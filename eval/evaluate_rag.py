import gc
import hashlib
import json
import math
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings

from app.rag.retriever import HybridTenderRetriever
from app.rag.vector_store import ChromaVectorStore
from app.schemas.document import DocumentChunk, DocumentMetadata, SearchRequest

DATASET_PATH = Path("eval/golden_qa_dataset.json")
RESULTS_MD_PATH = Path("eval/results.md")


class LocalTFHashEmbeddings(Embeddings):
    """Deterministic 64-d feature-hashing embeddings for reproducible offline benchmark runs."""

    DIM = 64

    def _embed(self, text: str) -> list[float]:
        tokens = re.findall(r"[a-z0-9àâäéèêëîïôöùûüç]{2,}", text.lower())
        vec = [0.0] * self.DIM
        if not tokens:
            return vec
        counts = Counter(tokens)
        for token, freq in counts.items():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = digest[0] % self.DIM
            sign = 1.0 if (digest[1] & 1) == 0 else -1.0
            vec[bucket] += sign * (1.0 + math.log(freq))
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@dataclass
class EvalMetrics:
    mode_name: str
    total_questions: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    per_question_rows: list[dict[str, Any]]


def evaluate_mode(
    retriever: HybridTenderRetriever,
    questions: list[dict[str, Any]],
    mode_name: str,
    use_hybrid: bool,
    use_rerank: bool,
) -> EvalMetrics:
    hits_1 = 0
    hits_3 = 0
    hits_5 = 0
    reciprocal_ranks: list[float] = []
    rows: list[dict[str, Any]] = []

    for q in questions:
        res = retriever.search(
            SearchRequest(
                query=q["question"],
                top_k=5,
                tender_id=q.get("tender_id"),
                section_title=q.get("section_title"),
                use_hybrid=use_hybrid,
                use_rerank=use_rerank,
            )
        )

        expected_file = q["expected_source_file"]
        expected_page = int(q["expected_page"])

        found_rank: int | None = None
        for rank, item in enumerate(res.results, start=1):
            if (
                item.metadata.get("source_file") == expected_file
                and int(item.metadata.get("page_number", -1)) == expected_page
            ):
                found_rank = rank
                break

        if found_rank is not None:
            if found_rank <= 1:
                hits_1 += 1
            if found_rank <= 3:
                hits_3 += 1
            if found_rank <= 5:
                hits_5 += 1
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)

        top1_label = (
            f"{res.results[0].metadata.get('source_file')} (p.{res.results[0].metadata.get('page_number')})"
            if res.results
            else "None"
        )
        rows.append(
            {
                "id": q["id"],
                "tender_id": q["tender_id"],
                "section": q.get("section_title") or "ALL",
                "expected": f"{expected_file} (p.{expected_page})",
                "top1": top1_label,
                "rank": found_rank if found_rank is not None else "Miss",
                "hit3": "PASS" if (found_rank is not None and found_rank <= 3) else "MISS",
            }
        )

    n = len(questions)
    return EvalMetrics(
        mode_name=mode_name,
        total_questions=n,
        hit_at_1=round((hits_1 / n) * 100, 2),
        hit_at_3=round((hits_3 / n) * 100, 2),
        hit_at_5=round((hits_5 / n) * 100, 2),
        mrr=round(sum(reciprocal_ranks) / n, 4),
        per_question_rows=rows,
    )


def run_benchmark(output_path: Path = RESULTS_MD_PATH) -> tuple[EvalMetrics, EvalMetrics]:
    raw = json.loads(DATASET_PATH.read_text(encoding="utf-8-sig"))
    corpus_chunks = [
        DocumentChunk(
            content=c["content"],
            metadata=DocumentMetadata(
                source_file=c["source_file"],
                file_type="pdf",
                page_number=c["page_number"],
                section_title=c["section_title"],
                chunk_index=idx,
                tender_id=c["tender_id"],
            ),
        )
        for idx, c in enumerate(raw["corpus_chunks"])
    ]
    questions = raw["questions"]

    with tempfile.TemporaryDirectory(prefix="rag_eval_", ignore_cleanup_errors=True) as tmp_dir:
        store = ChromaVectorStore(
            persist_directory=tmp_dir,
            collection_name="golden_eval_collection",
            embedding_function=LocalTFHashEmbeddings(),
        )
        store.add_chunks_idempotent(corpus_chunks)
        retriever = HybridTenderRetriever(vector_store=store)

        semantic_metrics = evaluate_mode(
            retriever=retriever,
            questions=questions,
            mode_name="Semantic-Only Baseline",
            use_hybrid=False,
            use_rerank=False,
        )
        hybrid_metrics = evaluate_mode(
            retriever=retriever,
            questions=questions,
            mode_name="Hybrid (Semantic + BM25 + RRF + Re-rank)",
            use_hybrid=True,
            use_rerank=True,
        )
        del retriever, store
        gc.collect()

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md_lines = [
        "# RAG Retrieval Evaluation Report (`Hit-rate@k` & `MRR`)",
        "",
        f"- **Dataset:** `{DATASET_PATH.as_posix()}` ({hybrid_metrics.total_questions} Golden Q&A pairs across `RC`, `CCTP`, `CCAP`, `BPU`)",
        f"- **Evaluated At:** {timestamp}",
        "- **Retrieval Pipeline:** ChromaDB Persistent Store + Metadata Filtering (`tender_id`, `section_title`) + BM25 RRF + Re-ranking",
        "",
        "## Summary Comparison",
        "",
        "| Retrieval Configuration | Hit-rate@1 | Hit-rate@3 | Hit-rate@5 | MRR |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| {semantic_metrics.mode_name} | {semantic_metrics.hit_at_1:.1f}% | {semantic_metrics.hit_at_3:.1f}% | {semantic_metrics.hit_at_5:.1f}% | {semantic_metrics.mrr:.4f} |",
        f"| **{hybrid_metrics.mode_name}** | **{hybrid_metrics.hit_at_1:.1f}%** | **{hybrid_metrics.hit_at_3:.1f}%** | **{hybrid_metrics.hit_at_5:.1f}%** | **{hybrid_metrics.mrr:.4f}** |",
        "",
        "## Per-Question Breakdown (Hybrid + RRF + Re-ranking)",
        "",
        "| ID | Tender | Section | Expected Source | Top-1 Retrieved | Rank | Hit@3 |",
        "| :--- | :--- | :---: | :--- | :--- | :---: | :---: |",
    ]

    for row in hybrid_metrics.per_question_rows:
        md_lines.append(
            f"| {row['id']} | `{row['tender_id']}` | {row['section']} | `{row['expected']}` | `{row['top1']}` | {row['rank']} | {row['hit3']} |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return semantic_metrics, hybrid_metrics


if __name__ == "__main__":
    sem, hyb = run_benchmark()
    print(f"Semantic Baseline -> Hit@1: {sem.hit_at_1}% | Hit@3: {sem.hit_at_3}% | Hit@5: {sem.hit_at_5}% | MRR: {sem.mrr}")
    print(f"Hybrid + Rerank   -> Hit@1: {hyb.hit_at_1}% | Hit@3: {hyb.hit_at_3}% | Hit@5: {hyb.hit_at_5}% | MRR: {hyb.mrr}")
    print(f"Saved detailed markdown report to {RESULTS_MD_PATH}")
