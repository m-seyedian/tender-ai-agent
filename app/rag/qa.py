from pathlib import Path
from typing import Protocol

from google import genai
from google.genai import errors, types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logger import get_logger
from app.rag.retriever import HybridTenderRetriever
from app.schemas.document import (
    AskRequest,
    AskResponse,
    CitationItem,
    GroundedLLMAnswer,
    SearchRequest,
    SearchResultItem,
)

logger = get_logger(__name__)

QA_PROMPT_PATH = Path("prompts/rag_qa.txt")

DEFAULT_NO_ANSWER_MESSAGE = (
    "Aucune information suffisante n'a été trouvée dans les documents fournis pour répondre à cette question."
)


def _is_retryable_gemini_error(exc: BaseException) -> bool:
    if isinstance(exc, errors.ClientError):
        return exc.code == 429
    return True


class GroundedLLMClient(Protocol):
    """Interface for any LLM backend returning structured GroundedLLMAnswer."""

    model_name: str

    def generate_grounded_answer(self, prompt: str) -> GroundedLLMAnswer:
        ...


class GeminiGroundedLLM:
    """Production Gemini client using Structured Outputs to enforce zero-hallucination Q&A."""

    def __init__(self, model_name: str = "gemini-2.5-flash") -> None:
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is missing in environment configuration.")
        self.client = genai.Client(
            api_key=settings.GEMINI_API_KEY,
            http_options=types.HttpOptions(api_version="v1beta"),
        )
        self.model_name = model_name

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(_is_retryable_gemini_error),
    )
    def generate_grounded_answer(self, prompt: str) -> GroundedLLMAnswer:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GroundedLLMAnswer,
                temperature=0.0,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        if not response.parsed:
            raise ValueError("Gemini returned an empty or unparseable structured response.")
        return response.parsed


class TenderQAEngine:
    """End-to-end RAG Q&A engine with hybrid retrieval, hallucination guardrails, and verified citations."""

    def __init__(
        self,
        retriever: HybridTenderRetriever | None = None,
        llm_client: GroundedLLMClient | None = None,
        prompt_path: Path = QA_PROMPT_PATH,
    ) -> None:
        self.retriever = retriever or HybridTenderRetriever()
        self.llm_client = llm_client or GeminiGroundedLLM()
        self.prompt_path = prompt_path

    def _load_prompt_template(self) -> str:
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"QA prompt file not found: {self.prompt_path}")
        return self.prompt_path.read_text(encoding="utf-8")

    @staticmethod
    def _format_context_blocks(results: list[SearchResultItem]) -> str:
        blocks: list[str] = []
        for idx, item in enumerate(results, start=1):
            source_file = item.metadata.get("source_file", "unknown")
            page_num = item.metadata.get("page_number", "?")
            section = item.metadata.get("section_title") or "General"
            header = f"[Source {idx}] (Document: {source_file} | Section: {section} | Page: {page_num})"
            blocks.append(f"{header}\n{item.content}")
        return "\n\n---\n\n".join(blocks)

    @staticmethod
    def _build_verified_citations(
        results: list[SearchResultItem],
        used_indices: list[int],
    ) -> list[CitationItem]:
        citations: list[CitationItem] = []
        seen: set[int] = set()

        for idx in used_indices:
            if idx in seen or idx < 1 or idx > len(results):
                continue
            seen.add(idx)
            item = results[idx - 1]
            excerpt = item.content.strip()
            if len(excerpt) > 220:
                excerpt = excerpt[:217].rstrip() + "..."

            citations.append(
                CitationItem(
                    source_index=idx,
                    source_file=str(item.metadata.get("source_file", "unknown")),
                    page_number=int(item.metadata.get("page_number", 1)),
                    section_title=item.metadata.get("section_title"),
                    chunk_id=item.chunk_id,
                    relevance_score=item.score,
                    excerpt=excerpt,
                )
            )
        return citations

    def ask(self, request: AskRequest) -> AskResponse:
        search_res = self.retriever.search(
            SearchRequest(
                query=request.question,
                top_k=request.top_k,
                tender_id=request.tender_id,
                section_title=request.section_title,
                file_type=request.file_type,
                use_hybrid=request.use_hybrid,
                use_rerank=request.use_rerank,
            )
        )

        # Guardrail 1: Filter chunks below minimum relevance threshold
        qualified_results = [
            item for item in search_res.results if item.score >= request.min_score
        ]

        # Guardrail 2: Short-circuit if no relevant context exists (zero hallucination, zero LLM cost)
        if not qualified_results:
            return AskResponse(
                question=request.question,
                answered=False,
                answer=DEFAULT_NO_ANSWER_MESSAGE,
                citations=[],
                sources_consulted=0,
                model_used=self.llm_client.model_name,
            )

        context_blocks = self._format_context_blocks(qualified_results)
        prompt = self._load_prompt_template().format(
            question=request.question.strip(),
            context_blocks=context_blocks,
        )

        llm_output = self.llm_client.generate_grounded_answer(prompt)

        # Guardrail 3: If LLM determines the answer is not explicitly supported, suppress citations
        if not llm_output.answered:
            return AskResponse(
                question=request.question,
                answered=False,
                answer=llm_output.answer.strip() or DEFAULT_NO_ANSWER_MESSAGE,
                citations=[],
                sources_consulted=len(qualified_results),
                model_used=self.llm_client.model_name,
            )

        citations = self._build_verified_citations(
            results=qualified_results,
            used_indices=llm_output.used_source_indices,
        )

        # Guardrail 4: If model claimed answered=True but cited zero valid sources, downgrade to unanswered
        if not citations:
            logger.warning("LLM returned answered=True without valid source indices; downgrading to unanswered.")
            return AskResponse(
                question=request.question,
                answered=False,
                answer=DEFAULT_NO_ANSWER_MESSAGE,
                citations=[],
                sources_consulted=len(qualified_results),
                model_used=self.llm_client.model_name,
            )

        return AskResponse(
            question=request.question,
            answered=True,
            answer=llm_output.answer.strip(),
            citations=citations,
            sources_consulted=len(qualified_results),
            model_used=self.llm_client.model_name,
        )
