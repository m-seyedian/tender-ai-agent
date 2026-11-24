# RAG Retrieval Evaluation Report (`Hit-rate@k` & `MRR`)

- **Dataset:** `eval/golden_qa_dataset.json` *(Curated domain-specific synthetic benchmark modeled on real BOAMP RC/CCTP/CCAP/BPU clauses for deterministic offline CI evaluation)* (20 Golden Q&A pairs across `RC`, `CCTP`, `CCAP`, `BPU`)
- **Evaluated At:** 2026-11-24 13:12 UTC
- **Retrieval Pipeline:** ChromaDB Persistent Store + Metadata Filtering (`tender_id`, `section_title`) + BM25 RRF + Re-ranking

## Summary Comparison

| Retrieval Configuration | Hit-rate@1 | Hit-rate@3 | Hit-rate@5 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| Semantic-Only Baseline | 90.0% | 95.0% | 100.0% | 0.9350 |
| **Hybrid (Semantic + BM25 + RRF + Re-rank)** | **100.0%** | **100.0%** | **100.0%** | **1.0000** |

## Per-Question Breakdown (Hybrid + RRF + Re-ranking)

| ID | Tender | Section | Expected Source | Top-1 Retrieved | Rank | Hit@3 |
| :--- | :--- | :---: | :--- | :--- | :---: | :---: |
| Q01 | `BOAMP-26-101` | RC | `rc_26_101.pdf (p.1)` | `rc_26_101.pdf (p.1)` | 1 | PASS |
| Q02 | `BOAMP-26-101` | RC | `rc_26_101.pdf (p.3)` | `rc_26_101.pdf (p.3)` | 1 | PASS |
| Q03 | `BOAMP-26-101` | RC | `rc_26_101.pdf (p.5)` | `rc_26_101.pdf (p.5)` | 1 | PASS |
| Q04 | `BOAMP-26-101` | RC | `rc_26_101.pdf (p.7)` | `rc_26_101.pdf (p.7)` | 1 | PASS |
| Q05 | `BOAMP-26-101` | RC | `rc_26_101.pdf (p.9)` | `rc_26_101.pdf (p.9)` | 1 | PASS |
| Q06 | `BOAMP-26-101` | CCTP | `cctp_26_101.pdf (p.2)` | `cctp_26_101.pdf (p.2)` | 1 | PASS |
| Q07 | `BOAMP-26-101` | CCTP | `cctp_26_101.pdf (p.6)` | `cctp_26_101.pdf (p.6)` | 1 | PASS |
| Q08 | `BOAMP-26-101` | CCTP | `cctp_26_101.pdf (p.11)` | `cctp_26_101.pdf (p.11)` | 1 | PASS |
| Q09 | `BOAMP-26-101` | CCTP | `cctp_26_101.pdf (p.14)` | `cctp_26_101.pdf (p.14)` | 1 | PASS |
| Q10 | `BOAMP-26-101` | CCAP | `ccap_26_101.pdf (p.4)` | `ccap_26_101.pdf (p.4)` | 1 | PASS |
| Q11 | `BOAMP-26-101` | CCAP | `ccap_26_101.pdf (p.8)` | `ccap_26_101.pdf (p.8)` | 1 | PASS |
| Q12 | `BOAMP-26-101` | CCAP | `ccap_26_101.pdf (p.12)` | `ccap_26_101.pdf (p.12)` | 1 | PASS |
| Q13 | `BOAMP-26-101` | CCAP | `ccap_26_101.pdf (p.15)` | `ccap_26_101.pdf (p.15)` | 1 | PASS |
| Q14 | `BOAMP-26-101` | BPU | `bpu_26_101.pdf (p.1)` | `bpu_26_101.pdf (p.1)` | 1 | PASS |
| Q15 | `BOAMP-26-101` | BPU | `bpu_26_101.pdf (p.2)` | `bpu_26_101.pdf (p.2)` | 1 | PASS |
| Q16 | `BOAMP-26-101` | ALL | `cctp_26_101.pdf (p.2)` | `cctp_26_101.pdf (p.2)` | 1 | PASS |
| Q17 | `BOAMP-26-101` | ALL | `ccap_26_101.pdf (p.12)` | `ccap_26_101.pdf (p.12)` | 1 | PASS |
| Q18 | `BOAMP-26-202` | RC | `rc_26_202.pdf (p.2)` | `rc_26_202.pdf (p.2)` | 1 | PASS |
| Q19 | `BOAMP-26-202` | CCTP | `cctp_26_202.pdf (p.5)` | `cctp_26_202.pdf (p.5)` | 1 | PASS |
| Q20 | `BOAMP-26-202` | CCAP | `ccap_26_202.pdf (p.6)` | `ccap_26_202.pdf (p.6)` | 1 | PASS |
