# Vector Database Architecture Notes: ChromaDB vs. Qdrant

## 1. Benchmark Summary (BOAMP French Tender Corpus)

| Embedding Model | Dimensions | Recall@1 | Recall@3 | Avg Query Latency | Deployment Footprint |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Gemini (`gemini-embedding-001`)** | 3072 | **100.0%** | **100.0%** | 366.7 ms | Cloud API (Zero local RAM/GPU) |
| **Multilingual-E5 Base** | 768 | 95.0% | 100.0% | 153.7 ms | Local (~1.1 GB weights + PyTorch) |

---

## 2. Vector Store Comparison: ChromaDB vs. Qdrant

| Dimension | ChromaDB | Qdrant |
| :--- | :--- | :--- |
| **Core Engine** | Python + SQLite / DuckDB | Rust (high-throughput memory-mapped storage) |
| **Deployment Topology** | Embedded / in-process (`PersistentClient`) or lightweight server | Dedicated Client-Server microservice (Docker / Qdrant Cloud) |
| **Metadata Filtering** | Basic `where` filters; slower on complex compound predicates | Native indexed payload filtering (CPV codes, department, deadline range) |
| **Hybrid Search (Dense + Sparse)** | Requires external BM25 orchestration | Native multi-vector & sparse vector support (exact legal/article ID matching) |
| **Operational Complexity** | Minimal (zero extra containers needed for MVP) | Moderate (requires separate container, healthchecks, and backup volume) |

---

## 3. Architectural Decision for `tender-ai-agent`

1. **Current Phase (MVP & Single-Node Docker)**: Use **ChromaDB** (`langchain-chroma`) with **`models/gemini-embedding-001`**. It persists directly to a local volume without adding container orchestration overhead while delivering 100% Recall@1 on French procurement clauses.
2. **Scale-Up Path (>100k BOAMP Documents / Multi-Tenant Production)**: Migrate the vector store adapter to **Qdrant** to leverage Rust-level query latency, native hybrid search (exact CPV/article codes + semantic similarity), and strict payload filtering.
