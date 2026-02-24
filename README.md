# Hybrid RAG for Financial Document Analysis

## 1. Project Summary

This repository implements and evaluates a **Hybrid Retrieval-Augmented Generation** system that fuses semantic vector search (ChromaDB) with a structured knowledge graph (Neo4j) to answer questions over SEC 10-K filings for Apple (AAPL) and Tesla (TSLA). The system performs parallel retrieval from both datastores, merges their contexts via late fusion, and synthesises a final answer through an LLM. A 53-question evaluation benchmark with LLM-as-a-judge grading and paired statistical tests is included to quantify the claim that hybrid retrieval yields comparable or improved answer quality relative to either retriever used in isolation.

---

## 2. System Architecture

### 2.1 VectorQueryEngine (Chroma + Azure OpenAI Embeddings)

Defined in `hybrid_rag.py` as `VectorQueryEngine`. On construction it reads Azure credentials from the environment (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION`) and instantiates `AzureOpenAIEmbeddings` (LangChain) with the deployment `text-embedding-3-small`. It then loads a persisted `Chroma` vector store from `./vector_db`. If the directory does not exist, vector retrieval degrades gracefully to an empty string (no-op).

At query time, `similarity_search(question, k=5)` returns the top-5 chunks. A **context-length guard** truncates the concatenated text at 8 000 characters (~2 000 tokens) to prevent exceeding the downstream LLM context window.

The vector store is built offline by `build_vector_db.py`, which chunks cleaned SEC filings with `RecursiveCharacterTextSplitter` (chunk size 1 000, overlap 200) and writes embeddings via OpenAI `text-embedding-3-small` into a local ChromaDB directory.

### 2.2 GraphRAGQueryEngine (Neo4j + Cypher)

Defined in `query_graph_rag.py` as `GraphRAGQueryEngine`. On init it validates six environment variables (`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, and the four `AZURE_OPENAI_*` keys), opens a Neo4j driver connection, and creates two `AzureChatOpenAI` instances: one for Cypher generation and one for answer synthesis (both use the deployment named by `AZURE_OPENAI_MODEL_NAME`, temperature 1.0).

Query flow:

1. **Company detection** — regex match on the question for "apple/aapl" or "tesla/tsla".
2. **Keyword expansion** — a synonym map (`METRIC_SYNONYMS`) expands financial terms (e.g., "revenue" to `["net sales", "total net sales", ...]`) to widen the Cypher `WHERE` filter.
3. **Cypher execution** — a source-file filter + keyword `CONTAINS` clause is run against `Entity` nodes. Garbage nodes are filtered via `GARBAGE_PATTERN`.
4. **Answer generation** — the retrieved node/relationship context is fed to the answer LLM for natural-language synthesis.

Results are cached in an in-process `QueryCache` (MD5-keyed, TTL 3 600 s).

Graph construction is a two-stage offline process:

- `build_graph_data.py` — chunks cleaned filings (3 000-char chunks, 300-char overlap), sends each chunk to `gpt-4o-mini` with structured output (`GraphExtraction` Pydantic model) to extract typed nodes and relationships, deduplicates entities, and exports `graph_nodes.csv` / `graph_edges.csv` to `graph_output/`.
- `ingest_to_neo4j.py` — batch-loads those CSV files into Neo4j Aura with transaction batching, APOC-based dynamic relationship creation, and index/constraint setup.

### 2.3 Hybrid Orchestration

`HybridRAGService` in `hybrid_rag.py` wires the two engines together:

| Mechanism | Implementation detail |
|---|---|
| **Parallel retrieval** | A `ThreadPoolExecutor(max_workers=2)` submits `VectorQueryEngine.query` and `GraphRAGQueryEngine.query` concurrently. `as_completed` collects results with a per-retriever timeout of 60 s. If one retriever fails or times out, the other's context is still used (graceful degradation). |
| **Context fusion** | `_fuse()` concatenates both contexts under labelled headers (`=== VECTOR DATABASE ===` / `=== KNOWLEDGE GRAPH ===`). Missing contexts are replaced with a warning sentinel. |
| **LLM synthesis** | A LangChain chain (`ChatPromptTemplate | AzureChatOpenAI | StrOutputParser`) receives the fused context. The prompt instructs the LLM to prefer specific figures from vector chunks and structural relationships from the graph, resolve conflicts by specificity, and cite sources. The LLM is `AzureChatOpenAI` using the `AZURE_OPENAI_MODEL_NAME` deployment (temperature 1.0, no max-token cap). |
| **TTL cache** | `_TTLCache` stores SHA-256-keyed results for 300 s (configurable) to avoid redundant API calls on repeated questions. |
| **Observability** | Every `QueryResult` dict includes per-retriever latency, generation latency, total latency, and context character counts. |

### 2.4 Models and Configuration Summary

| Role | Model / Deployment | Configured in |
|---|---|---|
| Vector embeddings | `text-embedding-3-small` (Azure OpenAI) | `hybrid_rag.py` `VectorQueryEngine.__init__` |
| Hybrid LLM synthesis | Azure deployment via `AZURE_OPENAI_MODEL_NAME` | `hybrid_rag.py` `HybridRAGService.__init__` |
| Graph Cypher + answer LLMs | Same Azure deployment | `query_graph_rag.py` `GraphRAGQueryEngine.__init__` |
| Graph extraction (offline) | `gpt-4o-mini` (OpenAI) | `build_graph_data.py` `Config.LLM_MODEL` |
| Evaluation judge | Azure deployment via `AZURE_OPENAI_MODEL_NAME` | `evaluate_hybrid.py` `HybridEvaluator.__init__` |

---

## 3. How to Reproduce

### 3.1 Environment Setup

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# POSIX (Linux / macOS)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3.2 Environment Variables

Create a `.env` file in the project root with the following keys:

```
OPENAI_API_KEY=sk-...
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<key>
AZURE_OPENAI_MODEL_NAME=<deployment-name>
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-small
NEO4J_URI=neo4j+s://<host>
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<password>
```

### 3.3 Execution Commands

Run each step sequentially from the project root:

```powershell
# 1. Download raw SEC filings
python download_data.py

# 2. Clean filings into plain text
python clean_sec_filings.py

# 3. Build the vector database (ChromaDB)
python build_vector_db.py

# 4. Extract knowledge graph (nodes + edges)
python build_graph_data.py

# 5. Ingest graph into Neo4j
python ingest_to_neo4j.py

# 6. Evaluate Vector RAG baseline
python rag_evaluator.py

# 7. Evaluate Graph RAG baseline
python evaluate_graph_rag.py

# 8. Evaluate Hybrid RAG
python evaluate_hybrid.py

# 9. Generate comparison charts and significance tests
python generate_charts.py
```

---

## 4. Evaluation Methodology

### 4.1 Dataset

The benchmark is `evaluation_dataset_openai_fixed.csv` containing **53 question-answer pairs** derived from the cleaned SEC 10-K filings. Questions span multiple types (Reasoning, Factual, Comparison) and cover both Apple and Tesla filings.

### 4.2 Metrics

Each generated answer is graded on five dimensions (integer scale 1--5):

| Metric | Definition |
|---|---|
| `financial_accuracy` | Correctness of numbers, dates, and entities versus ground truth |
| `comprehensiveness` | Coverage of the full scope of the question |
| `diversity` | Synthesis of multiple sources or perspectives |
| `empowerment` | Practical usefulness to a financial analyst |
| `directness` | Conciseness and absence of filler |

### 4.3 Grading Process

An **LLM-as-a-judge** pipeline (implemented in `evaluate_hybrid.py` and `rag_evaluator.py`) grades each answer:

1. The generated answer, ground truth, and question are formatted into a structured grading prompt.
2. The judge LLM returns a JSON object with integer scores for each metric plus a reasoning string.
3. On API failure, exponential back-off retries up to 3 attempts; if all fail, fallback scores of 1 are assigned and `grading_error` is flagged.
4. Results auto-save every 5 rows and support resume on interruption.

### 4.4 Result Files

| File | Role |
|---|---|
| `evaluation_dataset_openai_fixed.csv` | Ground-truth input (53 rows) |
| `vector_rag_results.csv` | Graded output for Vector RAG baseline (53 rows) |
| `graph_rag_results.csv` | Graded output for Graph RAG baseline (53 rows) |
| `hybrid_rag_results.csv` | Graded output for Hybrid RAG (53 rows) |
| `significance_tests.csv` | Paired t-test and Wilcoxon signed-rank results across all architecture pairs and metrics (n = 53 per comparison) |

### 4.5 Significance Testing

`generate_charts.py` computes paired t-tests and Wilcoxon signed-rank tests for every (architecture pair, metric) combination. Results are written to `significance_tests.csv` with columns: comparison, metric, test, statistic, p-value, significance at alpha = 0.05, and sample size.

---

## 5. Results

### 5.1 Bar Chart -- Mean Scores with Standard Deviation

![Grouped bar chart comparing mean scores (with std-dev error bars) of Vector RAG, Graph RAG, and Hybrid RAG across five evaluation metrics](comparison_barchart.svg)

**Figure 1.** Mean metric scores (1--5) with standard-deviation error bars for each architecture across 53 questions. Source data: `vector_rag_results.csv`, `graph_rag_results.csv`, `hybrid_rag_results.csv`.

Hybrid RAG matches or exceeds both baselines on financial accuracy and directness while retaining the diversity advantage contributed by graph-based retrieval.

### 5.2 Radar Chart -- Multi-Dimensional Profile

![Radar chart showing the five-metric profile of Vector RAG, Graph RAG, and Hybrid RAG](comparison_radarchart.svg)

**Figure 2.** Radar overlay of mean scores for all five metrics. Source data: same three result CSVs.

The Hybrid RAG polygon covers a broader area than either baseline, indicating a more balanced performance profile across complementary evaluation dimensions.

### 5.3 Box Plot -- Per-Question Score Distributions

![Box plot showing per-question average score distributions for each architecture](comparison_boxplot.svg)

**Figure 3.** Distribution of per-question average scores (mean of five metrics per question). Source data: same three result CSVs.

Hybrid RAG exhibits a higher median and tighter interquartile range than Graph RAG, indicating more consistent answer quality. Statistical significance for individual metrics is detailed in `significance_tests.csv`.

---

## 6. Project Layout

```
.
├── .env                                  # API keys and connection strings (not committed)
├── download_data.py                      # Downloads raw SEC 10-K filings via EDGAR
├── clean_sec_filings.py                  # Parses and cleans raw filings into plain text
├── build_vector_db.py                    # Chunks text, generates embeddings, builds ChromaDB
├── build_graph_data.py                   # LLM-based entity/relationship extraction to CSV
├── ingest_to_neo4j.py                    # Batch-loads graph CSVs into Neo4j Aura
├── rag_evaluator.py                      # Vector RAG engine + LLM-as-a-judge grading pipeline
├── query_graph_rag.py                    # Graph RAG engine (Cypher generation + answer synthesis)
├── hybrid_rag.py                         # Hybrid RAG orchestrator (parallel retrieval + fusion)
├── generate_eval_qa.py                   # Script to generate evaluation question-answer pairs
├── evaluate_graph_rag.py                 # Runs 53 questions through Graph RAG and grades results
├── evaluate_hybrid.py                    # Runs 53 questions through Hybrid RAG and grades results
├── generate_charts.py                    # Produces bar, radar, box charts + significance tests
├── evaluation_dataset_openai_fixed.csv   # Ground-truth benchmark (53 QA pairs)
├── vector_rag_results.csv                # Graded Vector RAG answers
├── graph_rag_results.csv                 # Graded Graph RAG answers
├── hybrid_rag_results.csv                # Graded Hybrid RAG answers
├── significance_tests.csv                # Paired t-test and Wilcoxon p-values
├── comparison_barchart.svg               # Bar chart (mean + std dev)
├── comparison_radarchart.svg             # Radar chart (multi-metric profile)
├── comparison_boxplot.svg                # Box plot (score distributions)
├── clean_data/                           # Cleaned SEC filing text files
├── sec-edgar-filings/                    # Raw downloaded SEC filings (AAPL, TSLA)
├── graph_output/                         # Extracted graph nodes/edges CSVs and statistics
│   ├── graph_nodes.csv
│   ├── graph_edges.csv
│   ├── graph_data.json
│   └── graph_statistics.json
└── vector_db/                            # Persisted ChromaDB store (embeddings + metadata)
```

---

## 7. Technologies and Rationale

- **ChromaDB** -- lightweight, file-based vector store with LangChain integration; avoids the operational overhead of a managed vector service for a research prototype.
- **Neo4j Aura** -- managed graph database with native Cypher support; chosen for its mature Python driver and suitability for entity-relationship queries over financial filings.
- **LangChain** (`langchain-openai`, `langchain-chroma`, `langchain-core`) -- provides composable prompt/chain abstractions and standardised interfaces for Azure OpenAI and ChromaDB, reducing boilerplate.
- **Azure OpenAI** (`AzureChatOpenAI`, `AzureOpenAIEmbeddings`) -- enterprise endpoint for `text-embedding-3-small` (embeddings) and the configured chat deployment (synthesis, grading, Cypher generation). Azure was selected for quota management and regional availability.
- **OpenAI `gpt-4o-mini`** -- used for offline graph extraction (`build_graph_data.py`) as a cost-effective model for structured-output entity extraction at scale.
- **SciPy** (`ttest_rel`, `wilcoxon`) -- standard non-parametric and parametric paired tests for statistical comparison of evaluation scores.
- **Matplotlib** -- publication-quality chart generation (SVG + PNG dual export) without JavaScript dependencies.

---

## 8. Limitations

1. **Sample size** -- the evaluation dataset contains 53 questions. This is sufficient for paired testing but limits the power to detect small effect sizes and makes results sensitive to individual outlier questions.
2. **LLM grading bias** -- the judge model (same Azure deployment used for synthesis) may systematically favour answers that match its own generation style, inflating scores for all architectures but especially for Hybrid RAG which uses the same model.
3. **No reranker** -- retrieved chunks from the vector store are passed directly to the LLM without a cross-encoder reranking stage. Adding a reranker could improve precision at the cost of latency.
4. **API quota sensitivity** -- all retrieval, synthesis, and grading depend on Azure OpenAI API availability. Rate limiting (HTTP 429) during evaluation can inflate latency and trigger fallback grading scores.
5. **In-process TTL cache only** -- the query-level cache (`_TTLCache`) is not persisted to disk and is lost on process restart.
6. **Single embedding model** -- both ingestion (`build_vector_db.py`) and query-time retrieval (`hybrid_rag.py`) must use the same embedding model and dimension. Changing the model requires a full re-embed.
7. **Reproducibility dependency on exact `.env` keys** -- the system reads `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_MODEL_NAME`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME`, `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, and `OPENAI_API_KEY`. Missing or incorrect values cause silent failures in individual retrievers.

---

## 9. Key Findings (Statistical Results)

Based on 53 paired evaluations, significance testing (`paired_ttest` and `wilcoxon`) in `significance_tests.csv` supports the following implementation-level conclusions:

1. **Hybrid improves key weaknesses of Graph RAG.**
    Hybrid RAG is significantly better than Graph RAG on `financial_accuracy` (p = 0.0001367, paired t-test; p = 0.0002830, Wilcoxon) and `directness` (p = 0.0007598, paired t-test).

2. **Graph contributes a diversity signal that vector-only retrieval can miss.**
    Graph RAG vs Vector RAG shows a significant difference on `diversity` (p = 0.0017393, paired t-test; p = 0.0036100, Wilcoxon), indicating complementary information structure from graph retrieval.

3. **Hybrid vs Vector is mostly parity on this dataset.**
    Across the five metrics, Hybrid vs Vector comparisons in `significance_tests.csv` are mostly non-significant at alpha = 0.05, suggesting Hybrid’s primary gain in this run is correction of Graph-only failure modes rather than a uniform lift over Vector.

---

## 10. Future Work

1. **Cross-encoder reranker** -- insert a reranking step (e.g., `ms-marco-MiniLM`) between vector retrieval and context fusion to improve chunk relevance before LLM synthesis.
2. **CI-integrated evaluation** -- run the 53-question benchmark in a GitHub Actions workflow on each commit to detect regressions in answer quality.
3. **Chunk de-duplication** -- detect near-duplicate chunks across the vector and graph contexts before fusion to reduce prompt bloat and improve LLM focus.
4. **Persistent caching** -- replace `_TTLCache` with a disk-backed store (e.g., SQLite or Redis) so cached answers survive process restarts during long evaluation runs.
5. **Extended dataset** -- scale the benchmark to 200+ questions across more SEC filers and filing types (10-Q, 8-K) to strengthen statistical claims.
6. **Separate grading model** -- use a different model family (e.g., Claude or Gemini) as the judge to mitigate self-preference bias.
7. **Production deployment** -- containerise the service behind a FastAPI endpoint with health checks, structured JSON logging, and OpenTelemetry tracing for latency attribution.

---

## 11. Troubleshooting and Notes for Examiners

| Symptom | Likely cause | Where to look |
|---|---|---|
| `openai.RateLimitError` / HTTP 429 | Azure OpenAI quota exceeded | Check Azure portal billing and usage; increase TPM quota or add retry delay |
| `similarity_search` raises dimension mismatch | Vector DB was built with a different embedding model than the one configured in `.env` | Re-run `python build_vector_db.py` with the correct `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` |
| `neo4j.exceptions.ServiceUnavailable` | Neo4j Aura instance is paused or URI is wrong | Verify `NEO4J_URI` in `.env`; resume instance in Neo4j Aura console |
| All grading scores are 1 | Every grading attempt failed and fallback was applied | Inspect `grading_error` column in the results CSV; check API key validity |
| `ImportError: query_graph_rag` | `query_graph_rag.py` not in the working directory | Run all scripts from the project root (`code/`) |
| Charts missing or empty | One or more result CSVs not found | Ensure all three evaluation scripts have completed; check for `*.csv` in `code/` |
| `KeyError: AZURE_OPENAI_*` | `.env` file missing or key misspelled | Compare your `.env` against the template in Section 3.2 |

---

## 12. Minimal Usage Example

```python
from hybrid_rag import HybridRAGService

with HybridRAGService() as svc:
    result = svc.query("What was Apple's total revenue in fiscal year 2024?")
    print(result["answer"])
    print(f"Latency: {result['timing']['total_s']}s")
```

To run the full evaluation and produce charts:

```powershell
python evaluate_hybrid.py        # writes hybrid_rag_results.csv
python generate_charts.py        # writes comparison_*.svg and significance_tests.csv
```

Output artifacts are written to the project root. Open the SVGs in any browser for inspection.

---

## Reproducibility Checklist

- [ ] Set all required keys in `.env` (see Section 3.2)
- [ ] Run `python build_vector_db.py` to create `vector_db/`
- [ ] Run `python build_graph_data.py` then `python ingest_to_neo4j.py` to populate Neo4j
- [ ] Run `python rag_evaluator.py` and `python evaluate_graph_rag.py` for baseline results
- [ ] Run `python evaluate_hybrid.py` to produce `hybrid_rag_results.csv`
- [ ] Run `python generate_charts.py` to produce SVG charts and `significance_tests.csv`
