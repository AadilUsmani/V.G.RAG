"""
Hybrid RAG System (Parallel Retrieval / Late Fusion)

Architecture:
    1. User Query -> Vector DB -> Top-K Chunks
    2. User Query -> Graph DB  -> Metadata/Node Context
    3. Context Fusion -> LLM  -> Final Answer

Improvements:
    - True parallel retrieval via asyncio / concurrent.futures
    - Dataclass-based config (no magic numbers scattered in code)
    - Result caching with TTL to avoid redundant LLM + DB calls
    - Typed return values (TypedDict)
    - Centralised prompt management
    - Graceful degradation when one retriever fails
    - Structured logging with per-query timing breakdowns
    - Context length guard to avoid exceeding LLM token limits
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

try:
    from query_graph_rag import GraphRAGQueryEngine
except ImportError as exc:
    raise ImportError(
        "❌ query_graph_rag.py not found. Ensure it is in the same directory."
    ) from exc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class VectorConfig:
    persist_directory: str = "./vector_db"
    embedding_model: str = "text-embedding-3-small"
    top_k: int = 5
    max_context_chars: int = 8_000   # ~2 k tokens; trim if larger


@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 1.0
    max_tokens: Optional[int] = None


@dataclass
class HybridConfig:
    vector: VectorConfig = field(default_factory=VectorConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval_timeout: float = 60.0   # seconds per retriever
    cache_ttl: float = 300.0          # seconds to keep cached answers
    max_workers: int = 2              # thread pool size


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("hybrid_rag")


# ---------------------------------------------------------------------------
# Typed result
# ---------------------------------------------------------------------------

class QueryResult(Dict[str, Any]):
    """Typed dict returned by HybridRAGService.query()."""
    question: str
    answer: str
    vector_context_chars: int
    graph_context_chars: int
    cached: bool
    timing: Dict[str, float]


# ---------------------------------------------------------------------------
# Simple in-process TTL cache
# ---------------------------------------------------------------------------

class _TTLCache:
    """Minimal TTL cache; no external dependency required."""

    def __init__(self, ttl: float) -> None:
        self._ttl = ttl
        self._store: Dict[str, Tuple[Any, float]] = {}

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, question: str) -> Optional[Any]:
        k = self._key(question)
        if k in self._store:
            value, ts = self._store[k]
            if time.monotonic() - ts < self._ttl:
                return value
            del self._store[k]
        return None

    def set(self, question: str, value: Any) -> None:
        self._store[self._key(question)] = (value, time.monotonic())

    def clear(self) -> None:
        self._store.clear()


# ---------------------------------------------------------------------------
# Vector retriever
# ---------------------------------------------------------------------------

class VectorQueryEngine:
    """Thin wrapper around a persisted Chroma vector store."""

    def __init__(self, cfg: VectorConfig) -> None:
        self._cfg = cfg
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_embedding_deployment = os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
        )
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

        self._embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            azure_deployment=azure_embedding_deployment,
            api_version=azure_api_version,
        )
        self._store: Optional[Chroma] = None
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._cfg.persist_directory):
            self._store = Chroma(
                persist_directory=self._cfg.persist_directory,
                embedding_function=self._embeddings,
            )
            logger.info("Vector DB loaded from '%s'", self._cfg.persist_directory)
        else:
            logger.warning(
                "Vector DB not found at '%s' — vector search disabled.",
                self._cfg.persist_directory,
            )

    def query(self, question: str) -> str:
        if self._store is None:
            return ""
        try:
            docs = self._store.similarity_search(question, k=self._cfg.top_k)
            chunks = [
                f"[Vector chunk {i + 1} | {d.metadata.get('source', 'doc')}]\n{d.page_content}"
                for i, d in enumerate(docs)
            ]
            context = "\n\n".join(chunks)
            # Guard against oversized contexts
            if len(context) > self._cfg.max_context_chars:
                context = context[: self._cfg.max_context_chars] + "\n... [truncated]"
            return context
        except Exception:
            logger.exception("Vector retrieval failed")
            return ""


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYNTHESIS_PROMPT = ChatPromptTemplate.from_template(
    """You are a senior financial analyst assistant with access to two retrieval databases.

COMBINED CONTEXT
================
{context}

QUESTION: {question}

INSTRUCTIONS
============
1. Precision first — use Vector Database for exact figures, dates, and verbatim quotes.
2. Structure first — use the Knowledge Graph to surface entity relationships and subsidiaries.
3. Conflict resolution — when both sources mention the same fact, prefer the more specific number (e.g. "$383 billion" beats a generic node labelled "Revenue").
4. Synthesis — where both sources are relevant, combine them (e.g. Vector states Apple revenue; Graph links Apple → Foxconn supply chain).
5. Citation — briefly tag key facts (e.g. "per 10-K text …" or "graph node indicates …").
6. If neither source contains enough information, say so explicitly rather than speculating.

ANSWER:"""
)


# ---------------------------------------------------------------------------
# Hybrid RAG service
# ---------------------------------------------------------------------------

class HybridRAGService:
    """
    Orchestrates parallel retrieval from Vector DB + Graph DB,
    fuses their contexts, and synthesises a final answer with an LLM.
    """

    def __init__(self, cfg: Optional[HybridConfig] = None) -> None:
        load_dotenv()
        self._cfg = cfg or HybridConfig()
        self._cache = _TTLCache(ttl=self._cfg.cache_ttl)
        self._executor = ThreadPoolExecutor(
            max_workers=self._cfg.max_workers, thread_name_prefix="rag-retriever"
        )

        self._vector_engine = VectorQueryEngine(self._cfg.vector)
        self._graph_engine = GraphRAGQueryEngine()

        # Initialize Azure OpenAI for LLM
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_deployment = os.getenv("AZURE_OPENAI_MODEL_NAME")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        
        self._llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            azure_deployment=azure_deployment,
            api_version=azure_api_version,
            temperature=self._cfg.llm.temperature,
            max_tokens=self._cfg.llm.max_tokens,
        )
        self._chain = _SYNTHESIS_PROMPT | self._llm | StrOutputParser()

        logger.info("HybridRAGService ready (model=%s)", self._cfg.llm.model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, question: str) -> QueryResult:
        """
        Run hybrid retrieval + synthesis for *question*.

        Returns a QueryResult dict with the answer, timing info, and
        context lengths for observability.
        """
        # 1. Cache check
        cached = self._cache.get(question)
        if cached is not None:
            logger.info("Cache hit for question: %s", question[:80])
            cached["cached"] = True
            return cached

        t0 = time.monotonic()
        logger.info("Hybrid query: %s", question)

        # 2. Parallel retrieval
        vector_ctx, graph_ctx, retrieval_timing = self._parallel_retrieve(question)

        # 3. Context fusion
        combined = self._fuse(vector_ctx, graph_ctx)

        # 4. LLM synthesis
        t_gen = time.monotonic()
        answer = self._chain.invoke({"question": question, "context": combined})
        gen_time = time.monotonic() - t_gen

        total_time = time.monotonic() - t0

        result: QueryResult = {
            "question": question,
            "answer": answer,
            "vector_context_chars": len(vector_ctx),
            "graph_context_chars": len(graph_ctx),
            "cached": False,
            "timing": {
                **retrieval_timing,
                "generation_s": round(gen_time, 3),
                "total_s": round(total_time, 3),
            },
        }

        self._cache.set(question, result)
        logger.info(
            "Done in %.2fs  (vector=%d chars, graph=%d chars)",
            total_time,
            len(vector_ctx),
            len(graph_ctx),
        )
        return result

    def clear_cache(self) -> None:
        """Evict all cached answers."""
        self._cache.clear()
        logger.info("Answer cache cleared")

    def close(self) -> None:
        """Release resources (graph DB connection, thread pool)."""
        self._executor.shutdown(wait=False)
        self._graph_engine.close()
        logger.info("HybridRAGService closed")

    # Context-manager support
    def __enter__(self) -> "HybridRAGService":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parallel_retrieve(
        self, question: str
    ) -> Tuple[str, str, Dict[str, float]]:
        """Fire both retrievers in parallel; return (vector_ctx, graph_ctx, timing)."""
        timing: Dict[str, float] = {}

        futures = {
            self._executor.submit(self._timed_vector, question): "vector",
            self._executor.submit(self._timed_graph, question): "graph",
        }

        results: Dict[str, str] = {"vector": "", "graph": ""}

        for future in as_completed(futures, timeout=self._cfg.retrieval_timeout + 1):
            key = futures[future]
            try:
                ctx, elapsed = future.result(timeout=self._cfg.retrieval_timeout)
                results[key] = ctx
                timing[f"{key}_retrieval_s"] = round(elapsed, 3)
            except FuturesTimeoutError:
                logger.warning("%s retrieval timed out after %.1fs", key, self._cfg.retrieval_timeout)
                timing[f"{key}_retrieval_s"] = self._cfg.retrieval_timeout
            except Exception:
                logger.exception("%s retrieval raised an unexpected error", key)
                timing[f"{key}_retrieval_s"] = -1.0

        return results["vector"], results["graph"], timing

    def _timed_vector(self, question: str) -> Tuple[str, float]:
        t = time.monotonic()
        ctx = self._vector_engine.query(question)
        return ctx, time.monotonic() - t

    def _timed_graph(self, question: str) -> Tuple[str, float]:
        t = time.monotonic()
        result = self._graph_engine.query(question, verbose=False)
        ctx = result.get("context", "") if isinstance(result, dict) else ""
        return ctx, time.monotonic() - t

    @staticmethod
    def _fuse(vector_ctx: str, graph_ctx: str) -> str:
        """Combine Vector and Graph contexts into a single block."""
        parts = []
        parts.append("=== VECTOR DATABASE (text chunks) ===")
        parts.append(vector_ctx or "⚠️  No vector data retrieved.")
        parts.append("\n=== KNOWLEDGE GRAPH (structured nodes) ===")
        parts.append(graph_ctx or "⚠️  No graph data retrieved.")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    questions = [
        "What was Apple's revenue in 2024?",          # Vector typically wins
        "What are the supply chain risks for Apple?", # Graph typically wins
        "Compare Apple and Tesla debt positions.",    # Hybrid required
    ]

    with HybridRAGService() as svc:
        for q in questions:
            result = svc.query(q)
            timing = result["timing"]
            print(
                f"\n{'─' * 60}\n"
                f"❓  {result['question']}\n\n"
                f"💡  {result['answer']}\n\n"
                f"⏱   total={timing['total_s']}s  "
                f"(vector={timing.get('vector_retrieval_s', '?')}s  "
                f"graph={timing.get('graph_retrieval_s', '?')}s  "
                f"gen={timing.get('generation_s', '?')}s)  "
                f"cached={result['cached']}\n"
            )


if __name__ == "__main__":
    main()
