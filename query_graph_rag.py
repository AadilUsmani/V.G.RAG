"""
Optimized Metadata-Based GraphRAG Engine
=========================================
Strategy: Source-file filtering + keyword matching on node IDs.
Works with inverted/disconnected graph topology.
Companies identified via source_file (AAPL_ / TSLA_).
"""

import os
import re
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CYPHER_MODEL  = "gpt-4o"
ANSWER_MODEL  = "gpt-4o"
MAX_RESULTS   = 150
QUERY_TIMEOUT = 30
CACHE_TTL     = 3600

METRIC_SYNONYMS: Dict[str, List[str]] = {
    "revenue":          ["net sales", "total net sales", "revenue", "net revenue", "total revenue"],
    "net sales":        ["net sales", "total net sales", "net revenue"],
    "income":           ["net income", "operating income", "income from operations", "earnings"],
    "profit":           ["net income", "gross margin", "operating income", "gross profit"],
    "gross margin":     ["gross margin", "gross profit"],
    "operating income": ["operating income", "income from operations"],
    "eps":              ["earnings per share", "eps", "basic eps", "diluted eps"],
    "earnings":         ["earnings per share", "net income", "eps"],
    "debt":             ["long-term debt", "total debt", "term debt", "notes payable", "debt"],
    "cash":             ["cash and cash equivalents", "free cash flow", "operating cash flow", "cash"],
    "assets":           ["total assets", "assets"],
    "liabilities":      ["total liabilities", "liabilities"],
    "equity":           ["shareholders equity", "stockholders equity", "total equity"],
    "r&d":              ["research and development", "r&d"],
    "capex":            ["capital expenditures", "capex", "capital expenditure"],
    "dividend":         ["dividends", "dividend per share"],
    "buyback":          ["share repurchase", "stock buyback", "repurchase"],
    "inventory":        ["inventory", "inventories"],
    "margin":           ["gross margin", "operating margin", "net margin", "profit margin"],
    "risk":             ["risk", "uncertainty", "exposure", "threat"],
    "segment":          ["segment", "americas", "europe", "greater china", "rest of asia"],
}

GARBAGE_PATTERN = re.compile(
    r"(\$[A-Z]{3,}_[A-Z\^\*\#\$\?]{3,}|^[A-Z0-9]{2,6}$|.*[\^\*\#\@\?]{2,}.*)"
)


class QueryCache:
    def __init__(self, ttl: int = CACHE_TTL):
        self._store: Dict[str, Tuple[Dict, float]] = {}
        self.ttl = ttl

    def _key(self, text: str) -> str:
        return hashlib.md5(text.lower().strip().encode()).hexdigest()

    def get(self, question: str) -> Optional[Dict]:
        key = self._key(question)
        if key in self._store:
            result, ts = self._store[key]
            if time.time() - ts < self.ttl:
                return result
            del self._store[key]
        return None

    def set(self, question: str, value: Dict) -> None:
        self._store[self._key(question)] = (value, time.time())


class GraphRAGQueryEngine:
    def __init__(self):
        load_dotenv()
        self._validate_env()
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self._test_connection()
        
        # Initialize Azure OpenAI for both LLMs
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_deployment = os.getenv("AZURE_OPENAI_MODEL_NAME")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        
        self.cypher_llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            azure_deployment=azure_deployment,
            api_version=azure_api_version,
            temperature=1.0
        )
        self.answer_llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            azure_deployment=azure_deployment,
            api_version=azure_api_version,
            temperature=1.0
        )
        self.cache = QueryCache()
        self.stats = {"total_queries": 0, "successful_queries": 0, "failed_queries": 0, "cache_hits": 0}
        logger.info("GraphRAG engine ready")

    def _validate_env(self):
        self.uri      = os.getenv("NEO4J_URI")
        self.user     = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")
        missing = [k for k, v in {"NEO4J_URI": self.uri, "NEO4J_USERNAME": self.user,
                                   "NEO4J_PASSWORD": self.password,
                                   "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
                                   "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
                                   "AZURE_OPENAI_MODEL_NAME": os.getenv("AZURE_OPENAI_MODEL_NAME"),
                                   "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION")}.items() if not v]
        if missing:
            raise ValueError(f"Missing env vars: {', '.join(missing)}")

    def _test_connection(self):
        with self.driver.session() as s:
            s.run("RETURN 1")
        logger.info(f"Connected to Neo4j at {self.uri}")

    def close(self):
        if self.driver:
            self.driver.close()

    def _detect_companies(self, question: str) -> List[str]:
        q = question.lower()
        targets = []
        if "apple" in q or "aapl" in q:
            targets.append("aapl")
        if "tesla" in q or "tsla" in q:
            targets.append("tsla")
        return targets or ["aapl", "tsla"]

    def _expand_keywords(self, question: str) -> List[str]:
        q = question.lower()
        keywords: set = set()
        for concept, synonyms in METRIC_SYNONYMS.items():
            if concept in q or any(s in q for s in synonyms):
                keywords.update(synonyms[:3])
        stopwords = {"what", "were", "when", "which", "their", "about", "these", "those",
                     "that", "with", "from", "have", "does", "both", "compare", "between",
                     "company", "apple", "tesla"}
        raw = [w for w in re.findall(r'\b[a-z]{4,}\b', q) if w not in stopwords]
        keywords.update(raw[:4])
        return list(keywords)[:8]

    def _build_cypher(self, file_filter: str, keywords: List[str]) -> str:
        kw_conditions = " OR ".join(
            [f"toLower(n.id) CONTAINS '{kw.lower()}'" for kw in keywords]
        ) if keywords else "TRUE"
        return (
            f"MATCH (n:Entity)\n"
            f"WHERE toLower(n.source_file) CONTAINS '{file_filter}'\n"
            f"  AND ({kw_conditions})\n"
            f"RETURN DISTINCT n.id AS id, n.type AS type, n.source_file AS source_file\n"
            f"ORDER BY n.type\n"
            f"LIMIT {MAX_RESULTS}"
        )

    def _llm_cypher_fallback(self, question: str, file_filter: str) -> str:
        prompt = ChatPromptTemplate.from_template(
            "You are a Neo4j Cypher expert. Node-only search — no relationships.\n"
            "Nodes: label :Entity, props: id, type, source_file\n"
            "Filter: WHERE toLower(n.source_file) CONTAINS '{file_filter}'\n"
            "Match keywords in n.id with toLower CONTAINS.\n"
            "Synonyms: revenue->'net sales', profit->'net income', debt->'term debt'\n"
            "RETURN DISTINCT n.id AS id, n.type AS type, n.source_file AS source_file\n"
            "LIMIT {max_results}\n"
            "Output raw Cypher only.\n"
            "Question: {question}\nCypher:"
        )
        chain = prompt | self.cypher_llm | StrOutputParser()
        raw = chain.invoke({"question": question, "file_filter": file_filter, "max_results": MAX_RESULTS})
        return raw.replace("```cypher", "").replace("```", "").strip()

    def _run_cypher(self, cypher: str) -> Tuple[List[Dict], Optional[str]]:
        try:
            with self.driver.session() as s:
                return [dict(r) for r in s.run(cypher, timeout=QUERY_TIMEOUT)], None
        except Exception as e:
            logger.error(f"Cypher error: {e}")
            return [], str(e)

    def _is_garbage(self, node_id: str) -> bool:
        if not node_id or len(node_id.strip()) < 2:
            return True
        if GARBAGE_PATTERN.match(node_id):
            return True
        special_ratio = sum(1 for c in node_id if not c.isalnum() and c not in " $.,%-()/'\"&:") / max(len(node_id), 1)
        return special_ratio > 0.4

    def _clean_records(self, records: List[Dict]) -> List[Dict]:
        seen: set = set()
        clean = []
        for r in records:
            node_id = str(r.get("id", ""))
            if node_id not in seen and not self._is_garbage(node_id):
                seen.add(node_id)
                clean.append(r)
        return clean

    def _format_context(self, records: List[Dict]) -> str:
        lines = []
        for r in records:
            src = str(r.get("source_file", ""))
            company = "Apple" if "aapl" in src.lower() else ("Tesla" if "tsla" in src.lower() else "Unknown")
            lines.append(f"[{r.get('type', '')}] {r.get('id', '')}  (company: {company})")
        return "\n".join(lines)

    _ANSWER_PROMPT = ChatPromptTemplate.from_template(
        "You are a senior financial analyst specializing in Apple and Tesla SEC 10-K filings.\n\n"
        "GRAPH DATA:\n{context}\n\n"
        "QUESTION: {question}\n\n"
        "INSTRUCTIONS:\n"
        "1. Extract ALL financial figures, metric names, years, and company references.\n"
        "2. [Amount]=dollar values, [Metric]=label names, [Year]=reporting periods.\n"
        "3. Comparisons: both companies side by side with exact values.\n"
        "4. Format large numbers clearly ('$391,035 million' = '$391.0 billion').\n"
        "5. If data exists but is messy, still extract useful facts — never say 'not available' if numbers exist.\n"
        "6. Only if context is truly empty: 'This specific data was not found in the knowledge graph.'\n"
        "7. Be concise and cite exact values.\n\nANSWER:"
    )

    def _generate_answer(self, question: str, context: str) -> str:
        chain = self._ANSWER_PROMPT | self.answer_llm | StrOutputParser()
        return chain.invoke({"question": question, "context": context})

    def query(self, question: str, verbose: bool = True, use_cache: bool = True) -> Dict[str, Any]:
        self.stats["total_queries"] += 1

        if use_cache:
            cached = self.cache.get(question)
            if cached:
                self.stats["cache_hits"] += 1
                if verbose:
                    print("Cache hit")
                return cached

        result: Dict[str, Any] = {
            "question": question, "answer": None, "cypher": None,
            "graph_results": [], "context": "", "success": False, "error": None, "timing": {}
        }
        t_total = time.time()

        try:
            if verbose:
                print(f"\n{'='*65}\n{question}\n{'='*65}")

            targets  = self._detect_companies(question)
            keywords = self._expand_keywords(question)
            all_records: List[Dict] = []
            cyphers_used: List[str] = []

            for target in targets:
                if verbose:
                    print(f"Searching: {target.upper()} | keywords: {keywords[:5]}")

                cypher = self._build_cypher(target, keywords)
                records, error = self._run_cypher(cypher)

                if not records or error:
                    if verbose:
                        print(f"  {len(records)} results, trying LLM fallback...")
                    cypher = self._llm_cypher_fallback(question, target)
                    records, error = self._run_cypher(cypher)

                cyphers_used.append(cypher)
                records = self._clean_records(records)
                all_records.extend(records)

                if verbose:
                    print(f"  {len(records)} clean records for {target.upper()}")

            result["cypher"] = "\n---\n".join(cyphers_used)
            result["graph_results"] = all_records

            if not all_records:
                result["answer"] = "This specific data was not found in the knowledge graph."
            else:
                context = self._format_context(all_records)
                result["context"] = context
                if verbose:
                    print(f"\nContext preview:\n{context[:400]}\n...")

                t_ans = time.time()
                answer = self._generate_answer(question, context)
                result["timing"]["answer_generation"] = round(time.time() - t_ans, 2)
                result["answer"] = answer
                result["success"] = True
                self.stats["successful_queries"] += 1

                if verbose:
                    print(f"\nAnswer:\n{answer}")

            if use_cache and result["success"]:
                self.cache.set(question, result)

        except Exception as e:
            result["error"] = str(e)
            result["answer"] = f"System error: {e}"
            self.stats["failed_queries"] += 1
            logger.error(f"Pipeline error: {e}", exc_info=True)

        finally:
            result["timing"]["total"] = round(time.time() - t_total, 2)
            if verbose:
                print(f"\nTotal: {result['timing']['total']}s")

        return result

    def batch_query(self, questions: List[str], verbose: bool = False) -> List[Dict]:
        results = []
        for i, q in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] {q[:70]}...")
            results.append(self.query(q, verbose=verbose))
        return results

    def debug_query(self, question: str):
        print(f"\n{'='*65}\nDEBUG: {question}\n{'='*65}")
        targets  = self._detect_companies(question)
        keywords = self._expand_keywords(question)
        print(f"Targets: {targets}\nKeywords: {keywords}")
        for target in targets:
            print(f"\n-- {target.upper()} --")
            cypher = self._build_cypher(target, keywords)
            print(f"Cypher:\n{cypher}")
            records, error = self._run_cypher(cypher)
            clean = self._clean_records(records)
            print(f"Raw: {len(records)}  Clean: {len(clean)}  Error: {error}")
            for r in clean[:8]:
                print(f"  {r}")

    def print_statistics(self):
        print("\n" + "="*50 + "\nSTATISTICS\n" + "="*50)
        for k, v in self.stats.items():
            print(f"  {k}: {v}")
        print("="*50)


def main():
    engine = GraphRAGQueryEngine()
    try:
        questions = [
            "What was Apple's revenue in 2024?",
            "What was Tesla's net income?",
            "Compare Apple and Tesla total revenue",
            "What are the main risks for Tesla?",
            "How did Apple's debt change over the last two years?",
            "What was Apple's gross margin?",
            "What are Apple's R&D expenses?",
            "Compare Apple and Tesla operating income",
        ]
        for q in questions:
            engine.query(q, verbose=True)
            time.sleep(0.3)
        engine.print_statistics()
    finally:
        engine.close()


if __name__ == "__main__":
    main()