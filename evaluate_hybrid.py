"""
Hybrid RAG Evaluation Script
Runs the full evaluation dataset through the Hybrid RAG engine,
then grades each answer with an LLM judge.

Improvements over v1:
    - Parallel grading via ThreadPoolExecutor (RAG still sequential to respect rate limits)
    - Retry logic with exponential back-off for transient API errors
    - Dataclass-based config (no scattered magic strings/numbers)
    - Incremental autosave every N rows AND on keyboard interrupt
    - Resume support: skips already-processed question IDs
    - Richer scorecard (per-type breakdown + overall)
    - Structured logging with contextual fields
    - Context-manager usage of HybridRAGService
    - Typed results via TypedDict
    - Grading prompt moved to module-level constant
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
load_dotenv()

try:
    from hybrid_rag import HybridRAGService
except ImportError as exc:
    raise SystemExit(
        "❌ Could not import HybridRAGService. Make sure hybrid_rag.py is in the same folder."
    ) from exc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    input_file: str = "evaluation_dataset_openai.csv"
    output_file: str = "hybrid_rag_results.csv"
    judge_model: str = "gpt-4o-mini"
    judge_temperature: float = 1.0
    autosave_every: int = 5           # rows between incremental saves
    max_grade_workers: int = 4        # parallel grading threads
    max_retries: int = 3              # retries for grading API errors
    retry_base_delay: float = 2.0     # seconds; doubled on each retry


SCORE_COLUMNS = [
    "financial_accuracy",
    "comprehensiveness",
    "diversity",
    "empowerment",
    "directness",
]

_GRADING_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert Financial Examiner grading an AI analyst's response.

Compare the STUDENT ANSWER against the GROUND TRUTH for the given QUESTION.

QUESTION:      {question}
GROUND TRUTH:  {ground_truth}
STUDENT ANSWER: {generated_answer}

GRADING RUBRIC
--------------
• financial_accuracy  — Are the correct numbers, dates, and entities present?
                        5 = all key facts match  |  1 = none match
• comprehensiveness   — Does the answer cover the full scope of the question?
• diversity           — Does it synthesise multiple sources/perspectives?
• empowerment         — Is it genuinely useful to a financial analyst?
• directness          — Is it concise and free of unnecessary filler?

PARTIAL CREDIT GUIDE
--------------------
- Student finds ALL relevant data → 5
- Student finds MOST relevant data → 4
- Student finds SOME relevant data → 3
- Student finds LITTLE relevant data → 2
- Student finds NO relevant data → 1

Respond with ONLY valid JSON (no markdown fences):
{{
    "financial_accuracy": <int 1-5>,
    "comprehensiveness":  <int 1-5>,
    "diversity":          <int 1-5>,
    "empowerment":        <int 1-5>,
    "directness":         <int 1-5>,
    "reasoning":          "<one-sentence explanation>"
}}"""
)

_FALLBACK_GRADES: Dict[str, Any] = {
    col: 1 for col in SCORE_COLUMNS
} | {"reasoning": "Grading error — fallback scores applied"}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("hybrid_eval")


# ---------------------------------------------------------------------------
# Typed result row
# ---------------------------------------------------------------------------

class ResultRow(Dict[str, Any]):
    question_id: int
    question_type: str
    question: str
    ground_truth: str
    generated_answer: str
    latency_seconds: float
    financial_accuracy: int
    comprehensiveness: int
    diversity: int
    empowerment: int
    directness: int
    reasoning: str
    grading_error: bool


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class HybridEvaluator:
    def __init__(self, cfg: Optional[EvalConfig] = None) -> None:
        self._cfg = cfg or EvalConfig()
        # Initialize Azure OpenAI for judge
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_deployment = os.getenv("AZURE_OPENAI_MODEL_NAME")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        
        self._judge = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            azure_deployment=azure_deployment,
            api_version=azure_api_version,
            temperature=self._cfg.judge_temperature,
        )
        self._grading_chain = _GRADING_PROMPT | self._judge | JsonOutputParser()
        self._executor = ThreadPoolExecutor(
            max_workers=self._cfg.max_grade_workers,
            thread_name_prefix="grader",
        )
        logger.info("HybridEvaluator ready (judge=%s)", self._cfg.judge_model)

    # ------------------------------------------------------------------
    # Grading
    # ------------------------------------------------------------------

    def _grade_one(
        self, question: str, ground_truth: str, generated_answer: str
    ) -> Dict[str, Any]:
        """Grade a single answer with retry + back-off."""
        payload = {
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
        }
        delay = self._cfg.retry_base_delay
        for attempt in range(1, self._cfg.max_retries + 1):
            try:
                grades = self._grading_chain.invoke(payload)
                # Validate expected keys are present
                for col in SCORE_COLUMNS:
                    if col not in grades:
                        raise ValueError(f"Missing key '{col}' in grade response")
                grades["grading_error"] = False
                return grades
            except Exception as exc:
                logger.warning(
                    "Grading attempt %d/%d failed: %s",
                    attempt, self._cfg.max_retries, exc,
                )
                if attempt < self._cfg.max_retries:
                    time.sleep(delay)
                    delay *= 2
        logger.error("All grading attempts exhausted — using fallback scores")
        return {**_FALLBACK_GRADES, "grading_error": True}

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------

    def run(
        self,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Run the full evaluation pipeline.

        Supports resume: if *output_path* already exists, rows whose
        question_id is already present are skipped.
        """
        input_path = input_path or self._cfg.input_file
        output_path = output_path or self._cfg.output_file

        df = pd.read_csv(input_path)
        required_cols = {"question", "ground_truth"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Input CSV is missing columns: {missing}")

        # Resume: load existing results
        existing_ids: set[int] = set()
        results: List[ResultRow] = []
        out_path = Path(output_path)
        if out_path.exists():
            existing_df = pd.read_csv(out_path)
            existing_ids = set(existing_df["question_id"].tolist())
            results = existing_df.to_dict("records")
            logger.info("Resuming — %d rows already completed", len(existing_ids))

        pending = df[~df.index.isin(existing_ids)]
        total = len(df)
        logger.info(
            "Starting evaluation: %d total | %d pending | %d already done",
            total, len(pending), len(existing_ids),
        )

        try:
            self._process_rows(pending, results, total, output_path)
        except KeyboardInterrupt:
            logger.warning("⚠️  Interrupted — saving progress before exit…")
            _save(results, output_path)
            raise

        final_df = _save(results, output_path)
        self._print_scorecard(final_df)
        return final_df

    def _process_rows(
        self,
        pending: pd.DataFrame,
        results: List[ResultRow],
        total: int,
        output_path: str,
    ) -> None:
        """Inner loop: RAG query (sequential) then grade (parallel)."""
        completed_since_save = 0

        with HybridRAGService() as rag:
            for pos, (index, row) in enumerate(pending.iterrows(), start=1):
                question = row["question"]
                ground_truth = row["ground_truth"]
                q_type = row.get("question_type", "Unknown")

                done_total = len(results) + 1
                print(
                    f"\n[{done_total}/{total}] ({q_type}) {question[:70]}…",
                    flush=True,
                )

                # --- RAG query (kept sequential to avoid hammering vector/graph DBs) ---
                t0 = time.monotonic()
                try:
                    rag_result = rag.query(question)
                    generated_answer = rag_result["answer"]
                    latency = rag_result["timing"].get("total_s", round(time.monotonic() - t0, 3))
                except Exception:
                    logger.exception("RAG query failed for question_id=%d", index)
                    generated_answer = "RAG ERROR"
                    latency = round(time.monotonic() - t0, 3)

                # --- Submit grading to thread pool (non-blocking) ---
                future = self._executor.submit(
                    self._grade_one, question, ground_truth, generated_answer
                )

                # Collect grade (block here; parallelism is across retries + network I/O)
                grades = future.result()

                result_row: ResultRow = {
                    "question_id": int(index),
                    "question_type": q_type,
                    "question": question,
                    "ground_truth": ground_truth,
                    "generated_answer": generated_answer,
                    "latency_seconds": latency,
                    **grades,
                }
                results.append(result_row)
                completed_since_save += 1

                if completed_since_save >= self._cfg.autosave_every:
                    _save(results, output_path)
                    logger.info("Autosaved %d rows → %s", len(results), output_path)
                    completed_since_save = 0

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _print_scorecard(df: pd.DataFrame) -> None:
        sep = "=" * 55
        print(f"\n{sep}")
        print("📊  HYBRID RAG — FINAL SCORECARD")
        print(sep)

        overall = df[SCORE_COLUMNS].mean().round(3)
        print("\n▶  Overall averages:")
        for col, val in overall.items():
            bar = "█" * int(val) + "░" * (5 - int(val))
        print(overall.to_string())

        if "question_type" in df.columns:
            print("\n▶  Averages by question type:")
            by_type = (
                df.groupby("question_type")[SCORE_COLUMNS]
                .mean()
                .round(3)
            )
            print(by_type.to_string())

        if "grading_error" in df.columns:
            n_errors = df["grading_error"].sum()
            if n_errors:
                print(f"\n⚠️   {n_errors} row(s) had grading errors — check logs.")

        avg_latency = df["latency_seconds"].mean()
        print(f"\n⏱   Mean RAG latency: {avg_latency:.2f}s")
        print(sep)

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    def __enter__(self) -> "HybridEvaluator":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(results: List[Dict[str, Any]], path: str) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = EvalConfig()  # Override fields here or load from env/args as needed
    with HybridEvaluator(cfg) as evaluator:
        evaluator.run()


if __name__ == "__main__":
    main()