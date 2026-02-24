"""
RAG System Evaluator - Optimized Version

Purpose:
    Evaluates Retrieval-Augmented Generation (RAG) systems by:
    1. Retrieving relevant context from ChromaDB vector store
    2. Generating answers using GPT-4 based on retrieved context
    3. Grading answers against ground truth using LLM-as-a-Judge
    4. Tracking 5 key metrics: Financial Accuracy, Comprehensiveness, 
       Diversity, Empowerment, and Directness

Workflow:
    Question → Vector DB Retrieval → Answer Generation → Multi-Metric Grading → Analysis
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict, field
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation scores"""
    financial_accuracy: int = 0
    comprehensiveness: int = 0
    diversity: int = 0
    empowerment: int = 0
    directness: int = 0
    reasoning: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with clean keys"""
        return {
            'Financial Accuracy': self.financial_accuracy,
            'Comprehensiveness': self.comprehensiveness,
            'Diversity': self.diversity,
            'Empowerment': self.empowerment,
            'Directness': self.directness,
            'reasoning': self.reasoning
        }
    
    @property
    def average_score(self) -> float:
        """Calculate average score across all metrics"""
        scores = [
            self.financial_accuracy,
            self.comprehensiveness,
            self.diversity,
            self.empowerment,
            self.directness
        ]
        return np.mean(scores)


@dataclass
class EvaluationResult:
    """Single evaluation result"""
    question_id: int
    question_type: str
    question: str
    ground_truth: str
    generated_answer: str
    retrieved_context: str
    metrics: EvaluationMetrics
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    grading_time: float = 0.0
    num_chunks_retrieved: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to flat dictionary for CSV export"""
        base = {
            'question_id': self.question_id,
            'question_type': self.question_type,
            'question': self.question,
            'ground_truth': self.ground_truth,
            'generated_answer': self.generated_answer,
            'retrieved_context': self.retrieved_context,
            'num_chunks_retrieved': self.num_chunks_retrieved,
            'retrieval_time_ms': round(self.retrieval_time * 1000, 2),
            'generation_time_ms': round(self.generation_time * 1000, 2),
            'grading_time_ms': round(self.grading_time * 1000, 2),
            'total_time_ms': round((self.retrieval_time + self.generation_time + self.grading_time) * 1000, 2),
            'average_score': round(self.metrics.average_score, 2)
        }
        base.update(self.metrics.to_dict())
        return base


class Config:
    """Configuration for RAG evaluation"""
    # Paths
    DB_PATH = "vector_db"
    RESULTS_FILE = "vector_stress_results.csv"
    DETAILED_RESULTS_FILE = "vector_stress_detailed_results.json"
    SUMMARY_FILE = "vector_stress_summary.txt"
    
    # Model Settings
    EMBEDDING_MODEL = "text-embedding-3-small"
    STUDENT_MODEL = "gpt-4o-mini"  # Generates answers
    JUDGE_MODEL = "gpt-4o-mini"    # Grades answers
    TEMPERATURE = 0  # Deterministic for consistency
    
    # Retrieval Settings
    TOP_K_CHUNKS = 5  # Number of chunks to retrieve
    SEARCH_TYPE = "similarity"
    
    # Processing Settings
    MAX_WORKERS = 3  # Parallel evaluation
    RATE_LIMIT_DELAY = 0.3  # Seconds between API calls
    MAX_RETRIES = 2
    
    # Context Settings
    MAX_CONTEXT_LENGTH = 500  # Characters to save in CSV
    FULL_CONTEXT_IN_JSON = True  # Save full context in detailed JSON


class RAGEvaluator:
    """
    Evaluates RAG system performance using LLM-as-a-Judge methodology
    """
    
    def __init__(self, config: Config = Config(), db_path: Optional[str] = None, results_file: Optional[str] = None):
        """Initialize RAGEvaluator.

        Args:
            config: Optional Config instance to override defaults.
            db_path: Optional path to Chroma DB (overrides config.DB_PATH).
            results_file: Optional results CSV path (overrides config.RESULTS_FILE).
        """
        self.config = config
        # Allow callers (like check_models.py) to override paths
        if db_path:
            # Normalize: if caller passed a sqlite file path, use its parent directory
            p = Path(db_path)
            if p.exists() and p.is_file():
                self.config.DB_PATH = str(p.parent)
                logger.info(f"Normalized DB path from file to directory: {self.config.DB_PATH}")
            elif p.suffix.lower() in ('.sqlite3', '.sqlite', '.db'):
                # Path may not exist yet but looks like a file path; use parent
                self.config.DB_PATH = str(p.parent)
                logger.info(f"Interpreting DB path file -> directory: {self.config.DB_PATH}")
            else:
                self.config.DB_PATH = db_path
        if results_file:
            self.config.RESULTS_FILE = results_file

        self.start_time = None
        self._validate_environment()
        self._initialize_components()
    
    def _validate_environment(self) -> None:
        """Validate environment and paths"""
        load_dotenv()
        
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("❌ OPENAI_API_KEY not found in .env file")
        
        db_path = Path(self.config.DB_PATH)
        # If a file was supplied accidentally, normalize to its parent
        if db_path.exists() and db_path.is_file():
            logger.warning(f"DB path points to a file. Using parent directory instead: {db_path.parent}")
            db_path = db_path.parent
            self.config.DB_PATH = str(db_path)

        if not db_path.exists():
            raise FileNotFoundError(
                f"❌ Vector database directory not found at: {db_path}\n"
                f"Please ensure ChromaDB is initialized at this location."
            )
        
        logger.info("✅ Environment validated successfully")
    
    def _initialize_components(self) -> None:
        """Initialize embeddings, vector DB, and LLMs"""
        logger.info(f"🔌 Connecting to Vector DB at '{self.config.DB_PATH}'...")
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.config.EMBEDDING_MODEL
        )
        
        # Initialize vector database
        self.vector_db = Chroma(
            persist_directory=self.config.DB_PATH,
            embedding_function=self.embeddings
        )
        
        # Initialize retriever
        self.retriever = self.vector_db.as_retriever(
            search_type=self.config.SEARCH_TYPE,
            search_kwargs={"k": self.config.TOP_K_CHUNKS}
        )
        
        # Initialize LLMs
        self.student_llm = ChatOpenAI(
            model=self.config.STUDENT_MODEL,
            temperature=self.config.TEMPERATURE
        )
        
        self.judge_llm = ChatOpenAI(
            model=self.config.JUDGE_MODEL,
            temperature=self.config.TEMPERATURE
        )
        
        logger.info("✅ All components initialized")
    
    def _create_rag_prompt(self) -> ChatPromptTemplate:
        """Create the RAG answer generation prompt"""
        template = """You are a financial analyst assistant with expertise in SEC filings and corporate finance.

Use the following retrieved context to answer the question accurately and comprehensively.

Context:
{context}

Question: {question}

Requirements:
1. Answer STRICTLY based on the context provided - do not use external knowledge
2. If the context contains specific financial figures (revenue, expenses, etc.), cite them accurately
3. For comparison questions, explicitly contrast the data points from different companies
4. Synthesize information from multiple context chunks when relevant
5. If the answer cannot be determined from the context, state: "Data not available in provided context"
6. Be concise but comprehensive - include all relevant details

Answer:"""
        
        return ChatPromptTemplate.from_template(template)
    
    def _create_grading_prompt(
        self, 
        question: str, 
        ground_truth: str, 
        generated_answer: str
    ) -> str:
        """Create the LLM-as-a-Judge grading prompt"""
        return f"""You are an expert Financial Research Evaluator with deep expertise in SEC filings and corporate finance analysis.

Your task is to evaluate how well the Generated Answer matches the Ground Truth for the given Question.

Question: {question}

Ground Truth (Expected Answer): {ground_truth}

Generated Answer (RAG System Output): {generated_answer}

Evaluate the Generated Answer on these 5 metrics (Score 1-5 for each):

1. **Financial Accuracy** (1-5):
   - 5: All numbers (revenue, debt, percentages, dates) are identical to Ground Truth
   - 4: Minor rounding differences only
   - 3: Some numbers are correct, others missing
   - 2: Numbers present but mostly incorrect
   - 1: Wrong numbers or no numbers when required

2. **Comprehensiveness** (1-5):
   - 5: Covers ALL aspects and entities mentioned in Ground Truth
   - 4: Covers most aspects, minor omissions
   - 3: Covers about half of the required information
   - 2: Addresses question but misses major points
   - 1: Barely addresses the question

3. **Diversity** (1-5):
   - 5: Synthesizes information from multiple sources/chunks (especially for comparisons)
   - 4: Uses multiple sources, good synthesis
   - 3: Uses 2-3 sources with basic synthesis
   - 2: Mostly single source
   - 1: Single narrow source or no synthesis

4. **Empowerment** (1-5):
   - 5: Provides deep, actionable insights that go beyond surface facts
   - 4: Good insights with clear implications
   - 3: Adequate information, some insights
   - 2: Basic facts only, minimal insight
   - 1: Surface-level or unhelpful information

5. **Directness** (1-5):
   - 5: Concise, professional, zero fluff, perfect clarity
   - 4: Mostly concise with minor verbosity
   - 3: Adequate directness, some unnecessary text
   - 2: Verbose or unclear structure
   - 1: Extremely verbose or confusing

**Critical Instructions:**
- Output MUST follow the exact format below
- Each metric MUST be scored 1-5 (integer only)
- Reasoning MUST be a single concise sentence

**Output Format (STRICT):**
Financial Accuracy: [1-5]
Comprehensiveness: [1-5]
Diversity: [1-5]
Empowerment: [1-5]
Directness: [1-5]
Reasoning: [One sentence explaining the overall evaluation]
"""
    
    def retrieve_and_generate(
        self, 
        question: str
    ) -> Tuple[str, str, List[Document], float, float]:
        """
        Retrieve context and generate answer
        
        Returns:
            (generated_answer, context_text, retrieved_docs, retrieval_time, generation_time)
        """
        # Retrieval phase
        retrieval_start = time.time()
        retrieved_docs = self.retriever.invoke(question)
        retrieval_time = time.time() - retrieval_start
        
        # Format context
        context_text = "\n\n".join(
            f"[Chunk {i+1}]\n{doc.page_content}" 
            for i, doc in enumerate(retrieved_docs)
        )
        
        # Generation phase
        prompt = self._create_rag_prompt()
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": lambda x: format_docs(retrieved_docs), "question": RunnablePassthrough()}
            | prompt
            | self.student_llm
            | StrOutputParser()
        )
        
        generation_start = time.time()
        generated_answer = rag_chain.invoke(question)
        generation_time = time.time() - generation_start
        
        return generated_answer, context_text, retrieved_docs, retrieval_time, generation_time
    
    def grade_answer(
        self, 
        question: str, 
        ground_truth: str, 
        generated_answer: str
    ) -> Tuple[EvaluationMetrics, float]:
        """
        Grade the generated answer using LLM-as-a-Judge
        
        Returns:
            (metrics, grading_time)
        """
        grading_start = time.time()
        
        grading_prompt = self._create_grading_prompt(
            question, 
            ground_truth, 
            generated_answer
        )
        
        try:
            response = self.judge_llm.invoke(grading_prompt).content
            metrics = self._parse_grading_response(response)
        except Exception as e:
            logger.warning(f"Grading error: {e}")
            metrics = EvaluationMetrics(reasoning=f"Grading Error: {str(e)}")
        
        grading_time = time.time() - grading_start
        
        return metrics, grading_time
    
    def _parse_grading_response(self, response: str) -> EvaluationMetrics:
        """Parse LLM judge response into structured metrics"""
        metrics = EvaluationMetrics()
        
        try:
            lines = response.strip().split('\n')
            
            metric_map = {
                'Financial Accuracy': 'financial_accuracy',
                'Comprehensiveness': 'comprehensiveness',
                'Diversity': 'diversity',
                'Empowerment': 'empowerment',
                'Directness': 'directness',
                'Reasoning': 'reasoning'
            }
            
            for line in lines:
                if ':' not in line:
                    continue
                
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if key in metric_map:
                    attr_name = metric_map[key]
                    if key == 'Reasoning':
                        setattr(metrics, attr_name, value)
                    else:
                        # Extract integer score
                        try:
                            score = int(value.split()[0])  # Handle "5 - Perfect match"
                            score = max(1, min(5, score))  # Clamp to 1-5
                            setattr(metrics, attr_name, score)
                        except (ValueError, IndexError):
                            logger.warning(f"Failed to parse score for {key}: {value}")
                            
        except Exception as e:
            logger.error(f"Error parsing grading response: {e}")
            metrics.reasoning = f"Parse Error: {str(e)}"
        
        return metrics
    
    def evaluate_single_question(
        self, 
        question_id: int,
        question_type: str,
        question: str,
        ground_truth: str
    ) -> Optional[EvaluationResult]:
        """Evaluate a single question through the full pipeline"""
        try:
            # 1. Retrieve & Generate
            (
                generated_answer, 
                context_text, 
                retrieved_docs,
                retrieval_time,
                generation_time
            ) = self.retrieve_and_generate(question)
            
            # 2. Grade
            metrics, grading_time = self.grade_answer(
                question, 
                ground_truth, 
                generated_answer
            )
            
            # 3. Prepare context for storage
            if self.config.FULL_CONTEXT_IN_JSON:
                stored_context = context_text
            else:
                stored_context = context_text[:self.config.MAX_CONTEXT_LENGTH]
                if len(context_text) > self.config.MAX_CONTEXT_LENGTH:
                    stored_context += "..."
            
            # 4. Create result
            result = EvaluationResult(
                question_id=question_id,
                question_type=question_type,
                question=question,
                ground_truth=ground_truth,
                generated_answer=generated_answer,
                retrieved_context=stored_context,
                metrics=metrics,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                grading_time=grading_time,
                num_chunks_retrieved=len(retrieved_docs)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating question {question_id}: {e}", exc_info=True)
            return None
    
    def run_evaluation(
        self, 
        csv_path: str,
        parallel: bool = True
    ) -> pd.DataFrame:
        """
        Run evaluation on entire dataset
        
        Args:
            csv_path: Path to evaluation dataset CSV
            parallel: Use parallel processing (faster but uses more API quota)
        
        Returns:
            DataFrame with all results
        """
        self.start_time = datetime.now()
        logger.info(f"🚀 Starting RAG Evaluation on: {csv_path}")
        
        # Load dataset
        df = pd.read_csv(csv_path)
        logger.info(f"📊 Loaded {len(df)} questions")
        
        results = []
        
        if parallel and self.config.MAX_WORKERS > 1:
            results = self._run_parallel_evaluation(df)
        else:
            results = self._run_sequential_evaluation(df)
        
        # Convert to DataFrame
        if not results:
            raise ValueError("❌ No results generated")
        
        results_df = pd.DataFrame([r.to_dict() for r in results])
        
        logger.info(f"✅ Evaluation complete: {len(results_df)} results")
        
        return results_df
    
    def _run_sequential_evaluation(self, df: pd.DataFrame) -> List[EvaluationResult]:
        """Run evaluation sequentially with progress bar"""
        results = []
        
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            result = self.evaluate_single_question(
                question_id=index,
                question_type=row['question_type'],
                question=row['question'],
                ground_truth=row['ground_truth']
            )
            
            if result:
                results.append(result)
            
            # Rate limiting
            time.sleep(self.config.RATE_LIMIT_DELAY)
        
        return results
    
    def _run_parallel_evaluation(self, df: pd.DataFrame) -> List[EvaluationResult]:
        """Run evaluation in parallel for faster processing"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    self.evaluate_single_question,
                    index,
                    row['question_type'],
                    row['question'],
                    row['ground_truth']
                ): index for index, row in df.iterrows()
            }
            
            with tqdm(total=len(futures), desc="Evaluating (parallel)") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        results.append(result)
                    pbar.update(1)
                    time.sleep(self.config.RATE_LIMIT_DELAY)
        
        # Sort by question_id to maintain order
        results.sort(key=lambda x: x.question_id)
        
        return results
    
    def save_results(self, results_df: pd.DataFrame) -> None:
        """Save results in multiple formats"""
        # 1. CSV for spreadsheet analysis
        results_df.to_csv(self.config.RESULTS_FILE, index=False)
        logger.info(f"💾 CSV saved to: {self.config.RESULTS_FILE}")
        
        # 2. JSON for detailed analysis (if enabled)
        if self.config.FULL_CONTEXT_IN_JSON:
            results_dict = results_df.to_dict(orient='records')
            with open(self.config.DETAILED_RESULTS_FILE, 'w') as f:
                json.dump(results_dict, f, indent=2)
            logger.info(f"💾 Detailed JSON saved to: {self.config.DETAILED_RESULTS_FILE}")
        
        # 3. Summary report
        self._generate_summary_report(results_df)
    
    def _generate_summary_report(self, results_df: pd.DataFrame) -> None:
        """Generate human-readable summary report"""
        metrics = [
            'Financial Accuracy', 
            'Comprehensiveness', 
            'Diversity', 
            'Empowerment', 
            'Directness'
        ]
        
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("RAG SYSTEM EVALUATION REPORT")
        report_lines.append("=" * 70)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Questions Evaluated: {len(results_df)}")
        
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            report_lines.append(f"Total Evaluation Time: {duration:.2f} seconds")
            report_lines.append(f"Average Time per Question: {duration/len(results_df):.2f} seconds")
        
        report_lines.append("\n" + "-" * 70)
        report_lines.append("OVERALL METRICS (Average Scores out of 5.0)")
        report_lines.append("-" * 70)
        
        for metric in metrics:
            avg_score = results_df[metric].mean()
            std_score = results_df[metric].std()
            min_score = results_df[metric].min()
            max_score = results_df[metric].max()
            
            report_lines.append(
                f"{metric:20s}: {avg_score:.2f} ± {std_score:.2f} "
                f"(min: {min_score}, max: {max_score})"
            )
        
        avg_overall = results_df['average_score'].mean()
        report_lines.append(f"\n{'Overall Average':20s}: {avg_overall:.2f} / 5.0")
        
        # Question type breakdown
        report_lines.append("\n" + "-" * 70)
        report_lines.append("BREAKDOWN BY QUESTION TYPE")
        report_lines.append("-" * 70)
        
        for q_type in results_df['question_type'].unique():
            type_df = results_df[results_df['question_type'] == q_type]
            type_avg = type_df['average_score'].mean()
            report_lines.append(
                f"{q_type:15s}: {len(type_df):3d} questions, "
                f"avg score: {type_avg:.2f}"
            )
        
        # Performance metrics
        report_lines.append("\n" + "-" * 70)
        report_lines.append("PERFORMANCE METRICS")
        report_lines.append("-" * 70)
        report_lines.append(f"Average Retrieval Time: {results_df['retrieval_time_ms'].mean():.2f} ms")
        report_lines.append(f"Average Generation Time: {results_df['generation_time_ms'].mean():.2f} ms")
        report_lines.append(f"Average Grading Time: {results_df['grading_time_ms'].mean():.2f} ms")
        report_lines.append(f"Average Total Time: {results_df['total_time_ms'].mean():.2f} ms")
        report_lines.append(f"Average Chunks Retrieved: {results_df['num_chunks_retrieved'].mean():.1f}")
        
        # Top and bottom performers
        report_lines.append("\n" + "-" * 70)
        report_lines.append("TOP 3 PERFORMING QUESTIONS")
        report_lines.append("-" * 70)
        
        top_3 = results_df.nlargest(3, 'average_score')
        for idx, row in top_3.iterrows():
            report_lines.append(f"\n[Score: {row['average_score']:.2f}] {row['question'][:80]}...")
        
        report_lines.append("\n" + "-" * 70)
        report_lines.append("BOTTOM 3 PERFORMING QUESTIONS")
        report_lines.append("-" * 70)
        
        bottom_3 = results_df.nsmallest(3, 'average_score')
        for idx, row in bottom_3.iterrows():
            report_lines.append(f"\n[Score: {row['average_score']:.2f}] {row['question'][:80]}...")
            report_lines.append(f"Reason: {row['reasoning']}")
        
        report_lines.append("\n" + "=" * 70)
        
        # Save report
        report_text = "\n".join(report_lines)
        with open(self.config.SUMMARY_FILE, 'w') as f:
            f.write(report_text)
        
        # Also print to console
        print("\n" + report_text)
        logger.info(f"📄 Summary report saved to: {self.config.SUMMARY_FILE}")

    # Compatibility helper methods for external scripts (e.g., check_models.py)
    def generate_answer(self, question: str) -> Tuple[str, str]:
        """Compatibility wrapper returning (generated_answer, context_text)."""
        generated_answer, context_text, _, _, _ = self.retrieve_and_generate(question)
        return generated_answer, context_text

    def grade_response(self, question: str, ground_truth: str, generated_answer: str) -> Dict[str, object]:
        """Compatibility wrapper returning metric dict similar to older tooling."""
        metrics, _ = self.grade_answer(question, ground_truth, generated_answer)
        return {
            'financial_accuracy': metrics.financial_accuracy,
            'comprehensiveness': metrics.comprehensiveness,
            'diversity': metrics.diversity,
            'empowerment': metrics.empowerment,
            'directness': metrics.directness,
            'reasoning': metrics.reasoning
        }
    
    def print_summary(self, results_df: pd.DataFrame) -> None:
        """Print summary statistics to console"""
        metrics = [
            'Financial Accuracy', 
            'Comprehensiveness', 
            'Diversity', 
            'Empowerment', 
            'Directness'
        ]
        
        print("\n" + "=" * 70)
        print("📊 EVALUATION SUMMARY")
        print("=" * 70)
        
        for metric in metrics:
            avg_score = results_df[metric].mean()
            print(f"• {metric:20s}: {avg_score:.2f} / 5.0")
        
        avg_overall = results_df['average_score'].mean()
        print(f"\n• {'Overall Average':20s}: {avg_overall:.2f} / 5.0")
        print(f"• {'Total Questions':20s}: {len(results_df)}")
        print("=" * 70)


def main():
    """Main execution function"""
    try:
        # Initialize evaluator
        evaluator = RAGEvaluator()
        
        # Run evaluation
        results_df = evaluator.run_evaluation(
            csv_path="stress_graph_test.csv",
            parallel=True  # Set to False for sequential processing
        )
        
        # Save results
        evaluator.save_results(results_df)
        
        # Print summary
        evaluator.print_summary(results_df)
        
        logger.info("\n🎉 Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"\n❌ Fatal Error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()