"""
GraphRAG Evaluation Script ("The Final Exam")

Purpose:
    Runs the full evaluation dataset through the GraphRAG engine and grades
    the responses using an LLM-as-a-Judge against the Ground Truth.

Metrics:
    1. Financial Accuracy (1-5)
    2. Comprehensiveness (1-5)
    3. Diversity (1-5)
    4. Empowerment (1-5)
    5. Directness (1-5)

Output:
    graph_rag_results.csv
"""

import pandas as pd
import time
import json
import logging
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Import your existing engine
# Ensure query_graph_rag.py is in the same directory
try:
    from query_graph_rag import GraphRAGQueryEngine
except ImportError:
    print("❌ Could not import GraphRAGQueryEngine. Make sure query_graph_rag.py is in the same folder.")
    exit(1)

# Configuration
INPUT_FILE = "evaluation_dataset_openai.csv"
OUTPUT_FILE = "graph_rag_results.csv"
JUDGE_MODEL = "gpt-4o-mini"  # The Teacher
JUDGE_TEMP = 0.0

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define Output Structure for the Judge
class GradeResult(BaseModel):
    financial_accuracy: int = Field(description="Score 1-5 for precision of numbers/entities")
    comprehensiveness: int = Field(description="Score 1-5 for covering all aspects")
    diversity: int = Field(description="Score 1-5 for synthesis of multiple sources")
    empowerment: int = Field(description="Score 1-5 for actionable insight")
    directness: int = Field(description="Score 1-5 for conciseness")
    reasoning: str = Field(description="Brief justification for the scores")

class EvaluationEngine:
    def __init__(self):
        self.rag_engine = GraphRAGQueryEngine()
        self.judge_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=JUDGE_TEMP)
        self.parser = JsonOutputParser(pydantic_object=GradeResult)
        
        # The Judge's Rubric - UPDATED FOR FAIRNESS
       # The Judge's Rubric - CHAIN OF THOUGHT ENABLED
        self.grading_prompt = ChatPromptTemplate.from_template(
            """You are an expert Financial Examiner grading an AI analyst.
            
            Compare the STUDENT ANSWER against the GROUND TRUTH for the given QUESTION.
            
            QUESTION: {question}
            
            GROUND TRUTH: {ground_truth}
            
            STUDENT ANSWER: {generated_answer}
            
            CRITICAL GRADING INSTRUCTIONS:
            1. **Ignore Formatting:** The Student Answer might be raw JSON, a list of tuples, or unformatted text. This is acceptable.
            2. **Fact Matching:** - Identify the specific numbers, dates, and entities in the GROUND TRUTH.
               - Search for these specific values in the STUDENT ANSWER.
               - If the values match (e.g., Ground Truth says "$5B" and Student says "Amount: 5,000,000,000"), give FULL CREDIT for Accuracy.
            3. **Partial Credit:** If the student finds *some* relevant data but misses others, score 3 or 4. If they find *all* relevant data, score 5.
            
            Grade the student on a scale of 1-5 for these metrics:
            
            1. Financial Accuracy: Are the correct numbers/entities present in the raw output? (5 = Yes, 1 = No)
            2. Comprehensiveness: Does the raw output cover the scope of the question?
            3. Diversity: Does it retrieve data from the correct entities (e.g. both Apple AND Tesla)?
            5. Directness: (Be lenient here - raw data is verbose) Is the signal-to-noise ratio acceptable?
            
            Output strictly valid JSON matching this schema:
            {format_instructions}
            """
        )

    def evaluate_response(self, question, ground_truth, generated_answer):
        """Runs the LLM-as-a-Judge"""
        try:
            chain = self.grading_prompt | self.judge_llm | self.parser
            result = chain.invoke({
                "question": question,
                "ground_truth": ground_truth,
                "generated_answer": generated_answer,
                "format_instructions": self.parser.get_format_instructions()
            })
            return result
        except Exception as e:
            logger.error(f"Grading failed: {e}")
            return {
                "financial_accuracy": 0, "comprehensiveness": 0, 
                "diversity": 0, "empowerment": 0, 
                "directness": 0, "reasoning": "Grading Error"
            }

    def run_full_evaluation(self, input_path, output_path):
        """Main loop"""
        logger.info(f"📂 Loading dataset: {input_path}")
        try:
            df = pd.read_csv(input_path)
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
            return

        results = []
        total = len(df)
        
        logger.info(f"🚀 Starting evaluation of {total} questions...")

        for index, row in df.iterrows():
            question = row['question']
            ground_truth = row['ground_truth']
            q_type = row.get('question_type', 'General')
            
            print(f"\n[{index+1}/{total}] Processing ({q_type}): {question[:50]}...")
            
            # 1. Get Answer from GraphRAG
            rag_start = time.time()
            rag_result = self.rag_engine.query(question, verbose=False, use_cache=True)
            rag_time = time.time() - rag_start
            
            generated_answer = rag_result.get('answer', "Error generating answer")
            cypher_query = rag_result.get('cypher', "")
            
            # 2. Grade the Answer
            grades = self.evaluate_response(question, ground_truth, generated_answer)
            
            # 3. Store Data
            result_row = {
                "question_id": index,
                "question_type": q_type,
                "question": question,
                "ground_truth": ground_truth,
                "generated_answer": generated_answer,
                "cypher_query": cypher_query,
                "latency_seconds": round(rag_time, 2),
                **grades  # Unpack scores
            }
            results.append(result_row)
            
            # Autosave every 5 questions
            if (index + 1) % 5 == 0:
                pd.DataFrame(results).to_csv(output_path, index=False)
                logger.info(f"💾 Autosaved progress to {output_path}")

        # Final Save
        final_df = pd.DataFrame(results)
        final_df.to_csv(output_path, index=False)
        logger.info(f"✅ Evaluation Complete! Results saved to {output_path}")
        
        # Print Summary Stats
        print("\n" + "="*50)
        print("📊 FINAL SCORECARD (AVERAGES)")
        print("="*50)
        print(final_df[['financial_accuracy', 'comprehensiveness', 'diversity', 'empowerment', 'directness']].mean())
        print("="*50)

    def close(self):
        self.rag_engine.close()

if __name__ == "__main__":
    evaluator = EvaluationEngine()
    try:
        evaluator.run_full_evaluation(INPUT_FILE, OUTPUT_FILE)
    finally:
        evaluator.close()
