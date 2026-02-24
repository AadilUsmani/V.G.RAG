"""
Synthetic Question Generation for RAG Evaluation Dataset
Generates high-quality questions from SEC 10-K filings using OpenAI GPT-4o-mini
"""

import os
import random
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QuestionRecord:
    """Data class for storing question information"""
    question_type: str
    question: str
    ground_truth: str
    source_file: str
    chunk_index: Optional[int] = None


class Config:
    """Configuration constants"""
    DATA_PATH = Path("clean_data")
    OUTPUT_FILE = "evaluation_dataset_openai.csv"
    NUM_QUESTIONS = 35  # Total new questions to generate
    NUM_COMPARISON_QUESTIONS = 15  # Target number of comparison questions
    CHUNK_SIZE = 3000
    CHUNK_OVERLAP = 200
    MAX_CONTENT_LENGTH = 4000
    MODEL_NAME = "gpt-4o-mini"
    TEMPERATURE = 0.7
    MAX_WORKERS = 3  # Parallel processing threads
    MAX_RETRIES = 2
    APPEND_MODE = True  # Append to existing CSV instead of overwriting
    

class QuestionGenerator:
    """Handles synthetic question generation from documents"""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self._validate_environment()
        self.llm = ChatOpenAI(
            model=self.config.MODEL_NAME,
            temperature=self.config.TEMPERATURE
        )
        
    def _validate_environment(self) -> None:
        """Validate environment setup"""
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "❌ OPENAI_API_KEY not found! Please check your .env file."
            )
        
        if not self.config.DATA_PATH.exists():
            raise FileNotFoundError(
                f"❌ Data directory not found: {self.config.DATA_PATH}"
            )
    
    def load_documents(self) -> List:
        """Load all text documents from data directory"""
        documents = []
        files = list(self.config.DATA_PATH.glob("*.txt"))
        
        if not files:
            raise ValueError(f"❌ No .txt files found in {self.config.DATA_PATH}")
        
        logger.info(f"📂 Loading {len(files)} files...")
        
        for file_path in files:
            try:
                loader = TextLoader(str(file_path), encoding="utf-8")
                documents.extend(loader.load())
            except Exception as e:
                logger.warning(f"⚠️ Skipping {file_path.name}: {e}")
        
        if not documents:
            raise ValueError("❌ No documents loaded successfully")
        
        logger.info(f"✅ Successfully loaded {len(documents)} documents")
        return documents
    
    def create_chunks(self, documents: List) -> List:
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"📚 Created {len(chunks)} chunks from documents")
        return chunks
    
    def _group_chunks_by_company(self, chunks: List) -> Dict[str, List]:
        """Group chunks by company (AAPL or TSLA)"""
        grouped = {"AAPL": [], "TSLA": []}
        for chunk in chunks:
            source = chunk.metadata.get('source', '')
            if 'AAPL' in source:
                grouped["AAPL"].append(chunk)
            elif 'TSLA' in source:
                grouped["TSLA"].append(chunk)
        logger.info(f"📊 Grouped: {len(grouped['AAPL'])} AAPL chunks, {len(grouped['TSLA'])} TSLA chunks")
        return grouped
    
    def _create_prompt(self, text_content: str) -> str:
        """Create the generation prompt for LLM"""
        return f"""You are a Financial Expert creating an evaluation dataset for a RAG system.
Analyze the following text from SEC 10-K filings (Apple/Tesla):

--- TEXT START ---
{text_content}
--- TEXT END ---

Task: Create ONE high-quality question based *strictly* on this text.

Criteria:
1. **Reasoning Question (Preferred):** "How did X affect Y?" or "Compare X and Y"
2. **Financial Fact:** "What was the exact revenue for X in 2024?"
3. Ignore 'us-gaap' tags, focus on human-readable numbers and text
4. If text is junk (just headers/tables without context), reply "SKIP"

Output Format (must follow exactly):
TYPE: [Reasoning OR Fact]
QUESTION: [Your Question]
ANSWER: [Ground Truth Answer from text]
"""
    
    def _create_comparison_prompt(self, apple_text: str, tesla_text: str) -> str:
        """Create prompt for comparison questions across both companies"""
        return f"""You are a Financial Expert creating comparison questions for RAG evaluation.

You have excerpts from BOTH Apple and Tesla 10-K filings:

--- APPLE TEXT ---
{apple_text}

--- TESLA TEXT ---
{tesla_text}

Task: Create ONE comparison question that requires information from BOTH companies.

Comparison Themes (choose one):
- Revenue/Profit growth comparison
- R&D spending as % of revenue
- Legal/regulatory risk differences
- Supply chain strategies
- Geographic revenue distribution
- Capital expenditure priorities
- Liquidity positions (cash, assets)
- Employee/workforce metrics

Criteria:
1. Question MUST require synthesizing data from BOTH companies
2. Ground truth answer MUST mention specific data from BOTH
3. Focus on meaningful financial/strategic differences
4. Use exact numbers when available
5. If texts don't allow meaningful comparison, reply "SKIP"

Output Format (must follow exactly):
TYPE: Comparison
QUESTION: [Your comparison question]
ANSWER: [Ground truth mentioning BOTH Apple and Tesla with specific data]
"""
    
    def _parse_response(self, response: str) -> Optional[Dict[str, str]]:
        """Parse LLM response into structured format"""
        if not response or "SKIP" in response.upper():
            return None
        
        if "QUESTION:" not in response:
            logger.warning("Response missing QUESTION field")
            return None
        
        parsed = {
            "question_type": "Reasoning",  # default
            "question": "",
            "answer": ""
        }
        
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("TYPE:"):
                parsed["question_type"] = line.replace("TYPE:", "").strip()
            elif line.startswith("QUESTION:"):
                parsed["question"] = line.replace("QUESTION:", "").strip()
            elif line.startswith("ANSWER:"):
                parsed["answer"] = line.replace("ANSWER:", "").strip()
        
        # Validate parsed data
        if not parsed["question"] or not parsed["answer"]:
            logger.warning("Parsed response missing question or answer")
            return None
        
        return parsed
    
    def _generate_single_question(
        self, 
        chunk, 
        chunk_index: int
    ) -> Optional[QuestionRecord]:
        """Generate a single question from a chunk"""
        source_file = chunk.metadata.get('source', 'unknown')
        text_content = chunk.page_content[:self.config.MAX_CONTENT_LENGTH]
        
        # Skip if content is too short
        if len(text_content.strip()) < 100:
            return None
        
        prompt = self._create_prompt(text_content)
        
        for attempt in range(self.config.MAX_RETRIES):
            try:
                response = self.llm.invoke(prompt).content
                parsed = self._parse_response(response)
                
                if parsed:
                    return QuestionRecord(
                        question_type=parsed["question_type"],
                        question=parsed["question"],
                        ground_truth=parsed["answer"],
                        source_file=source_file,
                        chunk_index=chunk_index
                    )
                break  # SKIP response, don't retry
                
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1} failed for chunk {chunk_index}: {e}"
                )
                if attempt == self.config.MAX_RETRIES - 1:
                    logger.error(f"Failed to process chunk {chunk_index}: {e}")
        
        return None
    
    def _generate_comparison_question(
        self,
        apple_chunk,
        tesla_chunk,
        apple_index: int,
        tesla_index: int
    ) -> Optional[QuestionRecord]:
        """Generate a comparison question from Apple and Tesla chunks"""
        apple_text = apple_chunk.page_content[:self.config.MAX_CONTENT_LENGTH // 2]
        tesla_text = tesla_chunk.page_content[:self.config.MAX_CONTENT_LENGTH // 2]
        
        # Skip if either content is too short
        if len(apple_text.strip()) < 100 or len(tesla_text.strip()) < 100:
            return None
        
        prompt = self._create_comparison_prompt(apple_text, tesla_text)
        
        for attempt in range(self.config.MAX_RETRIES):
            try:
                response = self.llm.invoke(prompt).content
                parsed = self._parse_response(response)
                
                if parsed and parsed.get("question_type") == "Comparison":
                    apple_source = apple_chunk.metadata.get('source', 'AAPL')
                    tesla_source = tesla_chunk.metadata.get('source', 'TSLA')
                    
                    return QuestionRecord(
                        question_type="Comparison",
                        question=parsed["question"],
                        ground_truth=parsed["answer"],
                        source_file=f"{apple_source}; {tesla_source}",
                        chunk_index=apple_index  # Primary chunk index
                    )
                break
                
            except Exception as e:
                logger.warning(
                    f"Comparison attempt {attempt + 1} failed: {e}"
                )
                if attempt == self.config.MAX_RETRIES - 1:
                    logger.error(f"Failed to generate comparison question: {e}")
        
        return None
    
    def generate_questions(self) -> pd.DataFrame:
        """Main generation workflow with comparison questions"""
        logger.info("🤖 Starting Synthetic Question Generation...")
        
        # Load and chunk documents
        documents = self.load_documents()
        chunks = self.create_chunks(documents)
        grouped_chunks = self._group_chunks_by_company(chunks)
        
        dataset = []
        
        # Generate Comparison Questions
        logger.info(f"🔄 Generating {self.config.NUM_COMPARISON_QUESTIONS} comparison questions...")
        comparison_pairs = min(
            self.config.NUM_COMPARISON_QUESTIONS,
            min(len(grouped_chunks["AAPL"]), len(grouped_chunks["TSLA"]))
        )
        
        apple_indices = random.sample(range(len(grouped_chunks["AAPL"])), comparison_pairs)
        tesla_indices = random.sample(range(len(grouped_chunks["TSLA"])), comparison_pairs)
        
        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    self._generate_comparison_question,
                    grouped_chunks["AAPL"][apple_idx],
                    grouped_chunks["TSLA"][tesla_idx],
                    apple_idx,
                    tesla_idx
                ): (apple_idx, tesla_idx) for apple_idx, tesla_idx in zip(apple_indices, tesla_indices)
            }
            
            with tqdm(total=len(futures), desc="Comparison questions") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        dataset.append(asdict(result))
                    pbar.update(1)
        
        # Generate Regular Questions (Fact/Reasoning)
        num_regular = self.config.NUM_QUESTIONS - len(dataset)
        if num_regular > 0:
            logger.info(f"📝 Generating {num_regular} regular questions...")
            num_to_sample = min(num_regular, len(chunks))
            selected_indices = random.sample(range(len(chunks)), num_to_sample)
            
            with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
                futures = {
                    executor.submit(
                        self._generate_single_question, 
                        chunks[i], 
                        i
                    ): i for i in selected_indices
                }
                
                with tqdm(total=len(futures), desc="Regular questions") as pbar:
                    for future in as_completed(futures):
                        result = future.result()
                        if result:
                            dataset.append(asdict(result))
                        pbar.update(1)
        
        if not dataset:
            raise ValueError("❌ No questions generated successfully")
        
        df = pd.DataFrame(dataset)
        logger.info(f"✅ Generated {len(df)} questions successfully")
        logger.info(f"   - Comparison: {len(df[df['question_type'] == 'Comparison'])}")
        logger.info(f"   - Reasoning: {len(df[df['question_type'] == 'Reasoning'])}")
        logger.info(f"   - Fact: {len(df[df['question_type'] == 'Fact'])}")
        
        return df
    
    def save_dataset(self, df: pd.DataFrame) -> None:
        """Save dataset to CSV (append or overwrite based on config)"""
        output_path = Path(self.config.OUTPUT_FILE)
        
        if self.config.APPEND_MODE and output_path.exists():
            logger.info(f"📂 Loading existing dataset from {output_path}")
            existing_df = pd.read_csv(output_path)
            logger.info(f"   Existing questions: {len(existing_df)}")
            
            # Combine datasets
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_csv(output_path, index=False)
            logger.info(f"💾 Appended {len(df)} new questions to {output_path}")
            logger.info(f"📊 Total questions now: {len(combined_df)}")
            
            # Print summary for combined dataset
            logger.info("\n📊 Combined Dataset Summary:")
            logger.info(f"Total Questions: {len(combined_df)}")
            logger.info(f"Question Types:\n{combined_df['question_type'].value_counts()}")
        else:
            df.to_csv(output_path, index=False)
            logger.info(f"💾 Dataset saved to {output_path}")
            
            # Print summary statistics
            logger.info("\n📊 Dataset Summary:")
            logger.info(f"Total Questions: {len(df)}")
            logger.info(f"Question Types:\n{df['question_type'].value_counts()}")
            logger.info(f"Sources:\n{df['source_file'].value_counts()}")


def main():
    """Main execution function"""
    try:
        generator = QuestionGenerator()
        df = generator.generate_questions()
        generator.save_dataset(df)
        
        logger.info("\n🎉 SUCCESS! Question generation complete.")
        
    except Exception as e:
        logger.error(f"\n❌ Fatal Error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()