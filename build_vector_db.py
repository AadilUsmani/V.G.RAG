import os
from pathlib import Path
from typing import List
import time
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import warnings
warnings.filterwarnings('ignore')


class VectorDBBuilder:
    """Optimized Vector Database Builder for SEC Filings"""
    
    def __init__(
        self, 
        data_path: str = "clean_data",
        db_path: str = "vector_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        batch_size: int = 100,
        embedding_model: str = "text-embedding-3-small",
        rate_limit_delay: float = 0.1
        ):
        """
        Initialize the Vector DB Builder
        
        Args:
            data_path: Directory containing cleaned text files
            db_path: Directory to store the vector database
            chunk_size: Size of text chunks for embeddings
            chunk_overlap: Overlap between consecutive chunks
            batch_size: Number of documents to process in each batch (default: 100)
            embedding_model: OpenAI embedding model name (e.g., 'text-embedding-3-small')
            rate_limit_delay: Delay in seconds between batches to respect API rate limits (default: 0.1s)
        """
        self.data_path = Path(data_path)
        self.db_path = Path(db_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.embedding_model = embedding_model
        self.rate_limit_delay = rate_limit_delay
        
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        # Validate configuration
        self._validate_config()
        
    def _validate_config(self):
        """Validate configuration and environment"""
        if not self.api_key:
            raise ValueError(
                "❌ OPENAI_API_KEY not found! "
                "Please add OPENAI_API_KEY to your .env file."
            )
        
        print(f"✅ OpenAI API key loaded")
        
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"❌ Data directory '{self.data_path}' not found! "
                f"Please ensure your cleaned files are in this directory."
            )
        
        # Create DB directory if it doesn't exist
        self.db_path.mkdir(parents=True, exist_ok=True)
    
    def _load_documents_batch(self, files: List[Path]) -> List[Document]:
        """
        Load documents from files with error handling
        
        Args:
            files: List of file paths to load
            
        Returns:
            List of loaded documents with metadata
        """
        documents = []
        
        for file in files:
            try:
                loader = TextLoader(str(file), encoding="utf-8")
                docs = loader.load()
                
                # Add enhanced metadata
                for doc in docs:
                    doc.metadata.update({
                        "source": file.name,
                        "file_path": str(file),
                        "file_size_kb": file.stat().st_size / 1024
                    })
                
                documents.extend(docs)
                
            except Exception as e:
                print(f"⚠️  Warning: Failed to load {file.name}: {e}")
                continue
        
        return documents
    
    def load_all_documents(self) -> List[Document]:
        """
        Load all text documents from data directory
        
        Returns:
            List of all loaded documents
        """
        print("📂 Loading documents from disk...")
        
        # Get all .txt files
        txt_files = list(self.data_path.glob("*.txt"))
        
        if not txt_files:
            raise FileNotFoundError(
                f"❌ No .txt files found in '{self.data_path}'! "
                f"Please run the cleaning script first."
            )
        
        print(f"📄 Found {len(txt_files)} files to process")
        
        # Load all documents
        all_documents = self._load_documents_batch(txt_files)
        
        print(f"✅ Loaded {len(all_documents)} documents successfully")
        
        return all_documents
    
    def create_chunks(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks for embedding
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        print(f"✂️  Splitting documents into chunks...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        
        print(f"✅ Created {len(chunks)} chunks")
        print(f"   Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
        
        return chunks
    
    def build_vector_store(self, chunks: List[Document]) -> Chroma:
        """
        Create vector store with embeddings using batch processing optimized for OpenAI API.
        Implements resumable processing with skip logic.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Chroma vector store instance
        """
        print("🧠 Creating embeddings with OpenAI API...")
        print(f"   Total chunks to process: {len(chunks)}")
        print(f"   Batch size: {self.batch_size}")
        
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
        print(f"   Total batches: {total_batches}")
        print(f"   Embedding model: {self.embedding_model}")
        print(f"   Rate limit: {self.rate_limit_delay}s between batches")
        
        # Open progress log file
        progress_log_path = self.db_path / "progress.log"
        def log_progress(msg):
            with open(progress_log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        
        # RESUMABLE MODE: Skip clean slate to preserve progress on restarts
        # Uncomment the block below if you want a completely fresh database:
        # if self.db_path.exists():
        #     print("\n🧹 Clearing old database for research consistency...")
        #     log_progress("🧹 Clearing old database for clean run")
        #     shutil.rmtree(self.db_path)
        #     print("   ✅ Old database cleared")
        
        # Create directory if it doesn't exist
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Test the OpenAI API with a sample embedding
        print(f"\n🧪 Testing embedding model: {self.embedding_model}")
        try:
            test_embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                openai_api_key=self.api_key
            )
            test_embedding = test_embeddings.embed_query("test")
            print(f"✅ Embedding model working! Dimension: {len(test_embedding)}")
            log_progress(f"✅ Embedding model working! Dimension: {len(test_embedding)}")
        except Exception as e:
            print(f"\n❌ EMBEDDING MODEL ERROR:")
            print(f"   {str(e)}")
            print("\n💡 Possible solutions:")
            print("   1. Check if your OPENAI_API_KEY is valid")
            print("   2. Verify you have credits in your OpenAI account")
            log_progress(f"❌ EMBEDDING MODEL ERROR: {str(e)}")
            raise
        
        # Initialize vector store with OpenAI embeddings
        embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            openai_api_key=self.api_key
        )
        vector_db = Chroma(
            persist_directory=str(self.db_path),
            embedding_function=embeddings
        )
        
        # Process in batches with rate limiting
        successful_batches = 0
        failed_batches = 0
        
        estimated_time = (total_batches * self.rate_limit_delay) / 60
        print(f"\n⏱️  Estimated completion time: ~{estimated_time:.1f} minutes")
        print(f"📊 Progress:\n")
        log_progress(f"\nStarting batch processing: {total_batches} batches")
        
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            
            # SKIP LOGIC: Check if this batch is already processed
            current_db_count = vector_db._collection.count()
            if i < current_db_count:
                skipped_msg = f"⏩ Skipping batch {batch_num} (already in DB: {current_db_count} chunks)"
                print(f"   {skipped_msg}")
                log_progress(skipped_msg)
                continue
            
            # Progress indicator
            progress_pct = (batch_num / total_batches) * 100
            remaining = total_batches - batch_num + 1
            msg = f"[{batch_num}/{total_batches}] ({progress_pct:.1f}%) Processing {len(batch)} chunks (DB: {current_db_count}/{len(chunks)})"
            print(f"   {msg}")
            log_progress(msg)
            
            # Retry logic for rate limit errors with exponential backoff
            max_retries = 5
            retry_count = 0
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    # Add documents to vector store with OpenAI embeddings
                    vector_db.add_documents(batch)
                    
                    successful_batches += 1
                    log_progress(f"✅ Batch {batch_num} succeeded")
                    success = True
                    
                    # Small delay to respect rate limits (except for last batch)
                    if i + self.batch_size < len(chunks):
                        time.sleep(self.rate_limit_delay)
                    
                except Exception as e:
                    error_msg = str(e)
                    
                    # Check if it's a rate limit error (429 or RESOURCE_EXHAUSTED)
                    if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg or "Quota exceeded" in error_msg:
                        retry_count += 1
                        if retry_count < max_retries:
                            # Exponential backoff: 30s, 60s, 120s, 240s, 480s
                            wait_time = (2 ** retry_count) * 30
                            print(f"\n   ⚠️ Rate limit hit. Waiting {wait_time}s before retry (attempt {retry_count}/{max_retries})...")
                            log_progress(f"⚠️ Rate limit hit. Waiting {wait_time}s (retry {retry_count}/{max_retries})")
                            time.sleep(wait_time)
                        else:
                            print(f"\n   ❌ Batch {batch_num} failed after {max_retries} retries")
                            log_progress(f"❌ Batch {batch_num} failed after {max_retries} retries")
                            failed_batches += 1
                            break
                    
                    # Check if it's a model error (fatal)
                    elif "NOT_FOUND" in error_msg or "not found" in error_msg.lower():
                        print(f"\n❌ MODEL ERROR in batch {batch_num}:")
                        print(f"   {error_msg}")
                        print("\n💡 The embedding model is incorrect or unavailable.")
                        log_progress(f"❌ MODEL ERROR in batch {batch_num}: {error_msg}")
                        raise  # Stop execution on model errors
                    
                    # Other errors
                    else:
                        print(f"\n⚠️  Warning: Batch {batch_num} failed: {error_msg}")
                        log_progress(f"⚠️ Batch {batch_num} failed: {error_msg}")
                        failed_batches += 1
                        break
            
            # Check if we've had too many failures
            if failed_batches > 20:
                print(f"\n❌ Too many failures ({failed_batches}). Stopping.")
                log_progress(f"❌ Too many failures ({failed_batches}). Stopping.")
                raise Exception("Too many batch failures. Check your API key and daily quota.")
        
        print(f"\n📊 Batch Processing Summary:")
        print(f"   ✅ Successful: {successful_batches}/{total_batches}")
        print(f"   ❌ Failed: {failed_batches}/{total_batches}")
        log_progress(f"\nBatch Processing Summary: Successful: {successful_batches}/{total_batches}, Failed: {failed_batches}/{total_batches}")
        
        return vector_db
    
    def build(self):
        """Main method to build the complete vector database"""
        print("="*80)
        print("🚀 STARTING VECTOR DATABASE BUILD")
        print("="*80)
        print(f"📁 Data Path: {self.data_path}")
        print(f"💾 DB Path: {self.db_path}")
        print(f"📏 Chunk Size: {self.chunk_size}")
        print(f"🔄 Chunk Overlap: {self.chunk_overlap}")
        print(f"🤖 Embedding Model: {self.embedding_model}")
        print("="*80 + "\n")
        
        try:
            # Step 1: Load documents
            documents = self.load_all_documents()
            
            # Step 2: Create chunks
            chunks = self.create_chunks(documents)
            
            # Step 3: Build vector store
            vector_db = self.build_vector_store(chunks)
            
            # Summary
            print("\n" + "="*80)
            print("🎉 VECTOR DATABASE BUILT SUCCESSFULLY!")
            print("="*80)
            print(f"📊 Statistics:")
            print(f"   • Documents processed: {len(documents)}")
            print(f"   • Total chunks: {len(chunks)}")
            print(f"   • Database location: {self.db_path}")
            print(f"   • Collection size: {vector_db._collection.count()} vectors")
            print("="*80)
            
            return vector_db
            
        except Exception as e:
            print(f"\n❌ ERROR: Vector DB build failed!")
            print(f"   {str(e)}")
            raise


def main():
    """Main execution function"""
    # Configuration - easily customizable
    config = {
        "data_path": "clean_data",
        "db_path": "vector_db",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "batch_size": 100,  # Process 100 chunks per batch
        "embedding_model": "text-embedding-3-small",  # OpenAI's efficient embedding model
        "rate_limit_delay": 0.1  # 0.1 seconds = very fast with OpenAI (500 req/min default limit)
    }
    
    # Build the vector database
    builder = VectorDBBuilder(**config)
    vector_db = builder.build()
    
    # Optional: Test the vector store
    print("\n🧪 Testing vector store with sample query...")
    results = vector_db.similarity_search("revenue", k=3)
    print(f"✅ Test successful! Found {len(results)} relevant chunks")
    

if __name__ == "__main__":
    main()