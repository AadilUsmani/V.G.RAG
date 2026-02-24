"""
Knowledge Graph Extraction from Financial Documents

Purpose:
    Extracts entities (nodes) and relationships (edges) from SEC 10-K filings
    to build a knowledge graph for financial analysis and GraphRAG.

Workflow:
    Documents → Chunking → LLM Extraction → Validation → Graph Construction → Export

Key Features:
    - Parallel processing for faster extraction
    - Structured output validation
    - Entity deduplication and normalization
    - Graph statistics and quality metrics
    - Multiple export formats
"""

import os
import glob
import json
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field,field_validator
from langchain_text_splitters import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# SCHEMA DEFINITIONS
# ============================================================================

class Node(BaseModel):
    """Entity node in the knowledge graph"""
    id: str = Field(description="Unique normalized name of the entity (e.g., 'Apple Inc.', 'Revenue', '2024')")
    type: str = Field(description="Entity type: Company, Metric, Year, Amount, Product, Person, Location, Risk, Event")
    
    @field_validator('id')
    def normalize_id(cls, v):
        """Normalize entity IDs for consistency"""
        return v.strip()
    
    @field_validator('type')
    def validate_type(cls, v):
        """Ensure type is valid"""
        valid_types = {
            'Company', 'Metric', 'Year', 'Amount', 'Product', 
            'Person', 'Location', 'Risk', 'Event', 'Technology', 'Other'
        }
        v_clean = v.strip()
        if v_clean not in valid_types:
            return 'Other'
        return v_clean


class Relationship(BaseModel):
    """Edge/relationship in the knowledge graph"""
    source_id: str = Field(description="ID of the source entity")
    target_id: str = Field(description="ID of the target entity")
    type: str = Field(description="Relationship type: REPORTED, INCREASED_BY, DECREASED_BY, ACQUIRED, INVESTED_IN, RELATES_TO, HAS_RISK, LOCATED_IN")
    
    @field_validator('source_id', 'target_id')
    def normalize_ids(cls, v):
        """Normalize entity IDs"""
        return v.strip()
    
    @field_validator('type')
    def normalize_type(cls, v):
        """Normalize relationship type"""
        return v.upper().replace(" ", "_").replace("-", "_")


class GraphExtraction(BaseModel):
    """Complete graph extraction from a text chunk"""
    nodes: List[Node] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for graph extraction"""
    # Paths
    DATA_DIR = "clean_data"
    OUTPUT_DIR = "graph_output"
    OUTPUT_NODES = "graph_nodes.csv"
    OUTPUT_EDGES = "graph_edges.csv"
    OUTPUT_STATS = "graph_statistics.json"
    OUTPUT_SUMMARY = "graph_summary.txt"
    
    # Model Settings
    LLM_MODEL = "gpt-4o-mini"  # Cost-effective for extraction
    TEMPERATURE = 0  # Deterministic
    
    # Chunking Settings
    CHUNK_SIZE = 3000
    CHUNK_OVERLAP = 300
    
    # Processing Settings
    MAX_WORKERS = 12  # Parallel processing
    MAX_CHUNKS_PER_FILE = None  # None = all chunks, or set to int for testing
    RATE_LIMIT_DELAY = 0.05  # Seconds between API calls
    MAX_RETRIES = 2
    
    # Validation Settings
    MIN_NODE_ID_LENGTH = 2
    MAX_NODE_ID_LENGTH = 200
    DEDUPLICATION_THRESHOLD = 0.9  # Similarity threshold for entity deduplication


# ============================================================================
# GRAPH EXTRACTION ENGINE
# ============================================================================

class GraphExtractor:
    """
    Extracts knowledge graph from financial documents
    """
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.start_time = None
        self._validate_environment()
        self._initialize_components()
        
        # Statistics tracking
        self.stats = {
            'total_chunks': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'total_nodes_extracted': 0,
            'total_edges_extracted': 0,
            'nodes_by_type': Counter(),
            'edges_by_type': Counter(),
            'files_processed': 0,
            'processing_time': 0
        }
    
    def _validate_environment(self) -> None:
        """Validate environment and paths"""
        load_dotenv()
        
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("❌ OPENAI_API_KEY not found in .env file")
        
        data_path = Path(self.config.DATA_DIR)
        if not data_path.exists():
            raise FileNotFoundError(f"❌ Data directory not found: {data_path}")
        
        # Create output directory
        output_path = Path(self.config.OUTPUT_DIR)
        output_path.mkdir(exist_ok=True)
        
        logger.info("✅ Environment validated")
    
    def _initialize_components(self) -> None:
        """Initialize LLM and text splitter"""
        # Initialize LLM with structured output
        self.llm = ChatOpenAI(
            model=self.config.LLM_MODEL,
            temperature=self.config.TEMPERATURE
        )
        self.structured_llm = self.llm.with_structured_output(GraphExtraction)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        logger.info("✅ Components initialized")
    
    def _create_extraction_prompt(self) -> ChatPromptTemplate:
        """Create the graph extraction prompt"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert financial knowledge graph extractor analyzing SEC 10-K filings.

Your task is to extract entities (nodes) and relationships (edges) from financial text.

**Entity Types (Nodes):**
- Company: Company names (e.g., "Apple Inc.", "Tesla")
- Metric: Financial metrics (e.g., "Revenue", "Net Income", "R&D Expenses")
- Year: Years or fiscal periods (e.g., "2024", "Q3 2024")
- Amount: Specific monetary values (e.g., "$394.3 billion", "15% increase")
- Product: Products or services (e.g., "iPhone", "Model 3")
- Person: People mentioned (e.g., "Tim Cook", "Elon Musk")
- Location: Geographic locations (e.g., "Cupertino", "China")
- Risk: Risk factors or challenges (e.g., "Supply chain disruption", "Regulatory risk")
- Event: Significant events (e.g., "Acquisition", "Product launch")
- Technology: Technologies mentioned (e.g., "AI", "Autonomous driving")

**Relationship Types (Edges):**
- REPORTED: Company reported a metric/amount
- INCREASED_BY: Metric increased by amount
- DECREASED_BY: Metric decreased by amount
- ACQUIRED: Company acquired another company/product
- INVESTED_IN: Company invested in technology/product
- RELATES_TO: General relationship
- HAS_RISK: Company/product has a risk factor
- LOCATED_IN: Entity is located in a location
- LED_BY: Company led by person
- MANUFACTURES: Company manufactures product

**Instructions:**
1. Extract ONLY entities explicitly mentioned in the text
2. Use consistent naming (e.g., always "Apple Inc." not "Apple" or "AAPL")
3. For amounts, include the full value (e.g., "$394.3 billion" not just "394")
4. Create relationships that capture meaningful financial insights
5. Avoid creating nodes for generic terms or articles
6. Focus on financial, operational, and strategic information

**Quality Guidelines:**
- Minimum 3 nodes and 2 relationships per chunk (if sufficient content)
- Ensure relationship source/target IDs match node IDs exactly
- Use specific relationship types rather than generic RELATES_TO when possible
"""),
            ("human", "Extract the knowledge graph from this financial text:\n\n{text}")
        ])
    
    def extract_from_chunk(
        self, 
        chunk: str, 
        chunk_index: int,
        filename: str
    ) -> Optional[GraphExtraction]:
        """
        Extract graph data from a single chunk
        
        Returns:
            GraphExtraction object or None if extraction fails
        """
        # Skip very short chunks
        if len(chunk.strip()) < 100:
            return None
        
        prompt = self._create_extraction_prompt()
        chain = prompt | self.structured_llm
        
        for attempt in range(self.config.MAX_RETRIES):
            try:
                result = chain.invoke({"text": chunk})
                
                # Validate extraction
                if self._validate_extraction(result):
                    return result
                else:
                    logger.warning(f"Invalid extraction for chunk {chunk_index}")
                    return None
                    
            except Exception as e:
                if attempt == self.config.MAX_RETRIES - 1:
                    logger.error(f"Failed to extract from chunk {chunk_index}: {e}")
                    return None
                time.sleep(0.5)  # Brief retry delay
        
        return None
    
    def _validate_extraction(self, extraction: GraphExtraction) -> bool:
        """Validate extracted graph data"""
        if not extraction.nodes:
            return False
        
        # Check node ID lengths
        for node in extraction.nodes:
            if len(node.id) < self.config.MIN_NODE_ID_LENGTH:
                return False
            if len(node.id) > self.config.MAX_NODE_ID_LENGTH:
                return False
        
        # Check that relationship IDs reference existing nodes
        node_ids = {node.id for node in extraction.nodes}
        for rel in extraction.relationships:
            if rel.source_id not in node_ids or rel.target_id not in node_ids:
                # Create missing nodes as "Other" type
                if rel.source_id not in node_ids:
                    extraction.nodes.append(Node(id=rel.source_id, type="Other"))
                if rel.target_id not in node_ids:
                    extraction.nodes.append(Node(id=rel.target_id, type="Other"))
        
        return True
    
    def process_file(self, file_path: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Process a single file and extract graph data
        
        Returns:
            (nodes_list, edges_list)
        """
        filename = os.path.basename(file_path)
        logger.info(f"📄 Processing: {filename}")
        
        # Load file
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            logger.error(f"Failed to read {filename}: {e}")
            return [], []
        
        # Split into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Limit chunks if configured
        if self.config.MAX_CHUNKS_PER_FILE:
            chunks = chunks[:self.config.MAX_CHUNKS_PER_FILE]
        
        total_chunks = len(chunks)
        logger.info(f"   🔹 Split into {total_chunks} chunks")
        
        nodes_list = []
        edges_list = []
        
        # Process chunks
        for i, chunk in enumerate(tqdm(chunks, desc=f"  {filename}", leave=False)):
            self.stats['total_chunks'] += 1
            
            extraction = self.extract_from_chunk(chunk, i, filename)
            
            if extraction:
                self.stats['successful_extractions'] += 1
                
                # Collect nodes
                for node in extraction.nodes:
                    nodes_list.append({
                        "id": node.id,
                        "type": node.type,
                        "source_file": filename,
                        "chunk_index": i
                    })
                    self.stats['nodes_by_type'][node.type] += 1
                
                # Collect edges
                for rel in extraction.relationships:
                    edges_list.append({
                        "source": rel.source_id,
                        "target": rel.target_id,
                        "type": rel.type,
                        "source_file": filename,
                        "chunk_index": i
                    })
                    self.stats['edges_by_type'][rel.type] += 1
            else:
                self.stats['failed_extractions'] += 1
            
            # Rate limiting
            time.sleep(self.config.RATE_LIMIT_DELAY)
        
        self.stats['total_nodes_extracted'] += len(nodes_list)
        self.stats['total_edges_extracted'] += len(edges_list)
        
        return nodes_list, edges_list
    
    def process_all_files(self, parallel: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process all files in data directory
        
        Args:
            parallel: Use parallel processing
        
        Returns:
            (nodes_df, edges_df)
        """
        self.start_time = datetime.now()
        
        # Find all text files
        files = glob.glob(os.path.join(self.config.DATA_DIR, "*.txt"))
        
        if not files:
            raise ValueError(f"No .txt files found in {self.config.DATA_DIR}")
        
        logger.info(f"📂 Found {len(files)} text files")
        
        all_nodes = []
        all_edges = []
        
        if parallel and self.config.MAX_WORKERS > 1:
            all_nodes, all_edges = self._process_parallel(files)
        else:
            all_nodes, all_edges = self._process_sequential(files)
        
        # Create DataFrames
        df_nodes = pd.DataFrame(all_nodes)
        df_edges = pd.DataFrame(all_edges)
        
        # Deduplicate
        df_nodes = self._deduplicate_nodes(df_nodes)
        df_edges = self._deduplicate_edges(df_edges)
        
        # Update stats
        self.stats['files_processed'] = len(files)
        self.stats['processing_time'] = (datetime.now() - self.start_time).total_seconds()
        
        logger.info(f"✅ Extraction complete")
        logger.info(f"   Nodes: {len(df_nodes)}, Edges: {len(df_edges)}")
        
        return df_nodes, df_edges
    
    def _process_sequential(self, files: List[str]) -> Tuple[List[Dict], List[Dict]]:
        """Process files sequentially"""
        all_nodes = []
        all_edges = []
        
        for file_path in files:
            nodes, edges = self.process_file(file_path)
            all_nodes.extend(nodes)
            all_edges.extend(edges)
        
        return all_nodes, all_edges
    
    def _process_parallel(self, files: List[str]) -> Tuple[List[Dict], List[Dict]]:
        """Process files in parallel"""
        all_nodes = []
        all_edges = []
        
        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            futures = {executor.submit(self.process_file, f): f for f in files}
            
            for future in as_completed(futures):
                nodes, edges = future.result()
                all_nodes.extend(nodes)
                all_edges.extend(edges)
        
        return all_nodes, all_edges
    
    def _deduplicate_nodes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate nodes while preserving metadata"""
        if df.empty:
            return df
        
        # Group by id and type, keep first occurrence but aggregate metadata
        df_dedup = df.groupby(['id', 'type']).agg({
            'source_file': lambda x: '|'.join(sorted(set(x))),
            'chunk_index': 'first'
        }).reset_index()
        
        logger.info(f"   Deduplication: {len(df)} → {len(df_dedup)} nodes")
        
        return df_dedup
    
    def _deduplicate_edges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate edges"""
        if df.empty:
            return df
        
        # Remove exact duplicates
        df_dedup = df.drop_duplicates(subset=['source', 'target', 'type'])
        
        logger.info(f"   Deduplication: {len(df)} → {len(df_dedup)} edges")
        
        return df_dedup
    
    def save_graph(
        self, 
        nodes_df: pd.DataFrame, 
        edges_df: pd.DataFrame
    ) -> None:
        """Save graph data in multiple formats"""
        output_dir = Path(self.config.OUTPUT_DIR)
        
        # 1. CSV files
        nodes_path = output_dir / self.config.OUTPUT_NODES
        edges_path = output_dir / self.config.OUTPUT_EDGES
        
        nodes_df.to_csv(nodes_path, index=False)
        edges_df.to_csv(edges_path, index=False)
        
        logger.info(f"💾 Saved nodes to: {nodes_path}")
        logger.info(f"💾 Saved edges to: {edges_path}")
        
        # 2. JSON format (for graph databases)
        self._save_json_format(nodes_df, edges_df, output_dir)
        
        # 3. Statistics
        self._save_statistics(nodes_df, edges_df, output_dir)
        
        # 4. Summary report
        self._generate_summary_report(nodes_df, edges_df, output_dir)
    
    def _save_json_format(
        self, 
        nodes_df: pd.DataFrame, 
        edges_df: pd.DataFrame,
        output_dir: Path
    ) -> None:
        """Save graph in JSON format for graph databases"""
        graph_data = {
            "nodes": nodes_df.to_dict(orient='records'),
            "edges": edges_df.to_dict(orient='records'),
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "num_nodes": len(nodes_df),
                "num_edges": len(edges_df),
                "model": self.config.LLM_MODEL
            }
        }
        
        json_path = output_dir / "graph_data.json"
        with open(json_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        logger.info(f"💾 Saved JSON to: {json_path}")
    
    def _save_statistics(
        self, 
        nodes_df: pd.DataFrame, 
        edges_df: pd.DataFrame,
        output_dir: Path
    ) -> None:
        """Save graph statistics"""
        # Calculate graph metrics
        node_degrees = edges_df['source'].value_counts().to_dict()
        
        stats = {
            "extraction_stats": self.stats,
            "graph_metrics": {
                "num_nodes": len(nodes_df),
                "num_edges": len(edges_df),
                "avg_degree": len(edges_df) / len(nodes_df) if len(nodes_df) > 0 else 0,
                "nodes_by_type": dict(nodes_df['type'].value_counts()),
                "edges_by_type": dict(edges_df['type'].value_counts()),
                "top_connected_nodes": dict(Counter(node_degrees).most_common(10)),
                "files_covered": sorted(nodes_df['source_file'].unique().tolist())
            },
            "quality_metrics": {
                "extraction_success_rate": self.stats['successful_extractions'] / max(self.stats['total_chunks'], 1),
                "avg_nodes_per_chunk": self.stats['total_nodes_extracted'] / max(self.stats['successful_extractions'], 1),
                "avg_edges_per_chunk": self.stats['total_edges_extracted'] / max(self.stats['successful_extractions'], 1)
            }
        }
        
        stats_path = output_dir / self.config.OUTPUT_STATS
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"📊 Saved statistics to: {stats_path}")
    
    def _generate_summary_report(
        self, 
        nodes_df: pd.DataFrame, 
        edges_df: pd.DataFrame,
        output_dir: Path
    ) -> None:
        """Generate human-readable summary report"""
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("KNOWLEDGE GRAPH EXTRACTION REPORT")
        report_lines.append("=" * 70)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Model: {self.config.LLM_MODEL}")
        
        # Processing stats
        report_lines.append("\n" + "-" * 70)
        report_lines.append("PROCESSING STATISTICS")
        report_lines.append("-" * 70)
        report_lines.append(f"Files Processed: {self.stats['files_processed']}")
        report_lines.append(f"Total Chunks: {self.stats['total_chunks']}")
        report_lines.append(f"Successful Extractions: {self.stats['successful_extractions']}")
        report_lines.append(f"Failed Extractions: {self.stats['failed_extractions']}")
        report_lines.append(f"Success Rate: {self.stats['successful_extractions']/max(self.stats['total_chunks'],1)*100:.1f}%")
        report_lines.append(f"Processing Time: {self.stats['processing_time']:.2f} seconds")
        
        # Graph structure
        report_lines.append("\n" + "-" * 70)
        report_lines.append("GRAPH STRUCTURE")
        report_lines.append("-" * 70)
        report_lines.append(f"Total Nodes: {len(nodes_df)}")
        report_lines.append(f"Total Edges: {len(edges_df)}")
        report_lines.append(f"Average Degree: {len(edges_df)/len(nodes_df):.2f}" if len(nodes_df) > 0 else "N/A")
        
        # Nodes by type
        report_lines.append("\n" + "-" * 70)
        report_lines.append("NODES BY TYPE")
        report_lines.append("-" * 70)
        for node_type, count in nodes_df['type'].value_counts().items():
            report_lines.append(f"{node_type:15s}: {count:6d} ({count/len(nodes_df)*100:5.1f}%)")
        
        # Edges by type
        report_lines.append("\n" + "-" * 70)
        report_lines.append("RELATIONSHIPS BY TYPE")
        report_lines.append("-" * 70)
        for edge_type, count in edges_df['type'].value_counts().items():
            report_lines.append(f"{edge_type:20s}: {count:6d} ({count/len(edges_df)*100:5.1f}%)")
        
        # Most connected nodes
        report_lines.append("\n" + "-" * 70)
        report_lines.append("TOP 10 MOST CONNECTED ENTITIES")
        report_lines.append("-" * 70)
        
        node_degrees = pd.concat([
            edges_df['source'].value_counts(),
            edges_df['target'].value_counts()
        ]).groupby(level=0).sum().sort_values(ascending=False)
        
        for i, (node, degree) in enumerate(node_degrees.head(10).items(), 1):
            node_type = nodes_df[nodes_df['id'] == node]['type'].iloc[0] if node in nodes_df['id'].values else "Unknown"
            report_lines.append(f"{i:2d}. {node:30s} ({node_type:10s}): {degree:3d} connections")
        
        report_lines.append("\n" + "=" * 70)
        
        # Save report
        report_path = output_dir / self.config.OUTPUT_SUMMARY
        report_text = "\n".join(report_lines)
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        # Print to console
        print("\n" + report_text)
        logger.info(f"📄 Summary saved to: {report_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    try:
        # Initialize extractor
        config = Config()
        
        # Configure for testing or full run
        # For testing: limit chunks per file
        config.MAX_CHUNKS_PER_FILE = None
        
        # For full production run: process all chunks
        # config.MAX_CHUNKS_PER_FILE = None
        
        extractor = GraphExtractor(config=config)
        
        # Extract graph
        logger.info("🚀 Starting knowledge graph extraction...")
        nodes_df, edges_df = extractor.process_all_files(parallel=True)
        
        # Save results
        extractor.save_graph(nodes_df, edges_df)
        
        logger.info("\n🎉 Graph extraction completed successfully!")
        
    except Exception as e:
        logger.error(f"\n❌ Fatal Error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()