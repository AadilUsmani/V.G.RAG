"""
Optimized Neo4j Knowledge Graph Ingester

Purpose:
    Loads extracted knowledge graph (nodes + edges) into Neo4j graph database
    for GraphRAG, relationship analysis, and graph queries.

Key Optimizations:
    - Batch processing with transactions (10-100x faster)
    - APOC library support for dynamic relationship types
    - Progress tracking with tqdm
    - Comprehensive error handling
    - Connection pooling
    - Index creation for performance
    - Statistics and validation

Usage:
    python neo4j_ingester_optimized.py
    
Requirements:
    pip install neo4j pandas python-dotenv tqdm

.env file should contain:
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USERNAME=neo4j
    NEO4J_PASSWORD=your_password
"""

import os
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv
from typing import List, Dict, Optional
from tqdm import tqdm
import logging
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration for Neo4j ingestion"""
    # File paths
    NODES_FILE = "graph_output/graph_nodes.csv"
    EDGES_FILE = "graph_output/graph_edges.csv"
    
    # Batch sizes (tune based on your machine)
    NODE_BATCH_SIZE = 1000  # Larger batches = faster, but more memory
    EDGE_BATCH_SIZE = 500   # Edges are more complex
    
    # Connection settings
    CONNECTION_TIMEOUT = 30  # seconds
    MAX_RETRY_TIME = 30  # seconds
    
    # Options
    CLEAR_EXISTING = True  # Clear database before import
    CREATE_INDEXES = True  # Create performance indexes
    USE_APOC = False  # Set True if APOC is installed (for dynamic relationships)


class Neo4jIngester:
    """
    Efficiently loads knowledge graph into Neo4j with batch processing and error handling
    """
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self._validate_environment()
        self._initialize_driver()
        self.stats = {
            'nodes_loaded': 0,
            'edges_loaded': 0,
            'nodes_failed': 0,
            'edges_failed': 0,
            'start_time': None,
            'end_time': None
        }
    
    def _validate_environment(self) -> None:
        """Validate environment variables and files"""
        load_dotenv()
        
        # Check credentials
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")
        
        if not all([self.uri, self.user, self.password]):
            raise ValueError(
                "❌ Missing Neo4j credentials in .env file\n"
                "Required: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD"
            )
        
        # Check files exist
        if not os.path.exists(self.config.NODES_FILE):
            raise FileNotFoundError(f"❌ Nodes file not found: {self.config.NODES_FILE}")
        if not os.path.exists(self.config.EDGES_FILE):
            raise FileNotFoundError(f"❌ Edges file not found: {self.config.EDGES_FILE}")
        
        logger.info("✅ Environment validated")
    
    def _initialize_driver(self) -> None:
        """Initialize Neo4j driver with connection pooling"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_timeout=self.config.CONNECTION_TIMEOUT
            )
            
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS test")
                result.single()
            
            logger.info(f"✅ Connected to Neo4j at {self.uri}")
            
        except Exception as e:
            raise ConnectionError(f"❌ Failed to connect to Neo4j: {e}")
    
    def close(self) -> None:
        """Close Neo4j driver"""
        if self.driver:
            self.driver.close()
            logger.info("🔌 Neo4j connection closed")
    
    def clear_database(self) -> None:
        """Clear all nodes and relationships"""
        logger.info("🧹 Clearing existing database...")
        
        with self.driver.session() as session:
            # Delete in batches to avoid memory issues with large graphs
            deleted_count = 1
            total_deleted = 0
            
            while deleted_count > 0:
                result = session.run("""
                    MATCH (n)
                    WITH n LIMIT 10000
                    DETACH DELETE n
                    RETURN count(n) AS deleted
                """)
                deleted_count = result.single()['deleted']
                total_deleted += deleted_count
                
                if deleted_count > 0:
                    logger.info(f"   Deleted {total_deleted} nodes...")
            
            logger.info(f"✅ Database cleared ({total_deleted} nodes deleted)")
    
    def create_indexes(self) -> None:
        """Create indexes and constraints for performance"""
        logger.info("⚡ Creating indexes and constraints...")
        
        with self.driver.session() as session:
            # Unique constraint on Entity.id (also creates index)
            try:
                session.run("""
                    CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
                    FOR (n:Entity) REQUIRE n.id IS UNIQUE
                """)
                logger.info("   ✓ Created unique constraint on Entity.id")
            except Exception as e:
                logger.warning(f"   ⚠️ Constraint already exists or error: {e}")
            
            # Index on Entity.type for filtering
            try:
                session.run("""
                    CREATE INDEX entity_type_index IF NOT EXISTS
                    FOR (n:Entity) ON (n.type)
                """)
                logger.info("   ✓ Created index on Entity.type")
            except Exception as e:
                logger.warning(f"   ⚠️ Index already exists or error: {e}")
            
            # Index on source_file for provenance queries
            try:
                session.run("""
                    CREATE INDEX entity_source_index IF NOT EXISTS
                    FOR (n:Entity) ON (n.source_file)
                """)
                logger.info("   ✓ Created index on Entity.source_file")
            except Exception as e:
                logger.warning(f"   ⚠️ Index already exists or error: {e}")
        
        logger.info("✅ Indexes created")
    
    def load_nodes(self) -> None:
        """Load nodes from CSV with batch processing"""
        logger.info(f"📦 Loading nodes from {self.config.NODES_FILE}...")
        
        # Load CSV
        nodes_df = pd.read_csv(self.config.NODES_FILE)
        nodes_df = nodes_df.fillna("")  # Replace NaN with empty string
        
        total_nodes = len(nodes_df)
        logger.info(f"   Found {total_nodes} nodes to load")
        
        # Process in batches
        batch_size = self.config.NODE_BATCH_SIZE
        
        with self.driver.session() as session:
            for i in tqdm(range(0, total_nodes, batch_size), desc="Loading nodes"):
                batch = nodes_df.iloc[i:i+batch_size].to_dict('records')
                
                # Clean batch data
                for record in batch:
                    record['chunk_index'] = int(record.get('chunk_index', 0))
                
                try:
                    # Use UNWIND for batch insert (much faster than individual queries)
                    query = """
                    UNWIND $batch AS row
                    MERGE (n:Entity {id: row.id})
                    SET n.type = row.type,
                        n.source_file = row.source_file,
                        n.chunk_index = row.chunk_index
                    """
                    session.run(query, batch=batch)
                    self.stats['nodes_loaded'] += len(batch)
                    
                except Exception as e:
                    logger.error(f"Error loading node batch {i}: {e}")
                    self.stats['nodes_failed'] += len(batch)
        
        logger.info(f"✅ Loaded {self.stats['nodes_loaded']} nodes")
        if self.stats['nodes_failed'] > 0:
            logger.warning(f"⚠️ Failed to load {self.stats['nodes_failed']} nodes")
    
    def load_edges_with_apoc(self) -> None:
        """Load edges using APOC for dynamic relationship types (requires APOC plugin)"""
        logger.info(f"🔗 Loading edges from {self.config.EDGES_FILE} (using APOC)...")
        
        edges_df = pd.read_csv(self.config.EDGES_FILE)
        edges_df = edges_df.fillna("")
        
        total_edges = len(edges_df)
        logger.info(f"   Found {total_edges} edges to load")
        
        batch_size = self.config.EDGE_BATCH_SIZE
        
        with self.driver.session() as session:
            for i in tqdm(range(0, total_edges, batch_size), desc="Loading edges"):
                batch = edges_df.iloc[i:i+batch_size].to_dict('records')
                
                # Clean relationship types
                for record in batch:
                    rel_type = str(record['type']).upper().replace(" ", "_").replace("-", "_")
                    # Remove any non-alphanumeric characters except underscore
                    rel_type = ''.join(c if c.isalnum() or c == '_' else '' for c in rel_type)
                    record['type'] = rel_type if rel_type else "RELATED_TO"
                
                try:
                    # Use APOC for dynamic relationship types
                    query = """
                    UNWIND $batch AS row
                    MATCH (source:Entity {id: row.source})
                    MATCH (target:Entity {id: row.target})
                    CALL apoc.create.relationship(source, row.type, {}, target) YIELD rel
                    RETURN count(rel)
                    """
                    session.run(query, batch=batch)
                    self.stats['edges_loaded'] += len(batch)
                    
                except Exception as e:
                    logger.error(f"Error loading edge batch {i}: {e}")
                    self.stats['edges_failed'] += len(batch)
        
        logger.info(f"✅ Loaded {self.stats['edges_loaded']} edges")
        if self.stats['edges_failed'] > 0:
            logger.warning(f"⚠️ Failed to load {self.stats['edges_failed']} edges")
    
    def load_edges_without_apoc(self) -> None:
        """Load edges without APOC by grouping by relationship type"""
        logger.info(f"🔗 Loading edges from {self.config.EDGES_FILE}...")
        
        edges_df = pd.read_csv(self.config.EDGES_FILE)
        edges_df = edges_df.fillna("")
        
        total_edges = len(edges_df)
        logger.info(f"   Found {total_edges} edges to load")
        
        # Clean relationship types
        edges_df['type'] = edges_df['type'].apply(
            lambda x: ''.join(c if c.isalnum() or c == '_' else '' 
                            for c in str(x).upper().replace(" ", "_").replace("-", "_"))
        )
        edges_df['type'] = edges_df['type'].replace('', 'RELATED_TO')
        
        # Group by relationship type for efficiency
        relationship_types = edges_df['type'].unique()
        logger.info(f"   Found {len(relationship_types)} unique relationship types")
        
        batch_size = self.config.EDGE_BATCH_SIZE
        
        with self.driver.session() as session:
            for rel_type in tqdm(relationship_types, desc="Processing relationship types"):
                # Get all edges of this type
                type_edges = edges_df[edges_df['type'] == rel_type]
                
                # Process in batches
                for i in range(0, len(type_edges), batch_size):
                    batch = type_edges.iloc[i:i+batch_size].to_dict('records')
                    
                    try:
                        # Build query with actual relationship type
                        query = f"""
                        UNWIND $batch AS row
                        MATCH (source:Entity {{id: row.source}})
                        MATCH (target:Entity {{id: row.target}})
                        MERGE (source)-[:{rel_type}]->(target)
                        """
                        session.run(query, batch=batch)
                        self.stats['edges_loaded'] += len(batch)
                        
                    except Exception as e:
                        logger.error(f"Error loading edges of type {rel_type}: {e}")
                        self.stats['edges_failed'] += len(batch)
        
        logger.info(f"✅ Loaded {self.stats['edges_loaded']} edges")
        if self.stats['edges_failed'] > 0:
            logger.warning(f"⚠️ Failed to load {self.stats['edges_failed']} edges")
    
    def verify_ingestion(self) -> Dict:
        """Verify and get statistics on loaded graph"""
        logger.info("🔍 Verifying ingestion...")
        
        with self.driver.session() as session:
            # Count nodes
            result = session.run("MATCH (n:Entity) RETURN count(n) AS count")
            node_count = result.single()['count']
            
            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) AS count")
            edge_count = result.single()['count']
            
            # Get relationship types
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) AS rel_type, count(*) AS count
                ORDER BY count DESC
                LIMIT 10
            """)
            top_relationships = [(record['rel_type'], record['count']) for record in result]
            
            # Get node types
            result = session.run("""
                MATCH (n:Entity)
                RETURN n.type AS node_type, count(*) AS count
                ORDER BY count DESC
            """)
            node_types = [(record['node_type'], record['count']) for record in result]
            
            stats = {
                'nodes': node_count,
                'edges': edge_count,
                'top_relationships': top_relationships,
                'node_types': node_types
            }
            
            return stats
    
    def print_statistics(self, db_stats: Dict) -> None:
        """Print ingestion statistics"""
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        print("\n" + "=" * 70)
        print("NEO4J INGESTION REPORT")
        print("=" * 70)
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {duration:.2f} seconds")
        
        print("\n" + "-" * 70)
        print("INGESTION STATISTICS")
        print("-" * 70)
        print(f"Nodes loaded:    {self.stats['nodes_loaded']:,}")
        print(f"Nodes failed:    {self.stats['nodes_failed']:,}")
        print(f"Edges loaded:    {self.stats['edges_loaded']:,}")
        print(f"Edges failed:    {self.stats['edges_failed']:,}")
        
        print("\n" + "-" * 70)
        print("DATABASE STATISTICS")
        print("-" * 70)
        print(f"Total nodes in DB:    {db_stats['nodes']:,}")
        print(f"Total edges in DB:    {db_stats['edges']:,}")
        
        print("\n" + "-" * 70)
        print("NODE TYPES")
        print("-" * 70)
        for node_type, count in db_stats['node_types']:
            print(f"{node_type:20s}: {count:,}")
        
        print("\n" + "-" * 70)
        print("TOP 10 RELATIONSHIP TYPES")
        print("-" * 70)
        for rel_type, count in db_stats['top_relationships']:
            print(f"{rel_type:30s}: {count:,}")
        
        print("\n" + "-" * 70)
        print("PERFORMANCE METRICS")
        print("-" * 70)
        print(f"Nodes per second:    {self.stats['nodes_loaded']/duration:.0f}")
        print(f"Edges per second:    {self.stats['edges_loaded']/duration:.0f}")
        
        print("\n" + "=" * 70)
    
    def run_ingestion(self) -> None:
        """Main ingestion workflow"""
        self.stats['start_time'] = datetime.now()
        
        try:
            # Step 1: Clear database (optional)
            if self.config.CLEAR_EXISTING:
                self.clear_database()
            
            # Step 2: Create indexes
            if self.config.CREATE_INDEXES:
                self.create_indexes()
            
            # Step 3: Load nodes
            self.load_nodes()
            
            # Step 4: Load edges
            if self.config.USE_APOC:
                self.load_edges_with_apoc()
            else:
                self.load_edges_without_apoc()
            
            # Step 5: Verify
            self.stats['end_time'] = datetime.now()
            db_stats = self.verify_ingestion()
            
            # Step 6: Print statistics
            self.print_statistics(db_stats)
            
            logger.info("🎉 Ingestion completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Ingestion failed: {e}", exc_info=True)
            raise


def main():
    """Main execution"""
    # Configuration
    config = Config()
    
    # You can customize these settings:
    # config.NODE_BATCH_SIZE = 2000  # Larger batches for faster loading
    # config.USE_APOC = True  # If you have APOC installed
    # config.CLEAR_EXISTING = False  # Keep existing data
    
    ingester = Neo4jIngester(config=config)
    
    try:
        ingester.run_ingestion()
    finally:
        ingester.close()


if __name__ == "__main__":
    main()