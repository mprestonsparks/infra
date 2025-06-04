"""
Simplified metrics database using only SQLite.

Provides core functionality for the knowledge repository
without external dependencies.
"""

import sqlite3
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricRecord:
    """Generic metric record for database storage."""
    table_name: str
    data: Dict[str, Any]
    id: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.timestamp is None:
            self.timestamp = datetime.now()


class QueryBuilder:
    """Simple query builder for SQLite."""
    
    def __init__(self):
        self.select_clause = []
        self.from_clause = ""
        self.where_clauses = []
        self.order_by_clause = []
        self.limit_clause = None
        self.params = []
    
    def select(self, *columns: str):
        self.select_clause.extend(columns)
        return self
    
    def from_table(self, table: str):
        self.from_clause = table
        return self
    
    def where(self, condition: str, *params):
        # Convert PostgreSQL %s to SQLite ?
        condition = condition.replace("%s", "?")
        self.where_clauses.append(condition)
        self.params.extend(params)
        return self
    
    def order_by(self, column: str, desc: bool = False):
        direction = "DESC" if desc else "ASC"
        self.order_by_clause.append(f"{column} {direction}")
        return self
    
    def limit(self, n: int):
        self.limit_clause = n
        return self
    
    def build(self) -> Tuple[str, List]:
        query_parts = []
        
        if self.select_clause:
            query_parts.append(f"SELECT {', '.join(self.select_clause)}")
        else:
            query_parts.append("SELECT *")
        
        query_parts.append(f"FROM {self.from_clause}")
        
        if self.where_clauses:
            query_parts.append(f"WHERE {' AND '.join(self.where_clauses)}")
        
        if self.order_by_clause:
            query_parts.append(f"ORDER BY {', '.join(self.order_by_clause)}")
        
        if self.limit_clause:
            query_parts.append(f"LIMIT {self.limit_clause}")
        
        return "\n".join(query_parts), self.params


class SimpleMetricsDatabase:
    """
    Simplified metrics database using SQLite.
    
    Core functionality for storing and querying evolution data.
    """
    
    def __init__(self, db_path: str = "dean_metrics.db"):
        """Initialize database."""
        if db_path.startswith("sqlite:///"):
            db_path = db_path.replace("sqlite:///", "")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.is_postgres = False  # Always False for simple version
        self._initialize_schema()
    
    def _initialize_schema(self) -> None:
        """Initialize database schema."""
        schema = """
        -- Core patterns table
        CREATE TABLE IF NOT EXISTS discovered_patterns (
            id TEXT PRIMARY KEY,
            pattern_hash TEXT UNIQUE,
            pattern_type TEXT,
            description TEXT,
            effectiveness_score REAL,
            token_efficiency REAL,
            confidence_score REAL,
            reuse_count INTEGER DEFAULT 0,
            pattern_data TEXT,
            metadata TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        
        -- Agents table
        CREATE TABLE IF NOT EXISTS agents (
            agent_id TEXT PRIMARY KEY,
            genome_hash TEXT,
            generation INTEGER,
            total_tokens_used INTEGER DEFAULT 0,
            total_value_generated REAL DEFAULT 0,
            efficiency_score REAL DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        );
        
        -- Performance metrics
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id TEXT PRIMARY KEY,
            agent_id TEXT,
            timestamp TEXT DEFAULT (datetime('now')),
            action_type TEXT,
            tokens_used INTEGER,
            value_generated REAL,
            efficiency REAL,
            metadata TEXT,
            FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
        );
        
        -- Pattern adoptions
        CREATE TABLE IF NOT EXISTS pattern_adoptions (
            id TEXT PRIMARY KEY,
            pattern_id TEXT,
            agent_id TEXT,
            adopted_at TEXT DEFAULT (datetime('now')),
            success INTEGER,
            performance_delta REAL,
            FOREIGN KEY (pattern_id) REFERENCES discovered_patterns(id),
            FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
        );
        
        -- Indexes for performance
        CREATE INDEX IF NOT EXISTS idx_pattern_effectiveness 
            ON discovered_patterns(effectiveness_score DESC);
        CREATE INDEX IF NOT EXISTS idx_pattern_efficiency 
            ON discovered_patterns(token_efficiency DESC);
        CREATE INDEX IF NOT EXISTS idx_agents_generation 
            ON agents(generation);
        CREATE INDEX IF NOT EXISTS idx_performance_agent 
            ON performance_metrics(agent_id);
        CREATE INDEX IF NOT EXISTS idx_performance_efficiency 
            ON performance_metrics(efficiency DESC);
        """
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript(schema)
        
        logger.info("Database schema initialized")
    
    def get_connection(self):
        """Get database connection context manager."""
        from contextlib import contextmanager
        
        @contextmanager
        def connection():
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
        
        return connection()
    
    def insert_pattern(self, pattern: Dict[str, Any]) -> str:
        """Insert a discovered pattern."""
        pattern_id = pattern.get('id', str(uuid.uuid4()))
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO discovered_patterns (
                    id, pattern_hash, pattern_type, description,
                    effectiveness_score, token_efficiency, confidence_score,
                    pattern_data, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern_id, pattern['pattern_hash'], pattern['pattern_type'],
                pattern.get('description'), pattern['effectiveness_score'],
                pattern['token_efficiency'], pattern.get('confidence_score', 0),
                json.dumps(pattern), json.dumps(pattern.get('metadata', {}))
            ))
            
            conn.commit()
        
        return pattern_id
    
    def record_agent_metrics(self, agent_id: str, metrics: Dict[str, Any]) -> None:
        """Record agent performance metrics."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Update or insert agent record
            cursor.execute("""
                INSERT OR REPLACE INTO agents (
                    agent_id, genome_hash, generation,
                    total_tokens_used, total_value_generated, efficiency_score
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                agent_id, metrics.get('genome_hash'), metrics.get('generation', 0),
                metrics.get('tokens_used', 0), metrics.get('value_generated', 0),
                metrics.get('efficiency', 0)
            ))
            
            # Insert performance record
            perf_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO performance_metrics (
                    id, agent_id, action_type, tokens_used,
                    value_generated, efficiency, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                perf_id, agent_id, metrics.get('action_type', 'unknown'),
                metrics.get('tokens_used', 0), metrics.get('value_generated', 0),
                metrics.get('efficiency', 0), json.dumps(metrics)
            ))
            
            conn.commit()
    
    def query_top_patterns(self,
                          pattern_type: Optional[str] = None,
                          min_effectiveness: float = 0.0,
                          limit: int = 10) -> List[Dict]:
        """Query top performing patterns."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = """
                SELECT * FROM discovered_patterns
                WHERE effectiveness_score >= ?
            """
            params = [min_effectiveness]
            
            if pattern_type:
                query += " AND pattern_type = ?"
                params.append(pattern_type)
            
            query += " ORDER BY effectiveness_score * token_efficiency DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def analyze_evolution_progress(self, generation_range: Optional[Tuple[int, int]] = None) -> Dict:
        """Analyze evolution progress across generations."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = """
            SELECT 
                generation,
                COUNT(DISTINCT agent_id) as population_size,
                AVG(efficiency_score) as avg_efficiency,
                MAX(efficiency_score) as best_efficiency,
                SUM(total_tokens_used) as total_tokens,
                SUM(total_value_generated) as total_value
            FROM agents
            """
            
            params = []
            if generation_range:
                query += " WHERE generation BETWEEN ? AND ?"
                params = list(generation_range)
            
            query += " GROUP BY generation ORDER BY generation"
            
            cursor.execute(query, params)
            results = [dict(row) for row in cursor.fetchall()]
            
            return {
                'generations': results,
                'summary': {
                    'total_generations': len(results),
                    'efficiency_improvement': (
                        results[-1]['avg_efficiency'] / results[0]['avg_efficiency'] - 1
                        if results and results[0]['avg_efficiency'] > 0 else 0
                    )
                }
            }
    
    def find_pattern_correlations(self, min_correlation: float = 0.5) -> List[Dict]:
        """Find patterns that tend to be used together."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = """
            WITH pattern_pairs AS (
                SELECT 
                    p1.pattern_id as pattern1,
                    p2.pattern_id as pattern2,
                    COUNT(*) as co_occurrence_count
                FROM pattern_adoptions p1
                JOIN pattern_adoptions p2 
                    ON p1.agent_id = p2.agent_id 
                    AND p1.pattern_id < p2.pattern_id
                WHERE p1.success = 1 AND p2.success = 1
                GROUP BY p1.pattern_id, p2.pattern_id
            )
            SELECT 
                pp.*,
                dp1.description as pattern1_desc,
                dp2.description as pattern2_desc,
                dp1.effectiveness_score * dp2.effectiveness_score as combined_effectiveness
            FROM pattern_pairs pp
            JOIN discovered_patterns dp1 ON pp.pattern1 = dp1.id
            JOIN discovered_patterns dp2 ON pp.pattern2 = dp2.id
            WHERE pp.co_occurrence_count >= 3
            ORDER BY pp.co_occurrence_count DESC
            LIMIT 20
            """
            
            cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]
    
    def export_knowledge_base(self, output_path: str) -> None:
        """Export the entire knowledge base for analysis."""
        export_data = {
            'export_date': datetime.now().isoformat(),
            'patterns': self.query_top_patterns(limit=1000),
            'evolution_progress': self.analyze_evolution_progress(),
            'pattern_correlations': self.find_pattern_correlations(),
            'metadata': {
                'database_type': 'sqlite',
                'total_agents': self._count_records('agents'),
                'total_patterns': self._count_records('discovered_patterns')
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Knowledge base exported to {output_path}")
    
    def _count_records(self, table: str) -> int:
        """Count records in a table."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            return cursor.fetchone()[0]