"""
Metrics database for comprehensive evolution tracking.

Provides a queryable knowledge base for pattern analysis
and strategy extraction, not merely a logging mechanism.
"""

import sqlite3
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
import logging
from contextlib import contextmanager

# Optional PostgreSQL support
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
    import psycopg2.pool
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False
    psycopg2 = None
    RealDictCursor = None
    Json = None

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
    """Fluent query builder for complex analytical queries."""
    
    def __init__(self):
        self.select_clause = []
        self.from_clause = ""
        self.join_clauses = []
        self.where_clauses = []
        self.group_by_clause = []
        self.having_clauses = []
        self.order_by_clause = []
        self.limit_clause = None
        self.params = []
    
    def select(self, *columns: str) -> 'QueryBuilder':
        """Add columns to select."""
        self.select_clause.extend(columns)
        return self
    
    def from_table(self, table: str) -> 'QueryBuilder':
        """Set the FROM clause."""
        self.from_clause = table
        return self
    
    def join(self, table: str, on: str) -> 'QueryBuilder':
        """Add a JOIN clause."""
        self.join_clauses.append(f"JOIN {table} ON {on}")
        return self
    
    def left_join(self, table: str, on: str) -> 'QueryBuilder':
        """Add a LEFT JOIN clause."""
        self.join_clauses.append(f"LEFT JOIN {table} ON {on}")
        return self
    
    def where(self, condition: str, *params: Any) -> 'QueryBuilder':
        """Add a WHERE condition."""
        self.where_clauses.append(condition)
        self.params.extend(params)
        return self
    
    def group_by(self, *columns: str) -> 'QueryBuilder':
        """Add GROUP BY columns."""
        self.group_by_clause.extend(columns)
        return self
    
    def having(self, condition: str, *params: Any) -> 'QueryBuilder':
        """Add a HAVING condition."""
        self.having_clauses.append(condition)
        self.params.extend(params)
        return self
    
    def order_by(self, column: str, desc: bool = False) -> 'QueryBuilder':
        """Add ORDER BY clause."""
        direction = "DESC" if desc else "ASC"
        self.order_by_clause.append(f"{column} {direction}")
        return self
    
    def limit(self, n: int) -> 'QueryBuilder':
        """Set LIMIT clause."""
        self.limit_clause = n
        return self
    
    def build(self) -> Tuple[str, List[Any]]:
        """Build the final query and parameters."""
        query_parts = []
        
        # SELECT
        if self.select_clause:
            query_parts.append(f"SELECT {', '.join(self.select_clause)}")
        else:
            query_parts.append("SELECT *")
        
        # FROM
        query_parts.append(f"FROM {self.from_clause}")
        
        # JOINs
        query_parts.extend(self.join_clauses)
        
        # WHERE
        if self.where_clauses:
            query_parts.append(f"WHERE {' AND '.join(self.where_clauses)}")
        
        # GROUP BY
        if self.group_by_clause:
            query_parts.append(f"GROUP BY {', '.join(self.group_by_clause)}")
        
        # HAVING
        if self.having_clauses:
            query_parts.append(f"HAVING {' AND '.join(self.having_clauses)}")
        
        # ORDER BY
        if self.order_by_clause:
            query_parts.append(f"ORDER BY {', '.join(self.order_by_clause)}")
        
        # LIMIT
        if self.limit_clause:
            query_parts.append(f"LIMIT {self.limit_clause}")
        
        query = "\n".join(query_parts)
        return query, self.params


class MetricsDatabase:
    """
    Core metrics database for the DEAN system.
    
    Designed as a queryable knowledge base for pattern analysis
    and meta-learning, supporting both SQLite and PostgreSQL.
    """
    
    def __init__(self,
                 db_url: str = "sqlite:///dean_metrics.db",
                 pool_size: int = 5):
        """
        Initialize metrics database.
        
        Args:
            db_url: Database URL (sqlite:///path or postgresql://...)
            pool_size: Connection pool size for PostgreSQL
        """
        self.db_url = db_url
        self.is_postgres = db_url.startswith("postgresql://") and HAS_POSTGRES
        
        if self.is_postgres:
            if not HAS_POSTGRES:
                raise ImportError("PostgreSQL support requires psycopg2: pip install psycopg2-binary")
            # PostgreSQL connection pool
            self.pool = psycopg2.pool.SimpleConnectionPool(
                1, pool_size, db_url
            )
        else:
            # SQLite connection
            db_path = db_url.replace("sqlite:///", "")
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._initialize_schema()
    
    @contextmanager
    def get_connection(self):
        """Get a database connection from the pool."""
        if self.is_postgres:
            conn = self.pool.getconn()
            try:
                yield conn
            finally:
                self.pool.putconn(conn)
        else:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
    
    def _initialize_schema(self) -> None:
        """Initialize database schema."""
        schema_path = Path(__file__).parent / "schema.sql"
        
        if not schema_path.exists():
            logger.warning("Schema file not found, using basic schema")
            self._create_basic_schema()
            return
        
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if self.is_postgres:
                # PostgreSQL - execute as-is
                cursor.execute(schema_sql)
            else:
                # SQLite - need to adapt some syntax
                schema_sql = self._adapt_schema_for_sqlite(schema_sql)
                cursor.executescript(schema_sql)
            
            conn.commit()
        
        logger.info("Database schema initialized")
    
    def _adapt_schema_for_sqlite(self, schema_sql: str) -> str:
        """Adapt PostgreSQL schema for SQLite."""
        # Replace PostgreSQL-specific syntax
        adaptations = [
            ("UUID", "TEXT"),
            ("JSONB", "TEXT"),
            ("gen_random_uuid()", "hex(randomblob(16))"),
            ("TIMESTAMP", "TEXT"),
            ("CURRENT_TIMESTAMP", "datetime('now')"),
            ("REAL GENERATED ALWAYS AS", "REAL AS"),
            ("STORED", ""),
            ("CREATE MATERIALIZED VIEW", "CREATE VIEW"),
            ("$$ LANGUAGE sql;", ""),
            ("$$", ""),
            ("::REAL", ""),
            ("::jsonb", ""),
            ("BOOLEAN", "INTEGER"),
            ("true", "1"),
            ("false", "0")
        ]
        
        for old, new in adaptations:
            schema_sql = schema_sql.replace(old, new)
        
        return schema_sql
    
    def _create_basic_schema(self) -> None:
        """Create a basic schema if the full schema is not available."""
        basic_schema = """
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
        
        CREATE INDEX IF NOT EXISTS idx_pattern_effectiveness 
            ON discovered_patterns(effectiveness_score DESC);
        CREATE INDEX IF NOT EXISTS idx_pattern_efficiency 
            ON discovered_patterns(token_efficiency DESC);
        
        CREATE TABLE IF NOT EXISTS agents (
            agent_id TEXT PRIMARY KEY,
            genome_hash TEXT,
            generation INTEGER,
            total_tokens_used INTEGER DEFAULT 0,
            total_value_generated REAL DEFAULT 0,
            efficiency_score REAL DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        );
        
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id TEXT PRIMARY KEY,
            agent_id TEXT,
            timestamp TEXT DEFAULT (datetime('now')),
            action_type TEXT,
            tokens_used INTEGER,
            value_generated REAL,
            efficiency REAL,
            metadata TEXT
        );
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_postgres:
                # Convert SQLite syntax to PostgreSQL
                basic_schema = basic_schema.replace("TEXT", "VARCHAR")
                basic_schema = basic_schema.replace("datetime('now')", "CURRENT_TIMESTAMP")
                cursor.execute(basic_schema)
            else:
                cursor.executescript(basic_schema)
            conn.commit()
    
    def insert_pattern(self, pattern: Dict[str, Any]) -> str:
        """Insert a discovered pattern."""
        pattern_id = pattern.get('id', str(uuid.uuid4()))
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if self.is_postgres:
                cursor.execute("""
                    INSERT INTO discovered_patterns (
                        id, pattern_hash, pattern_type, description,
                        effectiveness_score, token_efficiency, confidence_score,
                        pattern_data, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (pattern_hash) DO UPDATE SET
                        reuse_count = discovered_patterns.reuse_count + 1,
                        last_seen = CURRENT_TIMESTAMP
                    RETURNING id
                """, (
                    pattern_id, pattern['pattern_hash'], pattern['pattern_type'],
                    pattern.get('description'), pattern['effectiveness_score'],
                    pattern['token_efficiency'], pattern.get('confidence_score', 0),
                    Json(pattern), Json(pattern.get('metadata', {}))
                ))
                result = cursor.fetchone()
                pattern_id = result[0] if result else pattern_id
            else:
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
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Update agent record
            if self.is_postgres:
                cursor.execute("""
                    INSERT INTO agents (
                        agent_id, genome_hash, generation,
                        total_tokens_used, total_value_generated, efficiency_score
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (agent_id) DO UPDATE SET
                        total_tokens_used = agents.total_tokens_used + %s,
                        total_value_generated = agents.total_value_generated + %s,
                        efficiency_score = CASE 
                            WHEN agents.total_tokens_used + %s > 0 
                            THEN (agents.total_value_generated + %s) / (agents.total_tokens_used + %s)
                            ELSE 0 
                        END
                """, (
                    agent_id, metrics.get('genome_hash'), metrics.get('generation', 0),
                    metrics.get('tokens_used', 0), metrics.get('value_generated', 0),
                    metrics.get('efficiency', 0),
                    metrics.get('tokens_used', 0), metrics.get('value_generated', 0),
                    metrics.get('tokens_used', 0), metrics.get('value_generated', 0),
                    metrics.get('tokens_used', 0)
                ))
            else:
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
            if self.is_postgres:
                cursor.execute("""
                    INSERT INTO performance_metrics (
                        id, agent_id, action_type, tokens_used,
                        value_generated, efficiency, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    perf_id, agent_id, metrics.get('action_type', 'unknown'),
                    metrics.get('tokens_used', 0), metrics.get('value_generated', 0),
                    metrics.get('efficiency', 0), Json(metrics)
                ))
            else:
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
        query = QueryBuilder() \
            .select("*") \
            .from_table("discovered_patterns") \
            .where("effectiveness_score >= %s", min_effectiveness)
        
        if pattern_type:
            query.where("pattern_type = %s", pattern_type)
        
        query.order_by("effectiveness_score * token_efficiency", desc=True) \
             .limit(limit)
        
        sql, params = query.build()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if self.is_postgres:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute(sql, params)
                results = cursor.fetchall()
                return [dict(row) for row in results]
            else:
                # SQLite parameter substitution
                sql = sql.replace("%s", "?")
                cursor.execute(sql, params)
                return [dict(row) for row in cursor.fetchall()]
    
    def analyze_evolution_progress(self, generation_range: Optional[Tuple[int, int]] = None) -> Dict:
        """Analyze evolution progress across generations."""
        base_query = """
        SELECT 
            generation,
            COUNT(DISTINCT agent_id) as population_size,
            AVG(efficiency_score) as avg_efficiency,
            MAX(efficiency_score) as best_efficiency,
            SUM(total_tokens_used) as total_tokens,
            SUM(total_value_generated) as total_value
        FROM agents
        """
        
        if generation_range:
            where_clause = "WHERE generation BETWEEN %s AND %s"
            params = generation_range
        else:
            where_clause = ""
            params = []
        
        query = f"{base_query} {where_clause} GROUP BY generation ORDER BY generation"
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if self.is_postgres:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute(query, params)
            else:
                query = query.replace("%s", "?")
                cursor.execute(query, params)
            
            results = cursor.fetchall()
            
            return {
                'generations': [dict(row) for row in results],
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
            WHERE p1.success = %s AND p2.success = %s
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
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if self.is_postgres:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute(query, (True, True))
            else:
                query = query.replace("%s", "?")
                cursor.execute(query, (1, 1))  # SQLite uses 1/0 for boolean
            
            return [dict(row) for row in cursor.fetchall()]
    
    def export_knowledge_base(self, output_path: str) -> None:
        """Export the entire knowledge base for analysis."""
        export_data = {
            'export_date': datetime.now().isoformat(),
            'patterns': self.query_top_patterns(limit=1000),
            'evolution_progress': self.analyze_evolution_progress(),
            'pattern_correlations': self.find_pattern_correlations(),
            'metadata': {
                'database_type': 'postgresql' if self.is_postgres else 'sqlite',
                'total_agents': self._count_records('agents'),
                'total_patterns': self._count_records('discovered_patterns')
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Knowledge base exported to {output_path}")
    
    def _count_records(self, table: str) -> int:
        """Count records in a table."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            return cursor.fetchone()[0]