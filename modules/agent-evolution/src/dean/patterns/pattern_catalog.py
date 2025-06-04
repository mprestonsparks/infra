"""
Pattern catalog for storing and retrieving successful patterns.

Maintains a searchable catalog of discovered patterns and strategies
for cross-agent learning and meta-optimization.
"""

import json
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import logging
from pathlib import Path

from .pattern_detector import Pattern, PatternType

logger = logging.getLogger(__name__)


@dataclass
class CatalogEntry:
    """Entry in the pattern catalog."""
    entry_id: str
    pattern: Pattern
    discovered_by: str  # Agent that discovered it
    discovery_date: datetime
    adoption_count: int = 0
    success_rate: float = 0.0
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}


class PatternCatalog:
    """
    Searchable catalog of successful patterns and strategies.
    
    Enables meta-learning by storing and retrieving patterns
    that have proven effective across different agents.
    """
    
    def __init__(self, db_path: str = ":memory:"):
        """
        Initialize pattern catalog.
        
        Args:
            db_path: Path to SQLite database (":memory:" for in-memory)
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        
        self._create_tables()
        self._pattern_cache: Dict[str, CatalogEntry] = {}
    
    def _create_tables(self) -> None:
        """Create database tables for pattern storage."""
        cursor = self.conn.cursor()
        
        # Main patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                entry_id TEXT PRIMARY KEY,
                pattern_id TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                description TEXT,
                effectiveness REAL,
                confidence REAL,
                occurrences INTEGER,
                discovered_by TEXT,
                discovery_date TEXT,
                adoption_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                pattern_data TEXT,  -- JSON serialized pattern
                metadata TEXT       -- JSON serialized metadata
            )
        """)
        
        # Pattern sequences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_sequences (
                pattern_id TEXT,
                sequence_index INTEGER,
                sequence_value TEXT,
                PRIMARY KEY (pattern_id, sequence_index),
                FOREIGN KEY (pattern_id) REFERENCES patterns(pattern_id)
            )
        """)
        
        # Tags table for searching
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_tags (
                entry_id TEXT,
                tag TEXT,
                PRIMARY KEY (entry_id, tag),
                FOREIGN KEY (entry_id) REFERENCES patterns(entry_id)
            )
        """)
        
        # Adoption tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_adoptions (
                adoption_id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id TEXT,
                agent_id TEXT,
                adoption_date TEXT,
                success BOOLEAN,
                performance_delta REAL,
                FOREIGN KEY (entry_id) REFERENCES patterns(entry_id)
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pattern_type ON patterns(pattern_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_effectiveness ON patterns(effectiveness DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_adoption ON patterns(adoption_count DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags ON pattern_tags(tag)")
        
        self.conn.commit()
    
    def add_pattern(self,
                   pattern: Pattern,
                   discovered_by: str,
                   tags: List[str] = None) -> str:
        """
        Add a new pattern to the catalog.
        
        Returns entry ID of the cataloged pattern.
        """
        entry_id = f"{pattern.pattern_id}_{discovered_by}_{int(datetime.now().timestamp())}"
        
        entry = CatalogEntry(
            entry_id=entry_id,
            pattern=pattern,
            discovered_by=discovered_by,
            discovery_date=datetime.now(),
            tags=tags or []
        )
        
        cursor = self.conn.cursor()
        
        # Insert main pattern record
        cursor.execute("""
            INSERT INTO patterns (
                entry_id, pattern_id, pattern_type, description,
                effectiveness, confidence, occurrences,
                discovered_by, discovery_date, pattern_data, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.entry_id,
            pattern.pattern_id,
            pattern.pattern_type.value,
            pattern.description,
            pattern.effectiveness,
            pattern.confidence,
            pattern.occurrences,
            entry.discovered_by,
            entry.discovery_date.isoformat(),
            json.dumps(pattern.to_dict()),
            json.dumps(entry.metadata)
        ))
        
        # Insert sequence data
        for idx, value in enumerate(pattern.sequence):
            cursor.execute("""
                INSERT INTO pattern_sequences (pattern_id, sequence_index, sequence_value)
                VALUES (?, ?, ?)
            """, (pattern.pattern_id, idx, str(value)))
        
        # Insert tags
        for tag in entry.tags:
            cursor.execute("""
                INSERT INTO pattern_tags (entry_id, tag)
                VALUES (?, ?)
            """, (entry.entry_id, tag))
        
        self.conn.commit()
        
        # Update cache
        self._pattern_cache[entry_id] = entry
        
        logger.info(f"Added pattern {entry_id} to catalog")
        return entry_id
    
    def search_patterns(self,
                       pattern_type: Optional[PatternType] = None,
                       min_effectiveness: float = 0.0,
                       min_adoption: int = 0,
                       tags: List[str] = None,
                       limit: int = 10) -> List[CatalogEntry]:
        """
        Search for patterns matching criteria.
        
        Returns list of matching catalog entries.
        """
        query = """
            SELECT DISTINCT p.*
            FROM patterns p
            WHERE 1=1
        """
        params = []
        
        # Add filters
        if pattern_type:
            query += " AND p.pattern_type = ?"
            params.append(pattern_type.value)
        
        if min_effectiveness > 0:
            query += " AND p.effectiveness >= ?"
            params.append(min_effectiveness)
        
        if min_adoption > 0:
            query += " AND p.adoption_count >= ?"
            params.append(min_adoption)
        
        if tags:
            tag_placeholders = ",".join("?" * len(tags))
            query += f"""
                AND p.entry_id IN (
                    SELECT entry_id FROM pattern_tags
                    WHERE tag IN ({tag_placeholders})
                    GROUP BY entry_id
                    HAVING COUNT(DISTINCT tag) = ?
                )
            """
            params.extend(tags)
            params.append(len(tags))
        
        # Order by effectiveness and adoption
        query += " ORDER BY p.effectiveness * LOG(1 + p.adoption_count) DESC"
        query += f" LIMIT {limit}"
        
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        
        entries = []
        for row in cursor.fetchall():
            entry = self._row_to_entry(row)
            if entry:
                entries.append(entry)
        
        return entries
    
    def get_similar_patterns(self,
                           pattern: Pattern,
                           similarity_threshold: float = 0.7,
                           limit: int = 5) -> List[Tuple[CatalogEntry, float]]:
        """
        Find patterns similar to a given pattern.
        
        Returns list of (entry, similarity_score) tuples.
        """
        candidates = self.search_patterns(
            pattern_type=pattern.pattern_type,
            limit=50  # Get more candidates for similarity comparison
        )
        
        similar = []
        for candidate in candidates:
            if candidate.pattern.pattern_id != pattern.pattern_id:
                similarity = self._calculate_similarity(pattern, candidate.pattern)
                if similarity >= similarity_threshold:
                    similar.append((candidate, similarity))
        
        # Sort by similarity
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar[:limit]
    
    def _calculate_similarity(self, p1: Pattern, p2: Pattern) -> float:
        """Calculate similarity between two patterns."""
        scores = []
        
        # Type match
        if p1.pattern_type == p2.pattern_type:
            scores.append(1.0)
        else:
            scores.append(0.0)
        
        # Effectiveness similarity
        eff_diff = abs(p1.effectiveness - p2.effectiveness)
        scores.append(1.0 / (1.0 + eff_diff))
        
        # Sequence similarity (if applicable)
        if p1.sequence and p2.sequence:
            # Simple overlap metric
            set1 = set(str(x) for x in p1.sequence)
            set2 = set(str(x) for x in p2.sequence)
            
            if set1 or set2:
                overlap = len(set1 & set2) / len(set1 | set2)
                scores.append(overlap)
        
        # Context similarity
        if p1.context and p2.context:
            shared_keys = set(p1.context.keys()) & set(p2.context.keys())
            if shared_keys:
                scores.append(len(shared_keys) / max(len(p1.context), len(p2.context)))
        
        return np.mean(scores) if scores else 0.0
    
    def record_adoption(self,
                       entry_id: str,
                       agent_id: str,
                       success: bool,
                       performance_delta: float = 0.0) -> None:
        """Record pattern adoption by an agent."""
        cursor = self.conn.cursor()
        
        # Record adoption
        cursor.execute("""
            INSERT INTO pattern_adoptions (
                entry_id, agent_id, adoption_date, success, performance_delta
            ) VALUES (?, ?, ?, ?, ?)
        """, (entry_id, agent_id, datetime.now().isoformat(), success, performance_delta))
        
        # Update adoption count and success rate
        cursor.execute("""
            UPDATE patterns
            SET adoption_count = adoption_count + 1,
                success_rate = (
                    SELECT AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END)
                    FROM pattern_adoptions
                    WHERE entry_id = ?
                )
            WHERE entry_id = ?
        """, (entry_id, entry_id))
        
        self.conn.commit()
        
        # Update cache if present
        if entry_id in self._pattern_cache:
            self._pattern_cache[entry_id].adoption_count += 1
    
    def get_top_patterns(self,
                        metric: str = "effectiveness",
                        limit: int = 10) -> List[CatalogEntry]:
        """
        Get top patterns by specified metric.
        
        Metrics: effectiveness, adoption, success_rate, recent
        """
        if metric == "effectiveness":
            order_by = "effectiveness DESC"
        elif metric == "adoption":
            order_by = "adoption_count DESC"
        elif metric == "success_rate":
            order_by = "success_rate DESC"
        elif metric == "recent":
            order_by = "discovery_date DESC"
        else:
            order_by = "effectiveness DESC"
        
        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT * FROM patterns
            ORDER BY {order_by}
            LIMIT ?
        """, (limit,))
        
        entries = []
        for row in cursor.fetchall():
            entry = self._row_to_entry(row)
            if entry:
                entries.append(entry)
        
        return entries
    
    def _row_to_entry(self, row: sqlite3.Row) -> Optional[CatalogEntry]:
        """Convert database row to CatalogEntry."""
        try:
            # Deserialize pattern
            pattern_data = json.loads(row['pattern_data'])
            
            # Reconstruct Pattern object
            pattern = Pattern(
                pattern_id=pattern_data['pattern_id'],
                pattern_type=PatternType(pattern_data['pattern_type']),
                description=pattern_data['description'],
                occurrences=pattern_data['occurrences'],
                effectiveness=pattern_data['effectiveness'],
                confidence=pattern_data['confidence'],
                sequence=pattern_data.get('sequence', []),
                context=pattern_data.get('context', {}),
                metadata=pattern_data.get('metadata', {})
            )
            
            # Get tags
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT tag FROM pattern_tags WHERE entry_id = ?
            """, (row['entry_id'],))
            tags = [r['tag'] for r in cursor.fetchall()]
            
            # Create entry
            entry = CatalogEntry(
                entry_id=row['entry_id'],
                pattern=pattern,
                discovered_by=row['discovered_by'],
                discovery_date=datetime.fromisoformat(row['discovery_date']),
                adoption_count=row['adoption_count'],
                success_rate=row['success_rate'],
                tags=tags,
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
            
            return entry
            
        except Exception as e:
            logger.error(f"Error converting row to entry: {e}")
            return None
    
    def export_catalog(self, filepath: str) -> None:
        """Export entire catalog to JSON file."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM patterns")
        
        catalog_data = {
            'export_date': datetime.now().isoformat(),
            'total_patterns': cursor.rowcount,
            'entries': []
        }
        
        for row in cursor.fetchall():
            entry = self._row_to_entry(row)
            if entry:
                catalog_data['entries'].append({
                    'entry_id': entry.entry_id,
                    'pattern': entry.pattern.to_dict(),
                    'discovered_by': entry.discovered_by,
                    'discovery_date': entry.discovery_date.isoformat(),
                    'adoption_count': entry.adoption_count,
                    'success_rate': entry.success_rate,
                    'tags': entry.tags,
                    'metadata': entry.metadata
                })
        
        with open(filepath, 'w') as f:
            json.dump(catalog_data, f, indent=2)
        
        logger.info(f"Exported catalog to {filepath}")
    
    def get_statistics(self) -> Dict:
        """Get catalog statistics."""
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Total patterns
        cursor.execute("SELECT COUNT(*) as count FROM patterns")
        stats['total_patterns'] = cursor.fetchone()['count']
        
        # Patterns by type
        cursor.execute("""
            SELECT pattern_type, COUNT(*) as count
            FROM patterns
            GROUP BY pattern_type
        """)
        stats['by_type'] = {row['pattern_type']: row['count'] for row in cursor.fetchall()}
        
        # Adoption statistics
        cursor.execute("""
            SELECT 
                AVG(adoption_count) as avg_adoption,
                MAX(adoption_count) as max_adoption,
                AVG(success_rate) as avg_success_rate
            FROM patterns
        """)
        row = cursor.fetchone()
        stats['adoption'] = {
            'average': row['avg_adoption'] or 0,
            'maximum': row['max_adoption'] or 0,
            'avg_success_rate': row['avg_success_rate'] or 0
        }
        
        # Top tags
        cursor.execute("""
            SELECT tag, COUNT(*) as count
            FROM pattern_tags
            GROUP BY tag
            ORDER BY count DESC
            LIMIT 10
        """)
        stats['top_tags'] = [(row['tag'], row['count']) for row in cursor.fetchall()]
        
        return stats