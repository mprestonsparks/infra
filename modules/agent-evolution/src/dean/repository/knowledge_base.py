"""
Knowledge base for storing and retrieving learned insights.

Provides high-level abstractions for meta-learning and
strategy extraction from the metrics database.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
from collections import defaultdict

from .metrics_database import MetricsDatabase, QueryBuilder

logger = logging.getLogger(__name__)


class InsightType(Enum):
    """Types of insights that can be learned."""
    OPTIMIZATION_STRATEGY = "optimization_strategy"
    PATTERN_SYNERGY = "pattern_synergy"
    CONSTRAINT_DISCOVERY = "constraint_discovery"
    EMERGENT_BEHAVIOR = "emergent_behavior"
    FAILURE_MODE = "failure_mode"
    PERFORMANCE_PREDICTOR = "performance_predictor"


@dataclass
class KnowledgeEntry:
    """Entry in the knowledge base."""
    entry_id: str
    insight_type: InsightType
    description: str
    confidence: float
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    applicable_contexts: List[str] = field(default_factory=list)
    validation_count: int = 0
    success_rate: float = 0.0
    discovered_at: datetime = field(default_factory=datetime.now)
    last_validated: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_reliable(self, min_validations: int = 5, min_success_rate: float = 0.7) -> bool:
        """Check if this insight is reliable enough to use."""
        return (self.validation_count >= min_validations and 
                self.success_rate >= min_success_rate)
    
    def update_validation(self, success: bool) -> None:
        """Update validation statistics."""
        self.validation_count += 1
        # Running average of success rate
        self.success_rate = (
            (self.success_rate * (self.validation_count - 1) + (1.0 if success else 0.0)) / 
            self.validation_count
        )
        self.last_validated = datetime.now()


class KnowledgeBase:
    """
    High-level knowledge management for the DEAN system.
    
    Extracts, stores, and retrieves learned insights from
    the evolution process for meta-learning.
    """
    
    def __init__(self, metrics_db: MetricsDatabase):
        """Initialize knowledge base with metrics database."""
        self.metrics_db = metrics_db
        self.insights_cache: Dict[str, KnowledgeEntry] = {}
        self.learning_rules: List[LearningRule] = self._initialize_learning_rules()
    
    def extract_insights(self, min_confidence: float = 0.7) -> List[KnowledgeEntry]:
        """Extract new insights from recent metrics."""
        insights = []
        
        # Run each learning rule
        for rule in self.learning_rules:
            try:
                new_insights = rule.extract(self.metrics_db, min_confidence)
                insights.extend(new_insights)
            except Exception as e:
                logger.error(f"Error in learning rule {rule.name}: {e}")
        
        # Store new insights
        for insight in insights:
            self._store_insight(insight)
        
        return insights
    
    def query_insights(self,
                      insight_type: Optional[InsightType] = None,
                      context: Optional[str] = None,
                      min_confidence: float = 0.0,
                      only_reliable: bool = True) -> List[KnowledgeEntry]:
        """Query insights from the knowledge base."""
        query = QueryBuilder() \
            .select("*") \
            .from_table("learned_insights") \
            .where("confidence >= %s", min_confidence)
        
        if insight_type:
            query.where("insight_type = %s", insight_type.value)
        
        if context:
            # JSON containment query
            if self.metrics_db.is_postgres:
                query.where("applicable_contexts @> %s", json.dumps([context]))
            else:
                # SQLite doesn't have native JSON operators
                query.where("applicable_contexts LIKE %s", f'%"{context}"%')
        
        if only_reliable:
            query.where("validation_count >= %s", 5) \
                 .where("success_rate >= %s", 0.7)
        
        query.order_by("confidence", desc=True)
        
        sql, params = query.build()
        
        with self.metrics_db.get_connection() as conn:
            cursor = conn.cursor()
            
            if self.metrics_db.is_postgres:
                from psycopg2.extras import RealDictCursor
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute(sql, params)
            else:
                sql = sql.replace("%s", "?")
                cursor.execute(sql, params)
            
            results = []
            for row in cursor.fetchall():
                entry = self._row_to_entry(dict(row))
                if entry:
                    results.append(entry)
        
        return results
    
    def get_optimization_strategies(self, 
                                   target_metric: str,
                                   current_performance: float) -> List[Dict[str, Any]]:
        """Get recommended optimization strategies for a metric."""
        # Query successful patterns that improved the target metric
        patterns_query = """
        SELECT 
            dp.*,
            AVG(pm.value_generated / pm.tokens_used) as avg_efficiency,
            COUNT(DISTINCT pa.agent_id) as adopter_count,
            AVG(pa.performance_delta) as avg_improvement
        FROM discovered_patterns dp
        JOIN pattern_adoptions pa ON dp.id = pa.pattern_id
        JOIN performance_metrics pm ON pa.agent_id = pm.agent_id
        WHERE pa.success = %s
            AND pm.action_type = %s
            AND pa.performance_delta > 0
        GROUP BY dp.id
        HAVING AVG(pm.value_generated / pm.tokens_used) > %s
        ORDER BY avg_improvement DESC
        LIMIT 10
        """
        
        with self.metrics_db.get_connection() as conn:
            cursor = conn.cursor()
            
            if self.metrics_db.is_postgres:
                from psycopg2.extras import RealDictCursor
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute(patterns_query, (True, target_metric, current_performance))
            else:
                patterns_query = patterns_query.replace("%s", "?")
                cursor.execute(patterns_query, (1, target_metric, current_performance))
            
            strategies = []
            for row in cursor.fetchall():
                strategy = dict(row)
                
                # Add applicable insights
                related_insights = self.query_insights(
                    insight_type=InsightType.OPTIMIZATION_STRATEGY,
                    context=target_metric
                )
                
                strategy['insights'] = [
                    {
                        'description': i.description,
                        'confidence': i.confidence,
                        'success_rate': i.success_rate
                    }
                    for i in related_insights
                ]
                
                strategies.append(strategy)
        
        return strategies
    
    def identify_pattern_synergies(self, min_synergy: float = 1.5) -> List[Dict[str, Any]]:
        """Identify patterns that work well together."""
        synergy_query = """
        WITH pattern_combinations AS (
            SELECT 
                pa1.pattern_id as pattern1_id,
                pa2.pattern_id as pattern2_id,
                pa1.agent_id,
                pa1.performance_delta + pa2.performance_delta as combined_delta,
                (pa1.performance_delta + pa2.performance_delta) / 
                    GREATEST(pa1.performance_delta, pa2.performance_delta) as synergy_factor
            FROM pattern_adoptions pa1
            JOIN pattern_adoptions pa2 
                ON pa1.agent_id = pa2.agent_id 
                AND pa1.pattern_id < pa2.pattern_id
                AND pa1.adopted_at < pa2.adopted_at + INTERVAL '1 hour'
            WHERE pa1.success = %s AND pa2.success = %s
        )
        SELECT 
            dp1.pattern_hash as pattern1,
            dp2.pattern_hash as pattern2,
            dp1.description as pattern1_desc,
            dp2.description as pattern2_desc,
            COUNT(*) as combination_count,
            AVG(pc.synergy_factor) as avg_synergy,
            AVG(pc.combined_delta) as avg_combined_improvement
        FROM pattern_combinations pc
        JOIN discovered_patterns dp1 ON pc.pattern1_id = dp1.id
        JOIN discovered_patterns dp2 ON pc.pattern2_id = dp2.id
        GROUP BY dp1.pattern_hash, dp2.pattern_hash, dp1.description, dp2.description
        HAVING AVG(pc.synergy_factor) >= %s
        ORDER BY avg_synergy DESC
        """
        
        with self.metrics_db.get_connection() as conn:
            cursor = conn.cursor()
            
            if self.metrics_db.is_postgres:
                from psycopg2.extras import RealDictCursor
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute(synergy_query, (True, True, min_synergy))
            else:
                # SQLite doesn't have INTERVAL, need to adjust
                synergy_query = synergy_query.replace("+ INTERVAL '1 hour'", "")
                synergy_query = synergy_query.replace("%s", "?")
                cursor.execute(synergy_query, (1, 1, min_synergy))
            
            synergies = []
            for row in cursor.fetchall():
                synergy = dict(row)
                
                # Create insight for this synergy
                insight = KnowledgeEntry(
                    entry_id=f"synergy_{synergy['pattern1']}_{synergy['pattern2']}",
                    insight_type=InsightType.PATTERN_SYNERGY,
                    description=f"Patterns '{synergy['pattern1_desc']}' and '{synergy['pattern2_desc']}' "
                               f"show {synergy['avg_synergy']:.2f}x synergy",
                    confidence=min(1.0, synergy['combination_count'] / 10),
                    evidence=[{
                        'pattern1': synergy['pattern1'],
                        'pattern2': synergy['pattern2'],
                        'synergy_factor': synergy['avg_synergy'],
                        'occurrences': synergy['combination_count']
                    }],
                    validation_count=synergy['combination_count']
                )
                
                self._store_insight(insight)
                synergies.append(synergy)
        
        return synergies
    
    def predict_agent_performance(self, 
                                 genome_features: Dict[str, Any],
                                 planned_patterns: List[str]) -> Dict[str, float]:
        """Predict agent performance based on genome and patterns."""
        predictions = {
            'expected_efficiency': 0.0,
            'success_probability': 0.0,
            'token_consumption_estimate': 0,
            'confidence': 0.0
        }
        
        # Get historical data for similar configurations
        similar_agents_query = """
        SELECT 
            a.agent_id,
            a.efficiency_score,
            a.total_tokens_used,
            a.total_value_generated,
            COUNT(pa.pattern_id) as patterns_used
        FROM agents a
        LEFT JOIN pattern_adoptions pa ON a.agent_id = pa.agent_id
        WHERE a.generation >= %s
        GROUP BY a.agent_id, a.efficiency_score, a.total_tokens_used, a.total_value_generated
        HAVING COUNT(pa.pattern_id) >= %s
        """
        
        with self.metrics_db.get_connection() as conn:
            cursor = conn.cursor()
            
            min_generation = max(0, genome_features.get('generation', 0) - 5)
            min_patterns = len(planned_patterns) - 2
            
            if self.metrics_db.is_postgres:
                from psycopg2.extras import RealDictCursor
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute(similar_agents_query, (min_generation, max(0, min_patterns)))
            else:
                similar_agents_query = similar_agents_query.replace("%s", "?")
                cursor.execute(similar_agents_query, (min_generation, max(0, min_patterns)))
            
            similar_agents = [dict(row) for row in cursor.fetchall()]
        
        if similar_agents:
            # Calculate predictions based on similar agents
            efficiencies = [a['efficiency_score'] for a in similar_agents]
            tokens = [a['total_tokens_used'] for a in similar_agents]
            
            predictions['expected_efficiency'] = sum(efficiencies) / len(efficiencies)
            predictions['token_consumption_estimate'] = int(sum(tokens) / len(tokens))
            predictions['success_probability'] = len([e for e in efficiencies if e > 0.5]) / len(efficiencies)
            predictions['confidence'] = min(1.0, len(similar_agents) / 10)
        
        # Adjust based on planned patterns
        if planned_patterns:
            pattern_effectiveness = self._get_pattern_effectiveness(planned_patterns)
            predictions['expected_efficiency'] *= (1 + pattern_effectiveness)
            predictions['success_probability'] = min(1.0, predictions['success_probability'] * (1 + pattern_effectiveness))
        
        return predictions
    
    def _get_pattern_effectiveness(self, pattern_hashes: List[str]) -> float:
        """Get average effectiveness of patterns."""
        if not pattern_hashes:
            return 0.0
        
        placeholders = ",".join(["%s"] * len(pattern_hashes))
        query = f"""
        SELECT AVG(effectiveness_score) as avg_effectiveness
        FROM discovered_patterns
        WHERE pattern_hash IN ({placeholders})
        """
        
        with self.metrics_db.get_connection() as conn:
            cursor = conn.cursor()
            
            if self.metrics_db.is_postgres:
                cursor.execute(query, pattern_hashes)
            else:
                query = query.replace("%s", "?")
                cursor.execute(query, pattern_hashes)
            
            result = cursor.fetchone()
            return result[0] if result and result[0] else 0.0
    
    def _store_insight(self, insight: KnowledgeEntry) -> None:
        """Store insight in database and cache."""
        with self.metrics_db.get_connection() as conn:
            cursor = conn.cursor()
            
            if self.metrics_db.is_postgres:
                cursor.execute("""
                    INSERT INTO learned_insights (
                        id, insight_type, description, confidence,
                        supporting_evidence, applicable_contexts,
                        validation_count, success_rate, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        validation_count = learned_insights.validation_count + %s,
                        success_rate = %s,
                        confidence = %s
                """, (
                    insight.entry_id, insight.insight_type.value,
                    insight.description, insight.confidence,
                    json.dumps(insight.evidence),
                    json.dumps(insight.applicable_contexts),
                    insight.validation_count, insight.success_rate,
                    json.dumps(insight.metadata),
                    1, insight.success_rate, insight.confidence
                ))
            else:
                cursor.execute("""
                    INSERT OR REPLACE INTO learned_insights (
                        id, insight_type, description, confidence,
                        supporting_evidence, applicable_contexts,
                        validation_count, success_rate, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    insight.entry_id, insight.insight_type.value,
                    insight.description, insight.confidence,
                    json.dumps(insight.evidence),
                    json.dumps(insight.applicable_contexts),
                    insight.validation_count, insight.success_rate,
                    json.dumps(insight.metadata)
                ))
            
            conn.commit()
        
        # Update cache
        self.insights_cache[insight.entry_id] = insight
    
    def _row_to_entry(self, row: Dict) -> Optional[KnowledgeEntry]:
        """Convert database row to KnowledgeEntry."""
        try:
            return KnowledgeEntry(
                entry_id=row['id'],
                insight_type=InsightType(row['insight_type']),
                description=row['description'],
                confidence=row['confidence'],
                evidence=json.loads(row.get('supporting_evidence', '[]')),
                applicable_contexts=json.loads(row.get('applicable_contexts', '[]')),
                validation_count=row.get('validation_count', 0),
                success_rate=row.get('success_rate', 0.0),
                discovered_at=datetime.fromisoformat(row['discovered_at']) if 'discovered_at' in row else datetime.now(),
                metadata=json.loads(row.get('metadata', '{}'))
            )
        except Exception as e:
            logger.error(f"Error converting row to entry: {e}")
            return None
    
    def _initialize_learning_rules(self) -> List['LearningRule']:
        """Initialize the learning rules for insight extraction."""
        return [
            OptimizationStrategyRule(),
            PatternSynergyRule(),
            ConstraintDiscoveryRule(),
            FailureModeRule(),
            PerformancePredictorRule()
        ]


class LearningRule:
    """Base class for learning rules that extract insights."""
    
    @property
    def name(self) -> str:
        """Name of the learning rule."""
        return self.__class__.__name__
    
    def extract(self, 
                metrics_db: MetricsDatabase,
                min_confidence: float) -> List[KnowledgeEntry]:
        """Extract insights using this rule."""
        raise NotImplementedError


class OptimizationStrategyRule(LearningRule):
    """Identifies successful optimization strategies."""
    
    def extract(self, metrics_db: MetricsDatabase, min_confidence: float) -> List[KnowledgeEntry]:
        insights = []
        
        # Find patterns that consistently improve performance
        query = """
        SELECT 
            dp.pattern_type,
            dp.description,
            COUNT(DISTINCT pa.agent_id) as usage_count,
            AVG(pa.performance_delta) as avg_improvement,
            STDDEV(pa.performance_delta) as improvement_variance
        FROM discovered_patterns dp
        JOIN pattern_adoptions pa ON dp.id = pa.pattern_id
        WHERE pa.success = %s
            AND pa.performance_delta > 0
        GROUP BY dp.pattern_type, dp.description
        HAVING COUNT(DISTINCT pa.agent_id) >= 5
            AND AVG(pa.performance_delta) > 0.1
        """
        
        with metrics_db.get_connection() as conn:
            cursor = conn.cursor()
            
            if metrics_db.is_postgres:
                from psycopg2.extras import RealDictCursor
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute(query, (True,))
            else:
                query = query.replace("%s", "?")
                cursor.execute(query, (1,))
            
            for row in cursor.fetchall():
                row = dict(row)
                confidence = min(1.0, row['usage_count'] / 20) * (1 - min(0.5, row['improvement_variance']))
                
                if confidence >= min_confidence:
                    insight = KnowledgeEntry(
                        entry_id=f"opt_strategy_{hash(row['description'])}",
                        insight_type=InsightType.OPTIMIZATION_STRATEGY,
                        description=f"Strategy '{row['description']}' improves performance by {row['avg_improvement']:.1%} on average",
                        confidence=confidence,
                        evidence=[{
                            'pattern_type': row['pattern_type'],
                            'usage_count': row['usage_count'],
                            'avg_improvement': row['avg_improvement']
                        }],
                        applicable_contexts=[row['pattern_type']],
                        validation_count=row['usage_count']
                    )
                    insights.append(insight)
        
        return insights


class PatternSynergyRule(LearningRule):
    """Identifies synergistic pattern combinations."""
    
    def extract(self, metrics_db: MetricsDatabase, min_confidence: float) -> List[KnowledgeEntry]:
        # Implementation handled by identify_pattern_synergies method
        return []


class ConstraintDiscoveryRule(LearningRule):
    """Discovers system constraints and limits."""
    
    def extract(self, metrics_db: MetricsDatabase, min_confidence: float) -> List[KnowledgeEntry]:
        insights = []
        
        # Find performance ceilings
        query = """
        SELECT 
            action_type,
            MAX(efficiency) as max_efficiency,
            AVG(efficiency) as avg_efficiency,
            COUNT(*) as sample_count
        FROM performance_metrics
        GROUP BY action_type
        HAVING COUNT(*) >= 50
        """
        
        with metrics_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            
            for row in cursor.fetchall():
                row = dict(row)
                if row['max_efficiency'] > row['avg_efficiency'] * 2:
                    insight = KnowledgeEntry(
                        entry_id=f"constraint_{row['action_type']}",
                        insight_type=InsightType.CONSTRAINT_DISCOVERY,
                        description=f"Action '{row['action_type']}' has efficiency ceiling at {row['max_efficiency']:.2f}",
                        confidence=min(1.0, row['sample_count'] / 100),
                        evidence=[row],
                        applicable_contexts=[row['action_type']]
                    )
                    insights.append(insight)
        
        return insights


class FailureModeRule(LearningRule):
    """Identifies common failure patterns."""
    
    def extract(self, metrics_db: MetricsDatabase, min_confidence: float) -> List[KnowledgeEntry]:
        insights = []
        
        # Find patterns associated with poor performance
        query = """
        SELECT 
            dp.description,
            COUNT(*) as failure_count,
            AVG(pa.performance_delta) as avg_decline
        FROM discovered_patterns dp
        JOIN pattern_adoptions pa ON dp.id = pa.pattern_id
        WHERE pa.success = %s
            AND pa.performance_delta < -0.1
        GROUP BY dp.description
        HAVING COUNT(*) >= 3
        """
        
        with metrics_db.get_connection() as conn:
            cursor = conn.cursor()
            
            if metrics_db.is_postgres:
                from psycopg2.extras import RealDictCursor
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute(query, (False,))
            else:
                query = query.replace("%s", "?")
                cursor.execute(query, (0,))
            
            for row in cursor.fetchall():
                row = dict(row)
                insight = KnowledgeEntry(
                    entry_id=f"failure_{hash(row['description'])}",
                    insight_type=InsightType.FAILURE_MODE,
                    description=f"Pattern '{row['description']}' associated with {row['avg_decline']:.1%} performance decline",
                    confidence=min(1.0, row['failure_count'] / 10),
                    evidence=[row],
                    validation_count=row['failure_count']
                )
                insights.append(insight)
        
        return insights


class PerformancePredictorRule(LearningRule):
    """Learns to predict performance based on features."""
    
    def extract(self, metrics_db: MetricsDatabase, min_confidence: float) -> List[KnowledgeEntry]:
        # Complex implementation would use ML models
        # For now, return simple correlations
        return []