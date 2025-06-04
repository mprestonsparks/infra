"""
Meta-learning capabilities for the DEAN system.

Learns from the collective experience of all agents to
improve future evolution strategies and optimizations.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import logging
from collections import defaultdict

from .metrics_database import MetricsDatabase
from .knowledge_base import KnowledgeBase, InsightType, KnowledgeEntry

logger = logging.getLogger(__name__)


class LearningStrategy(Enum):
    """Meta-learning strategies."""
    PATTERN_TRANSFER = "pattern_transfer"  # Transfer successful patterns
    CONSTRAINT_LEARNING = "constraint_learning"  # Learn system constraints
    FAILURE_AVOIDANCE = "failure_avoidance"  # Learn from failures
    SYNERGY_DISCOVERY = "synergy_discovery"  # Find pattern combinations
    EFFICIENCY_OPTIMIZATION = "efficiency_optimization"  # Optimize resource usage
    DIVERSITY_BALANCING = "diversity_balancing"  # Balance exploration/exploitation


@dataclass
class MetaKnowledge:
    """Encapsulates meta-knowledge learned from evolution."""
    knowledge_id: str
    strategy: LearningStrategy
    description: str
    applicability_score: float  # How broadly applicable
    reliability_score: float  # How reliable/consistent
    impact_score: float  # Impact on performance
    learned_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    success_count: int = 0
    constraints: Dict[str, Any] = field(default_factory=dict)
    evidence: List[Dict] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of applying this knowledge."""
        return self.success_count / self.usage_count if self.usage_count > 0 else 0.0
    
    def is_applicable(self, context: Dict[str, Any]) -> bool:
        """Check if this knowledge is applicable in given context."""
        for key, required_value in self.constraints.items():
            if key not in context:
                return False
            if isinstance(required_value, (list, tuple)):
                if context[key] not in required_value:
                    return False
            elif context[key] != required_value:
                return False
        return True


class MetaLearner:
    """
    Meta-learning system that extracts high-level insights
    from the collective agent experience.
    """
    
    def __init__(self, 
                 metrics_db: MetricsDatabase,
                 knowledge_base: KnowledgeBase):
        """Initialize meta-learner with data sources."""
        self.metrics_db = metrics_db
        self.knowledge_base = knowledge_base
        self.meta_knowledge: Dict[str, MetaKnowledge] = {}
        self.learning_history: List[Dict] = []
    
    def learn_from_population(self,
                            generation_window: int = 10,
                            min_sample_size: int = 20) -> List[MetaKnowledge]:
        """
        Extract meta-knowledge from recent population data.
        
        Args:
            generation_window: Number of recent generations to analyze
            min_sample_size: Minimum data points for reliable learning
            
        Returns:
            List of newly learned meta-knowledge
        """
        new_knowledge = []
        
        # Run each learning strategy
        strategies = [
            self._learn_pattern_transfers,
            self._learn_constraints,
            self._learn_from_failures,
            self._discover_synergies,
            self._optimize_efficiency,
            self._balance_diversity
        ]
        
        for strategy_func in strategies:
            try:
                knowledge = strategy_func(generation_window, min_sample_size)
                new_knowledge.extend(knowledge)
            except Exception as e:
                logger.error(f"Error in meta-learning strategy {strategy_func.__name__}: {e}")
        
        # Store new knowledge
        for mk in new_knowledge:
            self.meta_knowledge[mk.knowledge_id] = mk
            self._record_learning_event(mk)
        
        return new_knowledge
    
    def apply_meta_knowledge(self,
                           context: Dict[str, Any],
                           objective: str = "maximize_efficiency") -> Dict[str, Any]:
        """
        Apply learned meta-knowledge to improve agent configuration.
        
        Args:
            context: Current context (agent state, environment, etc.)
            objective: What to optimize for
            
        Returns:
            Recommendations based on meta-knowledge
        """
        recommendations = {
            'suggested_patterns': [],
            'constraints_to_observe': [],
            'patterns_to_avoid': [],
            'synergistic_combinations': [],
            'resource_allocation': {},
            'confidence_scores': {}
        }
        
        # Find applicable meta-knowledge
        applicable_knowledge = [
            mk for mk in self.meta_knowledge.values()
            if mk.is_applicable(context) and mk.reliability_score > 0.6
        ]
        
        # Sort by relevance to objective
        if objective == "maximize_efficiency":
            applicable_knowledge.sort(key=lambda x: x.impact_score * x.reliability_score, reverse=True)
        
        # Apply each piece of knowledge
        for mk in applicable_knowledge[:10]:  # Top 10 most relevant
            if mk.strategy == LearningStrategy.PATTERN_TRANSFER:
                recommendations['suggested_patterns'].extend(
                    mk.evidence[0].get('successful_patterns', [])
                )
                recommendations['confidence_scores']['patterns'] = mk.reliability_score
                
            elif mk.strategy == LearningStrategy.CONSTRAINT_LEARNING:
                recommendations['constraints_to_observe'].append({
                    'constraint': mk.description,
                    'details': mk.constraints
                })
                
            elif mk.strategy == LearningStrategy.FAILURE_AVOIDANCE:
                recommendations['patterns_to_avoid'].extend(
                    mk.evidence[0].get('failure_patterns', [])
                )
                
            elif mk.strategy == LearningStrategy.SYNERGY_DISCOVERY:
                recommendations['synergistic_combinations'].append({
                    'patterns': mk.evidence[0].get('pattern_combination', []),
                    'expected_synergy': mk.impact_score
                })
                
            elif mk.strategy == LearningStrategy.EFFICIENCY_OPTIMIZATION:
                recommendations['resource_allocation'] = mk.evidence[0].get('optimal_allocation', {})
        
        return recommendations
    
    def _learn_pattern_transfers(self, 
                               generation_window: int,
                               min_sample_size: int) -> List[MetaKnowledge]:
        """Learn which patterns transfer well across contexts."""
        knowledge = []
        
        # Query patterns with high transferability
        query = """
        WITH pattern_contexts AS (
            SELECT 
                dp.pattern_hash,
                dp.description,
                dp.pattern_type,
                pm.action_type as context,
                COUNT(DISTINCT pa.agent_id) as adoption_count,
                AVG(pa.performance_delta) as avg_improvement,
                AVG(CASE WHEN pa.success THEN 1.0 ELSE 0.0 END) as success_rate
            FROM discovered_patterns dp
            JOIN pattern_adoptions pa ON dp.id = pa.pattern_id
            JOIN performance_metrics pm ON pa.agent_id = pm.agent_id
            GROUP BY dp.pattern_hash, dp.description, dp.pattern_type, pm.action_type
            HAVING COUNT(DISTINCT pa.agent_id) >= %s
        )
        SELECT 
            pattern_hash,
            description,
            pattern_type,
            COUNT(DISTINCT context) as context_count,
            AVG(avg_improvement) as overall_improvement,
            AVG(success_rate) as overall_success_rate,
            MIN(adoption_count) as min_adoptions
        FROM pattern_contexts
        GROUP BY pattern_hash, description, pattern_type
        HAVING COUNT(DISTINCT context) >= 3  -- Used in at least 3 contexts
            AND AVG(success_rate) >= 0.7
        """
        
        with self.metrics_db.get_connection() as conn:
            cursor = conn.cursor()
            
            if self.metrics_db.is_postgres:
                from psycopg2.extras import RealDictCursor
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute(query, (min_sample_size // 5,))
            else:
                query = query.replace("%s", "?")
                cursor.execute(query, (min_sample_size // 5,))
            
            for row in cursor.fetchall():
                row = dict(row)
                mk = MetaKnowledge(
                    knowledge_id=f"transfer_{row['pattern_hash']}",
                    strategy=LearningStrategy.PATTERN_TRANSFER,
                    description=f"Pattern '{row['description']}' transfers well across {row['context_count']} contexts",
                    applicability_score=row['context_count'] / 10,  # Normalize
                    reliability_score=row['overall_success_rate'],
                    impact_score=row['overall_improvement'],
                    evidence=[{
                        'pattern_hash': row['pattern_hash'],
                        'successful_patterns': [row['pattern_hash']],
                        'contexts': row['context_count'],
                        'avg_improvement': row['overall_improvement']
                    }]
                )
                knowledge.append(mk)
        
        return knowledge
    
    def _learn_constraints(self,
                          generation_window: int,
                          min_sample_size: int) -> List[MetaKnowledge]:
        """Learn system constraints and boundaries."""
        knowledge = []
        
        # Find performance ceilings and resource limits
        query = """
        SELECT 
            action_type,
            MAX(efficiency) as max_efficiency,
            AVG(efficiency) as avg_efficiency,
            STDDEV(efficiency) as efficiency_stddev,
            MAX(tokens_used) as max_tokens,
            AVG(tokens_used) as avg_tokens,
            COUNT(*) as sample_count
        FROM performance_metrics
        WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '%s days'
        GROUP BY action_type
        HAVING COUNT(*) >= %s
        """
        
        with self.metrics_db.get_connection() as conn:
            cursor = conn.cursor()
            
            if self.metrics_db.is_postgres:
                from psycopg2.extras import RealDictCursor
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute(query % (generation_window, min_sample_size))
            else:
                # SQLite datetime handling
                query = """
                SELECT 
                    action_type,
                    MAX(efficiency) as max_efficiency,
                    AVG(efficiency) as avg_efficiency,
                    MAX(tokens_used) as max_tokens,
                    AVG(tokens_used) as avg_tokens,
                    COUNT(*) as sample_count
                FROM performance_metrics
                WHERE datetime(timestamp) >= datetime('now', '-%d days')
                GROUP BY action_type
                HAVING COUNT(*) >= %d
                """ % (generation_window, min_sample_size)
                cursor.execute(query)
            
            for row in cursor.fetchall():
                row = dict(row)
                
                # Identify hard limits
                if row['max_efficiency'] < row['avg_efficiency'] * 1.5:
                    mk = MetaKnowledge(
                        knowledge_id=f"constraint_efficiency_{row['action_type']}",
                        strategy=LearningStrategy.CONSTRAINT_LEARNING,
                        description=f"Action '{row['action_type']}' has efficiency ceiling at {row['max_efficiency']:.2f}",
                        applicability_score=1.0,
                        reliability_score=min(1.0, row['sample_count'] / 100),
                        impact_score=row['max_efficiency'] - row['avg_efficiency'],
                        constraints={'action_type': row['action_type']},
                        evidence=[row]
                    )
                    knowledge.append(mk)
        
        return knowledge
    
    def _learn_from_failures(self,
                           generation_window: int,
                           min_sample_size: int) -> List[MetaKnowledge]:
        """Learn from failure patterns to avoid."""
        knowledge = []
        
        # Find patterns associated with failures
        failure_insights = self.knowledge_base.query_insights(
            insight_type=InsightType.FAILURE_MODE,
            min_confidence=0.6
        )
        
        for insight in failure_insights:
            if insight.validation_count >= min_sample_size // 10:
                mk = MetaKnowledge(
                    knowledge_id=f"avoid_{insight.entry_id}",
                    strategy=LearningStrategy.FAILURE_AVOIDANCE,
                    description=insight.description,
                    applicability_score=0.8,
                    reliability_score=insight.confidence,
                    impact_score=-abs(insight.evidence[0].get('avg_decline', 0)),
                    evidence=[{
                        'failure_patterns': [insight.entry_id],
                        'failure_rate': 1 - insight.success_rate
                    }]
                )
                knowledge.append(mk)
        
        return knowledge
    
    def _discover_synergies(self,
                          generation_window: int,
                          min_sample_size: int) -> List[MetaKnowledge]:
        """Discover synergistic pattern combinations."""
        knowledge = []
        
        # Get pattern synergies from knowledge base
        synergies = self.knowledge_base.identify_pattern_synergies(min_synergy=1.3)
        
        for synergy in synergies[:10]:  # Top 10 synergies
            if synergy['combination_count'] >= min_sample_size // 10:
                mk = MetaKnowledge(
                    knowledge_id=f"synergy_{synergy['pattern1']}_{synergy['pattern2']}",
                    strategy=LearningStrategy.SYNERGY_DISCOVERY,
                    description=f"Combining '{synergy['pattern1_desc']}' with '{synergy['pattern2_desc']}' "
                               f"yields {synergy['avg_synergy']:.1f}x synergy",
                    applicability_score=0.7,
                    reliability_score=min(1.0, synergy['combination_count'] / 20),
                    impact_score=synergy['avg_synergy'],
                    evidence=[{
                        'pattern_combination': [synergy['pattern1'], synergy['pattern2']],
                        'synergy_factor': synergy['avg_synergy']
                    }]
                )
                knowledge.append(mk)
        
        return knowledge
    
    def _optimize_efficiency(self,
                           generation_window: int,
                           min_sample_size: int) -> List[MetaKnowledge]:
        """Learn optimal resource allocation strategies."""
        knowledge = []
        
        # Analyze token allocation effectiveness
        query = """
        WITH agent_allocations AS (
            SELECT 
                a.agent_id,
                a.generation,
                ta.token_amount as allocated_tokens,
                a.total_tokens_used as used_tokens,
                a.total_value_generated as value_generated,
                a.efficiency_score,
                ta.performance_multiplier
            FROM agents a
            JOIN token_allocations ta ON a.agent_id = ta.agent_id
            WHERE a.generation >= (SELECT MAX(generation) - %s FROM agents)
        )
        SELECT 
            CASE 
                WHEN performance_multiplier < 0.8 THEN 'low_performer'
                WHEN performance_multiplier < 1.2 THEN 'average_performer'
                ELSE 'high_performer'
            END as performer_class,
            AVG(allocated_tokens) as avg_allocation,
            AVG(used_tokens::REAL / allocated_tokens) as utilization_rate,
            AVG(efficiency_score) as avg_efficiency,
            COUNT(*) as sample_count
        FROM agent_allocations
        GROUP BY performer_class
        HAVING COUNT(*) >= %s
        """
        
        with self.metrics_db.get_connection() as conn:
            cursor = conn.cursor()
            
            if self.metrics_db.is_postgres:
                from psycopg2.extras import RealDictCursor
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute(query, (generation_window, min_sample_size // 3))
            else:
                query = query.replace("%s", "?").replace("::REAL", "")
                cursor.execute(query, (generation_window, min_sample_size // 3))
            
            results = [dict(row) for row in cursor.fetchall()]
            
            if len(results) >= 2:
                # Learn optimal allocation strategy
                optimal_allocation = {}
                for row in results:
                    optimal_allocation[row['performer_class']] = {
                        'recommended_tokens': int(row['avg_allocation']),
                        'expected_utilization': row['utilization_rate'],
                        'expected_efficiency': row['avg_efficiency']
                    }
                
                mk = MetaKnowledge(
                    knowledge_id="efficiency_token_allocation",
                    strategy=LearningStrategy.EFFICIENCY_OPTIMIZATION,
                    description="Optimal token allocation based on performer class",
                    applicability_score=0.9,
                    reliability_score=min(1.0, sum(r['sample_count'] for r in results) / 100),
                    impact_score=max(r['avg_efficiency'] for r in results) - min(r['avg_efficiency'] for r in results),
                    evidence=[{'optimal_allocation': optimal_allocation}]
                )
                knowledge.append(mk)
        
        return knowledge
    
    def _balance_diversity(self,
                         generation_window: int,
                         min_sample_size: int) -> List[MetaKnowledge]:
        """Learn optimal diversity balance strategies."""
        knowledge = []
        
        # Analyze diversity vs performance trade-offs
        query = """
        SELECT 
            dm.generation,
            dm.genetic_diversity,
            dm.convergence_risk,
            AVG(a.efficiency_score) as avg_efficiency,
            COUNT(DISTINCT a.agent_id) as population_size
        FROM diversity_metrics dm
        JOIN agents a ON dm.generation = a.generation
        WHERE dm.generation >= (SELECT MAX(generation) - %s FROM agents)
        GROUP BY dm.generation, dm.genetic_diversity, dm.convergence_risk
        HAVING COUNT(DISTINCT a.agent_id) >= %s
        """
        
        with self.metrics_db.get_connection() as conn:
            cursor = conn.cursor()
            
            if self.metrics_db.is_postgres:
                from psycopg2.extras import RealDictCursor
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute(query, (generation_window, min_sample_size // 5))
            else:
                query = query.replace("%s", "?")
                cursor.execute(query, (generation_window, min_sample_size // 5))
            
            data_points = [dict(row) for row in cursor.fetchall()]
            
            if len(data_points) >= 5:
                # Find optimal diversity range
                diversities = [dp['genetic_diversity'] for dp in data_points]
                efficiencies = [dp['avg_efficiency'] for dp in data_points]
                
                # Simple correlation analysis
                if np.std(diversities) > 0 and np.std(efficiencies) > 0:
                    correlation = np.corrcoef(diversities, efficiencies)[0, 1]
                    
                    # Find sweet spot
                    best_points = sorted(data_points, key=lambda x: x['avg_efficiency'], reverse=True)[:3]
                    optimal_diversity = np.mean([bp['genetic_diversity'] for bp in best_points])
                    
                    mk = MetaKnowledge(
                        knowledge_id="diversity_balance",
                        strategy=LearningStrategy.DIVERSITY_BALANCING,
                        description=f"Optimal genetic diversity around {optimal_diversity:.2f} for best performance",
                        applicability_score=0.8,
                        reliability_score=abs(correlation),
                        impact_score=max(efficiencies) - min(efficiencies),
                        constraints={'target_diversity': (optimal_diversity - 0.1, optimal_diversity + 0.1)},
                        evidence=[{
                            'optimal_diversity': optimal_diversity,
                            'correlation': correlation,
                            'sample_points': len(data_points)
                        }]
                    )
                    knowledge.append(mk)
        
        return knowledge
    
    def validate_meta_knowledge(self, mk_id: str, success: bool) -> None:
        """Update meta-knowledge based on application results."""
        if mk_id in self.meta_knowledge:
            mk = self.meta_knowledge[mk_id]
            mk.usage_count += 1
            if success:
                mk.success_count += 1
            
            # Update reliability score
            mk.reliability_score = mk.reliability_score * 0.9 + mk.success_rate * 0.1
    
    def export_meta_knowledge(self, filepath: str) -> None:
        """Export meta-knowledge for analysis or transfer."""
        export_data = {
            'export_date': datetime.now().isoformat(),
            'total_knowledge_items': len(self.meta_knowledge),
            'knowledge': [
                {
                    'id': mk.knowledge_id,
                    'strategy': mk.strategy.value,
                    'description': mk.description,
                    'scores': {
                        'applicability': mk.applicability_score,
                        'reliability': mk.reliability_score,
                        'impact': mk.impact_score,
                        'success_rate': mk.success_rate
                    },
                    'usage_stats': {
                        'usage_count': mk.usage_count,
                        'success_count': mk.success_count
                    },
                    'constraints': mk.constraints,
                    'evidence': mk.evidence
                }
                for mk in self.meta_knowledge.values()
            ],
            'learning_history': self.learning_history[-100:]  # Last 100 events
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported {len(self.meta_knowledge)} meta-knowledge items to {filepath}")
    
    def _record_learning_event(self, mk: MetaKnowledge) -> None:
        """Record a learning event for audit trail."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'knowledge_id': mk.knowledge_id,
            'strategy': mk.strategy.value,
            'description': mk.description[:100] + '...' if len(mk.description) > 100 else mk.description
        }
        self.learning_history.append(event)
        
        # Keep history bounded
        if len(self.learning_history) > 1000:
            self.learning_history = self.learning_history[-1000:]