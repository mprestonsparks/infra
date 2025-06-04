"""
Repository manager that orchestrates the knowledge repository components.

Provides a unified interface for storing, retrieving, and analyzing
knowledge across the DEAN system evolution process.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import asyncio
from pathlib import Path
import numpy as np

try:
    from .metrics_database import MetricsDatabase, MetricRecord
except ImportError:
    from .simple_metrics_db import SimpleMetricsDatabase as MetricsDatabase, MetricRecord

from .knowledge_base import KnowledgeBase, KnowledgeEntry, InsightType
from .meta_learner import MetaLearner, LearningStrategy, MetaKnowledge
from ..patterns import Pattern, PatternCatalog
from ..diversity import AgentGenome
from ..economy import TokenEconomyManager

logger = logging.getLogger(__name__)


@dataclass
class RepositoryStats:
    """Statistics about the knowledge repository."""
    total_patterns: int
    total_agents: int
    total_insights: int
    meta_knowledge_items: int
    database_size_mb: float
    last_updated: datetime


class RepositoryManager:
    """
    Unified manager for the DEAN knowledge repository.
    
    Orchestrates all repository components and provides
    high-level interfaces for knowledge management.
    """
    
    def __init__(self,
                 db_url: str = "sqlite:///dean_repository.db",
                 pattern_catalog: Optional[PatternCatalog] = None,
                 auto_learn: bool = True,
                 learning_interval: int = 3600):  # 1 hour
        """
        Initialize repository manager.
        
        Args:
            db_url: Database connection URL
            pattern_catalog: External pattern catalog (optional)
            auto_learn: Enable automatic meta-learning
            learning_interval: Seconds between auto-learning cycles
        """
        self.metrics_db = MetricsDatabase(db_url)
        self.knowledge_base = KnowledgeBase(self.metrics_db)
        self.meta_learner = MetaLearner(self.metrics_db, self.knowledge_base)
        
        # Optional external pattern catalog
        self.pattern_catalog = pattern_catalog
        
        # Auto-learning configuration
        self.auto_learn = auto_learn
        self.learning_interval = learning_interval
        self._learning_task = None
        self._running = False
        
        # Caches for performance
        self._stats_cache: Optional[RepositoryStats] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=15)
    
    async def start(self) -> None:
        """Start the repository manager and auto-learning."""
        self._running = True
        
        if self.auto_learn:
            self._learning_task = asyncio.create_task(self._auto_learning_loop())
            logger.info("Started auto-learning with interval: {} seconds", self.learning_interval)
        
        logger.info("Repository manager started")
    
    async def stop(self) -> None:
        """Stop the repository manager."""
        self._running = False
        
        if self._learning_task:
            self._learning_task.cancel()
            try:
                await self._learning_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Repository manager stopped")
    
    def record_agent_evolution(self,
                              agent_id: str,
                              genome: AgentGenome,
                              performance_metrics: Dict[str, Any],
                              discovered_patterns: List[Pattern]) -> None:
        """
        Record complete agent evolution data.
        
        This is the main entry point for storing agent evolution results.
        """
        try:
            # Store agent metrics
            agent_metrics = {
                'agent_id': agent_id,
                'genome_hash': genome.calculate_hash(),
                'generation': genome.generation,
                'tokens_used': performance_metrics.get('tokens_used', 0),
                'value_generated': performance_metrics.get('value_generated', 0.0),
                'efficiency': performance_metrics.get('efficiency', 0.0),
                'action_type': performance_metrics.get('action_type', 'evolution'),
                **performance_metrics
            }
            
            self.metrics_db.record_agent_metrics(agent_id, agent_metrics)
            
            # Store discovered patterns
            for pattern in discovered_patterns:
                pattern_data = {
                    'pattern_hash': pattern.calculate_hash(),
                    'pattern_type': pattern.pattern_type.value,
                    'description': pattern.description,
                    'effectiveness_score': pattern.effectiveness,
                    'token_efficiency': performance_metrics.get('efficiency', 0.0),
                    'confidence_score': pattern.confidence,
                    'discovered_by': agent_id,
                    **pattern.to_dict()
                }
                
                pattern_id = self.metrics_db.insert_pattern(pattern_data)
                
                # Also store in pattern catalog if available
                if self.pattern_catalog:
                    self.pattern_catalog.add_pattern(
                        pattern=pattern,
                        discovered_by=agent_id,
                        tags=['discovered', f'gen_{genome.generation}']
                    )
            
            # Invalidate stats cache
            self._invalidate_cache()
            
        except Exception as e:
            logger.error(f"Error recording agent evolution for {agent_id}: {e}")
    
    def query_optimization_strategies(self,
                                    current_context: Dict[str, Any],
                                    objective: str = "maximize_efficiency") -> Dict[str, Any]:
        """
        Get optimization recommendations based on learned knowledge.
        
        Combines pattern analysis, insights, and meta-knowledge
        to provide comprehensive optimization guidance.
        """
        recommendations = {
            'patterns': [],
            'insights': [],
            'meta_knowledge': [],
            'confidence_score': 0.0,
            'estimated_improvement': 0.0
        }
        
        try:
            # Get pattern-based recommendations
            patterns = self.knowledge_base.get_optimization_strategies(
                target_metric=objective,
                current_performance=current_context.get('current_performance', 0.0)
            )
            recommendations['patterns'] = patterns[:5]  # Top 5
            
            # Get relevant insights
            insights = self.knowledge_base.query_insights(
                insight_type=InsightType.OPTIMIZATION_STRATEGY,
                context=objective,
                only_reliable=True
            )
            recommendations['insights'] = [
                {
                    'description': i.description,
                    'confidence': i.confidence,
                    'success_rate': i.success_rate
                }
                for i in insights[:3]
            ]
            
            # Apply meta-knowledge
            meta_recommendations = self.meta_learner.apply_meta_knowledge(
                context=current_context,
                objective=objective
            )
            recommendations['meta_knowledge'] = meta_recommendations
            
            # Calculate overall confidence
            pattern_confidence = np.mean([p.get('avg_effectiveness', 0) for p in patterns]) if patterns else 0
            insight_confidence = np.mean([i.confidence for i in insights]) if insights else 0
            meta_confidence = meta_recommendations.get('confidence_scores', {}).get('patterns', 0)
            
            recommendations['confidence_score'] = np.mean([
                c for c in [pattern_confidence, insight_confidence, meta_confidence] if c > 0
            ])
            
            # Estimate improvement
            if patterns:
                recommendations['estimated_improvement'] = np.mean([
                    p.get('avg_improvement', 0) for p in patterns
                ])
            
        except Exception as e:
            logger.error(f"Error generating optimization strategies: {e}")
        
        return recommendations
    
    def analyze_evolution_progress(self,
                                 generation_range: Optional[Tuple[int, int]] = None,
                                 include_predictions: bool = True) -> Dict[str, Any]:
        """
        Comprehensive analysis of evolution progress.
        
        Provides detailed analytics on how the population
        has evolved and predictions for future progress.
        """
        analysis = {
            'progress_metrics': {},
            'trend_analysis': {},
            'pattern_evolution': {},
            'efficiency_trends': {},
            'predictions': {} if include_predictions else None
        }
        
        try:
            # Basic progress metrics
            progress = self.metrics_db.analyze_evolution_progress(generation_range)
            analysis['progress_metrics'] = progress
            
            # Pattern evolution analysis
            pattern_trends = self._analyze_pattern_trends(generation_range)
            analysis['pattern_evolution'] = pattern_trends
            
            # Efficiency trend analysis
            efficiency_trends = self._analyze_efficiency_trends(generation_range)
            analysis['efficiency_trends'] = efficiency_trends
            
            # Meta-learning insights
            recent_insights = self.knowledge_base.query_insights(
                min_confidence=0.7,
                only_reliable=True
            )
            analysis['recent_insights'] = [
                {
                    'type': i.insight_type.value,
                    'description': i.description,
                    'confidence': i.confidence
                }
                for i in recent_insights[-10:]  # Last 10 insights
            ]
            
            if include_predictions:
                # Generate predictions using meta-learner
                analysis['predictions'] = self._generate_predictions(progress)
            
        except Exception as e:
            logger.error(f"Error analyzing evolution progress: {e}")
        
        return analysis
    
    def export_knowledge_snapshot(self, output_dir: str) -> Dict[str, str]:
        """
        Export complete knowledge snapshot for backup or analysis.
        
        Returns dictionary mapping export types to file paths.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exports = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Export metrics database
            db_export_path = output_path / f"metrics_db_{timestamp}.json"
            self.metrics_db.export_knowledge_base(str(db_export_path))
            exports['metrics_database'] = str(db_export_path)
            
            # Export pattern catalog
            if self.pattern_catalog:
                catalog_export_path = output_path / f"pattern_catalog_{timestamp}.json"
                self.pattern_catalog.export_catalog(str(catalog_export_path))
                exports['pattern_catalog'] = str(catalog_export_path)
            
            # Export meta-knowledge
            meta_export_path = output_path / f"meta_knowledge_{timestamp}.json"
            self.meta_learner.export_meta_knowledge(str(meta_export_path))
            exports['meta_knowledge'] = str(meta_export_path)
            
            # Export repository statistics
            stats_export_path = output_path / f"repository_stats_{timestamp}.json"
            stats = self.get_repository_stats()
            import json
            with open(stats_export_path, 'w') as f:
                json.dump(stats.__dict__, f, indent=2, default=str)
            exports['statistics'] = str(stats_export_path)
            
            logger.info(f"Exported knowledge snapshot to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error exporting knowledge snapshot: {e}")
        
        return exports
    
    def get_repository_stats(self, use_cache: bool = True) -> RepositoryStats:
        """Get comprehensive repository statistics."""
        if (use_cache and self._stats_cache and self._cache_timestamp and 
            datetime.now() - self._cache_timestamp < self._cache_ttl):
            return self._stats_cache
        
        try:
            # Query database for counts
            total_patterns = self.metrics_db._count_records('discovered_patterns')
            total_agents = self.metrics_db._count_records('agents')
            
            # Try to count insights (may not exist in basic schema)
            try:
                total_insights = self.metrics_db._count_records('learned_insights')
            except:
                total_insights = len(self.knowledge_base.insights_cache)
            
            meta_knowledge_items = len(self.meta_learner.meta_knowledge)
            
            # Estimate database size
            db_size_mb = 0.0
            if hasattr(self.metrics_db, 'db_path'):
                db_path = Path(self.metrics_db.db_path)
                if db_path.exists():
                    db_size_mb = db_path.stat().st_size / (1024 * 1024)
            
            stats = RepositoryStats(
                total_patterns=total_patterns,
                total_agents=total_agents,
                total_insights=total_insights,
                meta_knowledge_items=meta_knowledge_items,
                database_size_mb=db_size_mb,
                last_updated=datetime.now()
            )
            
            # Update cache
            self._stats_cache = stats
            self._cache_timestamp = datetime.now()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting repository stats: {e}")
            return RepositoryStats(0, 0, 0, 0, 0.0, datetime.now())
    
    async def _auto_learning_loop(self) -> None:
        """Background auto-learning loop."""
        while self._running:
            try:
                await asyncio.sleep(self.learning_interval)
                
                if not self._running:
                    break
                
                # Extract new insights
                insights = self.knowledge_base.extract_insights(min_confidence=0.6)
                if insights:
                    logger.info(f"Auto-learning extracted {len(insights)} new insights")
                
                # Learn meta-knowledge
                meta_knowledge = self.meta_learner.learn_from_population(
                    generation_window=10,
                    min_sample_size=20
                )
                if meta_knowledge:
                    logger.info(f"Auto-learning discovered {len(meta_knowledge)} meta-knowledge items")
                
                # Invalidate cache after learning
                self._invalidate_cache()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-learning loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    def _analyze_pattern_trends(self, generation_range: Optional[Tuple[int, int]]) -> Dict:
        """Analyze how patterns have evolved over generations."""
        trends = {
            'pattern_discovery_rate': 0.0,
            'pattern_adoption_rate': 0.0,
            'pattern_effectiveness_trend': 'stable',
            'top_emerging_patterns': []
        }
        
        try:
            # Query recent pattern discoveries
            query = """
            SELECT 
                DATE_TRUNC('day', created_at) as discovery_date,
                COUNT(*) as patterns_discovered,
                AVG(effectiveness_score) as avg_effectiveness
            FROM discovered_patterns
            WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '30 days'
            GROUP BY discovery_date
            ORDER BY discovery_date
            """
            
            with self.metrics_db.get_connection() as conn:
                cursor = conn.cursor()
                
                if self.metrics_db.is_postgres:
                    cursor.execute(query)
                else:
                    # SQLite version
                    query = """
                    SELECT 
                        DATE(created_at) as discovery_date,
                        COUNT(*) as patterns_discovered,
                        AVG(effectiveness_score) as avg_effectiveness
                    FROM discovered_patterns
                    WHERE datetime(created_at) >= datetime('now', '-30 days')
                    GROUP BY discovery_date
                    ORDER BY discovery_date
                    """
                    cursor.execute(query)
                
                results = [dict(row) for row in cursor.fetchall()]
                
                if results:
                    discovery_rates = [r['patterns_discovered'] for r in results]
                    trends['pattern_discovery_rate'] = np.mean(discovery_rates)
                    
                    effectiveness_scores = [r['avg_effectiveness'] for r in results]
                    if len(effectiveness_scores) >= 2:
                        if effectiveness_scores[-1] > effectiveness_scores[0] * 1.1:
                            trends['pattern_effectiveness_trend'] = 'improving'
                        elif effectiveness_scores[-1] < effectiveness_scores[0] * 0.9:
                            trends['pattern_effectiveness_trend'] = 'degrading'
        
        except Exception as e:
            logger.error(f"Error analyzing pattern trends: {e}")
        
        return trends
    
    def _analyze_efficiency_trends(self, generation_range: Optional[Tuple[int, int]]) -> Dict:
        """Analyze efficiency trends across generations."""
        trends = {
            'overall_trend': 'stable',
            'efficiency_improvement_rate': 0.0,
            'best_performing_generation': 0,
            'efficiency_variance': 0.0
        }
        
        try:
            progress = self.metrics_db.analyze_evolution_progress(generation_range)
            generations = progress.get('generations', [])
            
            if len(generations) >= 2:
                efficiencies = [g['avg_efficiency'] for g in generations]
                
                # Calculate trend
                first_eff = efficiencies[0]
                last_eff = efficiencies[-1]
                
                if last_eff > first_eff * 1.1:
                    trends['overall_trend'] = 'improving'
                elif last_eff < first_eff * 0.9:
                    trends['overall_trend'] = 'degrading'
                
                # Improvement rate per generation
                if len(generations) > 1:
                    improvements = []
                    for i in range(1, len(efficiencies)):
                        if efficiencies[i-1] > 0:
                            improvement = (efficiencies[i] - efficiencies[i-1]) / efficiencies[i-1]
                            improvements.append(improvement)
                    
                    if improvements:
                        trends['efficiency_improvement_rate'] = np.mean(improvements)
                
                # Best generation
                best_gen_idx = np.argmax(efficiencies)
                trends['best_performing_generation'] = generations[best_gen_idx]['generation']
                
                # Variance
                trends['efficiency_variance'] = np.var(efficiencies)
        
        except Exception as e:
            logger.error(f"Error analyzing efficiency trends: {e}")
        
        return trends
    
    def _generate_predictions(self, progress_data: Dict) -> Dict:
        """Generate predictions about future evolution."""
        predictions = {
            'next_generation_efficiency': 0.0,
            'convergence_generation': None,
            'efficiency_ceiling': 0.0,
            'confidence': 0.0
        }
        
        try:
            generations = progress_data.get('generations', [])
            if len(generations) >= 3:
                efficiencies = [g['avg_efficiency'] for g in generations]
                
                # Simple linear extrapolation
                x = list(range(len(efficiencies)))
                z = np.polyfit(x, efficiencies, 1)
                
                # Predict next generation
                next_gen = len(efficiencies)
                predictions['next_generation_efficiency'] = z[0] * next_gen + z[1]
                
                # Estimate confidence based on R²
                y_pred = [z[0] * i + z[1] for i in x]
                ss_res = sum((efficiencies[i] - y_pred[i]) ** 2 for i in range(len(efficiencies)))
                ss_tot = sum((efficiencies[i] - np.mean(efficiencies)) ** 2 for i in range(len(efficiencies)))
                
                if ss_tot > 0:
                    r_squared = 1 - (ss_res / ss_tot)
                    predictions['confidence'] = max(0.0, r_squared)
                
                # Estimate efficiency ceiling (asymptotic behavior)
                if len(efficiencies) >= 5:
                    recent_improvement = efficiencies[-1] - efficiencies[-3]
                    if recent_improvement < 0.01:  # Convergence threshold
                        predictions['efficiency_ceiling'] = max(efficiencies) * 1.1
        
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
        
        return predictions
    
    def _invalidate_cache(self) -> None:
        """Invalidate cached statistics."""
        self._stats_cache = None
        self._cache_timestamp = None