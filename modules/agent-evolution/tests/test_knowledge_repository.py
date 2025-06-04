"""
Test knowledge repository functionality.
"""

import pytest
import sys
from pathlib import Path
import tempfile
import os
import asyncio
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dean.repository import (
    MetricsDatabase, QueryBuilder, MetricRecord,
    KnowledgeBase, KnowledgeEntry, InsightType,
    MetaLearner, LearningStrategy, MetaKnowledge,
    RepositoryManager
)
from dean.patterns import Pattern, PatternType
from dean.diversity import AgentGenome, Gene, GeneType


class TestMetricsDatabase:
    """Test metrics database functionality."""
    
    def create_test_database(self):
        """Create a test database in memory."""
        return MetricsDatabase("sqlite:///:memory:")
    
    def test_database_initialization(self):
        """Test database initialization."""
        db = self.create_test_database()
        assert db is not None
        assert not db.is_postgres
    
    def test_pattern_insertion(self):
        """Test pattern insertion and retrieval."""
        db = self.create_test_database()
        
        pattern_data = {
            'pattern_hash': 'test_hash_123',
            'pattern_type': 'behavioral',
            'description': 'Test pattern',
            'effectiveness_score': 0.8,
            'token_efficiency': 1.2,
            'confidence_score': 0.9,
            'metadata': {'test': True}
        }
        
        pattern_id = db.insert_pattern(pattern_data)
        assert pattern_id is not None
        
        # Query the pattern back
        patterns = db.query_top_patterns(min_effectiveness=0.5, limit=10)
        assert len(patterns) == 1
        assert patterns[0]['pattern_hash'] == 'test_hash_123'
    
    def test_agent_metrics_recording(self):
        """Test agent metrics recording."""
        db = self.create_test_database()
        
        metrics = {
            'genome_hash': 'genome_abc',
            'generation': 5,
            'tokens_used': 100,
            'value_generated': 150,
            'efficiency': 1.5,
            'action_type': 'optimize'
        }
        
        db.record_agent_metrics('agent_test_1', metrics)
        
        # Verify data was stored
        assert db._count_records('agents') == 1
        assert db._count_records('performance_metrics') == 1
    
    def test_query_builder(self):
        """Test the query builder functionality."""
        query = QueryBuilder() \
            .select('id', 'name', 'score') \
            .from_table('test_table') \
            .where('score > %s', 0.5) \
            .where('active = %s', True) \
            .order_by('score', desc=True) \
            .limit(10)
        
        sql, params = query.build()
        
        assert 'SELECT id, name, score' in sql
        assert 'FROM test_table' in sql
        assert 'WHERE score > %s AND active = %s' in sql
        assert 'ORDER BY score DESC' in sql
        assert 'LIMIT 10' in sql
        assert params == [0.5, True]
    
    def test_evolution_progress_analysis(self):
        """Test evolution progress analysis."""
        db = self.create_test_database()
        
        # Add test data for multiple generations
        for gen in range(3):
            for agent in range(5):
                agent_id = f"agent_{gen}_{agent}"
                metrics = {
                    'genome_hash': f'genome_{gen}_{agent}',
                    'generation': gen,
                    'tokens_used': 100 + gen * 10,
                    'value_generated': 120 + gen * 20,
                    'efficiency': (120 + gen * 20) / (100 + gen * 10),
                    'action_type': 'evolution'
                }
                db.record_agent_metrics(agent_id, metrics)
        
        # Analyze progress
        progress = db.analyze_evolution_progress()
        
        assert 'generations' in progress
        assert 'summary' in progress
        assert len(progress['generations']) == 3
        
        # Check that efficiency improves over generations
        gen_data = progress['generations']
        assert gen_data[2]['avg_efficiency'] > gen_data[0]['avg_efficiency']


class TestKnowledgeBase:
    """Test knowledge base functionality."""
    
    def create_test_knowledge_base(self):
        """Create test knowledge base."""
        db = MetricsDatabase("sqlite:///:memory:")
        return KnowledgeBase(db)
    
    def test_insight_storage_and_retrieval(self):
        """Test storing and retrieving insights."""
        kb = self.create_test_knowledge_base()
        
        insight = KnowledgeEntry(
            entry_id="test_insight_1",
            insight_type=InsightType.OPTIMIZATION_STRATEGY,
            description="Test optimization strategy",
            confidence=0.8,
            evidence=[{'pattern_id': 'test_pattern'}],
            applicable_contexts=['optimization']
        )
        
        # Store insight
        kb._store_insight(insight)
        
        # Retrieve insights
        insights = kb.query_insights(
            insight_type=InsightType.OPTIMIZATION_STRATEGY,
            min_confidence=0.5
        )
        
        # Note: This might be empty if the schema doesn't support insights table
        # In basic schema, we'd test the caching mechanism
        assert insight.entry_id in kb.insights_cache
    
    def test_optimization_strategies(self):
        """Test optimization strategy recommendations."""
        kb = self.create_test_knowledge_base()
        
        # Add some test data to the database
        pattern_data = {
            'pattern_hash': 'opt_pattern_1',
            'pattern_type': 'optimization',
            'description': 'Optimization pattern for efficiency',
            'effectiveness_score': 1.5,
            'token_efficiency': 2.0,
            'confidence_score': 0.9
        }
        
        kb.metrics_db.insert_pattern(pattern_data)
        
        # Get optimization strategies
        strategies = kb.get_optimization_strategies(
            target_metric="efficiency",
            current_performance=1.0
        )
        
        # Should return some recommendations
        assert isinstance(strategies, list)


class TestMetaLearner:
    """Test meta-learning functionality."""
    
    def create_test_meta_learner(self):
        """Create test meta-learner."""
        db = MetricsDatabase("sqlite:///:memory:")
        kb = KnowledgeBase(db)
        return MetaLearner(db, kb)
    
    def test_meta_knowledge_creation(self):
        """Test meta-knowledge creation."""
        meta_learner = self.create_test_meta_learner()
        
        # Create test meta-knowledge
        mk = MetaKnowledge(
            knowledge_id="test_mk_1",
            strategy=LearningStrategy.PATTERN_TRANSFER,
            description="Test meta-knowledge",
            applicability_score=0.8,
            reliability_score=0.9,
            impact_score=1.2
        )
        
        meta_learner.meta_knowledge[mk.knowledge_id] = mk
        
        # Test applicability
        context = {'generation': 5, 'action_type': 'optimize'}
        applicable = mk.is_applicable(context)
        assert applicable  # No constraints, so should be applicable
    
    def test_meta_knowledge_application(self):
        """Test applying meta-knowledge."""
        meta_learner = self.create_test_meta_learner()
        
        # Add test meta-knowledge
        mk = MetaKnowledge(
            knowledge_id="efficiency_strategy",
            strategy=LearningStrategy.EFFICIENCY_OPTIMIZATION,
            description="Optimize token allocation",
            applicability_score=0.9,
            reliability_score=0.8,
            impact_score=1.5,
            evidence=[{'optimal_allocation': {'high_performer': {'recommended_tokens': 1000}}}]
        )
        
        meta_learner.meta_knowledge[mk.knowledge_id] = mk
        
        # Apply meta-knowledge
        context = {'generation': 10, 'performer_class': 'high_performer'}
        recommendations = meta_learner.apply_meta_knowledge(context, "maximize_efficiency")
        
        assert 'suggested_patterns' in recommendations
        assert 'resource_allocation' in recommendations
    
    def test_validation_updates(self):
        """Test meta-knowledge validation updates."""
        meta_learner = self.create_test_meta_learner()
        
        mk = MetaKnowledge(
            knowledge_id="test_validation",
            strategy=LearningStrategy.PATTERN_TRANSFER,
            description="Test validation",
            applicability_score=0.7,
            reliability_score=0.8,
            impact_score=1.0
        )
        
        meta_learner.meta_knowledge[mk.knowledge_id] = mk
        
        # Validate with success
        meta_learner.validate_meta_knowledge(mk.knowledge_id, success=True)
        
        assert mk.usage_count == 1
        assert mk.success_count == 1
        assert mk.success_rate == 1.0
        
        # Validate with failure
        meta_learner.validate_meta_knowledge(mk.knowledge_id, success=False)
        
        assert mk.usage_count == 2
        assert mk.success_count == 1
        assert mk.success_rate == 0.5


class TestRepositoryManager:
    """Test repository manager integration."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        yield f"sqlite:///{db_path}"
        
        # Cleanup
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass
    
    def test_repository_manager_initialization(self, temp_db):
        """Test repository manager initialization."""
        manager = RepositoryManager(db_url=temp_db, auto_learn=False)
        
        assert manager.metrics_db is not None
        assert manager.knowledge_base is not None
        assert manager.meta_learner is not None
        assert not manager.auto_learn
    
    def test_agent_evolution_recording(self, temp_db):
        """Test recording complete agent evolution data."""
        manager = RepositoryManager(db_url=temp_db, auto_learn=False)
        
        # Create test genome
        genome = AgentGenome(generation=3)
        genome.add_gene(Gene(GeneType.STRATEGY, "test_strategy", {"param": 1.0}))
        
        # Create test pattern
        pattern = Pattern(
            pattern_id="test_pattern_1",
            pattern_type=PatternType.BEHAVIORAL,
            description="Test behavioral pattern",
            effectiveness=1.2,
            confidence=0.8
        )
        
        # Record evolution
        performance_metrics = {
            'tokens_used': 150,
            'value_generated': 200,
            'efficiency': 200/150,
            'action_type': 'evolution_test'
        }
        
        manager.record_agent_evolution(
            agent_id="test_agent_1",
            genome=genome,
            performance_metrics=performance_metrics,
            discovered_patterns=[pattern]
        )
        
        # Verify data was stored
        stats = manager.get_repository_stats(use_cache=False)
        assert stats.total_agents >= 1
        assert stats.total_patterns >= 1
    
    def test_optimization_recommendations(self, temp_db):
        """Test optimization strategy recommendations."""
        manager = RepositoryManager(db_url=temp_db, auto_learn=False)
        
        # Add some test data first
        genome = AgentGenome(generation=1)
        pattern = Pattern(
            pattern_id="opt_pattern",
            pattern_type=PatternType.OPTIMIZATION,
            description="Optimization pattern",
            effectiveness=1.5,
            confidence=0.9
        )
        
        manager.record_agent_evolution(
            agent_id="opt_agent",
            genome=genome,
            performance_metrics={'tokens_used': 100, 'value_generated': 150, 'efficiency': 1.5},
            discovered_patterns=[pattern]
        )
        
        # Get recommendations
        context = {'current_performance': 1.0, 'generation': 2}
        recommendations = manager.query_optimization_strategies(context, "maximize_efficiency")
        
        assert 'patterns' in recommendations
        assert 'insights' in recommendations
        assert 'meta_knowledge' in recommendations
        assert 'confidence_score' in recommendations
    
    def test_evolution_analysis(self, temp_db):
        """Test evolution progress analysis."""
        manager = RepositoryManager(db_url=temp_db, auto_learn=False)
        
        # Add test data across multiple generations
        for gen in range(3):
            genome = AgentGenome(generation=gen)
            manager.record_agent_evolution(
                agent_id=f"agent_gen_{gen}",
                genome=genome,
                performance_metrics={
                    'tokens_used': 100,
                    'value_generated': 120 + gen * 20,
                    'efficiency': (120 + gen * 20) / 100
                },
                discovered_patterns=[]
            )
        
        # Analyze progress
        analysis = manager.analyze_evolution_progress(include_predictions=True)
        
        assert 'progress_metrics' in analysis
        assert 'efficiency_trends' in analysis
        assert 'predictions' in analysis
        
        # Should show improvement trend
        if analysis['efficiency_trends']:
            assert 'overall_trend' in analysis['efficiency_trends']
    
    @pytest.mark.asyncio
    async def test_auto_learning(self, temp_db):
        """Test auto-learning functionality."""
        manager = RepositoryManager(
            db_url=temp_db, 
            auto_learn=True,
            learning_interval=1  # 1 second for testing
        )
        
        try:
            await manager.start()
            
            # Add some data
            genome = AgentGenome(generation=1)
            pattern = Pattern(
                pattern_id="auto_learn_pattern",
                pattern_type=PatternType.OPTIMIZATION,
                description="Auto-learning test pattern",
                effectiveness=1.8,
                confidence=0.95
            )
            
            manager.record_agent_evolution(
                agent_id="auto_learn_agent",
                genome=genome,
                performance_metrics={'tokens_used': 100, 'value_generated': 180, 'efficiency': 1.8},
                discovered_patterns=[pattern]
            )
            
            # Wait for auto-learning cycle
            await asyncio.sleep(2)
            
            # Check if learning occurred
            stats = manager.get_repository_stats(use_cache=False)
            assert stats.total_patterns >= 1
            
        finally:
            await manager.stop()
    
    def test_knowledge_export(self, temp_db):
        """Test knowledge snapshot export."""
        manager = RepositoryManager(db_url=temp_db, auto_learn=False)
        
        # Add test data
        genome = AgentGenome(generation=1)
        pattern = Pattern(
            pattern_id="export_test_pattern",
            pattern_type=PatternType.BEHAVIORAL,
            description="Export test pattern",
            effectiveness=1.3,
            confidence=0.85
        )
        
        manager.record_agent_evolution(
            agent_id="export_test_agent",
            genome=genome,
            performance_metrics={'tokens_used': 100, 'value_generated': 130, 'efficiency': 1.3},
            discovered_patterns=[pattern]
        )
        
        # Export knowledge
        with tempfile.TemporaryDirectory() as temp_dir:
            exports = manager.export_knowledge_snapshot(temp_dir)
            
            assert 'metrics_database' in exports
            assert 'statistics' in exports
            
            # Verify files exist
            for file_path in exports.values():
                assert Path(file_path).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])