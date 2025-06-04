#!/usr/bin/env python3
"""
Test the knowledge repository functionality.
"""

import sys
from pathlib import Path
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Force use of simple database
from dean.repository.simple_metrics_db import SimpleMetricsDatabase
from dean.repository.knowledge_base import KnowledgeBase
from dean.repository.meta_learner import MetaLearner
from dean.repository.repository_manager import RepositoryManager

# Monkey patch to use simple database
import dean.repository.repository_manager
dean.repository.repository_manager.MetricsDatabase = SimpleMetricsDatabase
from dean.patterns import Pattern, PatternType
from dean.diversity import AgentGenome, Gene, GeneType


def test_basic_repository_functionality():
    """Test basic repository operations."""
    print("Testing Knowledge Repository...")
    print("=" * 50)
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        # Initialize repository
        repo = RepositoryManager(f"sqlite:///{db_path}", auto_learn=False)
        print("✓ Repository manager initialized")
        
        # Create test genome
        genome = AgentGenome(generation=1)
        genome.add_gene(Gene(GeneType.STRATEGY, "test_strategy", {"approach": "greedy"}))
        genome.add_gene(Gene(GeneType.HYPERPARAMETER, "learning_rate", 0.01))
        print("✓ Test genome created")
        
        # Create test patterns
        patterns = [
            Pattern(
                pattern_id="behavioral_001",
                pattern_type=PatternType.BEHAVIORAL,
                description="Efficient exploration sequence",
                effectiveness=1.5,
                confidence=0.9,
                sequence=["explore", "analyze", "exploit"]
            ),
            Pattern(
                pattern_id="optimization_001", 
                pattern_type=PatternType.OPTIMIZATION,
                description="Token efficiency optimization",
                effectiveness=2.1,
                confidence=0.85
            )
        ]
        print("✓ Test patterns created")
        
        # Record agent evolution
        performance_metrics = {
            'tokens_used': 150,
            'value_generated': 225,
            'efficiency': 225/150,
            'action_type': 'optimization'
        }
        
        repo.record_agent_evolution(
            agent_id="test_agent_001",
            genome=genome,
            performance_metrics=performance_metrics,
            discovered_patterns=patterns
        )
        print("✓ Agent evolution recorded")
        
        # Test repository statistics
        stats = repo.get_repository_stats(use_cache=False)
        print(f"✓ Repository stats: {stats.total_agents} agents, {stats.total_patterns} patterns")
        
        # Test optimization recommendations
        context = {
            'current_performance': 1.0,
            'generation': 2,
            'objective': 'efficiency'
        }
        
        recommendations = repo.query_optimization_strategies(context, "maximize_efficiency")
        print(f"✓ Optimization recommendations generated (confidence: {recommendations['confidence_score']:.2f})")
        
        # Test evolution analysis
        analysis = repo.analyze_evolution_progress(include_predictions=True)
        print("✓ Evolution progress analysis completed")
        
        if analysis['progress_metrics'].get('generations'):
            gen_count = len(analysis['progress_metrics']['generations'])
            print(f"  - Analyzed {gen_count} generations")
        
        # Test knowledge export
        with tempfile.TemporaryDirectory() as temp_dir:
            exports = repo.export_knowledge_snapshot(temp_dir)
            print(f"✓ Knowledge exported to {len(exports)} files")
            
            for export_type, file_path in exports.items():
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    print(f"  - {export_type}: {file_size} bytes")
        
        print("\n" + "=" * 50)
        print("✅ All repository tests PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Repository test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    success = test_basic_repository_functionality()
    exit(0 if success else 1)