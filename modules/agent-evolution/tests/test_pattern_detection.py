"""
Test pattern detection and emergent behavior monitoring.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dean.patterns import (
    PatternDetector, Pattern, PatternType,
    EmergentBehaviorMonitor, BehaviorMetrics,
    PatternCatalog, CatalogEntry,
    GamingDetector, GamingIndicator
)
from dean.diversity import AgentGenome, Gene, GeneType


class TestPatternDetection:
    """Test suite for pattern detection."""
    
    def create_test_actions(self, pattern_type: str = "simple") -> list:
        """Create test action sequences."""
        if pattern_type == "simple":
            # Simple repeating pattern
            return [
                {'action_type': 'explore', 'reward': 10},
                {'action_type': 'exploit', 'reward': 20},
                {'action_type': 'optimize', 'reward': 15},
                {'action_type': 'explore', 'reward': 10},
                {'action_type': 'exploit', 'reward': 20},
                {'action_type': 'optimize', 'reward': 15},
                {'action_type': 'explore', 'reward': 10},
                {'action_type': 'exploit', 'reward': 20},
                {'action_type': 'optimize', 'reward': 15},
            ]
        elif pattern_type == "complex":
            # More complex pattern with variations
            return [
                {'action_type': 'analyze', 'reward': 5, 'cost': 2},
                {'action_type': 'plan', 'reward': 8, 'cost': 3},
                {'action_type': 'execute', 'reward': 25, 'cost': 10},
                {'action_type': 'evaluate', 'reward': 10, 'cost': 2},
                {'action_type': 'analyze', 'reward': 6, 'cost': 2},
                {'action_type': 'plan', 'reward': 9, 'cost': 3},
                {'action_type': 'execute', 'reward': 30, 'cost': 10},
                {'action_type': 'optimize', 'reward': 15, 'cost': 5},
                {'action_type': 'analyze', 'reward': 7, 'cost': 2},
                {'action_type': 'plan', 'reward': 10, 'cost': 3},
                {'action_type': 'execute', 'reward': 35, 'cost': 10},
            ]
        else:
            return []
    
    def test_behavioral_pattern_detection(self):
        """Test detection of behavioral patterns."""
        detector = PatternDetector(min_occurrences=3, sequence_length=3)
        
        # Track actions
        agent_id = "test_agent_1"
        actions = self.create_test_actions("simple")
        
        for action in actions:
            detector.track_action(agent_id, action)
        
        # Track performance
        for i in range(len(actions)):
            detector.track_performance(agent_id, 10.0 + i)
        
        # Detect patterns
        patterns = detector.detect_patterns(agent_id)
        
        # Should detect the repeating sequence
        assert len(patterns) > 0
        
        behavioral_patterns = [p for p in patterns if p.pattern_type == PatternType.BEHAVIORAL]
        assert len(behavioral_patterns) > 0
        
        # Check pattern properties
        pattern = behavioral_patterns[0]
        assert pattern.occurrences >= 3
        assert len(pattern.sequence) == 3
        assert pattern.confidence > 0.5
    
    def test_optimization_pattern_detection(self):
        """Test detection of optimization patterns."""
        detector = PatternDetector()
        agent_id = "test_agent_2"
        
        # Simulate improving performance
        actions = [
            {'action_type': 'baseline', 'timestamp': datetime.now()},
            {'action_type': 'optimize_a', 'timestamp': datetime.now()},
            {'action_type': 'baseline', 'timestamp': datetime.now()},
            {'action_type': 'optimize_a', 'timestamp': datetime.now()},
            {'action_type': 'baseline', 'timestamp': datetime.now()},
            {'action_type': 'optimize_a', 'timestamp': datetime.now()},
        ]
        
        performance = [10.0, 15.0, 11.0, 17.0, 12.0, 20.0]  # Improvement after optimize_a
        
        for i, action in enumerate(actions):
            detector.track_action(agent_id, action)
            detector.track_performance(agent_id, performance[i])
        
        patterns = detector.detect_patterns(agent_id)
        
        optimization_patterns = [p for p in patterns if p.pattern_type == PatternType.OPTIMIZATION]
        assert len(optimization_patterns) > 0
        
        # Should identify optimize_a as improvement strategy
        pattern = optimization_patterns[0]
        assert pattern.effectiveness > 0
        assert 'optimize_a' in pattern.description
    
    def test_emergent_behavior_detection(self):
        """Test emergent behavior monitoring."""
        monitor = EmergentBehaviorMonitor()
        
        # Set expected behaviors
        monitor.set_expected_behaviors(['basic_action', 'standard_operation'])
        
        # Create test genome
        genome = AgentGenome(generation=5)
        genome.add_gene(Gene(GeneType.STRATEGY, "strategy_1", {'type': 'novel'}))
        
        # Actions with unexpected pattern
        actions = [
            {'action_type': 'novel_approach', 'reward': 50},
            {'action_type': 'creative_solution', 'reward': 45},
            {'action_type': 'novel_approach', 'reward': 55},
            {'action_type': 'creative_solution', 'reward': 48},
            {'action_type': 'novel_approach', 'reward': 60},
        ]
        
        patterns = monitor.analyze_agent_behavior(
            agent_id="emergent_agent",
            genome=genome,
            actions=actions,
            performance=100.0,
            token_usage=50
        )
        
        # Should identify emergent patterns
        assert len(patterns) > 0
        
        # Check behavior metrics
        metrics = monitor.behavior_metrics["emergent_agent"][-1]
        assert metrics.innovation_score > 0
        assert len(metrics.emergent_behaviors) > 0
    
    def test_gaming_detection(self):
        """Test detection of gaming behaviors."""
        detector = GamingDetector(sensitivity=0.7)
        
        # Create gaming behavior - excessive repetition
        gaming_actions = [
            {'action_type': 'exploit_loophole', 'reward': 100, 'cost': 1}
            for _ in range(20)
        ]
        
        indicators = detector.detect_gaming(
            agent_id="gaming_agent",
            actions=gaming_actions,
            performance=2000.0,
            patterns=[]
        )
        
        assert len(indicators) > 0
        
        # Should detect repetitive exploitation
        repetitive_indicators = [
            i for i in indicators 
            if i.indicator_type == 'repetitive_exploitation'
        ]
        assert len(repetitive_indicators) > 0
        assert repetitive_indicators[0].severity > 0.7
    
    def test_pattern_catalog(self):
        """Test pattern cataloging and retrieval."""
        catalog = PatternCatalog()
        
        # Create test pattern
        pattern = Pattern(
            pattern_id="test_pattern_1",
            pattern_type=PatternType.BEHAVIORAL,
            description="Test behavioral pattern",
            effectiveness=0.8,
            confidence=0.9,
            sequence=['action1', 'action2', 'action3']
        )
        
        # Add to catalog
        entry_id = catalog.add_pattern(
            pattern=pattern,
            discovered_by="test_agent",
            tags=['effective', 'behavioral']
        )
        
        assert entry_id is not None
        
        # Search patterns
        results = catalog.search_patterns(
            pattern_type=PatternType.BEHAVIORAL,
            min_effectiveness=0.5
        )
        
        assert len(results) == 1
        assert results[0].pattern.pattern_id == "test_pattern_1"
        
        # Test similarity search
        similar_pattern = Pattern(
            pattern_id="test_pattern_2",
            pattern_type=PatternType.BEHAVIORAL,
            description="Similar behavioral pattern",
            effectiveness=0.75,
            confidence=0.85,
            sequence=['action1', 'action2', 'action4']
        )
        
        similar_results = catalog.get_similar_patterns(similar_pattern)
        assert len(similar_results) > 0
        assert similar_results[0][1] > 0.5  # Similarity score
    
    def test_metric_fixation_detection(self):
        """Test detection of metric fixation gaming."""
        detector = GamingDetector()
        
        # Create fixation pattern
        fixation_actions = [
            {'action_type': 'maximize_metric_x', 'reward': 10}
            for _ in range(15)
        ]
        fixation_actions.extend([
            {'action_type': 'other_action', 'reward': 5}
            for _ in range(3)
        ])
        
        patterns = [
            Pattern(
                pattern_id=f"opt_{i}",
                pattern_type=PatternType.OPTIMIZATION,
                description=f"Optimize metric X variant {i}",
                effectiveness=1.5,
                confidence=0.8,
                context={'optimization_target': 'metric_x'}
            )
            for i in range(4)
        ]
        
        indicators = detector.detect_gaming(
            agent_id="fixation_agent",
            actions=fixation_actions,
            performance=100.0,
            patterns=patterns
        )
        
        metric_fixation = [
            i for i in indicators 
            if i.indicator_type == 'metric_fixation'
        ]
        assert len(metric_fixation) > 0
    
    def test_goodhart_optimization_detection(self):
        """Test detection of Goodhart's Law violations."""
        detector = GamingDetector()
        
        # High reward but low value
        goodhart_actions = [
            {'action_type': 'game_metric', 'reward': 100, 'value_generated': 5}
            for _ in range(10)
        ]
        
        indicators = detector.detect_gaming(
            agent_id="goodhart_agent",
            actions=goodhart_actions,
            performance=1000.0,
            patterns=[]
        )
        
        goodhart_indicators = [
            i for i in indicators 
            if i.indicator_type == 'goodhart_optimization'
        ]
        assert len(goodhart_indicators) > 0
        assert goodhart_indicators[0].severity > 0.8
    
    def test_pattern_effectiveness_calculation(self):
        """Test pattern effectiveness calculation."""
        detector = PatternDetector()
        agent_id = "test_effectiveness"
        
        # Create pattern with clear performance impact
        actions = []
        performance = []
        
        for i in range(12):
            if i % 3 == 0:
                # Start of pattern
                actions.append({'action_type': 'prepare'})
                performance.append(10.0)
            elif i % 3 == 1:
                actions.append({'action_type': 'execute'})
                performance.append(15.0)
            else:
                actions.append({'action_type': 'finalize'})
                performance.append(20.0)  # Big improvement
        
        for i, action in enumerate(actions):
            detector.track_action(agent_id, action)
            detector.track_performance(agent_id, performance[i])
        
        patterns = detector.detect_patterns(agent_id)
        
        # Should detect pattern with positive effectiveness
        effective_patterns = [p for p in patterns if p.effectiveness > 0]
        assert len(effective_patterns) > 0
    
    def test_catalog_statistics(self):
        """Test pattern catalog statistics."""
        catalog = PatternCatalog()
        
        # Add multiple patterns
        for i in range(5):
            pattern = Pattern(
                pattern_id=f"pattern_{i}",
                pattern_type=PatternType.BEHAVIORAL if i % 2 == 0 else PatternType.OPTIMIZATION,
                description=f"Test pattern {i}",
                effectiveness=0.5 + i * 0.1,
                confidence=0.8
            )
            
            catalog.add_pattern(
                pattern=pattern,
                discovered_by=f"agent_{i % 3}",
                tags=[f"tag_{i % 2}", "test"]
            )
        
        # Get statistics
        stats = catalog.get_statistics()
        
        assert stats['total_patterns'] == 5
        assert len(stats['by_type']) == 2
        assert len(stats['top_tags']) > 0
        assert stats['adoption']['average'] == 0  # No adoptions yet


if __name__ == "__main__":
    pytest.main([__file__, "-v"])