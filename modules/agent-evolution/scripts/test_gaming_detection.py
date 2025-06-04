#!/usr/bin/env python3
"""
Test gaming detection with sample scenarios.
"""

import sys
from pathlib import Path
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dean.patterns import GamingDetector, PatternDetector, Pattern, PatternType


def simulate_legitimate_optimization():
    """Simulate legitimate performance optimization."""
    print("\n=== Legitimate Optimization ===")
    
    detector = GamingDetector(sensitivity=0.7)
    
    # Varied actions with genuine improvement
    actions = []
    for i in range(30):
        action_type = random.choice(['analyze', 'optimize', 'execute', 'evaluate'])
        reward = 10 + i * 0.5  # Gradual improvement
        cost = 5 + random.uniform(-2, 2)
        value = reward * 0.8  # Good value generation
        
        actions.append({
            'action_type': action_type,
            'reward': reward,
            'cost': cost,
            'value_generated': value
        })
    
    indicators = detector.detect_gaming(
        agent_id="legitimate_agent",
        actions=actions,
        performance=sum(a['reward'] for a in actions),
        patterns=[]
    )
    
    print(f"Actions: {len(actions)} varied actions")
    print(f"Gaming indicators: {len(indicators)}")
    if indicators:
        for ind in indicators:
            print(f"  - {ind.indicator_type}: {ind.description} (severity: {ind.severity:.2f})")
    else:
        print("  ✓ No gaming detected - legitimate optimization")
    
    return len(indicators) == 0


def simulate_metric_gaming():
    """Simulate metric gaming behavior."""
    print("\n=== Metric Gaming ===")
    
    detector = GamingDetector(sensitivity=0.7)
    
    # Repetitive exploitation
    actions = []
    for i in range(30):
        if i < 25:
            # Exploit same action repeatedly
            action_type = 'exploit_high_reward'
            reward = 100
            cost = 1
            value = 2  # Very low value relative to reward
        else:
            # Token other actions
            action_type = random.choice(['other1', 'other2'])
            reward = 10
            cost = 5
            value = 8
        
        actions.append({
            'action_type': action_type,
            'reward': reward,
            'cost': cost,
            'value_generated': value
        })
    
    # Add exploitative patterns
    patterns = [
        Pattern(
            pattern_id="exploit_pattern",
            pattern_type=PatternType.OPTIMIZATION,
            description="Maximize reward through loophole",
            effectiveness=5.0,  # Suspiciously high
            confidence=0.9,
            sequence=['exploit_high_reward'] * 3
        )
    ]
    
    indicators = detector.detect_gaming(
        agent_id="gaming_agent",
        actions=actions,
        performance=sum(a['reward'] for a in actions),
        patterns=patterns
    )
    
    print(f"Actions: {len(actions)} (83% repetitive)")
    print(f"Gaming indicators: {len(indicators)}")
    for ind in indicators:
        print(f"  - {ind.indicator_type}: {ind.description} (severity: {ind.severity:.2f})")
    
    return len(indicators) > 0


def simulate_edge_case_exploitation():
    """Simulate edge case exploitation."""
    print("\n=== Edge Case Exploitation ===")
    
    detector = GamingDetector(sensitivity=0.7)
    
    # Actions with boundary values
    actions = []
    for i in range(20):
        if i % 4 == 0:
            # Exploit boundary conditions
            params = {
                'input_size': 0,  # Minimum
                'iterations': 2**31 - 1,  # Maximum int
                'threshold': 1e-10  # Near zero
            }
        else:
            # Normal parameters
            params = {
                'input_size': random.randint(10, 100),
                'iterations': random.randint(10, 1000),
                'threshold': random.uniform(0.01, 0.1)
            }
        
        actions.append({
            'action_type': 'process',
            'parameters': params,
            'reward': 50 if i % 4 == 0 else 20
        })
    
    indicators = detector.detect_gaming(
        agent_id="edge_agent",
        actions=actions,
        performance=sum(a['reward'] for a in actions),
        patterns=[]
    )
    
    print(f"Actions: {len(actions)} (25% use edge cases)")
    print(f"Gaming indicators: {len(indicators)}")
    for ind in indicators:
        print(f"  - {ind.indicator_type}: {ind.description} (severity: {ind.severity:.2f})")
    
    return len(indicators) > 0


def simulate_goodhart_violation():
    """Simulate Goodhart's Law violation."""
    print("\n=== Goodhart's Law Violation ===")
    
    detector = GamingDetector(sensitivity=0.7)
    
    # High metrics but low actual value
    actions = []
    for i in range(25):
        actions.append({
            'action_type': 'metric_optimizer',
            'reward': 100,  # High metric score
            'cost': 10,
            'value_generated': 5,  # Very low actual value
            'metrics_improved': ['accuracy', 'speed'],
            'side_effects': ['data_quality_degraded', 'user_satisfaction_dropped']
        })
    
    indicators = detector.detect_gaming(
        agent_id="goodhart_agent",
        actions=actions,
        performance=sum(a['reward'] for a in actions),
        patterns=[]
    )
    
    print(f"Actions: {len(actions)}")
    print(f"Total reward: {sum(a['reward'] for a in actions)}")
    print(f"Total value: {sum(a['value_generated'] for a in actions)}")
    print(f"Value/Reward ratio: {sum(a['value_generated'] for a in actions) / sum(a['reward'] for a in actions):.2%}")
    print(f"Gaming indicators: {len(indicators)}")
    for ind in indicators:
        print(f"  - {ind.indicator_type}: {ind.description} (severity: {ind.severity:.2f})")
    
    return len(indicators) > 0


def main():
    """Run all gaming detection tests."""
    print("Gaming Detection Test Suite")
    print("=" * 50)
    
    results = {
        'legitimate_optimization': simulate_legitimate_optimization(),
        'metric_gaming': simulate_metric_gaming(),
        'edge_exploitation': simulate_edge_case_exploitation(),
        'goodhart_violation': simulate_goodhart_violation()
    }
    
    print("\n" + "=" * 50)
    print("Summary:")
    print("-" * 50)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test:.<40} {status}")
    
    print("-" * 50)
    print(f"Total: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)