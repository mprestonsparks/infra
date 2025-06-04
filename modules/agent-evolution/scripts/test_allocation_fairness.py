#!/usr/bin/env python3
"""
Test allocation fairness across agents with different performance levels.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dean.economy import TokenEconomyManager


def test_allocation_fairness():
    """Test that allocation algorithm is fair and performance-based."""
    
    # Initialize manager with 100k tokens
    manager = TokenEconomyManager(global_budget=100000)
    
    # Simulate agents with different performance levels
    agents = [
        ("agent_poor", 0.1),      # Very low efficiency
        ("agent_avg", 0.5),       # Average efficiency
        ("agent_good", 1.2),      # Good efficiency
        ("agent_excellent", 2.0), # Excellent efficiency
        ("agent_new", None),      # New agent, no history
    ]
    
    allocations = {}
    for agent_id, efficiency in agents:
        allocation = manager.allocate_tokens(agent_id, efficiency)
        allocations[agent_id] = allocation
        print(f"{agent_id}: {allocation} tokens (efficiency: {efficiency})")
    
    # Verify allocations
    print("\nAllocation Analysis:")
    print("-" * 40)
    
    # New agents should get reasonable allocation
    assert allocations["agent_new"] > 0
    print(f"✓ New agent allocation: {allocations['agent_new']}")
    
    # Better performers should get more
    assert allocations["agent_excellent"] > allocations["agent_good"]
    assert allocations["agent_good"] > allocations["agent_avg"]
    assert allocations["agent_avg"] > allocations["agent_poor"]
    print("✓ Performance-based allocation ordering verified")
    
    # No agent should get everything
    total_allocated = sum(allocations.values())
    max_allocation = max(allocations.values())
    assert max_allocation < total_allocated * 0.5
    print(f"✓ Maximum allocation: {max_allocation/total_allocated:.1%} of total")
    
    # All agents should get minimum viable amount
    min_viable = manager.available_budget * 0.01
    for agent_id, allocation in allocations.items():
        assert allocation >= min_viable
    print(f"✓ All agents got minimum viable allocation (>= {min_viable})")
    
    # Total shouldn't exceed available budget
    assert total_allocated <= manager.available_budget
    print(f"✓ Total allocated ({total_allocated}) within budget ({manager.available_budget})")
    
    print("\nAllocation fairness test PASSED!")
    return True


if __name__ == "__main__":
    success = test_allocation_fairness()
    exit(0 if success else 1)