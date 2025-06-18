#!/usr/bin/env python3
"""
Test script for CellularAutomataEngine simulate method
"""

import asyncio
from indexagent.agents.evolution.cellular_automata import CellularAutomataEngine, CARule

async def test_ca_simulate():
    """Test the simulate method implementation"""
    
    print("Testing CellularAutomataEngine simulate method...")
    
    # Initialize engine
    engine = CellularAutomataEngine()
    
    # Test 1: Rule 110 with small initial state
    print("\nTest 1: Rule 110 simulation")
    initial_state = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
    result = await engine.simulate(
        initial_state=initial_state,
        rule=CARule.RULE_110,
        generations=5
    )
    
    print(f"Initial state: {initial_state}")
    print(f"Final state: {result['final_state'][:12]}...")  # Show first 12 cells
    print(f"Complexity: {result['complexity']:.4f}")
    print(f"Patterns found: {len(result.get('patterns', []))}")
    print(f"Evolution history length: {len(result.get('evolution_history', []))}")
    
    # Test 2: Rule 30 (if available)
    print("\nTest 2: Rule 30 simulation")
    try:
        result2 = await engine.simulate(
            initial_state=[0]*20 + [1] + [0]*20,
            rule=CARule.RULE_30,
            generations=10
        )
        print(f"Rule 30 complexity: {result2['complexity']:.4f}")
    except Exception as e:
        print(f"Rule 30 test failed (expected if not implemented): {e}")
    
    # Test 3: Deterministic results
    print("\nTest 3: Deterministic results check")
    result3a = await engine.simulate(
        initial_state=[1, 0, 1, 0, 1],
        rule=CARule.RULE_110,
        generations=3
    )
    result3b = await engine.simulate(
        initial_state=[1, 0, 1, 0, 1],
        rule=CARule.RULE_110,
        generations=3
    )
    
    deterministic = result3a['final_state'] == result3b['final_state']
    print(f"Deterministic results: {deterministic}")
    
    print("\nAll tests completed!")
    return True

if __name__ == "__main__":
    # Add current directory to path for imports
    import sys
    sys.path.append('/app')
    sys.path.append('/app/IndexAgent')
    
    # Apply the patch
    from ca_simulate_patch import simulate
    CellularAutomataEngine.simulate = simulate
    
    # Run tests
    asyncio.run(test_ca_simulate())