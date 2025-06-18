"""
Patch to add simulate method to CellularAutomataEngine
"""

import asyncio
from typing import List, Dict, Any, Optional
from indexagent.agents.evolution.cellular_automata import CellularAutomataEngine, CARule

# Add simulate method to CellularAutomataEngine
async def simulate(self, initial_state: List[int], rule: CARule, generations: int) -> Dict[str, Any]:
    """
    Simulate cellular automata evolution for the specified number of generations.
    
    This method provides compatibility with the DEAN API expectations while
    leveraging the existing apply_rule_110_complexity_generation functionality.
    
    Args:
        initial_state: Initial state of the cellular automata
        rule: The CA rule to apply (e.g., CARule.RULE_110)
        generations: Number of generations to simulate
        
    Returns:
        Dictionary containing:
        - final_state: The final state after all generations
        - complexity: Calculated complexity score
        - patterns: Any detected patterns
        - evolution_history: History of state changes
    """
    # Handle different rule types
    if rule == CARule.RULE_110:
        # Use the existing Rule 110 implementation
        result = await self.apply_rule_110_complexity_generation(
            initial_state=initial_state,
            generations=generations
        )
        
        # Extract relevant data from the comprehensive result
        return {
            "final_state": result.get("final_state", initial_state),
            "complexity": result.get("complexity_metrics", {}).get("shannon_entropy", 0.0),
            "patterns": result.get("detected_patterns", []),
            "evolution_history": result.get("evolution_grid", [])
        }
    else:
        # For other rules, use the general apply_rule method
        current_state = initial_state.copy()
        evolution_history = [current_state.copy()]
        
        for gen in range(generations):
            # Apply the rule (this method expects agent states but we can adapt)
            result = await self.apply_rule(
                rule=rule,
                agent_states=current_state,
                generations=1  # Apply one generation at a time
            )
            
            # Extract the new state from the result
            if "final_state" in result:
                current_state = result["final_state"]
            elif "states" in result:
                current_state = result["states"]
            else:
                # If format is unexpected, maintain current state
                pass
                
            evolution_history.append(current_state.copy())
        
        # Calculate complexity of the final state
        complexity = await self._calculate_shannon_entropy(current_state)
        
        return {
            "final_state": current_state,
            "complexity": complexity,
            "patterns": [],  # Pattern detection would need to be implemented for other rules
            "evolution_history": evolution_history
        }

# Monkey patch the method onto the class
CellularAutomataEngine.simulate = simulate

# Also create a synchronous wrapper for the API
def simulate_sync(engine: CellularAutomataEngine, initial_state: List[int], rule: CARule, generations: int) -> Dict[str, Any]:
    """Synchronous wrapper for the simulate method."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(engine.simulate(initial_state, rule, generations))
    finally:
        loop.close()