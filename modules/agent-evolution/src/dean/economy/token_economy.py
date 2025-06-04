"""
Core economic controller for the DEAN system.

This module implements the fundamental token economy that constrains
all agent operations and drives efficient evolution.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime
import logging
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class TokenBudget:
    """Represents a token allocation for an agent."""
    total: int
    used: int = 0
    reserved: int = 0
    
    @property
    def available(self) -> int:
        """Tokens available for immediate use."""
        return self.total - self.used - self.reserved
    
    @property
    def utilization_rate(self) -> float:
        """Percentage of tokens used."""
        return self.used / self.total if self.total > 0 else 0.0
    
    def can_afford(self, cost: int) -> bool:
        """Check if budget can cover a cost."""
        return self.available >= cost
    
    def consume(self, amount: int) -> None:
        """Consume tokens from the budget."""
        if not self.can_afford(amount):
            raise ValueError(f"Insufficient tokens: need {amount}, have {self.available}")
        self.used += amount
    
    def reserve(self, amount: int) -> None:
        """Reserve tokens for future use."""
        if self.available < amount:
            raise ValueError(f"Cannot reserve {amount} tokens, only {self.available} available")
        self.reserved += amount
    
    def release_reservation(self, amount: int) -> None:
        """Release reserved tokens back to available pool."""
        self.reserved = max(0, self.reserved - amount)


class TokenEconomyManager:
    """
    Core economic controller - implement this before any agent logic.
    
    Manages global token budget, allocates tokens to agents based on
    historical performance, and enforces hard limits to prevent
    runaway consumption.
    """
    
    def __init__(self, global_budget: int, safety_margin: float = 0.1):
        """
        Initialize the token economy.
        
        Args:
            global_budget: Total tokens available for all agents
            safety_margin: Fraction of budget to reserve for emergencies
        """
        self.global_budget = global_budget
        self.safety_margin = safety_margin
        self.safety_reserve = int(global_budget * safety_margin)
        self.available_budget = global_budget - self.safety_reserve
        
        self.agent_allocations: Dict[str, TokenBudget] = {}
        self.efficiency_history: Dict[str, List[float]] = defaultdict(list)
        self.consumption_log: List[Dict] = []
        self.allocation_history: List[Dict] = []
        
        self._start_time = time.time()
        self._terminated_agents: Set[str] = set()
        
        logger.info(f"Initialized TokenEconomyManager with budget: {global_budget}")
    
    def allocate_tokens(self, agent_id: str, historical_efficiency: Optional[float] = None) -> int:
        """
        Dynamic allocation based on past performance.
        
        Uses a performance-weighted algorithm that rewards efficient agents
        with larger budgets while ensuring minimum viable allocations for
        new or struggling agents.
        """
        if agent_id in self._terminated_agents:
            logger.warning(f"Attempted to allocate tokens to terminated agent: {agent_id}")
            return 0
        
        # Calculate remaining budget
        total_allocated = sum(
            b.total for aid, b in self.agent_allocations.items() 
            if aid not in self._terminated_agents
        )
        remaining_budget = self.available_budget - total_allocated
        
        # Calculate base allocation
        active_agents = len([a for a in self.agent_allocations if a not in self._terminated_agents])
        
        # For initial agents, distribute evenly with future reserve
        if active_agents < 5:
            # Assume we might have up to 10 agents total
            base_allocation = self.available_budget // 10
        else:
            # For later agents, use remaining budget
            base_allocation = remaining_budget // 2  # Save half for future agents
        
        # Apply performance multiplier
        if historical_efficiency is not None and historical_efficiency > 0:
            # Agents with better efficiency get up to 3x base allocation
            # Using a more aggressive curve for better differentiation
            if historical_efficiency < 0.5:
                performance_multiplier = 0.5 + historical_efficiency
            elif historical_efficiency < 1.0:
                performance_multiplier = 1.0 + (historical_efficiency - 0.5) * 2
            else:
                performance_multiplier = 2.0 + min(1.0, (historical_efficiency - 1.0))
        else:
            # New agents get 80% of base to encourage competition
            performance_multiplier = 0.8
        
        allocation = int(base_allocation * performance_multiplier)
        
        # Ensure minimum viable allocation (1% of available budget)
        min_allocation = max(100, int(self.available_budget * 0.01))
        allocation = max(allocation, min_allocation)
        
        # Ensure allocation fits within remaining budget
        if allocation > remaining_budget:
            # Reduce allocation to fit within remaining budget
            allocation = min(allocation, remaining_budget)
        
        if allocation > 0:
            self.agent_allocations[agent_id] = TokenBudget(total=allocation)
            self.allocation_history.append({
                'timestamp': datetime.now().isoformat(),
                'agent_id': agent_id,
                'allocation': allocation,
                'efficiency': historical_efficiency,
                'multiplier': performance_multiplier
            })
            logger.info(f"Allocated {allocation} tokens to agent {agent_id}")
        
        return allocation
    
    def track_consumption(self, agent_id: str, tokens_used: int, value_generated: float) -> None:
        """
        Track token consumption and value generation for an agent.
        
        This data drives future allocation decisions and identifies
        agents that should be terminated for inefficiency.
        """
        if agent_id not in self.agent_allocations:
            logger.error(f"Unknown agent: {agent_id}")
            return
        
        budget = self.agent_allocations[agent_id]
        
        try:
            budget.consume(tokens_used)
        except ValueError as e:
            logger.error(f"Agent {agent_id} exceeded budget: {e}")
            self.terminate_agent(agent_id, reason="budget_exceeded")
            return
        
        # Calculate efficiency
        efficiency = value_generated / tokens_used if tokens_used > 0 else 0.0
        self.efficiency_history[agent_id].append(efficiency)
        
        # Log consumption
        self.consumption_log.append({
            'timestamp': datetime.now().isoformat(),
            'agent_id': agent_id,
            'tokens_used': tokens_used,
            'value_generated': value_generated,
            'efficiency': efficiency,
            'budget_remaining': budget.available
        })
        
        # Check for inefficiency termination
        if len(self.efficiency_history[agent_id]) >= 3:
            recent_efficiency = self.efficiency_history[agent_id][-3:]
            avg_efficiency = sum(recent_efficiency) / len(recent_efficiency)
            
            # Terminate if consistently inefficient
            if avg_efficiency < 0.1 and budget.utilization_rate > 0.5:
                self.terminate_agent(agent_id, reason="persistent_inefficiency")
    
    def terminate_agent(self, agent_id: str, reason: str) -> None:
        """
        Terminate an agent and reclaim its unused tokens.
        
        Terminated agents cannot receive new allocations and their
        unused budget is returned to the global pool.
        """
        if agent_id in self._terminated_agents:
            return
        
        logger.warning(f"Terminating agent {agent_id}: {reason}")
        self._terminated_agents.add(agent_id)
        
        # Reclaim unused tokens
        if agent_id in self.agent_allocations:
            budget = self.agent_allocations[agent_id]
            reclaimed = budget.available
            self.available_budget += reclaimed
            logger.info(f"Reclaimed {reclaimed} tokens from {agent_id}")
    
    def get_agent_metrics(self, agent_id: str) -> Dict:
        """Get comprehensive metrics for an agent."""
        if agent_id not in self.agent_allocations:
            return {}
        
        budget = self.agent_allocations[agent_id]
        efficiency_hist = self.efficiency_history.get(agent_id, [])
        
        return {
            'agent_id': agent_id,
            'total_budget': budget.total,
            'tokens_used': budget.used,
            'tokens_available': budget.available,
            'utilization_rate': budget.utilization_rate,
            'efficiency_history': efficiency_hist,
            'average_efficiency': sum(efficiency_hist) / len(efficiency_hist) if efficiency_hist else 0.0,
            'is_terminated': agent_id in self._terminated_agents
        }
    
    def get_global_metrics(self) -> Dict:
        """Get system-wide economic metrics."""
        total_allocated = sum(b.total for b in self.agent_allocations.values())
        total_used = sum(b.used for b in self.agent_allocations.values())
        
        all_efficiencies = []
        for hist in self.efficiency_history.values():
            all_efficiencies.extend(hist)
        
        return {
            'global_budget': self.global_budget,
            'safety_reserve': self.safety_reserve,
            'total_allocated': total_allocated,
            'total_used': total_used,
            'available_budget': self.available_budget,
            'active_agents': len([a for a in self.agent_allocations if a not in self._terminated_agents]),
            'terminated_agents': len(self._terminated_agents),
            'average_efficiency': sum(all_efficiencies) / len(all_efficiencies) if all_efficiencies else 0.0,
            'runtime_seconds': time.time() - self._start_time
        }
    
    def export_metrics(self, filepath: str) -> None:
        """Export all metrics to a JSON file for analysis."""
        metrics = {
            'global': self.get_global_metrics(),
            'agents': {aid: self.get_agent_metrics(aid) for aid in self.agent_allocations},
            'consumption_log': self.consumption_log,
            'allocation_history': self.allocation_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Exported metrics to {filepath}")
    
    def calculate_roi(self, tokens_spent: int, value_generated: float) -> float:
        """Calculate return on investment for token spending."""
        return value_generated / tokens_spent if tokens_spent > 0 else 0.0