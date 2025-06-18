#!/usr/bin/env python3
"""
Phase 3: Token Economy System
Implements real economic constraints that affect agent behavior.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from sqlalchemy.orm import Session
from sqlalchemy import text


class TokenTransactionType(Enum):
    """Types of token transactions."""
    ALLOCATION = "allocation"          # Initial token grant
    EVOLUTION_COST = "evolution_cost"  # Cost of running evolution
    ANALYSIS_COST = "analysis_cost"    # Cost of code analysis
    IMPLEMENTATION = "implementation"   # Cost of implementing changes
    REWARD = "reward"                  # Reward for successful patterns
    PENALTY = "penalty"                # Penalty for failed attempts
    TRANSFER = "transfer"              # Transfer between agents


@dataclass
class TokenTransaction:
    """Represents a token transaction."""
    agent_id: str
    transaction_type: TokenTransactionType
    amount: int
    description: str
    metadata: Dict[str, any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class TokenEconomyEngine:
    """
    Real token economy that creates pressure for efficiency.
    Agents must balance exploration with conservation.
    """
    
    def __init__(self, db_session: Session, global_budget: int = 1000000):
        self.db = db_session
        self.global_budget = global_budget
        self.min_operation_cost = 50  # Minimum tokens for any operation
        self.scarcity_multiplier = 1.0  # Increases as tokens become scarce
        
    def get_agent_balance(self, agent_id: str) -> Tuple[int, int, int]:
        """
        Get agent's token balance.
        Returns: (allocated, consumed, remaining)
        """
        result = self.db.execute(text("""
            SELECT token_budget, token_consumed 
            FROM agent_evolution.agents 
            WHERE id = :id
        """), {"id": agent_id}).fetchone()
        
        if not result:
            return (0, 0, 0)
            
        allocated = result.token_budget
        consumed = result.token_consumed
        remaining = allocated - consumed
        
        return (allocated, consumed, remaining)
    
    def get_global_stats(self) -> Dict[str, any]:
        """Get global token economy statistics."""
        result = self.db.execute(text("""
            SELECT 
                COUNT(*) as agent_count,
                SUM(token_budget) as total_allocated,
                SUM(token_consumed) as total_consumed,
                AVG(token_efficiency) as avg_efficiency
            FROM agent_evolution.agents
            WHERE status = 'active'
        """)).fetchone()
        
        if not result:
            return {
                'agent_count': 0,
                'total_allocated': 0,
                'total_consumed': 0,
                'remaining_budget': self.global_budget,
                'scarcity_level': 0.0
            }
            
        total_allocated = result.total_allocated or 0
        total_consumed = result.total_consumed or 0
        remaining_budget = self.global_budget - total_allocated
        
        # Calculate scarcity level (0.0 = abundant, 1.0 = scarce)
        scarcity_level = 1.0 - (remaining_budget / self.global_budget)
        
        # Update scarcity multiplier
        self.scarcity_multiplier = 1.0 + (scarcity_level * 2.0)  # Up to 3x cost at full scarcity
        
        return {
            'agent_count': result.agent_count,
            'total_allocated': total_allocated,
            'total_consumed': total_consumed,
            'remaining_budget': remaining_budget,
            'avg_efficiency': float(result.avg_efficiency or 0.0),
            'scarcity_level': scarcity_level,
            'scarcity_multiplier': self.scarcity_multiplier
        }
    
    def calculate_operation_cost(self, operation_type: str, complexity: int = 1) -> int:
        """
        Calculate token cost for an operation.
        Costs increase with scarcity.
        """
        base_costs = {
            'code_analysis': 100,
            'ca_evolution': 50,
            'pattern_detection': 75,
            'todo_implementation': 500,
            'refactoring': 300,
            'cleanup': 100,
            'complexity_reduction': 400,
            'testing': 200
        }
        
        base_cost = base_costs.get(operation_type, 100)
        
        # Apply complexity multiplier
        cost = base_cost * complexity
        
        # Apply scarcity multiplier
        cost = int(cost * self.scarcity_multiplier)
        
        # Ensure minimum cost
        return max(cost, self.min_operation_cost)
    
    def can_afford_operation(self, agent_id: str, operation_type: str, 
                           complexity: int = 1) -> Tuple[bool, int, str]:
        """
        Check if agent can afford an operation.
        Returns: (can_afford, cost, reason)
        """
        _, _, remaining = self.get_agent_balance(agent_id)
        cost = self.calculate_operation_cost(operation_type, complexity)
        
        if remaining < cost:
            return (False, cost, f"Insufficient tokens: need {cost}, have {remaining}")
            
        # Check if operation would leave agent with too few tokens
        if remaining - cost < self.min_operation_cost * 2:
            return (False, cost, "Would leave insufficient tokens for future operations")
            
        return (True, cost, "OK")
    
    def charge_tokens(self, agent_id: str, amount: int, 
                     transaction_type: TokenTransactionType,
                     description: str, metadata: Dict = None) -> bool:
        """
        Charge tokens from an agent.
        Returns True if successful, False if insufficient funds.
        """
        _, _, remaining = self.get_agent_balance(agent_id)
        
        if remaining < amount:
            return False
            
        # Update agent's consumed tokens
        self.db.execute(text("""
            UPDATE agent_evolution.agents
            SET token_consumed = token_consumed + :amount,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = :id
        """), {"id": agent_id, "amount": amount})
        
        # Record transaction
        self._record_transaction(
            agent_id, transaction_type, -amount, description, metadata or {}
        )
        
        # Update token efficiency
        self._update_efficiency(agent_id)
        
        self.db.commit()
        return True
    
    def reward_tokens(self, agent_id: str, amount: int, 
                     description: str, metadata: Dict = None) -> bool:
        """
        Reward tokens to an agent for successful patterns.
        """
        # Give tokens back (reduce consumed amount)
        self.db.execute(text("""
            UPDATE agent_evolution.agents
            SET token_consumed = GREATEST(0, token_consumed - :amount),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = :id
        """), {"id": agent_id, "amount": amount})
        
        # Record transaction
        self._record_transaction(
            agent_id, TokenTransactionType.REWARD, amount, description, metadata or {}
        )
        
        # Update efficiency
        self._update_efficiency(agent_id)
        
        self.db.commit()
        return True
    
    def allocate_initial_tokens(self, agent_id: str, requested_amount: int) -> int:
        """
        Allocate initial tokens to a new agent.
        Amount may be reduced based on global budget constraints.
        """
        global_stats = self.get_global_stats()
        remaining_global = global_stats['remaining_budget']
        
        # Apply allocation strategy based on scarcity
        if remaining_global <= 0:
            return 0
            
        # Reduce allocation as budget depletes
        scarcity_factor = 1.0 - global_stats['scarcity_level']
        allocated_amount = int(requested_amount * scarcity_factor)
        
        # Ensure minimum viable allocation
        allocated_amount = max(allocated_amount, self.min_operation_cost * 10)
        
        # Don't exceed remaining budget
        allocated_amount = min(allocated_amount, remaining_global)
        
        # Update agent's token budget
        self.db.execute(text("""
            UPDATE agent_evolution.agents
            SET token_budget = :amount
            WHERE id = :id
        """), {"id": agent_id, "amount": allocated_amount})
        
        # Record allocation
        self._record_transaction(
            agent_id, TokenTransactionType.ALLOCATION, allocated_amount,
            "Initial token allocation", 
            {"requested": requested_amount, "scarcity_factor": scarcity_factor}
        )
        
        self.db.commit()
        return allocated_amount
    
    def calculate_pattern_reward(self, pattern_type: str, effectiveness: float,
                               reuse_count: int = 0) -> int:
        """
        Calculate reward for discovering a useful pattern.
        """
        base_rewards = {
            'optimization': 200,
            'todo_implementation': 150,
            'refactoring': 100,
            'cleanup': 50,
            'complexity_reduction': 250,
            'performance_improvement': 300
        }
        
        base_reward = base_rewards.get(pattern_type, 100)
        
        # Scale by effectiveness
        reward = int(base_reward * effectiveness)
        
        # Bonus for reusable patterns
        if reuse_count > 0:
            reward += reuse_count * 50
            
        return reward
    
    def enforce_budget_limits(self, agent_id: str) -> Dict[str, any]:
        """
        Enforce budget limits and potentially retire agent.
        Returns action taken.
        """
        _, consumed, remaining = self.get_agent_balance(agent_id)
        
        # Check if agent is out of tokens
        if remaining <= self.min_operation_cost:
            # Retire agent
            self.db.execute(text("""
                UPDATE agent_evolution.agents
                SET status = 'retired',
                    retirement_reason = 'insufficient_tokens',
                    terminated_at = CURRENT_TIMESTAMP
                WHERE id = :id
            """), {"id": agent_id})
            
            self.db.commit()
            
            return {
                'action': 'retired',
                'reason': 'insufficient_tokens',
                'final_balance': remaining
            }
            
        # Check if agent is inefficient
        result = self.db.execute(text("""
            SELECT token_efficiency, fitness_score
            FROM agent_evolution.agents
            WHERE id = :id
        """), {"id": agent_id}).fetchone()
        
        efficiency = result.token_efficiency if result else 0.5
        fitness = result.fitness_score if result else 0.0
        
        # Retire inefficient agents when tokens are scarce
        global_stats = self.get_global_stats()
        if global_stats['scarcity_level'] > 0.7 and efficiency < 0.3 and fitness < 0.5:
            self.db.execute(text("""
                UPDATE agent_evolution.agents
                SET status = 'retired',
                    retirement_reason = 'inefficient_under_scarcity',
                    terminated_at = CURRENT_TIMESTAMP
                WHERE id = :id
            """), {"id": agent_id})
            
            self.db.commit()
            
            return {
                'action': 'retired',
                'reason': 'inefficient_under_scarcity',
                'efficiency': efficiency,
                'scarcity_level': global_stats['scarcity_level']
            }
            
        return {
            'action': 'continue',
            'remaining_tokens': remaining,
            'efficiency': efficiency
        }
    
    def _record_transaction(self, agent_id: str, transaction_type: TokenTransactionType,
                          amount: int, description: str, metadata: Dict):
        """Record a token transaction."""
        # Get current balance before transaction
        result = self.db.execute(text("""
            SELECT token_budget - token_consumed as balance
            FROM agent_evolution.agents
            WHERE id = :id
        """), {"id": agent_id}).fetchone()
        
        balance_before = result.balance if result else 0
        balance_after = balance_before + amount
        
        self.db.execute(text("""
            INSERT INTO agent_evolution.token_transactions
            (agent_id, transaction_type, amount, reason, balance_before, balance_after)
            VALUES (:agent_id, :type, :amount, :reason, :balance_before, :balance_after)
        """), {
            "agent_id": agent_id,
            "type": transaction_type.value,
            "amount": amount,
            "reason": description,  # Changed from "description" to "reason"
            "balance_before": balance_before,
            "balance_after": balance_after
        })
    
    def _update_efficiency(self, agent_id: str):
        """Update agent's token efficiency metric."""
        # Calculate efficiency as fitness per token consumed
        result = self.db.execute(text("""
            SELECT fitness_score, token_consumed
            FROM agent_evolution.agents
            WHERE id = :id
        """), {"id": agent_id}).fetchone()
        
        if result and result.token_consumed > 0:
            efficiency = result.fitness_score / (result.token_consumed / 1000.0)
            
            self.db.execute(text("""
                UPDATE agent_evolution.agents
                SET token_efficiency = :efficiency
                WHERE id = :id
            """), {"id": agent_id, "efficiency": min(2.0, efficiency)})
    
    def get_economy_pressure_metrics(self) -> Dict[str, any]:
        """
        Get metrics showing how economic pressure affects the system.
        """
        # Get agent distribution by token consumption
        result = self.db.execute(text("""
            SELECT 
                COUNT(CASE WHEN token_consumed < token_budget * 0.25 THEN 1 END) as conservative,
                COUNT(CASE WHEN token_consumed BETWEEN token_budget * 0.25 AND token_budget * 0.75 THEN 1 END) as balanced,
                COUNT(CASE WHEN token_consumed > token_budget * 0.75 THEN 1 END) as aggressive,
                COUNT(CASE WHEN status = 'retired' AND retirement_reason = 'insufficient_tokens' THEN 1 END) as token_deaths
            FROM agent_evolution.agents
        """)).fetchone()
        
        # Get efficiency trends
        efficiency_result = self.db.execute(text("""
            SELECT 
                AVG(token_efficiency) as current_avg_efficiency,
                MIN(token_efficiency) as min_efficiency,
                MAX(token_efficiency) as max_efficiency
            FROM agent_evolution.agents
            WHERE status = 'active'
        """)).fetchone()
        
        global_stats = self.get_global_stats()
        
        return {
            'consumption_distribution': {
                'conservative': result.conservative or 0,
                'balanced': result.balanced or 0,
                'aggressive': result.aggressive or 0
            },
            'token_deaths': result.token_deaths or 0,
            'efficiency_metrics': {
                'average': float(efficiency_result.current_avg_efficiency or 0.0),
                'min': float(efficiency_result.min_efficiency or 0.0),
                'max': float(efficiency_result.max_efficiency or 0.0)
            },
            'scarcity_level': global_stats['scarcity_level'],
            'cost_multiplier': global_stats['scarcity_multiplier'],
            'selection_pressure': 'high' if global_stats['scarcity_level'] > 0.7 else 'medium' if global_stats['scarcity_level'] > 0.3 else 'low'
        }