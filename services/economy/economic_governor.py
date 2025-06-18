#!/usr/bin/env python3
"""
Economic Governor for DEAN System
Manages token budgets and allocation across agent populations
"""

import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import psycopg2
from psycopg2.extras import RealDictCursor
import os

logger = logging.getLogger(__name__)


@dataclass
class AgentBudget:
    """Token budget allocation for an agent"""
    agent_id: str
    current_budget: int
    total_allocated: int
    total_used: int
    efficiency_score: float  # tokens per unit of success
    last_allocation: datetime
    allocation_history: List[Dict[str, any]]


@dataclass
class BudgetAllocation:
    """Record of a budget allocation decision"""
    agent_id: str
    amount: int
    reason: str
    performance_metrics: Dict[str, float]
    timestamp: datetime


class GlobalBudgetManager:
    """Manages global token budget across all agents"""
    
    def __init__(self, total_budget: int, reserve_ratio: float = 0.2):
        """
        Initialize global budget manager
        
        Args:
            total_budget: Total tokens available for allocation
            reserve_ratio: Fraction to keep in reserve (0.0-1.0)
        """
        self.total_budget = total_budget
        self.reserve_ratio = reserve_ratio
        self.allocated_budget = 0
        self.used_budget = 0
        self.reserve_budget = int(total_budget * reserve_ratio)
        self.available_budget = total_budget - self.reserve_budget
        
        logger.info(f"GlobalBudgetManager initialized: total={total_budget}, available={self.available_budget}, reserve={self.reserve_budget}")
    
    def can_allocate(self, amount: int) -> bool:
        """Check if budget is available for allocation"""
        return (self.allocated_budget + amount) <= self.available_budget
    
    def allocate(self, amount: int) -> bool:
        """Allocate budget if available"""
        if self.can_allocate(amount):
            self.allocated_budget += amount
            return True
        return False
    
    def deallocate(self, amount: int):
        """Return unused budget to pool"""
        self.allocated_budget = max(0, self.allocated_budget - amount)
    
    def record_usage(self, amount: int):
        """Record actual token usage"""
        self.used_budget += amount
    
    def get_utilization(self) -> Dict[str, float]:
        """Get budget utilization metrics"""
        return {
            'total_budget': self.total_budget,
            'allocated_budget': self.allocated_budget,
            'used_budget': self.used_budget,
            'available_budget': self.available_budget - self.allocated_budget,
            'reserve_budget': self.reserve_budget,
            'allocation_rate': self.allocated_budget / self.available_budget if self.available_budget > 0 else 0,
            'usage_rate': self.used_budget / self.allocated_budget if self.allocated_budget > 0 else 0,
            'efficiency': self.used_budget / self.total_budget if self.total_budget > 0 else 0
        }


class AgentBudgetAllocator:
    """Allocates budgets to individual agents based on performance"""
    
    def __init__(self, base_allocation: int = 1000, 
                 performance_multiplier: float = 1.5,
                 decay_rate: float = 0.1):
        """
        Initialize agent budget allocator
        
        Args:
            base_allocation: Base token allocation for new agents
            performance_multiplier: Max multiplier for high performers
            decay_rate: Rate of budget decay for inactive agents
        """
        self.base_allocation = base_allocation
        self.performance_multiplier = performance_multiplier
        self.decay_rate = decay_rate
        self.agent_budgets: Dict[str, AgentBudget] = {}
        
    def create_agent_budget(self, agent_id: str) -> AgentBudget:
        """Create initial budget for new agent"""
        budget = AgentBudget(
            agent_id=agent_id,
            current_budget=self.base_allocation,
            total_allocated=self.base_allocation,
            total_used=0,
            efficiency_score=0.0,
            last_allocation=datetime.now(),
            allocation_history=[{
                'amount': self.base_allocation,
                'reason': 'initial_allocation',
                'timestamp': datetime.now().isoformat()
            }]
        )
        
        self.agent_budgets[agent_id] = budget
        logger.info(f"Created budget for agent {agent_id}: {self.base_allocation} tokens")
        
        return budget
    
    def calculate_performance_allocation(self, agent_id: str, 
                                       task_success: float,
                                       tokens_used: int,
                                       generation: int) -> int:
        """Calculate allocation based on agent performance"""
        if agent_id not in self.agent_budgets:
            return self.base_allocation
        
        budget = self.agent_budgets[agent_id]
        
        # Calculate efficiency (success per token)
        if tokens_used > 0:
            current_efficiency = task_success / (tokens_used / 1000)
            
            # Update rolling efficiency score
            alpha = 0.3  # Learning rate
            budget.efficiency_score = (alpha * current_efficiency + 
                                     (1 - alpha) * budget.efficiency_score)
        
        # Base allocation with performance modifier
        performance_factor = min(self.performance_multiplier, 
                               1.0 + budget.efficiency_score)
        
        # Generational bonus for consistent performers
        generation_bonus = 1.0 + (generation * 0.02) if task_success > 0.7 else 1.0
        
        allocation = int(self.base_allocation * performance_factor * generation_bonus)
        
        return allocation
    
    def allocate_budget(self, agent_id: str, amount: int, reason: str,
                       performance_metrics: Dict[str, float]) -> bool:
        """Allocate budget to agent"""
        if agent_id not in self.agent_budgets:
            self.create_agent_budget(agent_id)
        
        budget = self.agent_budgets[agent_id]
        budget.current_budget += amount
        budget.total_allocated += amount
        budget.last_allocation = datetime.now()
        
        budget.allocation_history.append({
            'amount': amount,
            'reason': reason,
            'performance': performance_metrics,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep history size manageable
        if len(budget.allocation_history) > 100:
            budget.allocation_history = budget.allocation_history[-50:]
        
        logger.info(f"Allocated {amount} tokens to agent {agent_id} (reason: {reason})")
        
        return True
    
    def use_budget(self, agent_id: str, amount: int) -> bool:
        """Use tokens from agent's budget"""
        if agent_id not in self.agent_budgets:
            return False
        
        budget = self.agent_budgets[agent_id]
        
        if budget.current_budget >= amount:
            budget.current_budget -= amount
            budget.total_used += amount
            return True
        
        return False
    
    def apply_decay(self, inactive_threshold_hours: int = 24):
        """Apply budget decay to inactive agents"""
        now = datetime.now()
        threshold = now - timedelta(hours=inactive_threshold_hours)
        
        for agent_id, budget in self.agent_budgets.items():
            if budget.last_allocation < threshold:
                # Apply decay
                decay_amount = int(budget.current_budget * self.decay_rate)
                budget.current_budget = max(0, budget.current_budget - decay_amount)
                
                if decay_amount > 0:
                    logger.info(f"Applied decay of {decay_amount} tokens to inactive agent {agent_id}")
    
    def get_agent_budget(self, agent_id: str) -> Optional[AgentBudget]:
        """Get current budget for agent"""
        return self.agent_budgets.get(agent_id)
    
    def get_top_performers(self, count: int = 10) -> List[Tuple[str, AgentBudget]]:
        """Get top performing agents by efficiency"""
        sorted_agents = sorted(
            self.agent_budgets.items(),
            key=lambda x: x[1].efficiency_score,
            reverse=True
        )
        
        return sorted_agents[:count]


class EconomicGovernor:
    """Main economic governance system for DEAN"""
    
    def __init__(self, db_url: str, total_budget: int = 1000000):
        """
        Initialize economic governor
        
        Args:
            db_url: PostgreSQL connection URL
            total_budget: Total token budget for system
        """
        self.db_url = db_url
        self.global_budget = GlobalBudgetManager(total_budget)
        self.agent_allocator = AgentBudgetAllocator()
        
        # Initialize database
        self._init_database()
        
        logger.info(f"EconomicGovernor initialized with budget: {total_budget}")
    
    def _init_database(self):
        """Initialize database tables"""
        conn = psycopg2.connect(self.db_url)
        cur = conn.cursor()
        
        # Create token allocations table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS token_allocations (
                id SERIAL PRIMARY KEY,
                agent_id VARCHAR(255) NOT NULL,
                allocation_amount INTEGER NOT NULL,
                reason VARCHAR(255),
                performance_metrics JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                generation INTEGER,
                efficiency_score FLOAT
            )
        """)
        
        # Create token usage table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS token_usage (
                id SERIAL PRIMARY KEY,
                agent_id VARCHAR(255) NOT NULL,
                tokens_used INTEGER NOT NULL,
                action_type VARCHAR(100),
                task_success FLOAT,
                quality_score FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indices
        cur.execute("CREATE INDEX IF NOT EXISTS idx_allocations_agent ON token_allocations(agent_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_allocations_created ON token_allocations(created_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_usage_agent ON token_usage(agent_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_usage_created ON token_usage(created_at)")
        
        conn.commit()
        conn.close()
    
    def allocate_to_agent(self, agent_id: str, task_success: float,
                         tokens_used: int, quality_score: float,
                         generation: int) -> Optional[int]:
        """
        Allocate tokens to agent based on performance
        
        Returns:
            Allocated amount or None if allocation failed
        """
        # Calculate allocation
        allocation = self.agent_allocator.calculate_performance_allocation(
            agent_id, task_success, tokens_used, generation
        )
        
        # Apply efficiency bonus
        agent_budget = self.agent_allocator.get_agent_budget(agent_id)
        if agent_budget and agent_budget.efficiency_score > 1.0:
            efficiency_bonus = min(1.5, agent_budget.efficiency_score)
            allocation = int(allocation * efficiency_bonus)
            logger.info(f"Applied efficiency bonus {efficiency_bonus:.2f}x to agent {agent_id}")
        
        # Check global budget
        if not self.global_budget.can_allocate(allocation):
            # Try reduced allocation
            allocation = int(allocation * 0.5)
            if not self.global_budget.can_allocate(allocation):
                logger.warning(f"Insufficient global budget for agent {agent_id}")
                return None
        
        # Perform allocation
        self.global_budget.allocate(allocation)
        
        performance_metrics = {
            'task_success': task_success,
            'tokens_used': tokens_used,
            'quality_score': quality_score,
            'generation': generation
        }
        
        self.agent_allocator.allocate_budget(
            agent_id, allocation, 'performance_based', performance_metrics
        )
        
        # Record in database
        self._record_allocation(agent_id, allocation, 'performance_based',
                              performance_metrics, generation)
        
        return allocation
    
    def use_tokens(self, agent_id: str, tokens: int, action_type: str,
                  task_success: float, quality_score: float) -> bool:
        """
        Use tokens from agent's budget
        
        Returns:
            True if successful, False if insufficient budget
        """
        # Check agent budget
        if not self.agent_allocator.use_budget(agent_id, tokens):
            logger.warning(f"Agent {agent_id} has insufficient budget for {tokens} tokens")
            return False
        
        # Record usage
        self.global_budget.record_usage(tokens)
        self._record_usage(agent_id, tokens, action_type, task_success, quality_score)
        
        return True
    
    def rebalance_budgets(self):
        """Rebalance budgets based on performance"""
        # Apply decay to inactive agents
        self.agent_allocator.apply_decay()
        
        # Get top and bottom performers
        all_agents = list(self.agent_allocator.agent_budgets.items())
        sorted_agents = sorted(all_agents, key=lambda x: x[1].efficiency_score, reverse=True)
        
        if len(sorted_agents) < 2:
            return
        
        # Transfer budget from bottom to top performers
        top_20_percent = int(len(sorted_agents) * 0.2)
        bottom_20_percent = int(len(sorted_agents) * 0.2)
        
        top_agents = sorted_agents[:top_20_percent]
        bottom_agents = sorted_agents[-bottom_20_percent:]
        
        # Calculate transfer amount
        total_to_transfer = 0
        for agent_id, budget in bottom_agents:
            transfer = int(budget.current_budget * 0.2)  # Transfer 20% from underperformers
            if transfer > 0:
                budget.current_budget -= transfer
                total_to_transfer += transfer
                logger.info(f"Reducing budget for underperforming agent {agent_id} by {transfer}")
        
        # Distribute to top performers
        if total_to_transfer > 0 and top_agents:
            per_agent = total_to_transfer // len(top_agents)
            for agent_id, budget in top_agents:
                budget.current_budget += per_agent
                self._record_allocation(agent_id, per_agent, 'rebalance_bonus', 
                                      {'efficiency': budget.efficiency_score}, 0)
                logger.info(f"Bonus allocation of {per_agent} to top performer {agent_id}")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system economic metrics"""
        global_metrics = self.global_budget.get_utilization()
        
        # Get agent statistics
        agent_count = len(self.agent_allocator.agent_budgets)
        
        if agent_count > 0:
            avg_efficiency = sum(b.efficiency_score for b in self.agent_allocator.agent_budgets.values()) / agent_count
            total_agent_budgets = sum(b.current_budget for b in self.agent_allocator.agent_budgets.values())
        else:
            avg_efficiency = 0.0
            total_agent_budgets = 0
        
        return {
            **global_metrics,
            'agent_count': agent_count,
            'average_efficiency': avg_efficiency,
            'total_agent_budgets': total_agent_budgets,
            'top_performers': [
                {'agent_id': aid, 'efficiency': b.efficiency_score, 'budget': b.current_budget}
                for aid, b in self.agent_allocator.get_top_performers(5)
            ]
        }
    
    def _record_allocation(self, agent_id: str, amount: int, reason: str,
                         metrics: Dict[str, float], generation: int):
        """Record allocation in database"""
        conn = psycopg2.connect(self.db_url)
        cur = conn.cursor()
        
        efficiency = self.agent_allocator.agent_budgets[agent_id].efficiency_score
        
        cur.execute("""
            INSERT INTO token_allocations 
            (agent_id, allocation_amount, reason, performance_metrics, generation, efficiency_score)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (agent_id, amount, reason, json.dumps(metrics), generation, efficiency))
        
        conn.commit()
        conn.close()
    
    def _record_usage(self, agent_id: str, tokens: int, action_type: str,
                     task_success: float, quality_score: float):
        """Record token usage in database"""
        conn = psycopg2.connect(self.db_url)
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO token_usage 
            (agent_id, tokens_used, action_type, task_success, quality_score)
            VALUES (%s, %s, %s, %s, %s)
        """, (agent_id, tokens, action_type, task_success, quality_score))
        
        conn.commit()
        conn.close()


if __name__ == "__main__":
    # Demo the economic governor
    db_url = os.environ.get('DATABASE_URL', 'postgresql://dean_user:dean_password_2024@localhost:5432/agent_evolution')
    
    governor = EconomicGovernor(db_url, total_budget=100000)
    
    # Simulate some agent allocations
    print("Economic Governor Demo")
    print("=" * 60)
    
    # Create agents with different performance levels
    agents = [
        ("agent_001", 0.9, 1500, 0.85),  # High performer
        ("agent_002", 0.7, 2000, 0.75),  # Medium performer
        ("agent_003", 0.4, 3000, 0.60),  # Low performer
    ]
    
    print("\n1. Initial allocations:")
    for agent_id, success, tokens, quality in agents:
        allocation = governor.allocate_to_agent(agent_id, success, tokens, quality, 1)
        print(f"   {agent_id}: allocated {allocation} tokens")
    
    print("\n2. Token usage:")
    for agent_id, _, _, _ in agents:
        success = governor.use_tokens(agent_id, 500, "implement_todos", 0.8, 0.8)
        print(f"   {agent_id}: used 500 tokens - {'success' if success else 'failed'}")
    
    print("\n3. System metrics:")
    metrics = governor.get_system_metrics()
    print(f"   Total budget: {metrics['total_budget']}")
    print(f"   Used budget: {metrics['used_budget']}")
    print(f"   Average efficiency: {metrics['average_efficiency']:.3f}")
    
    print("\n4. Top performers:")
    for performer in metrics['top_performers']:
        print(f"   {performer['agent_id']}: efficiency={performer['efficiency']:.3f}, budget={performer['budget']}")
    
    # Rebalance
    print("\n5. Rebalancing budgets...")
    governor.rebalance_budgets()
    
    print("\nDemo complete!")