"""
Token allocation strategies for the DEAN system.

Implements various algorithms for distributing tokens among agents
based on performance, diversity contribution, and innovation metrics.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentPerformance:
    """Performance metrics for allocation decisions."""
    agent_id: str
    efficiency: float  # value per token
    diversity_contribution: float  # 0-1, how unique the agent is
    innovation_score: float  # novel patterns discovered
    stability: float  # consistency of performance
    total_value: float  # cumulative value generated
    

class AllocationStrategy(ABC):
    """Base class for token allocation strategies."""
    
    @abstractmethod
    def allocate(self, 
                 total_budget: int,
                 performances: List[AgentPerformance],
                 min_allocation: int = 100) -> Dict[str, int]:
        """
        Allocate tokens among agents.
        
        Args:
            total_budget: Total tokens to distribute
            performances: Performance metrics for each agent
            min_allocation: Minimum tokens per agent
            
        Returns:
            Dict mapping agent_id to token allocation
        """
        pass


class PerformanceBasedAllocator(AllocationStrategy):
    """
    Allocates tokens based on multi-factor performance scoring.
    
    Balances efficiency, diversity contribution, and innovation
    to prevent monocultures while rewarding valuable agents.
    """
    
    def __init__(self,
                 efficiency_weight: float = 0.4,
                 diversity_weight: float = 0.3,
                 innovation_weight: float = 0.2,
                 stability_weight: float = 0.1):
        """
        Initialize allocator with scoring weights.
        
        Weights should sum to 1.0 for normalized scoring.
        """
        self.efficiency_weight = efficiency_weight
        self.diversity_weight = diversity_weight
        self.innovation_weight = innovation_weight
        self.stability_weight = stability_weight
        
        # Normalize weights
        total_weight = sum([efficiency_weight, diversity_weight, 
                           innovation_weight, stability_weight])
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total_weight}, normalizing...")
            self.efficiency_weight /= total_weight
            self.diversity_weight /= total_weight
            self.innovation_weight /= total_weight
            self.stability_weight /= total_weight
    
    def allocate(self,
                 total_budget: int,
                 performances: List[AgentPerformance],
                 min_allocation: int = 100) -> Dict[str, int]:
        """Allocate tokens using weighted performance scoring."""
        if not performances:
            return {}
        
        # Calculate composite scores
        scores = []
        for perf in performances:
            score = (
                self.efficiency_weight * self._normalize_efficiency(perf.efficiency) +
                self.diversity_weight * perf.diversity_contribution +
                self.innovation_weight * self._normalize_innovation(perf.innovation_score) +
                self.stability_weight * perf.stability
            )
            scores.append((perf.agent_id, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate allocations
        allocations = {}
        remaining_budget = total_budget
        num_agents = len(performances)
        
        # Ensure minimum allocation for all agents
        base_allocation = min(min_allocation, remaining_budget // num_agents)
        for agent_id, _ in scores:
            allocations[agent_id] = base_allocation
            remaining_budget -= base_allocation
        
        # Distribute remaining budget proportionally to scores
        if remaining_budget > 0 and sum(s[1] for s in scores) > 0:
            total_score = sum(s[1] for s in scores)
            for agent_id, score in scores:
                bonus = int(remaining_budget * (score / total_score))
                allocations[agent_id] += bonus
        
        # Log allocation decisions
        for agent_id, allocation in allocations.items():
            perf = next(p for p in performances if p.agent_id == agent_id)
            logger.debug(f"Agent {agent_id}: allocated {allocation} tokens "
                        f"(eff={perf.efficiency:.3f}, div={perf.diversity_contribution:.3f}, "
                        f"inn={perf.innovation_score:.3f}, stab={perf.stability:.3f})")
        
        return allocations
    
    def _normalize_efficiency(self, efficiency: float) -> float:
        """Normalize efficiency scores using sigmoid function."""
        # Map efficiency to 0-1 range with diminishing returns
        return 1 / (1 + np.exp(-2 * (efficiency - 1)))
    
    def _normalize_innovation(self, innovation_score: float) -> float:
        """Normalize innovation scores."""
        # Use log scale for innovation to reward early discoveries more
        return np.log1p(innovation_score) / np.log1p(10)


class TournamentAllocator(AllocationStrategy):
    """
    Tournament-based allocation strategy.
    
    Agents compete in small groups, with winners receiving
    larger allocations. Promotes local competition while
    maintaining global diversity.
    """
    
    def __init__(self, tournament_size: int = 4, elite_bonus: float = 1.5):
        self.tournament_size = tournament_size
        self.elite_bonus = elite_bonus
    
    def allocate(self,
                 total_budget: int,
                 performances: List[AgentPerformance],
                 min_allocation: int = 100) -> Dict[str, int]:
        """Run tournaments and allocate based on results."""
        if not performances:
            return {}
        
        # Shuffle for random tournament groupings
        agents = performances.copy()
        np.random.shuffle(agents)
        
        # Run tournaments
        tournament_results = {}
        for i in range(0, len(agents), self.tournament_size):
            group = agents[i:i + self.tournament_size]
            
            # Score agents in tournament
            group_scores = [(a.agent_id, self._score_agent(a)) for a in group]
            group_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Assign tournament rankings
            for rank, (agent_id, score) in enumerate(group_scores):
                tournament_results[agent_id] = {
                    'rank': rank,
                    'group_size': len(group),
                    'score': score
                }
        
        # Calculate allocations based on tournament results
        allocations = {}
        base_per_agent = total_budget // len(performances)
        
        for agent_id, result in tournament_results.items():
            if result['rank'] == 0:  # Tournament winner
                allocation = int(base_per_agent * self.elite_bonus)
            else:
                # Scale allocation by tournament performance
                rank_factor = 1 - (result['rank'] / result['group_size'])
                allocation = int(base_per_agent * (0.5 + 0.5 * rank_factor))
            
            allocations[agent_id] = max(allocation, min_allocation)
        
        # Normalize to fit budget
        total_allocated = sum(allocations.values())
        if total_allocated > total_budget:
            scale_factor = total_budget / total_allocated
            for agent_id in allocations:
                allocations[agent_id] = max(
                    int(allocations[agent_id] * scale_factor),
                    min_allocation
                )
        
        return allocations
    
    def _score_agent(self, agent: AgentPerformance) -> float:
        """Calculate tournament score for an agent."""
        # Balanced scoring for tournament selection
        return (
            0.4 * agent.efficiency +
            0.3 * agent.diversity_contribution +
            0.2 * agent.innovation_score +
            0.1 * agent.stability
        )


class AdaptiveAllocator(AllocationStrategy):
    """
    Adaptive allocation that adjusts strategy based on system state.
    
    Switches between exploration (diversity-focused) and exploitation
    (efficiency-focused) based on population metrics.
    """
    
    def __init__(self):
        self.performance_allocator = PerformanceBasedAllocator()
        self.exploration_allocator = PerformanceBasedAllocator(
            efficiency_weight=0.2,
            diversity_weight=0.5,
            innovation_weight=0.3,
            stability_weight=0.0
        )
        self.exploitation_allocator = PerformanceBasedAllocator(
            efficiency_weight=0.7,
            diversity_weight=0.1,
            innovation_weight=0.1,
            stability_weight=0.1
        )
    
    def allocate(self,
                 total_budget: int,
                 performances: List[AgentPerformance],
                 min_allocation: int = 100) -> Dict[str, int]:
        """Adaptively choose allocation strategy based on system state."""
        if not performances:
            return {}
        
        # Calculate system metrics
        avg_diversity = np.mean([p.diversity_contribution for p in performances])
        avg_efficiency = np.mean([p.efficiency for p in performances])
        innovation_rate = sum(p.innovation_score > 0 for p in performances) / len(performances)
        
        # Choose strategy based on metrics
        if avg_diversity < 0.3:
            # Low diversity - use exploration strategy
            logger.info("Using exploration strategy (low diversity)")
            return self.exploration_allocator.allocate(total_budget, performances, min_allocation)
        elif avg_efficiency > 2.0 and innovation_rate < 0.1:
            # High efficiency but low innovation - encourage exploration
            logger.info("Using exploration strategy (stagnant innovation)")
            return self.exploration_allocator.allocate(total_budget, performances, min_allocation)
        elif avg_diversity > 0.7 and avg_efficiency < 0.5:
            # High diversity but low efficiency - focus on exploitation
            logger.info("Using exploitation strategy (inefficient population)")
            return self.exploitation_allocator.allocate(total_budget, performances, min_allocation)
        else:
            # Balanced state - use default strategy
            logger.info("Using balanced strategy")
            return self.performance_allocator.allocate(total_budget, performances, min_allocation)