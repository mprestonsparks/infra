"""
FractalAgent implementation for DEAN system.

Core agent class with cellular automata evolution, token economics,
and git worktree isolation.
"""

import asyncio
import uuid
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
from pathlib import Path

from ..economy import TokenBudget, TokenEconomyManager
from ..diversity import AgentGenome, GeneticDiversityManager
from ..patterns import PatternDetector, EmergentBehaviorMonitor
from ..repository import RepositoryManager
from .cellular_automata import CellularAutomataEngine, CARule

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """States of agent lifecycle."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    EVOLVING = "evolving"
    BOTTLENECKED = "bottlenecked"
    CONVERGED = "converged"
    RETIRED = "retired"
    ERROR = "error"


@dataclass
class EvolutionResult:
    """Result of an agent evolution cycle."""
    agent_id: str
    generation: int
    efficiency_before: float
    efficiency_after: float
    tokens_consumed: int
    patterns_discovered: List[str]
    ca_rules_activated: List[CARule]
    children_created: List[str]
    success: bool
    error_message: Optional[str] = None
    meta_agent_created: bool = False


class FractalAgent:
    """
    Core DEAN agent with cellular automata evolution.
    
    Implements fractal architecture where agents can create improved
    versions of themselves, fork into parallel execution paths,
    and recurse to higher abstraction levels.
    """
    
    def __init__(self,
                 agent_id: str,
                 genome: AgentGenome,
                 level: int = 0,
                 parent_id: Optional[str] = None,
                 token_budget: Optional[TokenBudget] = None,
                 worktree_manager: Optional['GitWorktreeManager'] = None):
        """
        Initialize FractalAgent.
        
        Args:
            agent_id: Unique identifier for this agent
            genome: Genetic configuration defining agent behavior
            level: Abstraction level (0=base, higher=meta-levels)
            parent_id: ID of parent agent if this is a child
            token_budget: Economic constraints for operations
            worktree_manager: Manager for isolated git worktrees
        """
        self.agent_id = agent_id
        self.genome = genome
        self.level = level
        self.parent_id = parent_id
        self.children: List[str] = []
        
        # Economic constraints
        self.token_budget = token_budget or TokenBudget(total=1000)
        
        # Evolution tracking
        self.state = AgentState.INITIALIZING
        self.generation = genome.generation
        self.emergent_patterns: List[str] = []
        self.efficiency_history: List[float] = []
        
        # Execution environment
        self.worktree_manager = worktree_manager
        self.worktree_path: Optional[str] = None
        
        # Evolution components (injected)
        self.ca_engine: Optional[CellularAutomataEngine] = None
        self.diversity_manager: Optional[GeneticDiversityManager] = None
        self.pattern_detector: Optional[PatternDetector] = None
        self.behavior_monitor: Optional[EmergentBehaviorMonitor] = None
        self.repository: Optional[RepositoryManager] = None
        
        # Performance metrics
        self._current_efficiency = 0.0
        self._total_value_generated = 0.0
        self._last_evolution_time = datetime.now()
        
        logger.info(f"Initialized FractalAgent {agent_id} at level {level}")
    
    async def initialize(self, 
                        ca_engine: CellularAutomataEngine,
                        diversity_manager: GeneticDiversityManager,
                        pattern_detector: PatternDetector,
                        behavior_monitor: EmergentBehaviorMonitor,
                        repository: RepositoryManager) -> None:
        """Initialize agent with required evolution components."""
        self.ca_engine = ca_engine
        self.diversity_manager = diversity_manager
        self.pattern_detector = pattern_detector
        self.behavior_monitor = behavior_monitor
        self.repository = repository
        
        # Create isolated worktree if manager available
        if self.worktree_manager and not self.worktree_path:
            try:
                worktree = await self.worktree_manager.create_worktree(
                    branch_name=f"agent_{self.agent_id}",
                    agent_id=self.agent_id,
                    token_limit=self.token_budget.total
                )
                self.worktree_path = str(worktree)
                logger.info(f"Agent {self.agent_id} isolated in worktree: {self.worktree_path}")
            except Exception as e:
                logger.warning(f"Failed to create worktree for agent {self.agent_id}: {e}")
        
        self.state = AgentState.ACTIVE
        logger.info(f"Agent {self.agent_id} fully initialized")
    
    async def evolve(self, environment: Dict[str, Any]) -> EvolutionResult:
        """
        Execute one evolution cycle using cellular automata rules.
        
        Args:
            environment: Current environment context including population
            
        Returns:
            EvolutionResult with detailed evolution outcomes
        """
        if self.state not in [AgentState.ACTIVE, AgentState.BOTTLENECKED]:
            return EvolutionResult(
                agent_id=self.agent_id,
                generation=self.generation,
                efficiency_before=self._current_efficiency,
                efficiency_after=self._current_efficiency,
                tokens_consumed=0,
                patterns_discovered=[],
                ca_rules_activated=[],
                children_created=[],
                success=False,
                error_message=f"Cannot evolve in state: {self.state.value}"
            )
        
        if not await self._can_afford_evolution():
            self.state = AgentState.RETIRED
            return EvolutionResult(
                agent_id=self.agent_id,
                generation=self.generation,
                efficiency_before=self._current_efficiency,
                efficiency_after=self._current_efficiency,
                tokens_consumed=0,
                patterns_discovered=[],
                ca_rules_activated=[],
                children_created=[],
                success=False,
                error_message="Insufficient token budget for evolution"
            )
        
        self.state = AgentState.EVOLVING
        initial_efficiency = self._current_efficiency
        initial_tokens = self.token_budget.used
        initial_children_count = len(self.children)
        
        try:
            # Detect current patterns
            new_patterns = await self._detect_emergent_patterns()
            
            # Update efficiency based on performance
            await self._update_efficiency_metrics()
            
            # Apply diversity maintenance
            if self.diversity_manager:
                await self._maintain_genetic_diversity(environment.get('population', []))
            
            # Execute CA evolution step if in population context
            ca_result = {}
            activated_rules = []
            if self.ca_engine and 'population' in environment:
                population = environment['population']
                if self in population:
                    ca_result = await self.ca_engine.step(population)
                    activated_rules = list(ca_result.get('rule_activations', {}).keys())
            
            # Record evolution in repository
            if self.repository:
                await self._record_evolution_data(new_patterns, ca_result)
            
            # Update state based on performance
            await self._update_agent_state()
            
            final_efficiency = self._current_efficiency
            tokens_consumed = self.token_budget.used - initial_tokens
            children_created = self.children[initial_children_count:]
            
            # Check for meta-agent creation
            meta_agent_created = any(
                'meta_agent_created' in ca_result.get('rule_activations', {}).get(rule, [{}])[0].get('result', {})
                for rule in activated_rules
                if rule == CARule.RULE_1.value
            )
            
            result = EvolutionResult(
                agent_id=self.agent_id,
                generation=self.generation,
                efficiency_before=initial_efficiency,
                efficiency_after=final_efficiency,
                tokens_consumed=tokens_consumed,
                patterns_discovered=new_patterns,
                ca_rules_activated=[CARule(rule) for rule in activated_rules],
                children_created=children_created,
                success=True,
                meta_agent_created=meta_agent_created
            )
            
            self._last_evolution_time = datetime.now()
            logger.info(f"Agent {self.agent_id} evolution completed: "
                       f"efficiency {initial_efficiency:.3f} → {final_efficiency:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Evolution failed for agent {self.agent_id}: {e}")
            self.state = AgentState.ERROR
            
            return EvolutionResult(
                agent_id=self.agent_id,
                generation=self.generation,
                efficiency_before=initial_efficiency,
                efficiency_after=self._current_efficiency,
                tokens_consumed=self.token_budget.used - initial_tokens,
                patterns_discovered=[],
                ca_rules_activated=[],
                children_created=[],
                success=False,
                error_message=str(e)
            )
    
    def get_efficiency(self) -> float:
        """Get current token efficiency (value/tokens)."""
        return self._current_efficiency
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        return {
            'agent_id': self.agent_id,
            'generation': self.generation,
            'level': self.level,
            'efficiency': self._current_efficiency,
            'efficiency_history': self.efficiency_history.copy(),
            'total_value_generated': self._total_value_generated,
            'tokens_consumed': self.token_budget.used,
            'tokens_remaining': self.token_budget.available,
            'budget_utilization': self.token_budget.utilization_rate,
            'patterns_discovered': len(self.emergent_patterns),
            'children_count': len(self.children),
            'state': self.state.value,
            'worktree_path': self.worktree_path,
            'last_evolution': self._last_evolution_time.isoformat()
        }
    
    async def cleanup(self) -> None:
        """Clean up agent resources."""
        try:
            # Clean up worktree
            if self.worktree_manager and self.worktree_path:
                await self.worktree_manager.cleanup_worktree(self.agent_id)
                self.worktree_path = None
            
            self.state = AgentState.RETIRED
            logger.info(f"Agent {self.agent_id} cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error cleaning up agent {self.agent_id}: {e}")
    
    async def _can_afford_evolution(self) -> bool:
        """Check if agent can afford evolution operation."""
        min_evolution_cost = 100  # Minimum tokens needed for evolution
        return self.token_budget.can_afford(min_evolution_cost)
    
    async def _detect_emergent_patterns(self) -> List[str]:
        """Detect new emergent patterns in agent behavior."""
        if not self.pattern_detector:
            return []
        
        try:
            # Detect patterns for this agent
            patterns = self.pattern_detector.detect_patterns(self.agent_id)
            
            # Filter for novel patterns
            new_patterns = []
            for pattern in patterns:
                pattern_id = pattern.pattern_id if hasattr(pattern, 'pattern_id') else f"pattern_{uuid.uuid4().hex[:8]}"
                if pattern_id not in self.emergent_patterns:
                    self.emergent_patterns.append(pattern_id)
                    new_patterns.append(pattern_id)
                    
                    # Monitor with behavior monitor
                    if self.behavior_monitor:
                        await self.behavior_monitor.record_pattern_discovery(
                            agent_id=self.agent_id,
                            pattern=pattern
                        )
            
            return new_patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns for agent {self.agent_id}: {e}")
            return []
    
    async def _update_efficiency_metrics(self) -> None:
        """Update agent's efficiency based on recent performance."""
        try:
            # Calculate efficiency based on value generated vs tokens consumed
            if self.token_budget.used > 0:
                # Simulate value generation based on genome effectiveness
                genome_effectiveness = self._calculate_genome_effectiveness()
                
                # Base value on successful operations
                operation_value = genome_effectiveness * 10  # Base value per operation
                pattern_bonus = len(self.emergent_patterns) * 0.1  # Bonus for patterns
                
                self._total_value_generated += operation_value + pattern_bonus
                self._current_efficiency = self._total_value_generated / self.token_budget.used
            else:
                self._current_efficiency = 0.0
            
            # Track efficiency history
            self.efficiency_history.append(self._current_efficiency)
            
            # Keep only recent history (last 20 measurements)
            if len(self.efficiency_history) > 20:
                self.efficiency_history = self.efficiency_history[-20:]
                
        except Exception as e:
            logger.error(f"Error updating efficiency for agent {self.agent_id}: {e}")
    
    def _calculate_genome_effectiveness(self) -> float:
        """Calculate effectiveness score based on genome."""
        try:
            # Simple heuristic based on genome diversity and gene quality
            gene_count = len(self.genome.genes)
            gene_quality = sum(
                1.0 if hasattr(gene, 'value') and gene.value else 0.5 
                for gene in self.genome.genes
            )
            
            base_effectiveness = min(gene_quality / max(gene_count, 1), 2.0)
            
            # Bonus for higher generation (accumulated learning)
            generation_bonus = min(self.generation * 0.1, 1.0)
            
            return base_effectiveness + generation_bonus
            
        except Exception as e:
            logger.error(f"Error calculating genome effectiveness: {e}")
            return 1.0  # Default effectiveness
    
    async def _maintain_genetic_diversity(self, population: List['FractalAgent']) -> None:
        """Maintain genetic diversity within population."""
        if not self.diversity_manager or len(population) < 2:
            return
        
        try:
            # Check if diversity intervention is needed
            genomes = [agent.genome for agent in population if agent.genome]
            diversity_metrics = self.diversity_manager.calculate_diversity_metrics(genomes)
            
            if diversity_metrics.overall_diversity < self.diversity_manager.min_diversity:
                # Apply diversity-preserving mutations
                if self.token_budget.can_afford(50):
                    # Apply light mutation for diversity
                    mutated_genome = self.diversity_manager.mutation_engine.mutate(
                        self.genome, force_mutation=True
                    )
                    
                    if mutated_genome:
                        self.genome = mutated_genome
                        self.token_budget.consume(50)
                        logger.info(f"Applied diversity mutation to agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Error maintaining diversity for agent {self.agent_id}: {e}")
    
    async def _record_evolution_data(self, new_patterns: List[str], ca_result: Dict) -> None:
        """Record evolution data in repository."""
        if not self.repository:
            return
        
        try:
            # Create pattern objects for discovered patterns
            patterns = []
            for pattern_id in new_patterns:
                from ..patterns import Pattern, PatternType
                pattern = Pattern(
                    pattern_id=pattern_id,
                    pattern_type=PatternType.BEHAVIORAL,
                    description=f"Emergent pattern from agent {self.agent_id}",
                    effectiveness=self._current_efficiency,
                    confidence=0.8
                )
                patterns.append(pattern)
            
            # Record evolution in repository
            performance_metrics = {
                'tokens_used': self.token_budget.used,
                'value_generated': self._total_value_generated,
                'efficiency': self._current_efficiency,
                'action_type': 'ca_evolution',
                'ca_rules_activated': list(ca_result.get('rule_activations', {}).keys()),
                'agents_created': ca_result.get('agents_created', 0),
                'patterns_discovered': ca_result.get('patterns_discovered', 0)
            }
            
            self.repository.record_agent_evolution(
                agent_id=self.agent_id,
                genome=self.genome,
                performance_metrics=performance_metrics,
                discovered_patterns=patterns
            )
            
        except Exception as e:
            logger.error(f"Error recording evolution data for agent {self.agent_id}: {e}")
    
    async def _update_agent_state(self) -> None:
        """Update agent state based on current performance."""
        try:
            # Check for convergence
            if len(self.efficiency_history) >= 5:
                recent_variance = float(
                    sum((eff - self._current_efficiency) ** 2 for eff in self.efficiency_history[-5:]) / 5
                )
                
                if recent_variance < 0.001:  # Very low variance indicates convergence
                    self.state = AgentState.CONVERGED
                    return
            
            # Check for bottlenecks
            if len(self.efficiency_history) >= 3:
                recent_trend = [
                    self.efficiency_history[i] - self.efficiency_history[i-1] 
                    for i in range(-2, 0)
                ]
                
                if all(trend <= 0 for trend in recent_trend):
                    self.state = AgentState.BOTTLENECKED
                    return
            
            # Check token budget status
            if self.token_budget.utilization_rate > 0.9:
                self.state = AgentState.BOTTLENECKED
                return
            
            # Default to active if no issues
            if self.state == AgentState.EVOLVING:
                self.state = AgentState.ACTIVE
                
        except Exception as e:
            logger.error(f"Error updating agent state for {self.agent_id}: {e}")
            self.state = AgentState.ERROR