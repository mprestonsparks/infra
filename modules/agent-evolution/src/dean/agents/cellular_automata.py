"""
Cellular Automata Engine for DEAN agents.

Implements the five core CA rules with economic constraints,
diversity maintenance, and emergent behavior capture.
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
from abc import ABC, abstractmethod

from ..economy import TokenBudget, TokenEconomyManager
from ..diversity import AgentGenome, GeneticDiversityManager
from ..patterns import PatternDetector, EmergentBehaviorMonitor
from ..repository import RepositoryManager

logger = logging.getLogger(__name__)


class CARule(Enum):
    """Cellular Automata rules for agent evolution."""
    RULE_110 = "rule_110"  # Create improved neighbors
    RULE_30 = "rule_30"    # Fork into parallel worktrees
    RULE_90 = "rule_90"    # Abstract patterns
    RULE_184 = "rule_184"  # Learn from neighbors
    RULE_1 = "rule_1"      # Recurse to higher abstraction


@dataclass
class Neighborhood:
    """Represents an agent's neighborhood in the cellular automata space."""
    center_agent: 'FractalAgent'
    neighbors: List['FractalAgent'] = field(default_factory=list)
    generation: int = 0
    diversity_score: float = 0.0
    average_efficiency: float = 0.0
    
    def calculate_metrics(self) -> None:
        """Calculate neighborhood metrics."""
        if not self.neighbors:
            return
        
        # Calculate average efficiency
        efficiencies = [n.get_efficiency() for n in self.neighbors if n.get_efficiency() > 0]
        self.average_efficiency = np.mean(efficiencies) if efficiencies else 0.0
        
        # Calculate diversity (genetic distance variance)
        if len(self.neighbors) > 1:
            distances = []
            for i in range(len(self.neighbors)):
                for j in range(i + 1, len(self.neighbors)):
                    distance = self.neighbors[i].genome.distance_from(self.neighbors[j].genome)
                    distances.append(distance)
            self.diversity_score = np.var(distances) if distances else 0.0
        else:
            self.diversity_score = 1.0  # Single agent = max diversity
    
    def get_best_neighbor(self) -> Optional['FractalAgent']:
        """Get the highest performing neighbor."""
        if not self.neighbors:
            return None
        return max(self.neighbors, key=lambda a: a.get_efficiency())
    
    def get_performance_ranking(self, agent: 'FractalAgent') -> float:
        """Get agent's performance ranking in neighborhood (0-1)."""
        efficiencies = [n.get_efficiency() for n in self.neighbors + [agent]]
        efficiencies.sort()
        agent_eff = agent.get_efficiency()
        
        if len(efficiencies) <= 1:
            return 0.5
        
        rank = efficiencies.index(agent_eff)
        return rank / (len(efficiencies) - 1)


@dataclass
class CAState:
    """State of the cellular automata at a point in time."""
    agents: List['FractalAgent']
    neighborhoods: Dict[str, Neighborhood] = field(default_factory=dict)
    generation: int = 0
    global_metrics: Dict[str, float] = field(default_factory=dict)
    active_rules: Set[CARule] = field(default_factory=set)
    
    def update_neighborhoods(self, radius: int = 2) -> None:
        """Update neighborhood relationships."""
        self.neighborhoods.clear()
        
        for i, agent in enumerate(self.agents):
            # Create neighborhood with nearby agents
            start_idx = max(0, i - radius)
            end_idx = min(len(self.agents), i + radius + 1)
            
            neighbors = []
            for j in range(start_idx, end_idx):
                if j != i:  # Exclude self
                    neighbors.append(self.agents[j])
            
            neighborhood = Neighborhood(
                center_agent=agent,
                neighbors=neighbors,
                generation=self.generation
            )
            neighborhood.calculate_metrics()
            
            self.neighborhoods[agent.agent_id] = neighborhood
    
    def calculate_global_metrics(self) -> None:
        """Calculate population-wide metrics."""
        if not self.agents:
            return
        
        efficiencies = [a.get_efficiency() for a in self.agents]
        diversities = [n.diversity_score for n in self.neighborhoods.values()]
        
        self.global_metrics = {
            'population_size': len(self.agents),
            'average_efficiency': np.mean(efficiencies),
            'max_efficiency': np.max(efficiencies),
            'min_efficiency': np.min(efficiencies),
            'efficiency_variance': np.var(efficiencies),
            'average_diversity': np.mean(diversities) if diversities else 0.0,
            'generation': self.generation
        }


class BaseCARuleImplementation(ABC):
    """Base class for CA rule implementations."""
    
    def __init__(self, 
                 token_manager: TokenEconomyManager,
                 diversity_manager: GeneticDiversityManager,
                 repository: RepositoryManager):
        self.token_manager = token_manager
        self.diversity_manager = diversity_manager
        self.repository = repository
    
    @abstractmethod
    async def should_activate(self, 
                            agent: 'FractalAgent',
                            neighborhood: Neighborhood,
                            ca_state: CAState) -> bool:
        """Determine if this rule should activate for the given agent."""
        pass
    
    @abstractmethod
    async def execute(self,
                     agent: 'FractalAgent',
                     neighborhood: Neighborhood,
                     ca_state: CAState) -> Dict[str, Any]:
        """Execute the CA rule."""
        pass


class Rule110Implementation(BaseCARuleImplementation):
    """
    Rule 110: Create improved neighbors when detecting imperfections.
    
    Activated when: Agent detects suboptimal patterns in neighborhood
    Action: Creates mutated variants to explore better solutions
    Constraint: Only when token budget justifies exploration cost
    """
    
    async def should_activate(self, agent, neighborhood, ca_state) -> bool:
        # Check if agent has budget for neighbor creation
        creation_cost = 500  # Estimated cost for creating improved neighbor
        if not agent.token_budget.can_afford(creation_cost):
            return False
        
        # Check if neighborhood shows suboptimal patterns
        agent_efficiency = agent.get_efficiency()
        neighborhood_avg = neighborhood.average_efficiency
        
        # Activate if agent is performing better than neighborhood average
        # and has potential to create improvements
        if agent_efficiency > neighborhood_avg * 1.2:
            # Check for detectable imperfections
            imperfections = await self._detect_imperfections(agent, neighborhood)
            return len(imperfections) > 0
        
        return False
    
    async def execute(self, agent, neighborhood, ca_state) -> Dict[str, Any]:
        """Create improved neighbors through targeted mutations."""
        result = {
            'rule': CARule.RULE_110.value,
            'action': 'create_improved_neighbors',
            'neighbors_created': 0,
            'improvements': []
        }
        
        try:
            # Detect specific imperfections
            imperfections = await self._detect_imperfections(agent, neighborhood)
            
            # Create targeted improvements
            for imperfection in imperfections[:2]:  # Limit to 2 improvements
                if not agent.token_budget.can_afford(250):
                    break
                
                # Create improved neighbor
                improved_neighbor = await self._create_improved_neighbor(
                    agent, imperfection
                )
                
                if improved_neighbor:
                    ca_state.agents.append(improved_neighbor)
                    result['neighbors_created'] += 1
                    result['improvements'].append({
                        'imperfection_type': imperfection['type'],
                        'improvement_strategy': imperfection['solution'],
                        'neighbor_id': improved_neighbor.agent_id
                    })
                    
                    # Track token consumption
                    agent.token_budget.consume(250)
            
            logger.info(f"Rule 110: Agent {agent.agent_id} created {result['neighbors_created']} improved neighbors")
            
        except Exception as e:
            logger.error(f"Rule 110 execution failed for agent {agent.agent_id}: {e}")
            result['error'] = str(e)
        
        return result
    
    async def _detect_imperfections(self, agent, neighborhood) -> List[Dict]:
        """Detect patterns that could be improved."""
        imperfections = []
        
        # Check for efficiency gaps
        for neighbor in neighborhood.neighbors:
            efficiency_gap = agent.get_efficiency() - neighbor.get_efficiency()
            if efficiency_gap > 0.3:  # Significant efficiency difference
                imperfections.append({
                    'type': 'efficiency_gap',
                    'neighbor_id': neighbor.agent_id,
                    'gap': efficiency_gap,
                    'solution': 'transfer_efficient_patterns'
                })
        
        # Check for diversity deficits
        if neighborhood.diversity_score < 0.3:
            imperfections.append({
                'type': 'diversity_deficit',
                'score': neighborhood.diversity_score,
                'solution': 'inject_diverse_mutations'
            })
        
        # Check for pattern reuse opportunities
        agent_patterns = set(agent.emergent_patterns)
        for neighbor in neighborhood.neighbors:
            neighbor_patterns = set(neighbor.emergent_patterns)
            if len(agent_patterns - neighbor_patterns) > 2:
                imperfections.append({
                    'type': 'pattern_gap',
                    'neighbor_id': neighbor.agent_id,
                    'missing_patterns': list(agent_patterns - neighbor_patterns),
                    'solution': 'transfer_successful_patterns'
                })
        
        return imperfections
    
    async def _create_improved_neighbor(self, agent, imperfection) -> Optional['FractalAgent']:
        """Create a neighbor that addresses the detected imperfection."""
        # Import here to avoid circular dependency
        from .fractal_agent import FractalAgent
        
        # Clone the agent's genome for modification
        improved_genome = agent.genome.clone()
        improved_genome.generation += 1
        improved_genome.parent_ids = [agent.genome.calculate_hash()]
        
        # Apply improvement based on imperfection type
        if imperfection['type'] == 'efficiency_gap':
            # Enhance with efficiency-boosting mutations
            improved_genome = self.diversity_manager.mutation_engine.mutate(
                improved_genome, force_mutation=True
            )
            
        elif imperfection['type'] == 'diversity_deficit':
            # Apply diversity-increasing mutations
            improved_genome = self.diversity_manager.mutation_engine.hypermutate(
                improved_genome, intensity=0.6
            )
            
        elif imperfection['type'] == 'pattern_gap':
            # Transfer successful patterns
            for pattern_id in imperfection['missing_patterns']:
                # Add pattern transfer logic here
                pass
        
        # Create new agent with improved genome
        improved_agent = FractalAgent(
            agent_id=f"{agent.agent_id}_improved_{len(agent.children)}",
            genome=improved_genome,
            level=agent.level,
            parent_id=agent.agent_id,
            token_budget=TokenBudget(total=1000),  # Allocate initial budget
            worktree_manager=agent.worktree_manager
        )
        
        # Update parent's children list
        agent.children.append(improved_agent.agent_id)
        
        return improved_agent


class Rule30Implementation(BaseCARuleImplementation):
    """
    Rule 30: Fork into parallel worktrees when bottlenecked.
    
    Activated when: Agent encounters resource or computational bottlenecks
    Action: Creates parallel execution paths in separate worktrees
    Constraint: Only if economic benefit justifies parallel execution cost
    """
    
    async def should_activate(self, agent, neighborhood, ca_state) -> bool:
        # Check for bottleneck conditions
        is_bottlenecked = await self._detect_bottleneck(agent, neighborhood)
        
        if not is_bottlenecked:
            return False
        
        # Check if parallel execution would be beneficial
        parallel_cost = 1000  # Cost of creating and managing parallel worktree
        expected_benefit = await self._estimate_parallel_benefit(agent)
        
        # Only activate if ROI is positive and we have budget
        roi = expected_benefit / parallel_cost if parallel_cost > 0 else 0
        return roi > 1.2 and agent.token_budget.can_afford(parallel_cost)
    
    async def execute(self, agent, neighborhood, ca_state) -> Dict[str, Any]:
        """Fork agent into parallel worktrees."""
        result = {
            'rule': CARule.RULE_30.value,
            'action': 'fork_parallel_worktrees',
            'forks_created': 0,
            'worktree_paths': []
        }
        
        try:
            # Determine optimal number of forks
            max_forks = min(3, agent.token_budget.available // 1000)
            
            for fork_id in range(max_forks):
                if not agent.token_budget.can_afford(1000):
                    break
                
                # Create parallel worktree
                fork_agent = await self._create_parallel_fork(agent, fork_id)
                
                if fork_agent:
                    ca_state.agents.append(fork_agent)
                    result['forks_created'] += 1
                    result['worktree_paths'].append(fork_agent.worktree_path)
                    
                    # Consume tokens for fork creation
                    agent.token_budget.consume(1000)
            
            logger.info(f"Rule 30: Agent {agent.agent_id} created {result['forks_created']} parallel forks")
            
        except Exception as e:
            logger.error(f"Rule 30 execution failed for agent {agent.agent_id}: {e}")
            result['error'] = str(e)
        
        return result
    
    async def _detect_bottleneck(self, agent, neighborhood) -> bool:
        """Detect if agent is experiencing bottlenecks."""
        # Check efficiency trend
        if hasattr(agent, 'efficiency_history') and len(agent.efficiency_history) >= 3:
            recent = agent.efficiency_history[-3:]
            if all(recent[i] <= recent[i-1] for i in range(1, len(recent))):
                return True  # Declining efficiency trend
        
        # Check if agent is significantly underperforming neighborhood
        if agent.get_efficiency() < neighborhood.average_efficiency * 0.8:
            return True
        
        # Check for resource constraints
        if agent.token_budget.utilization_rate > 0.9:
            return True  # High token utilization suggests resource constraint
        
        return False
    
    async def _estimate_parallel_benefit(self, agent) -> float:
        """Estimate benefit of parallel execution."""
        # Simple heuristic based on agent's potential
        base_efficiency = agent.get_efficiency()
        parallelization_factor = 1.5  # Assume 50% improvement from parallelization
        
        # Estimate value generation over time horizon
        time_horizon_tokens = 500  # Expected tokens consumed over analysis period
        estimated_value = base_efficiency * parallelization_factor * time_horizon_tokens
        
        return estimated_value
    
    async def _create_parallel_fork(self, agent, fork_id: int):
        """Create a parallel fork of the agent."""
        from .fractal_agent import FractalAgent
        
        # Create genetic variant for exploration
        fork_genome = agent.genome.clone()
        fork_genome.generation += 1
        fork_genome.parent_ids = [agent.genome.calculate_hash()]
        
        # Apply light mutations for exploration diversity
        fork_genome = self.diversity_manager.mutation_engine.mutate(
            fork_genome, force_mutation=True
        )
        
        # Create fork agent
        fork_agent = FractalAgent(
            agent_id=f"{agent.agent_id}_fork_{fork_id}",
            genome=fork_genome,
            level=agent.level,
            parent_id=agent.agent_id,
            token_budget=TokenBudget(total=800),  # Smaller budget for forks
            worktree_manager=agent.worktree_manager
        )
        
        # Create isolated worktree for the fork
        if agent.worktree_manager:
            fork_worktree = await agent.worktree_manager.create_worktree(
                branch_name=f"fork_{fork_agent.agent_id}",
                agent_id=fork_agent.agent_id,
                token_limit=800
            )
            fork_agent.worktree_path = str(fork_worktree)
        
        agent.children.append(fork_agent.agent_id)
        return fork_agent


class Rule90Implementation(BaseCARuleImplementation):
    """
    Rule 90: Abstract patterns into reusable components.
    
    Activated when: Agent discovers effective patterns worth abstracting
    Action: Extracts patterns into reusable components for the pattern library
    Constraint: Only abstract patterns with proven token efficiency gains
    """
    
    async def should_activate(self, agent, neighborhood, ca_state) -> bool:
        # Check if agent has discovered valuable patterns
        if not agent.emergent_patterns:
            return False
        
        # Check if patterns have sufficient effectiveness to warrant abstraction
        pattern_effectiveness = await self._evaluate_pattern_effectiveness(agent)
        
        # Activate if we have effective patterns and budget for abstraction
        abstraction_cost = 200
        return (pattern_effectiveness > 1.5 and 
                agent.token_budget.can_afford(abstraction_cost))
    
    async def execute(self, agent, neighborhood, ca_state) -> Dict[str, Any]:
        """Abstract effective patterns into reusable components."""
        result = {
            'rule': CARule.RULE_90.value,
            'action': 'abstract_patterns',
            'patterns_abstracted': 0,
            'components_created': []
        }
        
        try:
            # Identify patterns worth abstracting
            valuable_patterns = await self._identify_valuable_patterns(agent)
            
            for pattern_id in valuable_patterns:
                if not agent.token_budget.can_afford(200):
                    break
                
                # Create abstracted component
                component = await self._create_abstracted_component(pattern_id, agent)
                
                if component:
                    # Store in pattern catalog
                    await self._store_pattern_component(component)
                    
                    result['patterns_abstracted'] += 1
                    result['components_created'].append({
                        'component_id': component['id'],
                        'pattern_source': pattern_id,
                        'reusability_score': component['reusability']
                    })
                    
                    # Consume tokens for abstraction
                    agent.token_budget.consume(200)
            
            logger.info(f"Rule 90: Agent {agent.agent_id} abstracted {result['patterns_abstracted']} patterns")
            
        except Exception as e:
            logger.error(f"Rule 90 execution failed for agent {agent.agent_id}: {e}")
            result['error'] = str(e)
        
        return result
    
    async def _evaluate_pattern_effectiveness(self, agent) -> float:
        """Evaluate the overall effectiveness of agent's patterns."""
        if not agent.emergent_patterns:
            return 0.0
        
        # Simple heuristic: effectiveness correlates with agent efficiency
        # In practice, this would analyze specific pattern contributions
        return agent.get_efficiency()
    
    async def _identify_valuable_patterns(self, agent) -> List[str]:
        """Identify patterns worth abstracting."""
        # For now, return all patterns that contributed to success
        # In practice, this would analyze pattern usage and effectiveness
        
        if agent.get_efficiency() > 1.0:
            return agent.emergent_patterns[:3]  # Top 3 patterns
        
        return []
    
    async def _create_abstracted_component(self, pattern_id: str, agent) -> Optional[Dict]:
        """Create an abstracted reusable component from a pattern."""
        # Extract pattern details (mock implementation)
        component = {
            'id': f"component_{pattern_id}_{agent.agent_id}",
            'source_pattern': pattern_id,
            'source_agent': agent.agent_id,
            'abstraction_level': 'medium',
            'reusability': 0.8,
            'effectiveness': agent.get_efficiency(),
            'token_efficiency': agent.get_efficiency(),
            'description': f"Abstracted component from pattern {pattern_id}",
            'component_type': 'strategy',
            'parameters': {
                'adaptable': True,
                'context_independent': True
            },
            'created_at': datetime.now()
        }
        
        return component
    
    async def _store_pattern_component(self, component: Dict) -> None:
        """Store the abstracted component in the pattern library."""
        # Store in repository for future use
        if self.repository:
            # Convert to pattern format for storage
            pattern_data = {
                'pattern_hash': component['id'],
                'pattern_type': 'abstracted_component',
                'description': component['description'],
                'effectiveness_score': component['effectiveness'],
                'token_efficiency': component['token_efficiency'],
                'confidence_score': component['reusability'],
                'metadata': component
            }
            
            self.repository.metrics_db.insert_pattern(pattern_data)


class Rule184Implementation(BaseCARuleImplementation):
    """
    Rule 184: Learn from higher-performing neighbors.
    
    Activated when: Agent detects neighbors with superior token efficiency
    Action: Adopts successful strategies from high-performing neighbors
    Constraint: Only learn from neighbors with significantly better value-per-token
    """
    
    async def should_activate(self, agent, neighborhood, ca_state) -> bool:
        # Check if there are higher-performing neighbors
        best_neighbor = neighborhood.get_best_neighbor()
        
        if not best_neighbor:
            return False
        
        # Only activate if neighbor is significantly better
        efficiency_threshold = agent.get_efficiency() * 1.3  # 30% better
        learning_cost = 300
        
        return (best_neighbor.get_efficiency() > efficiency_threshold and
                agent.token_budget.can_afford(learning_cost))
    
    async def execute(self, agent, neighborhood, ca_state) -> Dict[str, Any]:
        """Learn from higher-performing neighbors."""
        result = {
            'rule': CARule.RULE_184.value,
            'action': 'learn_from_neighbors',
            'patterns_learned': 0,
            'efficiency_gain': 0.0
        }
        
        try:
            # Identify best neighbors to learn from
            teachers = await self._identify_teachers(agent, neighborhood)
            
            initial_efficiency = agent.get_efficiency()
            
            for teacher in teachers:
                if not agent.token_budget.can_afford(300):
                    break
                
                # Learn from teacher
                learned_patterns = await self._learn_from_neighbor(agent, teacher)
                
                result['patterns_learned'] += len(learned_patterns)
                
                # Consume tokens for learning
                agent.token_budget.consume(300)
            
            # Calculate efficiency gain
            final_efficiency = agent.get_efficiency()
            result['efficiency_gain'] = final_efficiency - initial_efficiency
            
            logger.info(f"Rule 184: Agent {agent.agent_id} learned {result['patterns_learned']} patterns")
            
        except Exception as e:
            logger.error(f"Rule 184 execution failed for agent {agent.agent_id}: {e}")
            result['error'] = str(e)
        
        return result
    
    async def _identify_teachers(self, agent, neighborhood) -> List:
        """Identify neighbors worth learning from."""
        teachers = []
        agent_efficiency = agent.get_efficiency()
        
        for neighbor in neighborhood.neighbors:
            neighbor_efficiency = neighbor.get_efficiency()
            
            # Select neighbors with significantly better performance
            if neighbor_efficiency > agent_efficiency * 1.2:
                teachers.append({
                    'agent': neighbor,
                    'efficiency_ratio': neighbor_efficiency / agent_efficiency,
                    'patterns': neighbor.emergent_patterns
                })
        
        # Sort by efficiency ratio
        teachers.sort(key=lambda t: t['efficiency_ratio'], reverse=True)
        
        return teachers[:2]  # Learn from top 2 teachers
    
    async def _learn_from_neighbor(self, student, teacher_info) -> List[str]:
        """Learn patterns from a neighboring agent."""
        teacher = teacher_info['agent']
        learned_patterns = []
        
        # Identify patterns to learn
        student_patterns = set(student.emergent_patterns)
        teacher_patterns = set(teacher.emergent_patterns)
        
        # Learn patterns the student doesn't have
        novel_patterns = teacher_patterns - student_patterns
        
        for pattern in list(novel_patterns)[:2]:  # Learn up to 2 patterns
            # Adapt pattern to student's context
            adapted_pattern = await self._adapt_pattern(pattern, student, teacher)
            
            if adapted_pattern:
                student.emergent_patterns.append(adapted_pattern)
                learned_patterns.append(adapted_pattern)
                
                # Update student's genome with learned knowledge
                self._integrate_learned_pattern(student, adapted_pattern)
        
        return learned_patterns
    
    async def _adapt_pattern(self, pattern: str, student, teacher) -> Optional[str]:
        """Adapt a pattern from teacher to student's context."""
        # Simple pattern adaptation (in practice, this would be more sophisticated)
        adapted_pattern = f"{pattern}_adapted_from_{teacher.agent_id}"
        
        return adapted_pattern
    
    def _integrate_learned_pattern(self, agent, pattern: str) -> None:
        """Integrate learned pattern into agent's genome."""
        # Create a learning record in the genome
        from ..diversity import Gene, GeneType
        
        learning_gene = Gene(
            gene_type=GeneType.OPTIMIZATION_HINT,
            name=f"learned_pattern_{len(agent.genome.genes)}",
            value=pattern,
            metadata={'learned_from_neighbor': True, 'pattern': pattern}
        )
        
        agent.genome.add_gene(learning_gene)


class Rule1Implementation(BaseCARuleImplementation):
    """
    Rule 1: Recurse to higher abstraction levels when optimal.
    
    Activated when: Agent achieves local optimization and can benefit from meta-level thinking
    Action: Creates higher-level agent that operates on meta-strategies
    Constraint: Only recurse when token ROI of meta-level agent exceeds threshold
    """
    
    async def should_activate(self, agent, neighborhood, ca_state) -> bool:
        # Check if agent has achieved local optimization
        is_locally_optimal = await self._check_local_optimization(agent, neighborhood)
        
        if not is_locally_optimal:
            return False
        
        # Check if meta-level thinking would be beneficial
        meta_cost = 2000  # High cost for meta-level agent
        expected_meta_benefit = await self._estimate_meta_benefit(agent, ca_state)
        
        roi = expected_meta_benefit / meta_cost if meta_cost > 0 else 0
        roi_threshold = 2.0  # High threshold for meta-level recursion
        
        return roi > roi_threshold and agent.token_budget.can_afford(meta_cost)
    
    async def execute(self, agent, neighborhood, ca_state) -> Dict[str, Any]:
        """Create higher-abstraction level agent."""
        result = {
            'rule': CARule.RULE_1.value,
            'action': 'recurse_to_meta_level',
            'meta_agent_created': False,
            'abstraction_level': agent.level + 1
        }
        
        try:
            # Create meta-level agent
            meta_agent = await self._create_meta_agent(agent, ca_state)
            
            if meta_agent:
                ca_state.agents.append(meta_agent)
                result['meta_agent_created'] = True
                result['meta_agent_id'] = meta_agent.agent_id
                
                # Consume tokens for meta-level creation
                agent.token_budget.consume(2000)
                
                logger.info(f"Rule 1: Agent {agent.agent_id} created meta-level agent {meta_agent.agent_id}")
            
        except Exception as e:
            logger.error(f"Rule 1 execution failed for agent {agent.agent_id}: {e}")
            result['error'] = str(e)
        
        return result
    
    async def _check_local_optimization(self, agent, neighborhood) -> bool:
        """Check if agent has achieved local optimization."""
        # Agent is locally optimal if it's the best in neighborhood
        # and its efficiency has plateaued
        
        if not neighborhood.neighbors:
            return False
        
        # Check if agent is best in neighborhood
        is_best = all(agent.get_efficiency() >= n.get_efficiency() 
                     for n in neighborhood.neighbors)
        
        # Check efficiency plateau
        has_plateaued = False
        if hasattr(agent, 'efficiency_history') and len(agent.efficiency_history) >= 5:
            recent = agent.efficiency_history[-5:]
            variance = np.var(recent)
            has_plateaued = variance < 0.01  # Low variance indicates plateau
        
        return is_best and has_plateaued
    
    async def _estimate_meta_benefit(self, agent, ca_state) -> float:
        """Estimate benefit of creating meta-level agent."""
        # Meta-agent can optimize population-level strategies
        population_size = len(ca_state.agents)
        average_efficiency = ca_state.global_metrics.get('average_efficiency', 1.0)
        
        # Estimate potential improvement from meta-optimization
        meta_improvement_factor = 1.3  # 30% improvement potential
        time_horizon = 1000  # Tokens over which benefit is realized
        
        population_benefit = (population_size * average_efficiency * 
                            (meta_improvement_factor - 1) * time_horizon)
        
        return population_benefit
    
    async def _create_meta_agent(self, base_agent, ca_state):
        """Create a meta-level agent."""
        from .fractal_agent import FractalAgent
        
        # Create meta-genome that focuses on population-level optimization
        meta_genome = base_agent.genome.clone()
        meta_genome.generation += 1
        meta_genome.parent_ids = [base_agent.genome.calculate_hash()]
        
        # Add meta-level genes
        from ..diversity import Gene, GeneType
        
        meta_strategy_gene = Gene(
            gene_type=GeneType.STRATEGY,
            name="meta_optimization",
            value={
                'level': 'meta',
                'focus': 'population_optimization',
                'strategies': ['diversity_balancing', 'efficiency_maximization', 'pattern_synthesis']
            },
            metadata={'meta_level': True}
        )
        
        meta_genome.add_gene(meta_strategy_gene)
        
        # Create meta-agent
        meta_agent = FractalAgent(
            agent_id=f"{base_agent.agent_id}_meta",
            genome=meta_genome,
            level=base_agent.level + 1,
            parent_id=base_agent.agent_id,
            token_budget=TokenBudget(total=3000),  # Larger budget for meta operations
            worktree_manager=base_agent.worktree_manager
        )
        
        base_agent.children.append(meta_agent.agent_id)
        return meta_agent


class CellularAutomataEngine:
    """
    Engine that orchestrates the cellular automata evolution process.
    
    Manages the application of CA rules with economic constraints,
    diversity maintenance, and emergent behavior capture.
    """
    
    def __init__(self,
                 token_manager: TokenEconomyManager,
                 diversity_manager: GeneticDiversityManager,
                 repository: RepositoryManager,
                 behavior_monitor: EmergentBehaviorMonitor):
        """Initialize the CA engine with required managers."""
        self.token_manager = token_manager
        self.diversity_manager = diversity_manager
        self.repository = repository
        self.behavior_monitor = behavior_monitor
        
        # Initialize rule implementations
        self.rules = {
            CARule.RULE_110: Rule110Implementation(token_manager, diversity_manager, repository),
            CARule.RULE_30: Rule30Implementation(token_manager, diversity_manager, repository),
            CARule.RULE_90: Rule90Implementation(token_manager, diversity_manager, repository),
            CARule.RULE_184: Rule184Implementation(token_manager, diversity_manager, repository),
            CARule.RULE_1: Rule1Implementation(token_manager, diversity_manager, repository),
        }
        
        # Evolution state
        self.ca_state = CAState(agents=[])
        self.evolution_history: List[Dict] = []
    
    async def step(self, agents: List['FractalAgent']) -> Dict[str, Any]:
        """Execute one step of cellular automata evolution."""
        # Update CA state
        self.ca_state.agents = agents
        self.ca_state.generation += 1
        self.ca_state.update_neighborhoods()
        self.ca_state.calculate_global_metrics()
        
        step_result = {
            'generation': self.ca_state.generation,
            'initial_population': len(agents),
            'rule_activations': {},
            'global_metrics': self.ca_state.global_metrics.copy(),
            'agents_created': 0,
            'patterns_discovered': 0
        }
        
        # Apply diversity enforcement first
        genomes = [agent.genome for agent in agents if agent.genome]
        if genomes:
            enforced_genomes = self.diversity_manager.enforce_diversity(genomes)
            # Update agents with enforced genomes
            for i, agent in enumerate(agents):
                if i < len(enforced_genomes) and agent.genome:
                    agent.genome = enforced_genomes[i]
        
        # Apply CA rules to each agent
        for agent in agents[:]:  # Copy list since it may be modified
            if not agent.token_budget.can_afford(50):  # Minimum cost for rule evaluation
                continue
            
            neighborhood = self.ca_state.neighborhoods.get(agent.agent_id)
            if not neighborhood:
                continue
            
            # Check each rule for activation
            for rule_type, rule_impl in self.rules.items():
                try:
                    if await rule_impl.should_activate(agent, neighborhood, self.ca_state):
                        # Execute the rule
                        rule_result = await rule_impl.execute(agent, neighborhood, self.ca_state)
                        
                        # Track rule activation
                        rule_name = rule_type.value
                        if rule_name not in step_result['rule_activations']:
                            step_result['rule_activations'][rule_name] = []
                        
                        step_result['rule_activations'][rule_name].append({
                            'agent_id': agent.agent_id,
                            'result': rule_result
                        })
                        
                        # Update counters
                        if 'neighbors_created' in rule_result:
                            step_result['agents_created'] += rule_result['neighbors_created']
                        if 'forks_created' in rule_result:
                            step_result['agents_created'] += rule_result['forks_created']
                        if 'patterns_abstracted' in rule_result:
                            step_result['patterns_discovered'] += rule_result['patterns_abstracted']
                        
                        # Consume tokens for rule evaluation
                        agent.token_budget.consume(50)
                        
                        # Limit one rule activation per agent per step
                        break
                        
                except Exception as e:
                    logger.error(f"Error applying {rule_type.value} to agent {agent.agent_id}: {e}")
        
        # Update final population count
        step_result['final_population'] = len(self.ca_state.agents)
        
        # Record evolution step
        self.evolution_history.append(step_result)
        
        # Store metrics in repository
        if self.repository:
            await self._store_evolution_metrics(step_result)
        
        logger.info(f"CA Step {self.ca_state.generation}: "
                   f"{step_result['initial_population']} → {step_result['final_population']} agents, "
                   f"{len(step_result['rule_activations'])} rules activated")
        
        return step_result
    
    async def _store_evolution_metrics(self, step_result: Dict) -> None:
        """Store evolution metrics in the repository."""
        try:
            # Store global metrics
            metrics = {
                'generation': step_result['generation'],
                'population_size': step_result['final_population'],
                'rule_activations': len(step_result['rule_activations']),
                'agents_created': step_result['agents_created'],
                'patterns_discovered': step_result['patterns_discovered'],
                **step_result['global_metrics']
            }
            
            # Record in repository (simplified)
            # In practice, this would use proper database records
            
        except Exception as e:
            logger.error(f"Error storing evolution metrics: {e}")
    
    def get_population_state(self) -> CAState:
        """Get current CA state."""
        return self.ca_state
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of evolution process."""
        if not self.evolution_history:
            return {}
        
        total_activations = sum(
            len(step['rule_activations']) 
            for step in self.evolution_history
        )
        
        total_agents_created = sum(
            step['agents_created'] 
            for step in self.evolution_history
        )
        
        return {
            'total_generations': len(self.evolution_history),
            'total_rule_activations': total_activations,
            'total_agents_created': total_agents_created,
            'current_population': len(self.ca_state.agents),
            'final_metrics': self.ca_state.global_metrics
        }