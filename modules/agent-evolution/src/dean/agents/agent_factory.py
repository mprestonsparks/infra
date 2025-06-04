"""
Agent factory for creating and configuring FractalAgent instances.

Handles dependency injection, configuration, and initialization
of agents with proper cellular automata and economic constraints.
"""

import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import logging

from .fractal_agent import FractalAgent, AgentState
from .worktree_manager import GitWorktreeManager, WorktreeConstraints
from .cellular_automata import CellularAutomataEngine
from ..diversity import AgentGenome, GeneticDiversityManager, Gene, GeneType
from ..economy import TokenBudget, TokenEconomyManager
from ..patterns import PatternDetector, EmergentBehaviorMonitor
from ..repository import RepositoryManager

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for creating a new agent."""
    agent_id: Optional[str] = None
    level: int = 0
    parent_id: Optional[str] = None
    token_budget: int = 1000
    initial_genes: List[Dict[str, Any]] = field(default_factory=list)
    worktree_enabled: bool = True
    worktree_constraints: Optional[WorktreeConstraints] = None
    custom_genome: Optional[AgentGenome] = None
    
    def __post_init__(self):
        """Generate agent ID if not provided."""
        if self.agent_id is None:
            self.agent_id = f"agent_{uuid.uuid4().hex[:8]}"


class AgentFactory:
    """
    Factory for creating properly configured FractalAgent instances.
    
    Handles dependency injection and ensures all agents are created
    with consistent configuration and required components.
    """
    
    def __init__(self,
                 token_manager: TokenEconomyManager,
                 diversity_manager: GeneticDiversityManager,
                 ca_engine: CellularAutomataEngine,
                 pattern_detector: PatternDetector,
                 behavior_monitor: EmergentBehaviorMonitor,
                 repository: RepositoryManager,
                 worktree_manager: Optional[GitWorktreeManager] = None,
                 base_repo_path: Optional[Path] = None):
        """
        Initialize agent factory with required dependencies.
        
        Args:
            token_manager: Manages token economics
            diversity_manager: Handles genetic diversity
            ca_engine: Cellular automata evolution engine
            pattern_detector: Detects emergent patterns
            behavior_monitor: Monitors agent behaviors
            repository: Knowledge repository
            worktree_manager: Optional git worktree manager
            base_repo_path: Base path for creating worktree manager
        """
        self.token_manager = token_manager
        self.diversity_manager = diversity_manager
        self.ca_engine = ca_engine
        self.pattern_detector = pattern_detector
        self.behavior_monitor = behavior_monitor
        self.repository = repository
        
        # Set up worktree manager
        if worktree_manager:
            self.worktree_manager = worktree_manager
        elif base_repo_path:
            self.worktree_manager = GitWorktreeManager(base_repo_path=base_repo_path)
        else:
            self.worktree_manager = None
            logger.warning("No worktree manager configured - agents will not have isolated environments")
        
        # Factory statistics
        self.agents_created = 0
        self.creation_history: List[Dict[str, Any]] = []
        
        logger.info("AgentFactory initialized with all dependencies")
    
    async def create_agent(self, config: AgentConfig) -> FractalAgent:
        """
        Create a new FractalAgent with the specified configuration.
        
        Args:
            config: Agent configuration
            
        Returns:
            Fully initialized FractalAgent instance
            
        Raises:
            RuntimeError: If agent creation fails
        """
        try:
            # Generate or validate agent ID
            agent_id = config.agent_id or f"agent_{uuid.uuid4().hex[:8]}"
            
            # Create genome
            genome = config.custom_genome or await self._create_genome(config)
            
            # Create token budget
            token_budget = await self._create_token_budget(agent_id, config.token_budget)
            
            # Create agent instance
            agent = FractalAgent(
                agent_id=agent_id,
                genome=genome,
                level=config.level,
                parent_id=config.parent_id,
                token_budget=token_budget,
                worktree_manager=self.worktree_manager
            )
            
            # Initialize agent with dependencies
            await agent.initialize(
                ca_engine=self.ca_engine,
                diversity_manager=self.diversity_manager,
                pattern_detector=self.pattern_detector,
                behavior_monitor=self.behavior_monitor,
                repository=self.repository
            )
            
            # Track creation
            self.agents_created += 1
            self.creation_history.append({
                'agent_id': agent_id,
                'level': config.level,
                'parent_id': config.parent_id,
                'token_budget': config.token_budget,
                'creation_time': agent._last_evolution_time.isoformat(),
                'genome_hash': genome.calculate_hash()
            })
            
            # Keep only recent history
            if len(self.creation_history) > 100:
                self.creation_history = self.creation_history[-100:]
            
            logger.info(f"Created FractalAgent {agent_id} at level {config.level}")
            return agent
            
        except Exception as e:
            logger.error(f"Failed to create agent with config {config}: {e}")
            raise RuntimeError(f"Agent creation failed: {e}")
    
    async def create_population(self, 
                               population_size: int,
                               base_config: AgentConfig,
                               diversity_factor: float = 0.3) -> List[FractalAgent]:
        """
        Create a diverse population of agents.
        
        Args:
            population_size: Number of agents to create
            base_config: Base configuration (will be varied for diversity)
            diversity_factor: Amount of genetic diversity to inject (0.0 to 1.0)
            
        Returns:
            List of initialized FractalAgent instances
        """
        if population_size <= 0:
            raise ValueError("Population size must be positive")
        
        agents = []
        
        try:
            for i in range(population_size):
                # Create varied configuration
                agent_config = self._create_diverse_config(base_config, i, diversity_factor)
                
                # Create agent
                agent = await self.create_agent(agent_config)
                agents.append(agent)
                
                logger.debug(f"Created agent {i+1}/{population_size}: {agent.agent_id}")
            
            # Ensure population diversity
            await self._ensure_population_diversity(agents)
            
            logger.info(f"Created diverse population of {len(agents)} agents")
            return agents
            
        except Exception as e:
            # Cleanup any created agents on failure
            for agent in agents:
                try:
                    await agent.cleanup()
                except:
                    pass
            
            logger.error(f"Failed to create population: {e}")
            raise RuntimeError(f"Population creation failed: {e}")
    
    async def create_child_agent(self, 
                                parent: FractalAgent,
                                mutation_intensity: float = 0.1,
                                token_budget: Optional[int] = None) -> FractalAgent:
        """
        Create a child agent derived from a parent.
        
        Args:
            parent: Parent agent to derive from
            mutation_intensity: Intensity of genetic mutations (0.0 to 1.0)
            token_budget: Token budget for child (defaults to parent's remaining budget)
            
        Returns:
            Child FractalAgent instance
        """
        try:
            # Create child genome with mutations
            child_genome = parent.genome.clone()
            child_genome.generation += 1
            child_genome.parent_ids = [parent.genome.calculate_hash()]
            
            # Apply mutations
            if mutation_intensity > 0:
                child_genome = self.diversity_manager.mutation_engine.mutate(
                    child_genome, 
                    force_mutation=True
                )
            
            # Configure child
            child_config = AgentConfig(
                agent_id=f"{parent.agent_id}_child_{len(parent.children)}",
                level=parent.level,
                parent_id=parent.agent_id,
                token_budget=token_budget or min(parent.token_budget.available, 500),
                custom_genome=child_genome,
                worktree_enabled=True
            )
            
            # Create child agent
            child = await self.create_agent(child_config)
            
            # Update parent's children list
            parent.children.append(child.agent_id)
            
            logger.info(f"Created child agent {child.agent_id} from parent {parent.agent_id}")
            return child
            
        except Exception as e:
            logger.error(f"Failed to create child agent from {parent.agent_id}: {e}")
            raise RuntimeError(f"Child agent creation failed: {e}")
    
    async def create_meta_agent(self,
                               base_agents: List[FractalAgent],
                               meta_level: int = 1,
                               token_budget: int = 3000) -> FractalAgent:
        """
        Create a meta-level agent that operates on other agents.
        
        Args:
            base_agents: Base agents that the meta-agent will manage
            meta_level: Meta-abstraction level
            token_budget: Token budget for meta-operations
            
        Returns:
            Meta-level FractalAgent instance
        """
        try:
            # Create meta-genome by combining knowledge from base agents
            meta_genome = await self._create_meta_genome(base_agents, meta_level)
            
            # Configure meta-agent
            meta_config = AgentConfig(
                agent_id=f"meta_level_{meta_level}_{uuid.uuid4().hex[:8]}",
                level=meta_level,
                token_budget=token_budget,
                custom_genome=meta_genome,
                worktree_enabled=True
            )
            
            # Create meta-agent
            meta_agent = await self.create_agent(meta_config)
            
            logger.info(f"Created meta-agent {meta_agent.agent_id} at level {meta_level}")
            return meta_agent
            
        except Exception as e:
            logger.error(f"Failed to create meta-agent: {e}")
            raise RuntimeError(f"Meta-agent creation failed: {e}")
    
    def get_factory_stats(self) -> Dict[str, Any]:
        """Get factory statistics and creation history."""
        return {
            'agents_created': self.agents_created,
            'recent_creations': self.creation_history[-10:] if self.creation_history else [],
            'creation_history_length': len(self.creation_history),
            'worktree_manager_available': self.worktree_manager is not None,
            'active_worktrees': len(self.worktree_manager.active_worktrees) if self.worktree_manager else 0
        }
    
    async def _create_genome(self, config: AgentConfig) -> AgentGenome:
        """Create a genome based on configuration."""
        genome = AgentGenome(generation=0)
        
        # Add default genes
        default_genes = [
            Gene(
                gene_type=GeneType.STRATEGY,
                name="exploration_strategy",
                value={"approach": "balanced", "exploration_rate": 0.3}
            ),
            Gene(
                gene_type=GeneType.HYPERPARAMETER,
                name="learning_rate",
                value=0.01
            ),
            Gene(
                gene_type=GeneType.OPTIMIZATION_HINT,
                name="efficiency_focus",
                value="token_optimization"
            )
        ]
        
        for gene in default_genes:
            genome.add_gene(gene)
        
        # Add custom genes from config
        for gene_data in config.initial_genes:
            gene = Gene(
                gene_type=GeneType(gene_data.get('type', 'strategy')),
                name=gene_data.get('name', f'custom_gene_{len(genome.genes)}'),
                value=gene_data.get('value', {}),
                metadata=gene_data.get('metadata', {})
            )
            genome.add_gene(gene)
        
        return genome
    
    async def _create_token_budget(self, agent_id: str, budget_amount: int) -> TokenBudget:
        """Create and allocate token budget for an agent."""
        # Allocate budget through token manager
        allocated_amount = self.token_manager.allocate_tokens(agent_id, historical_efficiency=None)
        
        return TokenBudget(total=allocated_amount)
    
    def _create_diverse_config(self, 
                              base_config: AgentConfig, 
                              index: int, 
                              diversity_factor: float) -> AgentConfig:
        """Create a configuration with diversity variations."""
        import random
        
        # Vary token budget
        budget_variation = int(base_config.token_budget * diversity_factor * random.uniform(-0.5, 0.5))
        varied_budget = max(100, base_config.token_budget + budget_variation)
        
        # Create diverse initial genes
        diverse_genes = []
        
        # Add some variation to exploration strategy
        exploration_rate = 0.3 + (diversity_factor * random.uniform(-0.2, 0.2))
        diverse_genes.append({
            'type': 'strategy',
            'name': 'exploration_strategy',
            'value': {
                'approach': random.choice(['aggressive', 'balanced', 'conservative']),
                'exploration_rate': max(0.1, min(0.9, exploration_rate))
            }
        })
        
        # Vary learning rate
        learning_rate = 0.01 + (diversity_factor * random.uniform(-0.005, 0.005))
        diverse_genes.append({
            'type': 'hyperparameter',
            'name': 'learning_rate',
            'value': max(0.001, min(0.1, learning_rate))
        })
        
        # Add specialized genes based on index
        if index % 3 == 0:
            diverse_genes.append({
                'type': 'optimization_hint',
                'name': 'specialization',
                'value': 'efficiency_focused'
            })
        elif index % 3 == 1:
            diverse_genes.append({
                'type': 'optimization_hint',
                'name': 'specialization',
                'value': 'exploration_focused'
            })
        else:
            diverse_genes.append({
                'type': 'optimization_hint',
                'name': 'specialization',
                'value': 'pattern_focused'
            })
        
        return AgentConfig(
            agent_id=f"{base_config.agent_id}_{index}" if base_config.agent_id else None,
            level=base_config.level,
            parent_id=base_config.parent_id,
            token_budget=varied_budget,
            initial_genes=diverse_genes,
            worktree_enabled=base_config.worktree_enabled,
            worktree_constraints=base_config.worktree_constraints
        )
    
    async def _ensure_population_diversity(self, agents: List[FractalAgent]) -> None:
        """Ensure the created population meets diversity requirements."""
        if len(agents) < 2:
            return
        
        # Calculate diversity score
        genomes = [agent.genome for agent in agents]
        diversity_metrics = self.diversity_manager.calculate_diversity_metrics(genomes)
        
        if diversity_metrics.overall_diversity < self.diversity_manager.min_diversity:
            # Apply additional mutations to increase diversity
            logger.info(f"Population diversity too low ({diversity_metrics.overall_diversity:.3f}), applying additional mutations")
            
            for i, agent in enumerate(agents):
                if i % 2 == 0:  # Mutate every other agent
                    mutated_genome = self.diversity_manager.mutation_engine.mutate(
                        agent.genome, force_mutation=True
                    )
                    agent.genome = mutated_genome
                    
                    logger.debug(f"Applied diversity mutation to agent {agent.agent_id}")
    
    async def _create_meta_genome(self, 
                                 base_agents: List[FractalAgent], 
                                 meta_level: int) -> AgentGenome:
        """Create a meta-genome that combines knowledge from base agents."""
        meta_genome = AgentGenome(generation=0)
        
        # Add meta-level strategy genes
        meta_strategy = Gene(
            gene_type=GeneType.STRATEGY,
            name="meta_optimization",
            value={
                "level": meta_level,
                "focus": "population_optimization",
                "strategies": ["diversity_balancing", "efficiency_maximization", "pattern_synthesis"],
                "base_agent_count": len(base_agents)
            },
            metadata={"meta_level": True}
        )
        meta_genome.add_gene(meta_strategy)
        
        # Extract successful patterns from base agents
        successful_patterns = []
        for agent in base_agents:
            if agent.get_efficiency() > 1.0:  # Above-average performers
                for pattern in agent.emergent_patterns:
                    if pattern not in successful_patterns:
                        successful_patterns.append(pattern)
        
        # Add pattern synthesis gene
        if successful_patterns:
            pattern_gene = Gene(
                gene_type=GeneType.OPTIMIZATION_HINT,
                name="pattern_synthesis",
                value={
                    "successful_patterns": successful_patterns[:10],  # Top 10
                    "synthesis_strategy": "meta_combination"
                },
                metadata={"derived_from_population": True}
            )
            meta_genome.add_gene(pattern_gene)
        
        # Add population management genes
        management_gene = Gene(
            gene_type=GeneType.STRATEGY,
            name="population_management",
            value={
                "diversity_target": self.diversity_manager.min_diversity,
                "efficiency_target": 1.5,
                "convergence_detection": True,
                "intervention_threshold": 0.8
            }
        )
        meta_genome.add_gene(management_gene)
        
        return meta_genome