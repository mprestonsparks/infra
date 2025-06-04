"""
Mutation engine for introducing genetic variation.

Implements various mutation strategies to maintain diversity
and explore the solution space effectively.
"""

import random
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import logging
from dataclasses import dataclass

from .genome import AgentGenome, Gene, GeneType

logger = logging.getLogger(__name__)


class MutationType(Enum):
    """Types of mutations that can be applied."""
    POINT = "point"  # Single gene modification
    INSERTION = "insertion"  # Add new gene
    DELETION = "deletion"  # Remove gene
    DUPLICATION = "duplication"  # Duplicate existing gene
    INVERSION = "inversion"  # Reverse gene order
    TRANSLOCATION = "translocation"  # Move gene to different position
    HYPERMUTATION = "hypermutation"  # Multiple simultaneous mutations


@dataclass
class MutationRate:
    """Adaptive mutation rates."""
    base_rate: float = 0.1
    insertion_rate: float = 0.05
    deletion_rate: float = 0.05
    duplication_rate: float = 0.03
    inversion_rate: float = 0.02
    translocation_rate: float = 0.02
    
    def adapt(self, population_diversity: float, generation: int) -> None:
        """Adapt mutation rates based on population state."""
        # Increase rates when diversity is low
        if population_diversity < 0.2:
            scale_factor = 2.0
        elif population_diversity < 0.3:
            scale_factor = 1.5
        else:
            scale_factor = 1.0
        
        # Decrease rates over time (simulated annealing)
        time_factor = 1.0 / (1.0 + generation * 0.01)
        
        # Apply both factors
        final_factor = scale_factor * time_factor
        
        self.base_rate = min(0.5, 0.1 * final_factor)
        self.insertion_rate = min(0.2, 0.05 * final_factor)
        self.deletion_rate = min(0.2, 0.05 * final_factor)


class MutationEngine:
    """
    Engine for applying mutations to agent genomes.
    
    Supports multiple mutation types and adaptive mutation rates
    to maintain diversity while exploring the solution space.
    """
    
    def __init__(self, mutation_rates: Optional[MutationRate] = None):
        """Initialize mutation engine with configurable rates."""
        self.rates = mutation_rates or MutationRate()
        self.mutation_count = 0
        self.mutation_history: List[Dict] = []
        
        # Gene generators for insertion mutations
        self.gene_generators: Dict[GeneType, Callable] = {
            GeneType.PROMPT_TEMPLATE: self._generate_prompt_gene,
            GeneType.STRATEGY: self._generate_strategy_gene,
            GeneType.HYPERPARAMETER: self._generate_hyperparameter_gene,
            GeneType.MODULE_CONFIG: self._generate_module_config_gene,
            GeneType.BEHAVIOR_RULE: self._generate_behavior_rule_gene
        }
    
    def mutate(self, genome: AgentGenome, force_mutation: bool = False) -> AgentGenome:
        """
        Apply mutations to a genome.
        
        Args:
            genome: Genome to mutate
            force_mutation: Force at least one mutation
            
        Returns:
            Mutated genome (new instance)
        """
        # Clone the genome
        mutated = genome.clone()
        mutated.generation += 1
        mutated.parent_ids = [genome.calculate_hash()]
        
        mutations_applied = []
        
        # Point mutations on existing genes
        for gene_name, gene in list(mutated.genes.items()):
            if random.random() < self.rates.base_rate or (force_mutation and not mutations_applied):
                mutated_gene = gene.mutate(self.rates.base_rate)
                if mutated_gene != gene:
                    mutated.genes[gene_name] = mutated_gene
                    mutations_applied.append({
                        'type': MutationType.POINT.value,
                        'gene': gene_name,
                        'details': 'Modified gene value'
                    })
        
        # Insertion mutations
        if random.random() < self.rates.insertion_rate or (force_mutation and not mutations_applied):
            new_gene = self._insert_random_gene(mutated)
            if new_gene:
                mutations_applied.append({
                    'type': MutationType.INSERTION.value,
                    'gene': new_gene.name,
                    'details': f'Added {new_gene.gene_type.value} gene'
                })
        
        # Deletion mutations (protect essential genes)
        if len(mutated.genes) > 3 and random.random() < self.rates.deletion_rate:
            deleted = self._delete_random_gene(mutated)
            if deleted:
                mutations_applied.append({
                    'type': MutationType.DELETION.value,
                    'gene': deleted,
                    'details': 'Removed gene'
                })
        
        # Duplication mutations
        if random.random() < self.rates.duplication_rate:
            duplicated = self._duplicate_random_gene(mutated)
            if duplicated:
                mutations_applied.append({
                    'type': MutationType.DUPLICATION.value,
                    'gene': duplicated,
                    'details': 'Duplicated gene with variation'
                })
        
        # Record mutations
        if mutations_applied:
            mutated.mutation_history.extend(mutations_applied)
            self.mutation_count += len(mutations_applied)
            self.mutation_history.append({
                'genome_hash': mutated.calculate_hash(),
                'mutations': mutations_applied,
                'generation': mutated.generation
            })
            
            logger.debug(f"Applied {len(mutations_applied)} mutations to genome")
        
        return mutated
    
    def hypermutate(self, genome: AgentGenome, intensity: float = 0.5) -> AgentGenome:
        """
        Apply aggressive mutations for escaping local optima.
        
        Args:
            genome: Genome to hypermutate
            intensity: Mutation intensity (0-1)
            
        Returns:
            Heavily mutated genome
        """
        # Temporarily increase mutation rates
        original_rates = MutationRate(
            base_rate=self.rates.base_rate,
            insertion_rate=self.rates.insertion_rate,
            deletion_rate=self.rates.deletion_rate
        )
        
        # Scale up rates based on intensity
        self.rates.base_rate = min(0.8, original_rates.base_rate * (1 + intensity * 4))
        self.rates.insertion_rate = min(0.5, original_rates.insertion_rate * (1 + intensity * 4))
        self.rates.deletion_rate = min(0.3, original_rates.deletion_rate * (1 + intensity * 4))
        
        # Apply multiple rounds of mutation
        mutated = genome
        rounds = int(1 + intensity * 5)
        
        for _ in range(rounds):
            mutated = self.mutate(mutated, force_mutation=True)
        
        # Record hypermutation event
        mutated.mutation_history.append({
            'type': MutationType.HYPERMUTATION.value,
            'intensity': intensity,
            'rounds': rounds,
            'details': f'Hypermutation with {rounds} rounds'
        })
        
        # Restore original rates
        self.rates = original_rates
        
        return mutated
    
    def _insert_random_gene(self, genome: AgentGenome) -> Optional[Gene]:
        """Insert a randomly generated gene."""
        # Choose gene type weighted by usefulness
        gene_types = list(GeneType)
        # Ensure weights match number of gene types
        weights = [0.25, 0.20, 0.20, 0.15, 0.10, 0.10][:len(gene_types)]
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        gene_type = np.random.choice(gene_types, p=weights)
        
        generator = self.gene_generators.get(gene_type)
        if generator:
            gene = generator()
            genome.add_gene(gene)
            return gene
        return None
    
    def _delete_random_gene(self, genome: AgentGenome) -> Optional[str]:
        """Delete a random non-essential gene."""
        # Don't delete if too few genes
        if len(genome.genes) <= 3:
            return None
        
        # Identify non-essential genes (keep at least one of each type)
        gene_type_counts = {}
        for gene in genome.genes.values():
            gene_type_counts[gene.gene_type] = gene_type_counts.get(gene.gene_type, 0) + 1
        
        deletable = [
            name for name, gene in genome.genes.items()
            if gene_type_counts[gene.gene_type] > 1
        ]
        
        if deletable:
            gene_name = random.choice(deletable)
            genome.remove_gene(gene_name)
            return gene_name
        return None
    
    def _duplicate_random_gene(self, genome: AgentGenome) -> Optional[str]:
        """Duplicate a gene with slight variations."""
        if not genome.genes:
            return None
        
        # Choose gene to duplicate
        source_name = random.choice(list(genome.genes.keys()))
        source_gene = genome.genes[source_name]
        
        # Create variant
        new_name = f"{source_name}_variant_{random.randint(1000, 9999)}"
        new_gene = source_gene.mutate(mutation_rate=0.3)  # Higher mutation rate
        new_gene.name = new_name
        new_gene.metadata['duplicated_from'] = source_name
        
        genome.add_gene(new_gene)
        return new_name
    
    def _generate_prompt_gene(self) -> Gene:
        """Generate a new prompt template gene."""
        templates = [
            "Analyze the following data and identify key patterns: {input}",
            "Given the context: {context}, provide a solution for: {problem}",
            "Step by step, solve this problem: {task}",
            "Summarize the main points of: {content}",
            "Compare and contrast: {item1} vs {item2}",
            "Explain the relationship between: {concept1} and {concept2}",
            "Optimize the following process: {process}",
            "Debug this issue: {error} in context: {context}"
        ]
        
        base_template = random.choice(templates)
        variations = [
            lambda s: s + "\nProvide reasoning for each step.",
            lambda s: s + "\nBe concise and focus on key insights.",
            lambda s: s + "\nInclude confidence levels for each conclusion.",
            lambda s: "Think carefully. " + s,
            lambda s: s.replace("identify", "discover"),
            lambda s: s.replace("provide", "generate")
        ]
        
        # Apply 0-2 variations
        for _ in range(random.randint(0, 2)):
            if variations:
                variation = random.choice(variations)
                base_template = variation(base_template)
        
        return Gene(
            gene_type=GeneType.PROMPT_TEMPLATE,
            name=f"prompt_{random.randint(1000, 9999)}",
            value=base_template,
            metadata={'generated': True}
        )
    
    def _generate_strategy_gene(self) -> Gene:
        """Generate a new strategy gene."""
        strategies = [
            {
                'name': 'exploration_strategy',
                'explore_rate': random.uniform(0.1, 0.5),
                'depth_limit': random.randint(3, 10),
                'breadth_factor': random.uniform(0.5, 2.0)
            },
            {
                'name': 'optimization_strategy',
                'iterations': random.randint(10, 100),
                'convergence_threshold': random.uniform(0.001, 0.1),
                'learning_rate': random.uniform(0.01, 0.5)
            },
            {
                'name': 'memory_strategy',
                'cache_size': random.randint(100, 1000),
                'ttl_seconds': random.randint(60, 3600),
                'eviction_policy': random.choice(['lru', 'lfu', 'fifo'])
            }
        ]
        
        strategy = random.choice(strategies)
        return Gene(
            gene_type=GeneType.STRATEGY,
            name=f"strategy_{strategy['name']}_{random.randint(1000, 9999)}",
            value=strategy,
            metadata={'generated': True}
        )
    
    def _generate_hyperparameter_gene(self) -> Gene:
        """Generate a new hyperparameter gene."""
        hyperparams = [
            ('temperature', random.uniform(0.1, 2.0)),
            ('top_p', random.uniform(0.5, 1.0)),
            ('max_tokens', random.randint(100, 2000)),
            ('frequency_penalty', random.uniform(-2.0, 2.0)),
            ('presence_penalty', random.uniform(-2.0, 2.0)),
            ('timeout_seconds', random.randint(10, 300)),
            ('retry_attempts', random.randint(1, 5)),
            ('batch_size', random.choice([1, 2, 4, 8, 16]))
        ]
        
        param_name, param_value = random.choice(hyperparams)
        return Gene(
            gene_type=GeneType.HYPERPARAMETER,
            name=f"param_{param_name}_{random.randint(1000, 9999)}",
            value=param_value,
            metadata={'parameter_name': param_name, 'generated': True}
        )
    
    def _generate_module_config_gene(self) -> Gene:
        """Generate a new module configuration gene."""
        configs = [
            {
                'module': 'chain_of_thought',
                'enabled': True,
                'max_steps': random.randint(3, 10),
                'require_justification': random.choice([True, False])
            },
            {
                'module': 'self_critique',
                'enabled': True,
                'critique_rounds': random.randint(1, 3),
                'min_confidence': random.uniform(0.5, 0.9)
            },
            {
                'module': 'multi_agent',
                'enabled': True,
                'num_agents': random.randint(2, 5),
                'consensus_threshold': random.uniform(0.5, 1.0)
            }
        ]
        
        config = random.choice(configs)
        return Gene(
            gene_type=GeneType.MODULE_CONFIG,
            name=f"module_{config['module']}_{random.randint(1000, 9999)}",
            value=config,
            metadata={'generated': True}
        )
    
    def _generate_behavior_rule_gene(self) -> Gene:
        """Generate a new behavior rule gene."""
        rules = [
            "If confidence < 0.7, request additional context",
            "If token_usage > 0.8 * limit, summarize and conclude",
            "If error_count > 3, switch to fallback strategy",
            "If response_time > 10s, simplify approach",
            "If diversity_score < 0.3, increase exploration rate",
            "If success_rate > 0.9, attempt more complex tasks",
            "If memory_usage > 0.8, clear cache and continue",
            "If pattern_detected, store and reuse optimization"
        ]
        
        rule = random.choice(rules)
        return Gene(
            gene_type=GeneType.BEHAVIOR_RULE,
            name=f"rule_{random.randint(1000, 9999)}",
            value=rule,
            metadata={'generated': True}
        )