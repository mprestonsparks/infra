"""
Crossover operations for combining genetic material from multiple agents.

Implements various crossover strategies to create offspring that
inherit beneficial traits from multiple parents.
"""

import random
import numpy as np
from typing import List, Tuple, Dict, Optional
from enum import Enum
from dataclasses import dataclass
import logging

from .genome import AgentGenome, Gene, GeneType

logger = logging.getLogger(__name__)


class CrossoverStrategy(Enum):
    """Available crossover strategies."""
    UNIFORM = "uniform"  # Each gene randomly selected from parents
    SINGLE_POINT = "single_point"  # Single crossover point
    TWO_POINT = "two_point"  # Two crossover points
    GENE_TYPE = "gene_type"  # Crossover by gene type
    FITNESS_WEIGHTED = "fitness_weighted"  # Weighted by parent fitness
    ADAPTIVE = "adaptive"  # Choose strategy based on parent similarity


@dataclass
class CrossoverResult:
    """Result of a crossover operation."""
    offspring: List[AgentGenome]
    strategy_used: CrossoverStrategy
    crossover_points: Optional[List[int]] = None
    parent_contributions: Optional[Dict[str, float]] = None


class CrossoverEngine:
    """
    Engine for performing genetic crossover operations.
    
    Combines genetic material from multiple parents to create
    offspring with potentially beneficial trait combinations.
    """
    
    def __init__(self, default_strategy: CrossoverStrategy = CrossoverStrategy.ADAPTIVE):
        """Initialize crossover engine."""
        self.default_strategy = default_strategy
        self.crossover_count = 0
        self.crossover_history: List[Dict] = []
    
    def crossover(self,
                  parent1: AgentGenome,
                  parent2: AgentGenome,
                  strategy: Optional[CrossoverStrategy] = None,
                  num_offspring: int = 2) -> CrossoverResult:
        """
        Perform crossover between two parent genomes.
        
        Args:
            parent1: First parent genome
            parent2: Second parent genome
            strategy: Crossover strategy to use
            num_offspring: Number of offspring to generate
            
        Returns:
            CrossoverResult containing offspring and metadata
        """
        strategy = strategy or self.default_strategy
        
        # Choose appropriate strategy if adaptive
        if strategy == CrossoverStrategy.ADAPTIVE:
            strategy = self._choose_adaptive_strategy(parent1, parent2)
        
        # Perform crossover based on strategy
        if strategy == CrossoverStrategy.UNIFORM:
            result = self._uniform_crossover(parent1, parent2, num_offspring)
        elif strategy == CrossoverStrategy.SINGLE_POINT:
            result = self._single_point_crossover(parent1, parent2, num_offspring)
        elif strategy == CrossoverStrategy.TWO_POINT:
            result = self._two_point_crossover(parent1, parent2, num_offspring)
        elif strategy == CrossoverStrategy.GENE_TYPE:
            result = self._gene_type_crossover(parent1, parent2, num_offspring)
        elif strategy == CrossoverStrategy.FITNESS_WEIGHTED:
            result = self._fitness_weighted_crossover(parent1, parent2, num_offspring)
        else:
            # Fallback to uniform
            result = self._uniform_crossover(parent1, parent2, num_offspring)
        
        # Update metadata
        for offspring in result.offspring:
            offspring.parent_ids = [parent1.calculate_hash(), parent2.calculate_hash()]
            offspring.generation = max(parent1.generation, parent2.generation) + 1
        
        # Record crossover
        self.crossover_count += 1
        self.crossover_history.append({
            'parent1_hash': parent1.calculate_hash(),
            'parent2_hash': parent2.calculate_hash(),
            'strategy': strategy.value,
            'num_offspring': len(result.offspring),
            'offspring_hashes': [o.calculate_hash() for o in result.offspring]
        })
        
        return result
    
    def multi_parent_crossover(self,
                              parents: List[AgentGenome],
                              num_offspring: int = 1) -> List[AgentGenome]:
        """
        Perform crossover with multiple parents.
        
        Creates offspring that inherit traits from multiple parents,
        useful for combining diverse beneficial traits.
        """
        if len(parents) < 2:
            raise ValueError("Need at least 2 parents for crossover")
        
        offspring = []
        
        for _ in range(num_offspring):
            child_genes = {}
            
            # Collect all unique genes from parents
            all_gene_names = set()
            for parent in parents:
                all_gene_names.update(parent.genes.keys())
            
            # For each gene, randomly select from available parents
            for gene_name in all_gene_names:
                # Find parents that have this gene
                parents_with_gene = [p for p in parents if gene_name in p.genes]
                
                if parents_with_gene:
                    # Weight selection by parent fitness if available
                    if all(p.fitness_scores for p in parents_with_gene):
                        weights = [sum(p.fitness_scores.values()) for p in parents_with_gene]
                        total_weight = sum(weights)
                        if total_weight > 0:
                            weights = [w / total_weight for w in weights]
                            selected_parent = np.random.choice(parents_with_gene, p=weights)
                        else:
                            selected_parent = random.choice(parents_with_gene)
                    else:
                        selected_parent = random.choice(parents_with_gene)
                    
                    # Clone the gene
                    child_genes[gene_name] = Gene(
                        gene_type=selected_parent.genes[gene_name].gene_type,
                        name=gene_name,
                        value=selected_parent.genes[gene_name].value,
                        metadata=selected_parent.genes[gene_name].metadata.copy()
                    )
            
            # Create offspring
            child = AgentGenome(
                genes=child_genes,
                generation=max(p.generation for p in parents) + 1,
                parent_ids=[p.calculate_hash() for p in parents]
            )
            
            offspring.append(child)
        
        return offspring
    
    def _choose_adaptive_strategy(self, parent1: AgentGenome, parent2: AgentGenome) -> CrossoverStrategy:
        """Choose crossover strategy based on parent characteristics."""
        # Calculate genetic distance
        distance = parent1.distance_from(parent2)
        
        # Similar parents: use gene-type crossover to maintain structure
        if distance < 0.2:
            return CrossoverStrategy.GENE_TYPE
        
        # Very different parents: use uniform to explore combinations
        elif distance > 0.7:
            return CrossoverStrategy.UNIFORM
        
        # Moderate difference: use point crossover
        else:
            return CrossoverStrategy.TWO_POINT if len(parent1.genes) > 10 else CrossoverStrategy.SINGLE_POINT
    
    def _uniform_crossover(self,
                          parent1: AgentGenome,
                          parent2: AgentGenome,
                          num_offspring: int) -> CrossoverResult:
        """Each gene randomly selected from either parent."""
        offspring = []
        
        # Get all unique gene names
        all_genes = set(parent1.genes.keys()) | set(parent2.genes.keys())
        
        for i in range(num_offspring):
            child_genes = {}
            parent_contributions = {parent1.calculate_hash(): 0, parent2.calculate_hash(): 0}
            
            for gene_name in all_genes:
                # Randomly select parent
                if gene_name in parent1.genes and gene_name in parent2.genes:
                    # Both have it - random choice
                    if random.random() < 0.5:
                        source = parent1
                    else:
                        source = parent2
                elif gene_name in parent1.genes:
                    source = parent1
                else:
                    source = parent2
                
                # Clone gene
                child_genes[gene_name] = Gene(
                    gene_type=source.genes[gene_name].gene_type,
                    name=gene_name,
                    value=source.genes[gene_name].value,
                    metadata=source.genes[gene_name].metadata.copy()
                )
                
                parent_contributions[source.calculate_hash()] += 1
            
            # Normalize contributions
            total = sum(parent_contributions.values())
            if total > 0:
                for k in parent_contributions:
                    parent_contributions[k] /= total
            
            child = AgentGenome(genes=child_genes)
            offspring.append(child)
        
        return CrossoverResult(
            offspring=offspring,
            strategy_used=CrossoverStrategy.UNIFORM,
            parent_contributions=parent_contributions
        )
    
    def _single_point_crossover(self,
                               parent1: AgentGenome,
                               parent2: AgentGenome,
                               num_offspring: int) -> CrossoverResult:
        """Single crossover point splits genes between parents."""
        # Get sorted gene names for consistent ordering
        p1_genes = sorted(parent1.genes.keys())
        p2_genes = sorted(parent2.genes.keys())
        all_genes = sorted(set(p1_genes) | set(p2_genes))
        
        if len(all_genes) < 2:
            # Not enough genes for meaningful crossover
            return self._uniform_crossover(parent1, parent2, num_offspring)
        
        # Choose crossover point
        crossover_point = random.randint(1, len(all_genes) - 1)
        
        offspring = []
        
        # Create two complementary offspring
        for i in range(min(num_offspring, 2)):
            child_genes = {}
            
            for j, gene_name in enumerate(all_genes):
                # Determine which parent to use based on position and offspring number
                use_parent1 = (j < crossover_point) if (i == 0) else (j >= crossover_point)
                
                if use_parent1 and gene_name in parent1.genes:
                    source = parent1
                elif not use_parent1 and gene_name in parent2.genes:
                    source = parent2
                elif gene_name in parent1.genes:
                    source = parent1
                else:
                    source = parent2
                
                # Clone gene
                child_genes[gene_name] = Gene(
                    gene_type=source.genes[gene_name].gene_type,
                    name=gene_name,
                    value=source.genes[gene_name].value,
                    metadata=source.genes[gene_name].metadata.copy()
                )
            
            child = AgentGenome(genes=child_genes)
            offspring.append(child)
        
        # If more offspring requested, use uniform crossover for the rest
        if num_offspring > 2:
            additional = self._uniform_crossover(parent1, parent2, num_offspring - 2)
            offspring.extend(additional.offspring)
        
        return CrossoverResult(
            offspring=offspring,
            strategy_used=CrossoverStrategy.SINGLE_POINT,
            crossover_points=[crossover_point]
        )
    
    def _two_point_crossover(self,
                            parent1: AgentGenome,
                            parent2: AgentGenome,
                            num_offspring: int) -> CrossoverResult:
        """Two crossover points create three segments."""
        # Get sorted gene names
        all_genes = sorted(set(parent1.genes.keys()) | set(parent2.genes.keys()))
        
        if len(all_genes) < 3:
            # Not enough genes for two-point crossover
            return self._single_point_crossover(parent1, parent2, num_offspring)
        
        # Choose two crossover points
        points = sorted(random.sample(range(1, len(all_genes)), 2))
        
        offspring = []
        
        # Create two complementary offspring
        for i in range(min(num_offspring, 2)):
            child_genes = {}
            
            for j, gene_name in enumerate(all_genes):
                # Determine which parent based on segment
                if j < points[0]:
                    use_parent1 = (i == 0)
                elif j < points[1]:
                    use_parent1 = (i == 1)
                else:
                    use_parent1 = (i == 0)
                
                if use_parent1 and gene_name in parent1.genes:
                    source = parent1
                elif not use_parent1 and gene_name in parent2.genes:
                    source = parent2
                elif gene_name in parent1.genes:
                    source = parent1
                else:
                    source = parent2
                
                # Clone gene
                child_genes[gene_name] = Gene(
                    gene_type=source.genes[gene_name].gene_type,
                    name=gene_name,
                    value=source.genes[gene_name].value,
                    metadata=source.genes[gene_name].metadata.copy()
                )
            
            child = AgentGenome(genes=child_genes)
            offspring.append(child)
        
        # Additional offspring if needed
        if num_offspring > 2:
            additional = self._uniform_crossover(parent1, parent2, num_offspring - 2)
            offspring.extend(additional.offspring)
        
        return CrossoverResult(
            offspring=offspring,
            strategy_used=CrossoverStrategy.TWO_POINT,
            crossover_points=points
        )
    
    def _gene_type_crossover(self,
                            parent1: AgentGenome,
                            parent2: AgentGenome,
                            num_offspring: int) -> CrossoverResult:
        """Crossover by gene type - inherit complete gene types from each parent."""
        offspring = []
        
        # Get gene types present in either parent
        gene_types = set()
        for gene in parent1.genes.values():
            gene_types.add(gene.gene_type)
        for gene in parent2.genes.values():
            gene_types.add(gene.gene_type)
        
        gene_types = list(gene_types)
        
        for i in range(num_offspring):
            child_genes = {}
            
            # Randomly assign each gene type to a parent
            for gene_type in gene_types:
                source = parent1 if random.random() < 0.5 else parent2
                
                # Get all genes of this type from source
                type_genes = source.get_genes_by_type(gene_type)
                
                # If source doesn't have this type, try other parent
                if not type_genes:
                    other = parent2 if source == parent1 else parent1
                    type_genes = other.get_genes_by_type(gene_type)
                
                # Clone all genes of this type
                for gene in type_genes:
                    child_genes[gene.name] = Gene(
                        gene_type=gene.gene_type,
                        name=gene.name,
                        value=gene.value,
                        metadata=gene.metadata.copy()
                    )
            
            child = AgentGenome(genes=child_genes)
            offspring.append(child)
        
        return CrossoverResult(
            offspring=offspring,
            strategy_used=CrossoverStrategy.GENE_TYPE
        )
    
    def _fitness_weighted_crossover(self,
                                   parent1: AgentGenome,
                                   parent2: AgentGenome,
                                   num_offspring: int) -> CrossoverResult:
        """Weight gene selection by parent fitness scores."""
        # Calculate parent fitness weights
        p1_fitness = sum(parent1.fitness_scores.values()) if parent1.fitness_scores else 1.0
        p2_fitness = sum(parent2.fitness_scores.values()) if parent2.fitness_scores else 1.0
        
        total_fitness = p1_fitness + p2_fitness
        if total_fitness == 0:
            # No fitness info - fall back to uniform
            return self._uniform_crossover(parent1, parent2, num_offspring)
        
        p1_weight = p1_fitness / total_fitness
        
        offspring = []
        
        for _ in range(num_offspring):
            child_genes = {}
            
            # Get all unique genes
            all_genes = set(parent1.genes.keys()) | set(parent2.genes.keys())
            
            for gene_name in all_genes:
                # Select parent based on fitness weight
                if gene_name in parent1.genes and gene_name in parent2.genes:
                    if random.random() < p1_weight:
                        source = parent1
                    else:
                        source = parent2
                elif gene_name in parent1.genes:
                    source = parent1
                else:
                    source = parent2
                
                # Clone gene
                child_genes[gene_name] = Gene(
                    gene_type=source.genes[gene_name].gene_type,
                    name=gene_name,
                    value=source.genes[gene_name].value,
                    metadata=source.genes[gene_name].metadata.copy()
                )
            
            child = AgentGenome(genes=child_genes)
            offspring.append(child)
        
        return CrossoverResult(
            offspring=offspring,
            strategy_used=CrossoverStrategy.FITNESS_WEIGHTED,
            parent_contributions={
                parent1.calculate_hash(): p1_weight,
                parent2.calculate_hash(): 1 - p1_weight
            }
        )