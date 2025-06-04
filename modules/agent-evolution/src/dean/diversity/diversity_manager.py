"""
Genetic diversity management to prevent monocultures.

Maintains population health through enforced variance and
actively prevents premature convergence to local optima.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import random

from .genome import AgentGenome, Gene, GeneType
from .mutation import MutationEngine, MutationType
from .crossover import CrossoverEngine, CrossoverStrategy

logger = logging.getLogger(__name__)


@dataclass
class DiversityMetrics:
    """Population diversity measurements."""
    genetic_diversity: float  # 0-1, based on genome differences
    gene_type_diversity: float  # Distribution of gene types
    value_diversity: float  # Diversity in gene values
    cluster_count: int  # Number of distinct genetic clusters
    convergence_risk: float  # Risk of monoculture (0-1)
    unique_genes: int  # Count of unique genes in population
    average_distance: float  # Average pairwise genetic distance
    
    @property
    def overall_diversity(self) -> float:
        """Combined diversity score."""
        return (
            self.genetic_diversity * 0.4 +
            self.gene_type_diversity * 0.2 +
            self.value_diversity * 0.2 +
            min(1.0, self.cluster_count / 5) * 0.1 +
            (1 - self.convergence_risk) * 0.1
        )


class GeneticDiversityManager:
    """
    Maintains population health through enforced variance.
    
    Actively monitors and maintains genetic diversity to prevent
    monocultures and ensure continued evolution potential.
    """
    
    def __init__(self,
                 min_diversity: float = 0.3,
                 target_diversity: float = 0.5,
                 emergency_threshold: float = 0.15):
        """
        Initialize diversity manager.
        
        Args:
            min_diversity: Minimum acceptable diversity level
            target_diversity: Target diversity to maintain
            emergency_threshold: Diversity level triggering emergency measures
        """
        self.min_diversity = min_diversity
        self.target_diversity = target_diversity
        self.emergency_threshold = emergency_threshold
        
        self.mutation_engine = MutationEngine()
        self.crossover_engine = CrossoverEngine()
        
        # Tracking
        self.diversity_history: List[DiversityMetrics] = []
        self.intervention_count = 0
        self.emergency_interventions = 0
        
        # Pattern library for cross-domain import
        self.pattern_library: Dict[str, List[Gene]] = defaultdict(list)
    
    def enforce_diversity(self, population: List[AgentGenome]) -> List[AgentGenome]:
        """
        Actively prevent monocultures through diversity enforcement.
        
        Analyzes population diversity and applies interventions
        as needed to maintain healthy genetic variance.
        """
        if len(population) < 2:
            return population
        
        # Calculate current diversity
        metrics = self.calculate_diversity_metrics(population)
        self.diversity_history.append(metrics)
        
        logger.info(f"Population diversity: {metrics.overall_diversity:.3f} "
                   f"(genetic: {metrics.genetic_diversity:.3f}, "
                   f"convergence risk: {metrics.convergence_risk:.3f})")
        
        # Determine intervention level
        if metrics.overall_diversity < self.emergency_threshold:
            logger.warning(f"EMERGENCY: Diversity critically low ({metrics.overall_diversity:.3f})")
            population = self._emergency_diversification(population, metrics)
            self.emergency_interventions += 1
        elif metrics.overall_diversity < self.min_diversity:
            logger.info(f"Diversity below minimum ({metrics.overall_diversity:.3f}), intervening")
            population = self._standard_diversification(population, metrics)
            self.intervention_count += 1
        elif metrics.convergence_risk > 0.7:
            logger.info(f"High convergence risk ({metrics.convergence_risk:.3f}), injecting variation")
            population = self._inject_variation(population)
            self.intervention_count += 1
        
        return population
    
    def calculate_diversity_metrics(self, population: List[AgentGenome]) -> DiversityMetrics:
        """Calculate comprehensive diversity metrics for the population."""
        if not population:
            return DiversityMetrics(0, 0, 0, 0, 1.0, 0, 0)
        
        # Genetic diversity - average pairwise distance
        genetic_distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = population[i].distance_from(population[j])
                genetic_distances.append(distance)
        
        genetic_diversity = np.mean(genetic_distances) if genetic_distances else 0.0
        average_distance = genetic_diversity
        
        # Gene type diversity - Shannon entropy
        gene_type_counts = defaultdict(int)
        total_genes = 0
        for genome in population:
            for gene in genome.genes.values():
                gene_type_counts[gene.gene_type] += 1
                total_genes += 1
        
        if total_genes > 0:
            gene_type_probs = [count / total_genes for count in gene_type_counts.values()]
            gene_type_diversity = -sum(p * np.log(p) for p in gene_type_probs if p > 0)
            # Normalize by maximum possible entropy
            max_entropy = -np.log(1 / len(GeneType))
            gene_type_diversity = gene_type_diversity / max_entropy if max_entropy > 0 else 0
        else:
            gene_type_diversity = 0.0
        
        # Value diversity - unique values per gene name
        gene_values = defaultdict(set)
        for genome in population:
            for gene_name, gene in genome.genes.items():
                # Convert value to hashable representation
                value_repr = str(gene.value)
                gene_values[gene_name].add(value_repr)
        
        if gene_values:
            unique_value_ratios = [len(values) / len(population) 
                                  for values in gene_values.values()]
            value_diversity = np.mean(unique_value_ratios)
        else:
            value_diversity = 0.0
        
        # Cluster analysis - identify genetic clusters
        clusters = self._identify_clusters(population)
        cluster_count = len(clusters)
        
        # Convergence risk - largest cluster size / population size
        if clusters:
            largest_cluster = max(len(cluster) for cluster in clusters)
            convergence_risk = largest_cluster / len(population)
        else:
            convergence_risk = 1.0
        
        # Unique genes across population
        all_genes = set()
        for genome in population:
            all_genes.update(genome.genes.keys())
        unique_genes = len(all_genes)
        
        return DiversityMetrics(
            genetic_diversity=genetic_diversity,
            gene_type_diversity=gene_type_diversity,
            value_diversity=value_diversity,
            cluster_count=cluster_count,
            convergence_risk=convergence_risk,
            unique_genes=unique_genes,
            average_distance=average_distance
        )
    
    def _identify_clusters(self, population: List[AgentGenome], threshold: float = 0.2) -> List[List[int]]:
        """Identify genetic clusters using distance threshold."""
        n = len(population)
        if n == 0:
            return []
        
        # Build adjacency matrix
        adjacent = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                distance = population[i].distance_from(population[j])
                if distance < threshold:
                    adjacent[i][j] = adjacent[j][i] = True
        
        # Find connected components
        visited = [False] * n
        clusters = []
        
        for i in range(n):
            if not visited[i]:
                cluster = []
                stack = [i]
                
                while stack:
                    node = stack.pop()
                    if not visited[node]:
                        visited[node] = True
                        cluster.append(node)
                        
                        for j in range(n):
                            if adjacent[node][j] and not visited[j]:
                                stack.append(j)
                
                if cluster:
                    clusters.append(cluster)
        
        return clusters
    
    def _emergency_diversification(self, 
                                  population: List[AgentGenome],
                                  metrics: DiversityMetrics) -> List[AgentGenome]:
        """Emergency measures for critically low diversity."""
        logger.warning("Applying emergency diversification measures")
        
        # 1. Hypermutate a portion of the population
        hypermutate_count = max(2, len(population) // 3)
        for i in range(hypermutate_count):
            idx = random.randint(0, len(population) - 1)
            population[idx] = self.mutation_engine.hypermutate(
                population[idx], 
                intensity=0.8
            )
        
        # 2. Import patterns from library
        if self.pattern_library:
            import_count = min(3, len(population) // 4)
            for _ in range(import_count):
                if population:
                    idx = random.randint(0, len(population) - 1)
                    self._import_cross_domain_patterns(population[idx])
        
        # 3. Generate completely new individuals
        new_count = max(1, len(population) // 5)
        new_individuals = []
        for _ in range(new_count):
            new_genome = self._generate_random_genome()
            new_individuals.append(new_genome)
        
        # Replace lowest diversity individuals
        if new_individuals:
            # Find most similar pairs (lowest diversity contributors)
            similarity_scores = []
            for i, genome in enumerate(population):
                min_distance = min(
                    genome.distance_from(other) 
                    for j, other in enumerate(population) if i != j
                ) if len(population) > 1 else 0
                similarity_scores.append((i, min_distance))
            
            # Sort by similarity (ascending - most similar first)
            similarity_scores.sort(key=lambda x: x[1])
            
            # Replace most similar individuals
            for i, new_genome in enumerate(new_individuals[:len(similarity_scores)]):
                idx = similarity_scores[i][0]
                population[idx] = new_genome
        
        return population
    
    def _standard_diversification(self,
                                 population: List[AgentGenome],
                                 metrics: DiversityMetrics) -> List[AgentGenome]:
        """Standard diversity injection for low diversity."""
        logger.info("Applying standard diversification")
        
        # 1. Increase mutation rates temporarily
        original_base_rate = self.mutation_engine.rates.base_rate
        self.mutation_engine.rates.adapt(metrics.overall_diversity, 0)
        
        # 2. Mutate individuals in largest clusters
        clusters = self._identify_clusters(population)
        if clusters:
            # Sort clusters by size
            clusters.sort(key=len, reverse=True)
            
            # Mutate members of large clusters
            for cluster_indices in clusters[:2]:  # Top 2 largest clusters
                if len(cluster_indices) > 2:
                    # Mutate all but one member of the cluster
                    for idx in cluster_indices[1:]:
                        population[idx] = self.mutation_engine.mutate(
                            population[idx],
                            force_mutation=True
                        )
        
        # 3. Cross-pollinate between distant individuals
        if len(population) >= 4:
            # Find most distant pairs
            max_distance = 0
            best_pair = (0, 1)
            
            for i in range(len(population)):
                for j in range(i + 1, len(population)):
                    distance = population[i].distance_from(population[j])
                    if distance > max_distance:
                        max_distance = distance
                        best_pair = (i, j)
            
            # Create offspring from distant parents
            if max_distance > 0.5:
                result = self.crossover_engine.crossover(
                    population[best_pair[0]],
                    population[best_pair[1]],
                    strategy=CrossoverStrategy.UNIFORM,
                    num_offspring=2
                )
                
                # Replace random individuals with offspring
                for offspring in result.offspring:
                    idx = random.randint(0, len(population) - 1)
                    population[idx] = offspring
        
        # Restore mutation rate
        self.mutation_engine.rates.base_rate = original_base_rate
        
        return population
    
    def _inject_variation(self, population: List[AgentGenome]) -> List[AgentGenome]:
        """Inject variation to prevent convergence."""
        logger.info("Injecting genetic variation")
        
        # Select 20% of population for variation injection
        inject_count = max(1, len(population) // 5)
        indices = random.sample(range(len(population)), inject_count)
        
        for idx in indices:
            genome = population[idx]
            
            # Add new random genes
            if random.random() < 0.5:
                new_gene = self._generate_random_gene()
                genome.add_gene(new_gene)
            
            # Modify existing genes
            if genome.genes and random.random() < 0.7:
                gene_name = random.choice(list(genome.genes.keys()))
                genome.genes[gene_name] = genome.genes[gene_name].mutate(0.5)
            
            # Import patterns if available
            if self.pattern_library and random.random() < 0.3:
                self._import_cross_domain_patterns(genome)
        
        return population
    
    def store_successful_pattern(self, gene: Gene, domain: str = "default") -> None:
        """Store successful gene patterns for cross-domain import."""
        self.pattern_library[domain].append(gene)
        
        # Keep library size manageable
        if len(self.pattern_library[domain]) > 50:
            # Remove oldest patterns
            self.pattern_library[domain] = self.pattern_library[domain][-50:]
    
    def _import_cross_domain_patterns(self, genome: AgentGenome) -> None:
        """Import successful patterns from other domains."""
        if not self.pattern_library:
            return
        
        # Choose random domain
        domains = list(self.pattern_library.keys())
        if not domains:
            return
        
        domain = random.choice(domains)
        patterns = self.pattern_library[domain]
        
        if patterns:
            # Import 1-3 patterns
            import_count = min(random.randint(1, 3), len(patterns))
            imported = random.sample(patterns, import_count)
            
            for pattern in imported:
                # Create variant of the pattern
                new_gene = Gene(
                    gene_type=pattern.gene_type,
                    name=f"{pattern.name}_imported_{random.randint(1000, 9999)}",
                    value=pattern.value,
                    metadata={**pattern.metadata, 'imported_from': domain}
                )
                
                # Slight mutation to adapt to new context
                new_gene = new_gene.mutate(0.3)
                genome.add_gene(new_gene)
    
    def _generate_random_genome(self) -> AgentGenome:
        """Generate a completely random genome."""
        genome = AgentGenome()
        
        # Add 5-10 random genes
        gene_count = random.randint(5, 10)
        
        for _ in range(gene_count):
            gene = self._generate_random_gene()
            genome.add_gene(gene)
        
        return genome
    
    def _generate_random_gene(self) -> Gene:
        """Generate a single random gene."""
        # Use mutation engine's gene generators
        gene_type = random.choice(list(GeneType))
        
        # Simple random gene generation
        if gene_type == GeneType.HYPERPARAMETER:
            params = [
                ('learning_rate', random.uniform(0.001, 0.1)),
                ('batch_size', random.choice([8, 16, 32, 64])),
                ('temperature', random.uniform(0.1, 2.0)),
                ('iterations', random.randint(10, 100))
            ]
            name, value = random.choice(params)
            
        elif gene_type == GeneType.PROMPT_TEMPLATE:
            templates = [
                "Analyze this problem step by step: {input}",
                "Consider all aspects of: {topic}",
                "Provide a detailed solution for: {problem}",
                "Explain the concept of: {subject}"
            ]
            value = random.choice(templates)
            name = f"prompt_{random.randint(1000, 9999)}"
            
        elif gene_type == GeneType.STRATEGY:
            value = {
                'approach': random.choice(['greedy', 'exploratory', 'balanced']),
                'depth': random.randint(1, 5),
                'timeout': random.randint(10, 60)
            }
            name = f"strategy_{random.randint(1000, 9999)}"
            
        else:
            value = f"random_value_{random.randint(1000, 9999)}"
            name = f"{gene_type.value}_{random.randint(1000, 9999)}"
        
        return Gene(
            gene_type=gene_type,
            name=name,
            value=value,
            metadata={'randomly_generated': True}
        )
    
    def get_diversity_report(self) -> Dict:
        """Generate comprehensive diversity report."""
        if not self.diversity_history:
            return {}
        
        recent_metrics = self.diversity_history[-1]
        
        # Calculate trends
        if len(self.diversity_history) >= 5:
            recent_5 = self.diversity_history[-5:]
            diversity_trend = (recent_metrics.overall_diversity - 
                             recent_5[0].overall_diversity)
        else:
            diversity_trend = 0.0
        
        return {
            'current_metrics': {
                'overall': recent_metrics.overall_diversity,
                'genetic': recent_metrics.genetic_diversity,
                'gene_type': recent_metrics.gene_type_diversity,
                'value': recent_metrics.value_diversity,
                'clusters': recent_metrics.cluster_count,
                'convergence_risk': recent_metrics.convergence_risk,
                'unique_genes': recent_metrics.unique_genes
            },
            'trends': {
                'diversity_trend': diversity_trend,
                'history_length': len(self.diversity_history)
            },
            'interventions': {
                'total': self.intervention_count,
                'emergency': self.emergency_interventions
            },
            'pattern_library': {
                'domains': list(self.pattern_library.keys()),
                'total_patterns': sum(len(patterns) for patterns in self.pattern_library.values())
            }
        }