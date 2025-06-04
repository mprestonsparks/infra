"""
Test diversity enforcement mechanisms.
"""

import pytest
import sys
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dean.diversity import (
    GeneticDiversityManager, DiversityMetrics,
    AgentGenome, Gene, GeneType,
    MutationEngine, CrossoverEngine
)


class TestDiversityEnforcement:
    """Test suite for diversity enforcement."""
    
    def create_test_genome(self, variant: int = 0) -> AgentGenome:
        """Create a test genome with some variation."""
        genome = AgentGenome()
        
        # Add base genes
        genome.add_gene(Gene(
            gene_type=GeneType.HYPERPARAMETER,
            name="learning_rate",
            value=0.01 + variant * 0.001
        ))
        
        genome.add_gene(Gene(
            gene_type=GeneType.PROMPT_TEMPLATE,
            name="main_prompt",
            value=f"Solve this problem: {{input}} (variant {variant})"
        ))
        
        genome.add_gene(Gene(
            gene_type=GeneType.STRATEGY,
            name="search_strategy",
            value={'depth': 3 + variant, 'breadth': 2}
        ))
        
        return genome
    
    def create_monoculture_population(self, size: int = 10) -> List[AgentGenome]:
        """Create a population with very low diversity."""
        base_genome = self.create_test_genome()
        population = []
        
        for i in range(size):
            # Create nearly identical genomes
            genome = base_genome.clone()
            if i % 3 == 0:
                # Slight variation in some individuals
                genome.genes["learning_rate"].value += 0.0001
            population.append(genome)
        
        return population
    
    def create_diverse_population(self, size: int = 10) -> List[AgentGenome]:
        """Create a population with good diversity."""
        population = []
        
        for i in range(size):
            genome = self.create_test_genome(variant=i)
            
            # Add unique genes to some individuals
            if i % 2 == 0:
                genome.add_gene(Gene(
                    gene_type=GeneType.MODULE_CONFIG,
                    name=f"module_{i}",
                    value={'enabled': True, 'config': i}
                ))
            
            if i % 3 == 0:
                genome.add_gene(Gene(
                    gene_type=GeneType.BEHAVIOR_RULE,
                    name=f"rule_{i}",
                    value=f"If condition_{i}, then action_{i}"
                ))
            
            population.append(genome)
        
        return population
    
    def test_diversity_metrics_calculation(self):
        """Test diversity metrics calculation."""
        manager = GeneticDiversityManager()
        
        # Test with monoculture
        monoculture = self.create_monoculture_population()
        metrics = manager.calculate_diversity_metrics(monoculture)
        
        assert metrics.genetic_diversity < 0.1
        assert metrics.convergence_risk > 0.8
        assert metrics.overall_diversity < 0.3
        
        # Test with diverse population
        diverse_pop = self.create_diverse_population()
        metrics = manager.calculate_diversity_metrics(diverse_pop)
        
        assert metrics.genetic_diversity > 0.3
        assert metrics.convergence_risk < 0.5
        assert metrics.overall_diversity > 0.4
    
    def test_emergency_diversification(self):
        """Test emergency diversity intervention."""
        manager = GeneticDiversityManager(
            min_diversity=0.3,
            emergency_threshold=0.15
        )
        
        # Create critically low diversity population
        population = self.create_monoculture_population(10)
        
        # Measure before
        metrics_before = manager.calculate_diversity_metrics(population)
        assert metrics_before.overall_diversity < 0.15
        
        # Apply diversity enforcement
        population = manager.enforce_diversity(population)
        
        # Measure after
        metrics_after = manager.calculate_diversity_metrics(population)
        
        # Should have increased diversity
        assert metrics_after.overall_diversity > metrics_before.overall_diversity
        assert manager.emergency_interventions == 1
    
    def test_standard_diversification(self):
        """Test standard diversity maintenance."""
        manager = GeneticDiversityManager(min_diversity=0.4)
        
        # Create population with moderate diversity
        population = self.create_monoculture_population(8)
        # Add a couple diverse individuals
        population.extend(self.create_diverse_population(2))
        
        # Apply diversity enforcement
        population = manager.enforce_diversity(population)
        
        # Check intervention was applied
        assert manager.intervention_count > 0
    
    def test_mutation_injection(self):
        """Test mutation-based diversity injection."""
        engine = MutationEngine()
        genome = self.create_test_genome()
        
        # Apply mutations
        mutated = engine.mutate(genome, force_mutation=True)
        
        # Should be different from original
        assert mutated.calculate_hash() != genome.calculate_hash()
        assert len(mutated.mutation_history) > 0
        assert mutated.generation == genome.generation + 1
    
    def test_hypermutation(self):
        """Test hypermutation for escaping local optima."""
        engine = MutationEngine()
        genome = self.create_test_genome()
        
        # Apply hypermutation
        hypermutated = engine.hypermutate(genome, intensity=0.8)
        
        # Should have multiple mutations
        assert len(hypermutated.mutation_history) > 1
        assert any(m['type'] == 'hypermutation' for m in hypermutated.mutation_history)
        
        # Should be very different from original
        distance = genome.distance_from(hypermutated)
        assert distance > 0.3
    
    def test_crossover_diversity(self):
        """Test crossover operations maintain diversity."""
        engine = CrossoverEngine()
        
        # Create two different parents
        parent1 = self.create_test_genome(variant=0)
        parent2 = self.create_test_genome(variant=5)
        
        # Add unique genes to each parent
        parent1.add_gene(Gene(
            gene_type=GeneType.MODULE_CONFIG,
            name="module_p1",
            value={'parent': 1}
        ))
        
        parent2.add_gene(Gene(
            gene_type=GeneType.MODULE_CONFIG,
            name="module_p2",
            value={'parent': 2}
        ))
        
        # Perform crossover
        result = engine.crossover(parent1, parent2, num_offspring=2)
        
        assert len(result.offspring) == 2
        
        # Offspring should be different from each other and parents
        offspring1, offspring2 = result.offspring
        
        assert offspring1.calculate_hash() != offspring2.calculate_hash()
        assert offspring1.calculate_hash() != parent1.calculate_hash()
        assert offspring1.calculate_hash() != parent2.calculate_hash()
    
    def test_pattern_library(self):
        """Test cross-domain pattern import."""
        manager = GeneticDiversityManager()
        
        # Store successful patterns
        successful_gene = Gene(
            gene_type=GeneType.STRATEGY,
            name="successful_strategy",
            value={'approach': 'optimal', 'score': 0.95}
        )
        
        manager.store_successful_pattern(successful_gene, domain="optimization")
        
        # Create genome and import patterns
        genome = self.create_test_genome()
        original_gene_count = len(genome.genes)
        
        manager._import_cross_domain_patterns(genome)
        
        # Should have new genes
        assert len(genome.genes) > original_gene_count
        
        # Check for imported gene
        imported_genes = [g for g in genome.genes.values() 
                         if 'imported_from' in g.metadata]
        assert len(imported_genes) > 0
    
    def test_cluster_identification(self):
        """Test genetic cluster identification."""
        manager = GeneticDiversityManager()
        
        # Create population with clear clusters
        population = []
        
        # Cluster 1: Very similar genomes
        base1 = self.create_test_genome(0)
        for i in range(3):
            genome = base1.clone()
            genome.genes["learning_rate"].value += i * 0.0001
            population.append(genome)
        
        # Cluster 2: Different similar genomes
        base2 = self.create_test_genome(10)
        for i in range(3):
            genome = base2.clone()
            genome.genes["learning_rate"].value += i * 0.0001
            population.append(genome)
        
        # Outlier
        outlier = self.create_test_genome(20)
        outlier.add_gene(Gene(
            gene_type=GeneType.BEHAVIOR_RULE,
            name="unique_rule",
            value="Special behavior"
        ))
        population.append(outlier)
        
        # Identify clusters
        clusters = manager._identify_clusters(population)
        
        # Should identify at least 2 clusters
        assert len(clusters) >= 2
        
        # Check cluster sizes
        cluster_sizes = sorted([len(c) for c in clusters])
        assert cluster_sizes[-1] >= 3  # Largest cluster should have at least 3
    
    def test_diversity_trends(self):
        """Test diversity trend tracking."""
        manager = GeneticDiversityManager()
        
        # Simulate multiple generations
        population = self.create_monoculture_population(10)
        
        for _ in range(5):
            population = manager.enforce_diversity(population)
        
        # Get diversity report
        report = manager.get_diversity_report()
        
        assert 'current_metrics' in report
        assert 'trends' in report
        assert 'interventions' in report
        
        # Should have positive diversity trend after interventions
        assert len(manager.diversity_history) >= 5
        assert report['interventions']['total'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])