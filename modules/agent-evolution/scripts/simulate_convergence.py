#!/usr/bin/env python3
"""
Simulate population convergence and diversity maintenance.
"""

import sys
from pathlib import Path
import argparse
import random
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dean.diversity import (
    GeneticDiversityManager,
    AgentGenome, Gene, GeneType,
    MutationEngine
)


def create_initial_population(size: int) -> list:
    """Create initial diverse population."""
    population = []
    
    for i in range(size):
        genome = AgentGenome(generation=0)
        
        # Base genes
        genome.add_gene(Gene(
            gene_type=GeneType.HYPERPARAMETER,
            name="learning_rate",
            value=random.uniform(0.001, 0.1)
        ))
        
        genome.add_gene(Gene(
            gene_type=GeneType.HYPERPARAMETER,
            name="temperature",
            value=random.uniform(0.1, 2.0)
        ))
        
        genome.add_gene(Gene(
            gene_type=GeneType.PROMPT_TEMPLATE,
            name="main_prompt",
            value=f"Solve problem with approach {random.choice(['A', 'B', 'C'])}: {{input}}"
        ))
        
        # Some genomes get extra genes
        if random.random() < 0.5:
            genome.add_gene(Gene(
                gene_type=GeneType.STRATEGY,
                name=f"strategy_{i}",
                value={'type': random.choice(['greedy', 'exploratory']),
                       'param': random.random()}
            ))
        
        population.append(genome)
    
    return population


def simulate_selection_pressure(population: list, strength: float = 0.5) -> list:
    """Simulate selection pressure that tends toward convergence."""
    if not population:
        return population
    
    # Find "best" genome (arbitrary criteria)
    scores = []
    for genome in population:
        # Favor specific values (creates convergence pressure)
        score = 0
        if 'learning_rate' in genome.genes:
            lr = genome.genes['learning_rate'].value
            score += 1.0 - abs(lr - 0.05)  # Favor lr=0.05
        
        if 'temperature' in genome.genes:
            temp = genome.genes['temperature'].value  
            score += 1.0 - abs(temp - 1.0)  # Favor temp=1.0
        
        scores.append(score)
    
    # Select based on scores
    selected = []
    for _ in range(len(population)):
        if random.random() < strength:
            # Select based on fitness
            idx = np.random.choice(len(population), p=np.array(scores)/sum(scores))
        else:
            # Random selection
            idx = random.randint(0, len(population) - 1)
        
        selected.append(population[idx].clone())
    
    return selected


def run_simulation(generations: int, 
                   population_size: int,
                   selection_strength: float,
                   min_diversity: float,
                   use_diversity_manager: bool = True):
    """Run evolution simulation."""
    
    # Initialize
    if use_diversity_manager:
        diversity_manager = GeneticDiversityManager(min_diversity=min_diversity)
    else:
        diversity_manager = None
    
    mutation_engine = MutationEngine()
    
    # Create initial population
    population = create_initial_population(population_size)
    
    print(f"Starting simulation with {population_size} individuals")
    print(f"Selection strength: {selection_strength}")
    print(f"Diversity management: {'ENABLED' if use_diversity_manager else 'DISABLED'}")
    print(f"Target minimum diversity: {min_diversity}")
    print("-" * 60)
    
    diversity_history = []
    
    for gen in range(generations):
        # Apply selection pressure
        population = simulate_selection_pressure(population, selection_strength)
        
        # Apply mutations
        for i in range(len(population)):
            if random.random() < 0.1:  # 10% mutation rate
                population[i] = mutation_engine.mutate(population[i])
        
        # Apply diversity management if enabled
        if diversity_manager:
            metrics_before = diversity_manager.calculate_diversity_metrics(population)
            population = diversity_manager.enforce_diversity(population)
            metrics_after = diversity_manager.calculate_diversity_metrics(population)
            
            diversity = metrics_after.overall_diversity
            intervened = metrics_after.overall_diversity != metrics_before.overall_diversity
        else:
            # Just calculate metrics without intervention
            temp_manager = GeneticDiversityManager()
            metrics = temp_manager.calculate_diversity_metrics(population)
            diversity = metrics.overall_diversity
            intervened = False
        
        diversity_history.append(diversity)
        
        # Print progress
        if gen % 10 == 0:
            print(f"Generation {gen:3d}: Diversity = {diversity:.3f} "
                  f"{'[Intervention applied]' if intervened else ''}")
    
    # Final report
    print("-" * 60)
    print("Simulation complete!")
    print(f"Final diversity: {diversity_history[-1]:.3f}")
    print(f"Average diversity: {np.mean(diversity_history):.3f}")
    print(f"Minimum diversity: {min(diversity_history):.3f}")
    
    if diversity_manager:
        report = diversity_manager.get_diversity_report()
        print(f"Total interventions: {report['interventions']['total']}")
        print(f"Emergency interventions: {report['interventions']['emergency']}")
    
    return diversity_history


def main():
    parser = argparse.ArgumentParser(description="Simulate population convergence")
    parser.add_argument('--generations', type=int, default=50,
                       help='Number of generations to simulate')
    parser.add_argument('--population', type=int, default=20,
                       help='Population size')
    parser.add_argument('--selection', type=float, default=0.7,
                       help='Selection pressure strength (0-1)')
    parser.add_argument('--assert-diversity', type=float, default=None,
                       help='Assert minimum diversity maintained')
    parser.add_argument('--no-management', action='store_true',
                       help='Disable diversity management')
    
    args = parser.parse_args()
    
    # Run with diversity management
    print("\n=== WITH Diversity Management ===")
    history_with = run_simulation(
        generations=args.generations,
        population_size=args.population,
        selection_strength=args.selection,
        min_diversity=args.assert_diversity or 0.3,
        use_diversity_manager=True
    )
    
    if args.no_management:
        # Run without diversity management for comparison
        print("\n=== WITHOUT Diversity Management ===")
        history_without = run_simulation(
            generations=args.generations,
            population_size=args.population,
            selection_strength=args.selection,
            min_diversity=0.3,
            use_diversity_manager=False
        )
    
    # Assert diversity if requested
    if args.assert_diversity:
        min_achieved = min(history_with)
        if min_achieved < args.assert_diversity:
            print(f"\nERROR: Minimum diversity {min_achieved:.3f} "
                  f"below threshold {args.assert_diversity}")
            return False
        else:
            print(f"\nSUCCESS: Minimum diversity {min_achieved:.3f} "
                  f"maintained above {args.assert_diversity}")
            return True
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)