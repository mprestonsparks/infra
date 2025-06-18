#!/usr/bin/env python3
"""
Diversity Manager for DEAN Evolution
Maintains genetic diversity in agent populations to prevent premature convergence
"""

import logging
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GeneticSignature:
    """Represents the genetic makeup of an agent"""
    agent_id: str
    strategies: List[str]
    prompt_patterns: List[str]
    lineage: List[str]
    generation: int
    mutations: int = 0
    crossovers: int = 0


@dataclass
class DiversityMetrics:
    """Population diversity measurements"""
    genome_similarity: float  # Average Jaccard similarity
    population_variance: float  # Standard deviation of similarity matrix
    unique_strategies: int  # Number of unique strategies in population
    convergence_index: float  # 0.0 = diverse, 1.0 = converged
    clustering_coefficient: float  # Degree of clustering in population
    entropy: float  # Shannon entropy of strategy distribution


class DiversityManager:
    """Manages genetic diversity in DEAN agent populations"""
    
    def __init__(self, 
                 min_diversity: float = 0.3,
                 convergence_threshold: float = 0.7,
                 mutation_rate: float = 0.15,
                 crossover_rate: float = 0.25):
        """
        Initialize diversity manager
        
        Args:
            min_diversity: Minimum acceptable diversity index
            convergence_threshold: Threshold for detecting convergence
            mutation_rate: Base mutation rate for interventions
            crossover_rate: Rate of genetic crossover
        """
        self.min_diversity = min_diversity
        self.convergence_threshold = convergence_threshold
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Population tracking
        self.population: Dict[str, GeneticSignature] = {}
        self.diversity_history: List[DiversityMetrics] = []
        self.interventions: List[Dict[str, any]] = []
        
        # Pattern library for foreign patterns
        self.pattern_library = self._load_pattern_library()
        
        logger.info(f"DiversityManager initialized: min_diversity={min_diversity}, mutation_rate={mutation_rate}")
    
    def register_agent(self, agent_id: str, strategies: List[str], 
                      lineage: List[str], generation: int):
        """Register a new agent in the population"""
        signature = GeneticSignature(
            agent_id=agent_id,
            strategies=strategies,
            prompt_patterns=self._extract_prompt_patterns(strategies),
            lineage=lineage,
            generation=generation
        )
        
        self.population[agent_id] = signature
        logger.info(f"Registered agent {agent_id} with {len(strategies)} strategies")
    
    def calculate_diversity_metrics(self) -> DiversityMetrics:
        """Calculate comprehensive diversity metrics for the population"""
        if len(self.population) < 2:
            return DiversityMetrics(
                genome_similarity=0.0,
                population_variance=1.0,
                unique_strategies=len(self._get_all_strategies()),
                convergence_index=0.0,
                clustering_coefficient=0.0,
                entropy=1.0
            )
        
        # Calculate pairwise similarities
        similarities = self._calculate_similarity_matrix()
        
        # Genome similarity (average Jaccard index)
        genome_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
        
        # Population variance
        population_variance = np.std(similarities[np.triu_indices_from(similarities, k=1)])
        
        # Unique strategies
        all_strategies = self._get_all_strategies()
        unique_strategies = len(all_strategies)
        
        # Convergence index (inverse of variance)
        convergence_index = 1.0 - population_variance if population_variance <= 1.0 else 0.0
        
        # Clustering coefficient
        clustering_coefficient = self._calculate_clustering_coefficient(similarities)
        
        # Shannon entropy of strategy distribution
        entropy = self._calculate_strategy_entropy()
        
        metrics = DiversityMetrics(
            genome_similarity=genome_similarity,
            population_variance=population_variance,
            unique_strategies=unique_strategies,
            convergence_index=convergence_index,
            clustering_coefficient=clustering_coefficient,
            entropy=entropy
        )
        
        # Record history
        self.diversity_history.append(metrics)
        
        return metrics
    
    def check_intervention_needed(self) -> Optional[str]:
        """Check if diversity intervention is needed"""
        metrics = self.calculate_diversity_metrics()
        
        # Check various diversity indicators
        if metrics.population_variance < self.min_diversity:
            return "low_variance"
        
        if metrics.convergence_index > self.convergence_threshold:
            return "high_convergence"
        
        if metrics.entropy < 0.5:  # Low strategy diversity
            return "low_entropy"
        
        if metrics.clustering_coefficient > 0.8:  # Too much clustering
            return "high_clustering"
        
        # Check for stagnation
        if self._detect_stagnation():
            return "stagnation"
        
        return None
    
    def force_mutation(self, agent_id: str, mutation_rate: Optional[float] = None) -> GeneticSignature:
        """Force mutation on an agent's genetic signature"""
        if agent_id not in self.population:
            raise ValueError(f"Agent {agent_id} not found in population")
        
        rate = mutation_rate or self.mutation_rate
        signature = self.population[agent_id]
        
        # Mutate strategies
        mutated_strategies = []
        for strategy in signature.strategies:
            if random.random() < rate:
                # Mutate the strategy
                mutated = self._mutate_strategy(strategy)
                mutated_strategies.append(mutated)
                logger.info(f"Mutated {strategy} -> {mutated} for agent {agent_id}")
            else:
                mutated_strategies.append(strategy)
        
        # Add random new strategy with some probability
        if random.random() < rate * 0.5:
            new_strategy = self._generate_novel_strategy()
            mutated_strategies.append(new_strategy)
            logger.info(f"Added novel strategy {new_strategy} to agent {agent_id}")
        
        # Update signature
        signature.strategies = mutated_strategies
        signature.prompt_patterns = self._extract_prompt_patterns(mutated_strategies)
        signature.mutations += 1
        
        # Record intervention
        self._record_intervention("mutation", agent_id, {
            "rate": rate,
            "strategies_mutated": len(mutated_strategies) - len(signature.strategies)
        })
        
        return signature
    
    def import_foreign_pattern(self, agent_id: str, source: str = "pattern_library") -> GeneticSignature:
        """Import a foreign pattern into an agent's genome"""
        if agent_id not in self.population:
            raise ValueError(f"Agent {agent_id} not found in population")
        
        signature = self.population[agent_id]
        
        # Select foreign pattern
        if source == "pattern_library":
            foreign_pattern = random.choice(self.pattern_library)
        else:
            # Get pattern from another population or external source
            foreign_pattern = self._get_external_pattern(source)
        
        # Integrate pattern
        if foreign_pattern not in signature.strategies:
            signature.strategies.append(foreign_pattern)
            signature.prompt_patterns = self._extract_prompt_patterns(signature.strategies)
            
            logger.info(f"Imported foreign pattern '{foreign_pattern}' into agent {agent_id}")
            
            # Record intervention
            self._record_intervention("foreign_import", agent_id, {
                "pattern": foreign_pattern,
                "source": source
            })
        
        return signature
    
    def crossover_agents(self, parent1_id: str, parent2_id: str) -> Tuple[str, GeneticSignature]:
        """Perform genetic crossover between two agents"""
        if parent1_id not in self.population or parent2_id not in self.population:
            raise ValueError("Both parent agents must be in population")
        
        parent1 = self.population[parent1_id]
        parent2 = self.population[parent2_id]
        
        # Create offspring ID
        offspring_id = f"agent_cross_{parent1_id[-4:]}_{parent2_id[-4:]}_{random.randint(1000, 9999)}"
        
        # Perform crossover
        crossover_point = random.randint(1, min(len(parent1.strategies), len(parent2.strategies)) - 1)
        
        # Combine strategies
        offspring_strategies = (parent1.strategies[:crossover_point] + 
                              parent2.strategies[crossover_point:])
        
        # Ensure uniqueness
        offspring_strategies = list(dict.fromkeys(offspring_strategies))
        
        # Create offspring signature
        offspring = GeneticSignature(
            agent_id=offspring_id,
            strategies=offspring_strategies,
            prompt_patterns=self._extract_prompt_patterns(offspring_strategies),
            lineage=parent1.lineage + parent2.lineage + [parent1_id, parent2_id],
            generation=max(parent1.generation, parent2.generation) + 1,
            crossovers=1
        )
        
        self.population[offspring_id] = offspring
        
        logger.info(f"Created offspring {offspring_id} from crossover of {parent1_id} x {parent2_id}")
        
        # Record intervention
        self._record_intervention("crossover", offspring_id, {
            "parent1": parent1_id,
            "parent2": parent2_id,
            "strategies_inherited": len(offspring_strategies)
        })
        
        return offspring_id, offspring
    
    def reset_stagnant_lineage(self, agent_id: str, generations_threshold: int = 5):
        """Reset a stagnant lineage with fresh genetic material"""
        if agent_id not in self.population:
            raise ValueError(f"Agent {agent_id} not found in population")
        
        signature = self.population[agent_id]
        
        # Check if truly stagnant (simplified check)
        if signature.mutations == 0 and signature.crossovers == 0:
            # Reset with new strategies
            base_strategies = ["exploration", "optimization", "refactoring"]
            novel_strategies = [self._generate_novel_strategy() for _ in range(2)]
            
            signature.strategies = base_strategies + novel_strategies
            signature.prompt_patterns = self._extract_prompt_patterns(signature.strategies)
            signature.generation = 0  # Reset generation counter
            
            logger.info(f"Reset stagnant lineage for agent {agent_id}")
            
            # Record intervention
            self._record_intervention("lineage_reset", agent_id, {
                "reason": "stagnation",
                "new_strategies": signature.strategies
            })
        
        return signature
    
    def get_diversity_report(self) -> Dict[str, any]:
        """Get comprehensive diversity report"""
        current_metrics = self.calculate_diversity_metrics()
        
        # Calculate trends if we have history
        trends = {}
        if len(self.diversity_history) >= 2:
            prev_metrics = self.diversity_history[-2]
            trends = {
                'variance_trend': current_metrics.population_variance - prev_metrics.population_variance,
                'entropy_trend': current_metrics.entropy - prev_metrics.entropy,
                'convergence_trend': current_metrics.convergence_index - prev_metrics.convergence_index
            }
        
        # Get population statistics
        strategy_counts = defaultdict(int)
        for signature in self.population.values():
            for strategy in signature.strategies:
                strategy_counts[strategy] += 1
        
        most_common_strategies = sorted(strategy_counts.items(), 
                                      key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'current_metrics': {
                'genome_similarity': current_metrics.genome_similarity,
                'population_variance': current_metrics.population_variance,
                'unique_strategies': current_metrics.unique_strategies,
                'convergence_index': current_metrics.convergence_index,
                'clustering_coefficient': current_metrics.clustering_coefficient,
                'entropy': current_metrics.entropy
            },
            'population_size': len(self.population),
            'total_interventions': len(self.interventions),
            'recent_interventions': self.interventions[-5:] if self.interventions else [],
            'most_common_strategies': most_common_strategies,
            'trends': trends,
            'diversity_status': self._get_diversity_status(current_metrics)
        }
    
    def _calculate_similarity_matrix(self) -> np.ndarray:
        """Calculate pairwise Jaccard similarity between all agents"""
        agent_ids = list(self.population.keys())
        n = len(agent_ids)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sig1 = self.population[agent_ids[i]]
                    sig2 = self.population[agent_ids[j]]
                    
                    # Jaccard similarity of strategies
                    set1 = set(sig1.strategies)
                    set2 = set(sig2.strategies)
                    
                    if not set1 and not set2:
                        similarity = 0.0
                    else:
                        intersection = len(set1 & set2)
                        union = len(set1 | set2)
                        similarity = intersection / union if union > 0 else 0.0
                    
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
        
        return similarity_matrix
    
    def _calculate_clustering_coefficient(self, similarity_matrix: np.ndarray) -> float:
        """Calculate clustering coefficient based on similarity threshold"""
        threshold = 0.7  # Agents with >70% similarity are considered connected
        
        # Create adjacency matrix
        adjacency = (similarity_matrix > threshold).astype(int)
        np.fill_diagonal(adjacency, 0)  # No self-connections
        
        # Calculate clustering coefficient
        n = len(adjacency)
        clustering_coeffs = []
        
        for i in range(n):
            neighbors = np.where(adjacency[i] == 1)[0]
            k = len(neighbors)
            
            if k >= 2:
                # Count edges between neighbors
                edges = 0
                for j in range(k):
                    for m in range(j + 1, k):
                        if adjacency[neighbors[j], neighbors[m]] == 1:
                            edges += 1
                
                # Local clustering coefficient
                max_edges = k * (k - 1) / 2
                local_cc = edges / max_edges if max_edges > 0 else 0.0
                clustering_coeffs.append(local_cc)
        
        return np.mean(clustering_coeffs) if clustering_coeffs else 0.0
    
    def _calculate_strategy_entropy(self) -> float:
        """Calculate Shannon entropy of strategy distribution"""
        strategy_counts = defaultdict(int)
        total_strategies = 0
        
        for signature in self.population.values():
            for strategy in signature.strategies:
                strategy_counts[strategy] += 1
                total_strategies += 1
        
        if total_strategies == 0:
            return 0.0
        
        # Calculate probabilities and entropy
        entropy = 0.0
        for count in strategy_counts.values():
            p = count / total_strategies
            if p > 0:
                entropy -= p * np.log2(p)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(strategy_counts)) if len(strategy_counts) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy
    
    def _get_all_strategies(self) -> Set[str]:
        """Get all unique strategies in the population"""
        all_strategies = set()
        for signature in self.population.values():
            all_strategies.update(signature.strategies)
        return all_strategies
    
    def _extract_prompt_patterns(self, strategies: List[str]) -> List[str]:
        """Extract prompt patterns from strategies"""
        patterns = []
        for strategy in strategies:
            # Simple pattern extraction (can be enhanced)
            if "_" in strategy:
                pattern = strategy.split("_")[0]
                patterns.append(pattern)
            else:
                patterns.append(strategy[:4])
        return patterns
    
    def _mutate_strategy(self, strategy: str) -> str:
        """Mutate a strategy string"""
        mutations = [
            lambda s: s + "_enhanced",
            lambda s: "adaptive_" + s,
            lambda s: s.replace("_", "_deep_"),
            lambda s: "parallel_" + s,
            lambda s: s + "_v2",
            lambda s: "experimental_" + s
        ]
        
        mutation_func = random.choice(mutations)
        return mutation_func(strategy)
    
    def _generate_novel_strategy(self) -> str:
        """Generate a completely novel strategy"""
        prefixes = ["quantum", "neural", "hybrid", "meta", "dynamic", "adaptive"]
        cores = ["search", "optimize", "evolve", "transform", "analyze", "synthesize"]
        suffixes = ["parallel", "recursive", "iterative", "cascade", "fusion"]
        
        components = []
        if random.random() > 0.5:
            components.append(random.choice(prefixes))
        components.append(random.choice(cores))
        if random.random() > 0.5:
            components.append(random.choice(suffixes))
        
        return "_".join(components)
    
    def _load_pattern_library(self) -> List[str]:
        """Load foreign patterns from library"""
        return [
            "memoization_optimization",
            "parallel_execution",
            "lazy_evaluation",
            "cache_warming",
            "predictive_prefetch",
            "adaptive_sampling",
            "progressive_refinement",
            "hierarchical_decomposition",
            "constraint_propagation",
            "speculative_execution"
        ]
    
    def _get_external_pattern(self, source: str) -> str:
        """Get pattern from external source"""
        # In production, this would fetch from external systems
        external_patterns = {
            "research": ["academic_optimization", "theoretical_approach"],
            "industry": ["production_hardening", "scale_optimization"],
            "nature": ["swarm_intelligence", "evolutionary_pressure"]
        }
        
        patterns = external_patterns.get(source, self.pattern_library)
        return random.choice(patterns)
    
    def _detect_stagnation(self, window: int = 5) -> bool:
        """Detect if population is stagnating"""
        if len(self.diversity_history) < window:
            return False
        
        recent_history = self.diversity_history[-window:]
        
        # Check if variance has been consistently low
        variances = [m.population_variance for m in recent_history]
        avg_variance = np.mean(variances)
        
        # Check if entropy is decreasing
        entropies = [m.entropy for m in recent_history]
        entropy_trend = np.polyfit(range(len(entropies)), entropies, 1)[0]
        
        return avg_variance < self.min_diversity and entropy_trend < -0.01
    
    def _record_intervention(self, intervention_type: str, agent_id: str, details: Dict[str, any]):
        """Record a diversity intervention"""
        intervention = {
            'type': intervention_type,
            'agent_id': agent_id,
            'timestamp': datetime.now().isoformat(),
            'details': details,
            'metrics_before': self.diversity_history[-1] if self.diversity_history else None
        }
        
        self.interventions.append(intervention)
        
        # Save to file for analysis
        log_dir = Path("logs/diversity")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"intervention_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_dir / filename, 'w') as f:
            json.dump(intervention, f, indent=2, default=str)
    
    def _get_diversity_status(self, metrics: DiversityMetrics) -> str:
        """Get human-readable diversity status"""
        if metrics.population_variance < self.min_diversity:
            return "CRITICAL: Low diversity"
        elif metrics.convergence_index > self.convergence_threshold:
            return "WARNING: High convergence"
        elif metrics.entropy < 0.5:
            return "WARNING: Low strategy diversity"
        elif metrics.population_variance > 0.7:
            return "HEALTHY: High diversity"
        else:
            return "STABLE: Acceptable diversity"


if __name__ == "__main__":
    # Demo the diversity manager
    manager = DiversityManager()
    
    print("Diversity Manager Demo")
    print("=" * 60)
    
    # Create initial population
    print("\n1. Creating initial population...")
    agents = [
        ("agent_001", ["optimize", "refactor", "test"]),
        ("agent_002", ["optimize", "refactor", "debug"]),
        ("agent_003", ["optimize", "analyze", "test"]),
        ("agent_004", ["explore", "experiment", "innovate"]),
        ("agent_005", ["optimize", "refactor", "test"])  # Clone of agent_001
    ]
    
    for agent_id, strategies in agents:
        manager.register_agent(agent_id, strategies, [], 1)
    
    # Calculate initial diversity
    print("\n2. Initial diversity metrics:")
    metrics = manager.calculate_diversity_metrics()
    print(f"   Population variance: {metrics.population_variance:.3f}")
    print(f"   Convergence index: {metrics.convergence_index:.3f}")
    print(f"   Unique strategies: {metrics.unique_strategies}")
    print(f"   Entropy: {metrics.entropy:.3f}")
    
    # Check if intervention needed
    print("\n3. Checking for intervention...")
    intervention = manager.check_intervention_needed()
    print(f"   Intervention needed: {intervention or 'None'}")
    
    # Force mutations
    print("\n4. Applying mutations...")
    manager.force_mutation("agent_001")
    manager.force_mutation("agent_002")
    
    # Import foreign pattern
    print("\n5. Importing foreign patterns...")
    manager.import_foreign_pattern("agent_003")
    
    # Perform crossover
    print("\n6. Performing crossover...")
    offspring_id, offspring = manager.crossover_agents("agent_004", "agent_002")
    print(f"   Created offspring: {offspring_id}")
    print(f"   Strategies: {offspring.strategies}")
    
    # Final diversity check
    print("\n7. Final diversity metrics:")
    final_metrics = manager.calculate_diversity_metrics()
    print(f"   Population variance: {final_metrics.population_variance:.3f}")
    print(f"   Convergence index: {final_metrics.convergence_index:.3f}")
    print(f"   Unique strategies: {final_metrics.unique_strategies}")
    print(f"   Entropy: {final_metrics.entropy:.3f}")
    
    # Get report
    print("\n8. Diversity report:")
    report = manager.get_diversity_report()
    print(f"   Status: {report['diversity_status']}")
    print(f"   Population size: {report['population_size']}")
    print(f"   Total interventions: {report['total_interventions']}")
    
    print("\nDemo complete!")