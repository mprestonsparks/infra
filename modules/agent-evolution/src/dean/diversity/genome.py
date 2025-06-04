"""
Agent genome representation and manipulation.

Defines the genetic structure of agents and provides methods
for genome comparison, serialization, and manipulation.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class GeneType(Enum):
    """Types of genes that can exist in an agent genome."""
    PROMPT_TEMPLATE = "prompt_template"
    STRATEGY = "strategy"
    HYPERPARAMETER = "hyperparameter"
    MODULE_CONFIG = "module_config"
    BEHAVIOR_RULE = "behavior_rule"
    OPTIMIZATION_HINT = "optimization_hint"


@dataclass
class Gene:
    """Individual gene within an agent's genome."""
    gene_type: GeneType
    name: str
    value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def mutate(self, mutation_rate: float = 0.1) -> 'Gene':
        """Apply mutation to this gene."""
        if np.random.random() > mutation_rate:
            return self
        
        # Different mutation strategies based on gene type
        if self.gene_type == GeneType.HYPERPARAMETER:
            return self._mutate_hyperparameter()
        elif self.gene_type == GeneType.PROMPT_TEMPLATE:
            return self._mutate_prompt()
        elif self.gene_type == GeneType.STRATEGY:
            return self._mutate_strategy()
        else:
            # Generic mutation - slight modification
            return Gene(
                gene_type=self.gene_type,
                name=self.name,
                value=self._generic_mutate(self.value),
                metadata={**self.metadata, 'mutated': True}
            )
    
    def _mutate_hyperparameter(self) -> 'Gene':
        """Mutate numerical hyperparameters."""
        if isinstance(self.value, (int, float)):
            # Add gaussian noise
            noise_scale = abs(self.value) * 0.2 if self.value != 0 else 0.1
            new_value = self.value + np.random.normal(0, noise_scale)
            
            # Preserve type
            if isinstance(self.value, int):
                new_value = int(round(new_value))
            
            return Gene(
                gene_type=self.gene_type,
                name=self.name,
                value=new_value,
                metadata={**self.metadata, 'mutated': True, 'original': self.value}
            )
        return self
    
    def _mutate_prompt(self) -> 'Gene':
        """Mutate prompt templates."""
        if isinstance(self.value, str):
            mutations = [
                lambda s: s.replace("Please", "Kindly"),
                lambda s: s.replace("analyze", "examine"),
                lambda s: s.replace("identify", "discover"),
                lambda s: s + "\nBe concise in your response.",
                lambda s: s + "\nProvide step-by-step reasoning.",
                lambda s: s.replace("important", "crucial"),
            ]
            
            # Apply random mutation
            mutation = np.random.choice(mutations)
            return Gene(
                gene_type=self.gene_type,
                name=self.name,
                value=mutation(self.value),
                metadata={**self.metadata, 'mutated': True}
            )
        return self
    
    def _mutate_strategy(self) -> 'Gene':
        """Mutate strategy genes."""
        if isinstance(self.value, dict):
            # Mutate a random key in the strategy
            if self.value:
                key = np.random.choice(list(self.value.keys()))
                new_value = {**self.value}
                new_value[key] = self._generic_mutate(new_value[key])
                
                return Gene(
                    gene_type=self.gene_type,
                    name=self.name,
                    value=new_value,
                    metadata={**self.metadata, 'mutated': True, 'mutated_key': key}
                )
        return self
    
    def _generic_mutate(self, value: Any) -> Any:
        """Generic mutation for any value type."""
        if isinstance(value, bool):
            return not value
        elif isinstance(value, (int, float)):
            return value * np.random.uniform(0.8, 1.2)
        elif isinstance(value, str):
            # Simple character mutation
            if len(value) > 0 and np.random.random() < 0.3:
                idx = np.random.randint(len(value))
                chars = list(value)
                chars[idx] = chr(ord(chars[idx]) + np.random.randint(-5, 5))
                return ''.join(chars)
        return value
    
    def to_dict(self) -> Dict:
        """Serialize gene to dictionary."""
        return {
            'gene_type': self.gene_type.value,
            'name': self.name,
            'value': self.value,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Gene':
        """Deserialize gene from dictionary."""
        return cls(
            gene_type=GeneType(data['gene_type']),
            name=data['name'],
            value=data['value'],
            metadata=data.get('metadata', {})
        )


@dataclass
class AgentGenome:
    """
    Complete genetic representation of an agent.
    
    Contains all heritable traits that define agent behavior,
    including prompts, strategies, hyperparameters, and learned optimizations.
    """
    
    genes: Dict[str, Gene] = field(default_factory=dict)
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_history: List[Dict] = field(default_factory=list)
    fitness_scores: Dict[str, float] = field(default_factory=dict)
    
    def add_gene(self, gene: Gene) -> None:
        """Add a gene to the genome."""
        self.genes[gene.name] = gene
    
    def get_gene(self, name: str) -> Optional[Gene]:
        """Get a gene by name."""
        return self.genes.get(name)
    
    def remove_gene(self, name: str) -> None:
        """Remove a gene from the genome."""
        if name in self.genes:
            del self.genes[name]
    
    def get_genes_by_type(self, gene_type: GeneType) -> List[Gene]:
        """Get all genes of a specific type."""
        return [g for g in self.genes.values() if g.gene_type == gene_type]
    
    def calculate_hash(self) -> str:
        """Calculate a unique hash for this genome."""
        # Sort genes for consistent hashing
        sorted_genes = sorted(
            [(k, v.to_dict()) for k, v in self.genes.items()],
            key=lambda x: x[0]
        )
        
        genome_str = json.dumps({
            'genes': sorted_genes,
            'generation': self.generation
        }, sort_keys=True)
        
        return hashlib.sha256(genome_str.encode()).hexdigest()[:16]
    
    def distance_from(self, other: 'AgentGenome') -> float:
        """
        Calculate genetic distance from another genome.
        
        Returns a value between 0 (identical) and 1 (completely different).
        """
        if not self.genes and not other.genes:
            return 0.0
        
        all_gene_names = set(self.genes.keys()) | set(other.genes.keys())
        if not all_gene_names:
            return 0.0
        
        differences = 0
        comparisons = 0
        
        for gene_name in all_gene_names:
            gene1 = self.genes.get(gene_name)
            gene2 = other.genes.get(gene_name)
            
            if gene1 is None or gene2 is None:
                # Gene present in one but not the other
                differences += 1
                comparisons += 1
            else:
                # Compare gene values
                similarity = self._compare_gene_values(gene1, gene2)
                differences += (1 - similarity)
                comparisons += 1
        
        return differences / comparisons if comparisons > 0 else 0.0
    
    def _compare_gene_values(self, gene1: Gene, gene2: Gene) -> float:
        """Compare two gene values and return similarity (0-1)."""
        if gene1.gene_type != gene2.gene_type:
            return 0.0
        
        val1, val2 = gene1.value, gene2.value
        
        # Type-specific comparison
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            # Numerical similarity
            if val1 == val2:
                return 1.0
            max_val = max(abs(val1), abs(val2))
            if max_val == 0:
                return 1.0
            return 1.0 - min(1.0, abs(val1 - val2) / max_val)
        
        elif isinstance(val1, str) and isinstance(val2, str):
            # String similarity (simple character overlap)
            if val1 == val2:
                return 1.0
            longer = max(len(val1), len(val2))
            if longer == 0:
                return 1.0
            
            # Calculate Levenshtein distance
            return 1.0 - (self._levenshtein_distance(val1, val2) / longer)
        
        elif isinstance(val1, bool) and isinstance(val2, bool):
            return 1.0 if val1 == val2 else 0.0
        
        elif isinstance(val1, dict) and isinstance(val2, dict):
            # Dictionary similarity
            all_keys = set(val1.keys()) | set(val2.keys())
            if not all_keys:
                return 1.0
            
            matches = sum(1 for k in all_keys if val1.get(k) == val2.get(k))
            return matches / len(all_keys)
        
        else:
            # Generic comparison
            return 1.0 if val1 == val2 else 0.0
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def clone(self) -> 'AgentGenome':
        """Create a deep copy of this genome."""
        return AgentGenome(
            genes={k: Gene(
                gene_type=v.gene_type,
                name=v.name,
                value=v.value.copy() if hasattr(v.value, 'copy') else v.value,
                metadata=v.metadata.copy()
            ) for k, v in self.genes.items()},
            generation=self.generation,
            parent_ids=self.parent_ids.copy(),
            mutation_history=self.mutation_history.copy(),
            fitness_scores=self.fitness_scores.copy()
        )
    
    def to_dict(self) -> Dict:
        """Serialize genome to dictionary."""
        return {
            'genes': {k: v.to_dict() for k, v in self.genes.items()},
            'generation': self.generation,
            'parent_ids': self.parent_ids,
            'mutation_history': self.mutation_history,
            'fitness_scores': self.fitness_scores,
            'genome_hash': self.calculate_hash()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AgentGenome':
        """Deserialize genome from dictionary."""
        genome = cls(
            generation=data['generation'],
            parent_ids=data.get('parent_ids', []),
            mutation_history=data.get('mutation_history', []),
            fitness_scores=data.get('fitness_scores', {})
        )
        
        for name, gene_data in data['genes'].items():
            genome.genes[name] = Gene.from_dict(gene_data)
        
        return genome
    
    def summary(self) -> str:
        """Get a human-readable summary of the genome."""
        gene_counts = {}
        for gene in self.genes.values():
            gene_type = gene.gene_type.value
            gene_counts[gene_type] = gene_counts.get(gene_type, 0) + 1
        
        summary_parts = [
            f"Generation: {self.generation}",
            f"Total genes: {len(self.genes)}",
            f"Gene types: {', '.join(f'{k}={v}' for k, v in gene_counts.items())}",
            f"Parents: {len(self.parent_ids)}",
            f"Mutations: {len(self.mutation_history)}",
            f"Hash: {self.calculate_hash()}"
        ]
        
        return " | ".join(summary_parts)