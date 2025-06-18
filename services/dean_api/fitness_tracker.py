#!/usr/bin/env python3
"""
Phase 5: Fitness Progression System
Concrete metrics for tracking agent evolution progress.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text


class FitnessComponent(Enum):
    """Components that contribute to overall fitness."""
    CODE_QUALITY = "code_quality"           # Reduced complexity, better structure
    PERFORMANCE = "performance"             # Speed improvements
    PATTERN_DISCOVERY = "pattern_discovery" # Finding reusable patterns
    TOKEN_EFFICIENCY = "token_efficiency"   # Value per token spent
    GOAL_ALIGNMENT = "goal_alignment"       # Progress toward stated goal
    INNOVATION = "innovation"               # Novel solutions


@dataclass
class FitnessMetric:
    """Represents a measurable fitness metric."""
    component: FitnessComponent
    value: float  # 0.0 to 1.0
    weight: float = 1.0
    evidence: Dict = field(default_factory=dict)
    
    @property
    def weighted_value(self) -> float:
        return self.value * self.weight


@dataclass 
class FitnessSnapshot:
    """A point-in-time fitness measurement."""
    agent_id: str
    generation: int
    timestamp: datetime
    metrics: List[FitnessMetric]
    total_fitness: float
    token_cost: int
    patterns_applied: List[str]
    
    def get_component_value(self, component: FitnessComponent) -> float:
        """Get value for specific component."""
        for metric in self.metrics:
            if metric.component == component:
                return metric.value
        return 0.0


class FitnessProgressionTracker:
    """
    Tracks and calculates concrete fitness progression.
    Uses multiple metrics to create a comprehensive fitness score.
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.component_weights = {
            FitnessComponent.CODE_QUALITY: 2.0,      # High priority
            FitnessComponent.PERFORMANCE: 1.5,
            FitnessComponent.PATTERN_DISCOVERY: 1.5,
            FitnessComponent.TOKEN_EFFICIENCY: 1.0,
            FitnessComponent.GOAL_ALIGNMENT: 1.0,
            FitnessComponent.INNOVATION: 0.5
        }
        
    def calculate_code_quality_fitness(self, agent_id: str, 
                                     before_metrics: Dict, after_metrics: Dict) -> FitnessMetric:
        """Calculate fitness from code quality improvements."""
        improvements = 0
        total_checks = 0
        
        evidence = {
            'before': before_metrics,
            'after': after_metrics,
            'improvements': []
        }
        
        # Check TODO reduction
        if before_metrics.get('todo_count', 0) > after_metrics.get('todo_count', 0):
            improvements += 1
            evidence['improvements'].append('reduced_todos')
        total_checks += 1
        
        # Check complexity reduction
        if before_metrics.get('complexity', 0) > after_metrics.get('complexity', 0):
            improvements += 1
            evidence['improvements'].append('reduced_complexity')
        total_checks += 1
        
        # Check unused import cleanup
        if before_metrics.get('unused_imports', 0) > after_metrics.get('unused_imports', 0):
            improvements += 1
            evidence['improvements'].append('cleaned_imports')
        total_checks += 1
        
        # Check long function refactoring
        if before_metrics.get('long_functions', 0) > after_metrics.get('long_functions', 0):
            improvements += 1
            evidence['improvements'].append('refactored_functions')
        total_checks += 1
        
        # Calculate score
        value = improvements / total_checks if total_checks > 0 else 0.0
        
        return FitnessMetric(
            component=FitnessComponent.CODE_QUALITY,
            value=value,
            weight=self.component_weights[FitnessComponent.CODE_QUALITY],
            evidence=evidence
        )
    
    def calculate_performance_fitness(self, agent_id: str, 
                                    pattern_type: str, effectiveness: float) -> FitnessMetric:
        """Calculate fitness from performance improvements."""
        # Map pattern types to performance impact
        performance_impact = {
            'optimization': 0.8,
            'memoization': 0.9,
            'algorithm_improvement': 1.0,
            'caching': 0.7,
            'refactoring': 0.5,
            'cleanup': 0.3
        }
        
        base_impact = performance_impact.get(pattern_type, 0.4)
        value = base_impact * effectiveness
        
        return FitnessMetric(
            component=FitnessComponent.PERFORMANCE,
            value=value,
            weight=self.component_weights[FitnessComponent.PERFORMANCE],
            evidence={
                'pattern_type': pattern_type,
                'effectiveness': effectiveness,
                'impact_factor': base_impact
            }
        )
    
    def calculate_pattern_discovery_fitness(self, agent_id: str, 
                                          patterns_found: int, reuse_potential: float) -> FitnessMetric:
        """Calculate fitness from pattern discovery."""
        # Value increases with both quantity and quality
        if patterns_found == 0:
            value = 0.0
        else:
            # Diminishing returns on quantity, emphasis on reuse potential
            quantity_score = min(1.0, patterns_found / 5.0)
            value = (quantity_score * 0.4) + (reuse_potential * 0.6)
            
        return FitnessMetric(
            component=FitnessComponent.PATTERN_DISCOVERY,
            value=value,
            weight=self.component_weights[FitnessComponent.PATTERN_DISCOVERY],
            evidence={
                'patterns_found': patterns_found,
                'reuse_potential': reuse_potential
            }
        )
    
    def calculate_token_efficiency_fitness(self, agent_id: str) -> FitnessMetric:
        """Calculate fitness from token efficiency."""
        # Get agent's token usage
        result = self.db.execute(text("""
            SELECT token_budget, token_consumed, fitness_score
            FROM agent_evolution.agents
            WHERE id = :id
        """), {"id": agent_id}).fetchone()
        
        if not result or result.token_consumed == 0:
            return FitnessMetric(
                component=FitnessComponent.TOKEN_EFFICIENCY,
                value=0.5,
                weight=self.component_weights[FitnessComponent.TOKEN_EFFICIENCY]
            )
            
        # Calculate efficiency as fitness gained per 1000 tokens
        efficiency = (result.fitness_score * 1000) / result.token_consumed
        
        # Normalize to 0-1 range (assuming 1.0 efficiency is excellent)
        value = min(1.0, efficiency)
        
        return FitnessMetric(
            component=FitnessComponent.TOKEN_EFFICIENCY,
            value=value,
            weight=self.component_weights[FitnessComponent.TOKEN_EFFICIENCY],
            evidence={
                'tokens_consumed': result.token_consumed,
                'fitness_per_1k_tokens': efficiency,
                'total_fitness': result.fitness_score
            }
        )
    
    def calculate_goal_alignment_fitness(self, agent_id: str, goal: str, 
                                       actions_taken: List[str]) -> FitnessMetric:
        """Calculate fitness based on alignment with stated goal."""
        # Simple keyword matching for demonstration
        goal_keywords = {
            'optimize': ['optimization', 'performance', 'speed', 'efficiency'],
            'reduce': ['reduction', 'cleanup', 'simplify', 'refactor'],
            'implement': ['implementation', 'todo', 'feature', 'functionality'],
            'improve': ['improvement', 'enhancement', 'better', 'quality'],
            'pattern': ['pattern', 'discovery', 'reusable', 'template']
        }
        
        alignment_score = 0.0
        matches = 0
        
        goal_lower = goal.lower()
        for keyword, related_terms in goal_keywords.items():
            if keyword in goal_lower:
                # Check if actions align with goal
                for action in actions_taken:
                    action_lower = action.lower()
                    if any(term in action_lower for term in related_terms):
                        matches += 1
                        
        if actions_taken:
            alignment_score = min(1.0, matches / len(actions_taken))
            
        return FitnessMetric(
            component=FitnessComponent.GOAL_ALIGNMENT,
            value=alignment_score,
            weight=self.component_weights[FitnessComponent.GOAL_ALIGNMENT],
            evidence={
                'goal': goal,
                'actions': actions_taken,
                'alignment_matches': matches
            }
        )
    
    def calculate_innovation_fitness(self, agent_id: str, generation: int,
                                   pattern_novelty: float) -> FitnessMetric:
        """Calculate fitness from innovative solutions."""
        # Innovation increases with generation and pattern novelty
        generation_factor = min(1.0, generation / 50.0)  # Max at gen 50
        value = (generation_factor * 0.3) + (pattern_novelty * 0.7)
        
        return FitnessMetric(
            component=FitnessComponent.INNOVATION,
            value=value,
            weight=self.component_weights[FitnessComponent.INNOVATION],
            evidence={
                'generation': generation,
                'generation_factor': generation_factor,
                'pattern_novelty': pattern_novelty
            }
        )
    
    def calculate_total_fitness(self, agent_id: str, generation: int,
                              evolution_data: Dict) -> FitnessSnapshot:
        """Calculate comprehensive fitness score."""
        metrics = []
        
        # 1. Code quality fitness
        if 'before_metrics' in evolution_data and 'after_metrics' in evolution_data:
            code_quality = self.calculate_code_quality_fitness(
                agent_id, 
                evolution_data['before_metrics'],
                evolution_data['after_metrics']
            )
            metrics.append(code_quality)
            
        # 2. Performance fitness
        if 'pattern_type' in evolution_data:
            performance = self.calculate_performance_fitness(
                agent_id,
                evolution_data['pattern_type'],
                evolution_data.get('effectiveness', 0.5)
            )
            metrics.append(performance)
            
        # 3. Pattern discovery fitness
        patterns_found = evolution_data.get('patterns_discovered', 0)
        pattern_discovery = self.calculate_pattern_discovery_fitness(
            agent_id, patterns_found, 
            evolution_data.get('reuse_potential', 0.5)
        )
        metrics.append(pattern_discovery)
        
        # 4. Token efficiency
        token_efficiency = self.calculate_token_efficiency_fitness(agent_id)
        metrics.append(token_efficiency)
        
        # 5. Goal alignment
        if 'goal' in evolution_data and 'actions' in evolution_data:
            goal_alignment = self.calculate_goal_alignment_fitness(
                agent_id,
                evolution_data['goal'],
                evolution_data['actions']
            )
            metrics.append(goal_alignment)
            
        # 6. Innovation
        innovation = self.calculate_innovation_fitness(
            agent_id, generation,
            evolution_data.get('pattern_novelty', 0.3)
        )
        metrics.append(innovation)
        
        # Calculate total fitness
        total_weighted = sum(m.weighted_value for m in metrics)
        total_weights = sum(m.weight for m in metrics)
        total_fitness = total_weighted / total_weights if total_weights > 0 else 0.0
        
        # Create snapshot
        snapshot = FitnessSnapshot(
            agent_id=agent_id,
            generation=generation,
            timestamp=datetime.utcnow(),
            metrics=metrics,
            total_fitness=total_fitness,
            token_cost=evolution_data.get('token_cost', 0),
            patterns_applied=evolution_data.get('patterns_applied', [])
        )
        
        # Store in database
        self._store_fitness_snapshot(snapshot)
        
        return snapshot
    
    def _store_fitness_snapshot(self, snapshot: FitnessSnapshot):
        """Store fitness snapshot in database."""
        # Store as performance metric
        for metric in snapshot.metrics:
            self.db.execute(text("""
                INSERT INTO agent_evolution.performance_metrics
                (agent_id, metric_type, metric_value, tokens_used, task_type)
                VALUES (:agent_id, :metric_type, :metric_value, :tokens, :task)
            """), {
                "agent_id": snapshot.agent_id,
                "metric_type": f"fitness_{metric.component.value}",
                "metric_value": metric.value,
                "tokens": snapshot.token_cost // len(snapshot.metrics),
                "task": f"generation_{snapshot.generation}"
            })
            
        # Store total fitness
        self.db.execute(text("""
            INSERT INTO agent_evolution.performance_metrics
            (agent_id, metric_type, metric_value, tokens_used, task_type)
            VALUES (:agent_id, :metric_type, :metric_value, :tokens, :task)
        """), {
            "agent_id": snapshot.agent_id,
            "metric_type": "fitness_total",
            "metric_value": snapshot.total_fitness,
            "tokens": snapshot.token_cost,
            "task": f"generation_{snapshot.generation}"
        })
        
        self.db.commit()
    
    def get_fitness_progression(self, agent_id: str, 
                              start_date: Optional[datetime] = None) -> List[Dict]:
        """Get fitness progression over time."""
        query = """
            SELECT 
                timestamp,
                metric_type,
                metric_value,
                tokens_used,
                task_type
            FROM agent_evolution.performance_metrics
            WHERE agent_id = :agent_id
                AND metric_type LIKE 'fitness_%'
        """
        
        params = {"agent_id": agent_id}
        
        if start_date:
            query += " AND timestamp >= :start_date"
            params["start_date"] = start_date
            
        query += " ORDER BY timestamp"
        
        result = self.db.execute(text(query), params)
        
        progression = []
        for row in result:
            progression.append({
                'timestamp': row.timestamp.isoformat() if row.timestamp else None,
                'metric_type': row.metric_type,
                'value': float(row.metric_value),
                'tokens_used': row.tokens_used,
                'generation': int(row.task_type.split('_')[1]) if '_' in row.task_type else 0
            })
            
        return progression
    
    def calculate_improvement_rate(self, agent_id: str, window_days: int = 7) -> float:
        """Calculate rate of fitness improvement."""
        start_date = datetime.utcnow() - timedelta(days=window_days)
        progression = self.get_fitness_progression(agent_id, start_date)
        
        if len(progression) < 2:
            return 0.0
            
        # Get total fitness values
        fitness_values = [
            p['value'] for p in progression 
            if p['metric_type'] == 'fitness_total'
        ]
        
        if len(fitness_values) < 2:
            return 0.0
            
        # Calculate improvement rate (fitness gain per generation)
        first_fitness = fitness_values[0]
        last_fitness = fitness_values[-1]
        generations = len(fitness_values) - 1
        
        if generations > 0:
            return (last_fitness - first_fitness) / generations
        else:
            return 0.0
    
    def create_fitness_report(self, agent_id: str) -> Dict:
        """Create comprehensive fitness report for an agent."""
        # Get agent info
        agent_result = self.db.execute(text("""
            SELECT name, goal, generation, fitness_score, token_consumed
            FROM agent_evolution.agents
            WHERE id = :id
        """), {"id": agent_id}).fetchone()
        
        if not agent_result:
            return {}
            
        # Get fitness progression
        progression = self.get_fitness_progression(agent_id)
        
        # Calculate current component values
        latest_metrics = {}
        for p in reversed(progression):
            if p['metric_type'].startswith('fitness_') and p['metric_type'] != 'fitness_total':
                component = p['metric_type'].replace('fitness_', '')
                if component not in latest_metrics:
                    latest_metrics[component] = p['value']
                    
        # Calculate improvement rate
        improvement_rate = self.calculate_improvement_rate(agent_id)
        
        report = {
            'agent_id': agent_id,
            'agent_name': agent_result.name,
            'goal': agent_result.goal,
            'current_generation': agent_result.generation,
            'current_fitness': agent_result.fitness_score,
            'tokens_consumed': agent_result.token_consumed,
            'fitness_components': latest_metrics,
            'improvement_rate': improvement_rate,
            'progression_points': len(progression),
            'efficiency': agent_result.fitness_score / (agent_result.token_consumed / 1000) if agent_result.token_consumed > 0 else 0
        }
        
        return report