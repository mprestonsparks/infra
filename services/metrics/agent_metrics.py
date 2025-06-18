#!/usr/bin/env python3
"""
Prometheus Metrics for DEAN System
Exposes agent evolution metrics for monitoring
"""

from prometheus_client import Counter, Gauge, Histogram, Summary, Info
from prometheus_client import start_http_server, generate_latest
import time
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Define metrics

# Agent success metrics
agent_success_rate = Gauge(
    'dean_agent_success_rate',
    'Success rate of agent actions',
    ['action_type', 'agent_id']
)

# Token efficiency metrics
token_efficiency = Gauge(
    'dean_token_efficiency',
    'Token efficiency (success per token) for agents',
    ['agent_id']
)

# Diversity metrics
diversity_index = Gauge(
    'dean_diversity_index',
    'Population diversity index (0-1)',
    []
)

population_variance = Gauge(
    'dean_population_variance',
    'Genetic variance in agent population',
    []
)

convergence_index = Gauge(
    'dean_convergence_index',
    'Population convergence index (0-1)',
    []
)

# Pattern discovery metrics
patterns_discovered_total = Counter(
    'dean_patterns_discovered_total',
    'Total number of patterns discovered',
    ['pattern_type']
)

pattern_reuse_count = Counter(
    'dean_pattern_reuse_count',
    'Number of times patterns have been reused',
    ['pattern_id']
)

# CA rule triggers
ca_rule_triggers = Counter(
    'dean_ca_rule_triggers',
    'Number of times each CA rule has triggered',
    ['rule_id']
)

# Economic metrics
token_budget_allocated = Gauge(
    'dean_token_budget_allocated',
    'Tokens allocated to agents',
    []
)

token_budget_used = Gauge(
    'dean_token_budget_used',
    'Tokens actually used by agents',
    []
)

agent_budget_current = Gauge(
    'dean_agent_budget_current',
    'Current token budget for agent',
    ['agent_id']
)

# Performance metrics
action_execution_time = Histogram(
    'dean_action_execution_seconds',
    'Time taken to execute agent actions',
    ['action_type'],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600)
)

agent_fitness_score = Gauge(
    'dean_agent_fitness_score',
    'Fitness score of agent',
    ['agent_id']
)

# Meta-learning metrics
meta_patterns_extracted = Counter(
    'dean_meta_patterns_extracted_total',
    'Total meta-patterns extracted',
    ['abstraction_level']
)

dspy_training_examples = Counter(
    'dean_dspy_training_examples_total',
    'Training examples injected into DSPy',
    []
)

# System health metrics
active_agents = Gauge(
    'dean_active_agents',
    'Number of currently active agents',
    []
)

worktrees_active = Gauge(
    'dean_worktrees_active',
    'Number of active git worktrees',
    []
)

evolution_generation = Gauge(
    'dean_evolution_generation',
    'Current evolution generation',
    []
)

# Task-specific metrics
todos_implemented = Counter(
    'dean_todos_implemented_total',
    'Total TODO items implemented',
    []
)

test_coverage_delta = Summary(
    'dean_test_coverage_delta',
    'Change in test coverage percentage',
    []
)

complexity_reduction = Summary(
    'dean_complexity_reduction',
    'Reduction in code complexity',
    []
)


class MetricsCollector:
    """Collects and updates Prometheus metrics"""
    
    def __init__(self, port: int = 8090):
        """
        Initialize metrics collector
        
        Args:
            port: Port to expose metrics on
        """
        self.port = port
        self.running = False
        
    def start(self):
        """Start metrics HTTP server"""
        if not self.running:
            start_http_server(self.port)
            self.running = True
            logger.info(f"Metrics server started on port {self.port}")
    
    def update_agent_metrics(self, agent_id: str, action_type: str,
                           success_rate: float, efficiency: float,
                           fitness: float, budget: int):
        """Update metrics for an agent"""
        agent_success_rate.labels(action_type=action_type, agent_id=agent_id).set(success_rate)
        token_efficiency.labels(agent_id=agent_id).set(efficiency)
        agent_fitness_score.labels(agent_id=agent_id).set(fitness)
        agent_budget_current.labels(agent_id=agent_id).set(budget)
    
    def update_diversity_metrics(self, metrics: Dict[str, float]):
        """Update diversity metrics"""
        diversity_index.set(metrics.get('diversity_index', 0))
        population_variance.set(metrics.get('population_variance', 0))
        convergence_index.set(metrics.get('convergence_index', 0))
    
    def record_pattern_discovery(self, pattern_type: str, pattern_id: str = None):
        """Record pattern discovery"""
        patterns_discovered_total.labels(pattern_type=pattern_type).inc()
        if pattern_id:
            pattern_reuse_count.labels(pattern_id=pattern_id).inc()
    
    def record_ca_rule_trigger(self, rule_id: str):
        """Record CA rule trigger"""
        ca_rule_triggers.labels(rule_id=rule_id).inc()
    
    def update_economic_metrics(self, allocated: int, used: int):
        """Update economic metrics"""
        token_budget_allocated.set(allocated)
        token_budget_used.set(used)
    
    def record_action_execution(self, action_type: str, duration_seconds: float):
        """Record action execution time"""
        action_execution_time.labels(action_type=action_type).observe(duration_seconds)
    
    def record_meta_learning(self, abstraction_level: int, examples_count: int):
        """Record meta-learning activity"""
        meta_patterns_extracted.labels(abstraction_level=str(abstraction_level)).inc()
        dspy_training_examples.inc(examples_count)
    
    def update_system_health(self, active_agent_count: int, worktree_count: int,
                           current_generation: int):
        """Update system health metrics"""
        active_agents.set(active_agent_count)
        worktrees_active.set(worktree_count)
        evolution_generation.set(current_generation)
    
    def record_task_metrics(self, task_type: str, value: float):
        """Record task-specific metrics"""
        if task_type == 'todos_implemented':
            todos_implemented.inc(value)
        elif task_type == 'coverage_delta':
            test_coverage_delta.observe(value)
        elif task_type == 'complexity_reduction':
            complexity_reduction.observe(value)
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        return generate_latest().decode('utf-8')


# Global metrics collector instance
metrics_collector = MetricsCollector()


def update_metrics_from_result(result: Dict[str, Any]):
    """Update metrics from an action result"""
    if not result.get('success'):
        return
    
    agent_id = result.get('agent_id')
    action_type = result.get('action')
    metrics = result.get('metrics', {})
    
    # Update agent metrics
    if agent_id and action_type:
        metrics_collector.update_agent_metrics(
            agent_id=agent_id,
            action_type=action_type,
            success_rate=metrics.get('task_score', 0),
            efficiency=metrics.get('task_score', 0) / max(metrics.get('token_cost', 1), 1) * 1000,
            fitness=metrics.get('task_score', 0) * 0.7 + metrics.get('quality_score', 0) * 0.3,
            budget=0  # Would get from economic governor
        )
    
    # Record action execution time
    if 'execution_time' in metrics:
        metrics_collector.record_action_execution(action_type, metrics['execution_time'])
    
    # Record task-specific metrics
    if action_type == 'implement_todos' and 'todos_implemented' in result.get('metadata', {}):
        metrics_collector.record_task_metrics('todos_implemented', result['metadata']['todos_implemented'])
    elif action_type == 'improve_test_coverage' and 'coverage_delta' in result.get('metadata', {}):
        metrics_collector.record_task_metrics('coverage_delta', result['metadata']['coverage_delta'])
    elif action_type == 'refactor_complexity' and 'complexity_reduction' in result.get('metadata', {}):
        metrics_collector.record_task_metrics('complexity_reduction', result['metadata']['complexity_reduction'])


if __name__ == "__main__":
    # Demo the metrics
    import random
    
    print("DEAN Metrics Demo")
    print("=" * 60)
    
    # Start metrics server
    metrics_collector.start()
    print(f"Metrics available at http://localhost:{metrics_collector.port}/metrics")
    
    # Simulate some metrics
    print("\nSimulating agent activities...")
    
    for i in range(5):
        agent_id = f"agent_{i:03d}"
        
        # Update agent metrics
        metrics_collector.update_agent_metrics(
            agent_id=agent_id,
            action_type="implement_todos",
            success_rate=random.uniform(0.6, 0.95),
            efficiency=random.uniform(0.5, 2.0),
            fitness=random.uniform(0.5, 0.9),
            budget=random.randint(500, 2000)
        )
        
        # Record action execution
        metrics_collector.record_action_execution(
            "implement_todos",
            random.uniform(10, 120)
        )
        
        # Record pattern discovery
        if random.random() > 0.7:
            metrics_collector.record_pattern_discovery("optimization", f"opt_pattern_{i}")
        
        # Record CA rule trigger
        if random.random() > 0.5:
            rule = random.choice(["Rule110", "Rule30", "Rule90"])
            metrics_collector.record_ca_rule_trigger(rule)
        
        time.sleep(0.5)
    
    # Update diversity metrics
    metrics_collector.update_diversity_metrics({
        'diversity_index': 0.65,
        'population_variance': 0.72,
        'convergence_index': 0.28
    })
    
    # Update system health
    metrics_collector.update_system_health(
        active_agent_count=5,
        worktree_count=5,
        current_generation=10
    )
    
    print("\nMetrics updated. Press Ctrl+C to exit.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")