# Instructions for AI Coding Agent

## Implementation Instructions for Distributed Evolutionary Agent Network (DEAN)

You are tasked with implementing a distributed, self-evolving agent system based on the provided specifications. This system combines cellular automata principles with practical git worktree parallelization, while fundamentally incorporating economic constraints, diversity maintenance, and emergent behavior capture as core architectural elements.

**CRITICAL WARNING**: All documents provided are FIRST DRAFTS. Before beginning implementation, you MUST:

1. Review each document for compatibility with existing codebases
2. Identify any conflicts with current implementations
3. Report incompatibilities and propose resolutions
4. Only proceed with implementation after integration review is complete

## Core Implementation Requirements

The DEAN system must be built with four foundational pillars from the outset:

**Economic Efficiency**: Every component must track and optimize token consumption. This is not an optional feature but a fundamental constraint that shapes all architectural decisions.

**Genetic Diversity**: The system must actively maintain population variance to prevent premature convergence. Diversity mechanisms must be built into the core evolution loop, not added later.

**Emergent Behavior Capture**: Pattern detection and cataloging must be integral to the system design, enabling meta-learning from discovered optimizations.

**Knowledge Repository**: The metrics database must be designed as a queryable knowledge base for pattern analysis and strategy extraction, not merely a logging mechanism.

## Implementation Priority Order

### Phase 1: Economic Foundation

Begin by implementing the token economy system, as it constrains all other components:

```python
# First implementation priority
class TokenEconomyManager:
    """Core economic controller - implement this before any agent logic"""
    def __init__(self, global_budget: int):
        self.global_budget = global_budget
        self.agent_allocations = {}
        self.efficiency_history = {}
    
    def allocate_tokens(self, agent_id: str, historical_efficiency: float) -> int:
        """Dynamic allocation based on past performance"""
        # Implement value-based allocation algorithm
        pass
```

Create the economic safety mechanisms including hard API-level token limits, automatic agent termination on budget exhaustion, and real-time consumption tracking with alerts.

### Phase 2: Diversity Infrastructure

Implement diversity management as a core component, not an add-on:

```python
class GeneticDiversityManager:
    """Maintains population health through enforced variance"""
    def __init__(self, min_diversity: float = 0.3):
        self.min_diversity = min_diversity
        self.mutation_engine = MutationEngine()
    
    def enforce_diversity(self, population: List[Agent]) -> List[Agent]:
        """Actively prevent monocultures"""
        # Implement variance calculation and mutation injection
        pass
```

Build mutation injection, cross-domain pattern import, and convergence detection into the base evolution loop.

### Phase 3: Pattern Detection System

Implement emergent behavior monitoring from the start:

```python
class EmergentBehaviorMonitor:
    """Captures and catalogs novel agent strategies"""
    def __init__(self, metrics_db: MetricsDatabase):
        self.metrics_db = metrics_db
        self.pattern_detector = PatternDetector()
    
    def analyze_agent_behavior(self, agent: Agent) -> List[Pattern]:
        """Identify strategies not explicitly programmed"""
        # Implement pattern detection and classification
        pass
```

Design the system to distinguish between beneficial innovations and metric gaming from day one.

### Phase 4: Knowledge Repository Design

Create the metrics database with analytical capabilities built-in:

```sql
-- Design for analysis, not just logging
CREATE TABLE discovered_patterns (
    id UUID PRIMARY KEY,
    pattern_hash TEXT UNIQUE,
    effectiveness_score FLOAT,
    token_efficiency FLOAT,
    reuse_count INTEGER DEFAULT 0,
    metadata JSONB,
    created_at TIMESTAMP
);

-- Critical indexes for meta-learning
CREATE INDEX idx_pattern_effectiveness ON discovered_patterns(effectiveness_score DESC);
CREATE INDEX idx_pattern_efficiency ON discovered_patterns(token_efficiency DESC);
CREATE INDEX idx_pattern_reuse ON discovered_patterns(reuse_count DESC);
```

### Phase 5: Agent Implementation

Only after the foundational systems are in place, implement the agent logic:

```python
class FractalAgent:
    def __init__(self, genome: AgentGenome, token_budget: TokenBudget):
        self.genome = genome
        self.token_budget = token_budget
        self.efficiency_tracker = EfficiencyTracker()
        self.pattern_history = []
    
    async def evolve(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Evolution with built-in constraints and monitoring"""
        # Check token budget before any operation
        # Track efficiency for all actions
        # Detect and report novel patterns
        # Maintain diversity from population
        pass
```

## Critical Implementation Details

### Git Worktree Management with Resource Limits

```python
class GitWorktreeManager:
    def create_worktree(self, agent_id: str, token_limit: int) -> Path:
        """Create isolated environment with built-in constraints"""
        worktree_path = self._generate_unique_path(agent_id)
        
        # Set resource limits at creation time
        self._set_token_limit(worktree_path, token_limit)
        self._set_file_system_quota(worktree_path)
        
        return worktree_path
```

### DSPy Integration with Meta-Learning Hooks

```python
class DSPyEvolutionModule(dspy.Module):
    """DSPy module that discovers new optimization strategies"""
    def __init__(self):
        super().__init__()
        self.prompt_optimizer = dspy.ChainOfThought("optimize_prompt")
        self.strategy_discoverer = dspy.Predict("discover_strategy")
    
    def forward(self, current_prompt, performance_metrics):
        # Optimize existing prompt
        # Discover new module combinations
        # Export successful strategies
        pass
```

### Cellular Automata with Economic Constraints

Implement all five rules with token awareness:

- Rule 110: Create improved neighbors within token budget
- Rule 30: Fork into parallel worktrees if economically justified
- Rule 90: Abstract patterns that improve token efficiency
- Rule 184: Learn from neighbors with better value-per-token
- Rule 1: Recurse only when token ROI exceeds threshold

### Airflow DAG with Economic Governance

```python
def create_dean_dag():
    with DAG('dean_evolution', 
             default_args={'token_budget': 100000}) as dag:
        
        # Economic checkpoint before agent spawn
        check_budget = PythonOperator(
            task_id='check_token_budget',
            python_callable=verify_sufficient_tokens
        )
        
        # Spawn agents with individual budgets
        spawn_agents = PythonOperator(
            task_id='spawn_agents',
            python_callable=spawn_with_token_limits
        )
        
        # Continuous efficiency monitoring
        monitor_efficiency = PythonOperator(
            task_id='monitor_efficiency',
            python_callable=track_value_per_token
        )
```

## Verification Requirements

After implementing each component, verify it meets these criteria:

### Economic Verification

```bash
# Test token enforcement
pytest tests/test_token_limits.py -v

# Verify efficiency tracking
python -c "from dean import TokenEconomyManager; assert TokenEconomyManager().calculate_roi(1000, 10) == 0.01"

# Check budget allocation algorithm
python scripts/test_allocation_fairness.py
```

### Diversity Verification

```bash
# Test diversity maintenance
pytest tests/test_diversity_enforcement.py -v

# Verify mutation injection
python scripts/simulate_convergence.py --generations=10 --assert-diversity=0.3

# Check pattern import
python scripts/test_cross_domain_import.py
```

### Pattern Detection Verification

```bash
# Test pattern identification
pytest tests/test_pattern_detection.py -v

# Verify classification accuracy
python scripts/test_gaming_detection.py --samples=100

# Check pattern storage
sqlite3 dean.db "SELECT COUNT(*) FROM discovered_patterns;"
```

## Architecture Considerations

Remember that this system is designed to evolve and improve itself. Your implementation should:

1. **Prioritize Measurement**: Build comprehensive metrics into every component from the start. You cannot optimize what you do not measure.
    
2. **Embrace Emergence**: Create conditions for unexpected behaviors to arise and be captured. The most valuable optimizations will be those the system discovers on its own.
    
3. **Enforce Constraints**: Economic and diversity constraints are not limitations but enablers of sustainable evolution. Strict enforcement prevents pathological behaviors.
    
4. **Design for Query**: The knowledge repository should support complex analytical queries from day one. Future meta-learning depends on rich historical data.
    
5. **Plan for Scale**: While starting with 4-8 agents, design for 100+ agents. Architectural decisions made now will determine future scalability.
    

## Success Metrics

Your implementation succeeds when:

- Token consumption decreases while value generation increases over 10+ generations
- Population diversity remains above 0.3 throughout evolution
- Novel patterns are discovered and successfully reused
- The system identifies and exports strategies you did not explicitly program
- Meta-learning creates new DSPy module combinations autonomously

Focus on creating a robust foundation with these core capabilities built-in from the start. The system's power lies not in perfect initial performance but in its ability to discover optimizations through evolutionary pressure while maintaining economic efficiency and genetic diversity.