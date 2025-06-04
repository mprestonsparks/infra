# User Documentation (Final)

## Getting Started

### Prerequisites

Before installing the DEAN system, ensure your environment meets the following requirements. You need Docker and Docker Compose installed on your system. Git version 2.20 or higher is required for worktree management. Python 3.11 must be installed for local development and testing. You must have a Claude API key with sufficient token allocation for agent operations. A GitHub personal access token is necessary for automated pull request creation. The system requires a minimum of 16GB RAM for parallel agent execution, and SSD storage is strongly recommended for optimal worktree operations.

### Installation

#### Step 1: Clone Required Repositories

Clone all three repositories that comprise the DEAN system into a common parent directory:

```bash
mkdir dean-system && cd dean-system
git clone https://github.com/mprestonsparks/airflow-hub
git clone https://github.com/mprestonsparks/infra
git clone https://github.com/mprestonsparks/IndexAgent
```

#### Step 2: Configure Environment Variables

The DEAN system uses environment variables for configuration. Create the environment file in the infra directory:

```bash
cd infra
cp .env.example .env
```

Edit the `.env` file with your specific configuration:

```bash
# Core configuration
CLAUDE_API_KEY=your_claude_api_key_here
GITHUB_TOKEN=your_github_token_here

# Economic configuration
AGENT_EVOLUTION_MAX_POPULATION=100000
AGENT_TOKEN_LIMIT=4096
DEAN_MIN_VALUE_PER_TOKEN=0.001

# Diversity configuration
DEAN_MIN_DIVERSITY=0.3
DEAN_MUTATION_RATE=0.1
DEAN_PATTERN_IMPORT_ENABLED=true

# Operational configuration
AGENT_MAX_CONCURRENT=4
AGENT_EVOLUTION_PORT=8090
AGENT_REGISTRY_PORT=6380
AGENT_EVOLUTION_LOG_LEVEL=INFO

# Resource limits
AGENT_MAX_CPU=2.0
AGENT_MAX_MEMORY=2Gi

# Database configuration
AGENT_EVOLUTION_DATABASE_URL=postgresql://postgres:password@postgres:5432/agent_evolution
```

#### Step 3: Initialize DEAN Components

Create the necessary directory structure for the DEAN system:

```bash
# Create agent evolution module structure
mkdir -p modules/agent-evolution/{config,scripts,docker,data,logs}

# Copy configuration templates
cp -r ../IndexAgent/config/agents modules/agent-evolution/config/
```

#### Step 4: Build and Start Services

The DEAN system integrates with existing infrastructure through Docker Compose. Start all services:

```bash
# Build custom images
docker-compose build agent-evolution agent-registry

# Start all services including DEAN components
docker-compose up -d
```

#### Step 5: Verify Installation

After starting the services, verify that all components are running correctly:

```bash
# Check service health
docker-compose ps

# Verify DEAN services are healthy
curl http://localhost:8090/health

# Verify DEAN DAG registration in Airflow
docker exec airflow-service airflow dags list | grep dean

# Test metrics database connectivity
docker exec postgres psql -U postgres -d agent_evolution -c "SELECT version();"

# Verify Redis connectivity
docker exec agent-registry redis-cli ping
```

### Running Agent Evolution

#### Basic Evolution Run

The simplest way to start agent evolution is through the Airflow web interface. Access the Airflow UI at http://localhost:8080 using the default credentials. Navigate to the DAGs page and locate the `dean_agent_lifecycle` DAG. Click on the DAG name to view its details, then click "Trigger DAG" to start with default parameters.

The default configuration spawns 4 agents with 4096 tokens each, maintains a minimum diversity threshold of 0.3, and runs for 10 evolution generations.

#### Advanced Configuration

For more control over the evolution process, trigger the DAG with custom configuration. Click "Trigger DAG w/ Config" and provide JSON parameters:

```json
{
  "parallel_agents": 8,
  "token_budget": 50000,
  "target_repository": "IndexAgent",
  "evolution_generations": 20,
  "diversity_threshold": 0.4,
  "enable_pattern_import": true,
  "mutation_rate": 0.15,
  "crossover_rate": 0.7,
  "selection_method": "tournament",
  "fitness_weights": {
    "efficiency": 0.4,
    "quality": 0.3,
    "innovation": 0.3
  }
}
```

#### Monitoring Progress

The Airflow interface provides real-time monitoring of agent evolution. The Graph view displays the current status of each task in the DAG. Click on individual tasks to view detailed logs including token consumption, diversity metrics, and discovered patterns. The Gantt chart shows task execution timing and parallelism.

For more detailed monitoring, access the DEAN dashboard at http://localhost:8090/dashboard, which provides real-time metrics on agent performance, population diversity, token consumption rates, and pattern discovery.

### Configuration

#### Token Economy Settings

The token economy configuration controls how the system allocates and manages computational resources. Edit `infra/modules/agent-evolution/config/agents.yaml`:

```yaml
economy:
  global_budget: 100000              # Total tokens available per run
  agent_limits:
    base: 4096                       # Starting tokens per agent
    maximum: 8192                    # Maximum tokens any agent can have
    minimum: 1024                    # Minimum viable token allocation
  efficiency_thresholds:
    excellent: 0.01                  # tokens per value point (top tier)
    good: 0.05                       # tokens per value point (mid tier)
    acceptable: 0.1                  # tokens per value point (baseline)
  budget_decay:
    enabled: true                    # Enable decay for long-running agents
    rate: 0.9                        # 10% reduction per generation
    generations_before_decay: 5      # Grace period before decay starts
  allocation_strategy:
    type: "performance_weighted"     # Options: equal, performance_weighted, adaptive
    performance_lookback: 3          # Generations to consider for performance
```

#### Diversity Management

Genetic diversity parameters ensure the system maintains a healthy variety of strategies. Configure in `infra/modules/agent-evolution/config/evolution.yaml`:

```yaml
diversity:
  minimum_variance: 0.3              # Below this triggers intervention
  measurement_method: "hamming"      # Options: hamming, euclidean, cosine
  mutation_injection:
    threshold: 0.2                   # Trigger when variance drops below
    rate: 0.15                       # Percentage of population to mutate
    strategy: "targeted"             # Options: random, targeted, adaptive
  pattern_import:
    enabled: true
    sources:
      - name: "historical_patterns"
        path: "/app/data/patterns/historical"
        filter: "effectiveness > 0.8"
      - name: "cross_domain_strategies"
        path: "/app/data/patterns/external"
        compatibility_check: true
    frequency: 5                     # Import every N generations
    max_imports_per_cycle: 10        # Limit to prevent disruption
  convergence_prevention:
    enabled: true
    action: "force_mutation"         # Options: force_mutation, restart_subset, import_patterns
    severity_threshold: 0.15         # Critical convergence level
```

#### Pattern Detection

Configure how the system identifies and manages emergent behaviors:

```yaml
patterns:
  detection:
    sensitivity: "medium"            # Options: low, medium, high
    min_occurrences: 3              # Times pattern must appear
    confidence_threshold: 0.7        # Statistical confidence required
    detection_window: 10            # Generations to observe
  classification:
    enabled: true
    categories:
      - name: "optimization"
        indicators: ["token_reduction", "speed_improvement"]
        min_benefit: 0.2
      - name: "innovation"
        indicators: ["novel_approach", "unexpected_success"]
        min_benefit: 0.3
      - name: "gaming"
        indicators: ["metric_manipulation", "constraint_bypass"]
        max_benefit: 0.1
  storage:
    backend: "postgresql"           # Options: postgresql, redis, both
    retention_days: 90              # How long to keep patterns
    max_patterns: 1000              # Storage limit
    compression: true               # Compress pattern data
  reuse:
    min_effectiveness: 0.8          # Threshold for reuse eligibility
    test_before_deploy: true        # Validate in sandbox first
    compatibility_check: true       # Ensure pattern fits context
```

### Using the System

#### Viewing Agent Performance

The DEAN system provides comprehensive performance monitoring through multiple interfaces. The main dashboard at http://localhost:8090/dashboard displays real-time metrics including agent efficiency scores, token consumption rates, diversity variance, pattern discovery timeline, and economic performance indicators.

To access detailed agent information via the API:

```bash
# List all active agents
curl http://localhost:8090/api/v1/agents

# Get specific agent details
curl http://localhost:8090/api/v1/agents/{agent_id}

# View agent efficiency history
curl http://localhost:8090/api/v1/agents/{agent_id}/efficiency
```

#### Querying the Knowledge Repository

The DEAN system maintains a comprehensive knowledge repository of discovered patterns and successful strategies. Access this repository through the command-line interface:

```bash
# Find most efficient strategies
docker exec agent-evolution dean-cli query \
  --type=strategies \
  --sort=efficiency \
  --limit=10

# Analyze pattern emergence over time
docker exec agent-evolution dean-cli patterns \
  --since="7 days ago" \
  --min-reuse=5 \
  --format=json

# Export successful agent genomes
docker exec agent-evolution dean-cli export \
  --generation=latest \
  --top-performers=3 \
  --output=/app/data/exports/top_agents.json

# Search for specific pattern types
docker exec agent-evolution dean-cli search \
  --pattern-type=optimization \
  --min-effectiveness=0.8 \
  --used-by-agents=5
```

#### Managing Token Budgets

Token budget management is critical for maintaining system efficiency. Monitor and adjust allocations through the management interface:

```bash
# Check current global consumption
docker exec agent-evolution dean-cli budget status

# View detailed breakdown by agent
docker exec agent-evolution dean-cli budget breakdown \
  --show-efficiency \
  --show-trends

# Adjust global budget (requires confirmation)
docker exec agent-evolution dean-cli budget set \
  --global=150000 \
  --confirm

# Set agent-specific limits
docker exec agent-evolution dean-cli budget set \
  --agent-type=explorer \
  --limit=6000 \
  --reason="Increased complexity in target repository"

# Enable emergency budget reserve
docker exec agent-evolution dean-cli budget reserve \
  --percentage=10 \
  --trigger-threshold=0.9
```

### Troubleshooting

#### Common Issues and Solutions

**High Token Consumption**

When agents consume tokens rapidly without generating proportional value, several interventions can help. First, check the efficiency thresholds in your configuration to ensure they match your expectations. Reduce the parallel agent count to decrease overall consumption. Implement stricter budget decay by adjusting the decay rate in `evolution.yaml`. Enable aggressive token limits for exploration phases. Review recent pattern imports that might have introduced inefficient strategies.

**Low Diversity Warnings**

If diversity drops below the configured threshold, the system should automatically inject mutations. If this mechanism fails, you can manually intervene. Use the command `docker exec agent-evolution dean-cli diversity inject --rate=0.2` to force mutation injection. Verify that pattern import is enabled and functioning by checking the logs. Review the diversity calculation method to ensure it accurately captures strategy variance. Consider importing patterns from a more diverse source repository.

**Pattern Detection Failures**

When the system fails to detect novel patterns that you expect it to find, several factors may be responsible. Adjust the detection sensitivity in your configuration from "medium" to "high". Verify that the metrics database is recording all agent actions by checking the `performance_metrics` table. Review classification logs for errors that might indicate misconfiguration. Ensure the detection window is long enough to capture slower-emerging patterns.

**Agent Creation Timeouts**

Slow agent creation often indicates resource constraints or configuration issues. Verify available disk space for worktrees using `df -h /app/worktrees`. Check Docker resource allocations with `docker system df`. Consider implementing worktree pooling for faster allocation. Review the git repository size, as large repositories slow worktree creation. Enable worktree caching in the configuration.

**Database Performance Issues**

If queries slow down over time, database maintenance may be needed. Run `ANALYZE` on PostgreSQL tables to update statistics. Check for missing indexes using the query execution planner. Consider partitioning large tables by date. Implement regular cleanup of old metrics data. Monitor connection pool usage for bottlenecks.

### Best Practices

#### Economic Optimization

Effective token management requires a strategic approach. Start with conservative token budgets and increase gradually based on observed efficiency. Monitor value-per-token metrics closely during initial runs to establish baselines. Implement progressive budgets that reward efficient agents with larger allocations. Use budget decay to prevent runaway consumption in long-running agents. Regular review of efficiency thresholds ensures they remain aligned with your goals.

#### Diversity Maintenance

Maintaining genetic diversity is crucial for long-term system health. Regularly review population variance metrics through the dashboard. Import patterns from successful manual implementations to introduce proven strategies. Avoid over-optimizing for a single metric, which can lead to convergence. Encourage exploration through diversity bonuses in fitness calculations. Set up alerts for diversity drops below critical thresholds.

#### Pattern Management

Effective pattern management maximizes the value of discovered innovations. Regularly audit discovered patterns for quality and relevance. Export and version successful strategies in your version control system. Create pattern libraries for different task types to enable targeted reuse. Test patterns in isolation before broad deployment to ensure stability. Document pattern metadata to aid future understanding and use.

#### Operational Excellence

Running the DEAN system effectively requires attention to operational details. Schedule evolution runs during off-peak hours to minimize resource contention. Implement cost alerts for budget thresholds to prevent overruns. Maintain separate environments for experimentation and production use. Perform regular backups of the metrics database to preserve discovered knowledge. Monitor system logs for early warning signs of issues.

### Integration with Existing Workflows

#### Pull Request Integration

The DEAN system automatically creates pull requests for successful improvements. Configure your repository to work effectively with automated PRs by creating a pull request template at `.github/pull_request_template.md`:

```markdown
## DEAN Agent Evolution Results

**Agent ID**: {{ agent_id }}
**Generation**: {{ generation }}
**Token Efficiency**: {{ efficiency }}
**Improvement Type**: {{ improvement_type }}

### Metrics
- Lines Changed: {{ lines_changed }}
- Tokens Used: {{ tokens_used }}
- Value Generated: {{ value_score }}
- Diversity Impact: {{ diversity_delta }}

### Testing
- [ ] All tests pass
- [ ] No security vulnerabilities introduced
- [ ] Performance benchmarks maintained
```

Set up CI/CD to validate DEAN-generated changes automatically. Use branch protection rules to require human review before merging. Configure automated testing specific to DEAN contributions.

#### Metrics Export

Export DEAN metrics to your existing monitoring systems for unified observability:

```bash
# Prometheus export (automatic at /metrics endpoint)
curl http://localhost:8090/metrics

# DataDog integration
docker exec agent-evolution dean-cli metrics push \
  --provider=datadog \
  --api-key=$DD_API_KEY \
  --interval=60

# Custom export to JSON
docker exec agent-evolution dean-cli metrics export \
  --format=json \
  --output=/app/data/metrics/dean_metrics.json \
  --include=efficiency,diversity,patterns

# CSV export for analysis
docker exec agent-evolution dean-cli metrics export \
  --format=csv \
  --output=/app/data/metrics/dean_metrics.csv \
  --time-range="last 24 hours"
```

#### Custom Fitness Functions

Define project-specific fitness functions to align evolution with your goals. Create a custom fitness module at `IndexAgent/indexagent/agents/fitness/custom_fitness.py`:

```python
from typing import Dict, Any
from .base_fitness import BaseFitness

class CustomProjectFitness(BaseFitness):
    """Project-specific fitness calculation"""
    
    def calculate(self, agent_metrics: Dict[str, Any]) -> float:
        """Calculate fitness based on project priorities"""
        
        # Weight different metrics according to project needs
        test_coverage_weight = 0.3
        performance_weight = 0.2
        token_efficiency_weight = 0.5
        
        # Calculate component scores
        coverage_score = agent_metrics.get('test_coverage', 0) * test_coverage_weight
        performance_score = (1 / agent_metrics.get('execution_time', 1)) * performance_weight
        efficiency_score = agent_metrics.get('value_per_token', 0) * token_efficiency_weight
        
        # Apply penalties for violations
        if agent_metrics.get('security_issues', 0) > 0:
            return 0.0  # Reject agents that introduce security issues
        
        # Calculate final fitness
        fitness = coverage_score + performance_score + efficiency_score
        
        # Bonus for innovation
        if agent_metrics.get('novel_patterns', 0) > 0:
            fitness *= 1.2
        
        return fitness
```

Register your custom fitness function in the configuration:

```yaml
fitness:
  function: "custom_project_fitness"
  module: "indexagent.agents.fitness.custom_fitness"
  update_frequency: "every_generation"
```

### Advanced Features

#### Multi-Repository Evolution

The DEAN system can evolve agents across multiple repositories simultaneously. Configure multi-repository support in `infra/modules/agent-evolution/config/repositories.yaml`:

```yaml
repositories:
  - name: "IndexAgent"
    path: "/repos/IndexAgent"
    weight: 0.4
    constraints:
      - "no_changes_to_core"
      - "maintain_api_compatibility"
  - name: "airflow-hub"
    path: "/repos/airflow-hub"
    weight: 0.3
    constraints:
      - "no_dag_deletions"
      - "maintain_schedule_integrity"
  - name: "custom-project"
    path: "/repos/custom-project"
    weight: 0.3
    constraints:
      - "preserve_tests"
      - "no_config_changes"
```

#### Pattern Cross-Pollination

Enable patterns discovered in one domain to benefit others:

```bash
# Export patterns from one project
docker exec agent-evolution dean-cli patterns export \
  --project=IndexAgent \
  --min-effectiveness=0.8 \
  --output=/app/data/patterns/indexagent_patterns.json

# Import patterns to another project
docker exec agent-evolution dean-cli patterns import \
  --project=custom-project \
  --source=/app/data/patterns/indexagent_patterns.json \
  --compatibility-check=true \
  --test-first=true
```

#### Evolutionary Strategies

Configure different evolutionary strategies for different scenarios:

```yaml
strategies:
  exploration:
    name: "High Exploration"
    mutation_rate: 0.3
    crossover_rate: 0.5
    selection_pressure: 0.2
    use_when: "diversity < 0.25"
    
  exploitation:
    name: "Refinement Focus"
    mutation_rate: 0.05
    crossover_rate: 0.8
    selection_pressure: 0.8
    use_when: "generation > 20"
    
  balanced:
    name: "Standard Evolution"
    mutation_rate: 0.1
    crossover_rate: 0.7
    selection_pressure: 0.5
    use_when: "default"
```

### Security Considerations

#### API Authentication

The DEAN system implements API key authentication for all endpoints. Generate API keys through the management interface:

```bash
docker exec agent-evolution dean-cli auth create-key \
  --name="monitoring-system" \
  --permissions="read" \
  --expiry="90d"
```

Use the generated key in API requests:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:8090/api/v1/agents
```

#### Agent Isolation

Agents operate in isolated environments with strict boundaries. Each agent worktree has read-only access to source repositories. Write operations are limited to the agent's assigned worktree. Network access is restricted to approved endpoints only. File system access is controlled through Docker volume mounts.

#### Audit Logging

All agent actions are logged for security and compliance:

```bash
# View security-relevant events
docker exec agent-evolution dean-cli audit \
  --event-type=security \
  --last=24h

# Export audit logs for compliance
docker exec agent-evolution dean-cli audit export \
  --format=json \
  --output=/app/data/audit/dean_audit.json \
  --time-range="last 30 days"
```

### Maintenance and Updates

#### System Updates

Keep the DEAN system updated with the latest improvements:

```bash
# Pull latest changes
cd ../IndexAgent && git pull
cd ../airflow-hub && git pull
cd ../infra && git pull

# Rebuild containers with updates
cd ../infra
docker-compose build --no-cache agent-evolution
docker-compose up -d agent-evolution
```

#### Database Maintenance

Regular database maintenance ensures optimal performance:

```bash
# Run maintenance tasks
docker exec postgres psql -U postgres -d agent_evolution \
  -c "VACUUM ANALYZE agent_evolution.agents;"

# Backup database
docker exec postgres pg_dump -U postgres agent_evolution \
  > backups/agent_evolution_$(date +%Y%m%d).sql

# Archive old data
docker exec agent-evolution dean-cli maintenance archive \
  --older-than="90 days" \
  --destination="s3://dean-archives/"
```

#### Log Rotation

Configure log rotation to manage disk space:

```yaml
# /etc/logrotate.d/dean
/app/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 dean dean
}
```