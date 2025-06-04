# Maintenance Documentation (Final)

## 1. Routine Maintenance

### 1.1 Daily Tasks

**Token Economy Monitoring**

Each morning, system administrators should review the token consumption dashboard accessible at http://localhost:8090/dashboard. This review identifies any agents exceeding their allocated budgets and ensures global token utilization remains below 80% of the daily allocation configured in the AGENT_EVOLUTION_MAX_POPULATION environment variable. Administrators should examine value-per-token metrics stored in the agent_evolution.performance_metrics table for any significant degradation that might indicate inefficient agent behavior.

To perform these checks efficiently, use the following commands:

```bash
# Check global token usage
docker exec agent-evolution dean-cli budget status --detailed

# Identify high-consumption agents
docker exec postgres psql -U postgres -d agent_evolution -c \
  "SELECT agent_id, SUM(metric_value) as total_tokens 
   FROM agent_evolution.performance_metrics 
   WHERE metric_name = 'tokens_consumed' 
   AND recorded_at > NOW() - INTERVAL '24 hours' 
   GROUP BY agent_id 
   ORDER BY total_tokens DESC 
   LIMIT 10;"
```

**Diversity Health Checks**

Population diversity monitoring ensures the system maintains healthy genetic variance above the configured threshold of 0.3. Review the mutation injection logs located at /app/logs/diversity_monitor.log to confirm the system actively maintains diversity. Check for monoculture warnings in the Airflow task logs, which indicate premature convergence requiring intervention.

```bash
# Check current diversity score
curl http://localhost:8090/api/v1/metrics/diversity

# Review mutation injection history
docker exec agent-evolution grep "mutation_injection" /app/logs/evolution.log | tail -20
```

**Pattern Discovery Review**

Examine newly discovered patterns from the previous 24 hours through the pattern management interface. Each pattern requires classification as either a beneficial innovation worthy of preservation or a metric gaming behavior that should be blocked. Approved patterns must be added to the reusable component library stored in Redis and backed up to PostgreSQL.

```bash
# List recent discoveries
docker exec agent-evolution dean-cli patterns list \
  --discovered-after="24 hours ago" \
  --status=unclassified

# Classify a pattern
docker exec agent-evolution dean-cli patterns classify \
  --pattern-id=<UUID> \
  --classification=beneficial \
  --effectiveness=0.85
```

**Pull Request Management**

Review all DEAN-generated pull requests in your GitHub repositories. Each PR includes metadata about token consumption, efficiency metrics, and pattern usage. Merge approved changes and provide feedback on rejected submissions by updating the agent's fitness metrics in the database. This feedback loop improves future agent performance.

### 1.2 Weekly Tasks

**Comprehensive System Cleanup**

Every week, execute a thorough cleanup of accumulated artifacts and temporary files. The git worktree system requires regular pruning to remove orphaned worktrees that may result from agent failures or interrupted evolution cycles.

```bash
# Execute cleanup script
docker exec agent-evolution /app/scripts/weekly-cleanup.sh

# Manual worktree pruning if needed
docker exec agent-evolution bash -c "cd /app/worktrees && git worktree prune"

# Clean temporary files
docker exec agent-evolution find /tmp -name "dean_*" -mtime +7 -delete

# Archive logs older than 7 days
docker exec agent-evolution bash -c "cd /app/logs && tar -czf archive_$(date +%Y%m%d).tar.gz *.log.* && rm *.log.*"
```

**Metrics Database Optimization**

The PostgreSQL database accumulates performance data rapidly and requires weekly optimization to maintain query performance. Execute the following maintenance routine:

```bash
# Connect to database and run optimization
docker exec postgres psql -U postgres -d agent_evolution << EOF
-- Update statistics
ANALYZE agent_evolution.agents;
ANALYZE agent_evolution.performance_metrics;
ANALYZE agent_evolution.discovered_patterns;

-- Vacuum tables
VACUUM ANALYZE agent_evolution.evolution_history;

-- Check index health
SELECT schemaname, tablename, indexname, idx_scan 
FROM pg_stat_user_indexes 
WHERE schemaname = 'agent_evolution' 
ORDER BY idx_scan;
EOF

# Archive old data
docker exec agent-evolution dean-cli maintenance archive-metrics \
  --older-than="30 days" \
  --destination="/app/data/archives/"
```

**Pattern Library Curation**

The pattern library requires weekly review to maintain quality and relevance. Remove redundant patterns that provide similar functionality. Test high-reuse patterns to verify continued effectiveness. Export successful patterns to version control for permanent storage.

```bash
# Analyze pattern usage
docker exec agent-evolution dean-cli patterns analyze \
  --min-reuse=5 \
  --export-report=/app/data/reports/pattern_analysis.json

# Remove ineffective patterns
docker exec agent-evolution dean-cli patterns prune \
  --effectiveness-below=0.3 \
  --unused-for="14 days" \
  --confirm
```

**Economic Performance Analysis**

Generate comprehensive reports on token efficiency trends to identify optimization opportunities. These reports inform decisions about budget adjustments and allocation strategies.

```bash
# Generate weekly efficiency report
docker exec agent-evolution dean-cli reports generate \
  --type=economic \
  --period=week \
  --output=/app/data/reports/weekly_economic_report.pdf

# Export data for external analysis
docker exec postgres psql -U postgres -d agent_evolution -c \
  "COPY (SELECT * FROM agent_evolution.performance_metrics WHERE recorded_at > NOW() - INTERVAL '7 days') 
   TO '/tmp/weekly_metrics.csv' WITH CSV HEADER;"
```

### 1.3 Monthly Tasks

**System Evolution Review**

Monthly reviews analyze long-term evolutionary trends to identify successful adaptations and areas needing adjustment. Document emergent behaviors that have proven valuable for inclusion in future fitness functions. Plan strategic adjustments based on observed outcomes.

```bash
# Generate evolution report
docker exec agent-evolution dean-cli reports evolution \
  --period=month \
  --include-patterns \
  --include-lineage \
  --output=/app/data/reports/monthly_evolution.pdf

# Extract successful strategies
docker exec agent-evolution dean-cli strategies export \
  --top-performers=10 \
  --min-efficiency=0.01 \
  --output=/app/data/strategies/monthly_best.json
```

**Infrastructure Capacity Planning**

Review resource utilization trends to plan for capacity increases. Monitor CPU, memory, and storage usage patterns across all DEAN components.

```bash
# Collect resource metrics
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" \
  agent-evolution agent-registry postgres

# Check disk usage
docker exec agent-evolution df -h /app/worktrees /app/data /app/logs

# Database size analysis
docker exec postgres psql -U postgres -d agent_evolution -c \
  "SELECT pg_size_pretty(pg_database_size('agent_evolution')) as db_size;"
```

**Cross-Domain Pattern Import**

Monthly pattern imports from external sources maintain diversity and introduce proven strategies from other domains. This process requires careful curation and testing.

```bash
# Import external patterns
docker exec agent-evolution dean-cli patterns import \
  --source=/app/data/imports/external_patterns.json \
  --test-mode \
  --compatibility-threshold=0.8

# Validate imported patterns
docker exec agent-evolution dean-cli patterns validate \
  --recently-imported \
  --run-tests \
  --report=/app/data/reports/import_validation.json
```

## 2. Troubleshooting

### 2.1 Token Economy Issues

**Excessive Token Consumption**

When agents consume tokens without generating proportional value, systematic investigation identifies root causes. Begin by examining the specific agents consuming excessive tokens through database queries. Analyze their strategy genomes stored in the agent_evolution.agents table to understand their approach. Common causes include inefficient DSPy prompts, recursive behavior patterns, or exploration strategies that lack proper termination conditions.

Resolution steps include reducing token allocations for inefficient strategy families through configuration updates, implementing stricter budget decay rates in the evolution.yaml file, and reviewing recent pattern imports that might have introduced inefficient behaviors. Use the following diagnostic queries:

```bash
# Identify token-hungry strategies
docker exec postgres psql -U postgres -d agent_evolution -c \
  "SELECT a.id, a.genome_hash, pm.metric_value as tokens_used, a.fitness_score 
   FROM agent_evolution.agents a 
   JOIN agent_evolution.performance_metrics pm ON a.id = pm.agent_id 
   WHERE pm.metric_name = 'tokens_consumed' 
   AND pm.metric_value > 5000 
   ORDER BY pm.metric_value DESC;"

# Trace pattern origins
docker exec agent-evolution dean-cli patterns trace \
  --agent-id=<problematic_agent_id> \
  --show-inheritance
```

**Token Starvation**

High-performing agents sometimes lack sufficient tokens due to allocation algorithm biases. Diagnose this condition by examining the allocation history and efficiency metrics. Verify that efficiency calculations in the token_manager.py module correctly reflect agent value generation.

```bash
# Analyze allocation patterns
docker exec agent-evolution dean-cli budget analyze \
  --show-allocations \
  --show-efficiency \
  --time-range="last 7 days"

# Adjust allocation weights
docker exec agent-evolution dean-cli budget configure \
  --efficiency-weight=0.7 \
  --diversity-weight=0.3 \
  --apply-immediately
```

### 2.2 Diversity Problems

**Convergence Detection**

When diversity metrics indicate convergence below the 0.3 threshold, immediate intervention prevents complete monoculture. The system should automatically trigger mutation injection, but manual intervention may be necessary if this mechanism fails.

```bash
# Check diversity calculation details
docker exec agent-evolution dean-cli diversity analyze \
  --show-clusters \
  --show-distances \
  --export=/app/data/analysis/diversity_detail.json

# Force manual mutation if needed
docker exec agent-evolution dean-cli diversity mutate \
  --target-variance=0.4 \
  --mutation-rate=0.2 \
  --selection-strategy=bottom_performers
```

Review the evolutionary history to identify convergence causes. Common patterns include overly restrictive fitness functions, insufficient exploration incentives, or dominant strategies that crowd out alternatives.

**Over-Diversification**

Excessive diversity prevents agents from learning from successful strategies. This condition manifests as stagnant fitness scores despite continued evolution. Adjust mutation rates and increase successful pattern propagation weights to encourage beneficial convergence while maintaining healthy variance.

```bash
# Reduce mutation pressure
docker exec agent-evolution dean-cli config update \
  --set="diversity.mutation_injection.rate=0.05" \
  --set="evolution.crossover_rate=0.8"

# Increase pattern sharing
docker exec agent-evolution dean-cli patterns configure \
  --propagation-rate=0.5 \
  --min-effectiveness=0.7
```

### 2.3 Pattern Management Issues

**Pattern Detection Failures**

When the EmergentBehaviorMonitor fails to detect expected patterns, several factors require investigation. Verify the monitor service runs correctly by checking its health endpoint. Review detection sensitivity settings in the configuration. Examine agent action logs to ensure all behaviors are properly recorded.

```bash
# Check monitor health
curl http://localhost:8090/api/v1/health/pattern-monitor

# Adjust detection parameters
docker exec agent-evolution dean-cli patterns configure \
  --detection-sensitivity=high \
  --min-occurrences=2 \
  --confidence-threshold=0.6

# Verify data recording
docker exec postgres psql -U postgres -d agent_evolution -c \
  "SELECT COUNT(*) as action_count, 
   DATE(recorded_at) as date 
   FROM agent_evolution.agent_actions 
   GROUP BY DATE(recorded_at) 
   ORDER BY date DESC LIMIT 7;"
```

**Pattern Gaming**

Agents sometimes discover ways to game the pattern detection system, creating the appearance of innovation without genuine improvement. Implement stricter validation criteria and human review requirements for high-impact patterns. Adjust fitness functions to penalize obvious gaming behaviors.

```bash
# Identify suspicious patterns
docker exec agent-evolution dean-cli patterns audit \
  --suspicion-indicators="high_frequency,low_diversity" \
  --time-range="last 48 hours"

# Implement validation rules
docker exec agent-evolution dean-cli patterns add-validator \
  --name="anti-gaming" \
  --rule="effectiveness_delta > 0.2" \
  --rule="implementation_variety > 3"
```

### 2.4 Performance Issues

**Slow Agent Creation**

Agent creation bottlenecks typically stem from git worktree operations or resource constraints. Monitor worktree creation times and implement pooling for faster allocation.

```bash
# Check worktree performance
docker exec agent-evolution time git worktree add /tmp/test_worktree

# Enable worktree pooling
docker exec agent-evolution dean-cli config update \
  --set="agents.worktree_pooling.enabled=true" \
  --set="agents.worktree_pooling.size=10"

# Monitor disk I/O
docker exec agent-evolution iostat -x 1 10
```

**Database Query Degradation**

PostgreSQL query performance degrades over time without proper maintenance. Identify slow queries using pg_stat_statements and optimize accordingly.

```bash
# Enable query monitoring
docker exec postgres psql -U postgres -d agent_evolution -c \
  "CREATE EXTENSION IF NOT EXISTS pg_stat_statements;"

# Find slow queries
docker exec postgres psql -U postgres -d agent_evolution -c \
  "SELECT query, mean_exec_time, calls 
   FROM pg_stat_statements 
   WHERE query LIKE '%agent_evolution%' 
   ORDER BY mean_exec_time DESC 
   LIMIT 10;"

# Update table statistics
docker exec postgres psql -U postgres -d agent_evolution -c \
  "ANALYZE VERBOSE agent_evolution.performance_metrics;"
```

### 2.5 Debug Commands

The DEAN system provides comprehensive debugging utilities for troubleshooting complex issues:

```bash
# Economic debugging
docker exec agent-evolution dean-cli debug economy \
  --show-allocations \
  --show-consumption \
  --agent-id=<ID> \
  --trace-transactions

docker exec agent-evolution dean-cli debug token-trace \
  --time-range="1 hour" \
  --show-api-calls \
  --show-efficiency

# Diversity debugging
docker exec agent-evolution dean-cli debug diversity \
  --show-variance \
  --show-clusters \
  --generation=latest \
  --export-visualization=/app/data/debug/diversity_plot.png

docker exec agent-evolution dean-cli debug mutations \
  --list-pending \
  --show-targets \
  --explain-selection

# Pattern debugging
docker exec agent-evolution dean-cli debug patterns \
  --unclassified \
  --last=10 \
  --show-detection-scores \
  --show-features

docker exec agent-evolution dean-cli debug pattern-trace \
  --pattern-id=<ID> \
  --show-lineage \
  --show-applications

# System health
docker exec agent-evolution dean-cli health check \
  --comprehensive \
  --show-warnings \
  --show-recommendations

docker exec agent-evolution dean-cli health worktrees \
  --show-orphaned \
  --show-sizes \
  --suggest-cleanup

docker exec agent-evolution dean-cli health database \
  --analyze-performance \
  --check-indexes \
  --suggest-optimizations
```

## 3. Performance Tuning

### 3.1 Token Efficiency Optimization

**Allocation Algorithm Tuning**

The token allocation algorithm in IndexAgent/indexagent/agents/economy/token_manager.py requires periodic tuning based on observed performance patterns. Adjust weights to favor efficiency while maintaining exploration capacity:

```python
# Current allocation formula
allocation = base_amount * (
    efficiency_weight * historical_efficiency +
    diversity_weight * diversity_contribution +
    innovation_weight * pattern_discovery_rate
)
```

Tune these weights through configuration updates:

```bash
# Update allocation weights
docker exec agent-evolution dean-cli config update \
  --set="economy.allocation_weights.efficiency=0.6" \
  --set="economy.allocation_weights.diversity=0.2" \
  --set="economy.allocation_weights.innovation=0.2"

# Implement progressive bonuses
docker exec agent-evolution dean-cli config update \
  --set="economy.bonus_multipliers.sustained_efficiency=1.5" \
  --set="economy.bonus_multipliers.breakthrough_discovery=2.0"
```

**Budget Decay Optimization**

Calibrate budget decay rates based on typical task completion times in your repositories. Longer tasks require slower decay rates to prevent premature termination.

```bash
# Analyze task completion times
docker exec postgres psql -U postgres -d agent_evolution -c \
  "SELECT AVG(completion_time) as avg_time, 
   STDDEV(completion_time) as time_variance 
   FROM agent_evolution.task_metrics 
   WHERE status = 'completed';"

# Adjust decay based on findings
docker exec agent-evolution dean-cli config update \
  --set="economy.budget_decay.generations_before_decay=8" \
  --set="economy.budget_decay.rate=0.95"
```

### 3.2 Diversity Optimization

**Mutation Rate Calibration**

Optimal mutation rates depend on population size and convergence speed. Start with 10% baseline and adjust based on diversity trends:

```bash
# Monitor diversity trends
docker exec agent-evolution dean-cli diversity trends \
  --period="7 days" \
  --show-mutation-impact \
  --export=/app/data/analysis/diversity_trends.csv

# Implement adaptive mutation
docker exec agent-evolution dean-cli config update \
  --set="diversity.mutation_injection.strategy=adaptive" \
  --set="diversity.mutation_injection.min_rate=0.05" \
  --set="diversity.mutation_injection.max_rate=0.25"
```

**Pattern Import Scheduling**

Schedule pattern imports during low-activity periods to minimize disruption while maximizing diversity injection:

```bash
# Configure import schedule
docker exec agent-evolution dean-cli patterns schedule-import \
  --cron="0 3 * * *" \
  --source="/app/data/patterns/curated/" \
  --gradual-introduction \
  --test-first
```

### 3.3 Scalability Tuning

**Agent Parallelism**

Optimal agent count depends on available resources and repository complexity. Monitor resource utilization to find the sweet spot:

```bash
# Test different parallelism levels
for agents in 2 4 8 16; do
  docker exec agent-evolution dean-cli benchmark \
    --parallel-agents=$agents \
    --duration=3600 \
    --measure="throughput,efficiency,resource_usage"
done

# Analyze results
docker exec agent-evolution dean-cli benchmark analyze \
  --compare-runs \
  --recommend-optimal
```

**Database Optimization**

Implement PostgreSQL optimizations specific to the DEAN workload:

```sql
-- Create partial indexes for common queries
CREATE INDEX idx_recent_metrics ON agent_evolution.performance_metrics(agent_id, metric_name) 
WHERE recorded_at > NOW() - INTERVAL '7 days';

-- Implement table partitioning
CREATE TABLE agent_evolution.performance_metrics_2025_01 
PARTITION OF agent_evolution.performance_metrics 
FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

-- Configure connection pooling
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
```

### 3.4 Infrastructure Optimization

**Worktree Storage**

Optimize git worktree operations through strategic storage configuration:

```bash
# Move worktrees to fast storage
docker exec agent-evolution mkdir -p /nvme/worktrees
docker exec agent-evolution dean-cli config update \
  --set="agents.worktree_base_path=/nvme/worktrees"

# Enable worktree recycling
docker exec agent-evolution dean-cli config update \
  --set="agents.worktree_recycling.enabled=true" \
  --set="agents.worktree_recycling.pool_size=20"
```

**Container Resource Allocation**

Fine-tune Docker resource limits based on workload analysis:

```yaml
# Update docker-compose.yml
services:
  agent-evolution:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: '8G'
        reservations:
          cpus: '2.0'
          memory: '4G'
```

## 4. Backup and Recovery

### 4.1 Backup Procedures

**Automated Daily Backups**

The backup script at infra/modules/agent-evolution/scripts/backup-dean.sh executes daily via cron:

```bash
#!/bin/bash
# backup-dean.sh
set -euo pipefail

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/app/backups/${BACKUP_DATE}"
mkdir -p "${BACKUP_DIR}"

# Backup PostgreSQL
echo "Backing up database..."
pg_dump "${DATABASE_URL}" | gzip > "${BACKUP_DIR}/agent_evolution.sql.gz"

# Backup Redis
echo "Backing up Redis..."
redis-cli --rdb "${BACKUP_DIR}/redis_dump.rdb"

# Backup patterns and configurations
echo "Backing up patterns..."
tar -czf "${BACKUP_DIR}/patterns.tar.gz" /app/data/patterns
tar -czf "${BACKUP_DIR}/configs.tar.gz" /app/config

# Calculate checksums
cd "${BACKUP_DIR}"
sha256sum * > checksums.txt

# Upload to S3
aws s3 sync "${BACKUP_DIR}" "s3://dean-backups/${BACKUP_DATE}/" --storage-class GLACIER

# Clean local backups older than 7 days
find /app/backups -type d -mtime +7 -exec rm -rf {} +

echo "Backup completed: ${BACKUP_DATE}"
```

**Pattern Library Versioning**

Maintain version control for successful patterns:

```bash
# Export patterns to git repository
cd /app/data/pattern-library
git add patterns/*.json
git commit -m "Pattern library snapshot $(date +%Y%m%d)"
git push origin main
```

### 4.2 Recovery Procedures

**Complete System Recovery**

In case of catastrophic failure, follow this recovery sequence:

```bash
# 1. Restore database
LATEST_BACKUP=$(aws s3 ls s3://dean-backups/ | tail -1 | awk '{print $2}')
aws s3 sync "s3://dean-backups/${LATEST_BACKUP}" /tmp/restore/

gunzip -c /tmp/restore/agent_evolution.sql.gz | \
  psql "${DATABASE_URL}"

# 2. Restore Redis
redis-cli --rdb /tmp/restore/redis_dump.rdb

# 3. Restore configurations
tar -xzf /tmp/restore/configs.tar.gz -C /

# 4. Restart services
docker-compose down
docker-compose up -d

# 5. Verify recovery
docker exec agent-evolution dean-cli health check --post-recovery
```

**Partial Recovery**

For selective recovery of specific components:

```bash
# Recover specific generation
docker exec agent-evolution dean-cli recovery \
  --type=generation \
  --generation=42 \
  --source=s3://dean-backups/20250115/

# Recover patterns only
docker exec agent-evolution dean-cli recovery \
  --type=patterns \
  --after-date="2025-01-10" \
  --validate
```

## 5. Monitoring and Alerting

### 5.1 Key Metrics to Monitor

**Economic Health Indicators**

Configure Prometheus to track critical economic metrics:

```yaml
# prometheus/alerts/dean_economic.yml
groups:
  - name: dean_economic
    rules:
      - alert: HighTokenConsumption
        expr: rate(dean_tokens_consumed_total[5m]) > 1000
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High token consumption rate"
          
      - alert: LowEfficiency
        expr: dean_agent_efficiency < 0.5
        for: 30m
        labels:
          severity: critical
        annotations:
          summary: "Agent efficiency below threshold"
          
      - alert: BudgetNearExhaustion
        expr: dean_budget_remaining / dean_budget_total < 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Token budget nearly exhausted"
```

**Diversity Health Indicators**

Monitor genetic diversity continuously:

```yaml
# prometheus/alerts/dean_diversity.yml
groups:
  - name: dean_diversity
    rules:
      - alert: LowPopulationDiversity
        expr: dean_population_diversity < 0.3
        for: 15m
        labels:
          severity: critical
        annotations:
          summary: "Population diversity below minimum threshold"
          
      - alert: ConvergenceDetected
        expr: rate(dean_mutation_injections_total[1h]) > 5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Frequent mutation injections indicate convergence"
```

**System Health Indicators**

Track system-level health metrics:

```yaml
# prometheus/alerts/dean_system.yml
groups:
  - name: dean_system
    rules:
      - alert: HighAgentFailureRate
        expr: rate(dean_agent_failures_total[10m]) > 0.2
        for: 10m
        labels:
          severity: warning
          
      - alert: DatabaseSlowQueries
        expr: dean_database_query_duration_seconds > 1
        for: 15m
        labels:
          severity: warning
          
      - alert: WorktreeSpaceExhaustion
        expr: dean_worktree_disk_usage_percent > 80
        for: 5m
        labels:
          severity: critical
```

### 5.2 Alert Configuration

Configure alert routing through Alertmanager:

```yaml
# alertmanager/config.yml
global:
  slack_api_url: 'YOUR_SLACK_WEBHOOK_URL'

route:
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'dean-team'
  routes:
    - match:
        severity: critical
      receiver: 'dean-critical'
      
receivers:
  - name: 'dean-team'
    slack_configs:
      - channel: '#dean-alerts'
        title: 'DEAN System Alert'
        
  - name: 'dean-critical'
    slack_configs:
      - channel: '#dean-critical'
        title: 'CRITICAL: DEAN System Alert'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_KEY'
```

## 6. Emergency Procedures

### 6.1 Token Budget Exhaustion

When the global token budget is exhausted:

```bash
# 1. Immediately pause all agents
docker exec agent-evolution dean-cli agents pause --all

# 2. Analyze consumption patterns
docker exec agent-evolution dean-cli budget analyze \
  --breakdown-by-agent \
  --show-inefficient

# 3. Allocate emergency reserve
docker exec agent-evolution dean-cli budget allocate \
  --source=emergency_reserve \
  --amount=10000 \
  --restrict-to="high_efficiency_agents"

# 4. Resume critical agents only
docker exec agent-evolution dean-cli agents resume \
  --filter="efficiency>0.01"
```

### 6.2 Diversity Collapse

When diversity collapses below critical levels:

```bash
# 1. Trigger emergency diversification
docker exec agent-evolution dean-cli diversity emergency \
  --inject-random-mutations \
  --rate=0.5 \
  --import-external-patterns

# 2. Reset convergent population subset
docker exec agent-evolution dean-cli agents reset \
  --bottom-performers=30 \
  --randomize-genomes

# 3. Adjust evolution parameters
docker exec agent-evolution dean-cli config emergency \
  --set="evolution.exploration_bonus=2.0" \
  --set="diversity.minimum_variance=0.4" \
  --duration="10 generations"
```

### 6.3 System Overload

When the system becomes overloaded:

```bash
# 1. Reduce agent parallelism
docker exec agent-evolution dean-cli agents scale \
  --target=4 \
  --graceful

# 2. Clear pattern cache
docker exec agent-registry redis-cli FLUSHDB

# 3. Optimize database
docker exec postgres psql -U postgres -d agent_evolution -c \
  "VACUUM FULL agent_evolution.performance_metrics;"

# 4. Restart with reduced load
docker-compose restart agent-evolution
```

## 7. Maintenance Windows

### 7.1 Scheduled Maintenance

Plan maintenance windows during low-activity periods:

```bash
# Announce maintenance
docker exec agent-evolution dean-cli maintenance announce \
  --start="2025-01-20 02:00:00" \
  --duration="2 hours" \
  --reason="Database optimization and pattern library update"

# Execute maintenance
docker exec agent-evolution dean-cli maintenance start \
  --pause-agents \
  --complete-running-tasks \
  --backup-first
```

### 7.2 Rolling Updates

Perform updates without full system downtime:

```bash
# Update agent-evolution service
docker-compose up -d --no-deps --build agent-evolution

# Verify new version
docker exec agent-evolution dean-cli version

# Run migration if needed
docker exec agent-evolution dean-cli migrate \
  --check-compatibility \
  --backup-first
```

## 8. Documentation Maintenance

Keep operational documentation current:

```bash
# Generate operational report
docker exec agent-evolution dean-cli docs generate \
  --type=operational \
  --include-configs \
  --include-procedures \
  --output=/app/docs/operations_manual.md

# Update runbooks
docker exec agent-evolution dean-cli docs update-runbook \
  --from-recent-incidents \
  --validate-procedures
```