# Token Economy Service

## Overview

The Token Economy Service implements **hard enforcement** of token budget constraints for DEAN agents. This service ensures that agents cannot exceed their allocated token budgets and are automatically terminated when budgets are exhausted.

## Key Features

### 1. Hard Budget Enforcement
- Agents are **actually stopped** when token budget is exhausted (not just logged)
- No token consumption is possible beyond allocated limits
- Automatic agent termination with proper cleanup

### 2. Efficiency-Based Allocation
- Base allocation: `min(requested, 1000)` tokens
- Efficiency adjustment: `base * (0.5 + historical_efficiency)`
- Global constraint: Cannot exceed `global_budget - already_allocated`

### 3. Real-Time Tracking
- Prometheus metrics updated in real-time
- All allocations tracked in PostgreSQL database
- Redis used for fast, real-time state management

### 4. Value-Based Optimization
- Efficiency calculated as `value_generated / tokens_consumed`
- High-efficiency agents receive larger future allocations
- Automatic performance-based budget adjustments

## API Endpoints

### POST /api/v1/tokens/allocate
Allocate tokens to an agent with efficiency-based adjustment.

**Request:**
```json
{
  "agent_id": "agent-001",
  "requested_tokens": 1000.0,
  "task_complexity": 0.5
}
```

**Response:**
```json
{
  "agent_id": "agent-001",
  "allocated_tokens": 850.0,
  "efficiency_multiplier": 0.85,
  "remaining_global_budget": 99150.0,
  "warning": "Reduced allocation due to constraints"
}
```

### POST /api/v1/tokens/consume
Update token consumption for an agent. **Will terminate agent if budget exhausted!**

**Request:**
```json
{
  "agent_id": "agent-001",
  "tokens_consumed": 100.0,
  "value_generated": 5.0
}
```

**Response:**
```json
{
  "agent_id": "agent-001",
  "consumed": 100.0,
  "remaining": 750.0,
  "efficiency": 0.05,
  "warning": null
}
```

### GET /api/v1/tokens/status
Get comprehensive budget status.

**Response:**
```json
{
  "global_budget": 100000.0,
  "allocated_total": 25000.0,
  "consumed_total": 15000.0,
  "remaining_budget": 75000.0,
  "active_agents": 12,
  "average_efficiency": 0.023,
  "top_performers": [...],
  "budget_utilization_percent": 25.0
}
```

### GET /api/v1/tokens/agent/{agent_id}
Get current token status for a specific agent.

## Running the Service

### Docker Compose

```bash
# Start all services
docker-compose -f docker-compose.token_economy.yml up -d

# View logs
docker-compose -f docker-compose.token_economy.yml logs -f token-economy

# Check health
curl http://localhost:8091/health
```

### Environment Variables

- `GLOBAL_TOKEN_BUDGET`: Total token budget (default: 100000)
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `TOKEN_ECONOMY_PORT`: Service port (default: 8091)

## Client Usage

### Python Client

```python
from services.token_economy_client import TokenEconomyClient, TokenBudgetManager

async def agent_task():
    async with TokenEconomyClient() as client:
        # Method 1: Direct API calls
        allocation = await client.allocate_tokens("agent-001", 1000.0)
        
        # Do work...
        
        await client.update_consumption("agent-001", 100.0, value_generated=5.0)
        
        # Method 2: Automatic tracking with context manager
        async with TokenBudgetManager(client, "agent-002", 500.0) as budget:
            # Do work...
            await budget.consume(tokens=50.0, value=2.0)
            # Automatically reports consumption on exit
```

### Hard Stop Example

When an agent exceeds its budget:

```python
# This will raise TokenBudgetExhausted and terminate the agent
try:
    await client.update_consumption("agent-001", 1000.0)  # More than allocated
except TokenBudgetExhausted as e:
    print(f"Agent terminated: {e}")
    # Agent is now inactive and cannot consume more tokens
```

## Metrics

The service exposes Prometheus metrics at `/metrics`:

- `dean_tokens_allocated_total`: Total tokens allocated by agent
- `dean_tokens_consumed_total`: Total tokens consumed by agent
- `dean_token_efficiency`: Histogram of efficiency values
- `dean_active_token_allocation`: Current allocation per agent
- `dean_global_budget_remaining`: Remaining global budget
- `dean_agents_terminated_budget`: Count of agents terminated due to budget

## Testing

```bash
# Run tests
cd infra/modules/agent-evolution
pytest tests/test_token_economy.py -v

# Test hard enforcement
pytest tests/test_token_economy.py::TestTokenEconomy::test_hard_stop_on_budget_exhaustion -v
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Agent Code    │────▶│  Token Economy  │────▶│   PostgreSQL    │
│                 │     │    Service      │     │   (Tracking)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │                          │
                               ▼                          ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │     Redis       │     │   Prometheus    │
                        │  (Real-time)    │     │   (Metrics)     │
                        └─────────────────┘     └─────────────────┘
```

## Integration with DEAN

The Token Economy Service integrates with other DEAN components:

1. **IndexAgent**: Requests tokens before executing tasks
2. **Airflow DAGs**: Monitor budget consumption across workflows
3. **Evolution Service**: Allocates tokens to new agents based on parent efficiency
4. **DEAN Orchestration**: Provides unified budget management interface

## Troubleshooting

### Common Issues

1. **"Global token budget exhausted"**
   - Check budget status: `curl http://localhost:8091/api/v1/tokens/status`
   - Increase global budget or wait for next budget cycle

2. **"Agent terminated: Token budget exhausted"**
   - Agent exceeded allocation and was stopped
   - Review agent efficiency metrics
   - Consider adjusting task complexity estimates

3. **Database connection errors**
   - Ensure PostgreSQL is running and accessible
   - Check DATABASE_URL environment variable
   - Verify agent_evolution schema exists

### Debug Commands

```bash
# Check service health
curl http://localhost:8091/health

# View real-time metrics
curl http://localhost:8091/metrics | grep dean_

# Check specific agent status
curl http://localhost:8091/api/v1/tokens/agent/agent-001

# Monitor Redis state
docker exec dean-redis redis-cli
> HGETALL agent_allocation:agent-001
> GET global_budget_remaining
```