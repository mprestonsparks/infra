# CLAUDE.md - infra Repository

This file provides repository-specific guidance for infrastructure and services.

## Repository Purpose
This repository provides infrastructure and shared services:
- Docker Compose configurations
- Database schemas and migrations
- API services (Economic Governor, etc.)
- Deployment scripts and utilities

## Critical Implementation Requirements

### NO MOCK IMPLEMENTATIONS

When implementing any feature in this codebase, Claude Code MUST create actual, working code. The following are STRICTLY PROHIBITED:
- Mock implementations or stub functions presented as complete
- Placeholder code with TODO comments in "finished" work
- Simulated test results or hypothetical outputs
- Documentation of what "would" happen instead of what "does" happen
- Pseudocode or conceptual implementations claimed as functional

### REAL CODE ONLY

Every implementation in this project MUST:
- Be fully functional and executable with proper error handling
- Work with actual services and dependencies
- Be tested with real commands showing actual output
- Include complete implementations of all code paths

## Infrastructure-Specific Guidelines

### Service Development
- All services must expose health endpoints
- Implement proper database connection pooling
- Use environment variables for configuration
- Include comprehensive error logging

### API Implementation
- Follow RESTful conventions
- Implement request validation
- Return consistent error responses
- Document all endpoints with OpenAPI

### Database Management
- Always use migrations for schema changes
- Implement proper indexing strategies
- Use transactions for data consistency
- Monitor query performance

## Testing Requirements
- Test APIs with real database connections
- Verify Docker Compose configurations
- Test database migrations up and down
- Validate service health checks

## Common Commands
- Start services: `docker-compose -f docker-compose.dean.yml up`
- Run migrations: `alembic upgrade head`
- Test API: `pytest services/dean_api/tests/`
- Check service health: `docker-compose ps`
- View logs: `docker-compose logs -f service-name`

## Project Structure
```
infra/
├── database/                      # Database schemas and migrations
│   ├── dean_schema_complete.sql   # Complete DEAN schema
│   └── migrations/                # Alembic migrations
├── docker/                        # Docker configurations
├── modules/
│   └── agent-evolution/          # Agent evolution module
├── scripts/                      # Utility scripts
├── services/
│   └── dean_api/                 # DEAN API service
│       ├── main.py              # FastAPI application
│       └── endpoints/           # API endpoints
└── docker-compose.dean.yml       # Main compose file
```

## Critical Services
- PostgreSQL: Main database with agent_evolution schema
- Redis: Caching and agent registry
- DEAN API: Economic governance and coordination
- Prometheus: Metrics collection
- Grafana: Metrics visualization

## Docker Compose Configuration
```yaml
# Service template
service-name:
  build:
    context: ./services/service-name
    dockerfile: Dockerfile
  environment:
    - DATABASE_URL=${DATABASE_URL}
    - REDIS_URL=${REDIS_URL}
  ports:
    - "8091:8080"
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
    interval: 30s
    timeout: 10s
    retries: 3
  networks:
    - dean-network
```

## Environment Variables
```bash
# Database
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/dean_production
AGENT_EVOLUTION_DATABASE_URL=postgresql://postgres:postgres@postgres:5432/agent_evolution

# Redis
REDIS_URL=redis://redis:6379

# API Keys
DEAN_SERVICE_API_KEY=your-service-key
CLAUDE_API_KEY=your-claude-key

# Service URLs
DEAN_API_URL=http://dean-api:8082
INDEXAGENT_URL=http://indexagent:8081
AIRFLOW_URL=http://airflow:8080
```

## Database Schema
- Schema: `agent_evolution`
- Key tables:
  - `agents`: Agent metadata and configuration
  - `performance_metrics`: Token usage and efficiency
  - `evolution_history`: Generation tracking
  - `discovered_patterns`: Pattern catalog

## API Standards
```python
# Health endpoint (required for all services)
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "service-name",
        "timestamp": datetime.utcnow().isoformat()
    }

# Error response format
{
    "error": "Error message",
    "detail": "Detailed explanation",
    "status_code": 400,
    "timestamp": "2024-01-01T00:00:00Z"
}
```

## Performance Requirements
- API response time < 200ms
- Database queries < 100ms
- Health checks < 5s timeout
- Container startup < 30s

## Security Guidelines
- Never hardcode credentials
- Use environment variables for secrets
- Implement request rate limiting
- Validate all input data
- Use HTTPS in production

## Monitoring and Logging
- All services must log to stdout
- Use structured logging (JSON format)
- Include correlation IDs in logs
- Export metrics to Prometheus
- Set up alerts for critical metrics