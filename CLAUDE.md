# CLAUDE.md - infra Repository

This file provides repository-specific guidance for infrastructure and services.

## Review Report Requirements

<review-report-standards>
When creating review reports for completed tasks, Claude Code MUST follow these standards:

1. **Naming Convention**: 
   - XML format: `REVIEW_REPORT_YYYY-MM-DD-HHMM_XML.md`
   - Markdown format: `REVIEW_REPORT_YYYY-MM-DD-HHMM_MD.md`
   - Example: `REVIEW_REPORT_2025-06-27-1452_XML.md` and `REVIEW_REPORT_2025-06-27-1452_MD.md`
   - Time should be in 24-hour format (e.g., 1452 for 2:52 PM)

2. **Dual Format Requirement**:
   - Always create TWO versions of each review report
   - XML version: Use XML syntax throughout for structured data
   - Markdown version: Use standard markdown formatting for readability
   - Both must contain identical information, only formatting differs

3. **Storage Location**:
   - All review reports MUST be saved in: `.claude/review-reports/`
   - Create the directory if it doesn't exist: `mkdir -p .claude/review-reports`
   - This applies to ALL repositories in the DEAN system

4. **Required Metadata**:
   Each review report MUST include metadata at the top:
   ```xml
   <report-metadata>
     <creation-date>YYYY-MM-DD</creation-date>
     <creation-time>HH:MM PST/EST</creation-time>
     <report-type>Implementation Review/Bug Fix/Feature Addition/etc.</report-type>
     <author>Claude Code Assistant</author>
     <system>DEAN</system>
     <component>Component Name</component>
     <task-id>Unique Task Identifier</task-id>
   </report-metadata>
   ```
</review-report-standards>

## Repository Purpose
This repository provides infrastructure and shared services:
- Docker Compose configurations
- Database schemas and migrations
- API services (Economic Governor, etc.)
- Deployment scripts and utilities

## CRITICAL: This is Part of a Distributed System

<distributed_system_warning>
⚠️ **WARNING: The DEAN system spans FOUR repositories** ⚠️

This repository contains ONLY the infrastructure and deployment configurations. Other components are located in:
- **DEAN**: Orchestration, authentication, monitoring (Port 8082-8083)
- **IndexAgent**: Agent logic, evolution algorithms (Port 8081)
- **airflow-hub**: DAGs, operators, workflow orchestration (Port 8080)

**Specification Documents Location**: DEAN/specifications/ (read-only)

Always check all repositories before implementing features!
</distributed_system_warning>

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

## Key Components in This Repository

### DEAN-Related Infrastructure
```
# Docker Configurations
docker-compose.dean.yml           # Main DEAN deployment
docker-compose.dean.prod.yml      # Production configuration
docker-compose.dean-complete.yml  # Complete system setup

# Database
database/init_agent_evolution.sql # Complete agent_evolution schema
database/dean_schema_complete.sql # Additional schema definitions

# Services
services/dean_api/               # Evolution API implementation
services/economy/                # Economic governor service
services/evolution/              # Evolution support services

# Modules
modules/agent-evolution/         # Agent evolution module
  ├── docker/                   # Docker configurations
  ├── config/                   # Service configurations
  ├── monitoring/               # Grafana dashboards
  └── scripts/                  # Deployment scripts

# Monitoring
monitoring/prometheus.yml        # Prometheus configuration
monitoring/dean_alerts.yml       # DEAN alert rules
```

### What This Repository Does NOT Contain
- **Agent Implementation**: Located in IndexAgent/indexagent/agents/
- **Orchestration Logic**: Located in DEAN/src/dean_orchestration/
- **DAG Definitions**: Located in airflow-hub/dags/dean/
- **Agent CLI**: Located in DEAN/src/interfaces/cli/

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