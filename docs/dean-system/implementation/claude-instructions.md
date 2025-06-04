## Instruction Prompt for Claude Code
Execute the following implementation tasks to complete the DEAN system infrastructure and enable operational deployment. Focus on creating the missing service layer and integration components that will allow the existing core logic to function within the broader system architecture.

### Phase 1: FastAPI Service Implementation (Priority: Critical)
Navigate to the infra repository and create the FastAPI service that will expose the DEAN agent evolution capabilities:
1. Create `modules/agent-evolution/main.py` with the following implementation:
   - FastAPI application initialization with proper CORS and middleware configuration
   - Health check endpoint at `/health` returning service status and component connectivity
   - Agent management endpoints:
     - `POST /api/v1/agents/spawn` - Spawn new agent population with token budget constraints
     - `GET /api/v1/agents` - List all active agents with their status and metrics
     - `POST /api/v1/agents/{agent_id}/evolve` - Trigger evolution for specific agent
     - `GET /api/v1/patterns/discovered` - List discovered patterns with effectiveness scores
     - `GET /api/v1/metrics/efficiency` - Get population-wide efficiency metrics
   - Prometheus metrics endpoint at `/metrics` for monitoring
   - Proper error handling, logging, and request validation
   - Integration with the existing agent evolution logic in `src/dean/`

2. Create supporting files:
   - `modules/agent-evolution/requirements.txt` with FastAPI, uvicorn, prometheus-client, and other dependencies
   - `modules/agent-evolution/__init__.py` for proper module initialization

### Phase 2: Docker Infrastructure (Priority: Critical)
3. Create `modules/agent-evolution/docker/Dockerfile`:
   - Base image: `python:3.11-slim`
   - Install system dependencies for git operations
   - Copy and install Python requirements
   - Configure proper working directory and user permissions
   - Expose port 8080
   - Set uvicorn as the entry point with appropriate production settings

4. Update `docker-compose.yml` to add DEAN services:
   ```yaml
   agent-evolution:
     build:
       context: ./modules/agent-evolution
       dockerfile: docker/Dockerfile
     container_name: agent-evolution
     ports:
       - "${AGENT_EVOLUTION_PORT:-8090}:8080"
     environment:
       - DATABASE_URL=postgresql://postgres:password@postgres:5432/agent_evolution
       - REDIS_URL=redis://agent-registry:6379
       - INDEXAGENT_URL=http://indexagent:8080
       - AIRFLOW_URL=http://airflow-service:8080
     depends_on:
       - postgres
       - agent-registry
     networks:
       - agent-network
   
   agent-registry:
     image: redis:7-alpine
     container_name: agent-registry
     ports:
       - "${AGENT_REGISTRY_PORT:-6380}:6379"
     networks:
       - agent-network
   ```

5. Update `.env.example` with DEAN-specific variables:
   ```
   AGENT_EVOLUTION_PORT=8090
   AGENT_REGISTRY_PORT=6380
   AGENT_TOKEN_LIMIT=4096
   AGENT_MAX_CONCURRENT=10
   DEAN_MIN_DIVERSITY=0.3
   AGENT_EVOLUTION_DATABASE_URL=postgresql://postgres:password@postgres:5432/agent_evolution
   ```

### Phase 3: Database Infrastructure (Priority: Critical)
6. Create `modules/agent-evolution/scripts/init-database.sql` with the complete schema:
   - Create `agent_evolution` schema
   - Define tables: agents, discovered_patterns, performance_metrics, evolution_history
   - Add all necessary indexes for query optimization
   - Include initial seed data if appropriate

7. Create database initialization script `modules/agent-evolution/scripts/init-db.sh`:
   - Check if database exists
   - Execute SQL schema creation
   - Verify table creation
   - Set up initial configuration records

### Phase 4: Configuration Management (Priority: High)
8. Create `modules/agent-evolution/config/agents.yaml`:
   ```yaml
   agent_types:
     code_analyzer:
       capabilities: ["syntax_analysis", "pattern_detection"]
       resource_limits:
         cpu: "0.5"
         memory: "512Mi"
         token_budget: 4096
   ```

9. Create `modules/agent-evolution/config/evolution.yaml`:
   ```yaml
   evolution_parameters:
     population_size: 8
     generations: 50
     mutation_rate: 0.1
     crossover_rate: 0.7
     diversity_threshold: 0.3
   ```

### Phase 5: Airflow Integration (Priority: High)
Navigate to the airflow-hub repository:
10. Create DAG structure:
    ```bash
    mkdir -p dags/agent_evolution
    mkdir -p plugins/agent_evolution/operators
    ```

11. Create `dags/agent_evolution/agent_lifecycle.yaml` with the dean_agent_lifecycle DAG definition
12. Create `plugins/agent_evolution/operators/agent_spawn_operator.py` with AgentSpawnOperator implementation that calls the agent-evolution API
13. Create `plugins/agent_evolution/operators/agent_evolution_operator.py` with AgentEvolutionOperator implementation

### Validation Requirements
After implementing each phase:
- Verify file creation and content correctness
- Test API endpoints with curl commands
- Ensure Docker services build without errors
- Validate database schema creation
- Confirm Airflow can load the new DAGs

### Critical Implementation Notes
1. **Use Existing Core Logic**: The comprehensive agent evolution implementation already exists in `modules/agent-evolution/src/dean/`. The FastAPI service should import and utilize these existing modules rather than reimplementing functionality.
2. **Maintain Separation of Concerns**: Keep the API layer thin, delegating business logic to the existing core modules.
3. **Error Handling**: Implement proper error handling at the API layer to prevent agent failures from crashing the service.
4. **Asynchronous Operations**: Use FastAPI's async capabilities for non-blocking agent operations.
5. **Configuration Loading**: Imdplement proper configuration loading from YAML files with environment variable overrides.

Begin with Phase 1 and report progress after completing the FastAPI service implementation. 
Focus on creating a working service that can be tested before proceeding to containerization.