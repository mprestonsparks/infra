# Architectural Design Document (Final)

## 1. Architecture Overview

The DEAN system follows a distributed, microservices-inspired architecture with autonomous agents operating in parallel across four integrated repositories. Within the DEAN ecosystem, the DEAN repository serves as the orchestration layer, providing unified control, authentication, and monitoring for DEAN-specific workflows. Each service is designed to function independently, with DEAN providing orchestration when these services are deployed together as the complete DEAN system. The architecture fundamentally incorporates economic constraints, diversity maintenance, and emergent behavior capture as core architectural principles implemented through specific infrastructure components.

## 2. Service Independence and Reusability

### 2.1 Design Principles for Independence

Each service in the DEAN architecture is designed with independence as a core principle:

1. **Self-Contained Functionality**: Each repository provides complete, useful functionality on its own
   - **airflow-hub**: General-purpose workflow orchestration platform
   - **IndexAgent**: Code indexing and search service
   - **infra**: Infrastructure and deployment tools

2. **Loose Coupling via APIs**: Services communicate through well-defined REST APIs
   - No compile-time dependencies between services
   - Configuration-based service discovery
   - Graceful degradation when services unavailable

3. **Configuration Over Code**: Service endpoints and integration points are configurable
   - No hardcoded service addresses
   - Environment-specific configuration support
   - Optional integration points

### 2.2 Using Services Independently

Each service can be deployed and used without the others:

#### airflow-hub Standalone Usage
```bash
# Deploy only airflow-hub
cd airflow-hub
docker-compose up -d

# Use for any workflow orchestration needs
# DEAN DAGs are optional - in the dean/ subdirectory
```

#### IndexAgent Standalone Usage
```bash
# Deploy only IndexAgent
cd IndexAgent
docker-compose up -d

# Use for code search and indexing
# No dependency on DEAN orchestration
```

### 2.3 Integration Patterns

When services are used together, follow these patterns to maintain independence:

#### Good: Configuration-Based Integration
```python
# Use Airflow Connections
conn = BaseHook.get_connection("my_service_api")
api_url = f"{conn.schema}://{conn.host}:{conn.port}"
```

#### Bad: Hardcoded Integration
```python
# Avoid hardcoded URLs
api_url = "http://my-service:8080"  # ❌ Creates tight coupling
```

#### Good: Optional Dependencies
```python
# Check if service is available
try:
    response = requests.get(f"{api_url}/health")
    if response.status_code == 200:
        # Use enhanced features
except:
    # Continue with basic functionality
```

### 2.4 DEAN's Role in the Ecosystem

The DEAN orchestration layer provides value when deploying the complete DEAN system:

1. **Unified Interface**: Single point of interaction for the complete system
2. **Cross-Service Workflows**: Coordinates complex operations across services
3. **Centralized Auth**: Optional unified authentication layer
4. **Aggregated Monitoring**: Consolidated view of all services

However, DEAN is **not required** for:
- Running Airflow workflows (use airflow-hub directly)
- Code indexing and search (use IndexAgent directly)
- Infrastructure deployment (use infra tools directly)

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│           DEAN System Orchestration Layer (Optional)                │
│                    (DEAN/ - Port 8082)                             │
│    Provides Unified Control for DEAN System Deployments            │
├─────────────────────────────────────────────────────────────────────┤
│              Apache Airflow 3.0.0 Task Orchestration               │
│                 (airflow-hub/dags/agent_evolution/)                 │
│                    with Economic Governor                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │   Agent 1   │  │   Agent 2   │  │   Agent N   │                  │
│  │  Worktree   │  │  Worktree   │  │  Worktree   │                  │
│  │ Token: 4096 │  │ Token: 3072 │  │ Token: 2048 │                  │
│  └─────────────┘  └─────────────┘  └─────────────┘                  │
│         ↓                ↓                ↓                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │         IndexAgent Core Logic (Port 8081)                   │    │
│  │      (IndexAgent/indexagent/agents/patterns/)               │    │
│  │         Pattern Detection & Classification                  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                    Shared Infrastructure (infra/)                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │    DSPy     │  │ Claude CLI  │  │ PostgreSQL  │  │   Redis     │ │
│  │  Optimizer  │  │   Docker    │  │  Metrics    │  │  Registry   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Meta-Learning Engine (Port 8090)               │    │
│  │           (infra/modules/agent-evolution/)                  │    │
│  │      Pattern Extraction & Strategy Evolution                │    │ 
│  └─────────────────────────────────────────────────────────────┘    │ 
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Repository Integration Architecture

```
┌──────────────────────┐
│       DEAN/          │  Primary Orchestration & User Interface
├──────────────────────┤
│ src/                 │
│ ├─ dean_orchestration│──────┐
│ ├─ integration/      │      │ Coordinates all services
│ ├─ interfaces/       │      │
│ └─ auth/            │      ▼
└──────────────────────┘  ┌────────────────────────────────────────────┐
                         │        Service Integration Layer            │
┌──────────────────────┐  ├────────────────────────────────────────────┤
│      infra/          │  │  ┌──────────────┐  ┌──────────────────┐    │
├──────────────────────┤  │  │ airflow-hub/ │  │   IndexAgent/    │    │
│ modules/             │  │  ├──────────────┤  ├──────────────────┤    │
│ └─ agent-evolution/  │──┼─▶│ dags/        │  │ indexagent/      │    │
│     ├─ docker/       │  │  │ └─ agent_    │  │ └─ agents/       │    │
│     ├─ config/       │  │  │   evolution/ │  │     ├─ evolution/│    │
│     └─ scripts/      │  │  │              │  │     ├─ patterns/ │    │
│                      │  │  │ plugins/     │  │     └─ economy/  │    │
│ docker-compose.yml   │  │  └──────────────┘  └──────────────────┘    │
└──────────────────────┘  └────────────────────────────────────────────┘
         ▲                            ▲                      ▲
         │                            │                      │
         └────────────────────────────┴──────────────────────┘
                    Central Configuration via DEAN + .env
```

### 3.3 Service Communication Architecture

```yaml
# Primary orchestration endpoint (all external communication flows through DEAN)
DEAN_ORCHESTRATION_API: "http://dean-server:8082/api/v1"
DEAN_WEB_DASHBOARD: "http://dean-server:8083"

# Internal service endpoints (accessed via DEAN orchestration)
INDEXAGENT_API: "http://indexagent:8081/api/v1"
AIRFLOW_API: "http://airflow-service:8080/api/v1"
AGENT_EVOLUTION_API: "http://agent-evolution:8090/api/v1"
AGENT_REGISTRY: "redis://agent-registry:6379"
DATABASE_URL: "postgresql://postgres:password@postgres:5432/agent_evolution"

# Authentication flow
User → DEAN (Auth) → Service Authentication → Service Action
```

## 4. Component Architecture

### 4.1 DEAN Orchestration Components

The DEAN repository provides the primary orchestration layer with the following key components:

#### 3.1.1 Orchestration Server (`src/dean_orchestration/`)
- **Main Server**: FastAPI application running on port 8082
- **WebSocket Support**: Real-time updates and monitoring
- **Request Routing**: Intelligently routes requests to appropriate services
- **Health Aggregation**: Collects and reports health status from all services

#### 3.1.2 Authentication System (`src/auth/`)
- **Unified Authentication**: Single sign-on across all services
- **Service Authentication**: Inter-service authentication tokens
- **Role-Based Access Control**: Granular permissions management
- **JWT Token Management**: Secure token generation and validation

#### 3.1.3 Service Integration Layer (`src/integration/`)
- **IndexAgent Client**: Manages agent population and pattern detection
- **Airflow Client**: Triggers and monitors DAG execution
- **Evolution Client**: Coordinates evolution trials and metrics
- **Infra Client**: Manages deployment and infrastructure operations

#### 3.1.4 User Interfaces (`src/interfaces/`)
- **CLI Interface** (`dean-cli`): Command-line tool for system control
  - Evolution management: `dean-cli evolution start/stop/status`
  - Service health: `dean-cli service health`
  - Workflow execution: `dean-cli workflow execute`
- **Web Dashboard**: Browser-based monitoring and control
  - Real-time metrics visualization
  - Service status monitoring
  - Evolution trial management
- **REST API**: Programmatic access to all orchestration features

#### 3.1.5 Orchestration Logic (`src/orchestration/`)
- **Workflow Coordination**: Multi-service workflow execution
- **Evolution Trial Management**: Complete evolution lifecycle orchestration
- **Pattern Propagation**: Cross-domain pattern transfer coordination
- **Resource Management**: Token budget and resource allocation

### 4.2 Agent Runtime Layer

#### Container Architecture
```yaml
# Docker service definition in infra/docker-compose.yml
agent-evolution:
  build:
    context: ../infra/modules/agent-evolution
    dockerfile: Dockerfile
  container_name: agent-evolution
  platform: "${DOCKER_DEFAULT_PLATFORM}"
  ports:
    - "${AGENT_EVOLUTION_PORT}:8080"
  environment:
    - INDEXAGENT_URL=http://indexagent:8080
    - AIRFLOW_URL=http://airflow-service:8080
    - DATABASE_URL=${AGENT_EVOLUTION_DATABASE_URL}
    - REDIS_URL=redis://agent-registry:6379
    - CLAUDE_API_KEY=${CLAUDE_API_KEY}
    - TOKEN_LIMIT_PER_AGENT=${AGENT_TOKEN_LIMIT:-4096}
  volumes:
    - ../infra/modules/agent-evolution/data:/app/data
    - ../infra/modules/agent-evolution/logs:/app/logs
    - agent-models:/app/models
    - /var/run/docker.sock:/var/run/docker.sock  # For Claude CLI
  networks:
    - agent-network
  resource_limits:
    cpus: '${AGENT_MAX_CPU:-2.0}'
    memory: '${AGENT_MAX_MEMORY:-2Gi}'
```

#### Component Distribution
- **IndexAgent Repository**: Core agent logic, evolution algorithms, pattern detection
- **Airflow Hub Repository**: DAG definitions, custom operators, scheduling logic
- **Infrastructure Repository**: Docker configurations, deployment scripts, monitoring

### 3.2 Orchestration Layer

#### Airflow Architecture
```python
# airflow-hub/plugins/agent_evolution/operators/agent_spawn_operator.py
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults

class AgentSpawnOperator(BaseOperator):
    """Spawns DEAN agents with economic constraints"""
    
    @apply_defaults
    def __init__(self, population_size: int, token_limit: int, 
                 diversity_threshold: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.population_size = population_size
        self.token_limit = token_limit
        self.diversity_threshold = diversity_threshold
    
    def execute(self, context):
        # Connect to agent-evolution service
        agent_api = AgentEvolutionAPI(
            base_url="http://agent-evolution:8080"
        )
        
        # Spawn agents with constraints
        agents = []
        for i in range(self.population_size):
            agent = agent_api.create_agent(
                token_budget=self.token_limit,
                diversity_score=random.uniform(0.3, 1.0)
            )
            agents.append(agent)
        
        return agents
```

### 3.3 Knowledge Repository Layer

#### Database Architecture
```sql
-- PostgreSQL with dedicated schema
CREATE SCHEMA IF NOT EXISTS agent_evolution;

-- Connection pooling configuration
-- Max connections: 100
-- Pool size: 20
-- Overflow: 10

-- Table partitioning for time-series data
CREATE TABLE agent_evolution.performance_metrics_2025_01 
PARTITION OF agent_evolution.performance_metrics
FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

-- Materialized views for analytics
CREATE MATERIALIZED VIEW agent_evolution.agent_efficiency_summary AS
SELECT 
    agent_id,
    AVG(metric_value) as avg_efficiency,
    COUNT(*) as measurement_count,
    MAX(recorded_at) as last_updated
FROM agent_evolution.performance_metrics
WHERE metric_name = 'token_efficiency'
GROUP BY agent_id;

-- Refresh every hour
CREATE EXTENSION IF NOT EXISTS pg_cron;
SELECT cron.schedule('refresh_efficiency', '0 * * * *', 
    'REFRESH MATERIALIZED VIEW CONCURRENTLY agent_evolution.agent_efficiency_summary');
```

#### Redis Architecture
```python
# Agent registry using Redis for fast access
# infra/modules/agent-evolution/config/redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

### 3.4 Meta-Learning Layer

#### Pattern Extraction Pipeline
```
Raw Metrics → Pattern Detection → Classification → Cataloging → Reuse
     ↓              ↓                   ↓              ↓          ↓
PostgreSQL    IndexAgent API      ML Models      Redis Cache   Import

Pattern flow implementation:
1. PostgreSQL stores raw metrics
2. IndexAgent API analyzes agent behaviors
3. ML models classify patterns
4. Redis caches frequently used patterns
5. Import scripts enable cross-domain transfer
```

## 5. Data Architecture

### 4.1 Data Flow Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Airflow DAGs   │────▶│  Agent Runtime  │────▶│   PostgreSQL    │
│  (Triggers)     │     │  (Execution)    │     │   (Storage)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                        │
         │                       ▼                        ▼
         │              ┌─────────────────┐     ┌─────────────────┐
         └─────────────▶│   Prometheus    │     │   Redis Cache   │
                        │   (Metrics)     │     │   (Patterns)    │
                        └─────────────────┘     └─────────────────┘
```

### 4.2 Storage Architecture

#### Primary Storage (PostgreSQL)
- **Location**: Dedicated PostgreSQL instance
- **Schema**: `agent_evolution`
- **Backup**: Daily automated backups to S3
- **Replication**: Read replicas for analytics

#### Cache Layer (Redis)
- **Purpose**: Agent registry and pattern cache
- **Persistence**: AOF with fsync every second
- **Eviction**: LRU for pattern cache
- **Clustering**: Redis Sentinel for HA

#### File Storage
- **Worktrees**: Local SSD storage with cleanup policies
- **Logs**: JSON structured logs to `/app/logs`
- **Models**: Persistent volume for trained models

## 6. Deployment Architecture

### 5.1 Docker Compose Architecture

```yaml
# Complete service stack with DEAN orchestration
version: "3.8"

services:
  dean-server:
    build:
      context: ../DEAN
      dockerfile: Dockerfile
    container_name: dean-orchestration
    ports:
      - "${DEAN_SERVER_PORT:-8082}:8082"
      - "${DEAN_WEB_PORT:-8083}:8083"
    environment:
      - DEAN_ENV=${DEAN_ENV:-development}
      - DEAN_SERVICE_API_KEY=${DEAN_SERVICE_API_KEY}
      - INDEXAGENT_URL=http://indexagent:8081
      - AIRFLOW_URL=http://airflow-service:8080
      - EVOLUTION_API_URL=http://agent-evolution:8090
      - DATABASE_URL=${DEAN_DATABASE_URL}
    depends_on:
      - postgres
      - redis
    volumes:
      - ./DEAN/configs:/app/configs
      - dean-data:/app/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8082/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    networks:
      - agent-network

  agent-evolution:
    build:
      context: ../infra/modules/agent-evolution
      dockerfile: Dockerfile
      args:
        PYTHON_VERSION: "3.11"
    container_name: agent-evolution
    platform: "${DOCKER_DEFAULT_PLATFORM}"
    ports:
      - "${AGENT_EVOLUTION_PORT:-8090}:8080"
    env_file:
      - .env
    depends_on:
      - postgres
      - redis
      - indexagent
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    networks:
      - agent-network

  agent-registry:
    image: redis:7-alpine
    container_name: agent-registry
    ports:
      - "${AGENT_REGISTRY_PORT:-6380}:6379"
    volumes:
      - agent-registry-data:/data
      - ./infra/modules/agent-evolution/config/redis.conf:/usr/local/etc/redis/redis.conf
    command: redis-server /usr/local/etc/redis/redis.conf
    restart: unless-stopped
    networks:
      - agent-network

volumes:
  agent-models:
    driver: local
  agent-registry-data:
    driver: local

networks:
  agent-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### 5.2 Kubernetes Architecture (Future)

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: dean-system
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: dean-config
  namespace: dean-system
data:
  token_budget: "100000"
  diversity_threshold: "0.3"
  evolution_generations: "50"
  parallel_agents: "8"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-evolution
  namespace: dean-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-evolution
  template:
    metadata:
      labels:
        app: agent-evolution
    spec:
      containers:
      - name: agent-evolution
        image: dean/agent-evolution:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: dean-secrets
              key: database-url
        resources:
          requests:
            memory: "1Gi"
            cpu: "1000m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
```

## 7. Security Architecture

### 6.1 Network Security

```yaml
# Network isolation configuration
networks:
  agent-network:
    internal: true
  public-network:
    external: true

# Service network assignment
services:
  agent-evolution:
    networks:
      - agent-network
      - public-network  # Only for API exposure
```

### 6.2 Token Security

```python
# API-level token enforcement middleware
# IndexAgent/src/api/middleware/token_limiter.py
from fastapi import Request, HTTPException
from typing import Dict

class TokenLimiterMiddleware:
    def __init__(self, app, token_limits: Dict[str, int]):
        self.app = app
        self.token_limits = token_limits
        
    async def __call__(self, request: Request, call_next):
        agent_id = request.headers.get("X-Agent-ID")
        if agent_id and agent_id in self.token_limits:
            remaining = await self.check_token_budget(agent_id)
            if remaining <= 0:
                raise HTTPException(status_code=429, 
                                  detail="Token budget exhausted")
        
        response = await call_next(request)
        return response
```

### 6.3 Access Control

```yaml
# Environment-based access control
CLAUDE_API_KEY: ${CLAUDE_API_KEY}  # From secure vault
GITHUB_TOKEN: ${GITHUB_TOKEN}      # For PR creation
AGENT_EVOLUTION_API_KEY: ${AGENT_EVOLUTION_API_KEY}  # Internal auth

# Volume mount restrictions
volumes:
  - ../repos:/repos:ro  # Read-only access to repositories
  - ./worktrees:/worktrees:rw  # Read-write for agent worktrees
```

## 8. Scalability Architecture

### 7.1 Horizontal Scaling

#### Agent Scaling
```yaml
# Docker Compose scaling
docker-compose up -d --scale agent-evolution=3

# Environment variable control
AGENT_MAX_CONCURRENT: 16  # Maximum parallel agents
PARALLEL_WORKERS: 8       # Worker threads per service
```

#### Service Scaling
- **Load Balancer**: Nginx reverse proxy for API distribution
- **Database**: PostgreSQL read replicas for query distribution
- **Cache**: Redis Cluster for distributed caching

### 7.2 Resource Management

```python
# Resource allocation per agent
class ResourceManager:
    def __init__(self):
        self.cpu_per_agent = float(os.getenv("AGENT_CPU_LIMIT", "0.5"))
        self.memory_per_agent = os.getenv("AGENT_MEMORY_LIMIT", "512Mi")
        
    def allocate_resources(self, agent_count: int):
        total_cpu = multiprocessing.cpu_count()
        max_agents = int(total_cpu / self.cpu_per_agent)
        
        if agent_count > max_agents:
            raise ResourceError(f"Insufficient CPU for {agent_count} agents")
        
        return {
            "cpu_limit": self.cpu_per_agent,
            "memory_limit": self.memory_per_agent
        }
```

## 9. Monitoring Architecture

### 8.1 Metrics Collection

```python
# Prometheus metrics endpoint
# infra/modules/agent-evolution/src/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info
from prometheus_client.exposition import make_asgi_app

# Core metrics
agents_created = Counter('dean_agents_created_total', 'Total agents created')
evolution_duration = Histogram('dean_evolution_duration_seconds', 
                             'Evolution cycle duration',
                             buckets=[1, 5, 10, 30, 60, 120, 300])
active_agents = Gauge('dean_active_agents', 'Currently active agents')
diversity_score = Gauge('dean_population_diversity', 'Population diversity score')
token_efficiency = Histogram('dean_token_efficiency', 
                           'Tokens per value generated',
                           buckets=[0.1, 0.5, 1, 2, 5, 10])

# Mount metrics endpoint
app.mount("/metrics", make_asgi_app())
```

### 8.2 Logging Architecture

```python
# Structured logging configuration
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()
```

### 8.3 Distributed Tracing

```yaml
# OpenTelemetry integration
OTEL_EXPORTER_OTLP_ENDPOINT: "http://jaeger:4317"
OTEL_SERVICE_NAME: "dean-agent-evolution"
OTEL_TRACES_EXPORTER: "otlp"
```

## 10. Integration Points

### 9.1 Repository Integration Matrix

| Component | Repository | Location | Purpose |
|-----------|------------|----------|---------|
| Agent Core | IndexAgent | `indexagent/agents/` | Core agent logic |
| Evolution | IndexAgent | `indexagent/agents/evolution/` | Genetic algorithms |
| Patterns | IndexAgent | `indexagent/agents/patterns/` | Pattern detection |
| DAGs | airflow-hub | `dags/agent_evolution/` | Orchestration |
| Operators | airflow-hub | `plugins/agent_evolution/` | Custom operators |
| Docker | infra | `modules/agent-evolution/` | Service definition |
| Config | infra | `modules/agent-evolution/config/` | YAML configs |
| Scripts | infra | `modules/agent-evolution/scripts/` | Management |

### 9.2 API Integration Points

```python
# Service discovery and communication
SERVICE_URLS = {
    "indexagent": "http://indexagent:8080/api/v1",
    "airflow": "http://airflow-service:8080/api/v1",
    "agent_evolution": "http://agent-evolution:8080/api/v1",
    "prometheus": "http://prometheus:9090/api/v1",
    "redis": "redis://agent-registry:6379"
}
```

## 11. DEAN Integration Architecture

### 10.1 Service Integration Flow

The DEAN orchestration layer acts as the central coordinator for all service interactions:

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interfaces                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   CLI Tool   │  │ Web Dashboard│  │   REST API   │          │
│  │  (dean-cli)  │  │  (Port 8083) │  │  (Port 8082) │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         └─────────────────┼─────────────────┘                  │
│                           ▼                                     │
│                 ┌───────────────────┐                           │
│                 │ DEAN Orchestration│                           │
│                 │     Server        │                           │
│                 │   (Port 8082)     │                           │
│                 └─────────┬─────────┘                           │
│                           │                                     │
│         ┌─────────────────┼─────────────────┐                  │
│         ▼                 ▼                 ▼                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  IndexAgent  │  │   Airflow    │  │  Evolution   │          │
│  │  (Port 8081) │  │  (Port 8080) │  │    API       │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### 10.2 Authentication and Authorization Flow

```python
# DEAN authentication flow
1. User → DEAN API (login request)
2. DEAN → Auth Service (validate credentials)
3. Auth Service → DEAN (JWT token)
4. DEAN → User (authenticated token)
5. User → DEAN API (request with token)
6. DEAN → Service (forwarded request with service token)
7. Service → DEAN (response)
8. DEAN → User (aggregated response)
```

### 10.3 Workflow Orchestration

DEAN coordinates complex multi-service workflows:

```yaml
# Example: Evolution Trial Workflow
workflow: evolution_trial
steps:
  - name: initialize_population
    service: indexagent
    endpoint: /api/v1/agents/initialize
    params:
      population_size: 10
      diversity_threshold: 0.3
  
  - name: start_evolution
    service: evolution_api
    endpoint: /api/v1/evolution/start
    params:
      generations: 50
      token_budget: 100000
  
  - name: trigger_airflow_dag
    service: airflow
    endpoint: /api/v1/dags/agent_evolution_trial/dagRuns
    params:
      conf:
        trial_id: "{{ steps.start_evolution.trial_id }}"
  
  - name: monitor_progress
    service: dean
    type: websocket
    endpoint: /ws/evolution/{{ steps.start_evolution.trial_id }}
```

### 10.4 Service Registry and Discovery

```python
# DEAN maintains service registry
SERVICE_REGISTRY = {
    "indexagent": {
        "url": "http://indexagent:8081",
        "health": "/health",
        "auth": "bearer",
        "timeout": 30
    },
    "airflow": {
        "url": "http://airflow-service:8080",
        "health": "/api/v1/health",
        "auth": "basic",
        "timeout": 60
    },
    "evolution_api": {
        "url": "http://agent-evolution:8090",
        "health": "/health",
        "auth": "bearer",
        "timeout": 30
    }
}
```

## 12. Disaster Recovery Architecture

### 11.1 Backup Strategy

```bash
# Automated backup script
# infra/modules/agent-evolution/scripts/backup-dean.sh
#!/bin/bash
set -e

# Backup PostgreSQL
pg_dump $DATABASE_URL > backups/dean_$(date +%Y%m%d_%H%M%S).sql

# Backup Redis
redis-cli --rdb backups/redis_$(date +%Y%m%d_%H%M%S).rdb

# Backup patterns
tar -czf backups/patterns_$(date +%Y%m%d_%H%M%S).tar.gz /app/data/patterns

# Upload to S3
aws s3 sync backups/ s3://dean-backups/$(date +%Y%m%d)/
```

### 11.2 Recovery Procedures

1. **Service Recovery**: Docker Compose restart policies ensure automatic recovery
2. **Data Recovery**: Point-in-time recovery from PostgreSQL backups
3. **Pattern Recovery**: Redis cache rebuild from PostgreSQL source of truth
4. **Worktree Recovery**: Automatic cleanup and recreation on agent restart