# Architectural Design Document (Final)

## 1. Architecture Overview

The DEAN system follows a distributed, microservices-inspired architecture with autonomous agents operating in parallel across three integrated repositories. The architecture fundamentally incorporates economic constraints, diversity maintenance, and emergent behavior capture as core architectural principles implemented through specific infrastructure components.

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│              Apache Airflow 3.0.0 Orchestration Layer               │
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

### 2.2 Repository Integration Architecture

```
┌──────────────────────┐     ┌──────────────────────┐   ┌─────────────────────┐
│      infra/          │     │   airflow-hub/       │   │     IndexAgent/     │
├──────────────────────┤     ├──────────────────────    ├─────────────────────┤
│ modules/             │     │ dags/                │   │ indexagent/         |
│ └─ agent-evolution/  │     │ └─ agent_evolution/  │   │ └─ agents/          │
│     ├─ docker/       │     │     ├─ *.yaml        │   │     ├─ evolution/   │
│     ├─ config/       │────▶│     └─ *.py          │──▶│     ├─ patterns/    │
│     └─ scripts/      │     │                      │   │     └─ economy/     │
│                      │     │ plugins/             │   │                     │
│ docker-compose.yml   │     │ └─ agent_evolution/  │   │ src/api/agents.py   │
└──────────────────────┘     └──────────────────────┘   └─────────────────────┘
         ▲                            ▲                            ▲
         │                            │                            │
         └────────────────────────────┴────────────────────────────┘
                          Central Configuration via .env
```

### 2.3 Service Communication Architecture

```yaml
# Service endpoints for inter-service communication
INDEXAGENT_API: "http://indexagent:8080/api/v1"
AIRFLOW_API: "http://airflow-service:8080/api/v1"
AGENT_EVOLUTION_API: "http://agent-evolution:8080/api/v1"
AGENT_REGISTRY: "redis://agent-registry:6379"
DATABASE_URL: "postgresql://postgres:password@postgres:5432/agent_evolution"
```

## 3. Component Architecture

### 3.1 Agent Runtime Layer

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

## 4. Data Architecture

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

## 5. Deployment Architecture

### 5.1 Docker Compose Architecture

```yaml
# infra/docker-compose.yml additions
version: "3.8"

services:
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

## 6. Security Architecture

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

## 7. Scalability Architecture

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

## 8. Monitoring Architecture

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

## 9. Integration Points

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

## 10. Disaster Recovery Architecture

### 10.1 Backup Strategy

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

### 10.2 Recovery Procedures

1. **Service Recovery**: Docker Compose restart policies ensure automatic recovery
2. **Data Recovery**: Point-in-time recovery from PostgreSQL backups
3. **Pattern Recovery**: Redis cache rebuild from PostgreSQL source of truth
4. **Worktree Recovery**: Automatic cleanup and recreation on agent restart