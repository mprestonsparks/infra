# Software Design Document (Final)

## 1. System Overview

The Distributed Evolutionary Agent Network (DEAN) implements a fractal architecture where agents recursively improve themselves while maintaining isolated execution environments through git worktrees. The system is distributed across four repositories with the DEAN repository serving as the primary orchestration layer. The architecture incorporates economic constraints, genetic diversity maintenance, and emergent behavior capture as fundamental design elements, all coordinated through the DEAN orchestration server.

## 2. Design Principles

### 2.1 Cellular Automata Rules

The system implements five core rules with diversity enforcement in `IndexAgent/indexagent/agents/evolution/cellular_automata.py`:

1. **Rule 110**: Create improved neighbors when detecting imperfections (with mutation variance)
2. **Rule 30**: Fork into parallel worktrees when bottlenecked (maintaining strategy diversity)
3. **Rule 90**: Abstract patterns into reusable components (with pattern cataloging)
4. **Rule 184**: Learn from higher-performing neighbors (weighted by token efficiency)
5. **Rule 1**: Recurse to higher abstraction levels when optimal (preserving diversity)

### 2.2 Economic Design Principles

- **Token Efficiency First**: All operations optimize for value-per-token tracked in PostgreSQL
- **Budget Decay**: Long-running agents receive diminishing token allocations per `evolution.yaml`
- **Performance-Based Allocation**: High-efficiency agents receive larger budgets via dynamic allocation
- **Real-time Cost Tracking**: Continuous monitoring via Prometheus metrics at `/metrics`

### 2.3 Evolutionary Design Principles

- **Genetic Diversity**: Mandatory variance maintained above 0.3 threshold
- **Convergence Prevention**: Active detection and disruption of monocultures
- **Innovation Capture**: Systematic cataloging in `agent_evolution.discovered_patterns`
- **Meta-Learning**: Pattern extraction and reapplication across generations

## 3. Component Design

### 3.1 Core Components

#### FractalAgent Class
Location: `IndexAgent/indexagent/agents/base_agent.py`

```python
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from .economy import TokenBudget
from .evolution import AgentGenome

class FractalAgent(BaseModel):
    """Base agent class with economic and diversity constraints"""
    
    genome: AgentGenome
    level: int
    parent_id: Optional[str] = None
    children: List[str] = []
    token_budget: TokenBudget
    diversity_score: float = 0.0
    emergent_patterns: List[str] = []
    worktree_path: Optional[str] = None
    
    async def evolve(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Evolution with economic constraints and diversity maintenance"""
        # Check token budget before any operation
        if not await self.token_budget.can_afford(estimated_cost=100):
            return {"status": "budget_exhausted"}
            
        # Maintain diversity from population
        await self.maintain_diversity(environment['population'])
        
        # Execute evolution cycle
        result = await self._execute_evolution(environment)
        
        # Track emergent patterns
        patterns = await self._detect_patterns(result)
        self.emergent_patterns.extend(patterns)
        
        return result
    
    async def track_token_efficiency(self) -> float:
        """Calculate value generated per token consumed"""
        value = await self._calculate_value()
        tokens = self.token_budget.consumed
        return value / tokens if tokens > 0 else 0.0
```

#### TokenEconomyManager Class
Location: `IndexAgent/indexagent/agents/economy/token_manager.py`

```python
from typing import Dict
import asyncio
from prometheus_client import Counter, Gauge, Histogram

class TokenEconomyManager:
    """Manages global token budget with Prometheus metrics"""
    
    def __init__(self, global_budget: int):
        self.global_budget = global_budget
        self.agent_allocations: Dict[str, int] = {}
        self.efficiency_metrics: Dict[str, float] = {}
        
        # Prometheus metrics
        self.tokens_allocated = Counter('dean_tokens_allocated_total', 
                                      'Total tokens allocated')
        self.tokens_consumed = Counter('dean_tokens_consumed_total',
                                     'Total tokens consumed')
        self.efficiency_gauge = Gauge('dean_agent_efficiency',
                                    'Agent efficiency score',
                                    ['agent_id'])
        
    async def allocate_tokens(self, agent_id: str, base_amount: int) -> int:
        """Dynamic allocation based on historical efficiency"""
        efficiency = self.efficiency_metrics.get(agent_id, 1.0)
        
        # Apply performance multiplier
        allocation = int(base_amount * efficiency)
        
        # Apply global budget constraints
        remaining = self.global_budget - sum(self.agent_allocations.values())
        allocation = min(allocation, remaining)
        
        self.agent_allocations[agent_id] = allocation
        self.tokens_allocated.inc(allocation)
        
        return allocation
```

#### GeneticDiversityManager Class
Location: `IndexAgent/indexagent/agents/evolution/diversity_manager.py`

```python
from typing import List
import numpy as np
from .mutation_strategies import MutationEngine
from .pattern_import import CrossDomainImporter

class GeneticDiversityManager:
    """Maintains population diversity through active intervention"""
    
    def __init__(self, min_diversity_threshold: float = 0.3):
        self.min_diversity_threshold = min_diversity_threshold
        self.mutation_engine = MutationEngine()
        self.pattern_importer = CrossDomainImporter()
        
    async def enforce_diversity(self, population: List['FractalAgent']):
        """Detect convergence and inject mutations"""
        variance = await self._calculate_population_variance(population)
        
        if variance < self.min_diversity_threshold:
            # Inject mutations
            mutation_targets = self._select_mutation_targets(population)
            for agent in mutation_targets:
                mutated = await self.mutation_engine.mutate(agent)
                await self._replace_agent(agent, mutated)
            
            # Import foreign patterns
            if variance < self.min_diversity_threshold * 0.5:
                foreign_patterns = await self.pattern_importer.import_patterns()
                await self._inject_patterns(population, foreign_patterns)
```

#### EmergentBehaviorMonitor Class
Location: `IndexAgent/indexagent/agents/patterns/monitor.py`

```python
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from .detector import PatternDetector
from .classifier import BehaviorClassifier

class EmergentBehaviorMonitor:
    """Captures and catalogs novel agent strategies"""
    
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        self.pattern_detector = PatternDetector()
        self.classifier = BehaviorClassifier()
        
    async def detect_novel_behaviors(self, agent: 'FractalAgent') -> List[Pattern]:
        """Identify strategies not explicitly programmed"""
        behaviors = await self.pattern_detector.analyze_agent_actions(agent)
        novel_patterns = []
        
        for behavior in behaviors:
            if not await self._is_known_pattern(behavior):
                classification = await self.classifier.classify(behavior)
                if classification.is_beneficial:
                    await self._store_pattern(behavior, classification)
                    novel_patterns.append(behavior)
        
        return novel_patterns
```

#### GitWorktreeManager Class
Location: `IndexAgent/indexagent/agents/worktree_manager.py`

```python
import asyncio
from pathlib import Path
from typing import Optional

class GitWorktreeManager:
    """Manages isolated git worktrees with resource limits"""
    
    def __init__(self, base_path: Path = Path("/app/worktrees")):
        self.base_path = base_path
        self.active_worktrees: Dict[str, Path] = {}
        
    async def create_worktree(self, branch_name: str, agent_id: str, 
                            token_limit: int) -> Path:
        """Creates isolated worktree for agent with resource limits"""
        worktree_path = self.base_path / agent_id
        
        # Create worktree
        await asyncio.create_subprocess_exec(
            'git', 'worktree', 'add', str(worktree_path), branch_name,
            cwd=str(self.base_path.parent)
        )
        
        # Set resource limits via Docker constraints
        await self._apply_resource_limits(worktree_path, token_limit)
        
        self.active_worktrees[agent_id] = worktree_path
        return worktree_path
```

### 3.2 Knowledge Repository Design

Database: PostgreSQL with `agent_evolution` schema
Location: Configured via `AGENT_EVOLUTION_DATABASE_URL` environment variable

```sql
-- Core tables for queryable knowledge repository
CREATE SCHEMA IF NOT EXISTS agent_evolution;

CREATE TABLE agent_evolution.agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    generation INTEGER DEFAULT 0,
    fitness_score DECIMAL(10, 6) DEFAULT 0.0,
    parent_ids UUID[],
    genome_hash TEXT NOT NULL,
    token_efficiency FLOAT,
    diversity_score FLOAT,
    config JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE agent_evolution.discovered_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES agent_evolution.agents(id),
    pattern_type TEXT NOT NULL,
    pattern_content JSONB NOT NULL,
    effectiveness_score FLOAT,
    token_efficiency_delta FLOAT,
    reuse_count INTEGER DEFAULT 0,
    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE agent_evolution.strategy_evolution (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_hash TEXT NOT NULL,
    parent_strategy_hash TEXT,
    mutation_type TEXT,
    performance_delta FLOAT,
    token_efficiency_delta FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance indexes
CREATE INDEX idx_agents_fitness ON agent_evolution.agents(fitness_score DESC);
CREATE INDEX idx_patterns_effectiveness ON agent_evolution.discovered_patterns(effectiveness_score DESC);
CREATE INDEX idx_patterns_reuse ON agent_evolution.discovered_patterns(reuse_count DESC);
CREATE INDEX idx_evolution_performance ON agent_evolution.strategy_evolution(performance_delta DESC);
```

### 3.3 DEAN Orchestration Components

The DEAN repository provides the primary orchestration layer that coordinates all services:

#### OrchestrationServer Class
Location: `DEAN/src/dean_orchestration/server.py`

```python
from fastapi import FastAPI, WebSocket
from typing import Dict, List, Optional
from ..auth import AuthManager
from ..integration import ServiceRegistry

class OrchestrationServer:
    """Main orchestration server coordinating all DEAN services"""
    
    def __init__(self):
        self.app = FastAPI(title="DEAN Orchestration API")
        self.auth_manager = AuthManager()
        self.service_registry = ServiceRegistry()
        self.active_workflows: Dict[str, WorkflowState] = {}
        
    async def coordinate_evolution_trial(self, trial_config: EvolutionConfig) -> str:
        """Orchestrates complete evolution trial across services"""
        trial_id = str(uuid.uuid4())
        
        # Step 1: Initialize population in IndexAgent
        population = await self.service_registry.indexagent.initialize_population(
            size=trial_config.population_size,
            diversity_threshold=trial_config.diversity_threshold
        )
        
        # Step 2: Start evolution in Evolution API
        evolution_job = await self.service_registry.evolution_api.start_evolution(
            population_ids=population.agent_ids,
            generations=trial_config.generations,
            token_budget=trial_config.token_budget
        )
        
        # Step 3: Trigger Airflow DAG for monitoring
        dag_run = await self.service_registry.airflow.trigger_dag(
            dag_id="agent_evolution_trial",
            conf={"trial_id": trial_id, "job_id": evolution_job.id}
        )
        
        return trial_id
```

#### ServiceRegistry Class
Location: `DEAN/src/integration/registry.py`

```python
class ServiceRegistry:
    """Maintains registry of all DEAN services with health monitoring"""
    
    def __init__(self):
        self.services = {
            "indexagent": ServiceClient("http://indexagent:8081", "bearer"),
            "airflow": ServiceClient("http://airflow-service:8080", "basic"),
            "evolution_api": ServiceClient("http://agent-evolution:8090", "bearer")
        }
        self._health_check_interval = 30  # seconds
        
    async def health_check_all(self) -> Dict[str, bool]:
        """Performs health checks on all registered services"""
        results = {}
        for name, client in self.services.items():
            results[name] = await client.health_check()
        return results
```

#### AuthManager Class
Location: `DEAN/src/auth/manager.py`

```python
from jose import jwt, JWTError
from passlib.context import CryptContext

class AuthManager:
    """Unified authentication manager for all DEAN services"""
    
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.secret_key = os.getenv("DEAN_SECRET_KEY")
        self.algorithm = "HS256"
        
    def create_service_token(self, service_name: str) -> str:
        """Creates JWT token for service-to-service authentication"""
        payload = {
            "sub": service_name,
            "type": "service",
            "exp": datetime.utcnow() + timedelta(hours=24)
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
    async def authenticate_request(self, token: str) -> Optional[Dict]:
        """Validates incoming request authentication"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError:
            return None
```

#### WorkflowCoordinator Class
Location: `DEAN/src/orchestration/coordinator.py`

```python
class WorkflowCoordinator:
    """Coordinates complex multi-service workflows"""
    
    def __init__(self, service_registry: ServiceRegistry):
        self.registry = service_registry
        self.workflow_definitions = self._load_workflow_definitions()
        
    async def execute_workflow(self, workflow_name: str, params: Dict) -> WorkflowResult:
        """Executes a defined workflow across multiple services"""
        workflow = self.workflow_definitions.get(workflow_name)
        if not workflow:
            raise ValueError(f"Unknown workflow: {workflow_name}")
            
        context = WorkflowContext(params)
        
        for step in workflow.steps:
            service = self.registry.services.get(step.service)
            result = await service.call(step.endpoint, step.params, context)
            context.store_result(step.name, result)
            
            if step.wait_for_completion:
                await self._wait_for_completion(service, result)
                
        return WorkflowResult(context)
```

#### CLI Interface
Location: `DEAN/src/interfaces/cli.py`

```python
import click
from ..orchestration import OrchestrationClient

@click.group()
def dean_cli():
    """DEAN Command Line Interface for system orchestration"""
    pass

@dean_cli.command()
@click.option('--population-size', default=10, help='Initial agent population size')
@click.option('--generations', default=50, help='Number of evolution generations')
@click.option('--token-budget', default=100000, help='Total token budget for trial')
def evolution_start(population_size: int, generations: int, token_budget: int):
    """Start a new evolution trial"""
    client = OrchestrationClient()
    trial_id = client.start_evolution_trial(
        population_size=population_size,
        generations=generations,
        token_budget=token_budget
    )
    click.echo(f"Evolution trial started: {trial_id}")
    
@dean_cli.command()
def status():
    """Check system and service status"""
    client = OrchestrationClient()
    health = client.get_system_health()
    
    for service, is_healthy in health.items():
        status = "✓" if is_healthy else "✗"
        click.echo(f"{status} {service}")
```

## 4. Integration Design

### 4.1 DEAN Orchestration Integration

When deploying the complete DEAN system, the DEAN repository serves as the orchestration layer for coordinating DEAN-specific workflows:

#### Service Communication Flow
```python
# Example: Complete evolution trial orchestrated by DEAN
async def evolution_trial_flow():
    # 1. User initiates via DEAN CLI
    dean_cli evolution start --population-size 10 --generations 50
    
    # 2. DEAN Server receives request
    trial_request = {
        "population_size": 10,
        "generations": 50,
        "token_budget": 100000,
        "diversity_threshold": 0.3
    }
    
    # 3. DEAN orchestrates service calls
    # IndexAgent: Create agent population
    population = await dean.indexagent.create_population(trial_request)
    
    # Evolution API: Start evolution process
    evolution_job = await dean.evolution_api.start_evolution({
        "population_ids": population.agent_ids,
        "config": trial_request
    })
    
    # Airflow: Monitor via DAG
    dag_run = await dean.airflow.trigger_dag(
        "agent_evolution_trial",
        {"job_id": evolution_job.id}
    )
    
    # 4. DEAN provides real-time updates via WebSocket
    async with dean.websocket_connect(f"/ws/evolution/{evolution_job.id}") as ws:
        async for update in ws:
            print(f"Generation {update.generation}: {update.metrics}")
```

#### Authentication Integration
```python
# All services authenticate through DEAN
class ServiceAuthMiddleware:
    """Middleware for service-to-service authentication via DEAN"""
    
    async def __call__(self, request: Request, call_next):
        # Extract DEAN service token
        token = request.headers.get("X-DEAN-Service-Token")
        
        if not token:
            return JSONResponse(status_code=401, content={"error": "Missing service token"})
            
        # Validate with DEAN auth service
        auth_result = await dean_client.validate_service_token(token)
        
        if not auth_result.valid:
            return JSONResponse(status_code=403, content={"error": "Invalid service token"})
            
        # Add service identity to request
        request.state.service_identity = auth_result.service_name
        
        return await call_next(request)
```

#### Configuration Management
```yaml
# DEAN/configs/services.yaml
services:
  indexagent:
    url: "http://indexagent:8081"
    auth_type: "bearer"
    timeout: 30
    health_endpoint: "/health"
    
  airflow:
    url: "http://airflow-service:8080"
    auth_type: "basic"
    timeout: 60
    health_endpoint: "/api/v1/health"
    
  evolution_api:
    url: "http://agent-evolution:8090"
    auth_type: "bearer"
    timeout: 30
    health_endpoint: "/health"

orchestration:
  default_timeout: 300
  retry_policy:
    max_attempts: 3
    backoff_factor: 2
  circuit_breaker:
    failure_threshold: 5
    recovery_timeout: 60
```

### 4.2 DSPy Integration with Meta-Learning

Location: `IndexAgent/indexagent/agents/evolution/dspy_optimizer.py`

```python
import dspy
from typing import Dict, Any

class DEANOptimizer(dspy.Module):
    """DSPy module for agent prompt optimization with meta-learning"""
    
    def __init__(self):
        super().__init__()
        self.prompt_optimizer = dspy.ChainOfThought("optimize_prompt")
        self.strategy_discoverer = dspy.Predict("discover_strategy")
        self.pattern_extractor = dspy.Predict("extract_pattern")
        
    def forward(self, current_prompt: str, performance_metrics: Dict[str, Any]):
        # Optimize existing prompt
        optimized = self.prompt_optimizer(
            prompt=current_prompt,
            metrics=performance_metrics
        )
        
        # Discover new strategies
        new_strategies = self.strategy_discoverer(
            context=optimized.reasoning,
            performance=performance_metrics
        )
        
        # Extract reusable patterns
        patterns = self.pattern_extractor(
            strategies=new_strategies,
            effectiveness=performance_metrics['efficiency']
        )
        
        return optimized, patterns
```

### 4.2 Airflow Integration

DAG Location: `airflow-hub/dags/agent_evolution/agent_lifecycle.yaml`

```yaml
dag:
  dag_id: 'dean_agent_lifecycle'
  description: 'Manages DEAN agent creation, evolution, and retirement'
  schedule: '0 */6 * * *'
  default_args:
    owner: 'agent-evolution'
    retries: 1
    retry_delay: 300
  tags:
    - 'dean'
    - 'agent-evolution'
    - 'automated'

tasks:
  - id: 'check_token_budget'
    type: 'PythonOperator'
    python_callable: 'agent_evolution.operators.budget_operator:check_global_budget'
    
  - id: 'spawn_agents'
    type: 'agent_evolution.operators.AgentSpawnOperator'
    config:
      population_size: "{{ var.value.dean_population_size }}"
      token_limit_per_agent: 4096
      diversity_threshold: 0.3
      
  - id: 'evolve_population'
    type: 'agent_evolution.operators.AgentEvolutionOperator'
    config:
      generations: 10
      mutation_rate: 0.1
      crossover_rate: 0.7
      parallel_workers: 4
```

## 5. Integration Best Practices

### 5.1 Maintaining Loose Coupling

To ensure services remain independently deployable and reusable:

#### Configuration-Based Integration
```python
# Good: Use configuration for service endpoints
class ServiceClient:
    def __init__(self, service_name: str):
        # Get from environment or config file
        self.base_url = os.getenv(f"{service_name.upper()}_URL")
        if not self.base_url:
            raise ValueError(f"No URL configured for {service_name}")

# Bad: Hardcoded service locations
class ServiceClient:
    def __init__(self):
        self.base_url = "http://my-service:8080"  # ❌ Tight coupling
```

#### Optional Service Dependencies
```python
# Good: Services are optional with graceful degradation
class EnhancedWorkflow:
    def __init__(self):
        self.optional_services = {}
        
    def register_service(self, name: str, client: ServiceClient):
        """Register optional service enhancement"""
        try:
            if client.health_check():
                self.optional_services[name] = client
        except:
            logger.info(f"Service {name} not available - continuing without")
    
    def execute(self):
        # Core functionality works without optional services
        result = self.core_logic()
        
        # Enhance with optional services if available
        for name, service in self.optional_services.items():
            try:
                result = service.enhance(result)
            except:
                logger.warning(f"Enhancement from {name} failed")
        
        return result
```

#### Anti-Patterns to Avoid

1. **Cross-Repository Imports**
```python
# Bad: Direct import from another repository
from infra.services.economy import EconomicGovernor  # ❌

# Good: Use API or package
from dean_common import EconomicGovernor  # ✓ Shared package
# Or use REST API
economic_api = ServiceClient("economy")
```

2. **Synchronous Service Dependencies**
```python
# Bad: Service fails if dependency unavailable
def process():
    other_service = connect_to_service()  # Fails if down
    return other_service.required_data()  # ❌

# Good: Async with fallback
async def process():
    try:
        other_service = await try_connect_to_service()
        return await other_service.get_data()
    except ServiceUnavailable:
        return get_cached_or_default_data()  # ✓
```

### 5.2 Service Communication Patterns

#### Event-Driven Architecture
```python
# Services communicate through events, not direct calls
class EventPublisher:
    async def publish(self, event_type: str, data: dict):
        # Publish to message queue or event bus
        await self.queue.send({
            "type": event_type,
            "data": data,
            "timestamp": datetime.utcnow()
        })

# Services subscribe to relevant events
class EventSubscriber:
    async def handle_event(self, event):
        if event["type"] == "agent.evolved":
            await self.process_evolution(event["data"])
```

#### API Gateway Pattern
```python
# DEAN acts as API gateway for external clients
class DEANGateway:
    def route_request(self, path: str, request):
        service = self.identify_service(path)
        if service and service.is_healthy():
            return service.forward(request)
        return self.handle_unavailable_service()
```

## 6. Data Flow

### 5.1 Standard Evolution Flow

1. Airflow triggers agent spawn via `AgentSpawnOperator` with token budget from environment
2. GitWorktreeManager creates isolated environment in `IndexAgent/indexagent/agents/worktrees/`
3. Agent analyzes code and identifies conflicts using IndexAgent API
4. DSPy optimizes prompts for resolution within token constraints
5. Claude Code CLI implements changes via Docker service integration
6. EmergentBehaviorMonitor captures novel strategies to PostgreSQL
7. Successful changes create pull requests via GitHub API
8. Metrics update in PostgreSQL for future evolution and meta-learning

### 6.2 Meta-Learning Flow

1. Pattern analysis identifies successful strategies via SQL queries
2. Strategy extraction creates reusable components in pattern library
3. Component library updates with new patterns in Redis cache
4. Future agents inherit successful patterns through genome initialization
5. Cross-domain pattern transfer maintains diversity via import scripts

## 7. Safety and Constraint Design

### 7.1 Economic Safety

- Hard token limits enforced at API level via FastAPI middleware
- Automatic agent termination on budget exhaustion through Airflow monitoring
- Progressive budget reduction for inefficient agents per `evolution.yaml` config

### 7.2 Behavioral Safety

- Prevent agents from modifying safety constraints through read-only mounts
- Detect and block metric gaming behaviors via pattern classifier
- Maintain audit trail in PostgreSQL `agent_evolution.audit_log` table

### 7.3 Diversity Safety

- Prevent monoculture through forced mutations when variance < 0.3
- Maintain minimum strategy variance via continuous monitoring
- Regular injection of external patterns from pattern library

## 8. API Design

### 8.1 DEAN Orchestration API Endpoints

Base URL: `http://dean-server:8082/api/v1`

```python
# DEAN/src/dean_orchestration/api/routes.py
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict

router = APIRouter(prefix="/orchestration", tags=["orchestration"])

@router.post("/evolution/start")
async def start_evolution_trial(config: EvolutionTrialConfig) -> Dict:
    """Start a new evolution trial across all services"""
    
@router.get("/evolution/{trial_id}/status")
async def get_evolution_status(trial_id: str) -> EvolutionStatus:
    """Get real-time status of evolution trial"""
    
@router.post("/workflow/execute")
async def execute_workflow(workflow: WorkflowRequest) -> Dict:
    """Execute a multi-service workflow"""
    
@router.get("/services/health")
async def get_services_health() -> Dict[str, bool]:
    """Check health status of all registered services"""
    
@router.post("/auth/login")
async def login(credentials: LoginCredentials) -> Dict:
    """Authenticate and receive JWT token"""
    
@router.post("/auth/service-token")
async def create_service_token(service: ServiceTokenRequest) -> Dict:
    """Generate service-to-service authentication token"""
```

### 8.2 Service-Specific API Endpoints

#### IndexAgent API
Base URL: `http://indexagent:8081/api/v1`

```python
# IndexAgent/src/api/agents.py
from fastapi import APIRouter, HTTPException
from typing import List, Dict

router = APIRouter(prefix="/agents", tags=["agents"])

@router.post("/")
async def create_agent(agent_config: AgentConfig) -> Dict:
    """Create new agent with token budget"""
    
@router.get("/{agent_id}")
async def get_agent(agent_id: str) -> Agent:
    """Get agent details including efficiency metrics"""
    
@router.post("/{agent_id}/evolve")
async def evolve_agent(agent_id: str, params: EvolutionParams) -> Dict:
    """Trigger evolution for specific agent"""
    
@router.get("/patterns/discovered")
async def list_discovered_patterns(limit: int = 100) -> List[Pattern]:
    """List recently discovered patterns"""
    
@router.get("/metrics/efficiency")
async def get_efficiency_metrics() -> Dict:
    """Get population-wide efficiency metrics"""
```

### 8.3 WebSocket Support

#### DEAN WebSocket Endpoints
```python
# DEAN/src/dean_orchestration/api/websocket.py
@router.websocket("/ws/evolution/{trial_id}")
async def monitor_evolution_trial(websocket: WebSocket, trial_id: str):
    """Real-time evolution trial monitoring"""
    await websocket.accept()
    
    async for update in evolution_monitor.subscribe(trial_id):
        await websocket.send_json({
            "trial_id": trial_id,
            "generation": update.generation,
            "metrics": update.metrics,
            "best_agents": update.best_agents,
            "diversity_score": update.diversity_score
        })

@router.websocket("/ws/services/health")
async def monitor_service_health(websocket: WebSocket):
    """Real-time service health monitoring"""
    await websocket.accept()
    
    while True:
        health_status = await service_registry.health_check_all()
        await websocket.send_json(health_status)
        await asyncio.sleep(5)
```

#### Service-Specific WebSocket Support
```python
# IndexAgent WebSocket (accessed via DEAN)
@router.websocket("/agents/{agent_id}/monitor")
async def monitor_agent(websocket: WebSocket, agent_id: str):
    """Real-time agent monitoring via WebSocket"""
    await websocket.accept()
    while True:
        metrics = await get_agent_metrics(agent_id)
        await websocket.send_json(metrics)
        await asyncio.sleep(1)
```