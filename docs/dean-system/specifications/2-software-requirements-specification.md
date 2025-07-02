# Software Requirements Specification (Final)

## 1. Introduction

### 1.1 Purpose

This document specifies the requirements for the Distributed Evolutionary Agent Network (DEAN), a self-evolving agent system that leverages parallelized Claude Code CLI agents for automated code maintenance and improvement with built-in economic efficiency, genetic diversity, and emergent behavior capture.

### 1.2 Scope

The system will integrate with the existing infrastructure across four repositories:

- **DEAN**: Primary orchestration layer providing unified control, authentication, and monitoring via `DEAN/src/`
- **infra**: Infrastructure orchestration via `infra/modules/agent-evolution/`
- **airflow-hub**: Workflow management via `airflow-hub/dags/agent_evolution/`
- **IndexAgent**: Core logic via `IndexAgent/indexagent/agents/`

The system will enable parallel execution of multiple AI agents with strict economic constraints through DEAN orchestration, automatically optimize agent behavior through recursive evolution while maintaining genetic diversity, provide scalable code maintenance across multiple repositories with unified monitoring, capture and leverage emergent behaviors for continuous improvement via coordinated services, and maintain a queryable knowledge repository for meta-learning accessible through DEAN's unified API.

## 2. Functional Requirements

### 2.1 Agent Management

- FR-001: The system SHALL create isolated git worktrees for each agent in the `IndexAgent/indexagent/agents/worktrees/` directory
- FR-002: The system SHALL support spawning N parallel agents (configurable via `AGENT_MAX_CONCURRENT` environment variable)
- FR-003: Agents SHALL be able to create child agents for specialized tasks through the `IndexAgent/indexagent/agents/evolution/genetic_algorithm.py` module
- FR-004: The system SHALL track agent lineage and performance metrics in PostgreSQL `agent_evolution.agents` table
- FR-005: The system SHALL enforce token budget limits at the API level through `IndexAgent/indexagent/agents/economy/token_manager.py`
- FR-006: The system SHALL calculate and track value-per-token metrics in the `agent_evolution.performance_metrics` table
- FR-007: The system SHALL automatically terminate agents that exceed token budgets via Airflow task monitoring

### 2.2 Evolution Capabilities

- FR-008: Agents SHALL improve their own prompts using DSPy optimization integrated in `IndexAgent/indexagent/agents/evolution/dspy_optimizer.py`
- FR-009: The system SHALL implement cellular automata rules in `IndexAgent/indexagent/agents/evolution/cellular_automata.py`
- FR-010: Successful patterns SHALL propagate across the agent population via Redis agent registry at `redis://agent-registry:6379`
- FR-011: The system SHALL maintain evolutionary history in PostgreSQL `agent_evolution.evolution_history` table
- FR-012: The system SHALL enforce minimum genetic diversity thresholds configured in `infra/modules/agent-evolution/config/evolution.yaml`
- FR-013: The system SHALL detect and prevent premature convergence using `IndexAgent/indexagent/agents/evolution/diversity_monitor.py`
- FR-014: The system SHALL implement controlled mutation injection via `IndexAgent/indexagent/agents/evolution/mutation_strategies.py`
- FR-015: The system SHALL identify and catalog emergent behaviors in `agent_evolution.discovered_patterns` table

### 2.3 Economic Management

- FR-016: The system SHALL implement global token budget management via environment variable `AGENT_EVOLUTION_MAX_POPULATION`
- FR-017: The system SHALL allocate tokens based on historical agent efficiency stored in `agent_evolution.performance_metrics`
- FR-018: The system SHALL track real-time token consumption patterns through Prometheus metrics exposed at `/metrics`
- FR-019: The system SHALL implement budget decay for long-running agents configured in `infra/modules/agent-evolution/config/agents.yaml`
- FR-020: The system SHALL prioritize high-value-per-token agents for reproduction in `IndexAgent/indexagent/agents/evolution/selection_strategies.py`
- FR-021: The system SHALL provide economic performance alerts via Airflow alert configurations

### 2.4 Pattern Detection and Meta-Learning

- FR-022: The system SHALL detect novel agent strategies using `IndexAgent/indexagent/agents/patterns/detector.py`
- FR-023: The system SHALL distinguish between beneficial innovations and metric gaming via `IndexAgent/indexagent/agents/patterns/classifier.py`
- FR-024: The system SHALL maintain a searchable pattern library in PostgreSQL with indexes on effectiveness and reuse
- FR-025: The system SHALL enable pattern reuse across agent generations through `IndexAgent/indexagent/agents/patterns/library.py`
- FR-026: The system SHALL support cross-domain pattern import via `infra/modules/agent-evolution/scripts/import-patterns.py`
- FR-027: The system SHALL track pattern effectiveness in `agent_evolution.discovered_patterns` table

### 2.5 DEAN Orchestration Requirements

- FR-028: The DEAN orchestration server SHALL provide a unified API endpoint at `http://dean-server:8082/api/v1` for all system operations
- FR-029: DEAN SHALL authenticate all user requests using JWT tokens with configurable expiration via `DEAN_TOKEN_EXPIRY`
- FR-030: DEAN SHALL provide service-to-service authentication tokens for secure inter-service communication
- FR-031: DEAN SHALL coordinate evolution trials across IndexAgent, Airflow, and Evolution API services
- FR-032: DEAN SHALL aggregate health status from all registered services every 30 seconds
- FR-033: DEAN SHALL provide real-time evolution monitoring via WebSocket connections at `/ws/evolution/{trial_id}`
- FR-034: DEAN SHALL maintain a service registry with automatic failover capabilities
- FR-035: DEAN SHALL execute multi-service workflows with transactional semantics
- FR-036: DEAN SHALL provide a CLI interface (`dean-cli`) for system control and monitoring
- FR-037: DEAN SHALL offer a web dashboard on port 8083 for visual system monitoring
- FR-038: DEAN SHALL implement circuit breakers for all service connections with configurable thresholds
- FR-039: DEAN SHALL log all orchestration decisions to an audit trail in `dean.orchestration_log`
- FR-040: DEAN SHALL support batch operations for managing multiple evolution trials
- FR-041: DEAN SHALL enforce global resource limits across all services
- FR-042: DEAN SHALL provide service discovery mechanisms for dynamic service registration

### 2.6 Integration Requirements

- FR-043: The system SHALL integrate with Apache Airflow 3.0.0 for task orchestration via custom operators in `airflow-hub/plugins/agent_evolution/`
- FR-044: The system SHALL use Claude Code CLI for code modifications integrated through Docker service
- FR-045: The system SHALL create pull requests for successful improvements via GitHub API integration
- FR-046: The system SHALL support Docker Compose deployment via `infra/docker-compose.yml` modifications
- FR-047: The system SHALL provide queryable access to the knowledge repository via DEAN's unified API
- FR-048: All service integrations SHALL authenticate through DEAN's authentication layer
- FR-049: Service endpoints SHALL be configurable via DEAN's service registry

## 3. Non-Functional Requirements

### 3.1 Performance

- NFR-001: The system SHALL support at least 8 parallel agents as configured by `AGENT_MAX_CONCURRENT`
- NFR-002: Agent creation SHALL complete within 10 seconds using pooled git worktrees
- NFR-003: The system SHALL scale linearly with available CPU cores up to 16 cores
- NFR-004: The system SHALL optimize for token efficiency maintaining below 1000 tokens per meaningful change
- NFR-005: Pattern detection SHALL complete within 5 seconds using indexed PostgreSQL queries
- NFR-006: DEAN orchestration API SHALL respond to requests within 200ms for cached operations
- NFR-007: DEAN SHALL handle at least 100 concurrent WebSocket connections for monitoring
- NFR-008: Service health checks SHALL complete within 5 seconds across all registered services

### 3.2 Security

- NFR-009: Agents SHALL only modify code within assigned worktrees enforced by Docker volume mounts
- NFR-010: The system SHALL enforce token budget limits through API middleware and database constraints
- NFR-011: All agent actions SHALL be auditable via PostgreSQL `agent_evolution.audit_log` table
- NFR-012: The system SHALL prevent agents from modifying safety constraints through immutable configuration
- NFR-013: The system SHALL isolate agent execution environments using Docker network isolation
- NFR-014: DEAN SHALL enforce authentication on all API endpoints using JWT tokens
- NFR-015: DEAN SHALL implement role-based access control for administrative operations
- NFR-016: Service-to-service communication SHALL use time-limited authentication tokens
- NFR-017: DEAN SHALL log all authentication attempts and authorization decisions

### 3.3 Reliability

- NFR-018: Failed agents SHALL not affect other running agents through container isolation
- NFR-019: The system SHALL automatically clean up orphaned worktrees via `infra/modules/agent-evolution/scripts/cleanup-worktrees.sh`
- NFR-020: The system SHALL recover from partial failures using Airflow retry mechanisms
- NFR-021: The system SHALL maintain diversity even under failure conditions through forced mutation injection
- NFR-022: The system SHALL preserve pattern library integrity through database transactions and backups
- NFR-023: DEAN SHALL implement circuit breakers to prevent cascading service failures
- NFR-024: DEAN SHALL maintain service operation logs for post-mortem analysis
- NFR-025: DEAN SHALL support graceful degradation when individual services are unavailable

### 3.4 Observability

- NFR-026: The system SHALL provide real-time metrics on token consumption via Prometheus exposed through DEAN
- NFR-027: The system SHALL track diversity variance continuously in unified DEAN dashboards
- NFR-028: The system SHALL log all pattern discoveries to structured logs in JSON format
- NFR-029: The system SHALL enable historical trend analysis through PostgreSQL time-series queries
- NFR-030: The system SHALL support complex analytical queries via indexed database views
- NFR-031: DEAN SHALL provide unified logging aggregation across all services
- NFR-032: DEAN SHALL expose consolidated metrics endpoint for all system components
- NFR-033: DEAN SHALL support distributed tracing for multi-service operations

## 4. Technical Specifications

### 4.1 Technology Stack

- **Python**: 3.11
- **Apache Airflow**: 3.0.0
- **PostgreSQL**: Latest stable with agent_evolution schema
- **Redis**: 7-alpine for agent registry
- **Docker**: Compose version 3.8
- **FastAPI**: 0.100.0+ for API endpoints
- **DSPy**: Latest version for prompt optimization
- **Claude Code CLI**: Via Docker container

### 4.2 Repository Structure

```
IndexAgent/
├── indexagent/
│   └── agents/                      # DEAN core logic
│       ├── base_agent.py           # Base agent class
│       ├── evolution/              # Evolution algorithms
│       ├── registry/               # Agent registry interface
│       ├── patterns/               # Pattern detection
│       ├── economy/                # Token management
│       └── interfaces/             # External integrations

airflow-hub/
├── dags/
│   └── agent_evolution/            # DEAN DAGs
└── plugins/
    └── agent_evolution/            # Custom operators

infra/
└── modules/
    └── agent-evolution/            # Infrastructure config
        ├── config/                 # YAML configurations
        ├── scripts/                # Management scripts
        └── docker/                 # Container definitions
```

### 4.3 Database Schema

The system uses PostgreSQL with a dedicated `agent_evolution` schema containing tables for agents, evolution_history, performance_metrics, discovered_patterns, and strategy_evolution, all with appropriate indexes for query performance.