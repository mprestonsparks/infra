# Software Requirements Specification (Final)

## 1. Introduction

### 1.1 Purpose

This document specifies the requirements for the Distributed Evolutionary Agent Network (DEAN), a self-evolving agent system that leverages parallelized Claude Code CLI agents for automated code maintenance and improvement with built-in economic efficiency, genetic diversity, and emergent behavior capture.

### 1.2 Scope

The system will integrate with the existing infrastructure across three repositories:

- **infra**: Infrastructure orchestration via `infra/modules/agent-evolution/`
- **airflow-hub**: Workflow management via `airflow-hub/dags/agent_evolution/`
- **IndexAgent**: Core logic via `IndexAgent/indexagent/agents/`

The system will enable parallel execution of multiple AI agents with strict economic constraints, automatically optimize agent behavior through recursive evolution while maintaining genetic diversity, provide scalable code maintenance across multiple repositories, capture and leverage emergent behaviors for continuous improvement, and maintain a queryable knowledge repository for meta-learning.

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

### 2.5 Integration Requirements

- FR-028: The system SHALL integrate with Apache Airflow 3.0.0 for orchestration via custom operators in `airflow-hub/plugins/agent_evolution/`
- FR-029: The system SHALL use Claude Code CLI for code modifications integrated through Docker service
- FR-030: The system SHALL create pull requests for successful improvements via GitHub API integration
- FR-031: The system SHALL support Docker Compose deployment via `infra/docker-compose.yml` modifications
- FR-032: The system SHALL provide queryable access to the knowledge repository via FastAPI endpoints at `http://agent-evolution:8080/api/v1`

## 3. Non-Functional Requirements

### 3.1 Performance

- NFR-001: The system SHALL support at least 8 parallel agents as configured by `AGENT_MAX_CONCURRENT`
- NFR-002: Agent creation SHALL complete within 10 seconds using pooled git worktrees
- NFR-003: The system SHALL scale linearly with available CPU cores up to 16 cores
- NFR-004: The system SHALL optimize for token efficiency maintaining below 1000 tokens per meaningful change
- NFR-005: Pattern detection SHALL complete within 5 seconds using indexed PostgreSQL queries

### 3.2 Security

- NFR-006: Agents SHALL only modify code within assigned worktrees enforced by Docker volume mounts
- NFR-007: The system SHALL enforce token budget limits through API middleware and database constraints
- NFR-008: All agent actions SHALL be auditable via PostgreSQL `agent_evolution.audit_log` table
- NFR-009: The system SHALL prevent agents from modifying safety constraints through immutable configuration
- NFR-010: The system SHALL isolate agent execution environments using Docker network isolation

### 3.3 Reliability

- NFR-011: Failed agents SHALL not affect other running agents through container isolation
- NFR-012: The system SHALL automatically clean up orphaned worktrees via `infra/modules/agent-evolution/scripts/cleanup-worktrees.sh`
- NFR-013: The system SHALL recover from partial failures using Airflow retry mechanisms
- NFR-014: The system SHALL maintain diversity even under failure conditions through forced mutation injection
- NFR-015: The system SHALL preserve pattern library integrity through database transactions and backups

### 3.4 Observability

- NFR-016: The system SHALL provide real-time metrics on token consumption via Prometheus at `http://agent-evolution:8080/metrics`
- NFR-017: The system SHALL track diversity variance continuously in Grafana dashboards
- NFR-018: The system SHALL log all pattern discoveries to structured logs in JSON format
- NFR-019: The system SHALL enable historical trend analysis through PostgreSQL time-series queries
- NFR-020: The system SHALL support complex analytical queries via indexed database views

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