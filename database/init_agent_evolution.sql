-- DEAN System Database Initialization Script
-- Creates the agent_evolution database and schema per Data Architecture Section 4.1
-- REAL IMPLEMENTATION - NO MOCKS

-- Create database if it doesn't exist (run as superuser)
-- Note: This needs to be run separately as CREATE DATABASE cannot be executed in a transaction block
-- CREATE DATABASE agent_evolution;

-- Connect to agent_evolution database before running the rest
\c agent_evolution;

-- Create schema
CREATE SCHEMA IF NOT EXISTS agent_evolution;

-- Set search path
SET search_path TO agent_evolution, public;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create agents table per FR-004: Track agent lineage and performance metrics
CREATE TABLE IF NOT EXISTS agent_evolution.agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    goal TEXT NOT NULL,
    
    -- Lineage tracking
    parent_ids UUID[] DEFAULT '{}',
    generation INTEGER NOT NULL DEFAULT 0,
    
    -- Economic constraints per FR-005
    token_budget INTEGER NOT NULL CHECK (token_budget >= 0),
    token_consumed INTEGER NOT NULL DEFAULT 0 CHECK (token_consumed >= 0),
    token_efficiency FLOAT DEFAULT 0.5 CHECK (token_efficiency >= 0 AND token_efficiency <= 2),
    
    -- Agent characteristics
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    worktree_path VARCHAR(500),
    specialized_domain VARCHAR(100),
    
    -- Diversity and evolution tracking per FR-012
    diversity_weight FLOAT NOT NULL DEFAULT 0.5 CHECK (diversity_weight >= 0 AND diversity_weight <= 1),
    diversity_score FLOAT NOT NULL DEFAULT 0.5 CHECK (diversity_score >= 0 AND diversity_score <= 1),
    fitness_score FLOAT NOT NULL DEFAULT 0.0,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    terminated_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for agents table
CREATE INDEX IF NOT EXISTS ix_agents_status ON agent_evolution.agents(status);
CREATE INDEX IF NOT EXISTS ix_agents_generation ON agent_evolution.agents(generation);
CREATE INDEX IF NOT EXISTS ix_agents_fitness_score ON agent_evolution.agents(fitness_score DESC);
CREATE INDEX IF NOT EXISTS ix_agents_created_at ON agent_evolution.agents(created_at);
CREATE INDEX IF NOT EXISTS ix_agents_parent_ids ON agent_evolution.agents USING GIN(parent_ids);

-- Create evolution_history table per FR-011: Maintain evolutionary history
CREATE TABLE IF NOT EXISTS agent_evolution.evolution_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES agent_evolution.agents(id) ON DELETE CASCADE,
    
    -- Evolution details
    generation INTEGER NOT NULL,
    evolution_type VARCHAR(50) NOT NULL, -- cellular_automata, genetic_algorithm, mutation
    rule_applied VARCHAR(50), -- rule_110, rule_30, etc.
    
    -- Results
    fitness_before FLOAT NOT NULL,
    fitness_after FLOAT NOT NULL,
    fitness_delta FLOAT GENERATED ALWAYS AS (fitness_after - fitness_before) STORED,
    
    -- Pattern tracking
    patterns_applied UUID[],
    new_patterns_discovered INTEGER DEFAULT 0,
    
    -- Context
    population_size INTEGER NOT NULL,
    population_diversity FLOAT NOT NULL,
    
    -- Timestamps
    evolved_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for evolution_history
CREATE INDEX IF NOT EXISTS ix_evolution_history_agent_id ON agent_evolution.evolution_history(agent_id);
CREATE INDEX IF NOT EXISTS ix_evolution_history_evolved_at ON agent_evolution.evolution_history(evolved_at);
CREATE INDEX IF NOT EXISTS ix_evolution_history_fitness_delta ON agent_evolution.evolution_history(fitness_delta DESC);

-- Create performance_metrics table per FR-006: Track value-per-token metrics
-- Using partitioning for time-series data
CREATE TABLE IF NOT EXISTS agent_evolution.performance_metrics (
    id UUID DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES agent_evolution.agents(id) ON DELETE CASCADE,
    
    -- Metric details
    metric_type VARCHAR(50) NOT NULL, -- speed, quality, efficiency, pattern_discovery
    metric_value FLOAT NOT NULL,
    
    -- Token tracking
    tokens_used INTEGER NOT NULL,
    value_per_token FLOAT GENERATED ALWAYS AS (
        CASE WHEN tokens_used > 0 THEN metric_value / tokens_used ELSE 0 END
    ) STORED,
    
    -- Task context
    task_type VARCHAR(100),
    task_description TEXT,
    
    -- Timestamps
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (id, timestamp)
) PARTITION BY RANGE (timestamp);

-- Create partitions for current and next month
CREATE TABLE IF NOT EXISTS agent_evolution.performance_metrics_2025_01 
PARTITION OF agent_evolution.performance_metrics
FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

CREATE TABLE IF NOT EXISTS agent_evolution.performance_metrics_2025_02 
PARTITION OF agent_evolution.performance_metrics
FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');

-- Create indexes for performance_metrics
CREATE INDEX IF NOT EXISTS ix_performance_metrics_agent_id ON agent_evolution.performance_metrics(agent_id);
CREATE INDEX IF NOT EXISTS ix_performance_metrics_timestamp ON agent_evolution.performance_metrics(timestamp);
CREATE INDEX IF NOT EXISTS ix_performance_metrics_value_per_token ON agent_evolution.performance_metrics(value_per_token DESC);

-- Create discovered_patterns table per FR-015 and FR-027
CREATE TABLE IF NOT EXISTS agent_evolution.discovered_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES agent_evolution.agents(id) ON DELETE CASCADE,
    
    -- Pattern identification
    pattern_hash VARCHAR(64) UNIQUE NOT NULL,
    pattern_type VARCHAR(50) NOT NULL, -- efficiency_optimization, collaboration, innovation
    pattern_content JSONB NOT NULL,
    
    -- Effectiveness tracking per FR-027
    effectiveness_score FLOAT NOT NULL DEFAULT 0.0 CHECK (effectiveness_score >= -1 AND effectiveness_score <= 2),
    confidence_score FLOAT NOT NULL DEFAULT 0.5 CHECK (confidence_score >= 0 AND confidence_score <= 1),
    
    -- Economic impact
    token_efficiency_delta FLOAT DEFAULT 0.0,
    performance_improvement FLOAT DEFAULT 0.0,
    
    -- Usage tracking per FR-024
    reuse_count INTEGER NOT NULL DEFAULT 0,
    last_used_at TIMESTAMP WITH TIME ZONE,
    
    -- Timestamps
    discovered_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create indexes for discovered_patterns per FR-024: searchable pattern library
CREATE INDEX IF NOT EXISTS ix_discovered_patterns_pattern_type ON agent_evolution.discovered_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS ix_discovered_patterns_effectiveness ON agent_evolution.discovered_patterns(effectiveness_score DESC);
CREATE INDEX IF NOT EXISTS ix_discovered_patterns_reuse_count ON agent_evolution.discovered_patterns(reuse_count DESC);
CREATE INDEX IF NOT EXISTS ix_discovered_patterns_discovered_at ON agent_evolution.discovered_patterns(discovered_at);
CREATE INDEX IF NOT EXISTS ix_discovered_patterns_pattern_content ON agent_evolution.discovered_patterns USING GIN(pattern_content);

-- Create audit_log table per NFR-008: All agent actions SHALL be auditable
CREATE TABLE IF NOT EXISTS agent_evolution.audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES agent_evolution.agents(id) ON DELETE CASCADE,
    
    -- Action details
    action_type VARCHAR(100) NOT NULL,
    action_description TEXT NOT NULL,
    
    -- Resource tracking
    target_resource VARCHAR(500),
    tokens_consumed INTEGER DEFAULT 0,
    
    -- Results
    success BOOLEAN NOT NULL,
    error_message TEXT,
    execution_time_ms INTEGER,
    
    -- Timestamps
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for audit_log
CREATE INDEX IF NOT EXISTS ix_audit_log_agent_id ON agent_evolution.audit_log(agent_id);
CREATE INDEX IF NOT EXISTS ix_audit_log_timestamp ON agent_evolution.audit_log(timestamp);
CREATE INDEX IF NOT EXISTS ix_audit_log_action_type ON agent_evolution.audit_log(action_type);

-- Create token_transactions table for economic tracking
CREATE TABLE IF NOT EXISTS agent_evolution.token_transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES agent_evolution.agents(id) ON DELETE CASCADE,
    
    -- Transaction details
    transaction_type VARCHAR(50) NOT NULL, -- allocation, consumption, reallocation
    amount INTEGER NOT NULL CHECK (amount != 0),
    
    -- Context
    reason VARCHAR(255) NOT NULL,
    task_id UUID,
    pattern_id UUID REFERENCES agent_evolution.discovered_patterns(id),
    
    -- Balance tracking
    balance_before INTEGER NOT NULL,
    balance_after INTEGER NOT NULL,
    
    -- Timestamps
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for token_transactions
CREATE INDEX IF NOT EXISTS ix_token_transactions_agent_id ON agent_evolution.token_transactions(agent_id);
CREATE INDEX IF NOT EXISTS ix_token_transactions_timestamp ON agent_evolution.token_transactions(timestamp);
CREATE INDEX IF NOT EXISTS ix_token_transactions_type ON agent_evolution.token_transactions(transaction_type);

-- Create materialized views for analytics per Data Architecture Section 4.1
CREATE MATERIALIZED VIEW IF NOT EXISTS agent_evolution.agent_efficiency_summary AS
SELECT 
    a.id as agent_id,
    a.name,
    a.generation,
    a.token_consumed,
    a.token_budget,
    CASE WHEN a.token_consumed > 0 
         THEN a.token_budget::float / a.token_consumed 
         ELSE 1.0 END as efficiency_ratio,
    a.fitness_score,
    COUNT(DISTINCT pm.id) as metric_count,
    AVG(pm.value_per_token) as avg_value_per_token,
    MAX(pm.timestamp) as last_activity
FROM agent_evolution.agents a
LEFT JOIN agent_evolution.performance_metrics pm ON a.id = pm.agent_id
WHERE a.status = 'active'
GROUP BY a.id, a.name, a.generation, a.token_consumed, a.token_budget, a.fitness_score;

CREATE UNIQUE INDEX IF NOT EXISTS ix_agent_efficiency_summary_agent_id 
ON agent_evolution.agent_efficiency_summary(agent_id);

-- Pattern effectiveness summary
CREATE MATERIALIZED VIEW IF NOT EXISTS agent_evolution.pattern_effectiveness_summary AS
SELECT 
    pattern_type,
    COUNT(*) as pattern_count,
    AVG(effectiveness_score) as avg_effectiveness,
    SUM(reuse_count) as total_reuses,
    AVG(token_efficiency_delta) as avg_efficiency_improvement,
    MAX(discovered_at) as latest_discovery
FROM agent_evolution.discovered_patterns
GROUP BY pattern_type;

CREATE UNIQUE INDEX IF NOT EXISTS ix_pattern_effectiveness_summary_type
ON agent_evolution.pattern_effectiveness_summary(pattern_type);

-- Create functions for common operations

-- Function to update agent token consumption
CREATE OR REPLACE FUNCTION agent_evolution.update_agent_tokens(
    p_agent_id UUID,
    p_tokens_consumed INTEGER,
    p_reason VARCHAR
) RETURNS TABLE(success BOOLEAN, remaining_tokens INTEGER, message TEXT) AS $$
DECLARE
    v_current_consumed INTEGER;
    v_budget INTEGER;
    v_new_consumed INTEGER;
BEGIN
    -- Get current values with lock
    SELECT token_consumed, token_budget 
    INTO v_current_consumed, v_budget
    FROM agent_evolution.agents 
    WHERE id = p_agent_id 
    FOR UPDATE;
    
    IF NOT FOUND THEN
        RETURN QUERY SELECT FALSE, 0, 'Agent not found'::TEXT;
        RETURN;
    END IF;
    
    v_new_consumed := v_current_consumed + p_tokens_consumed;
    
    IF v_new_consumed > v_budget THEN
        RETURN QUERY SELECT FALSE, v_budget - v_current_consumed, 'Insufficient token budget'::TEXT;
        RETURN;
    END IF;
    
    -- Update agent
    UPDATE agent_evolution.agents 
    SET token_consumed = v_new_consumed,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_agent_id;
    
    -- Record transaction
    INSERT INTO agent_evolution.token_transactions 
    (agent_id, transaction_type, amount, reason, balance_before, balance_after)
    VALUES 
    (p_agent_id, 'consumption', -p_tokens_consumed, p_reason, 
     v_budget - v_current_consumed, v_budget - v_new_consumed);
    
    RETURN QUERY SELECT TRUE, v_budget - v_new_consumed, 'Success'::TEXT;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate population diversity
CREATE OR REPLACE FUNCTION agent_evolution.calculate_population_diversity()
RETURNS FLOAT AS $$
DECLARE
    v_diversity FLOAT;
BEGIN
    -- Simple diversity calculation based on fitness score variance
    SELECT COALESCE(STDDEV(fitness_score) / NULLIF(AVG(fitness_score), 0), 0)
    INTO v_diversity
    FROM agent_evolution.agents
    WHERE status = 'active';
    
    RETURN LEAST(v_diversity, 1.0);
END;
$$ LANGUAGE plpgsql;

-- Create update trigger for updated_at
CREATE OR REPLACE FUNCTION agent_evolution.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_agents_updated_at BEFORE UPDATE ON agent_evolution.agents
    FOR EACH ROW EXECUTE FUNCTION agent_evolution.update_updated_at_column();

-- Create dean_api user if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_user WHERE usename = 'dean_api') THEN
        CREATE USER dean_api WITH PASSWORD 'dean_api_password';
    END IF;
END
$$;

-- Grant permissions (adjust as needed)
GRANT USAGE ON SCHEMA agent_evolution TO dean_api;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA agent_evolution TO dean_api;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA agent_evolution TO dean_api;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA agent_evolution TO dean_api;

-- Initial data for testing (optional)
-- INSERT INTO agent_evolution.agents (name, goal, token_budget)
-- VALUES 
-- ('Agent_Alpha', 'Optimize code performance', 5000),
-- ('Agent_Beta', 'Detect security vulnerabilities', 5000),
-- ('Agent_Gamma', 'Improve test coverage', 5000);

-- Maintenance commands
COMMENT ON SCHEMA agent_evolution IS 'DEAN System agent evolution and metrics tracking';
COMMENT ON TABLE agent_evolution.agents IS 'Core agent registry tracking lineage and performance per FR-004';
COMMENT ON TABLE agent_evolution.performance_metrics IS 'Time-series performance metrics per FR-006';
COMMENT ON TABLE agent_evolution.discovered_patterns IS 'Emergent behavior patterns per FR-015 and FR-027';
COMMENT ON TABLE agent_evolution.audit_log IS 'Complete audit trail per NFR-008';