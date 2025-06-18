-- DEAN System Complete Database Schema
-- This schema aligns with both API expectations and IndexAgent requirements

-- Create schema if not exists
CREATE SCHEMA IF NOT EXISTS agent_evolution;

-- Set search path
SET search_path TO agent_evolution;

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Drop existing tables for clean recreation
DROP TABLE IF EXISTS performance_metrics CASCADE;
DROP TABLE IF EXISTS evolution_history CASCADE;

-- Performance Metrics table with API-expected columns
CREATE TABLE performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    metric_type VARCHAR(50) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    additional_data JSONB,  -- API expects this for complexity metrics
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Additional columns for comprehensive tracking
    tokens_used INTEGER DEFAULT 0,
    task_type VARCHAR(100),
    task_description TEXT,
    
    -- Computed column for efficiency
    value_per_token DOUBLE PRECISION GENERATED ALWAYS AS (
        CASE 
            WHEN tokens_used > 0 THEN metric_value / tokens_used::DOUBLE PRECISION
            ELSE 0::DOUBLE PRECISION
        END
    ) STORED
);

-- Indexes for performance_metrics
CREATE INDEX ix_performance_metrics_agent_id ON performance_metrics(agent_id);
CREATE INDEX ix_performance_metrics_recorded_at ON performance_metrics(recorded_at);
CREATE INDEX ix_performance_metrics_metric_type ON performance_metrics(metric_type);
CREATE INDEX ix_performance_metrics_value_per_token ON performance_metrics(value_per_token DESC);

-- Evolution History table with API-expected columns
CREATE TABLE evolution_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    generation INTEGER NOT NULL,
    fitness_before DOUBLE PRECISION NOT NULL,
    fitness_after DOUBLE PRECISION NOT NULL,
    mutation_applied BOOLEAN DEFAULT FALSE,  -- API expects this
    crossover_parent_id UUID,  -- API expects this
    new_patterns_discovered INTEGER DEFAULT 0,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,  -- API expects this name
    
    -- Additional columns for comprehensive tracking
    evolution_type VARCHAR(50) DEFAULT 'genetic',
    rule_applied VARCHAR(50),
    patterns_applied UUID[],
    population_size INTEGER DEFAULT 1,
    population_diversity DOUBLE PRECISION DEFAULT 0.0,
    
    -- Cellular automata specific columns
    ca_rule_number INTEGER,
    ca_initial_state JSONB,
    ca_final_state JSONB,
    ca_complexity_score DOUBLE PRECISION,
    
    -- Computed column
    fitness_delta DOUBLE PRECISION GENERATED ALWAYS AS (fitness_after - fitness_before) STORED
);

-- Indexes for evolution_history
CREATE INDEX ix_evolution_history_agent_id ON evolution_history(agent_id);
CREATE INDEX ix_evolution_history_timestamp ON evolution_history(timestamp);
CREATE INDEX ix_evolution_history_generation ON evolution_history(generation);
CREATE INDEX ix_evolution_history_fitness_delta ON evolution_history(fitness_delta DESC);
CREATE INDEX ix_evolution_history_mutation_applied ON evolution_history(mutation_applied);

-- Additional tables that might be needed for cellular automata patterns
CREATE TABLE IF NOT EXISTS ca_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pattern_name VARCHAR(100) NOT NULL,
    pattern_type VARCHAR(50) NOT NULL,
    pattern_data JSONB NOT NULL,
    frequency INTEGER DEFAULT 1,
    effectiveness_score DOUBLE PRECISION DEFAULT 0.5,
    discovered_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Pattern associations
CREATE TABLE IF NOT EXISTS agent_ca_patterns (
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    pattern_id UUID NOT NULL REFERENCES ca_patterns(id) ON DELETE CASCADE,
    discovered_generation INTEGER NOT NULL,
    application_count INTEGER DEFAULT 0,
    PRIMARY KEY (agent_id, pattern_id)
);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA agent_evolution TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA agent_evolution TO postgres;