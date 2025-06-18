-- create_databases.sql
-- Creates multiple databases for DEAN system components

-- Create databases
CREATE DATABASE airflow;
CREATE DATABASE indexagent;
CREATE DATABASE market_analysis;
CREATE DATABASE agent_evolution;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE airflow TO dean_user;
GRANT ALL PRIVILEGES ON DATABASE indexagent TO dean_user;
GRANT ALL PRIVILEGES ON DATABASE market_analysis TO dean_user;
GRANT ALL PRIVILEGES ON DATABASE agent_evolution TO dean_user;

-- Connect to agent_evolution database and create initial schema
\c agent_evolution;

-- Create schema for agent evolution data
CREATE SCHEMA IF NOT EXISTS evolution;

-- Agent table
CREATE TABLE IF NOT EXISTS evolution.agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL,
    token_budget INTEGER DEFAULT 1000,
    goals JSONB,
    behavior_params JSONB,
    fitness_score FLOAT DEFAULT 0.0,
    generation INTEGER DEFAULT 0,
    parent_id UUID REFERENCES evolution.agents(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Evolution trials table
CREATE TABLE IF NOT EXISTS evolution.trials (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trial_id VARCHAR(255) UNIQUE NOT NULL,
    repository_id VARCHAR(255),
    repository_name VARCHAR(255),
    config JSONB,
    status VARCHAR(50) DEFAULT 'running',
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    results JSONB
);

-- Improvements table
CREATE TABLE IF NOT EXISTS evolution.improvements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES evolution.agents(id),
    trial_id UUID REFERENCES evolution.trials(id),
    type VARCHAR(100),
    description TEXT,
    impact VARCHAR(20),
    code_before TEXT,
    code_after TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Metrics table
CREATE TABLE IF NOT EXISTS evolution.metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trial_id UUID REFERENCES evolution.trials(id),
    generation INTEGER,
    metric_name VARCHAR(100),
    metric_value FLOAT,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_agents_generation ON evolution.agents(generation);
CREATE INDEX idx_agents_fitness ON evolution.agents(fitness_score DESC);
CREATE INDEX idx_improvements_agent ON evolution.improvements(agent_id);
CREATE INDEX idx_improvements_trial ON evolution.improvements(trial_id);
CREATE INDEX idx_metrics_trial ON evolution.metrics(trial_id, generation);

-- Grant permissions
GRANT ALL ON SCHEMA evolution TO dean_user;
GRANT ALL ON ALL TABLES IN SCHEMA evolution TO dean_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA evolution TO dean_user;
EOF < /dev/null