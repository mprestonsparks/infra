-- DEAN System Knowledge Repository Schema
-- Designed for analysis, not just logging

-- ========================================
-- Core Tables
-- ========================================

-- Agent registry
CREATE TABLE IF NOT EXISTS agents (
    agent_id TEXT PRIMARY KEY,
    genome_hash TEXT NOT NULL,
    generation INTEGER NOT NULL,
    parent_ids TEXT,  -- JSON array of parent IDs
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    terminated_at TIMESTAMP,
    termination_reason TEXT,
    total_tokens_used INTEGER DEFAULT 0,
    total_value_generated REAL DEFAULT 0.0,
    efficiency_score REAL DEFAULT 0.0,
    metadata JSONB
);

CREATE INDEX idx_agents_generation ON agents(generation);
CREATE INDEX idx_agents_efficiency ON agents(efficiency_score DESC);
CREATE INDEX idx_agents_created ON agents(created_at);

-- ========================================
-- Pattern Storage
-- ========================================

-- Discovered patterns with analytical indexes
CREATE TABLE IF NOT EXISTS discovered_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_hash TEXT UNIQUE NOT NULL,
    pattern_type TEXT NOT NULL,
    description TEXT,
    effectiveness_score REAL NOT NULL,
    token_efficiency REAL NOT NULL,
    confidence_score REAL NOT NULL,
    reuse_count INTEGER DEFAULT 0,
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    discovered_by TEXT REFERENCES agents(agent_id),
    pattern_data JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Critical indexes for meta-learning
CREATE INDEX idx_pattern_effectiveness ON discovered_patterns(effectiveness_score DESC);
CREATE INDEX idx_pattern_efficiency ON discovered_patterns(token_efficiency DESC);
CREATE INDEX idx_pattern_reuse ON discovered_patterns(reuse_count DESC);
CREATE INDEX idx_pattern_type ON discovered_patterns(pattern_type);
CREATE INDEX idx_pattern_confidence ON discovered_patterns(confidence_score DESC);
CREATE INDEX idx_pattern_discovered_by ON discovered_patterns(discovered_by);

-- Pattern adoptions tracking
CREATE TABLE IF NOT EXISTS pattern_adoptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_id UUID REFERENCES discovered_patterns(id),
    agent_id TEXT REFERENCES agents(agent_id),
    adopted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    success BOOLEAN,
    performance_delta REAL,
    token_cost INTEGER,
    adaptation_details JSONB
);

CREATE INDEX idx_adoptions_pattern ON pattern_adoptions(pattern_id);
CREATE INDEX idx_adoptions_agent ON pattern_adoptions(agent_id);
CREATE INDEX idx_adoptions_success ON pattern_adoptions(success);

-- ========================================
-- Evolution Tracking
-- ========================================

-- Genome evolution history
CREATE TABLE IF NOT EXISTS genome_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT REFERENCES agents(agent_id),
    generation INTEGER NOT NULL,
    genome_hash TEXT NOT NULL,
    gene_count INTEGER,
    gene_types JSONB,  -- Array of gene type counts
    mutation_count INTEGER DEFAULT 0,
    crossover_info JSONB,
    fitness_scores JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_genome_agent ON genome_history(agent_id);
CREATE INDEX idx_genome_generation ON genome_history(generation);
CREATE INDEX idx_genome_hash ON genome_history(genome_hash);

-- Population diversity metrics
CREATE TABLE IF NOT EXISTS diversity_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    generation INTEGER NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    genetic_diversity REAL NOT NULL,
    gene_type_diversity REAL NOT NULL,
    value_diversity REAL NOT NULL,
    cluster_count INTEGER,
    convergence_risk REAL,
    unique_genes INTEGER,
    average_distance REAL,
    intervention_applied BOOLEAN DEFAULT FALSE,
    intervention_type TEXT,
    metadata JSONB
);

CREATE INDEX idx_diversity_generation ON diversity_metrics(generation);
CREATE INDEX idx_diversity_timestamp ON diversity_metrics(timestamp);
CREATE INDEX idx_diversity_risk ON diversity_metrics(convergence_risk DESC);

-- ========================================
-- Performance Analytics
-- ========================================

-- Detailed performance metrics
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT REFERENCES agents(agent_id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    action_type TEXT NOT NULL,
    tokens_used INTEGER NOT NULL,
    value_generated REAL NOT NULL,
    efficiency REAL GENERATED ALWAYS AS (
        CASE WHEN tokens_used > 0 THEN value_generated / tokens_used ELSE 0 END
    ) STORED,
    context JSONB,
    patterns_used JSONB,  -- Array of pattern IDs
    metadata JSONB
);

CREATE INDEX idx_perf_agent ON performance_metrics(agent_id);
CREATE INDEX idx_perf_timestamp ON performance_metrics(timestamp);
CREATE INDEX idx_perf_action ON performance_metrics(action_type);
CREATE INDEX idx_perf_efficiency ON performance_metrics(efficiency DESC);

-- Aggregated performance statistics
CREATE MATERIALIZED VIEW agent_performance_summary AS
SELECT 
    agent_id,
    COUNT(*) as total_actions,
    SUM(tokens_used) as total_tokens,
    SUM(value_generated) as total_value,
    AVG(efficiency) as avg_efficiency,
    STDDEV(efficiency) as efficiency_stddev,
    MAX(efficiency) as max_efficiency,
    MIN(timestamp) as first_action,
    MAX(timestamp) as last_action,
    EXTRACT(EPOCH FROM (MAX(timestamp) - MIN(timestamp))) as active_duration_seconds
FROM performance_metrics
GROUP BY agent_id;

CREATE UNIQUE INDEX idx_perf_summary_agent ON agent_performance_summary(agent_id);

-- ========================================
-- Emergent Behavior Tracking
-- ========================================

-- Emergent strategies catalog
CREATE TABLE IF NOT EXISTS emergent_strategies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_hash TEXT UNIQUE NOT NULL,
    description TEXT,
    first_observed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    agent_count INTEGER DEFAULT 1,
    effectiveness REAL NOT NULL,
    replicability REAL,
    component_patterns JSONB,  -- Array of pattern IDs
    prerequisites JSONB,
    metadata JSONB
);

CREATE INDEX idx_strategy_effectiveness ON emergent_strategies(effectiveness DESC);
CREATE INDEX idx_strategy_adoption ON emergent_strategies(agent_count DESC);
CREATE INDEX idx_strategy_replicability ON emergent_strategies(replicability DESC);

-- Gaming behavior incidents
CREATE TABLE IF NOT EXISTS gaming_incidents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT REFERENCES agents(agent_id),
    incident_type TEXT NOT NULL,
    severity REAL NOT NULL,
    description TEXT,
    evidence JSONB NOT NULL,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    remediation_applied BOOLEAN DEFAULT FALSE,
    remediation_details JSONB
);

CREATE INDEX idx_gaming_agent ON gaming_incidents(agent_id);
CREATE INDEX idx_gaming_type ON gaming_incidents(incident_type);
CREATE INDEX idx_gaming_severity ON gaming_incidents(severity DESC);
CREATE INDEX idx_gaming_timestamp ON gaming_incidents(detected_at);

-- ========================================
-- Token Economy Tracking
-- ========================================

-- Token allocations
CREATE TABLE IF NOT EXISTS token_allocations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT REFERENCES agents(agent_id),
    allocated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    token_amount INTEGER NOT NULL,
    allocation_reason TEXT,
    historical_efficiency REAL,
    performance_multiplier REAL,
    metadata JSONB
);

CREATE INDEX idx_allocation_agent ON token_allocations(agent_id);
CREATE INDEX idx_allocation_timestamp ON token_allocations(allocated_at);

-- Token consumption events
CREATE TABLE IF NOT EXISTS token_consumption (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT REFERENCES agents(agent_id),
    consumed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    operation_type TEXT NOT NULL,
    tokens_consumed INTEGER NOT NULL,
    value_produced REAL,
    efficiency REAL GENERATED ALWAYS AS (
        CASE WHEN tokens_consumed > 0 THEN value_produced / tokens_consumed ELSE 0 END
    ) STORED,
    consumption_rate REAL,  -- Tokens per second
    metadata JSONB
);

CREATE INDEX idx_consumption_agent ON token_consumption(agent_id);
CREATE INDEX idx_consumption_timestamp ON token_consumption(consumed_at);
CREATE INDEX idx_consumption_operation ON token_consumption(operation_type);
CREATE INDEX idx_consumption_efficiency ON token_consumption(efficiency DESC);

-- ========================================
-- Meta-Learning Tables
-- ========================================

-- Learned insights and meta-patterns
CREATE TABLE IF NOT EXISTS learned_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    insight_type TEXT NOT NULL,
    description TEXT,
    confidence REAL NOT NULL,
    supporting_evidence JSONB,  -- Pattern IDs, agent IDs, etc.
    applicable_contexts JSONB,
    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    validation_count INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 0.0,
    metadata JSONB
);

CREATE INDEX idx_insight_type ON learned_insights(insight_type);
CREATE INDEX idx_insight_confidence ON learned_insights(confidence DESC);
CREATE INDEX idx_insight_validation ON learned_insights(validation_count DESC);

-- Strategy combinations that work well
CREATE TABLE IF NOT EXISTS strategy_combinations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_ids JSONB NOT NULL,  -- Array of strategy IDs
    synergy_score REAL NOT NULL,
    combined_effectiveness REAL NOT NULL,
    usage_count INTEGER DEFAULT 0,
    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

CREATE INDEX idx_combo_synergy ON strategy_combinations(synergy_score DESC);
CREATE INDEX idx_combo_usage ON strategy_combinations(usage_count DESC);

-- ========================================
-- Analytical Views
-- ========================================

-- Top performing patterns by generation
CREATE MATERIALIZED VIEW top_patterns_by_generation AS
SELECT 
    dp.pattern_type,
    dp.pattern_hash,
    dp.description,
    dp.effectiveness_score,
    dp.token_efficiency,
    dp.reuse_count,
    COUNT(DISTINCT pa.agent_id) as adopter_count,
    AVG(pa.performance_delta) as avg_performance_gain,
    MAX(gh.generation) as latest_generation
FROM discovered_patterns dp
LEFT JOIN pattern_adoptions pa ON dp.id = pa.pattern_id
LEFT JOIN agents a ON pa.agent_id = a.agent_id
LEFT JOIN genome_history gh ON a.agent_id = gh.agent_id
WHERE pa.success = true
GROUP BY dp.id, dp.pattern_type, dp.pattern_hash, dp.description, 
         dp.effectiveness_score, dp.token_efficiency, dp.reuse_count
ORDER BY dp.effectiveness_score * dp.token_efficiency DESC;

-- Evolution progress tracking
CREATE MATERIALIZED VIEW evolution_progress AS
SELECT 
    generation,
    COUNT(DISTINCT agent_id) as population_size,
    AVG(efficiency_score) as avg_efficiency,
    MAX(efficiency_score) as best_efficiency,
    MIN(efficiency_score) as worst_efficiency,
    STDDEV(efficiency_score) as efficiency_variance,
    SUM(total_tokens_used) as generation_token_cost,
    SUM(total_value_generated) as generation_value,
    COUNT(DISTINCT genome_hash) as unique_genomes
FROM agents
GROUP BY generation
ORDER BY generation;

-- Pattern discovery timeline
CREATE MATERIALIZED VIEW pattern_discovery_timeline AS
SELECT 
    DATE_TRUNC('hour', first_seen) as time_bucket,
    COUNT(*) as patterns_discovered,
    AVG(effectiveness_score) as avg_effectiveness,
    AVG(token_efficiency) as avg_efficiency,
    SUM(reuse_count) as total_reuses
FROM discovered_patterns
GROUP BY time_bucket
ORDER BY time_bucket;

-- ========================================
-- Utility Functions
-- ========================================

-- Function to calculate agent lineage
CREATE OR REPLACE FUNCTION get_agent_lineage(target_agent_id TEXT)
RETURNS TABLE(
    agent_id TEXT,
    generation INTEGER,
    parent_ids TEXT,
    depth INTEGER
) AS $$
WITH RECURSIVE lineage AS (
    -- Base case: the target agent
    SELECT 
        a.agent_id,
        a.generation,
        a.parent_ids,
        0 as depth
    FROM agents a
    WHERE a.agent_id = target_agent_id
    
    UNION ALL
    
    -- Recursive case: parents
    SELECT 
        p.agent_id,
        p.generation,
        p.parent_ids,
        l.depth + 1
    FROM lineage l
    CROSS JOIN LATERAL (
        SELECT a.*
        FROM agents a
        WHERE a.agent_id = ANY(
            SELECT jsonb_array_elements_text(l.parent_ids::jsonb)
        )
    ) p
    WHERE l.depth < 10  -- Limit recursion depth
)
SELECT * FROM lineage
ORDER BY depth, generation DESC;
$$ LANGUAGE sql;

-- Function to find similar patterns
CREATE OR REPLACE FUNCTION find_similar_patterns(
    target_pattern_hash TEXT,
    similarity_threshold REAL DEFAULT 0.7
)
RETURNS TABLE(
    pattern_hash TEXT,
    similarity_score REAL,
    effectiveness_score REAL,
    description TEXT
) AS $$
SELECT 
    p2.pattern_hash,
    -- Simple similarity based on shared adoptions
    COUNT(DISTINCT pa1.agent_id)::REAL / 
        GREATEST(
            (SELECT COUNT(DISTINCT agent_id) FROM pattern_adoptions WHERE pattern_id = p1.id),
            (SELECT COUNT(DISTINCT agent_id) FROM pattern_adoptions WHERE pattern_id = p2.id)
        ) as similarity_score,
    p2.effectiveness_score,
    p2.description
FROM discovered_patterns p1
CROSS JOIN discovered_patterns p2
LEFT JOIN pattern_adoptions pa1 ON p1.id = pa1.pattern_id
LEFT JOIN pattern_adoptions pa2 ON p2.id = pa2.pattern_id AND pa1.agent_id = pa2.agent_id
WHERE p1.pattern_hash = target_pattern_hash
    AND p2.pattern_hash != target_pattern_hash
    AND p1.pattern_type = p2.pattern_type
GROUP BY p2.id, p2.pattern_hash, p2.effectiveness_score, p2.description
HAVING COUNT(DISTINCT pa1.agent_id)::REAL / 
    GREATEST(
        (SELECT COUNT(DISTINCT agent_id) FROM pattern_adoptions WHERE pattern_id = p1.id),
        (SELECT COUNT(DISTINCT agent_id) FROM pattern_adoptions WHERE pattern_id = p2.id)
    ) >= similarity_threshold
ORDER BY similarity_score DESC;
$$ LANGUAGE sql;