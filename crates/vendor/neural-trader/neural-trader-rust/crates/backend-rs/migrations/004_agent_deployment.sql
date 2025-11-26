-- Agent Deployment Tables
-- Migration: 004_agent_deployment.sql

-- Agent Deployments table
CREATE TABLE IF NOT EXISTS agent_deployments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    agent_type VARCHAR(100) NOT NULL, -- researcher, coder, tester, reviewer, etc.
    sandbox_id VARCHAR(255) UNIQUE,
    template VARCHAR(100) NOT NULL DEFAULT 'base', -- base, claude-code, python, node, etc.
    model VARCHAR(100) NOT NULL DEFAULT 'anthropic/claude-3.5-sonnet',
    status VARCHAR(50) NOT NULL DEFAULT 'pending', -- pending, running, completed, failed, terminated

    -- Configuration
    capabilities TEXT[], -- array of agent capabilities
    environment JSONB DEFAULT '{}', -- environment variables
    config JSONB DEFAULT '{}', -- agent-specific configuration

    -- E2B Sandbox details
    sandbox_url VARCHAR(500),
    sandbox_metadata JSONB DEFAULT '{}',

    -- Execution tracking
    task_description TEXT,
    execution_logs TEXT,
    error_message TEXT,
    result JSONB,

    -- Metrics
    tokens_used INTEGER DEFAULT 0,
    execution_time_ms INTEGER DEFAULT 0,
    cost_usd DECIMAL(10, 6) DEFAULT 0.0,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    terminated_at TIMESTAMP WITH TIME ZONE,

    -- Foreign keys
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    swarm_id UUID REFERENCES swarm_configurations(id) ON DELETE SET NULL
);

-- Swarm Configurations table
CREATE TABLE IF NOT EXISTS swarm_configurations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    topology VARCHAR(50) NOT NULL, -- mesh, hierarchical, ring, star
    strategy VARCHAR(50) NOT NULL DEFAULT 'balanced', -- balanced, specialized, adaptive
    max_agents INTEGER NOT NULL DEFAULT 5,
    status VARCHAR(50) NOT NULL DEFAULT 'active', -- active, completed, failed, terminated

    -- Configuration
    config JSONB DEFAULT '{}',

    -- Metrics
    total_agents INTEGER DEFAULT 0,
    active_agents INTEGER DEFAULT 0,
    completed_tasks INTEGER DEFAULT 0,
    total_cost_usd DECIMAL(10, 6) DEFAULT 0.0,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,

    -- Foreign keys
    user_id UUID REFERENCES users(id) ON DELETE CASCADE
);

-- Execution Logs table for streaming logs
CREATE TABLE IF NOT EXISTS execution_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES agent_deployments(id) ON DELETE CASCADE,
    log_level VARCHAR(20) NOT NULL DEFAULT 'info', -- debug, info, warn, error
    message TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Indexing for efficient queries
    CONSTRAINT fk_agent_deployment FOREIGN KEY (agent_id) REFERENCES agent_deployments(id)
);

-- Agent Tasks table for tracking individual tasks
CREATE TABLE IF NOT EXISTS agent_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES agent_deployments(id) ON DELETE CASCADE,
    task_type VARCHAR(100) NOT NULL, -- research, code, test, review, etc.
    description TEXT NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending', -- pending, running, completed, failed

    -- Execution details
    command TEXT,
    input_data JSONB,
    output_data JSONB,
    error_message TEXT,

    -- Metrics
    execution_time_ms INTEGER DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Swarm Tasks table for coordinated work
CREATE TABLE IF NOT EXISTS swarm_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    swarm_id UUID NOT NULL REFERENCES swarm_configurations(id) ON DELETE CASCADE,
    task_description TEXT NOT NULL,
    priority VARCHAR(20) NOT NULL DEFAULT 'medium', -- low, medium, high, critical
    execution_strategy VARCHAR(50) NOT NULL DEFAULT 'adaptive', -- parallel, sequential, adaptive
    status VARCHAR(50) NOT NULL DEFAULT 'pending',

    -- Agent assignment
    assigned_agents UUID[],
    max_agents INTEGER DEFAULT 5,

    -- Results
    result JSONB,
    error_message TEXT,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_agent_deployments_user_id ON agent_deployments(user_id);
CREATE INDEX IF NOT EXISTS idx_agent_deployments_status ON agent_deployments(status);
CREATE INDEX IF NOT EXISTS idx_agent_deployments_swarm_id ON agent_deployments(swarm_id);
CREATE INDEX IF NOT EXISTS idx_agent_deployments_created_at ON agent_deployments(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_swarm_configurations_user_id ON swarm_configurations(user_id);
CREATE INDEX IF NOT EXISTS idx_swarm_configurations_status ON swarm_configurations(status);

CREATE INDEX IF NOT EXISTS idx_execution_logs_agent_id ON execution_logs(agent_id);
CREATE INDEX IF NOT EXISTS idx_execution_logs_timestamp ON execution_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_execution_logs_log_level ON execution_logs(log_level);

CREATE INDEX IF NOT EXISTS idx_agent_tasks_agent_id ON agent_tasks(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_tasks_status ON agent_tasks(status);

CREATE INDEX IF NOT EXISTS idx_swarm_tasks_swarm_id ON swarm_tasks(swarm_id);
CREATE INDEX IF NOT EXISTS idx_swarm_tasks_status ON swarm_tasks(status);
CREATE INDEX IF NOT EXISTS idx_swarm_tasks_priority ON swarm_tasks(priority);

-- Create views for common queries
CREATE OR REPLACE VIEW agent_deployment_stats AS
SELECT
    user_id,
    COUNT(*) as total_deployments,
    COUNT(*) FILTER (WHERE status = 'running') as running_agents,
    COUNT(*) FILTER (WHERE status = 'completed') as completed_agents,
    COUNT(*) FILTER (WHERE status = 'failed') as failed_agents,
    SUM(tokens_used) as total_tokens,
    SUM(cost_usd) as total_cost,
    AVG(execution_time_ms) as avg_execution_time
FROM agent_deployments
GROUP BY user_id;

CREATE OR REPLACE VIEW swarm_stats AS
SELECT
    s.id as swarm_id,
    s.name,
    s.topology,
    s.status,
    COUNT(a.id) as agent_count,
    COUNT(a.id) FILTER (WHERE a.status = 'running') as running_agents,
    SUM(a.tokens_used) as total_tokens,
    SUM(a.cost_usd) as total_cost,
    s.created_at,
    s.completed_at
FROM swarm_configurations s
LEFT JOIN agent_deployments a ON s.id = a.swarm_id
GROUP BY s.id, s.name, s.topology, s.status, s.created_at, s.completed_at;
