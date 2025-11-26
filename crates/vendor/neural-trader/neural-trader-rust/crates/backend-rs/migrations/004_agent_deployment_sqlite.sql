-- Agent Deployment Tables (SQLite Compatible)
-- Migration: 004_agent_deployment_sqlite.sql

-- Agent Deployments table
CREATE TABLE IF NOT EXISTS agent_deployments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    agent_type TEXT NOT NULL,
    sandbox_id TEXT UNIQUE,
    template TEXT NOT NULL DEFAULT 'base',
    model TEXT NOT NULL DEFAULT 'anthropic/claude-3.5-sonnet',
    status TEXT NOT NULL DEFAULT 'pending',

    -- Configuration (stored as JSON strings)
    capabilities TEXT,
    environment TEXT DEFAULT '{}',
    config TEXT DEFAULT '{}',

    -- E2B Sandbox details
    sandbox_url TEXT,
    sandbox_metadata TEXT DEFAULT '{}',

    -- Execution tracking
    task_description TEXT,
    execution_logs TEXT,
    error_message TEXT,
    result TEXT,

    -- Metrics
    tokens_used INTEGER DEFAULT 0,
    execution_time_ms INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0.0,

    -- Timestamps (stored as TEXT in ISO 8601 format)
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    terminated_at TEXT,

    -- Foreign keys (TEXT for SQLite)
    user_id TEXT NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000',
    swarm_id TEXT
);

-- Swarm Configurations table
CREATE TABLE IF NOT EXISTS swarm_configurations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    topology TEXT NOT NULL,
    strategy TEXT NOT NULL DEFAULT 'balanced',
    max_agents INTEGER NOT NULL DEFAULT 5,
    status TEXT NOT NULL DEFAULT 'active',

    -- Configuration
    config TEXT DEFAULT '{}',

    -- Metrics
    total_agents INTEGER DEFAULT 0,
    active_agents INTEGER DEFAULT 0,
    completed_tasks INTEGER DEFAULT 0,
    total_cost_usd REAL DEFAULT 0.0,

    -- Timestamps
    created_at TEXT NOT NULL,
    completed_at TEXT,

    -- Foreign keys
    user_id TEXT NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000'
);

-- Execution Logs table for streaming logs
CREATE TABLE IF NOT EXISTS execution_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    log_level TEXT NOT NULL DEFAULT 'info',
    message TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    timestamp TEXT NOT NULL,

    FOREIGN KEY (agent_id) REFERENCES agent_deployments(id) ON DELETE CASCADE
);

-- Agent Tasks table for tracking individual tasks
CREATE TABLE IF NOT EXISTS agent_tasks (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    task_type TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',

    -- Execution details
    command TEXT,
    input_data TEXT,
    output_data TEXT,
    error_message TEXT,

    -- Metrics
    execution_time_ms INTEGER DEFAULT 0,

    -- Timestamps
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,

    FOREIGN KEY (agent_id) REFERENCES agent_deployments(id) ON DELETE CASCADE
);

-- Swarm Tasks table for coordinated work
CREATE TABLE IF NOT EXISTS swarm_tasks (
    id TEXT PRIMARY KEY,
    swarm_id TEXT NOT NULL,
    task_description TEXT NOT NULL,
    priority TEXT NOT NULL DEFAULT 'medium',
    execution_strategy TEXT NOT NULL DEFAULT 'adaptive',
    status TEXT NOT NULL DEFAULT 'pending',

    -- Agent assignment (stored as JSON array)
    assigned_agents TEXT,
    max_agents INTEGER DEFAULT 5,

    -- Results
    result TEXT,
    error_message TEXT,

    -- Timestamps
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,

    FOREIGN KEY (swarm_id) REFERENCES swarm_configurations(id) ON DELETE CASCADE
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
