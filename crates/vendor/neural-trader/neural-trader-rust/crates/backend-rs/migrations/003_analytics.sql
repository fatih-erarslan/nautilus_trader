-- Analytics Events Table
CREATE TABLE IF NOT EXISTS analytics_events (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_category TEXT NOT NULL,
    resource_type TEXT,
    resource_id TEXT,
    metadata TEXT NOT NULL,
    duration_ms INTEGER,
    status TEXT,
    ip_address TEXT,
    user_agent TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_analytics_user_id ON analytics_events(user_id);
CREATE INDEX IF NOT EXISTS idx_analytics_event_type ON analytics_events(event_type);
CREATE INDEX IF NOT EXISTS idx_analytics_category ON analytics_events(event_category);
CREATE INDEX IF NOT EXISTS idx_analytics_created_at ON analytics_events(created_at);

-- Activity Log Table
CREATE TABLE IF NOT EXISTS activity_log (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    action TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT,
    entity_name TEXT,
    description TEXT,
    changes TEXT,
    severity TEXT DEFAULT 'info',
    ip_address TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_activity_user_id ON activity_log(user_id);
CREATE INDEX IF NOT EXISTS idx_activity_action ON activity_log(action);
CREATE INDEX IF NOT EXISTS idx_activity_entity_type ON activity_log(entity_type);
CREATE INDEX IF NOT EXISTS idx_activity_created_at ON activity_log(created_at);

-- Performance Metrics Table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id TEXT PRIMARY KEY,
    metric_type TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT NOT NULL,
    tags TEXT,
    resource_id TEXT,
    recorded_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_metrics_type ON performance_metrics(metric_type);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON performance_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_metrics_recorded_at ON performance_metrics(recorded_at);

-- Usage Statistics (Aggregated)
CREATE TABLE IF NOT EXISTS usage_statistics (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    date TEXT NOT NULL,
    api_calls_count INTEGER DEFAULT 0,
    workflows_executed INTEGER DEFAULT 0,
    scans_performed INTEGER DEFAULT 0,
    total_execution_time_ms INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(user_id, date)
);

CREATE INDEX IF NOT EXISTS idx_usage_user_date ON usage_statistics(user_id, date);

-- Insert initial test data
INSERT OR IGNORE INTO analytics_events (id, user_id, event_type, event_category, resource_type, metadata, status, created_at)
VALUES
    ('evt-1', 'user-1', 'api_call', 'usage', 'api', '{"endpoint": "/api/stats", "method": "GET"}', 'success', datetime('now', '-1 day')),
    ('evt-2', 'user-1', 'workflow_execution', 'performance', 'workflow', '{"workflow_id": "wf-1", "duration_ms": 1234}', 'success', datetime('now', '-2 hours')),
    ('evt-3', 'user-1', 'scan_started', 'security', 'scan', '{"url": "https://api.example.com", "type": "openapi"}', 'success', datetime('now', '-30 minutes'));

INSERT OR IGNORE INTO activity_log (id, user_id, action, entity_type, entity_id, entity_name, description, severity, created_at)
VALUES
    ('act-1', 'user-1', 'create', 'workflow', 'wf-1', 'New API Workflow', 'Created new workflow for API integration', 'info', datetime('now', '-3 hours')),
    ('act-2', 'user-1', 'execute', 'workflow', 'wf-1', 'New API Workflow', 'Executed workflow successfully', 'info', datetime('now', '-2 hours')),
    ('act-3', 'user-1', 'create', 'scan', 'scan-1', 'API Security Scan', 'Started security scan for API endpoints', 'info', datetime('now', '-1 hour'));

INSERT OR IGNORE INTO performance_metrics (id, metric_type, metric_name, value, unit, recorded_at)
VALUES
    ('pm-1', 'api_latency', 'GET /api/stats', 45.3, 'ms', datetime('now', '-1 hour')),
    ('pm-2', 'api_latency', 'POST /api/workflows', 123.7, 'ms', datetime('now', '-2 hours')),
    ('pm-3', 'scan_duration', 'openapi_scan', 5420.0, 'ms', datetime('now', '-30 minutes'));

INSERT OR IGNORE INTO usage_statistics (id, user_id, date, api_calls_count, workflows_executed, scans_performed, success_count)
VALUES
    ('stat-1', 'user-1', date('now'), 15, 3, 2, 18),
    ('stat-2', 'user-1', date('now', '-1 day'), 23, 5, 4, 26);
