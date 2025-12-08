-- Initial schema for experiment tracking

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    status TEXT NOT NULL,
    start_time INTEGER NOT NULL,
    end_time INTEGER,
    config TEXT,
    metrics TEXT,
    tags TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    name TEXT NOT NULL,
    start_time INTEGER NOT NULL,
    end_time INTEGER,
    parameters TEXT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS run_metrics (
    run_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    timestamp INTEGER NOT NULL,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id),
    PRIMARY KEY (run_id, step, name)
);

CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    metadata TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

-- Indexes for performance
CREATE INDEX idx_experiments_status ON experiments(status);
CREATE INDEX idx_experiments_start_time ON experiments(start_time);
CREATE INDEX idx_runs_experiment_id ON runs(experiment_id);
CREATE INDEX idx_run_metrics_run_id ON run_metrics(run_id);
CREATE INDEX idx_artifacts_run_id ON artifacts(run_id);