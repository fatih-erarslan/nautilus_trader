-- =====================================================
-- Scanner Enhancement Migration v2.0
-- =====================================================
-- Adds missing CRUD operations for endpoints and vulnerabilities
-- Adds scan comparison and metrics capabilities
-- =====================================================

-- Note: The base tables already exist from scanner_schema.sql
-- This migration adds enhancement columns and helper tables

-- =====================================================
-- ENDPOINT ENHANCEMENTS
-- =====================================================

-- Add additional endpoint tracking columns if they don't exist
ALTER TABLE scan_endpoints ADD COLUMN IF NOT EXISTS rate_limit_detected BOOLEAN DEFAULT 0;
ALTER TABLE scan_endpoints ADD COLUMN IF NOT EXISTS cache_headers TEXT;
ALTER TABLE scan_endpoints ADD COLUMN IF NOT EXISTS security_score REAL;
ALTER TABLE scan_endpoints ADD COLUMN IF NOT EXISTS last_modified_at DATETIME;

-- Table: endpoint_versions
-- Purpose: Track endpoint changes over time for comparison
CREATE TABLE IF NOT EXISTS endpoint_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    endpoint_id INTEGER NOT NULL,
    scan_id INTEGER NOT NULL,

    -- Version tracking
    version_number INTEGER NOT NULL DEFAULT 1,
    change_type TEXT CHECK(change_type IN ('created', 'modified', 'deleted', 'unchanged')),

    -- Snapshot of endpoint state
    path TEXT NOT NULL,
    method TEXT NOT NULL,
    status_code INTEGER,
    response_time_ms REAL,
    requires_auth BOOLEAN,

    -- Change details
    changes_detected TEXT, -- JSON array of specific changes
    diff_summary TEXT,

    -- Timestamps
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (endpoint_id) REFERENCES scan_endpoints(id) ON DELETE CASCADE,
    FOREIGN KEY (scan_id) REFERENCES api_scans(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_endpoint_versions_endpoint ON endpoint_versions(endpoint_id);
CREATE INDEX IF NOT EXISTS idx_endpoint_versions_scan ON endpoint_versions(scan_id);

-- =====================================================
-- VULNERABILITY ENHANCEMENTS
-- =====================================================

-- Add additional vulnerability tracking columns
ALTER TABLE scan_vulnerabilities ADD COLUMN IF NOT EXISTS first_detected_scan_id INTEGER;
ALTER TABLE scan_vulnerabilities ADD COLUMN IF NOT EXISTS remediation_priority TEXT DEFAULT 'medium' CHECK(remediation_priority IN ('critical', 'high', 'medium', 'low'));
ALTER TABLE scan_vulnerabilities ADD COLUMN IF NOT EXISTS estimated_fix_time_hours REAL;

-- Table: vulnerability_versions
-- Purpose: Track vulnerability lifecycle and changes
CREATE TABLE IF NOT EXISTS vulnerability_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vulnerability_id INTEGER NOT NULL,
    scan_id INTEGER NOT NULL,

    -- Version tracking
    version_number INTEGER NOT NULL DEFAULT 1,
    change_type TEXT CHECK(change_type IN ('detected', 'modified', 'resolved', 'reappeared')),

    -- Snapshot of vulnerability state
    severity TEXT NOT NULL,
    status TEXT NOT NULL,
    cvss_score REAL,

    -- Change details
    changes_detected TEXT, -- JSON array of changes
    notes TEXT,

    -- Timestamps
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (vulnerability_id) REFERENCES scan_vulnerabilities(id) ON DELETE CASCADE,
    FOREIGN KEY (scan_id) REFERENCES api_scans(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_vulnerability_versions_vuln ON vulnerability_versions(vulnerability_id);
CREATE INDEX IF NOT EXISTS idx_vulnerability_versions_scan ON vulnerability_versions(scan_id);

-- =====================================================
-- SCAN COMPARISON SUPPORT
-- =====================================================

-- Table: scan_comparisons
-- Purpose: Store scan comparison results for analysis
CREATE TABLE IF NOT EXISTS scan_comparisons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Scans being compared
    base_scan_id INTEGER NOT NULL,
    compare_scan_id INTEGER NOT NULL,

    -- Comparison metadata
    comparison_type TEXT DEFAULT 'full' CHECK(comparison_type IN ('full', 'endpoints', 'vulnerabilities', 'performance')),

    -- Results summary
    endpoints_added INTEGER DEFAULT 0,
    endpoints_removed INTEGER DEFAULT 0,
    endpoints_modified INTEGER DEFAULT 0,
    endpoints_unchanged INTEGER DEFAULT 0,

    vulnerabilities_added INTEGER DEFAULT 0,
    vulnerabilities_removed INTEGER DEFAULT 0,
    vulnerabilities_modified INTEGER DEFAULT 0,
    vulnerabilities_unchanged INTEGER DEFAULT 0,

    -- Performance changes
    avg_response_time_change_ms REAL,
    error_rate_change_percent REAL,

    -- Overall scores
    improvement_score REAL, -- -100 to +100, negative = regression, positive = improvement
    security_score_change REAL,
    performance_score_change REAL,

    -- Detailed results
    comparison_data TEXT, -- JSON with detailed comparison

    -- AI analysis
    ai_summary TEXT,
    ai_insights TEXT, -- JSON array
    recommendations TEXT, -- JSON array

    -- Timestamps
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (base_scan_id) REFERENCES api_scans(id) ON DELETE CASCADE,
    FOREIGN KEY (compare_scan_id) REFERENCES api_scans(id) ON DELETE CASCADE,
    UNIQUE(base_scan_id, compare_scan_id)
);

CREATE INDEX IF NOT EXISTS idx_scan_comparisons_base ON scan_comparisons(base_scan_id);
CREATE INDEX IF NOT EXISTS idx_scan_comparisons_compare ON scan_comparisons(compare_scan_id);
CREATE INDEX IF NOT EXISTS idx_scan_comparisons_created ON scan_comparisons(created_at DESC);

-- =====================================================
-- ENHANCED METRICS TABLES
-- =====================================================

-- Table: endpoint_metrics
-- Purpose: Detailed metrics for individual endpoints
CREATE TABLE IF NOT EXISTS endpoint_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    endpoint_id INTEGER NOT NULL,
    scan_id INTEGER NOT NULL,

    -- Performance metrics
    response_time_p50_ms REAL,
    response_time_p95_ms REAL,
    response_time_p99_ms REAL,
    throughput_rps REAL,

    -- Reliability metrics
    success_rate REAL,
    error_rate REAL,
    timeout_rate REAL,

    -- Security metrics
    ssl_grade TEXT,
    security_headers_count INTEGER DEFAULT 0,
    security_score REAL,

    -- Resource usage
    cpu_usage_percent REAL,
    memory_usage_mb REAL,
    bandwidth_mbps REAL,

    -- Timestamps
    measured_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (endpoint_id) REFERENCES scan_endpoints(id) ON DELETE CASCADE,
    FOREIGN KEY (scan_id) REFERENCES api_scans(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_endpoint_metrics_endpoint ON endpoint_metrics(endpoint_id);
CREATE INDEX IF NOT EXISTS idx_endpoint_metrics_scan ON endpoint_metrics(scan_id);

-- =====================================================
-- TRIGGERS FOR ENHANCED FUNCTIONALITY
-- =====================================================

-- Trigger: Create endpoint version on modification
CREATE TRIGGER IF NOT EXISTS trigger_endpoint_version_tracking
AFTER UPDATE ON scan_endpoints
FOR EACH ROW
WHEN NEW.response_time_ms != OLD.response_time_ms
     OR NEW.status_code != OLD.status_code
     OR NEW.requires_auth != OLD.requires_auth
BEGIN
    INSERT INTO endpoint_versions (
        endpoint_id, scan_id, version_number, change_type,
        path, method, status_code, response_time_ms, requires_auth,
        changes_detected
    )
    SELECT
        NEW.id,
        NEW.scan_id,
        COALESCE((SELECT MAX(version_number) + 1 FROM endpoint_versions WHERE endpoint_id = NEW.id), 1),
        'modified',
        NEW.path,
        NEW.method,
        NEW.status_code,
        NEW.response_time_ms,
        NEW.requires_auth,
        json_array(
            CASE WHEN NEW.status_code != OLD.status_code
                THEN json_object('field', 'status_code', 'old', OLD.status_code, 'new', NEW.status_code)
                ELSE NULL END,
            CASE WHEN NEW.response_time_ms != OLD.response_time_ms
                THEN json_object('field', 'response_time_ms', 'old', OLD.response_time_ms, 'new', NEW.response_time_ms)
                ELSE NULL END
        );
END;

-- Trigger: Track vulnerability lifecycle changes
CREATE TRIGGER IF NOT EXISTS trigger_vulnerability_version_tracking
AFTER UPDATE ON scan_vulnerabilities
FOR EACH ROW
WHEN NEW.severity != OLD.severity
     OR NEW.status != OLD.status
     OR NEW.cvss_score != OLD.cvss_score
BEGIN
    INSERT INTO vulnerability_versions (
        vulnerability_id, scan_id, version_number, change_type,
        severity, status, cvss_score,
        changes_detected
    )
    SELECT
        NEW.id,
        NEW.scan_id,
        COALESCE((SELECT MAX(version_number) + 1 FROM vulnerability_versions WHERE vulnerability_id = NEW.id), 1),
        CASE
            WHEN NEW.status = 'resolved' AND OLD.status != 'resolved' THEN 'resolved'
            WHEN NEW.status != 'resolved' AND OLD.status = 'resolved' THEN 'reappeared'
            ELSE 'modified'
        END,
        NEW.severity,
        NEW.status,
        NEW.cvss_score,
        json_array(
            CASE WHEN NEW.severity != OLD.severity
                THEN json_object('field', 'severity', 'old', OLD.severity, 'new', NEW.severity)
                ELSE NULL END,
            CASE WHEN NEW.status != OLD.status
                THEN json_object('field', 'status', 'old', OLD.status, 'new', NEW.status)
                ELSE NULL END
        );
END;

-- =====================================================
-- ENHANCED VIEWS
-- =====================================================

-- View: Endpoint comparison summary
CREATE VIEW IF NOT EXISTS v_endpoint_changes AS
SELECT
    ev.scan_id,
    ev.endpoint_id,
    e.path,
    e.method,
    ev.change_type,
    ev.version_number,
    ev.created_at,
    ev.changes_detected
FROM endpoint_versions ev
JOIN scan_endpoints e ON ev.endpoint_id = e.id
ORDER BY ev.created_at DESC;

-- View: Vulnerability lifecycle
CREATE VIEW IF NOT EXISTS v_vulnerability_lifecycle AS
SELECT
    vv.vulnerability_id,
    v.title,
    v.type,
    vv.scan_id,
    vv.change_type,
    vv.severity,
    vv.status,
    vv.cvss_score,
    vv.version_number,
    vv.created_at
FROM vulnerability_versions vv
JOIN scan_vulnerabilities v ON vv.vulnerability_id = v.id
ORDER BY vv.created_at DESC;

-- View: Scan comparison overview
CREATE VIEW IF NOT EXISTS v_scan_comparison_summary AS
SELECT
    sc.id,
    sc.base_scan_id,
    sc.compare_scan_id,
    bs.name AS base_scan_name,
    cs.name AS compare_scan_name,
    sc.endpoints_added,
    sc.endpoints_removed,
    sc.vulnerabilities_added,
    sc.vulnerabilities_removed,
    sc.improvement_score,
    sc.avg_response_time_change_ms,
    sc.created_at
FROM scan_comparisons sc
JOIN api_scans bs ON sc.base_scan_id = bs.id
JOIN api_scans cs ON sc.compare_scan_id = cs.id
ORDER BY sc.created_at DESC;

-- View: Endpoint performance trends
CREATE VIEW IF NOT EXISTS v_endpoint_performance_trends AS
SELECT
    e.id AS endpoint_id,
    e.path,
    e.method,
    e.scan_id,
    em.response_time_p50_ms,
    em.response_time_p95_ms,
    em.success_rate,
    em.error_rate,
    em.security_score,
    em.measured_at
FROM scan_endpoints e
LEFT JOIN endpoint_metrics em ON e.id = em.endpoint_id
ORDER BY em.measured_at DESC;

-- =====================================================
-- UTILITY FUNCTIONS AND HELPERS
-- =====================================================

-- Update schema version
INSERT OR IGNORE INTO schema_version (version, description) VALUES
('2.0.0', 'Scanner enhancements: endpoint/vulnerability CRUD, comparison, metrics');

-- =====================================================
-- SAMPLE DATA FOR TESTING
-- =====================================================

-- Insert sample comparison
INSERT OR IGNORE INTO scan_comparisons (
    id, base_scan_id, compare_scan_id,
    endpoints_added, endpoints_removed, endpoints_modified,
    vulnerabilities_added, vulnerabilities_removed,
    improvement_score, avg_response_time_change_ms,
    comparison_data, ai_summary
) VALUES (
    1, 1, 1,
    2, 0, 1,
    1, 1,
    15.5, -23.4,
    '{"summary":"Overall improvement detected","details":{}}',
    'API performance improved by 15%. Reduced response time and fixed one critical vulnerability.'
);

-- =====================================================
-- END OF MIGRATION
-- =====================================================
