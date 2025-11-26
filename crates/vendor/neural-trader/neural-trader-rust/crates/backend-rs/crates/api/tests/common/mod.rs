//! Common test utilities and fixtures for Rust backend tests
//!
//! Provides shared test infrastructure including:
//! - Test database setup and teardown
//! - Mock data generators
//! - HTTP client helpers
//! - Authentication helpers

use anyhow::Result;
use serde_json::{json, Value};
use std::path::PathBuf;
use tempfile::{tempdir, TempDir};

/// Test database helper
pub struct TestDb {
    pub path: PathBuf,
    _temp_dir: TempDir,
}

impl TestDb {
    /// Create a new temporary test database
    pub fn new() -> Result<Self> {
        let temp_dir = tempdir()?;
        let db_path = temp_dir.path().join("test.db");

        // Initialize database schema
        let conn = rusqlite::Connection::open(&db_path)?;
        Self::init_schema(&conn)?;
        Self::seed_data(&conn)?;

        Ok(Self {
            path: db_path,
            _temp_dir: temp_dir,
        })
    }

    fn init_schema(conn: &rusqlite::Connection) -> Result<()> {
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS workflows (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                config TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS workflow_executions (
                id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                status TEXT NOT NULL,
                execution_time_ms INTEGER,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                FOREIGN KEY (workflow_id) REFERENCES workflows(id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS api_scans (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                url TEXT NOT NULL,
                base_url TEXT NOT NULL,
                scan_type TEXT NOT NULL,
                status TEXT NOT NULL,
                total_endpoints INTEGER DEFAULT 0,
                total_vulnerabilities INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS scan_endpoints (
                id TEXT PRIMARY KEY,
                scan_id TEXT NOT NULL,
                path TEXT NOT NULL,
                method TEXT NOT NULL,
                response_time_ms INTEGER,
                status_code INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY (scan_id) REFERENCES api_scans(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS scan_vulnerabilities (
                id TEXT PRIMARY KEY,
                scan_id TEXT NOT NULL,
                endpoint_id TEXT,
                severity TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                recommendation TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (scan_id) REFERENCES api_scans(id) ON DELETE CASCADE,
                FOREIGN KEY (endpoint_id) REFERENCES scan_endpoints(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS analytics_events (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                event_type TEXT NOT NULL,
                event_data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE INDEX idx_scans_user ON api_scans(user_id);
            CREATE INDEX idx_scans_status ON api_scans(status);
            CREATE INDEX idx_endpoints_scan ON scan_endpoints(scan_id);
            CREATE INDEX idx_vulnerabilities_scan ON scan_vulnerabilities(scan_id);
            CREATE INDEX idx_events_user ON analytics_events(user_id);
            CREATE INDEX idx_events_type ON analytics_events(event_type);
            "#,
        )?;

        Ok(())
    }

    fn seed_data(conn: &rusqlite::Connection) -> Result<()> {
        let now = chrono::Utc::now().to_rfc3339();

        // Seed test users
        conn.execute(
            "INSERT INTO users (id, email, password_hash, role, created_at, updated_at)
             VALUES ('user-1', 'test@example.com', '$2b$12$hashedpassword', 'admin', ?1, ?2)",
            rusqlite::params![now, now],
        )?;

        conn.execute(
            "INSERT INTO users (id, email, password_hash, role, created_at, updated_at)
             VALUES ('user-2', 'user@example.com', '$2b$12$hashedpassword2', 'user', ?1, ?2)",
            rusqlite::params![now, now],
        )?;

        // Seed test workflows
        conn.execute(
            "INSERT INTO workflows (id, user_id, name, description, config, status, created_at, updated_at)
             VALUES ('wf-1', 'user-1', 'Test Workflow', 'Test description', '{}', 'active', ?1, ?2)",
            rusqlite::params![now, now],
        )?;

        // Seed test scans
        conn.execute(
            "INSERT INTO api_scans (id, user_id, name, url, base_url, scan_type, status, total_endpoints, total_vulnerabilities, created_at, updated_at)
             VALUES ('scan-1', 'user-1', 'Test Scan', 'https://api.example.com', 'https://api.example.com', 'openapi', 'completed', 5, 2, ?1, ?2)",
            rusqlite::params![now, now],
        )?;

        Ok(())
    }

    /// Get database path as string
    pub fn path_str(&self) -> String {
        self.path.to_str().unwrap().to_string()
    }

    /// Get a connection to the test database
    pub fn conn(&self) -> Result<rusqlite::Connection> {
        Ok(rusqlite::Connection::open(&self.path)?)
    }
}

/// Mock JWT token generator
pub struct MockAuth;

impl MockAuth {
    pub fn generate_token(user_id: &str, role: &str) -> String {
        format!("mock_token_{}_{}", user_id, role)
    }

    pub fn admin_token() -> String {
        Self::generate_token("admin", "admin")
    }

    pub fn user_token() -> String {
        Self::generate_token("user-1", "user")
    }
}

/// Mock data generators
pub struct MockData;

impl MockData {
    pub fn openapi_spec() -> Value {
        json!({
            "openapi": "3.0.0",
            "info": {
                "title": "Test API",
                "version": "1.0.0"
            },
            "paths": {
                "/users": {
                    "get": {
                        "summary": "List users",
                        "responses": {
                            "200": {
                                "description": "Success"
                            }
                        }
                    },
                    "post": {
                        "summary": "Create user",
                        "requestBody": {
                            "required": true,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "email": {"type": "string"},
                                            "password": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        },
                        "responses": {
                            "201": {"description": "Created"}
                        }
                    }
                },
                "/users/{id}": {
                    "get": {
                        "parameters": [
                            {
                                "name": "id",
                                "in": "path",
                                "required": true,
                                "schema": {"type": "string"}
                            }
                        ],
                        "responses": {
                            "200": {"description": "Success"}
                        }
                    }
                }
            }
        })
    }

    pub fn scan_request() -> Value {
        json!({
            "url": "https://api.example.com/openapi.json",
            "scan_type": "openapi",
            "options": {
                "deep_scan": true,
                "check_auth": true
            }
        })
    }

    pub fn workflow_data() -> Value {
        json!({
            "name": "Test Workflow",
            "description": "A test workflow for scanning",
            "config": {
                "steps": [
                    {"type": "scan", "url": "https://api.example.com"},
                    {"type": "analyze", "model": "gpt-4"}
                ]
            }
        })
    }
}

/// SQL injection test payloads
pub struct SecurityPayloads;

impl SecurityPayloads {
    pub fn sql_injection_tests() -> Vec<String> {
        vec![
            "' OR '1'='1".to_string(),
            "'; DROP TABLE users; --".to_string(),
            "1' UNION SELECT * FROM users--".to_string(),
            "admin'--".to_string(),
            "' OR 1=1--".to_string(),
        ]
    }

    pub fn xss_tests() -> Vec<String> {
        vec![
            "<script>alert('XSS')</script>".to_string(),
            "<img src=x onerror=alert('XSS')>".to_string(),
            "javascript:alert('XSS')".to_string(),
            "<svg onload=alert('XSS')>".to_string(),
        ]
    }

    pub fn path_traversal_tests() -> Vec<String> {
        vec![
            "../../../etc/passwd".to_string(),
            "..\\..\\..\\windows\\system32\\config\\sam".to_string(),
            "....//....//....//etc/passwd".to_string(),
        ]
    }
}

/// Performance test helpers
pub struct PerfHelpers;

impl PerfHelpers {
    /// Generate large dataset for stress testing
    pub fn generate_scans(count: usize) -> Vec<Value> {
        (0..count)
            .map(|i| {
                json!({
                    "url": format!("https://api{}.example.com", i),
                    "scan_type": "openapi",
                    "options": {}
                })
            })
            .collect()
    }

    /// Generate concurrent requests
    pub fn concurrent_requests(count: usize, endpoint: &str) -> Vec<String> {
        vec![endpoint.to_string(); count]
    }
}
