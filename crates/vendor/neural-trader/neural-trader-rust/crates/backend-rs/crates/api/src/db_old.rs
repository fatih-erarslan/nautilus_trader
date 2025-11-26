use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    pub id: String,
    pub user_id: String,
    pub name: String,
    pub description: Option<String>,
    pub config: Value,
    pub status: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowExecution {
    pub id: String,
    pub workflow_id: String,
    pub user_id: String,
    pub status: String,
    pub execution_time_ms: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stats {
    pub total_workflows: i64,
    pub active_executions: i64,
    pub uptime_seconds: i64,
    pub mock_mode: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiScan {
    pub id: String,
    pub url: String,
    pub scan_type: String,
    pub status: String,
    pub endpoints_found: i64,
    pub vulnerabilities_count: i64,
    pub scan_data: Value,
    pub created_at: String,
    pub updated_at: String,
    pub completed_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanSummary {
    pub id: String,
    pub url: String,
    pub scan_type: String,
    pub status: String,
    pub endpoints_found: i64,
    pub vulnerabilities_count: i64,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScannerStats {
    pub total_scans: i64,
    pub endpoints_discovered: i64,
    pub vulnerabilities_found: i64,
    pub active_scans: i64,
}

pub struct Database {
    db_path: String,
    _conn: Arc<Mutex<()>>, // Placeholder for actual connection
}

impl Database {
    pub fn new(db_path: String) -> Result<Self> {
        Ok(Self {
            db_path,
            _conn: Arc::new(Mutex::new(())),
        })
    }

    pub async fn get_stats(&self) -> Result<Stats> {
        // Use rusqlite directly for simple queries
        let conn = rusqlite::Connection::open(&self.db_path)?;

        let total_workflows: i64 = conn.query_row(
            "SELECT COUNT(*) FROM workflows",
            [],
            |row| row.get(0),
        )?;

        let active_executions: i64 = conn.query_row(
            "SELECT COUNT(*) FROM workflow_executions WHERE status IN ('pending', 'running')",
            [],
            |row| row.get(0),
        )?;

        Ok(Stats {
            total_workflows,
            active_executions,
            uptime_seconds: 0, // Will be calculated in the handler
            mock_mode: false,
        })
    }

    pub async fn get_workflows(&self) -> Result<Vec<Workflow>> {
        let conn = rusqlite::Connection::open(&self.db_path)?;

        let mut stmt = conn.prepare(
            "SELECT id, user_id, name, description, config, status, created_at, updated_at FROM workflows ORDER BY created_at DESC"
        )?;

        let workflows = stmt
            .query_map([], |row| {
                Ok(Workflow {
                    id: row.get(0)?,
                    user_id: row.get(1)?,
                    name: row.get(2)?,
                    description: row.get(3)?,
                    config: serde_json::from_str(&row.get::<_, String>(4)?).unwrap_or(Value::Object(Default::default())),
                    status: row.get(5)?,
                    created_at: row.get(6)?,
                    updated_at: row.get(7)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(workflows)
    }

    pub async fn create_workflow(&self, user_id: &str, name: &str, description: Option<&str>, config: Value) -> Result<Workflow> {
        let conn = rusqlite::Connection::open(&self.db_path)?;
        let id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        let config_str = serde_json::to_string(&config)?;

        conn.execute(
            "INSERT INTO workflows (id, user_id, name, description, config, status, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, 'draft', ?6, ?7)",
            rusqlite::params![id, user_id, name, description, config_str, now, now],
        )?;

        Ok(Workflow {
            id: id.clone(),
            user_id: user_id.to_string(),
            name: name.to_string(),
            description: description.map(|s| s.to_string()),
            config,
            status: "draft".to_string(),
            created_at: now.clone(),
            updated_at: now,
        })
    }

    pub async fn create_execution(&self, workflow_id: &str, user_id: &str) -> Result<WorkflowExecution> {
        let conn = rusqlite::Connection::open(&self.db_path)?;
        let id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().to_rfc3339();

        conn.execute(
            "INSERT INTO workflow_executions (id, workflow_id, user_id, status, started_at)
             VALUES (?1, ?2, ?3, 'running', ?4)",
            rusqlite::params![id, workflow_id, user_id, now],
        )?;

        Ok(WorkflowExecution {
            id: id.clone(),
            workflow_id: workflow_id.to_string(),
            user_id: user_id.to_string(),
            status: "running".to_string(),
            execution_time_ms: None,
        })
    }

    // Scanner operations
    pub async fn create_scan(&self, url: &str, scan_type: &str, options: Value) -> Result<ApiScan> {
        let conn = rusqlite::Connection::open(&self.db_path)?;
        let id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().to_rfc3339();

        let scan_data = serde_json::json!({
            "options": options,
            "endpoints": [],
            "vulnerabilities": [],
            "metrics": {}
        });
        let scan_data_str = serde_json::to_string(&scan_data)?;

        // Use existing scanner schema with correct column names
        conn.execute(
            "INSERT INTO api_scans (id, user_id, name, url, base_url, scan_type, status, total_endpoints, total_vulnerabilities, created_at, updated_at)
             VALUES (?1, 'user-1', ?2, ?3, ?4, ?5, 'queued', 0, 0, ?6, ?7)",
            rusqlite::params![id, format!("Scan: {}", url), url, url, scan_type, now, now],
        )?;

        Ok(ApiScan {
            id: id.clone(),
            url: url.to_string(),
            scan_type: scan_type.to_string(),
            status: "queued".to_string(),
            endpoints_found: 0,
            vulnerabilities_count: 0,
            scan_data,
            created_at: now.clone(),
            updated_at: now,
            completed_at: None,
        })
    }

    pub async fn get_scans(&self, page: i64, limit: i64, status_filter: Option<&str>) -> Result<Vec<ScanSummary>> {
        let conn = rusqlite::Connection::open(&self.db_path)?;

        // Validate pagination parameters to prevent abuse
        let page = page.max(1);
        let limit = limit.clamp(1, 100); // Maximum 100 results per page
        let offset = (page - 1) * limit;

        let scans = if let Some(status) = status_filter {
            // Use parameterized query to prevent SQL injection
            let mut stmt = conn.prepare(
                "SELECT id, url, scan_type, status, total_endpoints, total_vulnerabilities, created_at
                 FROM api_scans
                 WHERE status = ?1
                 ORDER BY created_at DESC
                 LIMIT ?2 OFFSET ?3"
            )?;

            stmt.query_map(rusqlite::params![status, limit, offset], |row| {
                Ok(ScanSummary {
                    id: row.get(0)?,
                    url: row.get(1)?,
                    scan_type: row.get(2)?,
                    status: row.get(3)?,
                    endpoints_found: row.get(4)?,
                    vulnerabilities_count: row.get(5)?,
                    created_at: row.get(6)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            // Use parameterized query without status filter
            let mut stmt = conn.prepare(
                "SELECT id, url, scan_type, status, total_endpoints, total_vulnerabilities, created_at
                 FROM api_scans
                 ORDER BY created_at DESC
                 LIMIT ?1 OFFSET ?2"
            )?;

            stmt.query_map(rusqlite::params![limit, offset], |row| {
                Ok(ScanSummary {
                    id: row.get(0)?,
                    url: row.get(1)?,
                    scan_type: row.get(2)?,
                    status: row.get(3)?,
                    endpoints_found: row.get(4)?,
                    vulnerabilities_count: row.get(5)?,
                    created_at: row.get(6)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?
        };

        Ok(scans)
    }

    pub async fn get_scan(&self, scan_id: &str) -> Result<ApiScan> {
        let conn = rusqlite::Connection::open(&self.db_path)?;

        let scan = conn.query_row(
            "SELECT id, url, scan_type, status, total_endpoints, total_vulnerabilities, created_at, updated_at, completed_at
             FROM api_scans WHERE id = ?1",
            [scan_id],
            |row| {
                Ok(ApiScan {
                    id: row.get(0)?,
                    url: row.get(1)?,
                    scan_type: row.get(2)?,
                    status: row.get(3)?,
                    endpoints_found: row.get(4)?,
                    vulnerabilities_count: row.get(5)?,
                    scan_data: serde_json::json!({}),
                    created_at: row.get(6)?,
                    updated_at: row.get(7)?,
                    completed_at: row.get(8)?,
                })
            },
        )?;

        Ok(scan)
    }

    pub async fn update_scan_status(&self, scan_id: &str, status: &str, scan_data: Option<Value>) -> Result<()> {
        let conn = rusqlite::Connection::open(&self.db_path)?;
        let now = chrono::Utc::now().to_rfc3339();

        if let Some(data) = scan_data {
            let endpoints_found = data.get("endpoints").and_then(|e| e.as_array()).map(|a| a.len() as i64).unwrap_or(0);
            let vulnerabilities_count = data.get("vulnerabilities").and_then(|v| v.as_array()).map(|a| a.len() as i64).unwrap_or(0);

            if status == "completed" {
                conn.execute(
                    "UPDATE api_scans SET status = ?1, total_endpoints = ?2, total_vulnerabilities = ?3, updated_at = ?4, completed_at = ?5 WHERE id = ?6",
                    rusqlite::params![status, endpoints_found, vulnerabilities_count, now, now, scan_id],
                )?;
            } else {
                conn.execute(
                    "UPDATE api_scans SET status = ?1, total_endpoints = ?2, total_vulnerabilities = ?3, updated_at = ?4 WHERE id = ?5",
                    rusqlite::params![status, endpoints_found, vulnerabilities_count, now, scan_id],
                )?;
            }
        } else {
            conn.execute(
                "UPDATE api_scans SET status = ?1, updated_at = ?2 WHERE id = ?3",
                rusqlite::params![status, now, scan_id],
            )?;
        }

        Ok(())
    }

    pub async fn update_scan_endpoints(&self, scan_id: &str, endpoints_count: i64) -> Result<()> {
        let conn = rusqlite::Connection::open(&self.db_path)?;
        let now = chrono::Utc::now().to_rfc3339();

        conn.execute(
            "UPDATE api_scans SET total_endpoints = ?1, updated_at = ?2 WHERE id = ?3",
            rusqlite::params![endpoints_count, now, scan_id],
        )?;

        Ok(())
    }

    pub async fn delete_scan(&self, scan_id: &str) -> Result<()> {
        let conn = rusqlite::Connection::open(&self.db_path)?;
        conn.execute("DELETE FROM api_scans WHERE id = ?1", [scan_id])?;
        Ok(())
    }

    pub async fn get_scanner_stats(&self) -> Result<ScannerStats> {
        let conn = rusqlite::Connection::open(&self.db_path)?;

        let total_scans: i64 = conn.query_row(
            "SELECT COUNT(*) FROM api_scans",
            [],
            |row| row.get(0),
        ).unwrap_or(0);

        let endpoints_discovered: i64 = conn.query_row(
            "SELECT COALESCE(SUM(endpoints_found), 0) FROM api_scans",
            [],
            |row| row.get(0),
        ).unwrap_or(0);

        let vulnerabilities_found: i64 = conn.query_row(
            "SELECT COALESCE(SUM(vulnerabilities_count), 0) FROM api_scans",
            [],
            |row| row.get(0),
        ).unwrap_or(0);

        let active_scans: i64 = conn.query_row(
            "SELECT COUNT(*) FROM api_scans WHERE status IN ('queued', 'running')",
            [],
            |row| row.get(0),
        ).unwrap_or(0);

        Ok(ScannerStats {
            total_scans,
            endpoints_discovered,
            vulnerabilities_found,
            active_scans,
        })
    }
}
