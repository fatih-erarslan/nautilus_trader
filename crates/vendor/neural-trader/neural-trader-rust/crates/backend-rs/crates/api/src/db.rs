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

#[derive(Clone)]
pub struct Database {
    pub db_path: String,
    _conn: Arc<Mutex<()>>, // Placeholder for actual connection
}

impl Database {
    pub fn new(db_path: String) -> Result<Self> {
        // Run migrations on initialization
        let conn = rusqlite::Connection::open(&db_path)?;
        Self::run_migrations_sync(&conn)?;
        Ok(Self {
            db_path,
            _conn: Arc::new(Mutex::new(())),
        })
    }

    fn run_migrations_sync(conn: &rusqlite::Connection) -> Result<()> {
        // Check if agent_deployments table exists
        let table_exists: bool = conn.query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='agent_deployments'",
            [],
            |row| row.get(0)
        ).unwrap_or(0) > 0;

        if !table_exists {
            // Run migration from 004_agent_deployment.sql
            let migration_sql = include_str!("../../../migrations/004_agent_deployment.sql");
            conn.execute_batch(migration_sql)?;
        }

        Ok(())
    }

    pub async fn run_migrations(&self) -> Result<()> {
        let conn = rusqlite::Connection::open(&self.db_path)?;
        Self::run_migrations_sync(&conn)?;
        Ok(())
    }

    // Agent deployment CRUD operations
    pub async fn list_all_agents(&self) -> Result<Vec<serde_json::Value>> {
        let conn = rusqlite::Connection::open(&self.db_path)?;
        let mut stmt = conn.prepare(
            "SELECT id, name, agent_type, status, sandbox_id, task_description,
                    result, error_message, tokens_used, cost_usd, execution_time_ms,
                    created_at, completed_at
             FROM agent_deployments ORDER BY created_at DESC"
        )?;

        let agents = stmt.query_map([], |row| {
            Ok(serde_json::json!({
                "id": row.get::<_, String>(0)?,
                "name": row.get::<_, String>(1)?,
                "agent_type": row.get::<_, String>(2)?,
                "status": row.get::<_, String>(3)?,
                "sandbox_id": row.get::<_, Option<String>>(4)?,
                "task_description": row.get::<_, String>(5)?,
                "result": row.get::<_, Option<String>>(6)?
                    .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok()),
                "error_message": row.get::<_, Option<String>>(7)?,
                "tokens_used": row.get::<_, i32>(8)?,
                "cost_usd": row.get::<_, f64>(9)?,
                "execution_time_ms": row.get::<_, i32>(10)?,
                "created_at": row.get::<_, String>(11)?,
                "completed_at": row.get::<_, Option<String>>(12)?
            }))
        })?.collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(agents)
    }

    pub async fn get_agent_deployment(&self, agent_id: &str) -> Result<serde_json::Value> {
        let conn = rusqlite::Connection::open(&self.db_path)?;
        let row = conn.query_row(
            "SELECT id, name, agent_type, status, sandbox_id, task_description,
                    result, error_message, tokens_used, cost_usd, execution_time_ms,
                    created_at, completed_at
             FROM agent_deployments WHERE id = ?1",
            [agent_id],
            |row| {
                Ok(serde_json::json!({
                    "id": row.get::<_, String>(0)?,
                    "name": row.get::<_, String>(1)?,
                    "agent_type": row.get::<_, String>(2)?,
                    "status": row.get::<_, String>(3)?,
                    "sandbox_id": row.get::<_, Option<String>>(4)?,
                    "task_description": row.get::<_, String>(5)?,
                    "result": row.get::<_, Option<String>>(6)?
                        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok()),
                    "error_message": row.get::<_, Option<String>>(7)?,
                    "tokens_used": row.get::<_, i32>(8)?,
                    "cost_usd": row.get::<_, f64>(9)?,
                    "execution_time_ms": row.get::<_, i32>(10)?,
                    "created_at": row.get::<_, String>(11)?,
                    "completed_at": row.get::<_, Option<String>>(12)?
                }))
            }
        )?;
        Ok(row)
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

        // Validate and sanitize pagination parameters
        let page = page.max(1);
        let limit = limit.clamp(1, 100);
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

            let rows = stmt.query_map(rusqlite::params![status, limit, offset], |row| {
                Ok(ScanSummary {
                    id: row.get(0)?,
                    url: row.get(1)?,
                    scan_type: row.get(2)?,
                    status: row.get(3)?,
                    endpoints_found: row.get(4)?,
                    vulnerabilities_count: row.get(5)?,
                    created_at: row.get(6)?,
                })
            })?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            // Use parameterized query without status filter
            let mut stmt = conn.prepare(
                "SELECT id, url, scan_type, status, total_endpoints, total_vulnerabilities, created_at
                 FROM api_scans
                 ORDER BY created_at DESC
                 LIMIT ?1 OFFSET ?2"
            )?;

            let rows = stmt.query_map(rusqlite::params![limit, offset], |row| {
                Ok(ScanSummary {
                    id: row.get(0)?,
                    url: row.get(1)?,
                    scan_type: row.get(2)?,
                    status: row.get(3)?,
                    endpoints_found: row.get(4)?,
                    vulnerabilities_count: row.get(5)?,
                    created_at: row.get(6)?,
                })
            })?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
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

    // ============ Analytics Operations ============

    pub async fn get_dashboard_stats(&self, timeframe: &str) -> Result<crate::analytics::DashboardStats> {
        let conn = rusqlite::Connection::open(&self.db_path)?;

        // Calculate time filter based on timeframe
        let hours_back = match timeframe {
            "24h" => 24,
            "7d" => 24 * 7,
            "30d" => 24 * 30,
            _ => 24,
        };

        let time_filter = format!("datetime('now', '-{} hours')", hours_back);

        let total_api_calls: i64 = conn.query_row(
            &format!("SELECT COUNT(*) FROM analytics_events WHERE event_category = 'usage' AND created_at >= {}", time_filter),
            [],
            |row| row.get(0),
        ).unwrap_or(0);

        let total_workflows: i64 = conn.query_row(
            "SELECT COUNT(*) FROM workflows",
            [],
            |row| row.get(0),
        ).unwrap_or(0);

        let total_scans: i64 = conn.query_row(
            &format!("SELECT COUNT(*) FROM api_scans WHERE created_at >= {}", time_filter),
            [],
            |row| row.get(0),
        ).unwrap_or(0);

        let active_users: i64 = conn.query_row(
            &format!("SELECT COUNT(DISTINCT user_id) FROM analytics_events WHERE created_at >= {}", time_filter),
            [],
            |row| row.get(0),
        ).unwrap_or(1);

        let avg_response_time: f64 = conn.query_row(
            &format!("SELECT AVG(value) FROM performance_metrics WHERE metric_type = 'api_latency' AND recorded_at >= {}", time_filter),
            [],
            |row| row.get(0),
        ).unwrap_or(50.0);

        let total_events: i64 = conn.query_row(
            &format!("SELECT COUNT(*) FROM analytics_events WHERE created_at >= {}", time_filter),
            [],
            |row| row.get(0),
        ).unwrap_or(0);

        let success_count: i64 = conn.query_row(
            &format!("SELECT COUNT(*) FROM analytics_events WHERE status = 'success' AND created_at >= {}", time_filter),
            [],
            |row| row.get(0),
        ).unwrap_or(0);

        let error_rate = if total_events > 0 {
            (total_events - success_count) as f64 / total_events as f64 * 100.0
        } else {
            0.0
        };

        // Get recent activity
        let mut stmt = conn.prepare(
            &format!("SELECT id, user_id, action, entity_type, entity_name, description, severity, created_at
                     FROM activity_log
                     WHERE created_at >= {}
                     ORDER BY created_at DESC
                     LIMIT 10", time_filter)
        )?;

        let recent_activity = stmt.query_map([], |row| {
            Ok(crate::analytics::ActivitySummary {
                id: row.get(0)?,
                user_id: row.get(1)?,
                action: row.get(2)?,
                entity_type: row.get(3)?,
                entity_name: row.get(4)?,
                description: row.get(5)?,
                severity: row.get(6)?,
                created_at: row.get(7)?,
            })
        })?.collect::<std::result::Result<Vec<_>, _>>()?;

        // Calculate performance percentiles (simplified)
        let mut latencies: Vec<f64> = Vec::new();
        let mut metric_stmt = conn.prepare(
            &format!("SELECT value FROM performance_metrics WHERE metric_type = 'api_latency' AND recorded_at >= {} ORDER BY value", time_filter)
        )?;

        let latency_rows = metric_stmt.query_map([], |row| row.get::<_, f64>(0))?;
        for latency in latency_rows {
            latencies.push(latency?);
        }

        let (p50, p95, p99) = if !latencies.is_empty() {
            let len = latencies.len();
            let p50_idx = len / 2;
            let p95_idx = (len * 95) / 100;
            let p99_idx = (len * 99) / 100;
            (
                latencies.get(p50_idx).copied().unwrap_or(avg_response_time),
                latencies.get(p95_idx).copied().unwrap_or(avg_response_time * 1.5),
                latencies.get(p99_idx).copied().unwrap_or(avg_response_time * 2.0),
            )
        } else {
            (avg_response_time, avg_response_time * 1.5, avg_response_time * 2.0)
        };

        let success_rate = if total_events > 0 {
            success_count as f64 / total_events as f64 * 100.0
        } else {
            100.0
        };

        Ok(crate::analytics::DashboardStats {
            total_api_calls,
            total_workflows,
            total_scans,
            active_users,
            avg_response_time_ms: avg_response_time,
            error_rate,
            recent_activity,
            performance_metrics: crate::analytics::PerformanceSnapshot {
                api_latency_p50: p50,
                api_latency_p95: p95,
                api_latency_p99: p99,
                total_events,
                success_rate,
            },
            timeframe: timeframe.to_string(),
        })
    }

    pub async fn get_usage_analytics(&self, user_id: &str, start_date: Option<&str>, end_date: Option<&str>) -> Result<crate::analytics::UsageReport> {
        let conn = rusqlite::Connection::open(&self.db_path)?;

        let start = start_date.unwrap_or_else(|| "date('now', '-30 days')");
        let end = end_date.unwrap_or_else(|| "date('now')");

        let mut stmt = conn.prepare(
            &format!("SELECT date, api_calls_count, workflows_executed, scans_performed, success_count, error_count
                     FROM usage_statistics
                     WHERE user_id = ?1 AND date >= {} AND date <= {}
                     ORDER BY date DESC", start, end)
        )?;

        let daily_breakdown = stmt.query_map([user_id], |row| {
            Ok(crate::analytics::DailyUsage {
                date: row.get(0)?,
                api_calls: row.get(1)?,
                workflows: row.get(2)?,
                scans: row.get(3)?,
                success_count: row.get(4)?,
                error_count: row.get(5)?,
            })
        })?.collect::<std::result::Result<Vec<_>, _>>()?;

        let total_api_calls: i64 = daily_breakdown.iter().map(|d| d.api_calls).sum();
        let total_workflows: i64 = daily_breakdown.iter().map(|d| d.workflows).sum();
        let total_scans: i64 = daily_breakdown.iter().map(|d| d.scans).sum();
        let total_success: i64 = daily_breakdown.iter().map(|d| d.success_count).sum();
        let total_errors: i64 = daily_breakdown.iter().map(|d| d.error_count).sum();

        let success_rate = if total_success + total_errors > 0 {
            total_success as f64 / (total_success + total_errors) as f64 * 100.0
        } else {
            100.0
        };

        let total_execution_time: i64 = conn.query_row(
            "SELECT COALESCE(SUM(total_execution_time_ms), 0) FROM usage_statistics WHERE user_id = ?1",
            [user_id],
            |row| row.get(0),
        ).unwrap_or(0);

        Ok(crate::analytics::UsageReport {
            user_id: user_id.to_string(),
            period: format!("{} to {}", start, end),
            total_api_calls,
            total_workflows,
            total_scans,
            total_execution_time_ms: total_execution_time,
            success_rate,
            daily_breakdown,
        })
    }

    pub async fn get_activity_feed(
        &self,
        user_id: Option<&str>,
        page: i64,
        limit: i64,
        action_filter: Option<&str>,
        entity_type_filter: Option<&str>,
    ) -> Result<Vec<crate::analytics::ActivitySummary>> {
        let conn = rusqlite::Connection::open(&self.db_path)?;

        let mut query = "SELECT id, user_id, action, entity_type, entity_name, description, severity, created_at FROM activity_log WHERE 1=1".to_string();

        if let Some(uid) = user_id {
            query.push_str(&format!(" AND user_id = '{}'", uid));
        }
        if let Some(action) = action_filter {
            query.push_str(&format!(" AND action = '{}'", action));
        }
        if let Some(entity_type) = entity_type_filter {
            query.push_str(&format!(" AND entity_type = '{}'", entity_type));
        }

        query.push_str(&format!(" ORDER BY created_at DESC LIMIT {} OFFSET {}", limit, (page - 1) * limit));

        let mut stmt = conn.prepare(&query)?;

        let activities = stmt.query_map([], |row| {
            Ok(crate::analytics::ActivitySummary {
                id: row.get(0)?,
                user_id: row.get(1)?,
                action: row.get(2)?,
                entity_type: row.get(3)?,
                entity_name: row.get(4)?,
                description: row.get(5)?,
                severity: row.get(6)?,
                created_at: row.get(7)?,
            })
        })?.collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(activities)
    }

    pub async fn log_activity(
        &self,
        user_id: &str,
        action: &str,
        entity_type: &str,
        entity_id: Option<&str>,
        entity_name: Option<&str>,
        description: &str,
        severity: Option<&str>,
        metadata: Option<serde_json::Value>,
    ) -> Result<String> {
        let conn = rusqlite::Connection::open(&self.db_path)?;
        let id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        let severity = severity.unwrap_or("info");

        let changes = metadata.map(|m| serde_json::to_string(&m).unwrap_or_default());

        conn.execute(
            "INSERT INTO activity_log (id, user_id, action, entity_type, entity_id, entity_name, description, severity, changes, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            rusqlite::params![id, user_id, action, entity_type, entity_id, entity_name, description, severity, changes, now],
        )?;

        Ok(id)
    }

    pub async fn get_performance_metrics(&self, timeframe: &str) -> Result<serde_json::Value> {
        let conn = rusqlite::Connection::open(&self.db_path)?;

        let hours_back = match timeframe {
            "24h" => 24,
            "7d" => 24 * 7,
            "30d" => 24 * 30,
            _ => 24,
        };

        let time_filter = format!("datetime('now', '-{} hours')", hours_back);

        let mut stmt = conn.prepare(
            &format!("SELECT metric_type, metric_name, AVG(value) as avg_value, MIN(value) as min_value, MAX(value) as max_value, COUNT(*) as count
                     FROM performance_metrics
                     WHERE recorded_at >= {}
                     GROUP BY metric_type, metric_name
                     ORDER BY metric_type, metric_name", time_filter)
        )?;

        let metrics = stmt.query_map([], |row| {
            Ok(serde_json::json!({
                "metric_type": row.get::<_, String>(0)?,
                "metric_name": row.get::<_, String>(1)?,
                "avg_value": row.get::<_, f64>(2)?,
                "min_value": row.get::<_, f64>(3)?,
                "max_value": row.get::<_, f64>(4)?,
                "count": row.get::<_, i64>(5)?,
            }))
        })?.collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(serde_json::json!({
            "timeframe": timeframe,
            "metrics": metrics,
            "total_metrics": metrics.len()
        }))
    }
}

// =====================================================
// ENDPOINT CRUD OPERATIONS
// =====================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanEndpoint {
    pub id: String,
    pub scan_id: String,
    pub path: String,
    pub full_url: String,
    pub method: String,
    pub status_code: Option<i64>,
    pub response_time_ms: Option<f64>,
    pub requires_auth: bool,
    pub content_type: Option<String>,
    pub discovery_method: Option<String>,
    pub first_seen_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointCreate {
    pub scan_id: String,
    pub path: String,
    pub full_url: String,
    pub method: String,
    pub status_code: Option<i64>,
    pub response_time_ms: Option<f64>,
    pub requires_auth: Option<bool>,
    pub content_type: Option<String>,
    pub discovery_method: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointUpdate {
    pub status_code: Option<i64>,
    pub response_time_ms: Option<f64>,
    pub requires_auth: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanVulnerability {
    pub id: String,
    pub scan_id: String,
    pub vuln_type: String,
    pub category: String,
    pub severity: String,
    pub cvss_score: Option<f64>,
    pub title: String,
    pub description: String,
    pub recommendation: String,
    pub status: String,
    pub detected_at: String,
    pub affected_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulnerabilityCreate {
    pub scan_id: String,
    pub vuln_type: String,
    pub category: String,
    pub severity: String,
    pub cvss_score: Option<f64>,
    pub title: String,
    pub description: String,
    pub recommendation: String,
    pub affected_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulnerabilityUpdate {
    pub status: Option<String>,
    pub severity: Option<String>,
    pub cvss_score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanComparison {
    pub id: String,
    pub base_scan_id: String,
    pub compare_scan_id: String,
    pub endpoints_added: i64,
    pub endpoints_removed: i64,
    pub endpoints_modified: i64,
    pub vulnerabilities_added: i64,
    pub vulnerabilities_removed: i64,
    pub improvement_score: f64,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedScanMetrics {
    pub scan_id: String,
    pub total_endpoints: i64,
    pub total_vulnerabilities: i64,
    pub avg_response_time_ms: f64,
    pub critical_vulnerabilities: i64,
    pub high_vulnerabilities: i64,
    pub medium_vulnerabilities: i64,
    pub low_vulnerabilities: i64,
    pub calculated_at: String,
}

impl Database {
    pub async fn get_scan_endpoints(&self, scan_id: &str) -> Result<Vec<ScanEndpoint>> {
        let conn = rusqlite::Connection::open(&self.db_path)?;
        let mut stmt = conn.prepare(
            "SELECT id, scan_id, path, full_url, method, status_code, response_time_ms,
                    requires_auth, content_type, discovery_method, first_seen_at
             FROM scan_endpoints WHERE scan_id = ?1 ORDER BY path, method"
        )?;
        let endpoints = stmt.query_map([scan_id], |row| {
            Ok(ScanEndpoint {
                id: row.get(0)?, scan_id: row.get(1)?, path: row.get(2)?, full_url: row.get(3)?,
                method: row.get(4)?, status_code: row.get(5)?, response_time_ms: row.get(6)?,
                requires_auth: row.get(7)?, content_type: row.get(8)?, discovery_method: row.get(9)?,
                first_seen_at: row.get(10)?,
            })
        })?.collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(endpoints)
    }

    pub async fn create_endpoint(&self, endpoint: &EndpointCreate) -> Result<ScanEndpoint> {
        let conn = rusqlite::Connection::open(&self.db_path)?;
        let id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO scan_endpoints (id, scan_id, path, full_url, method, status_code,
                                        response_time_ms, requires_auth, content_type,
                                        discovery_method, first_seen_at, last_tested_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            rusqlite::params![id, endpoint.scan_id, endpoint.path, endpoint.full_url,
                endpoint.method, endpoint.status_code, endpoint.response_time_ms,
                endpoint.requires_auth.unwrap_or(false), endpoint.content_type,
                endpoint.discovery_method, now, now],
        )?;
        Ok(ScanEndpoint {
            id: id.clone(), scan_id: endpoint.scan_id.clone(), path: endpoint.path.clone(),
            full_url: endpoint.full_url.clone(), method: endpoint.method.clone(),
            status_code: endpoint.status_code, response_time_ms: endpoint.response_time_ms,
            requires_auth: endpoint.requires_auth.unwrap_or(false),
            content_type: endpoint.content_type.clone(), discovery_method: endpoint.discovery_method.clone(),
            first_seen_at: now,
        })
    }

    pub async fn update_endpoint(&self, endpoint_id: &str, updates: &EndpointUpdate) -> Result<()> {
        let conn = rusqlite::Connection::open(&self.db_path)?;
        let now = chrono::Utc::now().to_rfc3339();
        let mut set_clauses = vec!["last_tested_at = ?"];
        let mut params: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(now)];
        if let Some(sc) = updates.status_code { set_clauses.push("status_code = ?"); params.push(Box::new(sc)); }
        if let Some(rt) = updates.response_time_ms { set_clauses.push("response_time_ms = ?"); params.push(Box::new(rt)); }
        if let Some(auth) = updates.requires_auth { set_clauses.push("requires_auth = ?"); params.push(Box::new(auth)); }
        params.push(Box::new(endpoint_id.to_string()));
        let query = format!("UPDATE scan_endpoints SET {} WHERE id = ?", set_clauses.join(", "));
        conn.execute(&query, rusqlite::params_from_iter(params.iter().map(|p| p.as_ref())))?;
        Ok(())
    }

    pub async fn delete_endpoint(&self, endpoint_id: &str) -> Result<()> {
        let conn = rusqlite::Connection::open(&self.db_path)?;
        conn.execute("DELETE FROM scan_endpoints WHERE id = ?1", [endpoint_id])?;
        Ok(())
    }

    pub async fn get_scan_vulnerabilities(&self, scan_id: &str) -> Result<Vec<ScanVulnerability>> {
        let conn = rusqlite::Connection::open(&self.db_path)?;
        let mut stmt = conn.prepare(
            "SELECT id, scan_id, type, category, severity, cvss_score, title,
                    description, recommendation, status, detected_at, affected_url
             FROM scan_vulnerabilities WHERE scan_id = ?1 ORDER BY
                 CASE severity WHEN 'critical' THEN 1 WHEN 'high' THEN 2 WHEN 'medium' THEN 3 WHEN 'low' THEN 4 ELSE 5 END,
                 detected_at DESC"
        )?;
        let vulnerabilities = stmt.query_map([scan_id], |row| {
            Ok(ScanVulnerability {
                id: row.get(0)?, scan_id: row.get(1)?, vuln_type: row.get(2)?, category: row.get(3)?,
                severity: row.get(4)?, cvss_score: row.get(5)?, title: row.get(6)?,
                description: row.get(7)?, recommendation: row.get(8)?, status: row.get(9)?,
                detected_at: row.get(10)?, affected_url: row.get(11)?,
            })
        })?.collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(vulnerabilities)
    }

    pub async fn create_vulnerability(&self, vuln: &VulnerabilityCreate) -> Result<ScanVulnerability> {
        let conn = rusqlite::Connection::open(&self.db_path)?;
        let id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO scan_vulnerabilities (id, scan_id, type, category, severity,
                                               cvss_score, title, description, recommendation,
                                               status, detected_at, updated_at, affected_url)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            rusqlite::params![id, vuln.scan_id, vuln.vuln_type, vuln.category, vuln.severity,
                vuln.cvss_score, vuln.title, vuln.description, vuln.recommendation,
                "open", now, now, vuln.affected_url],
        )?;
        Ok(ScanVulnerability {
            id: id.clone(), scan_id: vuln.scan_id.clone(), vuln_type: vuln.vuln_type.clone(),
            category: vuln.category.clone(), severity: vuln.severity.clone(), cvss_score: vuln.cvss_score,
            title: vuln.title.clone(), description: vuln.description.clone(),
            recommendation: vuln.recommendation.clone(), status: "open".to_string(),
            detected_at: now.clone(), affected_url: vuln.affected_url.clone(),
        })
    }

    pub async fn update_vulnerability(&self, vuln_id: &str, updates: &VulnerabilityUpdate) -> Result<()> {
        let conn = rusqlite::Connection::open(&self.db_path)?;
        let now = chrono::Utc::now().to_rfc3339();
        let mut set_clauses = vec!["updated_at = ?"];
        let mut params: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(now)];
        if let Some(st) = &updates.status { set_clauses.push("status = ?"); params.push(Box::new(st.clone())); }
        if let Some(sv) = &updates.severity { set_clauses.push("severity = ?"); params.push(Box::new(sv.clone())); }
        if let Some(cvss) = updates.cvss_score { set_clauses.push("cvss_score = ?"); params.push(Box::new(cvss)); }
        params.push(Box::new(vuln_id.to_string()));
        let query = format!("UPDATE scan_vulnerabilities SET {} WHERE id = ?", set_clauses.join(", "));
        conn.execute(&query, rusqlite::params_from_iter(params.iter().map(|p| p.as_ref())))?;
        Ok(())
    }

    pub async fn delete_vulnerability(&self, vuln_id: &str) -> Result<()> {
        let conn = rusqlite::Connection::open(&self.db_path)?;
        conn.execute("DELETE FROM scan_vulnerabilities WHERE id = ?1", [vuln_id])?;
        Ok(())
    }

    pub async fn compare_scans(&self, base_scan_id: &str, compare_scan_id: &str) -> Result<ScanComparison> {
        let conn = rusqlite::Connection::open(&self.db_path)?;
        let base_endpoints: Vec<String> = {
            let mut stmt = conn.prepare("SELECT path, method FROM scan_endpoints WHERE scan_id = ?")?;
            let rows = stmt.query_map([base_scan_id], |row| Ok(format!("{}:{}", row.get::<_, String>(0)?, row.get::<_, String>(1)?)))?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };
        let compare_endpoints: Vec<String> = {
            let mut stmt = conn.prepare("SELECT path, method FROM scan_endpoints WHERE scan_id = ?")?;
            let rows = stmt.query_map([compare_scan_id], |row| Ok(format!("{}:{}", row.get::<_, String>(0)?, row.get::<_, String>(1)?)))?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };
        let base_set: std::collections::HashSet<_> = base_endpoints.iter().collect();
        let compare_set: std::collections::HashSet<_> = compare_endpoints.iter().collect();
        let endpoints_added = compare_set.difference(&base_set).count() as i64;
        let endpoints_removed = base_set.difference(&compare_set).count() as i64;
        let base_vulns: i64 = conn.query_row("SELECT COUNT(*) FROM scan_vulnerabilities WHERE scan_id = ?", [base_scan_id], |row| row.get(0))?;
        let compare_vulns: i64 = conn.query_row("SELECT COUNT(*) FROM scan_vulnerabilities WHERE scan_id = ?", [compare_scan_id], |row| row.get(0))?;
        let vulnerabilities_added = (compare_vulns - base_vulns).max(0);
        let vulnerabilities_removed = (base_vulns - compare_vulns).max(0);
        let improvement_score = ((vulnerabilities_removed - vulnerabilities_added) * 10) as f64;
        let id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO scan_comparisons (id, base_scan_id, compare_scan_id,
                                          endpoints_added, endpoints_removed,
                                          vulnerabilities_added, vulnerabilities_removed,
                                          improvement_score, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            rusqlite::params![id, base_scan_id, compare_scan_id, endpoints_added, endpoints_removed,
                vulnerabilities_added, vulnerabilities_removed, improvement_score, now],
        )?;
        Ok(ScanComparison {
            id, base_scan_id: base_scan_id.to_string(), compare_scan_id: compare_scan_id.to_string(),
            endpoints_added, endpoints_removed, endpoints_modified: 0,
            vulnerabilities_added, vulnerabilities_removed, improvement_score, created_at: now,
        })
    }

    pub async fn get_scan_metrics(&self, scan_id: &str) -> Result<DetailedScanMetrics> {
        let scan = self.get_scan(scan_id).await?;
        let endpoints = self.get_scan_endpoints(scan_id).await?;
        let vulnerabilities = self.get_scan_vulnerabilities(scan_id).await?;
        let response_times: Vec<f64> = endpoints.iter().filter_map(|e| e.response_time_ms).collect();
        let avg_response_time = if !response_times.is_empty() {
            response_times.iter().sum::<f64>() / response_times.len() as f64
        } else { 0.0 };
        let critical_count = vulnerabilities.iter().filter(|v| v.severity == "critical").count() as i64;
        let high_count = vulnerabilities.iter().filter(|v| v.severity == "high").count() as i64;
        let medium_count = vulnerabilities.iter().filter(|v| v.severity == "medium").count() as i64;
        let low_count = vulnerabilities.iter().filter(|v| v.severity == "low").count() as i64;
        Ok(DetailedScanMetrics {
            scan_id: scan_id.to_string(), total_endpoints: scan.endpoints_found,
            total_vulnerabilities: scan.vulnerabilities_count, avg_response_time_ms: avg_response_time,
            critical_vulnerabilities: critical_count, high_vulnerabilities: high_count,
            medium_vulnerabilities: medium_count, low_vulnerabilities: low_count,
            calculated_at: chrono::Utc::now().to_rfc3339(),
        })
    }
}
