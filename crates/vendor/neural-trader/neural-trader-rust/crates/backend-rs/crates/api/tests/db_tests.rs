//! Database Layer Tests
//!
//! Comprehensive tests for database operations including:
//! - CRUD operations for all models
//! - Transaction handling
//! - Query performance
//! - Data integrity constraints
//! - Error handling

mod common;

use common::{TestDb, MockData};
use anyhow::Result;
use serde_json::json;

// ============================================================================
// Database Connection Tests
// ============================================================================

#[test]
fn test_db_connection_creation() -> Result<()> {
    let test_db = TestDb::new()?;
    let conn = test_db.conn()?;

    // Verify connection is valid
    let result: i64 = conn.query_row("SELECT 1", [], |row| row.get(0))?;
    assert_eq!(result, 1);

    Ok(())
}

#[test]
fn test_db_schema_initialization() -> Result<()> {
    let test_db = TestDb::new()?;
    let conn = test_db.conn()?;

    // Verify all tables exist
    let tables = vec![
        "users",
        "workflows",
        "workflow_executions",
        "api_scans",
        "scan_endpoints",
        "scan_vulnerabilities",
        "analytics_events",
    ];

    for table in tables {
        let count: i64 = conn.query_row(
            &format!("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{}'", table),
            [],
            |row| row.get(0),
        )?;
        assert_eq!(count, 1, "Table {} should exist", table);
    }

    Ok(())
}

// ============================================================================
// User CRUD Operations
// ============================================================================

#[test]
fn test_create_user() -> Result<()> {
    let test_db = TestDb::new()?;
    let conn = test_db.conn()?;
    let now = chrono::Utc::now().to_rfc3339();

    conn.execute(
        "INSERT INTO users (id, email, password_hash, role, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        rusqlite::params![
            "new-user",
            "newuser@example.com",
            "$2b$12$hashedpassword",
            "user",
            now,
            now
        ],
    )?;

    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM users WHERE email = 'newuser@example.com'",
        [],
        |row| row.get(0),
    )?;
    assert_eq!(count, 1);

    Ok(())
}

#[test]
fn test_duplicate_email_constraint() -> Result<()> {
    let test_db = TestDb::new()?;
    let conn = test_db.conn()?;
    let now = chrono::Utc::now().to_rfc3339();

    // Try to insert duplicate email (test@example.com already exists in seed data)
    let result = conn.execute(
        "INSERT INTO users (id, email, password_hash, role, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        rusqlite::params![
            "duplicate-user",
            "test@example.com",
            "$2b$12$hashedpassword",
            "user",
            now,
            now
        ],
    );

    assert!(result.is_err(), "Duplicate email should be rejected");

    Ok(())
}

#[test]
fn test_read_user_by_id() -> Result<()> {
    let test_db = TestDb::new()?;
    let conn = test_db.conn()?;

    let (id, email, role): (String, String, String) = conn.query_row(
        "SELECT id, email, role FROM users WHERE id = ?1",
        ["user-1"],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )?;

    assert_eq!(id, "user-1");
    assert_eq!(email, "test@example.com");
    assert_eq!(role, "admin");

    Ok(())
}

#[test]
fn test_update_user() -> Result<()> {
    let test_db = TestDb::new()?;
    let conn = test_db.conn()?;
    let now = chrono::Utc::now().to_rfc3339();

    conn.execute(
        "UPDATE users SET role = ?1, updated_at = ?2 WHERE id = ?3",
        rusqlite::params!["superadmin", now, "user-1"],
    )?;

    let role: String = conn.query_row(
        "SELECT role FROM users WHERE id = ?1",
        ["user-1"],
        |row| row.get(0),
    )?;

    assert_eq!(role, "superadmin");

    Ok(())
}

#[test]
fn test_delete_user() -> Result<()> {
    let test_db = TestDb::new()?;
    let conn = test_db.conn()?;

    conn.execute("DELETE FROM users WHERE id = ?1", ["user-2"])?;

    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM users WHERE id = 'user-2'",
        [],
        |row| row.get(0),
    )?;

    assert_eq!(count, 0);

    Ok(())
}

// ============================================================================
// Workflow Operations
// ============================================================================

#[test]
fn test_create_workflow() -> Result<()> {
    let test_db = TestDb::new()?;
    let conn = test_db.conn()?;
    let now = chrono::Utc::now().to_rfc3339();
    let config = json!({"steps": []}).to_string();

    conn.execute(
        "INSERT INTO workflows (id, user_id, name, description, config, status, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        rusqlite::params![
            "wf-new",
            "user-1",
            "New Workflow",
            "Description",
            config,
            "draft",
            now,
            now
        ],
    )?;

    let (name, status): (String, String) = conn.query_row(
        "SELECT name, status FROM workflows WHERE id = ?1",
        ["wf-new"],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;

    assert_eq!(name, "New Workflow");
    assert_eq!(status, "draft");

    Ok(())
}

#[test]
fn test_list_workflows_by_user() -> Result<()> {
    let test_db = TestDb::new()?;
    let conn = test_db.conn()?;

    let mut stmt = conn.prepare(
        "SELECT COUNT(*) FROM workflows WHERE user_id = ?1"
    )?;

    let count: i64 = stmt.query_row(["user-1"], |row| row.get(0))?;

    assert!(count > 0, "User should have workflows");

    Ok(())
}

#[test]
fn test_foreign_key_constraint_workflow() -> Result<()> {
    let test_db = TestDb::new()?;
    let conn = test_db.conn()?;
    let now = chrono::Utc::now().to_rfc3339();

    // Enable foreign key constraints
    conn.execute("PRAGMA foreign_keys = ON", [])?;

    // Try to create workflow with non-existent user
    let result = conn.execute(
        "INSERT INTO workflows (id, user_id, name, config, status, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        rusqlite::params![
            "wf-invalid",
            "nonexistent-user",
            "Invalid Workflow",
            "{}",
            "draft",
            now,
            now
        ],
    );

    assert!(result.is_err(), "Foreign key constraint should be enforced");

    Ok(())
}

// ============================================================================
// API Scan Operations
// ============================================================================

#[test]
fn test_create_scan() -> Result<()> {
    let test_db = TestDb::new()?;
    let conn = test_db.conn()?;
    let now = chrono::Utc::now().to_rfc3339();

    conn.execute(
        "INSERT INTO api_scans (id, user_id, name, url, base_url, scan_type, status, total_endpoints, total_vulnerabilities, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        rusqlite::params![
            "scan-new",
            "user-1",
            "New Scan",
            "https://api.test.com",
            "https://api.test.com",
            "openapi",
            "pending",
            0,
            0,
            now,
            now
        ],
    )?;

    let status: String = conn.query_row(
        "SELECT status FROM api_scans WHERE id = ?1",
        ["scan-new"],
        |row| row.get(0),
    )?;

    assert_eq!(status, "pending");

    Ok(())
}

#[test]
fn test_update_scan_status() -> Result<()> {
    let test_db = TestDb::new()?;
    let conn = test_db.conn()?;
    let now = chrono::Utc::now().to_rfc3339();

    conn.execute(
        "UPDATE api_scans SET status = ?1, updated_at = ?2 WHERE id = ?3",
        rusqlite::params!["running", now, "scan-1"],
    )?;

    let status: String = conn.query_row(
        "SELECT status FROM api_scans WHERE id = ?1",
        ["scan-1"],
        |row| row.get(0),
    )?;

    assert_eq!(status, "running");

    Ok(())
}

#[test]
fn test_update_scan_results() -> Result<()> {
    let test_db = TestDb::new()?;
    let conn = test_db.conn()?;
    let now = chrono::Utc::now().to_rfc3339();

    conn.execute(
        "UPDATE api_scans SET status = ?1, total_endpoints = ?2, total_vulnerabilities = ?3, completed_at = ?4 WHERE id = ?5",
        rusqlite::params!["completed", 10, 3, now, "scan-1"],
    )?;

    let (endpoints, vulns): (i64, i64) = conn.query_row(
        "SELECT total_endpoints, total_vulnerabilities FROM api_scans WHERE id = ?1",
        ["scan-1"],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;

    assert_eq!(endpoints, 10);
    assert_eq!(vulns, 3);

    Ok(())
}

#[test]
fn test_delete_scan_cascade() -> Result<()> {
    let test_db = TestDb::new()?;
    let conn = test_db.conn()?;
    let now = chrono::Utc::now().to_rfc3339();

    // Enable foreign key constraints
    conn.execute("PRAGMA foreign_keys = ON", [])?;

    // Create scan with related data
    conn.execute(
        "INSERT INTO api_scans (id, user_id, name, url, base_url, scan_type, status, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        rusqlite::params![
            "scan-cascade",
            "user-1",
            "Cascade Test",
            "https://test.com",
            "https://test.com",
            "openapi",
            "completed",
            now,
            now
        ],
    )?;

    // Create endpoint
    conn.execute(
        "INSERT INTO scan_endpoints (id, scan_id, path, method, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        rusqlite::params!["ep-1", "scan-cascade", "/test", "GET", now],
    )?;

    // Create vulnerability
    conn.execute(
        "INSERT INTO scan_vulnerabilities (id, scan_id, endpoint_id, severity, category, description, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        rusqlite::params!["vuln-1", "scan-cascade", "ep-1", "high", "auth", "Test vuln", now],
    )?;

    // Delete scan - should cascade to endpoints and vulnerabilities
    conn.execute("DELETE FROM api_scans WHERE id = ?1", ["scan-cascade"])?;

    let ep_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM scan_endpoints WHERE scan_id = 'scan-cascade'",
        [],
        |row| row.get(0),
    )?;

    let vuln_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM scan_vulnerabilities WHERE scan_id = 'scan-cascade'",
        [],
        |row| row.get(0),
    )?;

    assert_eq!(ep_count, 0, "Endpoints should be deleted");
    assert_eq!(vuln_count, 0, "Vulnerabilities should be deleted");

    Ok(())
}

// ============================================================================
// Analytics Events
// ============================================================================

#[test]
fn test_create_analytics_event() -> Result<()> {
    let test_db = TestDb::new()?;
    let conn = test_db.conn()?;
    let now = chrono::Utc::now().to_rfc3339();
    let event_data = json!({"action": "scan_started", "scan_id": "scan-1"}).to_string();

    conn.execute(
        "INSERT INTO analytics_events (id, user_id, event_type, event_data, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        rusqlite::params![uuid::Uuid::new_v4().to_string(), "user-1", "scan", event_data, now],
    )?;

    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM analytics_events WHERE event_type = 'scan'",
        [],
        |row| row.get(0),
    )?;

    assert!(count > 0);

    Ok(())
}

#[test]
fn test_query_events_by_user() -> Result<()> {
    let test_db = TestDb::new()?;
    let conn = test_db.conn()?;
    let now = chrono::Utc::now().to_rfc3339();

    // Insert test events
    for i in 0..5 {
        conn.execute(
            "INSERT INTO analytics_events (id, user_id, event_type, event_data, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![
                uuid::Uuid::new_v4().to_string(),
                "user-1",
                format!("event-{}", i),
                "{}",
                now
            ],
        )?;
    }

    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM analytics_events WHERE user_id = 'user-1'",
        [],
        |row| row.get(0),
    )?;

    assert!(count >= 5);

    Ok(())
}

// ============================================================================
// Transaction Tests
// ============================================================================

#[test]
fn test_transaction_commit() -> Result<()> {
    let test_db = TestDb::new()?;
    let mut conn = test_db.conn()?;
    let now = chrono::Utc::now().to_rfc3339();

    let tx = conn.transaction()?;

    tx.execute(
        "INSERT INTO users (id, email, password_hash, role, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        rusqlite::params!["tx-user", "tx@example.com", "hash", "user", now, now],
    )?;

    tx.commit()?;

    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM users WHERE id = 'tx-user'",
        [],
        |row| row.get(0),
    )?;

    assert_eq!(count, 1);

    Ok(())
}

#[test]
fn test_transaction_rollback() -> Result<()> {
    let test_db = TestDb::new()?;
    let mut conn = test_db.conn()?;
    let now = chrono::Utc::now().to_rfc3339();

    {
        let tx = conn.transaction()?;

        tx.execute(
            "INSERT INTO users (id, email, password_hash, role, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params!["rollback-user", "rollback@example.com", "hash", "user", now, now],
        )?;

        // Transaction dropped without commit - should rollback
    }

    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM users WHERE id = 'rollback-user'",
        [],
        |row| row.get(0),
    )?;

    assert_eq!(count, 0);

    Ok(())
}

// ============================================================================
// Index Performance Tests
// ============================================================================

#[test]
fn test_indexes_exist() -> Result<()> {
    let test_db = TestDb::new()?;
    let conn = test_db.conn()?;

    let indexes = vec![
        "idx_scans_user",
        "idx_scans_status",
        "idx_endpoints_scan",
        "idx_vulnerabilities_scan",
        "idx_events_user",
        "idx_events_type",
    ];

    for index in indexes {
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name=?1",
            [index],
            |row| row.get(0),
        )?;
        assert_eq!(count, 1, "Index {} should exist", index);
    }

    Ok(())
}
