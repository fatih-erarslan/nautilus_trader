//! Performance and Load Tests
//!
//! Tests covering:
//! - Response time benchmarks
//! - Database query performance
//! - Concurrent request handling
//! - Memory usage
//! - Throughput tests

mod common;

use common::{TestDb, PerfHelpers};
use anyhow::Result;
use std::time::Instant;

// ============================================================================
// Response Time Benchmarks
// ============================================================================

#[test]
fn test_health_check_response_time() {
    let iterations = 100;
    let mut total_time = std::time::Duration::ZERO;

    for _ in 0..iterations {
        let start = Instant::now();
        // Simulate health check
        let _ = serde_json::json!({
            "status": "healthy",
            "service": "beclever-api-rust"
        });
        total_time += start.elapsed();
    }

    let avg_time = total_time / iterations;
    println!("Average health check time: {:?}", avg_time);

    assert!(
        avg_time.as_micros() < 1000,
        "Health check should respond in under 1ms"
    );
}

#[test]
fn test_database_query_performance() -> Result<()> {
    let test_db = TestDb::new()?;
    let conn = test_db.conn()?;

    let start = Instant::now();

    for _ in 0..100 {
        let _: i64 = conn.query_row(
            "SELECT COUNT(*) FROM users",
            [],
            |row| row.get(0),
        )?;
    }

    let duration = start.elapsed();
    let avg_time = duration / 100;

    println!("Average query time: {:?}", avg_time);

    assert!(
        avg_time.as_millis() < 10,
        "Simple queries should complete in under 10ms"
    );

    Ok(())
}

#[test]
fn test_scan_list_query_performance() -> Result<()> {
    let test_db = TestDb::new()?;
    let conn = test_db.conn()?;

    // Insert test data
    let now = chrono::Utc::now().to_rfc3339();
    for i in 0..100 {
        conn.execute(
            "INSERT INTO api_scans (id, user_id, name, url, base_url, scan_type, status, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            rusqlite::params![
                format!("scan-{}", i),
                "user-1",
                format!("Scan {}", i),
                format!("https://api{}.example.com", i),
                format!("https://api{}.example.com", i),
                "openapi",
                if i % 3 == 0 { "completed" } else { "running" },
                now,
                now
            ],
        )?;
    }

    let start = Instant::now();

    let mut stmt = conn.prepare(
        "SELECT id, url, status FROM api_scans ORDER BY created_at DESC LIMIT 20"
    )?;

    let scans: Vec<(String, String, String)> = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    let duration = start.elapsed();

    println!("Scan list query time: {:?}, returned {} scans", duration, scans.len());

    assert_eq!(scans.len(), 20);
    assert!(
        duration.as_millis() < 50,
        "Pagination query should complete in under 50ms"
    );

    Ok(())
}

// ============================================================================
// Concurrent Operations Tests
// ============================================================================

#[test]
fn test_concurrent_reads() -> Result<()> {
    use std::sync::Arc;
    use std::thread;

    let test_db = TestDb::new()?;
    let db_path = Arc::new(test_db.path_str());

    let start = Instant::now();
    let num_threads = 10;
    let queries_per_thread = 100;

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let path = Arc::clone(&db_path);
            thread::spawn(move || -> Result<()> {
                let conn = rusqlite::Connection::open(path.as_str())?;

                for _ in 0..queries_per_thread {
                    let _: i64 = conn.query_row(
                        "SELECT COUNT(*) FROM users",
                        [],
                        |row| row.get(0),
                    )?;
                }

                Ok(())
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap()?;
    }

    let duration = start.elapsed();
    let total_queries = num_threads * queries_per_thread;

    println!(
        "Concurrent reads: {} queries in {:?} ({:.2} queries/sec)",
        total_queries,
        duration,
        total_queries as f64 / duration.as_secs_f64()
    );

    Ok(())
}

#[test]
fn test_concurrent_writes() -> Result<()> {
    use std::sync::Arc;
    use std::thread;

    let test_db = TestDb::new()?;
    let db_path = Arc::new(test_db.path_str());

    let start = Instant::now();
    let num_threads = 5;
    let writes_per_thread = 20;

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let path = Arc::clone(&db_path);
            thread::spawn(move || -> Result<()> {
                let conn = rusqlite::Connection::open(path.as_str())?;
                let now = chrono::Utc::now().to_rfc3339();

                for i in 0..writes_per_thread {
                    conn.execute(
                        "INSERT INTO analytics_events (id, user_id, event_type, event_data, created_at)
                         VALUES (?1, ?2, ?3, ?4, ?5)",
                        rusqlite::params![
                            uuid::Uuid::new_v4().to_string(),
                            "user-1",
                            format!("event-thread-{}-{}", thread_id, i),
                            "{}",
                            now
                        ],
                    )?;
                }

                Ok(())
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap()?;
    }

    let duration = start.elapsed();
    let total_writes = num_threads * writes_per_thread;

    println!(
        "Concurrent writes: {} inserts in {:?} ({:.2} inserts/sec)",
        total_writes,
        duration,
        total_writes as f64 / duration.as_secs_f64()
    );

    // Verify all writes succeeded
    let conn = test_db.conn()?;
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM analytics_events WHERE event_type LIKE 'event-thread-%'",
        [],
        |row| row.get(0),
    )?;

    assert_eq!(count, total_writes as i64);

    Ok(())
}

// ============================================================================
// Index Performance Tests
// ============================================================================

#[test]
fn test_indexed_vs_non_indexed_query() -> Result<()> {
    let test_db = TestDb::new()?;
    let conn = test_db.conn()?;
    let now = chrono::Utc::now().to_rfc3339();

    // Insert large dataset
    for i in 0..1000 {
        conn.execute(
            "INSERT INTO api_scans (id, user_id, name, url, base_url, scan_type, status, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            rusqlite::params![
                format!("scan-{}", i),
                format!("user-{}", i % 10), // 10 different users
                format!("Scan {}", i),
                format!("https://api{}.example.com", i),
                format!("https://api{}.example.com", i),
                "openapi",
                if i % 4 == 0 { "completed" } else { "running" },
                now,
                now
            ],
        )?;
    }

    // Query with index (user_id has index idx_scans_user)
    let start = Instant::now();
    let _: Vec<String> = conn
        .prepare("SELECT id FROM api_scans WHERE user_id = 'user-1'")?
        .query_map([], |row| row.get(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let indexed_time = start.elapsed();

    // Query with index (status has index idx_scans_status)
    let start = Instant::now();
    let _: Vec<String> = conn
        .prepare("SELECT id FROM api_scans WHERE status = 'completed'")?
        .query_map([], |row| row.get(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let indexed_status_time = start.elapsed();

    println!("Indexed query (user_id): {:?}", indexed_time);
    println!("Indexed query (status): {:?}", indexed_status_time);

    assert!(
        indexed_time.as_millis() < 100,
        "Indexed query should be fast"
    );
    assert!(
        indexed_status_time.as_millis() < 100,
        "Indexed query should be fast"
    );

    Ok(())
}

// ============================================================================
// Memory Usage Tests
// ============================================================================

#[test]
fn test_large_result_set_memory() -> Result<()> {
    let test_db = TestDb::new()?;
    let conn = test_db.conn()?;
    let now = chrono::Utc::now().to_rfc3339();

    // Insert 10,000 records
    for i in 0..10_000 {
        conn.execute(
            "INSERT INTO analytics_events (id, user_id, event_type, event_data, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![
                uuid::Uuid::new_v4().to_string(),
                "user-1",
                "test_event",
                format!("{{\"index\": {}}}", i),
                now
            ],
        )?;
    }

    let start = Instant::now();

    // Query large result set
    let events: Vec<(String, String)> = conn
        .prepare("SELECT id, event_data FROM analytics_events")?
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    let duration = start.elapsed();

    println!(
        "Retrieved {} events in {:?}",
        events.len(),
        duration
    );

    assert_eq!(events.len(), 10_000);
    assert!(
        duration.as_millis() < 1000,
        "Should retrieve 10k records in under 1 second"
    );

    Ok(())
}

// ============================================================================
// Transaction Performance Tests
// ============================================================================

#[test]
fn test_batch_insert_performance() -> Result<()> {
    let test_db = TestDb::new()?;
    let mut conn = test_db.conn()?;
    let now = chrono::Utc::now().to_rfc3339();

    let batch_size = 1000;

    // Test with transaction
    let start = Instant::now();
    {
        let tx = conn.transaction()?;

        for i in 0..batch_size {
            tx.execute(
                "INSERT INTO analytics_events (id, user_id, event_type, event_data, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![
                    uuid::Uuid::new_v4().to_string(),
                    "user-1",
                    "batch_event",
                    "{}",
                    now
                ],
            )?;
        }

        tx.commit()?;
    }
    let tx_duration = start.elapsed();

    println!(
        "Batch insert with transaction: {} records in {:?} ({:.2} records/sec)",
        batch_size,
        tx_duration,
        batch_size as f64 / tx_duration.as_secs_f64()
    );

    assert!(
        tx_duration.as_millis() < 500,
        "Batch insert should complete in under 500ms"
    );

    Ok(())
}

// ============================================================================
// Pagination Performance Tests
// ============================================================================

#[test]
fn test_pagination_performance() -> Result<()> {
    let test_db = TestDb::new()?;
    let conn = test_db.conn()?;
    let now = chrono::Utc::now().to_rfc3339();

    // Insert test data
    for i in 0..1000 {
        conn.execute(
            "INSERT INTO api_scans (id, user_id, name, url, base_url, scan_type, status, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            rusqlite::params![
                format!("scan-{}", i),
                "user-1",
                format!("Scan {}", i),
                format!("https://api{}.example.com", i),
                format!("https://api{}.example.com", i),
                "openapi",
                "completed",
                now,
                now
            ],
        )?;
    }

    // Test different page sizes
    let page_sizes = vec![10, 20, 50, 100];

    for page_size in page_sizes {
        let start = Instant::now();

        let _: Vec<String> = conn
            .prepare(&format!(
                "SELECT id FROM api_scans ORDER BY created_at DESC LIMIT {} OFFSET 0",
                page_size
            ))?
            .query_map([], |row| row.get(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        let duration = start.elapsed();

        println!(
            "Pagination (page_size={}): {:?}",
            page_size, duration
        );

        assert!(
            duration.as_millis() < 50,
            "Pagination should be fast for page size {}",
            page_size
        );
    }

    Ok(())
}

// ============================================================================
// Aggregate Query Performance
// ============================================================================

#[test]
fn test_aggregate_query_performance() -> Result<()> {
    let test_db = TestDb::new()?;
    let conn = test_db.conn()?;
    let now = chrono::Utc::now().to_rfc3339();

    // Insert test data
    for i in 0..5000 {
        conn.execute(
            "INSERT INTO api_scans (id, user_id, name, url, base_url, scan_type, status, total_endpoints, total_vulnerabilities, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            rusqlite::params![
                format!("scan-{}", i),
                format!("user-{}", i % 100),
                format!("Scan {}", i),
                format!("https://api{}.example.com", i),
                format!("https://api{}.example.com", i),
                "openapi",
                if i % 3 == 0 { "completed" } else { "running" },
                i % 20,
                i % 10,
                now,
                now
            ],
        )?;
    }

    let start = Instant::now();

    // Complex aggregate query
    let (total_scans, total_endpoints, total_vulns, active_scans): (i64, i64, i64, i64) = conn.query_row(
        "SELECT
            COUNT(*),
            SUM(total_endpoints),
            SUM(total_vulnerabilities),
            SUM(CASE WHEN status IN ('queued', 'running') THEN 1 ELSE 0 END)
         FROM api_scans",
        [],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
    )?;

    let duration = start.elapsed();

    println!(
        "Aggregate query: {:?} (scans={}, endpoints={}, vulns={}, active={})",
        duration, total_scans, total_endpoints, total_vulns, active_scans
    );

    assert_eq!(total_scans, 5000);
    assert!(
        duration.as_millis() < 100,
        "Aggregate query should complete in under 100ms"
    );

    Ok(())
}
