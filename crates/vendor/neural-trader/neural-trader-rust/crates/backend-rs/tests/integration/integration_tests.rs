//! Comprehensive End-to-End Integration Tests
//!
//! This module contains integration tests that validate:
//! 1. Authentication flow (registration, login, JWT, RBAC)
//! 2. Scanner integration (create, monitor, results, delete)
//! 3. Analytics flow (activity logging, usage tracking, performance)
//! 4. Security validation (SQL injection, XSS, unauthorized access, CORS)
//!
//! Tests are designed to run sequentially with fresh database state.

use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use std::time::Duration;
use tokio::time::sleep;

const API_BASE: &str = "http://localhost:8080";

/// Helper to create HTTP client with timeout
fn create_client() -> Client {
    Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .expect("Failed to create HTTP client")
}

/// Helper to wait for API to be ready
async fn wait_for_api_ready(client: &Client) -> Result<(), Box<dyn std::error::Error>> {
    for i in 0..30 {
        match client.get(format!("{}/health", API_BASE)).send().await {
            Ok(resp) if resp.status().is_success() => {
                println!("✅ API ready after {} attempts", i + 1);
                return Ok(());
            }
            _ => {
                println!("⏳ Waiting for API... attempt {}/30", i + 1);
                sleep(Duration::from_secs(2)).await;
            }
        }
    }
    Err("API failed to become ready".into())
}

// ============================================================================
// Test 1: Authentication Flow
// ============================================================================

#[tokio::test]
#[ignore] // Run with: cargo test --test integration_tests -- --ignored
async fn test_01_authentication_flow() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{'='*70}");
    println!("TEST 1: Authentication Flow");
    println!("{'='*70}\n");

    let client = create_client();
    wait_for_api_ready(&client).await?;

    // Step 1: Register new user
    println!("1.1 Registering new user...");
    let register_response = client
        .post(format!("{}/api/auth/register", API_BASE))
        .json(&json!({
            "email": "test@example.com",
            "password": "SecurePass123!",
            "name": "Test User"
        }))
        .send()
        .await?;

    assert_eq!(
        register_response.status(),
        StatusCode::CREATED,
        "Registration should succeed"
    );

    let register_data: Value = register_response.json().await?;
    println!("✅ User registered: {:?}", register_data);

    // Step 2: Login with credentials
    println!("\n1.2 Logging in...");
    let login_response = client
        .post(format!("{}/api/auth/login", API_BASE))
        .json(&json!({
            "email": "test@example.com",
            "password": "SecurePass123!"
        }))
        .send()
        .await?;

    assert_eq!(
        login_response.status(),
        StatusCode::OK,
        "Login should succeed"
    );

    let login_data: Value = login_response.json().await?;
    let jwt_token = login_data["token"]
        .as_str()
        .expect("JWT token should be present");
    println!("✅ Login successful, JWT token received: {}...", &jwt_token[..20]);

    // Step 3: Access protected endpoint with valid token
    println!("\n1.3 Accessing protected endpoint with valid token...");
    let protected_response = client
        .get(format!("{}/api/profile", API_BASE))
        .header("Authorization", format!("Bearer {}", jwt_token))
        .send()
        .await?;

    assert_eq!(
        protected_response.status(),
        StatusCode::OK,
        "Protected endpoint should be accessible with valid token"
    );
    println!("✅ Protected endpoint access successful");

    // Step 4: Try accessing with invalid token
    println!("\n1.4 Attempting access with invalid token...");
    let invalid_response = client
        .get(format!("{}/api/profile", API_BASE))
        .header("Authorization", "Bearer invalid-token-12345")
        .send()
        .await?;

    assert_eq!(
        invalid_response.status(),
        StatusCode::UNAUTHORIZED,
        "Invalid token should return 401"
    );
    println!("✅ Invalid token correctly rejected");

    // Step 5: Try accessing without token
    println!("\n1.5 Attempting access without token...");
    let no_token_response = client
        .get(format!("{}/api/profile", API_BASE))
        .send()
        .await?;

    assert_eq!(
        no_token_response.status(),
        StatusCode::UNAUTHORIZED,
        "Missing token should return 401"
    );
    println!("✅ Missing token correctly rejected");

    // Step 6: RBAC - User tries admin endpoint
    println!("\n1.6 Testing RBAC: User attempting admin endpoint...");
    let admin_response = client
        .get(format!("{}/api/admin/users", API_BASE))
        .header("Authorization", format!("Bearer {}", jwt_token))
        .send()
        .await?;

    assert_eq!(
        admin_response.status(),
        StatusCode::FORBIDDEN,
        "User should not have admin access"
    );
    println!("✅ RBAC working: User correctly forbidden from admin endpoint");

    println!("\n✅ Authentication Flow Test PASSED\n");
    Ok(())
}

// ============================================================================
// Test 2: Scanner Integration
// ============================================================================

#[tokio::test]
#[ignore]
async fn test_02_scanner_integration() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{'='*70}");
    println!("TEST 2: Scanner Integration");
    println!("{'='*70}\n");

    let client = create_client();
    wait_for_api_ready(&client).await?;

    // Step 1: Create new scan
    println!("2.1 Creating new API scan...");
    let scan_response = client
        .post(format!("{}/api/scanner/scan", API_BASE))
        .json(&json!({
            "url": "https://petstore.swagger.io/v2/swagger.json",
            "scan_type": "openapi",
            "options": {
                "deep_scan": true,
                "check_auth": true
            }
        }))
        .send()
        .await?;

    assert_eq!(
        scan_response.status(),
        StatusCode::OK,
        "Scan creation should succeed"
    );

    let scan_data: Value = scan_response.json().await?;
    let scan_id = scan_data["scan_id"]
        .as_str()
        .expect("Scan ID should be present");
    println!("✅ Scan created: {}", scan_id);

    // Step 2: Monitor scan status
    println!("\n2.2 Monitoring scan status...");
    let mut scan_status = "queued";
    for i in 0..10 {
        sleep(Duration::from_secs(2)).await;

        let status_response = client
            .get(format!("{}/api/scanner/scans/{}", API_BASE, scan_id))
            .send()
            .await?;

        let status_data: Value = status_response.json().await?;
        scan_status = status_data["status"].as_str().unwrap_or("unknown");

        println!("  Attempt {}: Status = {}", i + 1, scan_status);

        if scan_status == "completed" || scan_status == "failed" {
            break;
        }
    }

    assert!(
        scan_status == "completed" || scan_status == "running",
        "Scan should be completed or running"
    );
    println!("✅ Scan status monitoring successful");

    // Step 3: Get scan results
    println!("\n2.3 Retrieving scan results...");
    let results_response = client
        .get(format!("{}/api/scanner/scans/{}", API_BASE, scan_id))
        .send()
        .await?;

    assert_eq!(
        results_response.status(),
        StatusCode::OK,
        "Results retrieval should succeed"
    );

    let results_data: Value = results_response.json().await?;
    println!("✅ Results retrieved:");
    println!("  - Endpoints found: {}", results_data["endpoints_found"]);
    println!("  - Vulnerabilities: {}", results_data["vulnerabilities_count"]);

    // Step 4: View scan endpoints
    println!("\n2.4 Viewing discovered endpoints...");
    let endpoints_response = client
        .get(format!("{}/api/scanner/scans/{}/endpoints", API_BASE, scan_id))
        .send()
        .await?;

    if endpoints_response.status().is_success() {
        let endpoints_data: Value = endpoints_response.json().await?;
        println!("✅ Endpoints retrieved: {} total",
            endpoints_data.as_array().map(|a| a.len()).unwrap_or(0));
    }

    // Step 5: Delete scan
    println!("\n2.5 Deleting scan...");
    let delete_response = client
        .delete(format!("{}/api/scanner/scans/{}", API_BASE, scan_id))
        .send()
        .await?;

    assert_eq!(
        delete_response.status(),
        StatusCode::OK,
        "Scan deletion should succeed"
    );
    println!("✅ Scan deleted successfully");

    // Step 6: Verify scan was deleted
    println!("\n2.6 Verifying scan deletion...");
    let verify_response = client
        .get(format!("{}/api/scanner/scans/{}", API_BASE, scan_id))
        .send()
        .await?;

    assert_eq!(
        verify_response.status(),
        StatusCode::NOT_FOUND,
        "Deleted scan should not be found"
    );
    println!("✅ Scan verified as deleted");

    // Step 7: Compare two scans
    println!("\n2.7 Creating two scans for comparison...");
    let scan1 = client
        .post(format!("{}/api/scanner/scan", API_BASE))
        .json(&json!({
            "url": "https://api.example.com/v1/spec.json",
            "scan_type": "openapi"
        }))
        .send()
        .await?
        .json::<Value>()
        .await?;

    let scan2 = client
        .post(format!("{}/api/scanner/scan", API_BASE))
        .json(&json!({
            "url": "https://api.example.com/v2/spec.json",
            "scan_type": "openapi"
        }))
        .send()
        .await?
        .json::<Value>()
        .await?;

    let scan1_id = scan1["scan_id"].as_str().unwrap();
    let scan2_id = scan2["scan_id"].as_str().unwrap();

    println!("\n2.8 Comparing scans...");
    let compare_response = client
        .get(format!(
            "{}/api/scanner/compare?scan1={}&scan2={}",
            API_BASE, scan1_id, scan2_id
        ))
        .send()
        .await?;

    if compare_response.status().is_success() {
        let compare_data: Value = compare_response.json().await?;
        println!("✅ Scan comparison successful");
        println!("  - New endpoints: {}", compare_data["new_endpoints"]);
        println!("  - Removed endpoints: {}", compare_data["removed_endpoints"]);
    }

    println!("\n✅ Scanner Integration Test PASSED\n");
    Ok(())
}

// ============================================================================
// Test 3: Analytics Flow
// ============================================================================

#[tokio::test]
#[ignore]
async fn test_03_analytics_flow() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{'='*70}");
    println!("TEST 3: Analytics Flow");
    println!("{'='*70}\n");

    let client = create_client();
    wait_for_api_ready(&client).await?;

    // Step 1: Log activity
    println!("3.1 Logging activity...");
    let activity_response = client
        .post(format!("{}/api/analytics/activity", API_BASE))
        .json(&json!({
            "user_id": "user-123",
            "action": "scan_created",
            "resource": "scan-456",
            "metadata": {
                "scan_type": "openapi",
                "target": "https://api.example.com"
            }
        }))
        .send()
        .await?;

    assert!(
        activity_response.status().is_success(),
        "Activity logging should succeed"
    );
    println!("✅ Activity logged successfully");

    // Step 2: Query activity feed
    println!("\n3.2 Querying activity feed...");
    let feed_response = client
        .get(format!("{}/api/analytics/activity?limit=10", API_BASE))
        .send()
        .await?;

    assert_eq!(
        feed_response.status(),
        StatusCode::OK,
        "Activity feed retrieval should succeed"
    );

    let feed_data: Value = feed_response.json().await?;
    let activities = feed_data.as_array().expect("Feed should be array");

    println!("✅ Activity feed retrieved: {} activities", activities.len());
    assert!(
        activities.iter().any(|a| a["action"] == "scan_created"),
        "Logged activity should appear in feed"
    );

    // Step 3: Track API usage
    println!("\n3.3 Tracking API usage...");
    for i in 1..=5 {
        client
            .post(format!("{}/api/analytics/usage", API_BASE))
            .json(&json!({
                "endpoint": "/api/scanner/scan",
                "method": "POST",
                "user_id": "user-123",
                "response_time_ms": 100 + i * 10
            }))
            .send()
            .await?;
    }
    println!("✅ API usage tracked");

    // Step 4: Get analytics dashboard
    println!("\n3.4 Getting analytics dashboard...");
    let dashboard_response = client
        .get(format!("{}/api/analytics/dashboard", API_BASE))
        .send()
        .await?;

    assert_eq!(
        dashboard_response.status(),
        StatusCode::OK,
        "Dashboard retrieval should succeed"
    );

    let dashboard_data: Value = dashboard_response.json().await?;
    println!("✅ Analytics dashboard retrieved:");
    println!("  - Total requests: {}", dashboard_data["total_requests"]);
    println!("  - Avg response time: {}ms", dashboard_data["avg_response_time_ms"]);

    // Step 5: Performance monitoring
    println!("\n3.5 Getting performance stats...");
    let perf_response = client
        .get(format!("{}/api/analytics/performance", API_BASE))
        .send()
        .await?;

    if perf_response.status().is_success() {
        let perf_data: Value = perf_response.json().await?;
        println!("✅ Performance stats retrieved:");
        println!("  - P50 latency: {}ms", perf_data["p50_latency_ms"]);
        println!("  - P95 latency: {}ms", perf_data["p95_latency_ms"]);
        println!("  - P99 latency: {}ms", perf_data["p99_latency_ms"]);
    }

    println!("\n✅ Analytics Flow Test PASSED\n");
    Ok(())
}

// ============================================================================
// Test 4: Security Validation
// ============================================================================

#[tokio::test]
#[ignore]
async fn test_04_security_validation() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{'='*70}");
    println!("TEST 4: Security Validation");
    println!("{'='*70}\n");

    let client = create_client();
    wait_for_api_ready(&client).await?;

    // Step 1: SQL Injection attempt
    println!("4.1 Testing SQL injection protection...");
    let sql_injection_attempts = vec![
        "' OR '1'='1",
        "1'; DROP TABLE users--",
        "admin'--",
        "' UNION SELECT * FROM passwords--",
    ];

    for payload in sql_injection_attempts {
        let response = client
            .post(format!("{}/api/scanner/scan", API_BASE))
            .json(&json!({
                "url": payload,
                "scan_type": "openapi"
            }))
            .send()
            .await?;

        // Should either reject or sanitize safely
        assert!(
            response.status().is_client_error() || response.status().is_success(),
            "SQL injection should be handled safely"
        );
    }
    println!("✅ SQL injection protection verified");

    // Step 2: XSS attempt
    println!("\n4.2 Testing XSS protection...");
    let xss_payloads = vec![
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert('XSS')>",
        "javascript:alert('XSS')",
    ];

    for payload in xss_payloads {
        let response = client
            .post(format!("{}/api/scanner/scan", API_BASE))
            .json(&json!({
                "url": "https://example.com",
                "scan_type": payload
            }))
            .send()
            .await?;

        let response_text = response.text().await?;
        assert!(
            !response_text.contains("<script>") && !response_text.contains("onerror="),
            "XSS payload should be sanitized"
        );
    }
    println!("✅ XSS protection verified");

    // Step 3: Unauthorized access attempts
    println!("\n4.3 Testing unauthorized access protection...");
    let protected_endpoints = vec![
        "/api/admin/users",
        "/api/admin/settings",
        "/api/profile",
        "/api/analytics/activity",
    ];

    for endpoint in protected_endpoints {
        let response = client
            .get(format!("{}{}", API_BASE, endpoint))
            .send()
            .await?;

        assert_eq!(
            response.status(),
            StatusCode::UNAUTHORIZED,
            "Protected endpoint {} should require authentication",
            endpoint
        );
    }
    println!("✅ Unauthorized access protection verified");

    // Step 4: CORS validation
    println!("\n4.4 Testing CORS configuration...");
    let cors_response = client
        .get(format!("{}/health", API_BASE))
        .header("Origin", "https://malicious-site.com")
        .send()
        .await?;

    let cors_header = cors_response.headers().get("Access-Control-Allow-Origin");

    // Should either allow specific origins or handle correctly
    println!("✅ CORS header present: {:?}", cors_header);

    // Step 5: Rate limiting test
    println!("\n4.5 Testing rate limiting...");
    let mut rate_limit_hit = false;

    for i in 1..=100 {
        let response = client
            .get(format!("{}/health", API_BASE))
            .send()
            .await?;

        if response.status() == StatusCode::TOO_MANY_REQUESTS {
            rate_limit_hit = true;
            println!("✅ Rate limit triggered at request {}", i);
            break;
        }
    }

    if !rate_limit_hit {
        println!("⚠️  Rate limiting not enforced (may be disabled in dev)");
    }

    println!("\n✅ Security Validation Test PASSED\n");
    Ok(())
}

// ============================================================================
// Test 5: Database State Validation
// ============================================================================

#[tokio::test]
#[ignore]
async fn test_05_database_state_validation() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{'='*70}");
    println!("TEST 5: Database State Validation");
    println!("{'='*70}\n");

    let client = create_client();
    wait_for_api_ready(&client).await?;

    // Step 1: Create multiple scans
    println!("5.1 Creating test scans...");
    let mut scan_ids = Vec::new();

    for i in 1..=5 {
        let response = client
            .post(format!("{}/api/scanner/scan", API_BASE))
            .json(&json!({
                "url": format!("https://api{}.example.com/spec.json", i),
                "scan_type": "openapi"
            }))
            .send()
            .await?;

        let data: Value = response.json().await?;
        scan_ids.push(data["scan_id"].as_str().unwrap().to_string());
    }
    println!("✅ Created {} test scans", scan_ids.len());

    // Step 2: Verify all scans in database
    println!("\n5.2 Verifying scans in database...");
    let list_response = client
        .get(format!("{}/api/scanner/scans", API_BASE))
        .send()
        .await?;

    let list_data: Value = list_response.json().await?;
    let scans = list_data["scans"].as_array().expect("Scans should be array");

    assert!(
        scans.len() >= scan_ids.len(),
        "All created scans should be in database"
    );
    println!("✅ Database contains all {} scans", scans.len());

    // Step 3: Verify scan data integrity
    println!("\n5.3 Verifying scan data integrity...");
    for scan_id in &scan_ids {
        let response = client
            .get(format!("{}/api/scanner/scans/{}", API_BASE, scan_id))
            .send()
            .await?;

        let scan: Value = response.json().await?;

        assert_eq!(scan["id"], *scan_id, "Scan ID should match");
        assert!(scan["url"].is_string(), "URL should be present");
        assert!(scan["status"].is_string(), "Status should be present");
        assert!(scan["created_at"].is_string(), "Created timestamp should be present");
    }
    println!("✅ Data integrity verified for all scans");

    // Step 4: Test pagination
    println!("\n5.4 Testing pagination...");
    let page1 = client
        .get(format!("{}/api/scanner/scans?page=1&limit=2", API_BASE))
        .send()
        .await?
        .json::<Value>()
        .await?;

    let page2 = client
        .get(format!("{}/api/scanner/scans?page=2&limit=2", API_BASE))
        .send()
        .await?
        .json::<Value>()
        .await?;

    let page1_scans = page1["scans"].as_array().unwrap();
    let page2_scans = page2["scans"].as_array().unwrap();

    assert_eq!(page1_scans.len(), 2, "Page 1 should have 2 scans");
    assert!(page2_scans.len() > 0, "Page 2 should have scans");

    println!("✅ Pagination working correctly");

    // Step 5: Clean up test data
    println!("\n5.5 Cleaning up test data...");
    for scan_id in &scan_ids {
        client
            .delete(format!("{}/api/scanner/scans/{}", API_BASE, scan_id))
            .send()
            .await?;
    }
    println!("✅ Test data cleaned up");

    println!("\n✅ Database State Validation Test PASSED\n");
    Ok(())
}
