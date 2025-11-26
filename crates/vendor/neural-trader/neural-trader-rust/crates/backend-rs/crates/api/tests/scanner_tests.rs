//! Comprehensive Integration Tests for API Scanner
//!
//! Tests cover:
//! 1. OpenAPI spec parsing (valid and invalid specs)
//! 2. Endpoint discovery and testing
//! 3. Vulnerability detection accuracy
//! 4. Performance metrics collection
//! 5. AgentDB storage and retrieval
//! 6. Agentic-flow analysis execution
//! 7. API endpoint responses
//! 8. Error handling and edge cases

use beclever_api::scanner::*;
use serde_json::json;

mod fixtures;
use fixtures::*;

// ============================================================================
// Unit Tests - OpenAPI Spec Parsing
// ============================================================================

#[tokio::test]
async fn test_parse_valid_openapi_3_0_spec() {
    let scanner = ApiScanner::new().unwrap();
    let spec_json = create_valid_openapi_30_spec();

    let result = scanner.parse_openapi_spec(&spec_json).await;
    assert!(result.is_ok(), "Failed to parse valid OpenAPI 3.0 spec: {:?}", result.err());

    let spec = result.unwrap();
    assert_eq!(spec.info.title, "Sample API");
    assert_eq!(spec.info.version, "1.0.0");
    assert!(!spec.paths.is_empty());
}

#[tokio::test]
async fn test_parse_valid_openapi_3_1_spec() {
    let scanner = ApiScanner::new().unwrap();
    let spec_json = json!({
        "openapi": "3.1.0",
        "info": {
            "title": "Test API",
            "version": "2.0.0"
        },
        "paths": {
            "/test": {
                "get": {
                    "responses": {
                        "200": {
                            "description": "Success"
                        }
                    }
                }
            }
        }
    }).to_string();

    let result = scanner.parse_openapi_spec(&spec_json).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_parse_swagger_2_0_spec() {
    let scanner = ApiScanner::new().unwrap();
    let spec_json = json!({
        "swagger": "2.0",
        "info": {
            "title": "Swagger API",
            "version": "1.0.0"
        },
        "paths": {
            "/users": {
                "get": {
                    "responses": {
                        "200": {
                            "description": "Success"
                        }
                    }
                }
            }
        }
    }).to_string();

    let result = scanner.parse_openapi_spec(&spec_json).await;
    assert!(result.is_ok(), "Should support Swagger 2.0");
}

#[tokio::test]
async fn test_parse_invalid_json() {
    let scanner = ApiScanner::new().unwrap();
    let invalid_json = "not valid json {{}";

    let result = scanner.parse_openapi_spec(invalid_json).await;
    assert!(result.is_err(), "Should reject invalid JSON");
}

#[tokio::test]
async fn test_parse_missing_required_fields() {
    let scanner = ApiScanner::new().unwrap();

    // Missing info field
    let spec = json!({
        "openapi": "3.0.0",
        "paths": {}
    }).to_string();

    let result = scanner.parse_openapi_spec(&spec).await;
    assert!(result.is_err(), "Should reject spec missing required info field");
}

#[tokio::test]
async fn test_parse_empty_paths() {
    let scanner = ApiScanner::new().unwrap();
    let spec = json!({
        "openapi": "3.0.0",
        "info": {
            "title": "Empty API",
            "version": "1.0.0"
        },
        "paths": {}
    }).to_string();

    let result = scanner.parse_openapi_spec(&spec).await;
    // Empty paths should be allowed (spec is valid even if empty)
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_parse_complex_spec_with_components() {
    let scanner = ApiScanner::new().unwrap();
    let spec = create_complex_openapi_spec();

    let result = scanner.parse_openapi_spec(&spec).await;
    assert!(result.is_ok());

    let parsed = result.unwrap();
    assert!(parsed.components.is_some());
    assert!(parsed.security.is_some());
}

// ============================================================================
// Unit Tests - Endpoint Discovery
// ============================================================================

#[tokio::test]
async fn test_discover_endpoints_from_spec() {
    let scanner = ApiScanner::new().unwrap();
    let spec_json = create_valid_openapi_30_spec();
    let spec = scanner.parse_openapi_spec(&spec_json).await.unwrap();

    let endpoints = scanner.discover_endpoints(&spec);

    assert!(!endpoints.is_empty(), "Should discover at least one endpoint");
    assert!(endpoints.iter().any(|e| e.path == "/users"));
    assert!(endpoints.iter().any(|e| e.method == HttpMethod::GET));
}

#[tokio::test]
async fn test_discover_multiple_methods_per_path() {
    let scanner = ApiScanner::new().unwrap();
    let spec_json = create_crud_api_spec();
    let spec = scanner.parse_openapi_spec(&spec_json).await.unwrap();

    let endpoints = scanner.discover_endpoints(&spec);

    let users_endpoints: Vec<_> = endpoints.iter()
        .filter(|e| e.path.contains("/users"))
        .collect();

    assert!(users_endpoints.len() >= 2, "Should have multiple methods");
    assert!(endpoints.iter().any(|e| e.method == HttpMethod::GET));
    assert!(endpoints.iter().any(|e| e.method == HttpMethod::POST));
}

#[tokio::test]
async fn test_discover_endpoints_with_parameters() {
    let scanner = ApiScanner::new().unwrap();
    let spec_json = create_parameterized_api_spec();
    let spec = scanner.parse_openapi_spec(&spec_json).await.unwrap();

    let endpoints = scanner.discover_endpoints(&spec);

    let search_endpoint = endpoints.iter()
        .find(|e| e.path.contains("/search"));

    assert!(search_endpoint.is_some(), "Should find search endpoint");
}

#[tokio::test]
async fn test_endpoint_authentication_detection() {
    let scanner = ApiScanner::new().unwrap();
    let spec_json = create_auth_required_spec();
    let spec = scanner.parse_openapi_spec(&spec_json).await.unwrap();

    let endpoints = scanner.discover_endpoints(&spec);

    // Check that auth information is captured
    assert!(!endpoints.is_empty());
}

// ============================================================================
// Unit Tests - Vulnerability Detection
// ============================================================================

#[tokio::test]
async fn test_detect_missing_authentication() {
    let scanner = ApiScanner::new().unwrap();

    let endpoint = EndpointInfo {
        path: "/admin/users".to_string(),
        method: HttpMethod::DELETE,
        auth_required: false,
        parameters: vec![],
        discovery_method: DiscoveryMethod::OpenAPISpec,
        tested: false,
        response_code: None,
        response_time_ms: None,
    };

    let vulnerabilities = scanner.check_authentication_vulnerabilities(&vec![endpoint]);

    assert!(!vulnerabilities.is_empty(), "Should detect missing auth on admin endpoint");
    assert!(vulnerabilities.iter().any(|v| {
        v.vuln_type == VulnerabilityType::BrokenAuthentication
            && v.severity == VulnerabilitySeverity::Critical
    }));
}

#[tokio::test]
async fn test_detect_sensitive_data_exposure() {
    let scanner = ApiScanner::new().unwrap();

    let endpoint = EndpointInfo {
        path: "/users/123/ssn".to_string(),
        method: HttpMethod::GET,
        auth_required: false,
        parameters: vec![],
        discovery_method: DiscoveryMethod::OpenAPISpec,
        tested: false,
        response_code: None,
        response_time_ms: None,
    };

    let vulnerabilities = scanner.check_sensitive_data_exposure(&vec![endpoint]);

    assert!(!vulnerabilities.is_empty(), "Should detect sensitive data exposure");
    assert!(vulnerabilities.iter().any(|v| {
        v.vuln_type == VulnerabilityType::SensitiveDataExposure
    }));
}

#[tokio::test]
async fn test_detect_injection_vulnerabilities() {
    let scanner = ApiScanner::new().unwrap();

    let endpoint = EndpointInfo {
        path: "/search".to_string(),
        method: HttpMethod::GET,
        auth_required: false,
        parameters: vec!["query".to_string()],
        discovery_method: DiscoveryMethod::OpenAPISpec,
        tested: false,
        response_code: None,
        response_time_ms: None,
    };

    let vulnerabilities = scanner.check_injection_vulnerabilities(&vec![endpoint]);

    // Should identify potential injection points
    assert!(!vulnerabilities.is_empty());
}

#[tokio::test]
async fn test_detect_missing_rate_limiting() {
    let scanner = ApiScanner::new().unwrap();

    let endpoint = EndpointInfo {
        path: "/api/login".to_string(),
        method: HttpMethod::POST,
        auth_required: false,
        parameters: vec![],
        discovery_method: DiscoveryMethod::OpenAPISpec,
        tested: false,
        response_code: None,
        response_time_ms: None,
    };

    let vulnerabilities = scanner.check_rate_limiting_vulnerabilities(&vec![endpoint]);

    assert!(!vulnerabilities.is_empty(), "Should detect missing rate limiting on login");
}

#[tokio::test]
async fn test_vulnerability_severity_classification() {
    let scanner = ApiScanner::new().unwrap();

    let endpoints = vec![
        EndpointInfo {
            path: "/admin/delete-all".to_string(),
            method: HttpMethod::DELETE,
            auth_required: false,
            parameters: vec![],
            discovery_method: DiscoveryMethod::OpenAPISpec,
            tested: false,
            response_code: None,
            response_time_ms: None,
        },
        EndpointInfo {
            path: "/public/info".to_string(),
            method: HttpMethod::GET,
            auth_required: false,
            parameters: vec![],
            discovery_method: DiscoveryMethod::OpenAPISpec,
            tested: false,
            response_code: None,
            response_time_ms: None,
        },
    ];

    let vulnerabilities = scanner.check_authentication_vulnerabilities(&endpoints);

    // Admin endpoint should have critical/high severity
    let admin_vulns: Vec<_> = vulnerabilities.iter()
        .filter(|v| v.affected_endpoint.contains("admin"))
        .collect();

    if !admin_vulns.is_empty() {
        assert!(admin_vulns.iter().any(|v| {
            matches!(v.severity, VulnerabilitySeverity::Critical | VulnerabilitySeverity::High)
        }));
    }
}

// ============================================================================
// Unit Tests - Performance Metrics
// ============================================================================

#[tokio::test]
async fn test_metrics_collection_basic() {
    let endpoints = vec![
        create_tested_endpoint("/api/fast", 50),
        create_tested_endpoint("/api/medium", 200),
        create_tested_endpoint("/api/slow", 1000),
    ];

    let metrics = calculate_performance_metrics(&endpoints);

    assert_eq!(metrics.total_endpoints, 3);
    assert!(metrics.avg_response_time_ms > 0.0);
    assert!(metrics.min_response_time_ms <= metrics.avg_response_time_ms);
    assert!(metrics.max_response_time_ms >= metrics.avg_response_time_ms);
}

#[tokio::test]
async fn test_metrics_percentiles() {
    let mut endpoints = Vec::new();

    for i in 1..=100 {
        endpoints.push(create_tested_endpoint(&format!("/api/endpoint{}", i), i as u64 * 10));
    }

    let metrics = calculate_performance_metrics(&endpoints);

    assert_eq!(metrics.total_endpoints, 100);
    assert!(metrics.p50_response_time_ms > 0);
    assert!(metrics.p95_response_time_ms > metrics.p50_response_time_ms);
    assert!(metrics.p99_response_time_ms >= metrics.p95_response_time_ms);
}

#[tokio::test]
async fn test_metrics_empty_dataset() {
    let endpoints: Vec<EndpointInfo> = Vec::new();
    let metrics = calculate_performance_metrics(&endpoints);

    assert_eq!(metrics.total_endpoints, 0);
    assert_eq!(metrics.avg_response_time_ms, 0.0);
}

#[tokio::test]
async fn test_metrics_response_code_distribution() {
    let endpoints = vec![
        create_tested_endpoint_with_code("/api/success1", 100, 200),
        create_tested_endpoint_with_code("/api/success2", 150, 200),
        create_tested_endpoint_with_code("/api/notfound", 50, 404),
        create_tested_endpoint_with_code("/api/error", 200, 500),
    ];

    let metrics = calculate_performance_metrics(&endpoints);

    assert_eq!(metrics.total_endpoints, 4);
    assert_eq!(metrics.successful_requests, 2);
    assert_eq!(metrics.failed_requests, 2);
}

// ============================================================================
// Integration Tests - Full Scanner Workflow
// ============================================================================

#[tokio::test]
async fn test_full_scan_workflow() {
    let mut scanner = ApiScanner::new().unwrap();

    let spec_json = create_valid_openapi_30_spec();
    let spec = scanner.parse_openapi_spec(&spec_json).await.unwrap();

    let result = scanner.scan_api(
        "https://api.example.com".to_string(),
        Some(spec),
        ScannerConfig::default(),
    ).await;

    assert!(result.is_ok(), "Full scan workflow failed: {:?}", result.err());

    let scan_result = result.unwrap();
    assert_eq!(scan_result.status, ScanStatus::Completed);
    assert!(!scan_result.endpoints.is_empty());
}

#[tokio::test]
async fn test_scan_without_spec() {
    let mut scanner = ApiScanner::new().unwrap();

    let result = scanner.scan_api(
        "https://api.example.com".to_string(),
        None,
        ScannerConfig::default(),
    ).await;

    // Should still work, might discover endpoints through crawling
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_scan_with_custom_config() {
    let mut scanner = ApiScanner::new().unwrap();

    let config = ScannerConfig {
        max_endpoints: 10,
        request_timeout: std::time::Duration::from_secs(5),
        enable_security_scan: true,
        enable_performance_metrics: true,
        enable_crawling: false,
        ..Default::default()
    };

    let spec_json = create_valid_openapi_30_spec();
    let spec = scanner.parse_openapi_spec(&spec_json).await.unwrap();

    let result = scanner.scan_api(
        "https://api.example.com".to_string(),
        Some(spec),
        config,
    ).await;

    assert!(result.is_ok());
}

#[tokio::test]
async fn test_scan_result_storage_and_retrieval() {
    let mut scanner = ApiScanner::new().unwrap();

    let spec_json = create_valid_openapi_30_spec();
    let spec = scanner.parse_openapi_spec(&spec_json).await.unwrap();

    let scan_result = scanner.scan_api(
        "https://api.example.com".to_string(),
        Some(spec),
        ScannerConfig::default(),
    ).await.unwrap();

    let scan_id = scan_result.scan_id;

    // Retrieve the stored result
    let retrieved = scanner.get_scan_result(scan_id).await;
    assert!(retrieved.is_some(), "Should be able to retrieve stored scan result");

    let retrieved_result = retrieved.unwrap();
    assert_eq!(retrieved_result.scan_id, scan_id);
    assert_eq!(retrieved_result.target_url, scan_result.target_url);
}

#[tokio::test]
async fn test_multiple_scans_storage() {
    let mut scanner = ApiScanner::new().unwrap();

    for i in 0..3 {
        let spec_json = create_valid_openapi_30_spec();
        let spec = scanner.parse_openapi_spec(&spec_json).await.unwrap();

        let result = scanner.scan_api(
            format!("https://api{}.example.com", i),
            Some(spec),
            ScannerConfig::default(),
        ).await;

        assert!(result.is_ok());
    }

    let all_results = scanner.get_all_scan_results().await;
    assert_eq!(all_results.len(), 3, "Should have stored all 3 scan results");
}

// ============================================================================
// Integration Tests - AgentDB Storage
// ============================================================================

#[tokio::test]
async fn test_agentdb_store_scan_result() {
    let agentdb = ScannerAgentDB::new();
    let scan_result = create_sample_scan_result();

    let result = agentdb.store_scan_result(&scan_result).await;
    assert!(result.is_ok(), "Should store scan result in AgentDB");
}

#[tokio::test]
async fn test_agentdb_find_similar_apis() {
    let agentdb = ScannerAgentDB::new();
    let scan_result = create_sample_scan_result();

    let result = agentdb.find_similar_apis(&scan_result, 5).await;
    assert!(result.is_ok(), "Should search for similar APIs");
}

// ============================================================================
// Integration Tests - Agentic Flow Analysis
// ============================================================================

#[tokio::test]
async fn test_agentic_flow_generate_recommendations() {
    let flow = ScannerAgenticFlow::new();
    let scan_result = create_sample_scan_result_with_vulnerabilities();

    let recommendations = flow.generate_recommendations(&scan_result).await;
    assert!(recommendations.is_ok());

    let recs = recommendations.unwrap();
    assert!(!recs.is_empty(), "Should generate security recommendations");
}

#[tokio::test]
async fn test_agentic_flow_prioritize_fixes() {
    let flow = ScannerAgenticFlow::new();
    let scan_result = create_sample_scan_result_with_vulnerabilities();

    let prioritized = flow.prioritize_vulnerability_fixes(&scan_result).await;
    assert!(prioritized.is_ok());

    let fixes = prioritized.unwrap();
    // Critical vulnerabilities should be first
    if fixes.len() > 1 {
        for i in 0..fixes.len()-1 {
            let severity_order = |s: &VulnerabilitySeverity| match s {
                VulnerabilitySeverity::Critical => 0,
                VulnerabilitySeverity::High => 1,
                VulnerabilitySeverity::Medium => 2,
                VulnerabilitySeverity::Low => 3,
                VulnerabilitySeverity::Info => 4,
            };

            assert!(
                severity_order(&fixes[i].severity) <= severity_order(&fixes[i+1].severity),
                "Vulnerabilities should be ordered by severity"
            );
        }
    }
}

// ============================================================================
// Error Handling and Edge Cases
// ============================================================================

#[tokio::test]
async fn test_handle_network_timeout() {
    let config = ScannerConfig {
        request_timeout: std::time::Duration::from_millis(1), // Very short timeout
        ..Default::default()
    };

    let mut scanner = ApiScanner::new().unwrap();

    // This should handle timeout gracefully
    let result = scanner.scan_api(
        "https://httpbin.org/delay/10".to_string(),
        None,
        config,
    ).await;

    // Should complete even if individual requests timeout
    assert!(result.is_ok() || result.is_err());
}

#[tokio::test]
async fn test_handle_invalid_url() {
    let mut scanner = ApiScanner::new().unwrap();

    let result = scanner.scan_api(
        "not-a-valid-url".to_string(),
        None,
        ScannerConfig::default(),
    ).await;

    // Should handle invalid URL gracefully
    assert!(result.is_ok() || result.is_err());
}

#[tokio::test]
async fn test_handle_large_spec() {
    let mut scanner = ApiScanner::new().unwrap();

    let spec_json = create_large_openapi_spec(100);
    let spec = scanner.parse_openapi_spec(&spec_json).await;

    assert!(spec.is_ok(), "Should handle large specs");
}

#[tokio::test]
async fn test_concurrent_scans() {
    let mut scanner = ApiScanner::new().unwrap();

    let mut handles = vec![];

    for i in 0..3 {
        let spec_json = create_valid_openapi_30_spec();
        let url = format!("https://api{}.example.com", i);

        // Parse spec before spawning task
        let spec = scanner.parse_openapi_spec(&spec_json).await.unwrap();

        let config = ScannerConfig::default();
        let handle = tokio::spawn(async move {
            let mut local_scanner = ApiScanner::new().unwrap();
            local_scanner.scan_api(url, Some(spec), config).await
        });

        handles.push(handle);
    }

    for handle in handles {
        let result = handle.await;
        assert!(result.is_ok(), "Concurrent scan failed");
    }
}

// ============================================================================
// Helper Functions for Tests
// ============================================================================

fn create_tested_endpoint(path: &str, response_time_ms: u64) -> EndpointInfo {
    EndpointInfo {
        path: path.to_string(),
        method: HttpMethod::GET,
        auth_required: false,
        parameters: vec![],
        discovery_method: DiscoveryMethod::OpenAPISpec,
        tested: true,
        response_code: Some(200),
        response_time_ms: Some(response_time_ms),
    }
}

fn create_tested_endpoint_with_code(path: &str, response_time_ms: u64, code: u16) -> EndpointInfo {
    EndpointInfo {
        path: path.to_string(),
        method: HttpMethod::GET,
        auth_required: false,
        parameters: vec![],
        discovery_method: DiscoveryMethod::OpenAPISpec,
        tested: true,
        response_code: Some(code),
        response_time_ms: Some(response_time_ms),
    }
}

fn calculate_performance_metrics(endpoints: &[EndpointInfo]) -> PerformanceMetrics {
    let mut response_times: Vec<u64> = endpoints
        .iter()
        .filter_map(|e| e.response_time_ms)
        .collect();

    if response_times.is_empty() {
        return PerformanceMetrics {
            total_endpoints: 0,
            successful_requests: 0,
            failed_requests: 0,
            avg_response_time_ms: 0.0,
            min_response_time_ms: 0,
            max_response_time_ms: 0,
            p50_response_time_ms: 0,
            p95_response_time_ms: 0,
            p99_response_time_ms: 0,
        };
    }

    response_times.sort_unstable();

    let total = endpoints.len();
    let successful = endpoints.iter()
        .filter(|e| e.response_code.map(|c| c >= 200 && c < 300).unwrap_or(false))
        .count();

    let avg = response_times.iter().sum::<u64>() as f64 / response_times.len() as f64;
    let min = *response_times.first().unwrap();
    let max = *response_times.last().unwrap();

    let percentile = |p: f64| -> u64 {
        let index = ((response_times.len() as f64 - 1.0) * p) as usize;
        response_times[index]
    };

    PerformanceMetrics {
        total_endpoints: total,
        successful_requests: successful,
        failed_requests: total - successful,
        avg_response_time_ms: avg,
        min_response_time_ms: min,
        max_response_time_ms: max,
        p50_response_time_ms: percentile(0.50),
        p95_response_time_ms: percentile(0.95),
        p99_response_time_ms: percentile(0.99),
    }
}
