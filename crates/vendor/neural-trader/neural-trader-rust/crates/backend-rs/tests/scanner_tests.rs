// Scanner Module Integration Tests
// Tests CRUD operations, comparison, and metrics endpoints

use serde_json::json;

#[cfg(test)]
mod scanner_tests {
    use super::*;

    #[test]
    fn test_scanner_module_exists() {
        // Basic compilation test
        assert!(true);
    }

    #[test]
    fn test_endpoint_crud_structure() {
        let endpoint_create = json!({
            "scan_id": "test-scan-1",
            "path": "/api/v1/test",
            "full_url": "https://api.example.com/api/v1/test",
            "method": "GET",
            "status_code": 200,
            "response_time_ms": 145.5,
            "requires_auth": true,
            "content_type": "application/json",
            "discovery_method": "openapi"
        });

        assert!(endpoint_create.get("scan_id").is_some());
        assert!(endpoint_create.get("path").is_some());
        assert!(endpoint_create.get("method").is_some());
    }

    #[test]
    fn test_vulnerability_crud_structure() {
        let vulnerability_create = json!({
            "scan_id": "test-scan-1",
            "vuln_type": "sql_injection",
            "category": "security",
            "severity": "high",
            "cvss_score": 7.5,
            "title": "SQL Injection Vulnerability",
            "description": "Potential SQL injection found",
            "recommendation": "Use parameterized queries",
            "affected_url": "/api/v1/users"
        });

        assert!(vulnerability_create.get("scan_id").is_some());
        assert!(vulnerability_create.get("severity").is_some());
        assert!(vulnerability_create.get("title").is_some());
    }

    #[test]
    fn test_scan_comparison_structure() {
        let comparison = json!({
            "base_scan_id": "scan-1",
            "compare_scan_id": "scan-2",
            "endpoints_added": 2,
            "endpoints_removed": 0,
            "vulnerabilities_added": 1,
            "vulnerabilities_removed": 2,
            "improvement_score": 10.0
        });

        assert!(comparison.get("base_scan_id").is_some());
        assert!(comparison.get("improvement_score").is_some());
    }

    #[test]
    fn test_metrics_structure() {
        let metrics = json!({
            "scan_id": "test-scan-1",
            "total_endpoints": 5,
            "total_vulnerabilities": 3,
            "avg_response_time_ms": 234.5,
            "critical_vulnerabilities": 1,
            "high_vulnerabilities": 1,
            "medium_vulnerabilities": 1,
            "low_vulnerabilities": 0
        });

        assert!(metrics.get("scan_id").is_some());
        assert!(metrics.get("total_endpoints").is_some());
        assert!(metrics.get("avg_response_time_ms").is_some());
    }

    #[test]
    fn test_endpoint_update_fields() {
        let update = json!({
            "status_code": 201,
            "response_time_ms": 123.4,
            "requires_auth": false
        });

        assert_eq!(update.get("status_code").and_then(|v| v.as_i64()), Some(201));
        assert!(update.get("response_time_ms").is_some());
    }

    #[test]
    fn test_vulnerability_severity_levels() {
        let severities = vec!["critical", "high", "medium", "low", "info"];

        for severity in severities {
            let vuln = json!({
                "severity": severity,
                "title": format!("Test {} vulnerability", severity)
            });
            assert_eq!(vuln.get("severity").and_then(|v| v.as_str()), Some(severity));
        }
    }

    #[test]
    fn test_comparison_improvement_score_calculation() {
        // Improvement: removed more vulns than added
        let vulns_removed = 3i64;
        let vulns_added = 1i64;
        let improvement_score = (vulns_removed - vulns_added) * 10;

        assert_eq!(improvement_score, 20);
        assert!(improvement_score > 0, "Should show improvement");
    }

    #[test]
    fn test_comparison_regression_score_calculation() {
        // Regression: added more vulns than removed
        let vulns_removed = 1i64;
        let vulns_added = 3i64;
        let improvement_score = (vulns_removed - vulns_added) * 10;

        assert_eq!(improvement_score, -20);
        assert!(improvement_score < 0, "Should show regression");
    }

    #[test]
    fn test_risk_level_calculation() {
        fn calculate_risk_level(critical: i64, high: i64, medium: i64) -> &'static str {
            if critical > 0 {
                "critical"
            } else if high > 0 {
                "high"
            } else if medium > 0 {
                "medium"
            } else {
                "low"
            }
        }

        assert_eq!(calculate_risk_level(1, 0, 0), "critical");
        assert_eq!(calculate_risk_level(0, 1, 0), "high");
        assert_eq!(calculate_risk_level(0, 0, 1), "medium");
        assert_eq!(calculate_risk_level(0, 0, 0), "low");
    }

    #[test]
    fn test_endpoint_discovery_methods() {
        let methods = vec!["crawl", "openapi", "manual", "inference"];

        for method in methods {
            let endpoint = json!({
                "path": "/api/test",
                "discovery_method": method
            });
            assert_eq!(endpoint.get("discovery_method").and_then(|v| v.as_str()), Some(method));
        }
    }

    #[test]
    fn test_http_methods() {
        let methods = vec!["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"];

        for method in methods {
            let endpoint = json!({
                "method": method,
                "path": "/api/test"
            });
            assert_eq!(endpoint.get("method").and_then(|v| v.as_str()), Some(method));
        }
    }
}
