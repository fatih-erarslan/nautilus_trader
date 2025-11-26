//! Test fixtures and helpers for scanner integration tests

use beclever_api::scanner::*;
use serde_json::json;
use uuid::Uuid;

/// Create a valid OpenAPI 3.0 specification for testing
pub fn create_valid_openapi_30_spec() -> String {
    json!({
        "openapi": "3.0.0",
        "info": {
            "title": "Sample API",
            "version": "1.0.0",
            "description": "A sample API for testing the scanner"
        },
        "servers": [
            {
                "url": "https://api.example.com",
                "description": "Production server"
            }
        ],
        "paths": {
            "/users": {
                "get": {
                    "summary": "List all users",
                    "responses": {
                        "200": {
                            "description": "Success",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array"
                                    }
                                }
                            }
                        }
                    }
                },
                "post": {
                    "summary": "Create a new user",
                    "responses": {
                        "201": {
                            "description": "Created"
                        }
                    }
                }
            },
            "/users/{id}": {
                "get": {
                    "summary": "Get user by ID",
                    "parameters": [
                        {
                            "name": "id",
                            "in": "path",
                            "required": true,
                            "schema": {
                                "type": "string"
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Success"
                        }
                    }
                }
            }
        }
    }).to_string()
}

/// Create a CRUD API specification for testing
pub fn create_crud_api_spec() -> String {
    json!({
        "openapi": "3.0.0",
        "info": {
            "title": "CRUD API",
            "version": "1.0.0"
        },
        "paths": {
            "/resources": {
                "get": {
                    "summary": "List resources",
                    "responses": { "200": { "description": "OK" } }
                },
                "post": {
                    "summary": "Create resource",
                    "responses": { "201": { "description": "Created" } }
                }
            },
            "/resources/{id}": {
                "get": {
                    "summary": "Get resource",
                    "parameters": [
                        { "name": "id", "in": "path", "required": true, "schema": { "type": "string" } }
                    ],
                    "responses": { "200": { "description": "OK" } }
                },
                "put": {
                    "summary": "Update resource",
                    "parameters": [
                        { "name": "id", "in": "path", "required": true, "schema": { "type": "string" } }
                    ],
                    "responses": { "200": { "description": "OK" } }
                },
                "delete": {
                    "summary": "Delete resource",
                    "parameters": [
                        { "name": "id", "in": "path", "required": true, "schema": { "type": "string" } }
                    ],
                    "responses": { "204": { "description": "No Content" } }
                }
            }
        }
    }).to_string()
}

/// Create a specification with parameterized endpoints
pub fn create_parameterized_api_spec() -> String {
    json!({
        "openapi": "3.0.0",
        "info": {
            "title": "Search API",
            "version": "1.0.0"
        },
        "paths": {
            "/search": {
                "get": {
                    "summary": "Search endpoint",
                    "parameters": [
                        {
                            "name": "q",
                            "in": "query",
                            "required": true,
                            "schema": { "type": "string" }
                        },
                        {
                            "name": "limit",
                            "in": "query",
                            "required": false,
                            "schema": { "type": "integer" }
                        }
                    ],
                    "responses": {
                        "200": { "description": "Results" }
                    }
                }
            }
        }
    }).to_string()
}

/// Create specification requiring authentication
pub fn create_auth_required_spec() -> String {
    json!({
        "openapi": "3.0.0",
        "info": {
            "title": "Secured API",
            "version": "1.0.0"
        },
        "paths": {
            "/admin/users": {
                "get": {
                    "summary": "List users (admin)",
                    "security": [
                        { "bearerAuth": [] }
                    ],
                    "responses": {
                        "200": { "description": "OK" }
                    }
                }
            }
        },
        "components": {
            "securitySchemes": {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer"
                }
            }
        }
    }).to_string()
}

/// Create a complex specification with multiple components
pub fn create_complex_openapi_spec() -> String {
    json!({
        "openapi": "3.0.0",
        "info": {
            "title": "Complex API",
            "version": "2.0.0",
            "contact": {
                "name": "API Support",
                "email": "support@example.com"
            }
        },
        "servers": [
            { "url": "https://api.example.com" },
            { "url": "https://staging.api.example.com" }
        ],
        "paths": {
            "/users": {
                "get": {
                    "summary": "List users",
                    "tags": ["users"],
                    "responses": { "200": { "description": "OK" } }
                }
            }
        },
        "components": {
            "schemas": {
                "User": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "name": { "type": "string" }
                    }
                }
            },
            "securitySchemes": {
                "oauth2": {
                    "type": "oauth2",
                    "flows": {
                        "authorizationCode": {
                            "authorizationUrl": "https://example.com/oauth/authorize",
                            "tokenUrl": "https://example.com/oauth/token",
                            "scopes": {
                                "read": "Read access",
                                "write": "Write access"
                            }
                        }
                    }
                }
            }
        },
        "security": [
            { "oauth2": ["read"] }
        ]
    }).to_string()
}

/// Create a large specification for testing performance
pub fn create_large_openapi_spec(num_endpoints: usize) -> String {
    let mut paths = serde_json::Map::new();

    for i in 0..num_endpoints {
        let path = format!("/endpoint{}", i);
        paths.insert(path, json!({
            "get": {
                "summary": format!("Endpoint {}", i),
                "responses": {
                    "200": { "description": "OK" }
                }
            }
        }));
    }

    json!({
        "openapi": "3.0.0",
        "info": {
            "title": "Large API",
            "version": "1.0.0"
        },
        "paths": paths
    }).to_string()
}

/// Create a sample scan result for testing
pub fn create_sample_scan_result() -> ScanResult {
    ScanResult {
        scan_id: Uuid::new_v4(),
        target_url: "https://api.example.com".to_string(),
        status: ScanStatus::Completed,
        started_at: chrono::Utc::now(),
        completed_at: Some(chrono::Utc::now()),
        spec: None,
        endpoints: vec![],
        vulnerabilities: vec![],
        performance_metrics: PerformanceMetrics {
            total_endpoints: 5,
            successful_requests: 4,
            failed_requests: 1,
            avg_response_time_ms: 150.0,
            min_response_time_ms: 50,
            max_response_time_ms: 300,
            p50_response_time_ms: 150,
            p95_response_time_ms: 280,
            p99_response_time_ms: 295,
        },
        config: ScannerConfig::default(),
    }
}

/// Create a scan result with vulnerabilities for testing
pub fn create_sample_scan_result_with_vulnerabilities() -> ScanResult {
    let mut result = create_sample_scan_result();

    result.vulnerabilities = vec![
        Vulnerability {
            vuln_id: Uuid::new_v4(),
            vuln_type: VulnerabilityType::BrokenAuthentication,
            severity: VulnerabilitySeverity::Critical,
            title: "Missing Authentication".to_string(),
            description: "Admin endpoint accessible without authentication".to_string(),
            affected_endpoint: "/admin/users".to_string(),
            affected_method: HttpMethod::DELETE,
            remediation: "Implement authentication middleware".to_string(),
            cwe_id: Some("CWE-306".to_string()),
            owasp_category: Some("A01:2021 - Broken Access Control".to_string()),
        },
        Vulnerability {
            vuln_id: Uuid::new_v4(),
            vuln_type: VulnerabilitySeverity::Medium,
            title: "Missing Rate Limiting".to_string(),
            description: "No rate limiting detected on login endpoint".to_string(),
            affected_endpoint: "/api/login".to_string(),
            affected_method: HttpMethod::POST,
            remediation: "Implement rate limiting to prevent brute force attacks".to_string(),
            cwe_id: Some("CWE-770".to_string()),
            owasp_category: Some("A04:2021 - Insecure Design".to_string()),
        },
        Vulnerability {
            vuln_id: Uuid::new_v4(),
            vuln_type: VulnerabilityType::SensitiveDataExposure,
            severity: VulnerabilitySeverity::High,
            title: "Sensitive Data Exposure".to_string(),
            description: "SSN field accessible without proper authorization".to_string(),
            affected_endpoint: "/users/123/ssn".to_string(),
            affected_method: HttpMethod::GET,
            remediation: "Restrict access to sensitive fields and implement field-level security".to_string(),
            cwe_id: Some("CWE-359".to_string()),
            owasp_category: Some("A02:2021 - Cryptographic Failures".to_string()),
        },
    ];

    result
}
