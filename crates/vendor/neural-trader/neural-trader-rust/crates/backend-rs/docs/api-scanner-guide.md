# API Scanner Module Guide

## Overview

The API Scanner is a comprehensive Rust-based middleware component that provides:

- OpenAPI/Swagger specification parsing (JSON and YAML)
- Automated HTTP endpoint discovery and testing
- Security vulnerability detection based on OWASP Top 10
- Performance metrics collection and analysis
- Integration with AgentDB for persistent storage and vector search
- Integration with agentic-flow for AI-powered analysis

## Architecture

### Core Components

```
ApiScanner
├── OpenAPI Parser (JSON/YAML)
├── Endpoint Discovery (Crawling)
├── Security Scanner (OWASP Top 10)
├── Performance Metrics Collector
├── ScannerAgentDB (Vector Storage)
└── ScannerAgenticFlow (AI Analysis)
```

### Key Types

- **`ApiScanner`**: Main scanner orchestrator
- **`ScanResult`**: Complete scan result with all findings
- **`EndpointInfo`**: Discovered endpoint metadata
- **`Vulnerability`**: Security issue details
- **`PerformanceMetrics`**: Response times and statistics

## Usage Examples

### Basic Scan

```rust
use beclever_api::scanner::ApiScanner;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let scanner = ApiScanner::new()?;
    let scan_id = scanner.scan("https://api.example.com").await?;

    let result = scanner.get_scan_result(scan_id).await.unwrap();
    println!("Found {} endpoints", result.endpoints.len());
    println!("Found {} vulnerabilities", result.vulnerabilities.len());

    Ok(())
}
```

### Custom Configuration

```rust
use beclever_api::scanner::{ApiScanner, ScannerConfig};
use std::time::Duration;

let mut config = ScannerConfig::default();
config.max_endpoints = 500;
config.request_timeout = Duration::from_secs(15);
config.max_concurrent = 20;
config.enable_crawling = true;

let scanner = ApiScanner::with_config(config)?;
let scan_id = scanner.scan("https://api.example.com").await?;
```

### AgentDB Integration

```rust
use beclever_api::scanner::{ApiScanner, ScannerAgentDB};

let scanner = ApiScanner::new()?;
let agentdb = ScannerAgentDB::new();

// Run scan
let scan_id = scanner.scan("https://api.example.com").await?;
let result = scanner.get_scan_result(scan_id).await.unwrap();

// Store in AgentDB with vector embeddings
agentdb.store_scan_result(&result).await?;

// Find similar APIs
let similar = agentdb.find_similar_apis(&result, 5).await?;
println!("Found {} similar APIs", similar.len());

// Get historical data
let history = agentdb.get_scan_history("https://api.example.com").await?;
println!("Historical scans: {}", history.len());
```

### Agentic-Flow Integration

```rust
use beclever_api::scanner::{ApiScanner, ScannerAgenticFlow};

let scanner = ApiScanner::new()?;
let agentic_flow = ScannerAgenticFlow::new();

// Run scan
let scan_id = scanner.scan("https://api.example.com").await?;
let result = scanner.get_scan_result(scan_id).await.unwrap();

// AI pattern analysis
let patterns = agentic_flow.analyze_patterns(&result).await?;
for pattern in patterns {
    println!("Pattern: {}", pattern);
}

// AI security analysis
let ai_vulns = agentic_flow.detect_security_risks(&result).await?;

// AI recommendations
let recommendations = agentic_flow.generate_recommendations(&result).await?;

// Generate comprehensive report
let report = agentic_flow.generate_report(&result).await?;
println!("{}", report);
```

## Security Checks

The scanner performs comprehensive security analysis:

### 1. Broken Authentication
- Detects unprotected endpoints
- Identifies missing authentication on state-changing operations
- Validates authentication method configuration

### 2. CORS Issues
- Checks for overly permissive CORS policies
- Detects missing CORS headers
- Validates origin restrictions

### 3. Security Headers
- Verifies X-Content-Type-Options
- Checks X-Frame-Options
- Validates Strict-Transport-Security
- Identifies missing security headers

### 4. SSL/TLS
- Ensures HTTPS usage
- Detects unencrypted HTTP endpoints
- Validates TLS configuration (in production)

### 5. Rate Limiting
- Detects missing rate limit headers
- Tests rate limit enforcement (in production)

### 6. Sensitive Data Exposure
- Identifies publicly accessible API specs
- Detects exposed error details
- Validates response sanitization

## Performance Metrics

The scanner collects comprehensive performance data:

```rust
pub struct PerformanceMetrics {
    pub total_requests: usize,
    pub successful_requests: usize,
    pub failed_requests: usize,
    pub avg_response_time_ms: f64,
    pub min_response_time_ms: u64,
    pub max_response_time_ms: u64,
    pub p95_response_time_ms: u64,  // 95th percentile
    pub p99_response_time_ms: u64,  // 99th percentile
    pub endpoints_scanned: usize,
    pub vulnerabilities_found: usize,
}
```

## OpenAPI Support

The scanner supports both OpenAPI 3.x and Swagger 2.x specifications:

### Supported Formats
- JSON (`.json`)
- YAML (`.yaml`, `.yml`)

### Common Spec Locations
The scanner automatically checks:
- `/openapi.json`
- `/swagger.json`
- `/api-docs`
- `/v2/api-docs`
- `/v3/api-docs`
- `/swagger/v1/swagger.json`

### Extracted Information
- API metadata (title, version, description)
- Server URLs
- All endpoints with methods
- Parameters and schemas
- Authentication requirements
- Response codes and formats

## Endpoint Discovery

The scanner uses multiple methods to discover endpoints:

### 1. OpenAPI Specification
Parses spec and extracts all defined endpoints

### 2. Crawling
Tests common API paths:
- `/api/v1`, `/api/v2`
- `/api`
- `/v1`, `/v2`
- `/health`, `/status`, `/ping`

### 3. Manual Addition
Endpoints can be added programmatically

## Vulnerability Severity Levels

```rust
pub enum VulnerabilitySeverity {
    Critical,  // Immediate action required
    High,      // Priority fix needed
    Medium,    // Should be addressed
    Low,       // Minor issue
    Info,      // Informational only
}
```

## Scan Status States

```rust
pub enum ScanStatus {
    Pending,    // Queued for execution
    Running,    // Currently scanning
    Completed,  // Successfully finished
    Failed,     // Encountered errors
    Cancelled,  // Manually stopped
}
```

## Best Practices

### 1. Configuration Tuning
```rust
// For production APIs with many endpoints
config.max_endpoints = 5000;
config.max_concurrent = 50;
config.request_timeout = Duration::from_secs(60);

// For quick scans
config.max_endpoints = 100;
config.max_concurrent = 5;
config.enable_crawling = false;
```

### 2. Error Handling
```rust
match scanner.scan("https://api.example.com").await {
    Ok(scan_id) => {
        println!("Scan started: {}", scan_id);
        // Wait for completion or poll status
    }
    Err(e) => {
        eprintln!("Scan failed: {}", e);
        // Handle error appropriately
    }
}
```

### 3. Rate Limiting Respect
```rust
config.max_concurrent = 5; // Limit concurrent requests
config.request_timeout = Duration::from_secs(30); // Allow time for responses
```

### 4. Logging
The scanner uses `tracing` for comprehensive logging:
```rust
tracing_subscriber::fmt()
    .with_env_filter("info,beclever_api::scanner=debug")
    .init();
```

## Integration Patterns

### Middleware Integration
```rust
use axum::{Router, routing::post};
use tower::ServiceBuilder;

async fn scan_api(body: Json<ScanRequest>) -> Json<ScanResponse> {
    let scanner = ApiScanner::new()?;
    let scan_id = scanner.scan(&body.target_url).await?;
    Json(ScanResponse { scan_id })
}

let app = Router::new()
    .route("/api/scan", post(scan_api));
```

### Background Job Processing
```rust
use tokio::spawn;

let scanner = Arc::new(ApiScanner::new()?);

spawn(async move {
    loop {
        // Poll for pending scans
        // Execute scans in background
        tokio::time::sleep(Duration::from_secs(60)).await;
    }
});
```

### Webhook Notifications
```rust
async fn scan_with_webhook(target: &str, webhook_url: &str) -> Result<()> {
    let scanner = ApiScanner::new()?;
    let scan_id = scanner.scan(target).await?;

    let result = scanner.get_scan_result(scan_id).await.unwrap();

    // Send notification
    reqwest::Client::new()
        .post(webhook_url)
        .json(&result)
        .send()
        .await?;

    Ok(())
}
```

## Future Enhancements

### Planned Features
- GraphQL endpoint discovery and scanning
- WebSocket endpoint testing
- gRPC API scanning
- API rate limit testing
- Load testing capabilities
- Automated penetration testing
- Machine learning for anomaly detection
- Automated remediation suggestions

### AgentDB Enhancements
- Real-time similarity search during scanning
- Pattern recognition across multiple APIs
- Trend analysis and forecasting
- Automated vulnerability correlation

### Agentic-Flow Enhancements
- Multi-agent collaborative analysis
- Automated security testing orchestration
- Intelligent test case generation
- Self-learning vulnerability detection

## Troubleshooting

### Common Issues

**Issue**: Scanner times out on large APIs
```rust
// Solution: Increase timeout and limit endpoints
config.request_timeout = Duration::from_secs(120);
config.max_endpoints = 500;
```

**Issue**: Rate limited by target API
```rust
// Solution: Reduce concurrent requests
config.max_concurrent = 3;
```

**Issue**: No OpenAPI spec found
```rust
// Solution: Enable crawling or add endpoints manually
config.enable_crawling = true;
```

## Performance Considerations

### Memory Usage
- Each scan result is stored in memory
- Use AgentDB integration for persistent storage
- Implement cleanup for old scan results

### Concurrency
- Respect target API rate limits
- Use `max_concurrent` to control load
- Consider implementing exponential backoff

### Network
- Set appropriate timeouts
- Handle transient network errors
- Implement retry logic for critical operations

## Security Considerations

### Scanner Security
- Never store API credentials in scan results
- Sanitize sensitive data in logs
- Validate target URLs before scanning
- Implement authorization for scan operations

### Ethical Scanning
- Only scan APIs you have permission to test
- Respect robots.txt and security.txt
- Implement rate limiting to avoid DoS
- Report vulnerabilities responsibly

## API Reference

See the inline documentation in `/workspaces/FoxRev/beclever/backend-rs/crates/api/src/scanner.rs` for detailed API reference.

## License

MIT License - See project root for details.
