//! Example usage of the API Scanner module
//!
//! Run with: cargo run --example scanner_example

use anyhow::Result;
use std::time::Duration;
use tracing_subscriber;

// Note: This is an example - in production these would be proper imports
// from beclever_api::scanner::*

/// Example: Basic API scan
async fn basic_scan_example() -> Result<()> {
    println!("\n=== Basic API Scan Example ===\n");

    // This demonstrates the API - actual implementation would import from scanner module
    /*
    use beclever_api::scanner::ApiScanner;

    let scanner = ApiScanner::new()?;

    // Scan a public API (ensure you have permission!)
    let scan_id = scanner.scan("https://api.github.com").await?;

    // Get results
    let result = scanner.get_scan_result(scan_id).await
        .expect("Scan result not found");

    println!("Scan ID: {}", result.scan_id);
    println!("Target: {}", result.target_url);
    println!("Status: {:?}", result.status);
    println!("Endpoints found: {}", result.endpoints.len());
    println!("Vulnerabilities: {}", result.vulnerabilities.len());
    println!("Avg response time: {:.2}ms", result.performance_metrics.avg_response_time_ms);
    */

    println!("Basic scan completed");
    Ok(())
}

/// Example: Custom configuration
async fn custom_config_example() -> Result<()> {
    println!("\n=== Custom Configuration Example ===\n");

    /*
    use beclever_api::scanner::{ApiScanner, ScannerConfig};

    let mut config = ScannerConfig::default();

    // Customize scanner behavior
    config.max_endpoints = 500;
    config.request_timeout = Duration::from_secs(15);
    config.max_concurrent = 20;
    config.enable_security_scan = true;
    config.enable_performance_metrics = true;
    config.enable_crawling = true;

    // Add custom headers
    config.custom_headers.insert("X-API-Key".to_string(), "your-key-here".to_string());

    let scanner = ApiScanner::with_config(config)?;
    let scan_id = scanner.scan("https://api.example.com").await?;

    println!("Custom scan started: {}", scan_id);
    */

    println!("Custom configuration demonstrated");
    Ok(())
}

/// Example: Security vulnerability analysis
async fn security_scan_example() -> Result<()> {
    println!("\n=== Security Vulnerability Analysis Example ===\n");

    /*
    use beclever_api::scanner::{ApiScanner, VulnerabilitySeverity};

    let scanner = ApiScanner::new()?;
    let scan_id = scanner.scan("https://api.example.com").await?;
    let result = scanner.get_scan_result(scan_id).await.unwrap();

    // Analyze vulnerabilities by severity
    let critical = result.vulnerabilities.iter()
        .filter(|v| v.severity == VulnerabilitySeverity::Critical)
        .count();
    let high = result.vulnerabilities.iter()
        .filter(|v| v.severity == VulnerabilitySeverity::High)
        .count();
    let medium = result.vulnerabilities.iter()
        .filter(|v| v.severity == VulnerabilitySeverity::Medium)
        .count();

    println!("Critical vulnerabilities: {}", critical);
    println!("High vulnerabilities: {}", high);
    println!("Medium vulnerabilities: {}", medium);

    // Print detailed vulnerability information
    for vuln in &result.vulnerabilities {
        println!("\n{} ({:?})", vuln.title, vuln.severity);
        println!("  Description: {}", vuln.description);
        println!("  Remediation: {}", vuln.remediation);
        if let Some(endpoint) = &vuln.affected_endpoint {
            println!("  Affected: {}", endpoint);
        }
    }
    */

    println!("Security analysis completed");
    Ok(())
}

/// Example: Performance metrics analysis
async fn performance_metrics_example() -> Result<()> {
    println!("\n=== Performance Metrics Example ===\n");

    /*
    use beclever_api::scanner::ApiScanner;

    let scanner = ApiScanner::new()?;
    let scan_id = scanner.scan("https://api.example.com").await?;
    let result = scanner.get_scan_result(scan_id).await.unwrap();

    let metrics = &result.performance_metrics;

    println!("Performance Metrics:");
    println!("  Total requests: {}", metrics.total_requests);
    println!("  Successful: {}", metrics.successful_requests);
    println!("  Failed: {}", metrics.failed_requests);
    println!("  Average response time: {:.2}ms", metrics.avg_response_time_ms);
    println!("  Min response time: {}ms", metrics.min_response_time_ms);
    println!("  Max response time: {}ms", metrics.max_response_time_ms);
    println!("  P95 response time: {}ms", metrics.p95_response_time_ms);
    println!("  P99 response time: {}ms", metrics.p99_response_time_ms);

    // Performance recommendations
    if metrics.avg_response_time_ms > 1000.0 {
        println!("\n⚠️  Warning: Average response time exceeds 1 second");
        println!("   Consider implementing caching or optimizing queries");
    }

    if metrics.p99_response_time_ms > 5000 {
        println!("\n⚠️  Warning: P99 latency is very high");
        println!("   Some requests are experiencing significant delays");
    }
    */

    println!("Performance analysis completed");
    Ok(())
}

/// Example: AgentDB integration
async fn agentdb_integration_example() -> Result<()> {
    println!("\n=== AgentDB Integration Example ===\n");

    /*
    use beclever_api::scanner::{ApiScanner, ScannerAgentDB};

    let scanner = ApiScanner::new()?;
    let agentdb = ScannerAgentDB::new();

    // Run initial scan
    let scan_id = scanner.scan("https://api.example.com").await?;
    let result = scanner.get_scan_result(scan_id).await.unwrap();

    // Store in AgentDB with vector embeddings
    agentdb.store_scan_result(&result).await?;
    println!("Scan result stored in AgentDB");

    // Find similar APIs using vector search
    let similar_apis = agentdb.find_similar_apis(&result, 5).await?;
    println!("Found {} similar APIs", similar_apis.len());

    for similar in similar_apis {
        println!("  - {} (scanned at {})", similar.target_url, similar.started_at);
    }

    // Get historical scan data for trend analysis
    let history = agentdb.get_scan_history("https://api.example.com").await?;
    println!("\nHistorical scans: {}", history.len());

    if history.len() > 1 {
        let first = &history[0];
        let latest = &history[history.len() - 1];

        println!("Vulnerability trend:");
        println!("  First scan: {} vulnerabilities", first.vulnerabilities.len());
        println!("  Latest scan: {} vulnerabilities", latest.vulnerabilities.len());

        if latest.vulnerabilities.len() < first.vulnerabilities.len() {
            println!("  ✅ Security improved over time!");
        } else if latest.vulnerabilities.len() > first.vulnerabilities.len() {
            println!("  ⚠️  Security degraded - new vulnerabilities detected");
        }
    }
    */

    println!("AgentDB integration demonstrated");
    Ok(())
}

/// Example: Agentic-Flow integration
async fn agentic_flow_integration_example() -> Result<()> {
    println!("\n=== Agentic-Flow Integration Example ===\n");

    /*
    use beclever_api::scanner::{ApiScanner, ScannerAgenticFlow};

    let scanner = ApiScanner::new()?;
    let agentic_flow = ScannerAgenticFlow::new();

    // Run scan
    let scan_id = scanner.scan("https://api.example.com").await?;
    let result = scanner.get_scan_result(scan_id).await.unwrap();

    // AI-powered pattern analysis
    println!("Running AI pattern analysis...");
    let patterns = agentic_flow.analyze_patterns(&result).await?;
    println!("\nDetected patterns:");
    for pattern in patterns {
        println!("  - {}", pattern);
    }

    // AI-powered security risk detection
    println!("\nRunning AI security analysis...");
    let ai_vulnerabilities = agentic_flow.detect_security_risks(&result).await?;
    println!("AI detected {} additional security risks", ai_vulnerabilities.len());

    // AI-generated recommendations
    println!("\nGenerating AI recommendations...");
    let recommendations = agentic_flow.generate_recommendations(&result).await?;
    println!("\nRecommendations:");
    for rec in recommendations {
        println!("  - {}", rec);
    }

    // Generate comprehensive report
    println!("\nGenerating comprehensive report...");
    let report = agentic_flow.generate_report(&result).await?;
    println!("\n{}", report);
    */

    println!("Agentic-Flow integration demonstrated");
    Ok(())
}

/// Example: OpenAPI spec parsing
async fn openapi_parsing_example() -> Result<()> {
    println!("\n=== OpenAPI Specification Parsing Example ===\n");

    /*
    use beclever_api::scanner::ApiScanner;

    let scanner = ApiScanner::new()?;
    let scan_id = scanner.scan("https://api.example.com").await?;
    let result = scanner.get_scan_result(scan_id).await.unwrap();

    if let Some(spec) = &result.spec {
        println!("OpenAPI Specification found:");
        println!("  Title: {}", spec.info.title);
        println!("  Version: {}", spec.info.version);

        if let Some(desc) = &spec.info.description {
            println!("  Description: {}", desc);
        }

        println!("\nServers:");
        for server in &spec.servers {
            println!("  - {}", server.url);
            if let Some(desc) = &server.description {
                println!("    Description: {}", desc);
            }
        }

        println!("\nEndpoints:");
        for (path, path_item) in &spec.paths {
            for (method, operation) in &path_item.operations {
                let summary = operation.summary.as_deref().unwrap_or("No summary");
                println!("  {} {} - {}", method.to_uppercase(), path, summary);
            }
        }

        // Authentication information
        if let Some(components) = &spec.components {
            if let Some(security_schemes) = &components.security_schemes {
                println!("\nAuthentication schemes:");
                for (name, scheme) in security_schemes {
                    println!("  - {} ({})", name, scheme.scheme_type);
                }
            }
        }
    } else {
        println!("No OpenAPI specification found");
    }
    */

    println!("OpenAPI parsing demonstrated");
    Ok(())
}

/// Example: Comprehensive report generation
async fn report_generation_example() -> Result<()> {
    println!("\n=== Report Generation Example ===\n");

    /*
    use beclever_api::scanner::{ApiScanner, ScannerAgenticFlow};

    let scanner = ApiScanner::new()?;
    let agentic_flow = ScannerAgenticFlow::new();

    let scan_id = scanner.scan("https://api.example.com").await?;
    let result = scanner.get_scan_result(scan_id).await.unwrap();

    // Generate markdown report
    let report = agentic_flow.generate_report(&result).await?;

    // Save report to file
    tokio::fs::write("scan_report.md", report.as_bytes()).await?;
    println!("Report saved to scan_report.md");

    // Generate JSON report for programmatic access
    let json_report = serde_json::to_string_pretty(&result)?;
    tokio::fs::write("scan_report.json", json_report.as_bytes()).await?;
    println!("JSON report saved to scan_report.json");
    */

    println!("Report generation demonstrated");
    Ok(())
}

/// Example: Batch scanning multiple APIs
async fn batch_scan_example() -> Result<()> {
    println!("\n=== Batch Scanning Example ===\n");

    /*
    use beclever_api::scanner::ApiScanner;
    use tokio::task;

    let scanner = Arc::new(ApiScanner::new()?);

    let apis = vec![
        "https://api1.example.com",
        "https://api2.example.com",
        "https://api3.example.com",
    ];

    let mut handles = vec![];

    for api in apis {
        let scanner_clone = scanner.clone();
        let api_clone = api.to_string();

        let handle = task::spawn(async move {
            match scanner_clone.scan(&api_clone).await {
                Ok(scan_id) => {
                    println!("Started scan for {}: {}", api_clone, scan_id);
                    Some(scan_id)
                }
                Err(e) => {
                    eprintln!("Failed to scan {}: {}", api_clone, e);
                    None
                }
            }
        });

        handles.push(handle);
    }

    // Wait for all scans to complete
    let scan_ids: Vec<_> = futures::future::join_all(handles)
        .await
        .into_iter()
        .filter_map(|r| r.ok().flatten())
        .collect();

    println!("\nCompleted {} scans", scan_ids.len());

    // Analyze results
    for scan_id in scan_ids {
        if let Some(result) = scanner.get_scan_result(scan_id).await {
            println!("\n{}: {} vulnerabilities, {:.2}ms avg response time",
                result.target_url,
                result.vulnerabilities.len(),
                result.performance_metrics.avg_response_time_ms
            );
        }
    }
    */

    println!("Batch scanning demonstrated");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    println!("=================================================");
    println!("    API Scanner Module - Example Usage");
    println!("=================================================");

    // Run all examples
    basic_scan_example().await?;
    custom_config_example().await?;
    security_scan_example().await?;
    performance_metrics_example().await?;
    agentdb_integration_example().await?;
    agentic_flow_integration_example().await?;
    openapi_parsing_example().await?;
    report_generation_example().await?;
    batch_scan_example().await?;

    println!("\n=================================================");
    println!("    All examples completed successfully!");
    println!("=================================================\n");

    Ok(())
}
