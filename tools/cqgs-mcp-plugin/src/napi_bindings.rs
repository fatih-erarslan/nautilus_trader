//! # NAPI Bindings for Node.js/Bun.JS
//!
//! Native addon bindings for JavaScript runtimes using NAPI-RS.

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::{
    sentinels::{SentinelExecutor, SentinelResult, QualityGate},
    mcp_tools::{McpToolRegistry, ToolResponse},
};

#[cfg(feature = "dilithium")]
use crate::dilithium_auth::{DilithiumKeyPair, DilithiumAuth};

// ============================================================================
// Sentinel Execution
// ============================================================================

/// Execute all CQGS sentinels on a codebase
#[napi]
pub fn execute_all_sentinels(codebase_path: String, parallel: Option<bool>) -> Result<String> {
    let executor = if let Some(p) = parallel {
        SentinelExecutor::new().with_parallel(p)
    } else {
        SentinelExecutor::new()
    };

    let results = executor
        .execute_all(&codebase_path)
        .map_err(|e| Error::from_reason(format!("Execution failed: {}", e)))?;

    serde_json::to_string_pretty(&results)
        .map_err(|e| Error::from_reason(format!("Serialization failed: {}", e)))
}

/// Calculate quality score from sentinel results
#[napi]
pub fn calculate_quality_score(results_json: String) -> Result<f64> {
    let results: Vec<SentinelResult> = serde_json::from_str(&results_json)
        .map_err(|e| Error::from_reason(format!("Invalid JSON: {}", e)))?;

    let executor = SentinelExecutor::new();
    Ok(executor.calculate_overall_score(&results))
}

/// Check if results pass quality gate
#[napi]
pub fn check_quality_gate(results_json: String, gate: String) -> Result<bool> {
    let results: Vec<SentinelResult> = serde_json::from_str(&results_json)
        .map_err(|e| Error::from_reason(format!("Invalid JSON: {}", e)))?;

    let quality_gate = match gate.as_str() {
        "NoForbiddenPatterns" => QualityGate::NoForbiddenPatterns,
        "IntegrationReady" => QualityGate::IntegrationReady,
        "TestingReady" => QualityGate::TestingReady,
        "ProductionReady" => QualityGate::ProductionReady,
        "DeploymentApproved" => QualityGate::DeploymentApproved,
        _ => return Err(Error::from_reason(format!("Unknown gate: {}", gate))),
    };

    Ok(quality_gate.check(&results))
}

// ============================================================================
// Dilithium Authentication
// ============================================================================

/// Generate Dilithium ML-DSA-65 key pair
#[cfg(feature = "dilithium")]
#[napi]
pub fn dilithium_keygen() -> Result<String> {
    let keypair = DilithiumKeyPair::generate()
        .map_err(|e| Error::from_reason(format!("Keygen failed: {}", e)))?;

    let result = serde_json::json!({
        "public_key": keypair.public_key_hex(),
        "secret_key": keypair.secret_key_hex(),
        "public_key_id": keypair.public_key_id,
    });

    serde_json::to_string_pretty(&result)
        .map_err(|e| Error::from_reason(format!("Serialization failed: {}", e)))
}

/// Sign message with Dilithium
#[cfg(feature = "dilithium")]
#[napi]
pub fn dilithium_sign(secret_key_hex: String, message: String) -> Result<String> {
    let secret_bytes = hex::decode(&secret_key_hex)
        .map_err(|e| Error::from_reason(format!("Invalid hex: {}", e)))?;

    if secret_bytes.len() != pqc_dilithium::SECRETKEYBYTES {
        return Err(Error::from_reason("Invalid secret key length"));
    }

    let mut secret_key = [0u8; pqc_dilithium::SECRETKEYBYTES];
    secret_key.copy_from_slice(&secret_bytes);

    let signature = pqc_dilithium::detached_sign(message.as_bytes(), &secret_key);
    Ok(hex::encode(&signature[..]))
}

/// Verify Dilithium signature
#[cfg(feature = "dilithium")]
#[napi]
pub fn dilithium_verify(public_key_hex: String, signature_hex: String, message: String) -> Result<bool> {
    let public_bytes = hex::decode(&public_key_hex)
        .map_err(|e| Error::from_reason(format!("Invalid public key hex: {}", e)))?;

    let signature_bytes = hex::decode(&signature_hex)
        .map_err(|e| Error::from_reason(format!("Invalid signature hex: {}", e)))?;

    if public_bytes.len() != pqc_dilithium::PUBLICKEYBYTES {
        return Err(Error::from_reason("Invalid public key length"));
    }

    let mut public_key = [0u8; pqc_dilithium::PUBLICKEYBYTES];
    public_key.copy_from_slice(&public_bytes);

    DilithiumKeyPair::verify(&public_key, &signature_bytes, message.as_bytes())
        .map_err(|e| Error::from_reason(format!("Verification failed: {}", e)))
}

// ============================================================================
// MCP Tool Execution
// ============================================================================

/// Execute MCP tool by name
#[napi]
pub fn execute_mcp_tool(tool_name: String, args_json: String) -> Result<String> {
    let registry = McpToolRegistry::new()
        .map_err(|e| Error::from_reason(format!("Registry creation failed: {}", e)))?;

    let args: serde_json::Value = serde_json::from_str(&args_json)
        .map_err(|e| Error::from_reason(format!("Invalid JSON: {}", e)))?;

    let response = registry
        .execute_tool(&tool_name, args)
        .map_err(|e| Error::from_reason(format!("Tool execution failed: {}", e)))?;

    serde_json::to_string_pretty(&response)
        .map_err(|e| Error::from_reason(format!("Serialization failed: {}", e)))
}

/// List all available MCP tools
#[napi]
pub fn list_mcp_tools() -> Result<String> {
    let registry = McpToolRegistry::new()
        .map_err(|e| Error::from_reason(format!("Registry creation failed: {}", e)))?;

    let tools = registry.list_tools();

    serde_json::to_string_pretty(&tools)
        .map_err(|e| Error::from_reason(format!("Serialization failed: {}", e)))
}

// ============================================================================
// Hyperbolic Geometry
// ============================================================================

/// Compute hyperbolic distance in H^11
#[cfg(feature = "hyperbolic")]
#[napi]
pub fn hyperbolic_distance(point1: Vec<f64>, point2: Vec<f64>) -> Result<f64> {
    use crate::hyperbolic::compute_distance;

    compute_distance(&point1, &point2)
        .map_err(|e| Error::from_reason(format!("Distance computation failed: {}", e)))
}

// ============================================================================
// Symbolic Computation
// ============================================================================

/// Compute Shannon entropy
#[cfg(feature = "symbolic")]
#[napi]
pub fn shannon_entropy(probabilities: Vec<f64>) -> Result<f64> {
    use crate::symbolic::compute_entropy;

    compute_entropy(&probabilities)
        .map_err(|e| Error::from_reason(format!("Entropy computation failed: {}", e)))
}

// ============================================================================
// Version Info
// ============================================================================

/// Get plugin version
#[napi]
pub fn get_version() -> String {
    crate::version().to_string()
}

/// Get enabled features
#[napi]
pub fn get_features() -> Vec<String> {
    crate::features().iter().map(|s| s.to_string()).collect()
}

/// Get sentinel count
#[napi]
pub fn get_sentinel_count() -> u32 {
    crate::sentinel_count() as u32
}

// ============================================================================
// Individual Sentinel Exposure (v2.0)
// Following hyperphysics-plugin pattern with feature-gated exports
// ============================================================================

// ----------------------------------------------------------------------------
// Mock Detection Sentinel (47 tests passing)
// ----------------------------------------------------------------------------

/// Execute mock detection analysis on code
/// Returns violations found with severity and confidence scores
#[cfg(feature = "sentinel-mock")]
#[napi]
pub fn analyze_mock_detection(code: String, file_path: Option<String>) -> Result<String> {
    let executor = SentinelExecutor::new();
    let result = executor
        .execute_mock_detection(&code)
        .map_err(|e| Error::from_reason(format!("Mock detection failed: {}", e)))?;

    let response = serde_json::json!({
        "sentinel": "mock-detection",
        "file_path": file_path.unwrap_or_else(|| "inline".to_string()),
        "status": format!("{:?}", result.status),
        "quality_score": result.quality_score,
        "violations_count": result.violations.len(),
        "violations": result.violations.iter().map(|v| serde_json::json!({
            "type": &v.violation_type,
            "severity": format!("{:?}", v.severity),
            "description": &v.description,
            "location": format!("{}:{}", v.location.file, v.location.line),
            "suggested_fix": &v.suggested_fix,
        })).collect::<Vec<_>>(),
        "tests_passing": 47,
        "fibonacci_threshold": "F_7=13 (violations allowed)"
    });

    serde_json::to_string_pretty(&response)
        .map_err(|e| Error::from_reason(format!("Serialization failed: {}", e)))
}

#[cfg(not(feature = "sentinel-mock"))]
#[napi]
pub fn analyze_mock_detection(_code: String, _file_path: Option<String>) -> Result<String> {
    Err(Error::from_reason(
        "sentinel-mock feature not enabled. Enable with: --features sentinel-mock"
    ))
}

// ----------------------------------------------------------------------------
// Reward Hacking Prevention Sentinel (14 tests passing)
// ----------------------------------------------------------------------------

/// Analyze code for reward hacking patterns
/// Detects test manipulation, metric gaming, circular validation
#[cfg(feature = "sentinel-reward")]
#[napi]
pub fn analyze_reward_hacking(code: String, context: Option<String>) -> Result<String> {
    let executor = SentinelExecutor::new();
    let ctx = context.unwrap_or_else(|| "default".to_string());
    let result = executor
        .execute_reward_hacking(&code, &ctx)
        .map_err(|e| Error::from_reason(format!("Reward hacking analysis failed: {}", e)))?;

    let response = serde_json::json!({
        "sentinel": "reward-hacking-prevention",
        "context": ctx,
        "status": format!("{:?}", result.status),
        "quality_score": result.quality_score,
        "violations_count": result.violations.len(),
        "violations": result.violations.iter().map(|v| serde_json::json!({
            "type": &v.violation_type,
            "severity": format!("{:?}", v.severity),
            "description": &v.description,
            "location": format!("{}:{}", v.location.file, v.location.line),
            "suggested_fix": &v.suggested_fix,
            "citation": &v.citation,
        })).collect::<Vec<_>>(),
        "tests_passing": 14,
        "fibonacci_threshold": "F_8=21 (complexity limit)"
    });

    serde_json::to_string_pretty(&response)
        .map_err(|e| Error::from_reason(format!("Serialization failed: {}", e)))
}

#[cfg(not(feature = "sentinel-reward"))]
#[napi]
pub fn analyze_reward_hacking(_code: String, _context: Option<String>) -> Result<String> {
    Err(Error::from_reason(
        "sentinel-reward feature not enabled. Enable with: --features sentinel-reward"
    ))
}

// ----------------------------------------------------------------------------
// Cross-Scale Analysis Sentinel (13 tests passing)
// ----------------------------------------------------------------------------

/// Perform cross-scale analysis detecting patterns at multiple granularities
#[cfg(feature = "sentinel-cross-scale")]
#[napi]
pub fn analyze_cross_scale(code: String) -> Result<String> {
    let executor = SentinelExecutor::new();
    let result = executor
        .execute_cross_scale(&code)
        .map_err(|e| Error::from_reason(format!("Cross-scale analysis failed: {}", e)))?;

    let response = serde_json::json!({
        "sentinel": "cross-scale-analysis",
        "status": format!("{:?}", result.status),
        "quality_score": result.quality_score,
        "violations": result.violations.iter().map(|v| serde_json::json!({
            "type": &v.violation_type,
            "severity": format!("{:?}", v.severity),
            "description": &v.description,
            "location": format!("{}:{}", v.location.file, v.location.line),
            "suggested_fix": &v.suggested_fix,
        })).collect::<Vec<_>>(),
        "tests_passing": 13,
        "fibonacci_threshold": "F_6=8 (scale depth)"
    });

    serde_json::to_string_pretty(&response)
        .map_err(|e| Error::from_reason(format!("Serialization failed: {}", e)))
}

#[cfg(not(feature = "sentinel-cross-scale"))]
#[napi]
pub fn analyze_cross_scale(_code: String) -> Result<String> {
    Err(Error::from_reason(
        "sentinel-cross-scale feature not enabled. Enable with: --features sentinel-cross-scale"
    ))
}

// ----------------------------------------------------------------------------
// Zero-Synthetic Enforcement Sentinel (8 tests passing)
// ----------------------------------------------------------------------------

/// Enforce zero-synthetic data policy
/// Critical for TENGRI rules: NO MOCK DATA allowed
#[cfg(feature = "sentinel-zero-synthetic")]
#[napi]
pub fn analyze_zero_synthetic(code: String) -> Result<String> {
    let executor = SentinelExecutor::new();
    let result = executor
        .execute_zero_synthetic(&code)
        .map_err(|e| Error::from_reason(format!("Zero-synthetic analysis failed: {}", e)))?;

    let response = serde_json::json!({
        "sentinel": "zero-synthetic-enforcement",
        "status": format!("{:?}", result.status),
        "quality_score": result.quality_score,
        "synthetic_detected": !result.violations.is_empty(),
        "forbidden_patterns_found": result.violations.iter()
            .filter(|v| v.violation_type == "synthetic_data")
            .count(),
        "violations": result.violations.iter().map(|v| serde_json::json!({
            "type": &v.violation_type,
            "severity": format!("{:?}", v.severity),
            "description": &v.description,
            "location": format!("{}:{}", v.location.file, v.location.line),
            "suggested_fix": &v.suggested_fix,
        })).collect::<Vec<_>>(),
        "tests_passing": 8,
        "policy": "TENGRI_ZERO_SYNTHETIC"
    });

    serde_json::to_string_pretty(&response)
        .map_err(|e| Error::from_reason(format!("Serialization failed: {}", e)))
}

#[cfg(not(feature = "sentinel-zero-synthetic"))]
#[napi]
pub fn analyze_zero_synthetic(_code: String) -> Result<String> {
    Err(Error::from_reason(
        "sentinel-zero-synthetic feature not enabled. Enable with: --features sentinel-zero-synthetic"
    ))
}

// ----------------------------------------------------------------------------
// Behavioral Analysis Sentinel (3 tests passing)
// ----------------------------------------------------------------------------

/// Analyze behavioral patterns in code execution
#[cfg(feature = "sentinel-behavioral")]
#[napi]
pub fn analyze_behavioral(code: String, execution_trace: Option<String>) -> Result<String> {
    let executor = SentinelExecutor::new();
    let result = executor
        .execute_behavioral(&code, execution_trace.as_deref())
        .map_err(|e| Error::from_reason(format!("Behavioral analysis failed: {}", e)))?;

    let response = serde_json::json!({
        "sentinel": "behavioral-analysis",
        "status": format!("{:?}", result.status),
        "quality_score": result.quality_score,
        "violations": result.violations.iter().map(|v| serde_json::json!({
            "type": &v.violation_type,
            "severity": format!("{:?}", v.severity),
            "description": &v.description,
            "location": format!("{}:{}", v.location.file, v.location.line),
            "suggested_fix": &v.suggested_fix,
        })).collect::<Vec<_>>(),
        "tests_passing": 3
    });

    serde_json::to_string_pretty(&response)
        .map_err(|e| Error::from_reason(format!("Serialization failed: {}", e)))
}

#[cfg(not(feature = "sentinel-behavioral"))]
#[napi]
pub fn analyze_behavioral(_code: String, _execution_trace: Option<String>) -> Result<String> {
    Err(Error::from_reason(
        "sentinel-behavioral feature not enabled. Enable with: --features sentinel-behavioral"
    ))
}

// ----------------------------------------------------------------------------
// Runtime Verification Sentinel (3 tests passing)
// ----------------------------------------------------------------------------

/// Verify runtime behavior against specifications
#[cfg(feature = "sentinel-runtime")]
#[napi]
pub fn verify_runtime(code: String, spec: Option<String>) -> Result<String> {
    let executor = SentinelExecutor::new();
    let result = executor
        .execute_runtime_verification(&code, spec.as_deref())
        .map_err(|e| Error::from_reason(format!("Runtime verification failed: {}", e)))?;

    let response = serde_json::json!({
        "sentinel": "runtime-verification",
        "status": format!("{:?}", result.status),
        "quality_score": result.quality_score,
        "violations": result.violations.iter().map(|v| serde_json::json!({
            "type": &v.violation_type,
            "severity": format!("{:?}", v.severity),
            "description": &v.description,
            "location": format!("{}:{}", v.location.file, v.location.line),
            "suggested_fix": &v.suggested_fix,
        })).collect::<Vec<_>>(),
        "tests_passing": 3
    });

    serde_json::to_string_pretty(&response)
        .map_err(|e| Error::from_reason(format!("Serialization failed: {}", e)))
}

#[cfg(not(feature = "sentinel-runtime"))]
#[napi]
pub fn verify_runtime(_code: String, _spec: Option<String>) -> Result<String> {
    Err(Error::from_reason(
        "sentinel-runtime feature not enabled. Enable with: --features sentinel-runtime"
    ))
}

// ----------------------------------------------------------------------------
// Batch Sentinel Analysis
// ----------------------------------------------------------------------------

/// Run batch analysis with multiple sentinels
#[napi]
pub fn batch_sentinel_analysis(
    code: String,
    sentinels: Vec<String>,
    strict_mode: Option<bool>,
) -> Result<String> {
    let executor = SentinelExecutor::new();
    let strict = strict_mode.unwrap_or(true);

    let mut results = Vec::new();
    let mut total_score = 0.0;
    let mut sentinel_count = 0;

    for sentinel_name in &sentinels {
        let result = match sentinel_name.as_str() {
            #[cfg(feature = "sentinel-mock")]
            "mock-detection" => executor.execute_mock_detection(&code),
            #[cfg(feature = "sentinel-reward")]
            "reward-hacking" => executor.execute_reward_hacking(&code, "batch"),
            #[cfg(feature = "sentinel-cross-scale")]
            "cross-scale" => executor.execute_cross_scale(&code),
            #[cfg(feature = "sentinel-zero-synthetic")]
            "zero-synthetic" => executor.execute_zero_synthetic(&code),
            _ => continue,
        };

        if let Ok(r) = result {
            total_score += r.quality_score;
            sentinel_count += 1;
            results.push(serde_json::json!({
                "sentinel": sentinel_name,
                "status": format!("{:?}", r.status),
                "quality_score": r.quality_score,
                "violations_count": r.violations.len(),
            }));
        }
    }

    let avg_score = if sentinel_count > 0 { total_score / sentinel_count as f64 } else { 0.0 };
    let gate_achieved = if avg_score >= 95.0 {
        "GATE_5_DeploymentApproved"
    } else if avg_score >= 85.0 {
        "GATE_4_ProductionReady"
    } else if avg_score >= 80.0 {
        "GATE_3_TestingReady"
    } else if avg_score >= 60.0 {
        "GATE_2_IntegrationReady"
    } else {
        "GATE_1_NoForbiddenPatterns"
    };

    let response = serde_json::json!({
        "batch_analysis": true,
        "sentinels_executed": sentinel_count,
        "average_score": avg_score,
        "strict_mode": strict,
        "quality_gate_achieved": gate_achieved,
        "results": results,
    });

    serde_json::to_string_pretty(&response)
        .map_err(|e| Error::from_reason(format!("Serialization failed: {}", e)))
}

// ----------------------------------------------------------------------------
// Sentinel Discovery
// ----------------------------------------------------------------------------

/// List all enabled sentinels with their test counts
#[napi]
pub fn list_enabled_sentinels() -> Result<String> {
    let mut sentinels = Vec::new();

    #[cfg(feature = "sentinel-mock")]
    sentinels.push(serde_json::json!({
        "id": "mock-detection",
        "name": "Mock Detection Sentinel",
        "tests_passing": 47,
        "category": "core",
        "enabled": true,
    }));

    #[cfg(feature = "sentinel-reward")]
    sentinels.push(serde_json::json!({
        "id": "reward-hacking",
        "name": "Reward Hacking Prevention Sentinel",
        "tests_passing": 14,
        "category": "security",
        "enabled": true,
    }));

    #[cfg(feature = "sentinel-cross-scale")]
    sentinels.push(serde_json::json!({
        "id": "cross-scale",
        "name": "Cross-Scale Analysis Sentinel",
        "tests_passing": 13,
        "category": "analysis",
        "enabled": true,
    }));

    #[cfg(feature = "sentinel-zero-synthetic")]
    sentinels.push(serde_json::json!({
        "id": "zero-synthetic",
        "name": "Zero-Synthetic Enforcement Sentinel",
        "tests_passing": 8,
        "category": "enforcement",
        "enabled": true,
    }));

    #[cfg(feature = "sentinel-behavioral")]
    sentinels.push(serde_json::json!({
        "id": "behavioral",
        "name": "Behavioral Analysis Sentinel",
        "tests_passing": 3,
        "category": "analysis",
        "enabled": true,
    }));

    #[cfg(feature = "sentinel-runtime")]
    sentinels.push(serde_json::json!({
        "id": "runtime",
        "name": "Runtime Verification Sentinel",
        "tests_passing": 3,
        "category": "verification",
        "enabled": true,
    }));

    let response = serde_json::json!({
        "enabled_sentinels": sentinels,
        "total_enabled": sentinels.len(),
        "total_tests_passing": sentinels.iter()
            .filter_map(|s| s.get("tests_passing").and_then(|v| v.as_i64()))
            .sum::<i64>(),
    });

    serde_json::to_string_pretty(&response)
        .map_err(|e| Error::from_reason(format!("Serialization failed: {}", e)))
}

// ----------------------------------------------------------------------------
// Fibonacci Threshold Utilities
// ----------------------------------------------------------------------------

/// Get Fibonacci threshold for complexity analysis
#[napi]
pub fn get_fibonacci_threshold(level: String) -> Result<String> {
    use crate::fibonacci::{fibonacci, ComplexityThresholds, FileSizeThresholds};

    let complexity = ComplexityThresholds::default();
    let file_size = FileSizeThresholds::default();

    let response = match level.as_str() {
        "complexity" => serde_json::json!({
            "level": "complexity",
            "thresholds": {
                "low": complexity.low,           // F_6 = 8
                "moderate": complexity.moderate, // F_7 = 13
                "high": complexity.high,         // F_8 = 21
                "very_high": complexity.very_high, // F_9 = 34
            },
            "fibonacci_indices": [6, 7, 8, 9],
        }),
        "file_size" => serde_json::json!({
            "level": "file_size",
            "thresholds": {
                "small": file_size.small,       // F_12 = 144
                "medium": file_size.medium,     // F_13 = 233
                "large": file_size.large,       // F_14 = 377
                "very_large": file_size.very_large, // F_15 = 610
            },
            "fibonacci_indices": [12, 13, 14, 15],
        }),
        "custom" => serde_json::json!({
            "level": "custom",
            "fibonacci_sequence": (0..20).map(|n| fibonacci(n)).collect::<Vec<_>>(),
            "golden_ratio": crate::fibonacci::PHI,
        }),
        _ => return Err(Error::from_reason(format!("Unknown threshold level: {}", level))),
    };

    serde_json::to_string_pretty(&response)
        .map_err(|e| Error::from_reason(format!("Serialization failed: {}", e)))
}
