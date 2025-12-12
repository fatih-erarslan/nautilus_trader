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
