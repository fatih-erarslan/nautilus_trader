//! # Dilithium MCP Client Integration
//!
//! Provides a Rust client for interacting with the Dilithium MCP server,
//! enabling post-quantum secure communications, hyperbolic computations,
//! pBit dynamics, and formal verification via the Model Context Protocol.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    DILITHIUM MCP CLIENT                         │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
//! │  │   Crypto    │  │  Hyperbolic │  │    pBit     │              │
//! │  │  ML-DSA-65  │  │    H^11     │  │  Dynamics   │              │
//! │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
//! │         │                │                │                     │
//! │         └────────────────┼────────────────┘                     │
//! │                          │                                      │
//! │                  ┌───────▼───────┐                              │
//! │                  │   MCP Client  │                              │
//! │                  │    (stdio)    │                              │
//! │                  └───────┬───────┘                              │
//! │                          │                                      │
//! └──────────────────────────┼──────────────────────────────────────┘
//!                            │
//!                    ┌───────▼───────┐
//!                    │  Dilithium    │
//!                    │  MCP Server   │
//!                    │   (Bun.js)    │
//!                    └───────────────┘
//! ```
//!
//! ## Features
//!
//! - **Post-Quantum Authentication**: Dilithium ML-DSA-65 signatures
//! - **Hyperbolic Geometry**: H^11 Lorentz model computations
//! - **pBit Dynamics**: Probabilistic computing with Boltzmann statistics
//! - **STDP Learning**: Spike-timing dependent plasticity
//! - **Symbolic Math**: Computer algebra system integration
//! - **Swarm Coordination**: Multi-agent orchestration
//! - **Formal Verification**: Wolfram/Z3 theorem proving
//!
//! ## Usage
//!
//! ```rust,ignore
//! use hyperphysics_plugin::dilithium_mcp::{DilithiumClient, McpConfig};
//!
//! // Create client with default config
//! let client = DilithiumClient::new(McpConfig::default()).await?;
//!
//! // Generate post-quantum key pair
//! let keypair = client.dilithium_keygen().await?;
//!
//! // Compute hyperbolic distance
//! let dist = client.hyperbolic_distance(&point1, &point2).await?;
//!
//! // Sample pBit probability
//! let prob = client.pbit_sample(field, bias, temperature).await?;
//! ```

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use parking_lot::Mutex;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{HyperPhysicsError, Result};

// ============================================================================
// Constants
// ============================================================================

/// Default path to Dilithium MCP server
pub const DEFAULT_MCP_PATH: &str = "tools/dilithium-mcp/dist/index.js";

/// Default runtime for MCP server
pub const DEFAULT_RUNTIME: &str = "bun";

/// MCP JSON-RPC version
pub const JSONRPC_VERSION: &str = "2.0";

/// Golden ratio for Fibonacci-based computations
pub const PHI: f64 = 1.618033988749895;

/// Inverse golden ratio
pub const PHI_INV: f64 = 0.618033988749895;

/// Ising critical temperature (Onsager solution)
pub const ISING_CRITICAL_TEMP: f64 = 2.269185314213022;

// ============================================================================
// Configuration
// ============================================================================

/// MCP Client Configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct McpConfig {
    /// Path to MCP server script
    pub server_path: String,
    /// Runtime command (bun, node, deno)
    pub runtime: String,
    /// Environment variables
    pub env: HashMap<String, String>,
    /// Request timeout in milliseconds
    pub timeout_ms: u64,
    /// Enable native module (for best performance)
    pub use_native: bool,
    /// Native module path
    pub native_path: Option<String>,
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            server_path: DEFAULT_MCP_PATH.to_string(),
            runtime: DEFAULT_RUNTIME.to_string(),
            env: HashMap::new(),
            timeout_ms: 30000,
            use_native: true,
            native_path: None,
        }
    }
}

impl McpConfig {
    /// Create config with custom server path
    pub fn with_path(path: impl Into<String>) -> Self {
        Self {
            server_path: path.into(),
            ..Default::default()
        }
    }

    /// Set native module path
    pub fn native_module(mut self, path: impl Into<String>) -> Self {
        self.native_path = Some(path.into());
        self.env.insert(
            "DILITHIUM_NATIVE_PATH".to_string(),
            self.native_path.clone().unwrap(),
        );
        self
    }

    /// Set timeout
    pub fn timeout(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }
}

// ============================================================================
// Types
// ============================================================================

/// Dilithium key pair
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DilithiumKeyPair {
    /// Public key (hex encoded)
    pub public_key: String,
    /// Secret key (hex encoded)
    pub secret_key: String,
}

/// Hyperbolic point in Lorentz coordinates (H^11)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LorentzPoint {
    /// Coordinates (12D: time + 11 spatial)
    pub coords: Vec<f64>,
}

impl LorentzPoint {
    /// Create origin point on hyperboloid
    pub fn origin() -> Self {
        let mut coords = vec![0.0; 12];
        coords[0] = 1.0; // Time component
        Self { coords }
    }

    /// Create from Poincaré ball coordinates
    pub fn from_poincare(ball: &[f64]) -> Self {
        let norm_sq: f64 = ball.iter().map(|x| x * x).sum();
        let scale = 2.0 / (1.0 - norm_sq);

        let mut coords = Vec::with_capacity(12);
        coords.push((1.0 + norm_sq) / (1.0 - norm_sq)); // Time component
        for &x in ball {
            coords.push(scale * x);
        }
        while coords.len() < 12 {
            coords.push(0.0);
        }
        Self { coords }
    }

    /// Validate Lorentz constraint: ⟨x,x⟩_L = -1
    pub fn is_valid(&self) -> bool {
        if self.coords.len() < 2 {
            return false;
        }
        let inner = -self.coords[0] * self.coords[0]
            + self.coords[1..].iter().map(|x| x * x).sum::<f64>();
        (inner + 1.0).abs() < 1e-6
    }
}

/// pBit state and probability
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PBitState {
    /// Current state (+1 or -1)
    pub state: i8,
    /// Probability of being +1
    pub probability: f64,
    /// Effective field
    pub field: f64,
    /// Temperature
    pub temperature: f64,
}

/// STDP weight change result
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StdpResult {
    /// Weight change
    pub delta_weight: f64,
    /// Whether LTP (potentiation) occurred
    pub is_ltp: bool,
    /// Time difference (ms)
    pub delta_t: f64,
}

/// Swarm state for multi-agent coordination
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SwarmState {
    /// Swarm ID
    pub id: String,
    /// Active agents
    pub agent_count: usize,
    /// Topology type
    pub topology: String,
    /// Current iteration
    pub iteration: usize,
    /// Best fitness
    pub best_fitness: f64,
}

/// Tool call result from MCP
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct McpToolResult {
    /// Whether call succeeded
    pub success: bool,
    /// Result data (JSON)
    pub data: String,
    /// Error message if failed
    pub error: Option<String>,
}

// ============================================================================
// MCP Client
// ============================================================================

/// Dilithium MCP Client
///
/// Provides high-performance access to the Dilithium MCP server for:
/// - Post-quantum cryptographic operations
/// - Hyperbolic geometry computations
/// - pBit dynamics and sampling
/// - STDP learning rules
/// - Swarm coordination
/// - Formal verification
pub struct DilithiumClient {
    config: McpConfig,
    process: Option<Arc<Mutex<Child>>>,
    request_id: AtomicU64,
    connected: bool,
}

impl DilithiumClient {
    /// Create new client (does not connect yet)
    pub fn new(config: McpConfig) -> Self {
        Self {
            config,
            process: None,
            request_id: AtomicU64::new(1),
            connected: false,
        }
    }

    /// Create with default config
    pub fn default_config() -> Self {
        Self::new(McpConfig::default())
    }

    /// Connect to MCP server (spawn process)
    pub fn connect(&mut self) -> Result<()> {
        if self.connected {
            return Ok(());
        }

        let mut cmd = Command::new(&self.config.runtime);
        cmd.arg("run")
            .arg(&self.config.server_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Set environment
        for (key, value) in &self.config.env {
            cmd.env(key, value);
        }

        match cmd.spawn() {
            Ok(child) => {
                self.process = Some(Arc::new(Mutex::new(child)));
                self.connected = true;
                tracing::info!("Connected to Dilithium MCP server");
                Ok(())
            }
            Err(e) => Err(HyperPhysicsError::Config(format!(
                "Failed to spawn MCP server: {}",
                e
            ))),
        }
    }

    /// Disconnect from MCP server
    pub fn disconnect(&mut self) -> Result<()> {
        if let Some(process) = self.process.take() {
            let mut child = process.lock();
            let _ = child.kill();
            let _ = child.wait();
        }
        self.connected = false;
        Ok(())
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    /// Get next request ID
    fn next_request_id(&self) -> u64 {
        self.request_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Send JSON-RPC request
    fn send_request(&self, method: &str, params: serde_json::Value) -> Result<serde_json::Value> {
        let process = self.process.as_ref().ok_or_else(|| {
            HyperPhysicsError::Config("Not connected to MCP server".to_string())
        })?;

        let request_id = self.next_request_id();
        let request = serde_json::json!({
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "method": method,
            "params": params
        });

        let mut child = process.lock();

        // Write request
        if let Some(stdin) = child.stdin.as_mut() {
            writeln!(stdin, "{}", request.to_string())
                .map_err(|e| HyperPhysicsError::Config(format!("Failed to write request: {}", e)))?;
            stdin.flush()
                .map_err(|e| HyperPhysicsError::Config(format!("Failed to flush: {}", e)))?;
        } else {
            return Err(HyperPhysicsError::Config("No stdin available".to_string()));
        }

        // Read response
        if let Some(stdout) = child.stdout.as_mut() {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            reader.read_line(&mut line)
                .map_err(|e| HyperPhysicsError::Config(format!("Failed to read response: {}", e)))?;

            let response: serde_json::Value = serde_json::from_str(&line)
                .map_err(|e| HyperPhysicsError::Config(format!("Failed to parse response: {}", e)))?;

            if let Some(error) = response.get("error") {
                return Err(HyperPhysicsError::Config(format!("MCP error: {}", error)));
            }

            Ok(response.get("result").cloned().unwrap_or(serde_json::Value::Null))
        } else {
            Err(HyperPhysicsError::Config("No stdout available".to_string()))
        }
    }

    /// Call a tool on the MCP server
    pub fn call_tool(&self, name: &str, args: serde_json::Value) -> Result<McpToolResult> {
        let params = serde_json::json!({
            "name": name,
            "arguments": args
        });

        match self.send_request("tools/call", params) {
            Ok(result) => {
                let content = result.get("content")
                    .and_then(|c| c.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|item| item.get("text"))
                    .and_then(|t| t.as_str())
                    .unwrap_or("{}");

                Ok(McpToolResult {
                    success: true,
                    data: content.to_string(),
                    error: None,
                })
            }
            Err(e) => Ok(McpToolResult {
                success: false,
                data: String::new(),
                error: Some(e.to_string()),
            }),
        }
    }

    // ========================================================================
    // Dilithium Cryptography
    // ========================================================================

    /// Generate Dilithium ML-DSA key pair
    pub fn dilithium_keygen(&self) -> Result<DilithiumKeyPair> {
        let result = self.call_tool("dilithium_keygen", serde_json::json!({}))?;
        if !result.success {
            return Err(HyperPhysicsError::Config(
                result.error.unwrap_or_else(|| "Keygen failed".to_string())
            ));
        }

        let data: serde_json::Value = serde_json::from_str(&result.data)
            .map_err(|e| HyperPhysicsError::Config(format!("Parse error: {}", e)))?;

        Ok(DilithiumKeyPair {
            public_key: data["public_key"].as_str().unwrap_or("").to_string(),
            secret_key: data["secret_key"].as_str().unwrap_or("").to_string(),
        })
    }

    /// Sign message with Dilithium
    pub fn dilithium_sign(&self, secret_key: &str, message: &str) -> Result<String> {
        let result = self.call_tool("dilithium_sign", serde_json::json!({
            "secret_key": secret_key,
            "message": message
        }))?;

        if !result.success {
            return Err(HyperPhysicsError::Config(
                result.error.unwrap_or_else(|| "Sign failed".to_string())
            ));
        }

        Ok(result.data)
    }

    /// Verify Dilithium signature
    pub fn dilithium_verify(&self, public_key: &str, signature: &str, message: &str) -> Result<bool> {
        let result = self.call_tool("dilithium_verify", serde_json::json!({
            "public_key": public_key,
            "signature": signature,
            "message": message
        }))?;

        if !result.success {
            return Ok(false);
        }

        let data: serde_json::Value = serde_json::from_str(&result.data)
            .map_err(|e| HyperPhysicsError::Config(format!("Parse error: {}", e)))?;

        Ok(data["valid"].as_bool().unwrap_or(false))
    }

    /// Hash data with BLAKE3
    pub fn blake3_hash(&self, data: &str) -> Result<String> {
        let result = self.call_tool("blake3_hash", serde_json::json!({
            "data": data
        }))?;

        if !result.success {
            return Err(HyperPhysicsError::Config(
                result.error.unwrap_or_else(|| "Hash failed".to_string())
            ));
        }

        let parsed: serde_json::Value = serde_json::from_str(&result.data)
            .map_err(|e| HyperPhysicsError::Config(format!("Parse error: {}", e)))?;

        Ok(parsed["hash"].as_str().unwrap_or("").to_string())
    }

    // ========================================================================
    // Hyperbolic Geometry (H^11)
    // ========================================================================

    /// Compute hyperbolic distance in Lorentz model
    pub fn hyperbolic_distance(&self, point1: &[f64], point2: &[f64]) -> Result<f64> {
        let result = self.call_tool("hyperbolic_distance", serde_json::json!({
            "point1": point1,
            "point2": point2
        }))?;

        if !result.success {
            return Err(HyperPhysicsError::Config(
                result.error.unwrap_or_else(|| "Distance computation failed".to_string())
            ));
        }

        let data: serde_json::Value = serde_json::from_str(&result.data)
            .map_err(|e| HyperPhysicsError::Config(format!("Parse error: {}", e)))?;

        Ok(data["distance"].as_f64().unwrap_or(0.0))
    }

    /// Lift Euclidean point to Lorentz hyperboloid
    pub fn lift_to_hyperboloid(&self, point: &[f64]) -> Result<LorentzPoint> {
        let result = self.call_tool("lift_to_hyperboloid", serde_json::json!({
            "point": point
        }))?;

        if !result.success {
            return Err(HyperPhysicsError::Config(
                result.error.unwrap_or_else(|| "Lift failed".to_string())
            ));
        }

        let data: serde_json::Value = serde_json::from_str(&result.data)
            .map_err(|e| HyperPhysicsError::Config(format!("Parse error: {}", e)))?;

        let coords: Vec<f64> = data["lorentz_point"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect())
            .unwrap_or_default();

        Ok(LorentzPoint { coords })
    }

    /// Möbius addition in Poincaré ball
    pub fn mobius_add(&self, x: &[f64], y: &[f64], curvature: f64) -> Result<Vec<f64>> {
        let result = self.call_tool("mobius_add", serde_json::json!({
            "x": x,
            "y": y,
            "curvature": curvature
        }))?;

        if !result.success {
            return Err(HyperPhysicsError::Config(
                result.error.unwrap_or_else(|| "Möbius add failed".to_string())
            ));
        }

        let data: serde_json::Value = serde_json::from_str(&result.data)
            .map_err(|e| HyperPhysicsError::Config(format!("Parse error: {}", e)))?;

        Ok(data["result"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect())
            .unwrap_or_default())
    }

    // ========================================================================
    // pBit Dynamics
    // ========================================================================

    /// Sample pBit probability using Boltzmann statistics
    pub fn pbit_sample(&self, field: f64, bias: f64, temperature: f64) -> Result<f64> {
        let result = self.call_tool("pbit_sample", serde_json::json!({
            "field": field,
            "bias": bias,
            "temperature": temperature
        }))?;

        if !result.success {
            return Err(HyperPhysicsError::Config(
                result.error.unwrap_or_else(|| "pBit sampling failed".to_string())
            ));
        }

        let data: serde_json::Value = serde_json::from_str(&result.data)
            .map_err(|e| HyperPhysicsError::Config(format!("Parse error: {}", e)))?;

        Ok(data["probability"].as_f64().unwrap_or(0.5))
    }

    /// Compute Boltzmann weight exp(-E/T)
    pub fn boltzmann_weight(&self, energy: f64, temperature: f64) -> Result<f64> {
        let result = self.call_tool("boltzmann_weight", serde_json::json!({
            "energy": energy,
            "temperature": temperature
        }))?;

        if !result.success {
            return Err(HyperPhysicsError::Config(
                result.error.unwrap_or_else(|| "Boltzmann weight failed".to_string())
            ));
        }

        let data: serde_json::Value = serde_json::from_str(&result.data)
            .map_err(|e| HyperPhysicsError::Config(format!("Parse error: {}", e)))?;

        Ok(data["weight"].as_f64().unwrap_or(0.0))
    }

    /// Get Ising critical temperature (Onsager solution)
    pub fn ising_critical_temp(&self) -> Result<f64> {
        let result = self.call_tool("ising_critical_temp", serde_json::json!({}))?;

        if !result.success {
            return Ok(ISING_CRITICAL_TEMP); // Fallback to constant
        }

        let data: serde_json::Value = serde_json::from_str(&result.data)
            .map_err(|_| HyperPhysicsError::Config("Parse error".to_string()))?;

        Ok(data["critical_temperature"].as_f64().unwrap_or(ISING_CRITICAL_TEMP))
    }

    // ========================================================================
    // STDP Learning
    // ========================================================================

    /// Compute STDP weight change
    pub fn stdp_weight_change(
        &self,
        delta_t: f64,
        a_plus: Option<f64>,
        a_minus: Option<f64>,
        tau: Option<f64>,
    ) -> Result<StdpResult> {
        let result = self.call_tool("stdp_weight_change", serde_json::json!({
            "delta_t": delta_t,
            "a_plus": a_plus.unwrap_or(0.1),
            "a_minus": a_minus.unwrap_or(0.12),
            "tau": tau.unwrap_or(20.0)
        }))?;

        if !result.success {
            // Compute locally as fallback
            let a_p = a_plus.unwrap_or(0.1);
            let a_m = a_minus.unwrap_or(0.12);
            let t = tau.unwrap_or(20.0);

            let delta_weight = if delta_t > 0.0 {
                a_p * (-delta_t / t).exp()
            } else {
                -a_m * (delta_t / t).exp()
            };

            return Ok(StdpResult {
                delta_weight,
                is_ltp: delta_t > 0.0,
                delta_t,
            });
        }

        let data: serde_json::Value = serde_json::from_str(&result.data)
            .map_err(|e| HyperPhysicsError::Config(format!("Parse error: {}", e)))?;

        let delta_weight = data["weight_change"].as_f64().unwrap_or(0.0);

        Ok(StdpResult {
            delta_weight,
            is_ltp: delta_t > 0.0,
            delta_t,
        })
    }

    // ========================================================================
    // Swarm Coordination
    // ========================================================================

    /// Initialize swarm on MCP server
    pub fn swarm_init(&self, topology: &str, max_agents: usize) -> Result<SwarmState> {
        let result = self.call_tool("swarm_init", serde_json::json!({
            "topology": topology,
            "max_agents": max_agents
        }))?;

        if !result.success {
            return Err(HyperPhysicsError::Config(
                result.error.unwrap_or_else(|| "Swarm init failed".to_string())
            ));
        }

        let data: serde_json::Value = serde_json::from_str(&result.data)
            .map_err(|e| HyperPhysicsError::Config(format!("Parse error: {}", e)))?;

        Ok(SwarmState {
            id: data["swarm_id"].as_str().unwrap_or("unknown").to_string(),
            agent_count: data["agent_count"].as_u64().unwrap_or(0) as usize,
            topology: topology.to_string(),
            iteration: 0,
            best_fitness: f64::INFINITY,
        })
    }

    /// Spawn agent in swarm
    pub fn swarm_spawn_agent(&self, agent_type: &str, task: Option<&str>) -> Result<String> {
        let result = self.call_tool("swarm_spawn_agent", serde_json::json!({
            "type": agent_type,
            "task": task
        }))?;

        if !result.success {
            return Err(HyperPhysicsError::Config(
                result.error.unwrap_or_else(|| "Spawn failed".to_string())
            ));
        }

        let data: serde_json::Value = serde_json::from_str(&result.data)
            .map_err(|e| HyperPhysicsError::Config(format!("Parse error: {}", e)))?;

        Ok(data["agent_id"].as_str().unwrap_or("").to_string())
    }

    /// Get swarm status
    pub fn swarm_status(&self) -> Result<SwarmState> {
        let result = self.call_tool("swarm_status", serde_json::json!({}))?;

        if !result.success {
            return Err(HyperPhysicsError::Config(
                result.error.unwrap_or_else(|| "Status failed".to_string())
            ));
        }

        let data: serde_json::Value = serde_json::from_str(&result.data)
            .map_err(|e| HyperPhysicsError::Config(format!("Parse error: {}", e)))?;

        Ok(SwarmState {
            id: data["swarm_id"].as_str().unwrap_or("unknown").to_string(),
            agent_count: data["active_agents"].as_u64().unwrap_or(0) as usize,
            topology: data["topology"].as_str().unwrap_or("mesh").to_string(),
            iteration: data["iteration"].as_u64().unwrap_or(0) as usize,
            best_fitness: data["best_fitness"].as_f64().unwrap_or(f64::INFINITY),
        })
    }

    // ========================================================================
    // Symbolic Mathematics
    // ========================================================================

    /// Compute mathematical expression
    pub fn compute(&self, expression: &str) -> Result<String> {
        let result = self.call_tool("compute", serde_json::json!({
            "expression": expression
        }))?;

        if !result.success {
            return Err(HyperPhysicsError::Config(
                result.error.unwrap_or_else(|| "Compute failed".to_string())
            ));
        }

        Ok(result.data)
    }

    /// Symbolic mathematics operations
    pub fn symbolic(
        &self,
        operation: &str,
        expression: &str,
        variable: Option<&str>,
    ) -> Result<String> {
        let result = self.call_tool("symbolic", serde_json::json!({
            "operation": operation,
            "expression": expression,
            "variable": variable.unwrap_or("x")
        }))?;

        if !result.success {
            return Err(HyperPhysicsError::Config(
                result.error.unwrap_or_else(|| "Symbolic op failed".to_string())
            ));
        }

        Ok(result.data)
    }

    // ========================================================================
    // Design Thinking & Systems Dynamics
    // ========================================================================

    /// Run design thinking pipeline
    pub fn design_thinking(&self, challenge: &str, stage: &str) -> Result<String> {
        let result = self.call_tool("design_thinking_pipeline", serde_json::json!({
            "challenge": challenge,
            "stage": stage
        }))?;

        Ok(result.data)
    }

    /// Systems dynamics modeling
    pub fn systems_dynamics(&self, model_type: &str, params: serde_json::Value) -> Result<String> {
        let result = self.call_tool("systems_dynamics_model", serde_json::json!({
            "model_type": model_type,
            "params": params
        }))?;

        Ok(result.data)
    }
}

impl Drop for DilithiumClient {
    fn drop(&mut self) {
        let _ = self.disconnect();
    }
}

// ============================================================================
// Local Computation Fallbacks
// ============================================================================

/// Local computation functions (when MCP server unavailable)
pub mod local {
    use super::*;

    /// Compute Lorentz inner product: ⟨x,y⟩_L = -x₀y₀ + Σxᵢyᵢ
    pub fn lorentz_inner(x: &[f64], y: &[f64]) -> f64 {
        if x.is_empty() || y.is_empty() {
            return 0.0;
        }
        -x[0] * y[0] + x[1..].iter().zip(&y[1..]).map(|(a, b)| a * b).sum::<f64>()
    }

    /// Compute hyperbolic distance in Lorentz model
    pub fn hyperbolic_distance(x: &[f64], y: &[f64]) -> f64 {
        let inner = -lorentz_inner(x, y);
        stable_acosh(inner.max(1.0))
    }

    /// Stable acosh computation
    pub fn stable_acosh(x: f64) -> f64 {
        if x < 1.0001 {
            (2.0 * (x - 1.0).max(0.0)).sqrt()
        } else {
            x.acosh()
        }
    }

    /// Lift Euclidean point to hyperboloid
    pub fn lift_to_hyperboloid(z: &[f64]) -> Vec<f64> {
        let norm_sq: f64 = z.iter().map(|x| x * x).sum();
        let mut result = Vec::with_capacity(z.len() + 1);
        result.push((1.0 + norm_sq).sqrt());
        result.extend(z);
        result
    }

    /// Möbius addition in Poincaré ball
    pub fn mobius_add(x: &[f64], y: &[f64], curvature: f64) -> Vec<f64> {
        let c = -curvature;
        let xy: f64 = x.iter().zip(y).map(|(a, b)| a * b).sum();
        let x_norm_sq: f64 = x.iter().map(|a| a * a).sum();
        let y_norm_sq: f64 = y.iter().map(|a| a * a).sum();

        let denom = 1.0 + 2.0 * c * xy + c * c * x_norm_sq * y_norm_sq;
        let coef_x = 1.0 + 2.0 * c * xy + c * y_norm_sq;
        let coef_y = 1.0 - c * x_norm_sq;

        x.iter()
            .zip(y)
            .map(|(xi, yi)| (coef_x * xi + coef_y * yi) / denom)
            .collect()
    }

    /// pBit probability with Boltzmann statistics
    pub fn pbit_probability(field: f64, bias: f64, temperature: f64) -> f64 {
        let x = (field - bias) / temperature.max(1e-10);
        1.0 / (1.0 + (-x).exp())
    }

    /// Boltzmann weight
    pub fn boltzmann_weight(energy: f64, temperature: f64) -> f64 {
        (-energy / temperature.max(1e-10)).exp()
    }

    /// STDP weight change
    pub fn stdp_weight_change(delta_t: f64, a_plus: f64, a_minus: f64, tau: f64) -> f64 {
        if delta_t > 0.0 {
            a_plus * (-delta_t / tau).exp()
        } else {
            -a_minus * (delta_t / tau).exp()
        }
    }

    /// Fibonacci STDP with multi-scale time constants
    pub fn fibonacci_stdp(delta_t: f64) -> f64 {
        const TAU: [f64; 5] = [13.0, 21.0, 34.0, 55.0, 89.0];
        const A_PLUS: f64 = 0.1;
        const A_MINUS: f64 = 0.12;

        let mut dw = 0.0;
        let abs_dt = delta_t.abs();

        for (i, &tau) in TAU.iter().enumerate() {
            let amplitude = if delta_t > 0.0 {
                A_PLUS * PHI_INV.powi(i as i32)
            } else {
                A_MINUS * PHI_INV.powi(i as i32)
            };
            let decay = (-abs_dt / tau).exp();
            dw += amplitude * decay;
        }

        if delta_t > 0.0 {
            dw.min(1.0)
        } else {
            -dw.min(1.0)
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::local::*;

    #[test]
    fn test_lorentz_inner() {
        let x = vec![1.0, 0.0, 0.0];
        let y = vec![1.0, 0.0, 0.0];
        assert!((lorentz_inner(&x, &y) + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hyperbolic_distance_same_point() {
        let p = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        assert!(hyperbolic_distance(&p, &p).abs() < 1e-10);
    }

    #[test]
    fn test_lift_to_hyperboloid() {
        let z = vec![0.0, 0.0, 0.0];
        let lifted = lift_to_hyperboloid(&z);
        assert_eq!(lifted[0], 1.0); // Time component
        assert!(lifted[1..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_pbit_probability_bounds() {
        for field in [-10.0, 0.0, 10.0] {
            let p = pbit_probability(field, 0.0, 1.0);
            assert!(p >= 0.0 && p <= 1.0);
        }
    }

    #[test]
    fn test_stdp_symmetry() {
        let ltp = stdp_weight_change(10.0, 0.1, 0.12, 20.0);
        let ltd = stdp_weight_change(-10.0, 0.1, 0.12, 20.0);
        assert!(ltp > 0.0);
        assert!(ltd < 0.0);
    }

    #[test]
    fn test_fibonacci_stdp() {
        let ltp = fibonacci_stdp(10.0);
        let ltd = fibonacci_stdp(-10.0);
        assert!(ltp > 0.0);
        assert!(ltd < 0.0);
    }

    #[test]
    fn test_lorentz_point_origin() {
        let origin = LorentzPoint::origin();
        assert!(origin.is_valid());
    }

    #[test]
    fn test_mcp_config_builder() {
        let config = McpConfig::with_path("/custom/path")
            .native_module("/native/module.so")
            .timeout(60000);

        assert_eq!(config.server_path, "/custom/path");
        assert_eq!(config.timeout_ms, 60000);
        assert!(config.native_path.is_some());
    }
}
