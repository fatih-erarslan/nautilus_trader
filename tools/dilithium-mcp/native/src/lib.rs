//! # Dilithium Native - Post-Quantum Secure MCP Infrastructure
//!
//! Native Rust bindings for the Dilithium MCP Server providing:
//! - Post-quantum cryptographic authentication (Dilithium ML-DSA)
//! - Hyperbolic geometry computations
//! - pBit dynamics engine
//! - Symbolic mathematics
//!
//! ## Security Model
//!
//! All MCP requests are authenticated using Dilithium digital signatures,
//! providing resistance against quantum computer attacks.

use napi_derive::napi;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use parking_lot::RwLock;
use dashmap::DashMap;
use pqcrypto_traits::sign::{SecretKey as SkTrait, DetachedSignature as SigTrait};

// ============================================================================
// Dilithium Cryptography Module
// ============================================================================

pub mod dilithium {
    use pqcrypto_dilithium::dilithium3;
    use pqcrypto_traits::sign::{PublicKey, SecretKey, DetachedSignature, SignedMessage};
    
    /// Dilithium key pair
    #[derive(Clone)]
    pub struct DilithiumKeyPair {
        pub public_key: dilithium3::PublicKey,
        pub secret_key: dilithium3::SecretKey,
    }
    
    impl DilithiumKeyPair {
        /// Generate a new key pair
        pub fn generate() -> Self {
            let (pk, sk) = dilithium3::keypair();
            Self {
                public_key: pk,
                secret_key: sk,
            }
        }
        
        /// Sign a message
        pub fn sign(&self, message: &[u8]) -> Vec<u8> {
            let sm = dilithium3::sign(message, &self.secret_key);
            sm.as_bytes().to_vec()
        }
        
        /// Sign with detached signature
        pub fn sign_detached(&self, message: &[u8]) -> Vec<u8> {
            let sig = dilithium3::detached_sign(message, &self.secret_key);
            sig.as_bytes().to_vec()
        }
        
        /// Verify a detached signature
        pub fn verify_detached(&self, signature: &[u8], message: &[u8]) -> bool {
            if let Ok(sig) = dilithium3::DetachedSignature::from_bytes(signature) {
                dilithium3::verify_detached_signature(&sig, message, &self.public_key).is_ok()
            } else {
                false
            }
        }
        
        /// Export public key
        pub fn public_key_bytes(&self) -> Vec<u8> {
            self.public_key.as_bytes().to_vec()
        }
        
        /// Export secret key (use with caution)
        pub fn secret_key_bytes(&self) -> Vec<u8> {
            self.secret_key.as_bytes().to_vec()
        }
    }
    
    /// Verify signature with public key
    pub fn verify_with_public_key(public_key_bytes: &[u8], signature: &[u8], message: &[u8]) -> bool {
        if let Ok(pk) = dilithium3::PublicKey::from_bytes(public_key_bytes) {
            if let Ok(sig) = dilithium3::DetachedSignature::from_bytes(signature) {
                dilithium3::verify_detached_signature(&sig, message, &pk).is_ok()
            } else {
                false
            }
        } else {
            false
        }
    }
}

// ============================================================================
// MCP Authentication Types
// ============================================================================

/// Authenticated MCP request
#[derive(Debug, Clone, Serialize, Deserialize)]
#[napi(object)]
pub struct AuthenticatedRequest {
    /// Client's public key (hex encoded)
    pub client_id: String,
    /// Request timestamp (ISO 8601)
    pub timestamp: String,
    /// Request nonce (prevents replay)
    pub nonce: String,
    /// Request payload (JSON string)
    pub payload: String,
    /// Dilithium signature (hex encoded)
    pub signature: String,
}

/// Authentication result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[napi(object)]
pub struct AuthResult {
    pub valid: bool,
    pub client_id: String,
    pub error: Option<String>,
    pub timestamp: String,
}

/// Client registration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[napi(object)]
pub struct ClientRegistration {
    pub client_id: String,
    pub public_key: String,
    pub registered_at: String,
    pub capabilities: Vec<String>,
    pub quota_remaining: i64,
}

// ============================================================================
// NAPI Exports - Dilithium Crypto
// ============================================================================

/// Generate a new Dilithium key pair
#[napi]
pub fn dilithium_keygen() -> KeyPairResult {
    let kp = dilithium::DilithiumKeyPair::generate();
    KeyPairResult {
        public_key: hex::encode(&kp.public_key_bytes()),
        secret_key: hex::encode(&kp.secret_key_bytes()),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[napi(object)]
pub struct KeyPairResult {
    pub public_key: String,
    pub secret_key: String,
}

/// Sign a message with Dilithium
#[napi]
pub fn dilithium_sign(secret_key_hex: String, message: String) -> String {
    let sk_bytes = hex::decode(&secret_key_hex).unwrap_or_default();
    if let Ok(sk) = <pqcrypto_dilithium::dilithium3::SecretKey as SkTrait>::from_bytes(&sk_bytes) {
        let sig = pqcrypto_dilithium::dilithium3::detached_sign(message.as_bytes(), &sk);
        hex::encode(sig.as_bytes())
    } else {
        String::new()
    }
}

/// Verify a Dilithium signature
#[napi]
pub fn dilithium_verify(public_key_hex: String, signature_hex: String, message: String) -> bool {
    let pk_bytes = hex::decode(&public_key_hex).unwrap_or_default();
    let sig_bytes = hex::decode(&signature_hex).unwrap_or_default();
    dilithium::verify_with_public_key(&pk_bytes, &sig_bytes, message.as_bytes())
}

/// Hash data with BLAKE3
#[napi]
pub fn blake3_hash(data: String) -> String {
    let hash = blake3::hash(data.as_bytes());
    hex::encode(hash.as_bytes())
}

/// Generate secure nonce
#[napi]
pub fn generate_nonce() -> String {
    let mut bytes = [0u8; 32];
    getrandom::getrandom(&mut bytes).unwrap_or_default();
    hex::encode(&bytes)
}

// ============================================================================
// NAPI Exports - Hyperbolic Geometry
// ============================================================================

/// Lorentz inner product for H^11
#[napi]
pub fn lorentz_inner(x: Vec<f64>, y: Vec<f64>) -> f64 {
    if x.len() != 12 || y.len() != 12 {
        return f64::NAN;
    }
    -x[0] * y[0] + x[1..].iter().zip(y[1..].iter()).map(|(a, b)| a * b).sum::<f64>()
}

/// Hyperbolic distance in H^11
#[napi]
pub fn hyperbolic_distance(x: Vec<f64>, y: Vec<f64>) -> f64 {
    let inner = -lorentz_inner(x, y);
    if inner < 1.0 {
        return 0.0;
    }
    inner.acosh()
}

/// Lift Euclidean to Lorentz hyperboloid
#[napi]
pub fn lift_to_hyperboloid(z: Vec<f64>) -> Vec<f64> {
    let spatial_norm_sq: f64 = z.iter().map(|x| x * x).sum();
    let x0 = (1.0 + spatial_norm_sq).sqrt();
    
    let mut result = vec![x0];
    result.extend(z);
    result
}

/// Mobius addition in Poincare ball
#[napi]
pub fn mobius_add(x: Vec<f64>, y: Vec<f64>, curvature: f64) -> Vec<f64> {
    let c = -curvature;
    
    let xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let x_norm_sq: f64 = x.iter().map(|v| v * v).sum();
    let y_norm_sq: f64 = y.iter().map(|v| v * v).sum();
    
    let denom = 1.0 + 2.0 * c * xy + c * c * x_norm_sq * y_norm_sq;
    let coef_x = 1.0 + 2.0 * c * xy + c * y_norm_sq;
    let coef_y = 1.0 - c * x_norm_sq;
    
    x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (coef_x * xi + coef_y * yi) / denom)
        .collect()
}

// ============================================================================
// NAPI Exports - pBit Dynamics
// ============================================================================

/// pBit sampling probability: P(s=1) = sigmoid((h - bias) / T)
#[napi]
pub fn pbit_probability(field: f64, bias: f64, temperature: f64) -> f64 {
    let x = (field - bias) / temperature.max(1e-10);
    1.0 / (1.0 + (-x).exp())
}

/// Batch pBit probabilities
#[napi]
pub fn pbit_probabilities_batch(fields: Vec<f64>, biases: Vec<f64>, temperature: f64) -> Vec<f64> {
    fields.iter()
        .zip(biases.iter())
        .map(|(&h, &b)| pbit_probability(h, b, temperature))
        .collect()
}

/// Boltzmann weight: exp(-E/T)
#[napi]
pub fn boltzmann_weight(energy: f64, temperature: f64) -> f64 {
    (-energy / temperature.max(1e-10)).exp()
}

/// Ising critical temperature (2D square lattice)
#[napi]
pub fn ising_critical_temp() -> f64 {
    2.0 / (1.0 + 2.0_f64.sqrt()).ln()
}

/// STDP weight change
#[napi]
pub fn stdp_weight_change(delta_t: f64, a_plus: f64, a_minus: f64, tau: f64) -> f64 {
    if delta_t > 0.0 {
        a_plus * (-delta_t / tau).exp()
    } else {
        -a_minus * (delta_t / tau).exp()
    }
}

// ============================================================================
// NAPI Exports - Symbolic Math (Local Computation)
// ============================================================================

/// Fast exp approximation (6th order Remez)
#[napi]
pub fn fast_exp(x: f64) -> f64 {
    use std::f64::consts::{LN_2 as LN2, LOG2_E as INV_LN2};
    
    let x_clamped = x.clamp(-87.0, 88.0);
    let k = (x_clamped * INV_LN2).floor();
    let r = x_clamped - k * LN2;
    
    // Horner's method for polynomial
    let mut result = 1.0 / 720.0;
    result = result * r + 1.0 / 120.0;
    result = result * r + 1.0 / 24.0;
    result = result * r + 1.0 / 6.0;
    result = result * r + 0.5;
    result = result * r + 1.0;
    result = result * r + 1.0;
    
    result * 2.0_f64.powi(k as i32)
}

/// Stable acosh for hyperbolic distance
#[napi]
pub fn stable_acosh(x: f64) -> f64 {
    if x < 1.0 + 1e-10 {
        (2.0 * (x - 1.0).max(0.0)).sqrt()
    } else {
        x.acosh()
    }
}

// ============================================================================
// Server State Management
// ============================================================================

/// Global server state (thread-safe)
static SERVER_STATE: once_cell::sync::Lazy<Arc<ServerState>> = 
    once_cell::sync::Lazy::new(|| Arc::new(ServerState::new()));

pub struct ServerState {
    /// Registered clients
    clients: DashMap<String, ClientRegistration>,
    /// Used nonces (for replay protection)
    used_nonces: DashMap<String, chrono::DateTime<chrono::Utc>>,
    /// Server key pair
    server_keypair: RwLock<Option<dilithium::DilithiumKeyPair>>,
}

impl Default for ServerState {
    fn default() -> Self {
        Self::new()
    }
}

impl ServerState {
    pub fn new() -> Self {
        Self {
            clients: DashMap::new(),
            used_nonces: DashMap::new(),
            server_keypair: RwLock::new(None),
        }
    }
    
    pub fn init_server_keys(&self) {
        let mut kp = self.server_keypair.write();
        if kp.is_none() {
            *kp = Some(dilithium::DilithiumKeyPair::generate());
        }
    }
}

/// Initialize server with key pair
#[napi]
pub fn init_server() -> String {
    SERVER_STATE.init_server_keys();
    let kp = SERVER_STATE.server_keypair.read();
    if let Some(ref keypair) = *kp {
        hex::encode(&keypair.public_key_bytes())
    } else {
        String::new()
    }
}

/// Register a client
#[napi]
pub fn register_client(client_id: String, public_key: String, capabilities: Vec<String>) -> bool {
    let registration = ClientRegistration {
        client_id: client_id.clone(),
        public_key,
        registered_at: chrono::Utc::now().to_rfc3339(),
        capabilities,
        quota_remaining: 10000, // Default quota
    };
    SERVER_STATE.clients.insert(client_id, registration);
    true
}

/// Verify authenticated request
#[napi]
pub fn verify_request(request: AuthenticatedRequest) -> AuthResult {
    let now = chrono::Utc::now();
    
    // Check nonce hasn't been used
    if SERVER_STATE.used_nonces.contains_key(&request.nonce) {
        return AuthResult {
            valid: false,
            client_id: request.client_id,
            error: Some("Nonce already used (replay attack detected)".into()),
            timestamp: now.to_rfc3339(),
        };
    }
    
    // Get client's public key
    let client = match SERVER_STATE.clients.get(&request.client_id) {
        Some(c) => c.clone(),
        None => {
            return AuthResult {
                valid: false,
                client_id: request.client_id,
                error: Some("Client not registered".into()),
                timestamp: now.to_rfc3339(),
            };
        }
    };
    
    // Construct message to verify
    let message = format!("{}{}{}", request.timestamp, request.nonce, request.payload);
    
    // Verify signature
    let pk_bytes = hex::decode(&client.public_key).unwrap_or_default();
    let sig_bytes = hex::decode(&request.signature).unwrap_or_default();
    
    if !dilithium::verify_with_public_key(&pk_bytes, &sig_bytes, message.as_bytes()) {
        return AuthResult {
            valid: false,
            client_id: request.client_id,
            error: Some("Invalid signature".into()),
            timestamp: now.to_rfc3339(),
        };
    }
    
    // Mark nonce as used
    SERVER_STATE.used_nonces.insert(request.nonce, now);
    
    AuthResult {
        valid: true,
        client_id: request.client_id,
        error: None,
        timestamp: now.to_rfc3339(),
    }
}

// ============================================================================
// HyperPhysics Agency Integration
// ============================================================================

use hyperphysics_agency::{
    CyberneticAgent, AgencyConfig, AgentState, Observation,
    FreeEnergyEngine, SurvivalDrive, HomeostaticController, AgencyDynamics,
};
use ndarray::Array1;

/// Agent storage for persistent agents
static AGENT_REGISTRY: once_cell::sync::Lazy<Arc<DashMap<String, Arc<parking_lot::RwLock<CyberneticAgent>>>>> =
    once_cell::sync::Lazy::new(|| Arc::new(DashMap::new()));

// ============================================================================
// NAPI Exports - Agency Functions
// ============================================================================

/// Result for agency operations
#[derive(Debug, Clone, Serialize, Deserialize)]
#[napi(object)]
pub struct AgencyResult {
    pub success: bool,
    pub agent_id: Option<String>,
    pub error: Option<String>,
    pub data: Option<String>, // JSON-encoded data
}

/// Create a new cybernetic agent
///
/// # Arguments
/// * `config_json` - JSON string with AgencyConfig fields:
///   - observation_dim: number
///   - action_dim: number
///   - hidden_dim: number
///   - learning_rate?: number (default: 0.01)
///   - survival_strength?: number (default: 1.0)
///
/// # Returns
/// AgencyResult with agent_id on success
#[napi]
pub fn agency_create_agent(config_json: String) -> AgencyResult {
    // Parse configuration
    let config: AgencyConfig = match serde_json::from_str(&config_json) {
        Ok(c) => c,
        Err(e) => {
            return AgencyResult {
                success: false,
                agent_id: None,
                error: Some(format!("Failed to parse config: {}", e)),
                data: None,
            };
        }
    };

    // Create agent
    let agent = CyberneticAgent::new(config);

    // Generate unique ID
    let agent_id = uuid::Uuid::new_v4().to_string();

    // Store agent
    AGENT_REGISTRY.insert(agent_id.clone(), Arc::new(parking_lot::RwLock::new(agent)));

    AgencyResult {
        success: true,
        agent_id: Some(agent_id),
        error: None,
        data: None,
    }
}

/// Execute one agent step: observation → inference → action
///
/// # Arguments
/// * `agent_id` - Agent ID from agency_create_agent
/// * `observation_json` - JSON array of observation values
///
/// # Returns
/// AgencyResult with JSON data containing:
/// - action: number[] (motor commands)
/// - state: { phi, free_energy, survival, control, model_accuracy }
/// - metrics: { phi, free_energy, survival, control }
#[napi]
pub fn agency_agent_step(agent_id: String, observation_json: String) -> AgencyResult {
    // Get agent
    let agent_lock = match AGENT_REGISTRY.get(&agent_id) {
        Some(a) => a.clone(),
        None => {
            return AgencyResult {
                success: false,
                agent_id: Some(agent_id),
                error: Some("Agent not found".into()),
                data: None,
            };
        }
    };

    // Parse observation
    let obs_vec: Vec<f64> = match serde_json::from_str(&observation_json) {
        Ok(v) => v,
        Err(e) => {
            return AgencyResult {
                success: false,
                agent_id: Some(agent_id),
                error: Some(format!("Failed to parse observation: {}", e)),
                data: None,
            };
        }
    };

    let observation = Observation {
        sensory: Array1::from_vec(obs_vec),
        timestamp: chrono::Utc::now().timestamp() as u64,
    };

    // Execute step
    let mut agent = agent_lock.write();
    let action = agent.step(&observation);

    // Collect results
    let result = serde_json::json!({
        "action": action.motor.to_vec(),
        "state": {
            "phi": agent.state.phi,
            "free_energy": agent.state.free_energy,
            "survival": agent.state.survival,
            "control": agent.state.control,
            "model_accuracy": agent.state.model_accuracy,
        },
        "metrics": {
            "phi": agent.integrated_information(),
            "free_energy": agent.free_energy(),
            "survival": agent.survival_drive(),
            "control": agent.control_authority(),
        },
    });

    AgencyResult {
        success: true,
        agent_id: Some(agent_id),
        error: None,
        data: Some(result.to_string()),
    }
}

/// Compute variational free energy: F = D_KL[q||p] + accuracy
///
/// # Arguments
/// * `observation` - Observation vector
/// * `beliefs` - Belief vector (approximate posterior)
/// * `precision` - Precision vector (inverse variance)
///
/// # Returns
/// AgencyResult with JSON data:
/// - F: total free energy (nats)
/// - complexity: KL divergence term
/// - accuracy: prediction accuracy term
#[napi]
pub fn agency_compute_free_energy(
    observation: Vec<f64>,
    beliefs: Vec<f64>,
    precision: Vec<f64>,
) -> AgencyResult {
    let mut engine = FreeEnergyEngine::new(beliefs.len());

    let obs = Array1::from_vec(observation);
    let bel = Array1::from_vec(beliefs);
    let prec = Array1::from_vec(precision);

    let free_energy = engine.compute(&obs, &bel, &prec);

    let result = serde_json::json!({
        "F": free_energy,
        "complexity": engine.accumulated_surprise(),
        "accuracy": free_energy - engine.accumulated_surprise(),
    });

    AgencyResult {
        success: true,
        agent_id: None,
        error: None,
        data: Some(result.to_string()),
    }
}

/// Compute survival drive from free energy and position
///
/// # Arguments
/// * `free_energy` - Current free energy (F)
/// * `position` - Position in H^11 (12D Lorentz coordinates)
///
/// # Returns
/// AgencyResult with JSON data:
/// - drive: survival urgency [0,1]
/// - threat_level: threat assessment [0,1]
/// - homeostatic_status: "safe" | "stressed" | "critical"
#[napi]
pub fn agency_compute_survival_drive(free_energy: f64, position: Vec<f64>) -> AgencyResult {
    if position.len() != 12 {
        return AgencyResult {
            success: false,
            agent_id: None,
            error: Some("Position must be 12D Lorentz coordinates".into()),
            data: None,
        };
    }

    let mut survival = SurvivalDrive::new(1.0);
    let pos = Array1::from_vec(position);

    let drive = survival.compute_drive(free_energy, &pos);
    let distance = survival.compute_distance(&pos);
    let assessment = survival.threat_assessment(free_energy, distance);

    let result = serde_json::json!({
        "drive": drive,
        "threat_level": assessment.level,
        "homeostatic_status": survival.homeostatic_status(),
        "in_crisis": survival.in_crisis,
        "threat_assessment": {
            "detected": assessment.detected,
            "free_energy_contribution": assessment.free_energy_contribution,
            "distance_contribution": assessment.distance_contribution,
            "rate_of_change": assessment.rate_of_change,
        },
    });

    AgencyResult {
        success: true,
        agent_id: None,
        error: None,
        data: Some(result.to_string()),
    }
}

/// Compute integrated information Φ (consciousness metric)
///
/// # Arguments
/// * `network_state` - Neural network state vector
///
/// # Returns
/// AgencyResult with JSON data:
/// - phi: integrated information (bits)
/// - consciousness_level: "minimal" | "moderate" | "high"
#[napi]
pub fn agency_compute_phi(network_state: Vec<f64>) -> AgencyResult {
    // Simplified Φ calculation based on coherence
    let state = Array1::from_vec(network_state);

    let coherence = state.iter()
        .map(|x| x.abs())
        .sum::<f64>() / state.len() as f64;

    let phi = coherence.max(0.0).min(10.0);

    let consciousness_level = if phi < 0.5 {
        "minimal"
    } else if phi < 2.0 {
        "moderate"
    } else {
        "high"
    };

    let result = serde_json::json!({
        "phi": phi,
        "consciousness_level": consciousness_level,
        "coherence": coherence,
    });

    AgencyResult {
        success: true,
        agent_id: None,
        error: None,
        data: Some(result.to_string()),
    }
}

/// Analyze criticality (self-organized criticality markers)
///
/// # Arguments
/// * `timeseries` - Activity time series
///
/// # Returns
/// AgencyResult with JSON data:
/// - branching_ratio: σ (≈1.0 at criticality)
/// - at_criticality: boolean
/// - hurst_exponent: H (long-range correlations)
#[napi]
pub fn agency_analyze_criticality(timeseries: Vec<f64>) -> AgencyResult {
    if timeseries.len() < 10 {
        return AgencyResult {
            success: false,
            agent_id: None,
            error: Some("Time series too short (need at least 10 points)".into()),
            data: None,
        };
    }

    let mut dynamics = AgencyDynamics::new();

    // Create dummy agent states from time series
    for &phi in &timeseries {
        let mut position = Array1::zeros(12);
        position[0] = 1.0;

        let state = AgentState {
            position,
            beliefs: Array1::from_elem(32, 0.0),
            precision: Array1::from_elem(32, 1.0),
            prediction_errors: std::collections::VecDeque::new(),
            phi,
            free_energy: 1.0,
            control: 0.5,
            survival: 0.5,
            model_accuracy: 0.8,
        };

        dynamics.record_state(&state);
    }

    let branching_ratio = dynamics.branching_ratio().unwrap_or(1.0);
    let at_criticality = (branching_ratio - 1.0).abs() < 0.1;
    let hurst = dynamics.hurst_exponent().unwrap_or(0.5);

    let result = serde_json::json!({
        "branching_ratio": branching_ratio,
        "at_criticality": at_criticality,
        "hurst_exponent": hurst,
        "criticality_distance": (branching_ratio - 1.0).abs(),
    });

    AgencyResult {
        success: true,
        agent_id: None,
        error: None,
        data: Some(result.to_string()),
    }
}

/// Perform homeostatic regulation
///
/// # Arguments
/// * `current_state_json` - JSON with { phi, free_energy, survival }
/// * `setpoints_json` - Optional JSON with { phi_setpoint, fe_setpoint, survival_setpoint }
///
/// # Returns
/// AgencyResult with JSON data:
/// - control_signals: { phi_correction, fe_correction, survival_correction }
/// - allostatic_adjustment: number
/// - disturbance_rejection: number
#[napi]
pub fn agency_regulate_homeostasis(
    current_state_json: String,
    setpoints_json: Option<String>,
) -> AgencyResult {
    // Parse current state
    let state_data: serde_json::Value = match serde_json::from_str(&current_state_json) {
        Ok(v) => v,
        Err(e) => {
            return AgencyResult {
                success: false,
                agent_id: None,
                error: Some(format!("Failed to parse current_state: {}", e)),
                data: None,
            };
        }
    };

    // Create agent state
    let mut position = Array1::zeros(12);
    position[0] = 1.0;

    let mut state = AgentState {
        position,
        beliefs: Array1::from_elem(32, 0.0),
        precision: Array1::from_elem(32, 1.0),
        prediction_errors: std::collections::VecDeque::new(),
        phi: state_data["phi"].as_f64().unwrap_or(1.0),
        free_energy: state_data["free_energy"].as_f64().unwrap_or(1.0),
        control: state_data.get("control").and_then(|v| v.as_f64()).unwrap_or(0.5),
        survival: state_data["survival"].as_f64().unwrap_or(0.5),
        model_accuracy: state_data.get("model_accuracy").and_then(|v| v.as_f64()).unwrap_or(0.8),
    };

    // Create controller
    let mut controller = HomeostaticController::new();

    // Set custom setpoints if provided
    if let Some(sp_json) = setpoints_json {
        if let Ok(sp_data) = serde_json::from_str::<serde_json::Value>(&sp_json) {
            let phi_sp = sp_data.get("phi_setpoint").and_then(|v| v.as_f64()).unwrap_or(2.0);
            let fe_sp = sp_data.get("fe_setpoint").and_then(|v| v.as_f64()).unwrap_or(0.5);
            let surv_sp = sp_data.get("survival_setpoint").and_then(|v| v.as_f64()).unwrap_or(0.3);
            controller.set_setpoints(phi_sp, fe_sp, surv_sp);
        }
    }

    // Store initial values
    let initial_phi = state.phi;
    let initial_fe = state.free_energy;
    let initial_survival = state.survival;

    // Regulate
    controller.regulate(&mut state);

    // Compute corrections (changes made by controller)
    let phi_correction = state.phi - initial_phi;
    let fe_correction = state.free_energy - initial_fe;
    let survival_correction = state.survival - initial_survival;

    let result = serde_json::json!({
        "control_signals": {
            "phi_correction": phi_correction,
            "fe_correction": fe_correction,
            "survival_correction": survival_correction,
        },
        "allostatic_adjustment": controller.allostatic_adjustment(),
        "disturbance_rejection": controller.disturbance_rejection(),
        "prediction_confidence": controller.prediction_confidence(),
        "regulated_state": {
            "phi": state.phi,
            "free_energy": state.free_energy,
            "survival": state.survival,
        },
    });

    AgencyResult {
        success: true,
        agent_id: None,
        error: None,
        data: Some(result.to_string()),
    }
}

// ============================================================================
// Utility
// ============================================================================

mod hex {
    pub fn encode(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }

    pub fn decode(s: &str) -> Result<Vec<u8>, ()> {
        if s.len() % 2 != 0 {
            return Err(());
        }
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|_| ()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dilithium_keygen_sign_verify() {
        let kp = dilithium::DilithiumKeyPair::generate();
        let message = b"test message";
        let signature = kp.sign_detached(message);
        assert!(kp.verify_detached(&signature, message));
    }
    
    #[test]
    fn test_lorentz_inner() {
        let origin = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let inner = lorentz_inner(origin.clone(), origin);
        assert!((inner + 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_pbit_probability() {
        let p = pbit_probability(0.0, 0.0, 1.0);
        assert!((p - 0.5).abs() < 1e-10);
    }
    
    #[test]
    fn test_ising_critical() {
        let tc = ising_critical_temp();
        assert!((tc - 2.269185314213022).abs() < 1e-10);
    }
}
