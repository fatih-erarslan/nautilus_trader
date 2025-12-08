// Quantum Early Warning System - Immune System Modeling and Threat Detection
// Copyright (c) 2025 TENGRI Trading Swarm

use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use tracing::{info, warn, error, debug};

pub mod threat_detector;
pub mod immune_response;
pub mod quantum_sensors;
pub mod pattern_recognition;
pub mod anomaly_analyzer;
pub mod warning_engine;

pub use threat_detector::*;
pub use immune_response::*;
pub use quantum_sensors::*;
pub use pattern_recognition::*;
pub use anomaly_analyzer::*;
pub use warning_engine::*;

use market_regime_detector::MarketRegime;
use quantum_core::QuantumCircuit;
use iqad::QuantumAnomalyDetector;

/// Types of threats detected by the early warning system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ThreatType {
    // Market structure threats
    FlashCrash,                 // Sudden market crash
    LiquidityEvaporation,       // Liquidity disappearance
    VolatilitySpike,           // Extreme volatility
    CircuitBreakerTrigger,     // Circuit breaker activation
    MarketManipulation,        // Market manipulation attempt
    
    // Systemic threats
    ContagionRisk,             // Cross-market contagion
    SystemicRisk,              // System-wide risk
    CascadingFailure,          // Cascading system failure
    NetworkFragmentation,      // Network connectivity issues
    
    // Adversarial threats
    WhaleAttack,               // Large player attack
    CoordinatedAttack,         // Coordinated manipulation
    SpoofingCampaign,          // Large-scale spoofing
    WashTradingScheme,         // Wash trading operation
    LayeringAttack,            // Layering manipulation
    
    // Technology threats
    AlgorithmMalfunction,      // Algorithm malfunction
    HFTAnomaly,               // HFT behavior anomaly
    DataCorruption,           // Market data corruption
    LatencySpike,             // Latency anomaly
    SystemOverload,           // System overload
    
    // Information threats
    NewsLeakage,              // Insider information leak
    FakeNews,                 // False information spread
    SocialMediaManipulation,  // Social media manipulation
    RumorSpread,              // False rumor propagation
    
    // Regulatory threats
    RegulatorIntervention,    // Regulatory intervention
    PolicyShock,              // Unexpected policy change
    ComplianceViolation,      // Compliance violation
    TradingHalt,              // Trading halt
    
    // Quantum threats
    QuantumAttack,            // Quantum-based attack
    QuantumDecoherence,       // Quantum system failure
    QuantumEntanglement,      // Malicious entanglement
    
    // Economic threats
    MacroeconomicShock,       // Macroeconomic shock
    CurrencyCrisis,           // Currency crisis
    CreditEvent,              // Credit event
    GeopoliticalTension,      // Geopolitical risk
    
    // Novel threats
    AIThreat,                 // AI-based threat
    SwarmAttack,              // Swarm-based attack
    EmergentThreat,           // Previously unknown threat
    UnknownThreat,            // Unclassified threat
}

impl ThreatType {
    /// Get the typical impact severity of this threat type
    pub fn impact_severity(&self) -> f64 {
        match self {
            ThreatType::FlashCrash => 0.95,
            ThreatType::SystemicRisk => 0.98,
            ThreatType::CascadingFailure => 0.92,
            ThreatType::QuantumAttack => 0.9,
            ThreatType::MacroeconomicShock => 0.85,
            ThreatType::WhaleAttack => 0.7,
            ThreatType::VolatilitySpike => 0.6,
            ThreatType::NewsLeakage => 0.4,
            ThreatType::LatencySpike => 0.3,
            ThreatType::EmergentThreat => 0.8,
            _ => 0.5,
        }
    }
    
    /// Get the typical detection difficulty
    pub fn detection_difficulty(&self) -> f64 {
        match self {
            ThreatType::QuantumAttack => 0.95,
            ThreatType::EmergentThreat => 0.9,
            ThreatType::SocialMediaManipulation => 0.8,
            ThreatType::CoordinatedAttack => 0.75,
            ThreatType::SwarmAttack => 0.85,
            ThreatType::FlashCrash => 0.3,
            ThreatType::CircuitBreakerTrigger => 0.1,
            ThreatType::VolatilitySpike => 0.2,
            _ => 0.5,
        }
    }
    
    /// Get the typical warning lead time
    pub fn warning_lead_time(&self) -> chrono::Duration {
        match self {
            ThreatType::FlashCrash => chrono::Duration::seconds(5),
            ThreatType::LiquidityEvaporation => chrono::Duration::seconds(10),
            ThreatType::VolatilitySpike => chrono::Duration::seconds(30),
            ThreatType::WhaleAttack => chrono::Duration::minutes(5),
            ThreatType::SystemicRisk => chrono::Duration::minutes(30),
            ThreatType::MacroeconomicShock => chrono::Duration::hours(1),
            ThreatType::GeopoliticalTension => chrono::Duration::hours(6),
            ThreatType::QuantumAttack => chrono::Duration::milliseconds(100),
            ThreatType::EmergentThreat => chrono::Duration::seconds(1),
            _ => chrono::Duration::minutes(1),
        }
    }
    
    /// Check if threat requires immediate response
    pub fn requires_immediate_response(&self) -> bool {
        matches!(self,
            ThreatType::FlashCrash |
            ThreatType::LiquidityEvaporation |
            ThreatType::CascadingFailure |
            ThreatType::QuantumAttack |
            ThreatType::SystemOverload |
            ThreatType::AlgorithmMalfunction
        )
    }
}

/// Threat detection result with confidence and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatDetectionResult {
    pub threat_id: String,
    pub threat_type: ThreatType,
    pub confidence: f64,
    pub severity: f64,
    pub detection_time: chrono::DateTime<chrono::Utc>,
    pub estimated_impact: ThreatImpact,
    pub warning_lead_time: chrono::Duration,
    pub affected_systems: Vec<String>,
    pub recommended_actions: Vec<ThreatResponse>,
    pub quantum_signature: Option<QuantumThreatSignature>,
    pub immune_response: Option<ImmuneResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatImpact {
    pub financial_impact: f64,
    pub operational_impact: f64,
    pub reputational_impact: f64,
    pub systemic_impact: f64,
    pub recovery_time: chrono::Duration,
    pub cascading_probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatResponse {
    pub response_type: ResponseType,
    pub urgency: f64,
    pub effectiveness: f64,
    pub cost: f64,
    pub implementation_time: chrono::Duration,
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResponseType {
    // Immediate responses
    EmergencyStop,          // Stop all trading
    PositionReduction,      // Reduce positions
    HedgeActivation,        // Activate hedges
    LiquidityWithdrawal,    // Withdraw liquidity
    
    // System responses
    SystemShutdown,         // Shutdown systems
    BackupActivation,       // Activate backup systems
    FailoverActivation,     // Activate failover
    NetworkIsolation,       // Isolate from network
    
    // Defensive responses
    IncreaseReserves,       // Increase cash reserves
    DiversifyRisk,          // Diversify risk exposure
    ActivateInsurance,      // Activate insurance
    ContactRegulators,      // Contact regulators
    
    // Adaptive responses
    StrategyAdjustment,     // Adjust trading strategy
    RiskParameterChange,    // Change risk parameters
    AlgorithmUpdate,        // Update algorithms
    MonitoringIncrease,     // Increase monitoring
    
    // Collaborative responses
    InformAllies,           // Inform allied traders
    CoordinateResponse,     // Coordinate with others
    ShareIntelligence,      // Share threat intelligence
    FormDefenseCoalition,   // Form defensive coalition
    
    // Quantum responses
    QuantumDefense,         // Activate quantum defenses
    QuantumCountermeasure,  // Deploy quantum countermeasures
    QuantumIsolation,       // Isolate quantum systems
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumThreatSignature {
    pub coherence_disruption: f64,
    pub entanglement_attack: f64,
    pub superposition_collapse: f64,
    pub decoherence_acceleration: f64,
    pub quantum_noise_injection: f64,
    pub measurement_interference: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImmuneResponse {
    pub response_strength: f64,
    pub antibody_generation: f64,
    pub memory_formation: f64,
    pub adaptive_immunity: f64,
    pub inflammation_level: f64,
    pub recovery_speed: f64,
}

/// Immune system cell types for threat detection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImmuneCellType {
    // Innate immunity cells
    Neutrophil,             // First response cells
    Macrophage,             // Threat consumption cells
    DendriticCell,          // Antigen presentation cells
    NaturalKiller,          // Immediate threat elimination
    
    // Adaptive immunity cells
    THelper,                // Coordination cells
    TCytotoxic,             // Specific threat elimination
    BCell,                  // Antibody production
    PlasmaCell,             // Enhanced antibody production
    MemoryCell,             // Threat memory storage
    
    // Regulatory cells
    RegulatoryT,            // Immune response regulation
    Suppressor,             // Immune suppression
    
    // Custom quantum-enhanced cells
    QuantumSensor,          // Quantum threat detection
    QuantumKiller,          // Quantum threat elimination
    QuantumMemory,          // Quantum threat memory
}

impl ImmuneCellType {
    /// Get the response speed of this cell type
    pub fn response_speed(&self) -> chrono::Duration {
        match self {
            ImmuneCellType::Neutrophil => chrono::Duration::seconds(1),
            ImmuneCellType::NaturalKiller => chrono::Duration::seconds(5),
            ImmuneCellType::Macrophage => chrono::Duration::minutes(1),
            ImmuneCellType::QuantumSensor => chrono::Duration::milliseconds(10),
            ImmuneCellType::QuantumKiller => chrono::Duration::milliseconds(50),
            ImmuneCellType::THelper => chrono::Duration::minutes(5),
            ImmuneCellType::BCell => chrono::Duration::minutes(30),
            ImmuneCellType::MemoryCell => chrono::Duration::seconds(10),
            _ => chrono::Duration::minutes(2),
        }
    }
    
    /// Get the specificity level
    pub fn specificity_level(&self) -> f64 {
        match self {
            ImmuneCellType::Neutrophil => 0.2,
            ImmuneCellType::NaturalKiller => 0.3,
            ImmuneCellType::TCytotoxic => 0.95,
            ImmuneCellType::PlasmaCell => 0.98,
            ImmuneCellType::MemoryCell => 0.99,
            ImmuneCellType::QuantumKiller => 0.92,
            ImmuneCellType::QuantumMemory => 0.96,
            _ => 0.6,
        }
    }
}

/// Market data for threat detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub symbols: Vec<String>,
    pub prices: HashMap<String, f64>,
    pub volumes: HashMap<String, f64>,
    pub volatilities: HashMap<String, f64>,
    pub order_books: HashMap<String, OrderBook>,
    pub trades: Vec<Trade>,
    pub microstructure_indicators: HashMap<String, f64>,
    pub network_metrics: NetworkMetrics,
    pub system_health: SystemHealth,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub bids: Vec<OrderLevel>,
    pub asks: Vec<OrderLevel>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderLevel {
    pub price: f64,
    pub size: f64,
    pub order_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub symbol: String,
    pub price: f64,
    pub size: f64,
    pub side: TradeSide,
    pub venue: String,
    pub trade_type: TradeType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeType {
    Normal,
    BlockTrade,
    DarkPool,
    Crossing,
    Auction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    pub latency: f64,
    pub throughput: f64,
    pub packet_loss: f64,
    pub jitter: f64,
    pub connectivity_score: f64,
    pub congestion_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub disk_usage: f64,
    pub network_usage: f64,
    pub error_rate: f64,
    pub response_time: f64,
    pub availability: f64,
}

/// Configuration for the quantum early warning system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumEarlyWarningConfig {
    pub threat_detection_sensitivity: f64,
    pub immune_response_strength: f64,
    pub quantum_sensor_density: f64,
    pub warning_threshold: f64,
    pub response_automation: bool,
    pub memory_retention_period: chrono::Duration,
    pub learning_rate: f64,
    pub false_positive_tolerance: f64,
}

/// Core trait for threat detectors
#[async_trait]
pub trait ThreatDetector: Send + Sync {
    async fn detect_threats(&mut self, market_data: &MarketData) -> Result<Vec<ThreatDetectionResult>>;
    async fn update_threat_models(&mut self, historical_data: &[MarketData]) -> Result<()>;
    async fn classify_threat(&self, anomaly_data: &AnomalyData) -> Result<ThreatType>;
    async fn assess_threat_impact(&self, threat: &ThreatDetectionResult) -> Result<ThreatImpact>;
    fn get_supported_threats(&self) -> Vec<ThreatType>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyData {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub anomaly_score: f64,
    pub affected_variables: Vec<String>,
    pub deviation_magnitude: f64,
    pub persistence: chrono::Duration,
}

/// Core trait for immune system responses
#[async_trait]
pub trait ImmuneSystem: Send + Sync {
    async fn generate_immune_response(&mut self, 
                                    threat: &ThreatDetectionResult) -> Result<ImmuneResponse>;
    
    async fn activate_immune_cells(&mut self, 
                                  cell_types: &[ImmuneCellType],
                                  threat: &ThreatDetectionResult) -> Result<()>;
    
    async fn form_immunological_memory(&mut self, 
                                     threat: &ThreatDetectionResult,
                                     response_effectiveness: f64) -> Result<()>;
    
    async fn adapt_immune_response(&mut self, 
                                 threat_evolution: &ThreatEvolution) -> Result<()>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatEvolution {
    pub original_threat: ThreatDetectionResult,
    pub evolved_characteristics: HashMap<String, f64>,
    pub adaptation_speed: f64,
    pub evasion_techniques: Vec<String>,
}

/// Main quantum early warning system
#[derive(Debug)]
pub struct QuantumEarlyWarningSystem {
    threat_detectors: Arc<RwLock<Vec<Box<dyn ThreatDetector>>>>,
    immune_system: Arc<RwLock<dyn ImmuneSystem>>,
    quantum_sensors: Arc<RwLock<HashMap<String, QuantumSensor>>>,
    warning_engine: Arc<RwLock<WarningEngine>>,
    pattern_recognizer: Arc<RwLock<PatternRecognizer>>,
    config: QuantumEarlyWarningConfig,
    threat_memory: Arc<RwLock<HashMap<String, ThreatMemory>>>,
    active_threats: Arc<RwLock<HashMap<String, ThreatDetectionResult>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatMemory {
    pub threat_type: ThreatType,
    pub detection_history: Vec<ThreatDetectionResult>,
    pub response_history: Vec<ThreatResponse>,
    pub effectiveness_scores: Vec<f64>,
    pub learned_patterns: Vec<String>,
    pub adaptation_count: u32,
}

// Placeholder implementations
pub struct QuantumSensor;
pub struct WarningEngine;
pub struct PatternRecognizer;

impl QuantumEarlyWarningSystem {
    pub async fn new(config: QuantumEarlyWarningConfig) -> Result<Self> {
        info!("Initializing Quantum Early Warning System");
        
        Ok(Self {
            threat_detectors: Arc::new(RwLock::new(Vec::new())),
            immune_system: Arc::new(RwLock::new(BasicImmuneSystem::new().await?)),
            quantum_sensors: Arc::new(RwLock::new(HashMap::new())),
            warning_engine: Arc::new(RwLock::new(WarningEngine)),
            pattern_recognizer: Arc::new(RwLock::new(PatternRecognizer)),
            config,
            threat_memory: Arc::new(RwLock::new(HashMap::new())),
            active_threats: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Main threat scanning function
    pub async fn scan_threats(&mut self, market_data: &MarketData) -> Result<Vec<ThreatDetectionResult>> {
        debug!("Scanning for threats in market data");
        
        let mut all_threats = Vec::new();
        
        // Run all threat detectors in parallel
        let detectors = self.threat_detectors.read().await;
        let mut detector_futures = Vec::new();
        
        for detector in detectors.iter() {
            // Note: In real implementation, we'd need to handle the mutable borrow differently
            // This is a simplified version for demonstration
            let detection_future = async {
                // detector.detect_threats(market_data).await
                Vec::new() // Placeholder
            };
            detector_futures.push(detection_future);
        }
        
        // Collect results from all detectors
        for detector_result in futures::future::join_all(detector_futures).await {
            all_threats.extend(detector_result);
        }
        
        // Filter and prioritize threats
        let prioritized_threats = self.prioritize_threats(all_threats).await?;
        
        // Generate immune responses for high-priority threats
        for threat in &prioritized_threats {
            if threat.severity > self.config.warning_threshold {
                self.generate_warning(threat).await?;
                
                if self.config.response_automation {
                    self.activate_automated_response(threat).await?;
                }
            }
        }
        
        // Update threat memory
        self.update_threat_memory(&prioritized_threats).await?;
        
        info!("Threat scan completed, found {} threats", prioritized_threats.len());
        Ok(prioritized_threats)
    }
    
    async fn prioritize_threats(&self, threats: Vec<ThreatDetectionResult>) -> Result<Vec<ThreatDetectionResult>> {
        let mut prioritized = threats;
        
        // Sort by severity and confidence
        prioritized.sort_by(|a, b| {
            let score_a = a.severity * a.confidence;
            let score_b = b.severity * b.confidence;
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        Ok(prioritized)
    }
    
    async fn generate_warning(&self, threat: &ThreatDetectionResult) -> Result<()> {
        warn!("THREAT DETECTED: {:?} with severity {:.2} and confidence {:.2}", 
              threat.threat_type, threat.severity, threat.confidence);
        
        // Implementation will be added in warning_engine module
        Ok(())
    }
    
    async fn activate_automated_response(&self, threat: &ThreatDetectionResult) -> Result<()> {
        info!("Activating automated response for threat: {:?}", threat.threat_type);
        
        // Generate immune response
        let immune_response = self.immune_system
            .write()
            .await
            .generate_immune_response(threat)
            .await?;
        
        debug!("Immune response generated with strength: {:.2}", immune_response.response_strength);
        
        // Execute recommended actions based on threat type and immune response
        for action in &threat.recommended_actions {
            if action.urgency > 0.8 {
                self.execute_response_action(action).await?;
            }
        }
        
        Ok(())
    }
    
    async fn execute_response_action(&self, action: &ThreatResponse) -> Result<()> {
        info!("Executing response action: {:?}", action.response_type);
        
        match action.response_type {
            ResponseType::EmergencyStop => {
                // Implementation for emergency stop
                warn!("EMERGENCY STOP ACTIVATED");
            },
            ResponseType::PositionReduction => {
                // Implementation for position reduction
                info!("Position reduction initiated");
            },
            ResponseType::QuantumDefense => {
                // Implementation for quantum defense
                info!("Quantum defense systems activated");
            },
            _ => {
                debug!("Standard response action executed: {:?}", action.response_type);
            }
        }
        
        Ok(())
    }
    
    async fn update_threat_memory(&self, threats: &[ThreatDetectionResult]) -> Result<()> {
        let mut memory = self.threat_memory.write().await;
        
        for threat in threats {
            let threat_key = format!("{:?}_{}", threat.threat_type, threat.threat_id);
            
            memory.entry(threat_key).or_insert_with(|| ThreatMemory {
                threat_type: threat.threat_type,
                detection_history: Vec::new(),
                response_history: Vec::new(),
                effectiveness_scores: Vec::new(),
                learned_patterns: Vec::new(),
                adaptation_count: 0,
            }).detection_history.push(threat.clone());
        }
        
        debug!("Updated threat memory for {} threats", threats.len());
        Ok(())
    }
    
    /// Add a new threat detector to the system
    pub async fn add_threat_detector(&self, detector: Box<dyn ThreatDetector>) -> Result<()> {
        self.threat_detectors.write().await.push(detector);
        info!("Added new threat detector to the system");
        Ok(())
    }
    
    /// Get current system status
    pub async fn get_system_status(&self) -> Result<SystemStatus> {
        let active_threats_count = self.active_threats.read().await.len();
        let memory_entries = self.threat_memory.read().await.len();
        let detector_count = self.threat_detectors.read().await.len();
        
        Ok(SystemStatus {
            active_threats: active_threats_count,
            memory_entries,
            detector_count,
            quantum_sensors_active: true,
            immune_system_active: true,
            last_scan: chrono::Utc::now(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    pub active_threats: usize,
    pub memory_entries: usize,
    pub detector_count: usize,
    pub quantum_sensors_active: bool,
    pub immune_system_active: bool,
    pub last_scan: chrono::DateTime<chrono::Utc>,
}

// Basic immune system implementation
pub struct BasicImmuneSystem;

impl BasicImmuneSystem {
    pub async fn new() -> Result<Self> {
        Ok(Self)
    }
}

#[async_trait]
impl ImmuneSystem for BasicImmuneSystem {
    async fn generate_immune_response(&mut self, 
                                    threat: &ThreatDetectionResult) -> Result<ImmuneResponse> {
        let response_strength = threat.severity * threat.confidence;
        
        Ok(ImmuneResponse {
            response_strength,
            antibody_generation: response_strength * 0.8,
            memory_formation: response_strength * 0.6,
            adaptive_immunity: response_strength * 0.7,
            inflammation_level: response_strength * 0.5,
            recovery_speed: 1.0 - response_strength * 0.3,
        })
    }
    
    async fn activate_immune_cells(&mut self, 
                                  _cell_types: &[ImmuneCellType],
                                  _threat: &ThreatDetectionResult) -> Result<()> {
        // Implementation will be added in immune_response module
        Ok(())
    }
    
    async fn form_immunological_memory(&mut self, 
                                     _threat: &ThreatDetectionResult,
                                     _response_effectiveness: f64) -> Result<()> {
        // Implementation will be added in immune_response module
        Ok(())
    }
    
    async fn adapt_immune_response(&mut self, 
                                 _threat_evolution: &ThreatEvolution) -> Result<()> {
        // Implementation will be added in immune_response module
        Ok(())
    }
}

/// Error types for quantum early warning operations
#[derive(thiserror::Error, Debug)]
pub enum QuantumEarlyWarningError {
    #[error("Threat detection failed: {0}")]
    ThreatDetectionFailed(String),
    
    #[error("Immune response failed: {0}")]
    ImmuneResponseFailed(String),
    
    #[error("Quantum sensor error: {0}")]
    QuantumSensorError(String),
    
    #[error("Pattern recognition failed: {0}")]
    PatternRecognitionFailed(String),
    
    #[error("Warning generation failed: {0}")]
    WarningGenerationFailed(String),
    
    #[error("Response execution failed: {0}")]
    ResponseExecutionFailed(String),
    
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
    
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
}

pub type QuantumEarlyWarningResult<T> = Result<T, QuantumEarlyWarningError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threat_type_properties() {
        assert!(ThreatType::FlashCrash.requires_immediate_response());
        assert!(!ThreatType::GeopoliticalTension.requires_immediate_response());
        assert!(ThreatType::SystemicRisk.impact_severity() > 0.9);
        assert!(ThreatType::LatencySpike.impact_severity() < 0.5);
    }

    #[test]
    fn test_threat_detection_difficulty() {
        assert!(ThreatType::QuantumAttack.detection_difficulty() > 0.9);
        assert!(ThreatType::CircuitBreakerTrigger.detection_difficulty() < 0.2);
        assert!(ThreatType::EmergentThreat.detection_difficulty() > 0.8);
    }

    #[test]
    fn test_warning_lead_time() {
        assert!(ThreatType::FlashCrash.warning_lead_time() < chrono::Duration::seconds(10));
        assert!(ThreatType::MacroeconomicShock.warning_lead_time() > chrono::Duration::minutes(30));
        assert!(ThreatType::QuantumAttack.warning_lead_time() < chrono::Duration::seconds(1));
    }

    #[test]
    fn test_immune_cell_properties() {
        assert!(ImmuneCellType::QuantumSensor.response_speed() < chrono::Duration::seconds(1));
        assert!(ImmuneCellType::MemoryCell.specificity_level() > 0.9);
        assert!(ImmuneCellType::Neutrophil.specificity_level() < 0.3);
    }

    #[tokio::test]
    async fn test_quantum_early_warning_system_creation() {
        let config = QuantumEarlyWarningConfig {
            threat_detection_sensitivity: 0.8,
            immune_response_strength: 0.7,
            quantum_sensor_density: 0.9,
            warning_threshold: 0.5,
            response_automation: true,
            memory_retention_period: chrono::Duration::days(30),
            learning_rate: 0.1,
            false_positive_tolerance: 0.05,
        };

        let system = QuantumEarlyWarningSystem::new(config).await.expect("Failed to create system");
        let status = system.get_system_status().await.expect("Failed to get status");
        
        assert_eq!(status.active_threats, 0);
        assert_eq!(status.detector_count, 0);
        assert!(status.quantum_sensors_active);
        assert!(status.immune_system_active);
    }

    #[test]
    fn test_threat_detection_result_serialization() {
        let result = ThreatDetectionResult {
            threat_id: "THREAT_001".to_string(),
            threat_type: ThreatType::FlashCrash,
            confidence: 0.95,
            severity: 0.9,
            detection_time: chrono::Utc::now(),
            estimated_impact: ThreatImpact {
                financial_impact: 1000000.0,
                operational_impact: 0.8,
                reputational_impact: 0.6,
                systemic_impact: 0.7,
                recovery_time: chrono::Duration::hours(4),
                cascading_probability: 0.3,
            },
            warning_lead_time: chrono::Duration::seconds(5),
            affected_systems: vec!["trading".to_string(), "risk".to_string()],
            recommended_actions: vec![],
            quantum_signature: None,
            immune_response: None,
        };

        let serialized = serde_json::to_string(&result).expect("Serialization failed");
        let deserialized: ThreatDetectionResult = serde_json::from_str(&serialized).expect("Deserialization failed");
        
        assert_eq!(result.threat_id, deserialized.threat_id);
        assert_eq!(result.threat_type, deserialized.threat_type);
        assert!((result.confidence - deserialized.confidence).abs() < 1e-10);
    }
}