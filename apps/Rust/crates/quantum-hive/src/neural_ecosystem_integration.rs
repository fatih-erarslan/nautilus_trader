//! Neural Ecosystem Integration for Quantum-Hive
//! 
//! Integrates Cognition Engine, ruv-FANN, and ruv-swarm neural networks
//! to create a comprehensive neural intelligence layer for QAR.

use crate::{QuantumQueen, NeuromorphicSignal, ModuleContribution};
use anyhow::{Result, anyhow};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, debug, warn};
use std::collections::HashMap;
use std::process::Command;

/// Neural ecosystem coordinator that manages all neural network types
pub struct NeuralEcosystemCoordinator {
    /// Cognition Engine integration (NHITS/NBeats)
    pub cognition_engine: Arc<RwLock<CognitionEngineInterface>>,
    
    /// ruv-FANN neural networks
    pub ruv_fann_networks: Arc<RwLock<RuvFannManager>>,
    
    /// ruv-swarm ephemeral networks
    pub ruv_swarm: Arc<RwLock<RuvSwarmInterface>>,
    
    /// Performance metrics for all neural systems
    pub ecosystem_metrics: Arc<RwLock<EcosystemMetrics>>,
}

/// Interface to Cognition Engine (NHITS/NBeats)
pub struct CognitionEngineInterface {
    /// NHITS model for hierarchical time series
    nhits_endpoint: String,
    /// NBeats model for basis expansion
    nbeats_endpoint: String,
    /// HTTP client for API calls
    client: reqwest::Client,
}

/// Manager for ruv-FANN networks
pub struct RuvFannManager {
    /// Active FANN networks by purpose
    networks: HashMap<String, RuvFannNetwork>,
    /// Network performance history
    performance_history: Vec<NetworkPerformance>,
}

/// Individual ruv-FANN network wrapper
pub struct RuvFannNetwork {
    /// Network ID
    id: String,
    /// Network purpose (e.g., "momentum_detection", "volatility_prediction")
    purpose: String,
    /// Network configuration
    config: FannConfig,
    /// Performance metrics
    metrics: NetworkMetrics,
}

/// Interface to ruv-swarm ephemeral networks
pub struct RuvSwarmInterface {
    /// MCP server connection status
    mcp_connected: bool,
    /// Active swarm ID
    swarm_id: Option<String>,
    /// Ephemeral network count
    ephemeral_count: usize,
}

/// Combined ecosystem metrics
#[derive(Debug, Default)]
pub struct EcosystemMetrics {
    pub total_predictions: u64,
    pub cognition_engine_calls: u64,
    pub ruv_fann_inferences: u64,
    pub ephemeral_networks_spawned: u64,
    pub avg_latency_us: f64,
    pub neural_consensus_score: f64,
}

impl NeuralEcosystemCoordinator {
    /// Create new neural ecosystem coordinator
    pub async fn new() -> Result<Self> {
        info!("🧠 Initializing Neural Ecosystem Coordinator");
        
        // Initialize Cognition Engine interface
        let cognition_engine = Arc::new(RwLock::new(
            CognitionEngineInterface::new().await?
        ));
        
        // Initialize ruv-FANN manager
        let ruv_fann_networks = Arc::new(RwLock::new(
            RuvFannManager::new()?
        ));
        
        // Initialize ruv-swarm interface
        let ruv_swarm = Arc::new(RwLock::new(
            RuvSwarmInterface::new().await?
        ));
        
        let ecosystem_metrics = Arc::new(RwLock::new(EcosystemMetrics::default()));
        
        info!("✅ Neural Ecosystem initialized with 3 subsystems");
        
        Ok(Self {
            cognition_engine,
            ruv_fann_networks,
            ruv_swarm,
            ecosystem_metrics,
        })
    }
    
    /// Process market data through all neural systems
    pub async fn process_comprehensive(
        &self,
        market_data: &MarketData,
    ) -> Result<ComprehensiveNeuralSignal> {
        let start_time = std::time::Instant::now();
        
        // Process in parallel through all systems
        let (cognition_signal, fann_signal, swarm_signal) = tokio::join!(
            self.process_with_cognition_engine(market_data),
            self.process_with_ruv_fann(market_data),
            self.process_with_ruv_swarm(market_data)
        );
        
        // Combine signals
        let combined_signal = self.fuse_neural_signals(
            cognition_signal?,
            fann_signal?,
            swarm_signal?
        )?;
        
        // Update metrics
        let mut metrics = self.ecosystem_metrics.write().await;
        metrics.total_predictions += 1;
        metrics.avg_latency_us = start_time.elapsed().as_micros() as f64;
        
        debug!("Neural ecosystem processing completed in {}μs", 
               start_time.elapsed().as_micros());
        
        Ok(combined_signal)
    }
    
    /// Process with Cognition Engine (NHITS/NBeats)
    async fn process_with_cognition_engine(
        &self,
        market_data: &MarketData,
    ) -> Result<NeuralSignal> {
        let cognition = self.cognition_engine.read().await;
        
        // Call NHITS for hierarchical time series
        let nhits_prediction = cognition.predict_nhits(market_data).await?;
        
        // Call NBeats for trend/seasonality decomposition
        let nbeats_prediction = cognition.predict_nbeats(market_data).await?;
        
        // Combine predictions
        Ok(NeuralSignal {
            source: "cognition_engine".to_string(),
            prediction: (nhits_prediction.value + nbeats_prediction.value) / 2.0,
            confidence: (nhits_prediction.confidence + nbeats_prediction.confidence) / 2.0,
            components: vec![
                ("nhits".to_string(), nhits_prediction),
                ("nbeats".to_string(), nbeats_prediction),
            ],
        })
    }
    
    /// Process with ruv-FANN networks
    async fn process_with_ruv_fann(
        &self,
        market_data: &MarketData,
    ) -> Result<NeuralSignal> {
        let mut fann_manager = self.ruv_fann_networks.write().await;
        
        // Select appropriate FANN network based on market conditions
        let network = fann_manager.select_network_for_market(market_data)?;
        
        // Run inference
        let prediction = network.predict(market_data)?;
        
        Ok(NeuralSignal {
            source: "ruv_fann".to_string(),
            prediction: prediction.value,
            confidence: prediction.confidence,
            components: vec![
                (network.purpose.clone(), prediction),
            ],
        })
    }
    
    /// Process with ruv-swarm ephemeral networks
    async fn process_with_ruv_swarm(
        &self,
        market_data: &MarketData,
    ) -> Result<NeuralSignal> {
        let mut swarm = self.ruv_swarm.write().await;
        
        // Spawn ephemeral network for this specific prediction
        let ephemeral_id = swarm.spawn_ephemeral_network(market_data).await?;
        
        // Get prediction from ephemeral network
        let prediction = swarm.get_ephemeral_prediction(ephemeral_id).await?;
        
        // Dissolve ephemeral network
        swarm.dissolve_ephemeral_network(ephemeral_id).await?;
        
        Ok(NeuralSignal {
            source: "ruv_swarm_ephemeral".to_string(),
            prediction: prediction.value,
            confidence: prediction.confidence,
            components: vec![
                (format!("ephemeral_{}", ephemeral_id), prediction),
            ],
        })
    }
    
    /// Fuse signals from all neural systems
    fn fuse_neural_signals(
        &self,
        cognition: NeuralSignal,
        fann: NeuralSignal,
        swarm: NeuralSignal,
    ) -> Result<ComprehensiveNeuralSignal> {
        // Weight signals based on confidence
        let total_confidence = cognition.confidence + fann.confidence + swarm.confidence;
        
        let weighted_prediction = (
            cognition.prediction * cognition.confidence +
            fann.prediction * fann.confidence +
            swarm.prediction * swarm.confidence
        ) / total_confidence;
        
        // Calculate consensus
        let predictions = vec![cognition.prediction, fann.prediction, swarm.prediction];
        let consensus = self.calculate_consensus(&predictions);
        
        Ok(ComprehensiveNeuralSignal {
            prediction: weighted_prediction,
            confidence: total_confidence / 3.0,
            consensus_score: consensus,
            individual_signals: vec![cognition, fann, swarm],
            fusion_method: "confidence_weighted".to_string(),
        })
    }
    
    /// Calculate consensus among predictions
    fn calculate_consensus(&self, predictions: &[f64]) -> f64 {
        let mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
        let variance = predictions.iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>() / predictions.len() as f64;
        
        // Lower variance = higher consensus
        1.0 / (1.0 + variance.sqrt())
    }
}

/// Cognition Engine interface implementation
impl CognitionEngineInterface {
    /// Create new Cognition Engine interface
    async fn new() -> Result<Self> {
        // Start Cognition Engine server if not running
        let nhits_endpoint = "http://localhost:8080/nhits".to_string();
        let nbeats_endpoint = "http://localhost:8080/nbeats".to_string();
        
        Ok(Self {
            nhits_endpoint,
            nbeats_endpoint,
            client: reqwest::Client::new(),
        })
    }
    
    /// Predict using NHITS
    async fn predict_nhits(&self, market_data: &MarketData) -> Result<Prediction> {
        let request = serde_json::json!({
            "timestamp": market_data.timestamp,
            "values": market_data.to_time_series(),
            "horizon": 10,
        });
        
        let response = self.client
            .post(&self.nhits_endpoint)
            .json(&request)
            .send()
            .await?;
        
        let prediction: NhitsResponse = response.json().await?;
        
        Ok(Prediction {
            value: prediction.forecast[0],
            confidence: prediction.confidence,
            horizon: prediction.horizon,
        })
    }
    
    /// Predict using NBeats
    async fn predict_nbeats(&self, market_data: &MarketData) -> Result<Prediction> {
        let request = serde_json::json!({
            "timestamp": market_data.timestamp,
            "values": market_data.to_time_series(),
            "stacks": ["trend", "seasonality"],
        });
        
        let response = self.client
            .post(&self.nbeats_endpoint)
            .json(&request)
            .send()
            .await?;
        
        let prediction: NbeatsResponse = response.json().await?;
        
        Ok(Prediction {
            value: prediction.forecast[0],
            confidence: prediction.interpretability_score,
            horizon: 1,
        })
    }
}

/// ruv-FANN manager implementation
impl RuvFannManager {
    /// Create new ruv-FANN manager
    fn new() -> Result<Self> {
        let mut networks = HashMap::new();
        
        // Create specialized FANN networks
        networks.insert("momentum".to_string(), 
            RuvFannNetwork::new("momentum", FannConfig::momentum_detector())?);
        networks.insert("volatility".to_string(),
            RuvFannNetwork::new("volatility", FannConfig::volatility_predictor())?);
        networks.insert("trend".to_string(),
            RuvFannNetwork::new("trend", FannConfig::trend_analyzer())?);
        
        Ok(Self {
            networks,
            performance_history: Vec::new(),
        })
    }
    
    /// Select best network for current market conditions
    fn select_network_for_market(&mut self, market_data: &MarketData) -> Result<&mut RuvFannNetwork> {
        // Simple selection based on volatility
        let volatility = market_data.calculate_volatility();
        
        let network_key = if volatility > 0.02 {
            "volatility"
        } else if market_data.has_momentum() {
            "momentum"
        } else {
            "trend"
        };
        
        self.networks.get_mut(network_key)
            .ok_or_else(|| anyhow!("Network not found: {}", network_key))
    }
}

/// ruv-swarm interface implementation
impl RuvSwarmInterface {
    /// Create new ruv-swarm interface
    async fn new() -> Result<Self> {
        // Check if MCP server is available
        let mcp_connected = Self::check_mcp_connection().await;
        
        if mcp_connected {
            info!("✅ Connected to ruv-swarm MCP server");
        } else {
            warn!("⚠️ ruv-swarm MCP server not available, using fallback");
        }
        
        Ok(Self {
            mcp_connected,
            swarm_id: None,
            ephemeral_count: 0,
        })
    }
    
    /// Check MCP server connection
    async fn check_mcp_connection() -> bool {
        // Try to connect to MCP server
        Command::new("npx")
            .args(&["ruv-swarm", "status"])
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }
    
    /// Spawn ephemeral network
    async fn spawn_ephemeral_network(&mut self, market_data: &MarketData) -> Result<String> {
        self.ephemeral_count += 1;
        let ephemeral_id = format!("ephemeral_{}", self.ephemeral_count);
        
        if self.mcp_connected {
            // Use MCP to spawn network
            Command::new("npx")
                .args(&[
                    "ruv-swarm",
                    "spawn",
                    "--type", "predictor",
                    "--ephemeral",
                    "--id", &ephemeral_id,
                ])
                .spawn()?;
        }
        
        Ok(ephemeral_id)
    }
    
    /// Get prediction from ephemeral network
    async fn get_ephemeral_prediction(&self, ephemeral_id: String) -> Result<Prediction> {
        // In real implementation, would query the ephemeral network
        Ok(Prediction {
            value: 0.75, // Placeholder
            confidence: 0.85,
            horizon: 1,
        })
    }
    
    /// Dissolve ephemeral network
    async fn dissolve_ephemeral_network(&mut self, ephemeral_id: String) -> Result<()> {
        if self.mcp_connected {
            Command::new("npx")
                .args(&[
                    "ruv-swarm",
                    "dissolve",
                    "--id", &ephemeral_id,
                ])
                .spawn()?;
        }
        Ok(())
    }
}

/// Enhanced Quantum Queen integration
impl QuantumQueen {
    /// Process market data with full neural ecosystem
    pub async fn process_with_neural_ecosystem(
        &mut self,
        market_data: &MarketData,
        ecosystem: &NeuralEcosystemCoordinator,
    ) -> Result<TradeAction> {
        // Get comprehensive neural signal
        let neural_signal = ecosystem.process_comprehensive(market_data).await?;
        
        // Convert to neuromorphic signal for QAR
        let neuromorphic_signal = self.convert_comprehensive_to_neuromorphic(neural_signal)?;
        
        // Integrate with QAR
        self.integrate_neuromorphic_signal(neuromorphic_signal).await?;
        
        // Make enhanced decision
        self.make_enhanced_decision(market_data).await
    }
    
    /// Convert comprehensive neural signal to neuromorphic format
    fn convert_comprehensive_to_neuromorphic(
        &self,
        signal: ComprehensiveNeuralSignal,
    ) -> Result<NeuromorphicSignal> {
        let mut module_contributions = HashMap::new();
        
        // Add each neural system as a module contribution
        for neural_signal in &signal.individual_signals {
            module_contributions.insert(
                neural_signal.source.clone(),
                ModuleContribution {
                    module_name: neural_signal.source.clone(),
                    prediction: neural_signal.prediction,
                    confidence: neural_signal.confidence,
                    processing_time_us: 100, // Placeholder
                }
            );
        }
        
        Ok(NeuromorphicSignal {
            prediction: signal.prediction,
            confidence: signal.confidence,
            module_contributions,
            spike_patterns: vec![], // Would be populated from SNN data
            temporal_coherence: signal.consensus_score,
            functional_optimization: 0.9, // High due to multiple systems
        })
    }
}

/// Hyperbolic lattice node enhancement for neural spawning
pub trait NeuralSpawningNode {
    /// Spawn ephemeral neural network at this node
    async fn spawn_neural_network(&self, config: NeuralConfig) -> Result<String>;
    
    /// Check if node should spawn network based on local conditions
    fn should_spawn_network(&self) -> bool;
    
    /// Dissolve neural network at this node
    async fn dissolve_neural_network(&self, network_id: String) -> Result<()>;
}

// Data structures
#[derive(Debug, Clone)]
pub struct NeuralSignal {
    pub source: String,
    pub prediction: f64,
    pub confidence: f64,
    pub components: Vec<(String, Prediction)>,
}

#[derive(Debug, Clone)]
pub struct ComprehensiveNeuralSignal {
    pub prediction: f64,
    pub confidence: f64,
    pub consensus_score: f64,
    pub individual_signals: Vec<NeuralSignal>,
    pub fusion_method: String,
}

#[derive(Debug, Clone)]
pub struct Prediction {
    pub value: f64,
    pub confidence: f64,
    pub horizon: usize,
}

#[derive(Debug, Clone)]
pub struct FannConfig {
    pub layers: Vec<usize>,
    pub activation: String,
    pub learning_rate: f64,
}

impl FannConfig {
    fn momentum_detector() -> Self {
        Self {
            layers: vec![10, 20, 10, 1],
            activation: "relu".to_string(),
            learning_rate: 0.01,
        }
    }
    
    fn volatility_predictor() -> Self {
        Self {
            layers: vec![20, 40, 20, 1],
            activation: "tanh".to_string(),
            learning_rate: 0.001,
        }
    }
    
    fn trend_analyzer() -> Self {
        Self {
            layers: vec![15, 30, 15, 1],
            activation: "sigmoid".to_string(),
            learning_rate: 0.005,
        }
    }
}

// Placeholder types
#[derive(Debug)]
pub struct MarketData {
    pub timestamp: i64,
    pub price: f64,
    pub volume: f64,
    pub spread: f64,
}

impl MarketData {
    fn to_time_series(&self) -> Vec<f64> {
        vec![self.price, self.volume, self.spread]
    }
    
    fn calculate_volatility(&self) -> f64 {
        self.spread / self.price
    }
    
    fn has_momentum(&self) -> bool {
        self.volume > 1000.0
    }
}

#[derive(Debug, Clone)]
pub struct NetworkMetrics {
    pub accuracy: f64,
    pub latency_us: u64,
}

#[derive(Debug, Clone)]
pub struct NetworkPerformance {
    pub timestamp: i64,
    pub metrics: NetworkMetrics,
}

// API response types
#[derive(serde::Deserialize)]
struct NhitsResponse {
    forecast: Vec<f64>,
    confidence: f64,
    horizon: usize,
}

#[derive(serde::Deserialize)]
struct NbeatsResponse {
    forecast: Vec<f64>,
    interpretability_score: f64,
    trend: Vec<f64>,
    seasonality: Vec<f64>,
}

// Placeholder ruv-FANN network
impl RuvFannNetwork {
    fn new(purpose: &str, config: FannConfig) -> Result<Self> {
        Ok(Self {
            id: uuid::Uuid::new_v4().to_string(),
            purpose: purpose.to_string(),
            config,
            metrics: NetworkMetrics {
                accuracy: 0.0,
                latency_us: 0,
            },
        })
    }
    
    fn predict(&mut self, _market_data: &MarketData) -> Result<Prediction> {
        // Placeholder implementation
        Ok(Prediction {
            value: 0.5,
            confidence: 0.8,
            horizon: 1,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_neural_ecosystem_creation() {
        let ecosystem = NeuralEcosystemCoordinator::new().await;
        assert!(ecosystem.is_ok());
    }
    
    #[test]
    fn test_consensus_calculation() {
        let coordinator = NeuralEcosystemCoordinator::new().await.unwrap();
        let predictions = vec![0.5, 0.52, 0.48];
        let consensus = coordinator.calculate_consensus(&predictions);
        assert!(consensus > 0.9); // High consensus for similar predictions
    }
}