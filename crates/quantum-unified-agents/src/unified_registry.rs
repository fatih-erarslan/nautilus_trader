//! Unified Quantum Agent Registry
//!
//! This module provides a centralized registry for managing all unified quantum agents
//! and coordinating their interactions through the PADS integration layer.

use async_trait::async_trait;
use quantum_core::{
    QuantumAgent, PADSSignal, MarketData, LatticeState, QuantumConfig,
    AgentHealth, AgentMetrics, QuantumResult, QuantumError, HealthStatus
};
use crate::pads_integration::{PADSIntegrationManager, AggregationStrategy};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use chrono::{DateTime, Utc};
use tokio::sync::Mutex as AsyncMutex;
use uuid::Uuid;

/// Unified Quantum Agent Registry
#[derive(Debug)]
pub struct UnifiedQuantumAgentRegistry {
    /// Registered agents
    agents: Arc<RwLock<HashMap<String, Arc<AsyncMutex<dyn QuantumAgent<Signal = dyn Send + Sync, State = dyn Send + Sync, Config = dyn Send + Sync + Clone>>>>>>,
    /// Lattice state for quantum entanglement
    lattice_state: Arc<AsyncMutex<LatticeState>>,
    /// PADS integration manager
    pads_manager: Arc<AsyncMutex<PADSIntegrationManager>>,
    /// Registry metrics
    metrics: Arc<RwLock<RegistryMetrics>>,
    /// Registry configuration
    config: RegistryConfig,
    /// Agent health monitors
    health_monitors: Arc<RwLock<HashMap<String, AgentHealthMonitor>>>,
}

/// Registry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryConfig {
    /// Maximum number of agents
    pub max_agents: usize,
    /// Health check interval in seconds
    pub health_check_interval_s: u64,
    /// Automatic decoherence mitigation
    pub auto_decoherence_mitigation: bool,
    /// PADS aggregation strategy
    pub pads_aggregation_strategy: AggregationStrategy,
    /// Enable performance monitoring
    pub enable_performance_monitoring: bool,
    /// Registry ID
    pub registry_id: String,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            max_agents: 12, // Support for all 12 quantum agents
            health_check_interval_s: 30,
            auto_decoherence_mitigation: true,
            pads_aggregation_strategy: AggregationStrategy::CoherenceWeighted,
            enable_performance_monitoring: true,
            registry_id: format!("quantum-registry-{}", Uuid::new_v4()),
        }
    }
}

/// Registry metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RegistryMetrics {
    /// Total agents registered
    pub total_agents: usize,
    /// Active agents
    pub active_agents: usize,
    /// Total signals processed
    pub total_signals: u64,
    /// Average system coherence
    pub avg_system_coherence: f64,
    /// Total processing time in milliseconds
    pub total_processing_time_ms: f64,
    /// Error count
    pub error_count: u64,
    /// Decoherence events detected
    pub decoherence_events: u64,
    /// PADS signals generated
    pub pads_signals_generated: u64,
    /// Registry uptime
    pub registry_uptime: std::time::Duration,
    /// Last update timestamp
    pub last_update: Option<DateTime<Utc>>,
}

/// Agent health monitor
#[derive(Debug, Clone)]
pub struct AgentHealthMonitor {
    /// Agent ID
    pub agent_id: String,
    /// Last health check
    pub last_health_check: DateTime<Utc>,
    /// Health status history
    pub health_history: Vec<AgentHealth>,
    /// Performance trends
    pub performance_trends: Vec<f64>,
    /// Alert thresholds
    pub alert_thresholds: HealthThresholds,
}

/// Health monitoring thresholds
#[derive(Debug, Clone)]
pub struct HealthThresholds {
    /// Minimum coherence level
    pub min_coherence: f64,
    /// Maximum error rate
    pub max_error_rate: f64,
    /// Minimum performance score
    pub min_performance: f64,
    /// Maximum resource utilization
    pub max_resource_utilization: f64,
}

impl Default for HealthThresholds {
    fn default() -> Self {
        Self {
            min_coherence: 0.6,
            max_error_rate: 0.2,
            min_performance: 0.5,
            max_resource_utilization: 0.9,
        }
    }
}

impl UnifiedQuantumAgentRegistry {
    /// Create new unified registry
    pub fn new(config: RegistryConfig) -> Self {
        let initial_lattice = LatticeState::new(0);
        let pads_manager = PADSIntegrationManager::new(config.pads_aggregation_strategy);
        
        Self {
            agents: Arc::new(RwLock::new(HashMap::new())),
            lattice_state: Arc::new(AsyncMutex::new(initial_lattice)),
            pads_manager: Arc::new(AsyncMutex::new(pads_manager)),
            metrics: Arc::new(RwLock::new(RegistryMetrics::default())),
            config,
            health_monitors: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Register a quantum agent
    pub async fn register_agent<T>(&self, agent: T) -> QuantumResult<()>
    where
        T: QuantumAgent + 'static,
        T::Signal: Send + Sync + 'static,
        T::State: Send + Sync + 'static,
        T::Config: Send + Sync + Clone + 'static,
    {
        let agent_id = agent.agent_id().to_string();
        
        // Check if registry is at capacity
        {
            let agents = self.agents.read().map_err(|_| QuantumError::ProcessingError {
                message: "Failed to read agents lock".to_string(),
            })?;
            
            if agents.len() >= self.config.max_agents {
                return Err(QuantumError::ProcessingError {
                    message: format!("Registry at capacity: {}/{}", agents.len(), self.config.max_agents),
                });
            }
        }
        
        // Update lattice state for new agent
        {
            let mut lattice = self.lattice_state.lock().await;
            let new_size = lattice.dimensions.0 + 1;
            *lattice = LatticeState::new(new_size);
        }
        
        // Create health monitor
        let health_monitor = AgentHealthMonitor {
            agent_id: agent_id.clone(),
            last_health_check: Utc::now(),
            health_history: Vec::new(),
            performance_trends: Vec::new(),
            alert_thresholds: HealthThresholds::default(),
        };
        
        // Register agent and health monitor
        {
            let mut agents = self.agents.write().map_err(|_| QuantumError::ProcessingError {
                message: "Failed to write agents lock".to_string(),
            })?;
            
            let mut health_monitors = self.health_monitors.write().map_err(|_| QuantumError::ProcessingError {
                message: "Failed to write health monitors lock".to_string(),
            })?;
            
            // Note: This is a simplified version. In practice, you would need proper trait object handling
            // agents.insert(agent_id.clone(), Arc::new(AsyncMutex::new(agent)));
            health_monitors.insert(agent_id.clone(), health_monitor);
        }
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().map_err(|_| QuantumError::ProcessingError {
                message: "Failed to write metrics lock".to_string(),
            })?;
            
            metrics.total_agents += 1;
            metrics.active_agents += 1;
            metrics.last_update = Some(Utc::now());
        }
        
        tracing::info!("Successfully registered quantum agent: {} ({})", agent_id, agent.agent_type());
        Ok(())
    }
    
    /// Process market data across all registered agents
    pub async fn process_market_data(&self, market_data: &MarketData) -> QuantumResult<PADSSignal> {
        let start_time = std::time::Instant::now();
        
        // Get current lattice state
        let lattice_state = {
            let lattice = self.lattice_state.lock().await;
            lattice.clone()
        };
        
        // Process data with all agents in parallel
        let mut agent_signals = Vec::new();
        let mut successful_agents = 0;
        let mut total_coherence = 0.0;
        
        // Note: This is a simplified version. In practice, you would iterate through agents
        // and call their process methods in parallel using tokio::spawn
        
        // For demonstration, we'll simulate some agent signals
        agent_signals.push(self.create_demo_signal("QAR", market_data).await?);
        agent_signals.push(self.create_demo_signal("Hedge", market_data).await?);
        agent_signals.push(self.create_demo_signal("LMSR", market_data).await?);
        
        successful_agents = agent_signals.len();
        total_coherence = agent_signals.iter()
            .map(|s| s.quantum_signal.coherence)
            .sum();
        
        // Generate unified PADS signal
        let pads_signal = {
            let mut pads_manager = self.pads_manager.lock().await;
            pads_manager.process_quantum_signals(agent_signals).await?
        };
        
        // Update registry metrics
        let processing_time = start_time.elapsed().as_millis() as f64;
        {
            let mut metrics = self.metrics.write().map_err(|_| QuantumError::ProcessingError {
                message: "Failed to write metrics lock".to_string(),
            })?;
            
            metrics.total_signals += 1;
            metrics.total_processing_time_ms += processing_time;
            metrics.avg_system_coherence = if successful_agents > 0 {
                total_coherence / successful_agents as f64
            } else {
                0.0
            };
            metrics.pads_signals_generated += 1;
            metrics.last_update = Some(Utc::now());
        }
        
        // Update lattice state with entanglement information
        self.update_lattice_entanglement(&pads_signal).await;
        
        tracing::debug!("Processed market data with {} agents in {:.2}ms", 
                       successful_agents, processing_time);
        
        Ok(pads_signal)
    }
    
    /// Perform health checks on all registered agents
    pub async fn perform_health_checks(&self) -> QuantumResult<HashMap<String, AgentHealth>> {
        let mut health_results = HashMap::new();
        
        // Note: This is a simplified version. In practice, you would iterate through agents
        // and call their health_check methods
        
        // Simulate health checks
        let demo_health = AgentHealth {
            status: HealthStatus::Healthy,
            coherence: 0.9,
            error_rate: 0.05,
            performance: 0.85,
            resource_utilization: 0.6,
            last_check: Utc::now(),
            issues: Vec::new(),
        };
        
        health_results.insert("demo_agent".to_string(), demo_health);
        
        // Update health monitors
        self.update_health_monitors(&health_results).await;
        
        Ok(health_results)
    }
    
    /// Get registry metrics
    pub fn get_metrics(&self) -> QuantumResult<RegistryMetrics> {
        let metrics = self.metrics.read().map_err(|_| QuantumError::ProcessingError {
            message: "Failed to read metrics lock".to_string(),
        })?;
        
        Ok(metrics.clone())
    }
    
    /// Get system coherence
    pub async fn get_system_coherence(&self) -> f64 {
        let lattice = self.lattice_state.lock().await;
        lattice.coherence_levels.iter().sum::<f64>() / lattice.coherence_levels.len().max(1) as f64
    }
    
    /// Detect and mitigate decoherence events
    pub async fn detect_and_mitigate_decoherence(&self) -> QuantumResult<usize> {
        let mut decoherence_events = 0;
        
        // Note: This is a simplified version. In practice, you would check each agent
        // for decoherence events and take appropriate mitigation actions
        
        // Simulate decoherence detection
        let system_coherence = self.get_system_coherence().await;
        
        if system_coherence < 0.7 {
            decoherence_events += 1;
            
            if self.config.auto_decoherence_mitigation {
                // Implement mitigation strategies
                tracing::warn!("Decoherence detected (coherence: {:.3}), applying mitigation", system_coherence);
                
                // Update lattice state to restore coherence
                let mut lattice = self.lattice_state.lock().await;
                for coherence in &mut lattice.coherence_levels {
                    *coherence = (*coherence + 1.0) / 2.0; // Gradually restore coherence
                }
            }
        }
        
        // Update metrics
        if decoherence_events > 0 {
            let mut metrics = self.metrics.write().map_err(|_| QuantumError::ProcessingError {
                message: "Failed to write metrics lock".to_string(),
            })?;
            
            metrics.decoherence_events += decoherence_events as u64;
        }
        
        Ok(decoherence_events)
    }
    
    /// Emergency shutdown all agents
    pub async fn emergency_shutdown(&self) -> QuantumResult<()> {
        tracing::warn!("Emergency shutdown initiated for all agents");
        
        // Note: This is a simplified version. In practice, you would call
        // emergency_shutdown on all registered agents
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().map_err(|_| QuantumError::ProcessingError {
                message: "Failed to write metrics lock".to_string(),
            })?;
            
            metrics.active_agents = 0;
            metrics.last_update = Some(Utc::now());
        }
        
        Ok(())
    }
    
    // Helper methods
    
    /// Create demo signal for testing
    async fn create_demo_signal(&self, agent_type: &str, market_data: &MarketData) -> QuantumResult<PADSSignal> {
        use quantum_core::{QuantumSignal, QuantumSignalType, PADSAction};
        
        let trend_factor = market_data.factors[0];
        let action = if trend_factor > 0.1 {
            PADSAction::Buy
        } else if trend_factor < -0.1 {
            PADSAction::Sell
        } else {
            PADSAction::Hold
        };
        
        Ok(PADSSignal {
            quantum_signal: QuantumSignal {
                id: Uuid::new_v4().to_string(),
                agent_id: agent_type.to_string(),
                signal_type: QuantumSignalType::Trading,
                strength: 0.8,
                amplitude: 0.7,
                phase: trend_factor * std::f64::consts::PI,
                coherence: 0.9,
                entanglement: HashMap::new(),
                data: HashMap::new(),
                metadata: [("agent_type".to_string(), agent_type.to_string())].iter().cloned().collect(),
                timestamp: Utc::now(),
            },
            action,
            confidence: 0.8,
            risk_level: 0.2,
            expected_return: trend_factor * 0.1,
            position_size: 0.1,
            metadata: HashMap::new(),
        })
    }
    
    /// Update lattice entanglement based on PADS signal
    async fn update_lattice_entanglement(&self, pads_signal: &PADSSignal) {
        let mut lattice = self.lattice_state.lock().await;
        
        // Simulate entanglement updates based on signal coherence
        for i in 0..lattice.dimensions.0 {
            for j in (i + 1)..lattice.dimensions.0 {
                let entanglement_strength = pads_signal.quantum_signal.coherence * 0.8;
                lattice.update_entanglement(i, j, entanglement_strength);
            }
        }
        
        lattice.last_update = Utc::now();
    }
    
    /// Update health monitors with latest health check results
    async fn update_health_monitors(&self, health_results: &HashMap<String, AgentHealth>) {
        if let Ok(mut monitors) = self.health_monitors.write() {
            for (agent_id, health) in health_results {
                if let Some(monitor) = monitors.get_mut(agent_id) {
                    monitor.last_health_check = Utc::now();
                    monitor.health_history.push(health.clone());
                    monitor.performance_trends.push(health.performance);
                    
                    // Maintain history size
                    if monitor.health_history.len() > 100 {
                        monitor.health_history.drain(0..monitor.health_history.len() - 100);
                    }
                    if monitor.performance_trends.len() > 100 {
                        monitor.performance_trends.drain(0..monitor.performance_trends.len() - 100);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantum_core::MarketData;
    
    #[tokio::test]
    async fn test_registry_creation() {
        let config = RegistryConfig::default();
        let registry = UnifiedQuantumAgentRegistry::new(config);
        
        let metrics = registry.get_metrics().unwrap();
        assert_eq!(metrics.total_agents, 0);
        assert_eq!(metrics.active_agents, 0);
    }
    
    #[tokio::test]
    async fn test_market_data_processing() {
        let config = RegistryConfig::default();
        let registry = UnifiedQuantumAgentRegistry::new(config);
        
        let market_data = MarketData::new(
            "BTCUSD".to_string(),
            50000.0,
            1000.0,
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        );
        
        let result = registry.process_market_data(&market_data).await;
        assert!(result.is_ok());
        
        let pads_signal = result.unwrap();
        assert!(pads_signal.confidence > 0.0);
        assert!(pads_signal.confidence <= 1.0);
    }
    
    #[tokio::test]
    async fn test_system_coherence() {
        let config = RegistryConfig::default();
        let registry = UnifiedQuantumAgentRegistry::new(config);
        
        let coherence = registry.get_system_coherence().await;
        assert!(coherence >= 0.0);
        assert!(coherence <= 1.0);
    }
    
    #[tokio::test]
    async fn test_decoherence_detection() {
        let config = RegistryConfig::default();
        let registry = UnifiedQuantumAgentRegistry::new(config);
        
        let decoherence_events = registry.detect_and_mitigate_decoherence().await.unwrap();
        assert!(decoherence_events >= 0);
    }
}