//! Core traits for Quantum Agentic Reasoning

use async_trait::async_trait;
use std::collections::HashMap;
use super::types::*;
use super::{FactorMap, MarketContext, TradingDecision, QarResult, QuantumResult, CoreCircuitParams, CoreExecutionContext};

/// Extended QuantumCircuit trait that builds on the core trait with QAR-specific methods
#[async_trait]
pub trait QuantumCircuit: Send + Sync {
    /// Execute the quantum circuit with given parameters
    async fn execute(&self, params: &CircuitParams, context: &ExecutionContext) -> QarResult<QuantumResult>;
    
    /// Get the circuit name/identifier
    fn name(&self) -> &str;
    
    /// Get the number of qubits required
    fn num_qubits(&self) -> usize;
    
    /// Get the estimated execution time in milliseconds
    fn estimated_execution_time_ms(&self) -> u64;
    
    /// Check if the circuit supports classical fallback
    fn supports_classical_fallback(&self) -> bool;
    
    /// Execute classical fallback if quantum execution fails
    async fn classical_fallback(&self, params: &CircuitParams) -> QarResult<QuantumResult>;
    
    /// Validate circuit parameters
    fn validate_parameters(&self, params: &CircuitParams) -> QarResult<()>;
    
    /// Get circuit metadata
    fn metadata(&self) -> HashMap<String, String> {
        HashMap::new()
    }
}

/// Re-export core parameter types with aliases
pub use super::CoreCircuitParams as CircuitParams;
pub use super::CoreExecutionContext as ExecutionContext;

/// Trait for decision making engines
#[async_trait]
pub trait DecisionEngine: Send + Sync {
    /// Make a trading decision based on factors and context
    async fn make_decision(
        &self,
        factors: &FactorMap,
        context: &MarketContext,
    ) -> QarResult<TradingDecision>;
    
    /// Update the engine with feedback from a previous decision
    async fn update_with_feedback(
        &mut self,
        decision_id: &str,
        outcome: DecisionOutcome,
    ) -> QarResult<()>;
    
    /// Get the current confidence threshold
    fn confidence_threshold(&self) -> f64;
    
    /// Set the confidence threshold
    fn set_confidence_threshold(&mut self, threshold: f64);
    
    /// Get performance metrics
    fn get_metrics(&self) -> DecisionMetrics;
}

/// Trait for market analysis
#[async_trait]
pub trait MarketAnalyzer: Send + Sync {
    /// Analyze market regime from factors
    async fn analyze_regime(&self, factors: &FactorMap) -> QarResult<RegimeAnalysis>;
    
    /// Detect patterns in market data
    async fn detect_patterns(&self, factors: &FactorMap) -> QarResult<Vec<PatternMatch>>;
    
    /// Predict market direction
    async fn predict_direction(&self, factors: &FactorMap) -> QarResult<MarketPrediction>;
    
    /// Calculate market volatility
    async fn calculate_volatility(&self, factors: &FactorMap) -> QarResult<f64>;
}

/// Trait for pattern recognition
#[async_trait]
pub trait PatternRecognizer: Send + Sync {
    /// Recognize patterns in factor data
    async fn recognize_patterns(&self, factors: &FactorMap) -> QarResult<Vec<PatternMatch>>;
    
    /// Learn a new pattern
    async fn learn_pattern(&mut self, pattern: &PatternData) -> QarResult<()>;
    
    /// Get similarity between two patterns
    fn calculate_similarity(&self, pattern1: &PatternData, pattern2: &PatternData) -> f64;
    
    /// Get stored patterns count
    fn pattern_count(&self) -> usize;
    
    /// Clear stored patterns
    fn clear_patterns(&mut self);
}

/// Trait for hardware abstraction
#[async_trait]
pub trait HardwareInterface: Send + Sync {
    /// Check if quantum hardware is available
    async fn is_quantum_available(&self) -> bool;
    
    /// Get available quantum backends
    async fn get_quantum_backends(&self) -> Vec<String>;
    
    /// Execute quantum circuit on hardware
    async fn execute_quantum(
        &self,
        circuit: &dyn QuantumCircuit,
        params: &CircuitParams,
    ) -> QarResult<QuantumResult>;
    
    /// Get hardware capabilities
    async fn get_capabilities(&self) -> HardwareCapabilities;
    
    /// Get hardware metrics
    async fn get_metrics(&self) -> HardwareMetrics;
}

/// Trait for caching operations
pub trait CacheManager<K, V>: Send + Sync {
    /// Get value from cache
    fn get(&self, key: &K) -> Option<V>;
    
    /// Put value in cache
    fn put(&self, key: K, value: V);
    
    /// Remove value from cache
    fn remove(&self, key: &K) -> Option<V>;
    
    /// Clear all cache entries
    fn clear(&self);
    
    /// Get cache statistics
    fn stats(&self) -> CacheStats;
    
    /// Check if cache contains key
    fn contains(&self, key: &K) -> bool;
    
    /// Get cache size
    fn size(&self) -> usize;
    
    /// Get cache capacity
    fn capacity(&self) -> usize;
}

/// Trait for memory management
pub trait MemoryManager: Send + Sync {
    /// Store decision in memory
    fn store_decision(&mut self, decision: TradingDecision);
    
    /// Store pattern in memory
    fn store_pattern(&mut self, pattern: PatternData);
    
    /// Get recent decisions
    fn get_recent_decisions(&self, limit: usize) -> Vec<TradingDecision>;
    
    /// Get stored patterns
    fn get_patterns(&self) -> Vec<PatternData>;
    
    /// Clear old memories
    fn cleanup(&mut self);
    
    /// Get memory usage statistics
    fn get_memory_stats(&self) -> MemoryStats;
}

/// Trait for performance monitoring
pub trait PerformanceMonitor: Send + Sync {
    /// Record execution time
    fn record_execution_time(&self, operation: &str, duration_ms: f64);
    
    /// Record quantum vs classical usage
    fn record_quantum_usage(&self, used_quantum: bool);
    
    /// Record cache hit/miss
    fn record_cache_event(&self, hit: bool);
    
    /// Get performance metrics
    fn get_metrics(&self) -> PerformanceMetrics;
    
    /// Reset metrics
    fn reset_metrics(&self);
}

/// Hardware capabilities
#[derive(Debug, Clone)]
pub struct HardwareCapabilities {
    /// Maximum qubits available
    pub max_qubits: usize,
    /// Available quantum backends
    pub quantum_backends: Vec<String>,
    /// GPU acceleration available
    pub gpu_available: bool,
    /// Supported quantum gates
    pub supported_gates: Vec<String>,
}

/// Trait for serialization support
pub trait Serializable {
    /// Serialize to JSON string
    fn to_json(&self) -> QarResult<String>;
    
    /// Deserialize from JSON string
    fn from_json(json: &str) -> QarResult<Self>
    where
        Self: Sized;
    
    /// Serialize to binary format
    fn to_binary(&self) -> QarResult<Vec<u8>>;
    
    /// Deserialize from binary format
    fn from_binary(data: &[u8]) -> QarResult<Self>
    where
        Self: Sized;
}

/// Trait for state management
pub trait StatefulComponent {
    /// Get current state
    fn get_state(&self) -> QarResult<ComponentState>;
    
    /// Set state
    fn set_state(&mut self, state: ComponentState) -> QarResult<()>;
    
    /// Save state to storage
    fn save_state(&self, path: &str) -> QarResult<()>;
    
    /// Load state from storage
    fn load_state(&mut self, path: &str) -> QarResult<()>;
    
    /// Reset component to initial state
    fn reset(&mut self) -> QarResult<()>;
}

/// Component state representation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ComponentState {
    /// Component name
    pub name: String,
    /// State data
    pub data: HashMap<String, serde_json::Value>,
    /// State timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// State version
    pub version: u32,
}

impl ComponentState {
    /// Create new component state
    pub fn new(name: String) -> Self {
        Self {
            name,
            data: HashMap::new(),
            timestamp: chrono::Utc::now(),
            version: 1,
        }
    }

    /// Add state data
    pub fn with_data(mut self, key: String, value: serde_json::Value) -> Self {
        self.data.insert(key, value);
        self
    }

    /// Get state data
    pub fn get_data(&self, key: &str) -> Option<&serde_json::Value> {
        self.data.get(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_component_state() {
        let state = ComponentState::new("test_component".to_string())
            .with_data("param1".to_string(), serde_json::Value::Number(serde_json::Number::from(42)));
        
        assert_eq!(state.name, "test_component");
        assert_eq!(state.version, 1);
        assert!(state.get_data("param1").is_some());
    }

    #[test]
    fn test_decision_outcome() {
        let outcome = DecisionOutcome::Success {
            profit: 100.0,
            duration_ms: 5000,
        };
        
        match outcome {
            DecisionOutcome::Success { profit, duration_ms } => {
                assert_eq!(profit, 100.0);
                assert_eq!(duration_ms, 5000);
            }
            _ => panic!("Expected success outcome"),
        }
    }

    #[test]
    fn test_pattern_data() {
        let pattern = PatternData {
            id: "test_pattern".to_string(),
            features: vec![0.1, 0.2, 0.3],
            timestamp: chrono::Utc::now(),
            metadata: HashMap::new(),
        };
        
        assert_eq!(pattern.id, "test_pattern");
        assert_eq!(pattern.features.len(), 3);
    }
}