//! QBMIA & Whale Defense Hive-Mind Integration
//! 
//! Coordinates specialized agent swarms for quantum-biological intelligence
//! and sub-microsecond whale defense with the QAR decision engine.

use crate::execution_context::ExecutionContext;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Configuration for QBMIA & Whale Defense integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QBMIAWhaleConfig {
    /// Enable QBMIA biological pattern analysis
    pub enable_qbmia: bool,
    /// Enable whale defense systems
    pub enable_whale_defense: bool,
    /// Hive-mind coordination level
    pub coordination_level: CoordinationLevel,
    /// Performance constraints
    pub max_response_time_ns: u64,
    /// Agent specialization depth
    pub specialization_depth: u8,
    /// Cross-system communication frequency
    pub communication_frequency_hz: u32,
}

/// Levels of hive-mind coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationLevel {
    Minimal,    // Independent operation
    Standard,   // Periodic coordination
    Intensive,  // Continuous coordination
    Symbiotic,  // Deep integration
}

impl Default for QBMIAWhaleConfig {
    fn default() -> Self {
        Self {
            enable_qbmia: true,
            enable_whale_defense: true,
            coordination_level: CoordinationLevel::Intensive,
            max_response_time_ns: 500, // Sub-microsecond
            specialization_depth: 3,
            communication_frequency_hz: 1000, // 1kHz coordination
        }
    }
}

/// QBMIA Intelligence Report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QBMIAIntelligence {
    /// Biological pattern confidence
    pub biological_confidence: f64,
    /// Market intelligence score
    pub market_intelligence: f64,
    /// Quantum enhancement factor
    pub quantum_enhancement: f64,
    /// Adaptive learning insights
    pub learning_insights: Vec<String>,
    /// Performance metrics
    pub performance_metrics: HashMap<String, f64>,
    /// Acceleration factors
    pub acceleration_factors: AccelerationMetrics,
}

/// Acceleration performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccelerationMetrics {
    pub gpu_utilization: f64,
    pub simd_efficiency: f64,
    pub memory_bandwidth: f64,
    pub cache_hit_rate: f64,
}

/// Whale Defense Status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleDefenseStatus {
    /// Threat level (0.0 = no threat, 1.0 = maximum threat)
    pub threat_level: f64,
    /// Whale activity detected
    pub whale_detected: bool,
    /// Defense mechanisms active
    pub defenses_active: Vec<DefenseMechanism>,
    /// Real-time metrics
    pub realtime_metrics: RealtimeMetrics,
    /// ML analysis results
    pub ml_analysis: MLAnalysisResult,
}

/// Types of defense mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DefenseMechanism {
    Steganography,
    TimingRandomization,
    OrderFragmentation,
    LatencyMasking,
    VolumeObfuscation,
    PatternScrambling,
}

/// Real-time performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeMetrics {
    pub response_time_ns: u64,
    pub throughput_ops_per_sec: u64,
    pub memory_usage_mb: f64,
    pub cpu_utilization: f64,
}

/// ML analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLAnalysisResult {
    pub anomaly_score: f64,
    pub behavioral_classification: String,
    pub confidence: f64,
    pub patterns_detected: Vec<String>,
}

/// Integrated decision enhancement from QBMIA & Whale Defense
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegratedEnhancement {
    /// QBMIA intelligence contribution
    pub qbmia_intelligence: QBMIAIntelligence,
    /// Whale defense status
    pub whale_defense: WhaleDefenseStatus,
    /// Combined threat assessment
    pub threat_assessment: f64,
    /// Recommended risk adjustments
    pub risk_adjustments: HashMap<String, f64>,
    /// Performance optimizations
    pub performance_optimizations: Vec<String>,
    /// Execution timing recommendations
    pub timing_recommendations: TimingRecommendations,
}

/// Execution timing recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingRecommendations {
    /// Optimal execution delay (nanoseconds)
    pub optimal_delay_ns: u64,
    /// Randomization window
    pub randomization_window_ns: u64,
    /// Steganography enabled
    pub use_steganography: bool,
    /// Fragment large orders
    pub fragment_orders: bool,
}

/// Hive-Mind Coordinator for QBMIA & Whale Defense
#[derive(Debug)]
pub struct QBMIAWhaleHiveMind {
    config: QBMIAWhaleConfig,
    
    // Agent communication channels (simulated)
    qbmia_agents: QBMIAAgentCluster,
    whale_defense_agents: WhaleDefenseAgentCluster,
    
    // Coordination state
    coordination_state: Arc<Mutex<CoordinationState>>,
    performance_history: Vec<IntegratedEnhancement>,
    
    // Real-time metrics
    metrics: Arc<Mutex<HiveMindMetrics>>,
}

/// QBMIA Agent Cluster coordinator
#[derive(Debug)]
struct QBMIAAgentCluster {
    core_agent_status: AgentStatus,
    acceleration_agent_status: AgentStatus,
    quantum_agent_status: AgentStatus,
    biological_agent_status: AgentStatus,
}

/// Whale Defense Agent Cluster coordinator  
#[derive(Debug)]
struct WhaleDefenseAgentCluster {
    core_agent_status: AgentStatus,
    ml_agent_status: AgentStatus,
    realtime_agent_status: AgentStatus,
}

/// Individual agent status
#[derive(Debug, Clone)]
struct AgentStatus {
    agent_id: String,
    performance_score: f64,
    response_time_ns: u64,
    active_tasks: u32,
    specialization_focus: f64,
}

/// Cross-system coordination state
#[derive(Debug, Clone)]
struct CoordinationState {
    last_sync_timestamp: Instant,
    coordination_quality: f64,
    cross_system_latency_ns: u64,
    active_coordinations: u32,
}

/// Hive-mind performance metrics
#[derive(Debug, Clone)]
struct HiveMindMetrics {
    total_coordinations: u64,
    average_response_time_ns: u64,
    coordination_success_rate: f64,
    threat_detection_accuracy: f64,
    qbmia_intelligence_quality: f64,
}

impl QBMIAWhaleHiveMind {
    /// Initialize the hive-mind coordinator
    pub fn new(config: QBMIAWhaleConfig) -> Result<Self, crate::QARError> {
        let qbmia_agents = QBMIAAgentCluster {
            core_agent_status: AgentStatus::new("QBMIA-Core-Agent".to_string()),
            acceleration_agent_status: AgentStatus::new("QBMIA-Acceleration-Agent".to_string()),
            quantum_agent_status: AgentStatus::new("QBMIA-Quantum-Agent".to_string()),
            biological_agent_status: AgentStatus::new("QBMIA-Biological-Agent".to_string()),
        };
        
        let whale_defense_agents = WhaleDefenseAgentCluster {
            core_agent_status: AgentStatus::new("Whale-Defense-Core-Agent".to_string()),
            ml_agent_status: AgentStatus::new("Whale-Defense-ML-Agent".to_string()),
            realtime_agent_status: AgentStatus::new("Whale-Defense-Realtime-Agent".to_string()),
        };
        
        let coordination_state = Arc::new(Mutex::new(CoordinationState {
            last_sync_timestamp: Instant::now(),
            coordination_quality: 1.0,
            cross_system_latency_ns: 100, // Start with 100ns baseline
            active_coordinations: 0,
        }));
        
        let metrics = Arc::new(Mutex::new(HiveMindMetrics {
            total_coordinations: 0,
            average_response_time_ns: 0,
            coordination_success_rate: 1.0,
            threat_detection_accuracy: 0.95,
            qbmia_intelligence_quality: 0.9,
        }));
        
        Ok(Self {
            config,
            qbmia_agents,
            whale_defense_agents,
            coordination_state,
            performance_history: Vec::with_capacity(1000),
            metrics,
        })
    }
    
    /// Get integrated enhancement for QAR decision making
    pub async fn get_integrated_enhancement(&mut self, 
                                          market_data: &crate::MarketData,
                                          execution_context: &ExecutionContext) -> Result<IntegratedEnhancement, crate::QARError> {
        let start_time = Instant::now();
        
        // Coordinate with QBMIA agents for intelligence gathering
        let qbmia_intelligence = if self.config.enable_qbmia {
            self.coordinate_qbmia_analysis(market_data).await?
        } else {
            QBMIAIntelligence::neutral()
        };
        
        // Coordinate with Whale Defense agents for threat assessment
        let whale_defense = if self.config.enable_whale_defense {
            self.coordinate_whale_defense_analysis(market_data, execution_context).await?
        } else {
            WhaleDefenseStatus::safe()
        };
        
        // Perform cross-system coordination
        let threat_assessment = self.assess_combined_threat(&qbmia_intelligence, &whale_defense);
        let risk_adjustments = self.calculate_risk_adjustments(&qbmia_intelligence, &whale_defense);
        let performance_optimizations = self.suggest_performance_optimizations(&qbmia_intelligence, &whale_defense);
        let timing_recommendations = self.generate_timing_recommendations(&whale_defense);
        
        // Update coordination metrics
        let elapsed = start_time.elapsed();
        self.update_coordination_metrics(elapsed).await;
        
        // Check performance constraints
        if elapsed.as_nanos() as u64 > self.config.max_response_time_ns {
            return Err(crate::QARError::Performance { 
                message: format!("Hive-mind coordination exceeded {}ns target: {}ns", 
                               self.config.max_response_time_ns, elapsed.as_nanos()) 
            });
        }
        
        let enhancement = IntegratedEnhancement {
            qbmia_intelligence,
            whale_defense,
            threat_assessment,
            risk_adjustments,
            performance_optimizations,
            timing_recommendations,
        };
        
        // Store in history
        self.performance_history.push(enhancement.clone());
        if self.performance_history.len() > 1000 {
            self.performance_history.remove(0);
        }
        
        Ok(enhancement)
    }
    
    /// Coordinate with QBMIA agents for biological intelligence
    async fn coordinate_qbmia_analysis(&mut self, market_data: &crate::MarketData) -> Result<QBMIAIntelligence, crate::QARError> {
        // Simulate hive-mind coordination with specialized QBMIA agents
        // In production, this would use actual agent communication protocols
        
        // Core agent: biological pattern analysis
        let biological_confidence = self.simulate_biological_analysis(market_data);
        
        // Acceleration agent: performance optimization
        let acceleration_metrics = self.simulate_acceleration_analysis();
        
        // Quantum agent: quantum-enhanced ML
        let quantum_enhancement = self.simulate_quantum_analysis(market_data);
        
        // Biological agent: adaptive learning
        let learning_insights = self.simulate_biological_learning(market_data);
        
        // Market intelligence synthesis
        let market_intelligence = (biological_confidence + quantum_enhancement) / 2.0;
        
        let mut performance_metrics = HashMap::new();
        performance_metrics.insert("biological_accuracy".to_string(), biological_confidence);
        performance_metrics.insert("quantum_coherence".to_string(), quantum_enhancement);
        performance_metrics.insert("learning_rate".to_string(), 0.85);
        performance_metrics.insert("adaptation_speed".to_string(), 0.92);
        
        Ok(QBMIAIntelligence {
            biological_confidence,
            market_intelligence,
            quantum_enhancement,
            learning_insights,
            performance_metrics,
            acceleration_factors: acceleration_metrics,
        })
    }
    
    /// Coordinate with Whale Defense agents for threat analysis
    async fn coordinate_whale_defense_analysis(&mut self, 
                                             market_data: &crate::MarketData,
                                             execution_context: &ExecutionContext) -> Result<WhaleDefenseStatus, crate::QARError> {
        // Core agent: whale detection and steganography
        let (whale_detected, threat_level) = self.simulate_whale_detection(market_data);
        
        // ML agent: behavioral analysis
        let ml_analysis = self.simulate_ml_analysis(market_data);
        
        // Realtime agent: performance metrics
        let realtime_metrics = self.simulate_realtime_metrics();
        
        // Determine active defenses based on threat level
        let defenses_active = self.select_defense_mechanisms(threat_level);
        
        Ok(WhaleDefenseStatus {
            threat_level,
            whale_detected,
            defenses_active,
            realtime_metrics,
            ml_analysis,
        })
    }
    
    /// Assess combined threat from both systems
    fn assess_combined_threat(&self, qbmia: &QBMIAIntelligence, whale_defense: &WhaleDefenseStatus) -> f64 {
        let biological_threat = 1.0 - qbmia.biological_confidence;
        let whale_threat = whale_defense.threat_level;
        let market_uncertainty = 1.0 - qbmia.market_intelligence;
        
        // Weighted combination of threat factors
        (biological_threat * 0.3 + whale_threat * 0.5 + market_uncertainty * 0.2).min(1.0)
    }
    
    /// Calculate risk adjustments based on integrated analysis
    fn calculate_risk_adjustments(&self, qbmia: &QBMIAIntelligence, whale_defense: &WhaleDefenseStatus) -> HashMap<String, f64> {
        let mut adjustments = HashMap::new();
        
        // QBMIA-based adjustments
        adjustments.insert("biological_risk_factor".to_string(), 1.0 - qbmia.biological_confidence);
        adjustments.insert("quantum_uncertainty".to_string(), 1.0 - qbmia.quantum_enhancement);
        
        // Whale defense adjustments
        adjustments.insert("whale_threat_adjustment".to_string(), whale_defense.threat_level);
        adjustments.insert("steganography_overhead".to_string(), 
                          if whale_defense.defenses_active.contains(&DefenseMechanism::Steganography) { 0.05 } else { 0.0 });
        
        // Combined system risk
        let combined_risk = self.assess_combined_threat(qbmia, whale_defense);
        adjustments.insert("integrated_risk_multiplier".to_string(), 1.0 + combined_risk * 0.2);
        
        adjustments
    }
    
    /// Suggest performance optimizations
    fn suggest_performance_optimizations(&self, qbmia: &QBMIAIntelligence, whale_defense: &WhaleDefenseStatus) -> Vec<String> {
        let mut optimizations = Vec::new();
        
        // QBMIA optimizations
        if qbmia.acceleration_factors.gpu_utilization < 0.8 {
            optimizations.push("Increase GPU utilization for biological pattern analysis".to_string());
        }
        
        if qbmia.acceleration_factors.simd_efficiency < 0.9 {
            optimizations.push("Optimize SIMD operations for quantum calculations".to_string());
        }
        
        // Whale defense optimizations
        if whale_defense.realtime_metrics.response_time_ns > 300 {
            optimizations.push("Optimize lockfree algorithms for faster whale detection".to_string());
        }
        
        if whale_defense.realtime_metrics.memory_usage_mb > 100.0 {
            optimizations.push("Reduce memory footprint in real-time processing".to_string());
        }
        
        // Cross-system optimizations
        optimizations.push("Enable cross-agent memory sharing for reduced latency".to_string());
        optimizations.push("Implement adaptive coordination frequency based on market volatility".to_string());
        
        optimizations
    }
    
    /// Generate timing recommendations for execution
    fn generate_timing_recommendations(&self, whale_defense: &WhaleDefenseStatus) -> TimingRecommendations {
        let base_delay = if whale_defense.whale_detected { 200 } else { 50 }; // nanoseconds
        let randomization_window = if whale_defense.threat_level > 0.7 { 500 } else { 100 };
        
        TimingRecommendations {
            optimal_delay_ns: base_delay,
            randomization_window_ns: randomization_window,
            use_steganography: whale_defense.threat_level > 0.6,
            fragment_orders: whale_defense.threat_level > 0.8,
        }
    }
    
    /// Update coordination performance metrics
    async fn update_coordination_metrics(&self, elapsed: Duration) {
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.total_coordinations += 1;
            let elapsed_ns = elapsed.as_nanos() as u64;
            
            metrics.average_response_time_ns = 
                (metrics.average_response_time_ns * (metrics.total_coordinations - 1) + elapsed_ns) 
                / metrics.total_coordinations;
            
            // Update coordination success rate
            let success = elapsed_ns <= self.config.max_response_time_ns;
            metrics.coordination_success_rate = 
                (metrics.coordination_success_rate * (metrics.total_coordinations - 1) as f64 + if success { 1.0 } else { 0.0 })
                / metrics.total_coordinations as f64;
        }
    }
    
    // Simulation methods (would be replaced with actual agent communication)
    fn simulate_biological_analysis(&self, _market_data: &crate::MarketData) -> f64 {
        0.87 + (rand::random::<f64>() - 0.5) * 0.1
    }
    
    fn simulate_acceleration_analysis(&self) -> AccelerationMetrics {
        AccelerationMetrics {
            gpu_utilization: 0.85 + rand::random::<f64>() * 0.1,
            simd_efficiency: 0.92 + rand::random::<f64>() * 0.05,
            memory_bandwidth: 0.78 + rand::random::<f64>() * 0.15,
            cache_hit_rate: 0.94 + rand::random::<f64>() * 0.05,
        }
    }
    
    fn simulate_quantum_analysis(&self, _market_data: &crate::MarketData) -> f64 {
        0.91 + (rand::random::<f64>() - 0.5) * 0.08
    }
    
    fn simulate_biological_learning(&self, _market_data: &crate::MarketData) -> Vec<String> {
        vec![
            "Swarm intelligence detected positive market sentiment".to_string(),
            "Biological pattern suggests trend continuation".to_string(),
            "Adaptive learning recommends increased position size".to_string(),
        ]
    }
    
    fn simulate_whale_detection(&self, _market_data: &crate::MarketData) -> (bool, f64) {
        let threat_level = rand::random::<f64>() * 0.3; // Usually low threat
        (threat_level > 0.2, threat_level)
    }
    
    fn simulate_ml_analysis(&self, _market_data: &crate::MarketData) -> MLAnalysisResult {
        MLAnalysisResult {
            anomaly_score: rand::random::<f64>() * 0.4,
            behavioral_classification: "Normal Market Behavior".to_string(),
            confidence: 0.87 + rand::random::<f64>() * 0.1,
            patterns_detected: vec!["volume_clustering".to_string(), "price_momentum".to_string()],
        }
    }
    
    fn simulate_realtime_metrics(&self) -> RealtimeMetrics {
        RealtimeMetrics {
            response_time_ns: 150 + (rand::random::<u64>() % 100),
            throughput_ops_per_sec: 50000 + (rand::random::<u64>() % 10000),
            memory_usage_mb: 45.0 + rand::random::<f64>() * 20.0,
            cpu_utilization: 0.15 + rand::random::<f64>() * 0.1,
        }
    }
    
    fn select_defense_mechanisms(&self, threat_level: f64) -> Vec<DefenseMechanism> {
        let mut defenses = Vec::new();
        
        if threat_level > 0.3 {
            defenses.push(DefenseMechanism::TimingRandomization);
        }
        
        if threat_level > 0.5 {
            defenses.push(DefenseMechanism::OrderFragmentation);
            defenses.push(DefenseMechanism::LatencyMasking);
        }
        
        if threat_level > 0.7 {
            defenses.push(DefenseMechanism::Steganography);
            defenses.push(DefenseMechanism::VolumeObfuscation);
        }
        
        if threat_level > 0.9 {
            defenses.push(DefenseMechanism::PatternScrambling);
        }
        
        defenses
    }
    
    /// Get hive-mind performance metrics
    pub fn get_hive_mind_metrics(&self) -> HiveMindMetrics {
        if let Ok(metrics) = self.metrics.lock() {
            metrics.clone()
        } else {
            HiveMindMetrics {
                total_coordinations: 0,
                average_response_time_ns: 0,
                coordination_success_rate: 0.0,
                threat_detection_accuracy: 0.0,
                qbmia_intelligence_quality: 0.0,
            }
        }
    }
}

impl AgentStatus {
    fn new(agent_id: String) -> Self {
        Self {
            agent_id,
            performance_score: 0.9,
            response_time_ns: 200,
            active_tasks: 0,
            specialization_focus: 0.95,
        }
    }
}

impl QBMIAIntelligence {
    fn neutral() -> Self {
        Self {
            biological_confidence: 0.5,
            market_intelligence: 0.5,
            quantum_enhancement: 1.0,
            learning_insights: vec!["QBMIA disabled".to_string()],
            performance_metrics: HashMap::new(),
            acceleration_factors: AccelerationMetrics {
                gpu_utilization: 0.0,
                simd_efficiency: 0.0,
                memory_bandwidth: 0.0,
                cache_hit_rate: 0.0,
            },
        }
    }
}

impl WhaleDefenseStatus {
    fn safe() -> Self {
        Self {
            threat_level: 0.0,
            whale_detected: false,
            defenses_active: Vec::new(),
            realtime_metrics: RealtimeMetrics {
                response_time_ns: 100,
                throughput_ops_per_sec: 100000,
                memory_usage_mb: 10.0,
                cpu_utilization: 0.05,
            },
            ml_analysis: MLAnalysisResult {
                anomaly_score: 0.0,
                behavioral_classification: "Whale Defense Disabled".to_string(),
                confidence: 1.0,
                patterns_detected: Vec::new(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hive_mind_initialization() {
        let config = QBMIAWhaleConfig::default();
        let hive_mind = QBMIAWhaleHiveMind::new(config);
        assert!(hive_mind.is_ok());
    }

    #[tokio::test]
    async fn test_integrated_enhancement() {
        let config = QBMIAWhaleConfig::default();
        let mut hive_mind = QBMIAWhaleHiveMind::new(config).unwrap();
        
        let market_data = crate::MarketData {
            symbol: "BTC/USDT".to_string(),
            current_price: 50000.0,
            possible_outcomes: vec![52000.0, 51000.0, 49000.0, 48000.0],
            buy_probabilities: vec![0.3, 0.3, 0.2, 0.2],
            sell_probabilities: vec![0.2, 0.2, 0.3, 0.3],
            hold_probabilities: vec![0.25, 0.25, 0.25, 0.25],
            frame: prospect_theory::FramingContext {
                frame_type: prospect_theory::FrameType::Neutral,
                emphasis: 0.5,
            },
            timestamp: 1640995200000,
        };
        
        let context = ExecutionContext::new(&crate::QARConfig::default()).unwrap();
        let enhancement = hive_mind.get_integrated_enhancement(&market_data, &context).await;
        
        assert!(enhancement.is_ok());
        let enhancement = enhancement.unwrap();
        
        assert!(enhancement.threat_assessment >= 0.0 && enhancement.threat_assessment <= 1.0);
        assert!(!enhancement.risk_adjustments.is_empty());
        assert!(!enhancement.performance_optimizations.is_empty());
    }

    #[test]
    fn test_threat_assessment() {
        let config = QBMIAWhaleConfig::default();
        let hive_mind = QBMIAWhaleHiveMind::new(config).unwrap();
        
        let qbmia = QBMIAIntelligence {
            biological_confidence: 0.8,
            market_intelligence: 0.9,
            quantum_enhancement: 0.95,
            learning_insights: Vec::new(),
            performance_metrics: HashMap::new(),
            acceleration_factors: AccelerationMetrics {
                gpu_utilization: 0.85,
                simd_efficiency: 0.92,
                memory_bandwidth: 0.78,
                cache_hit_rate: 0.94,
            },
        };
        
        let whale_defense = WhaleDefenseStatus {
            threat_level: 0.3,
            whale_detected: false,
            defenses_active: Vec::new(),
            realtime_metrics: RealtimeMetrics {
                response_time_ns: 150,
                throughput_ops_per_sec: 60000,
                memory_usage_mb: 50.0,
                cpu_utilization: 0.2,
            },
            ml_analysis: MLAnalysisResult {
                anomaly_score: 0.1,
                behavioral_classification: "Normal".to_string(),
                confidence: 0.9,
                patterns_detected: Vec::new(),
            },
        };
        
        let threat = hive_mind.assess_combined_threat(&qbmia, &whale_defense);
        assert!(threat >= 0.0 && threat <= 1.0);
        assert!(threat < 0.5); // Should be low threat given inputs
    }

    #[test]
    fn test_defense_mechanism_selection() {
        let config = QBMIAWhaleConfig::default();
        let hive_mind = QBMIAWhaleHiveMind::new(config).unwrap();
        
        // Low threat
        let defenses_low = hive_mind.select_defense_mechanisms(0.2);
        assert!(defenses_low.is_empty());
        
        // Medium threat
        let defenses_medium = hive_mind.select_defense_mechanisms(0.6);
        assert!(!defenses_medium.is_empty());
        assert!(defenses_medium.contains(&DefenseMechanism::TimingRandomization));
        
        // High threat
        let defenses_high = hive_mind.select_defense_mechanisms(0.9);
        assert!(defenses_high.len() >= 3);
        assert!(defenses_high.contains(&DefenseMechanism::Steganography));
    }
}