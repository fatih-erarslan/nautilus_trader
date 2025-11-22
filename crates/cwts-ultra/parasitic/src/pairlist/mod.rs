//! # Parasitic Pairlist Module
//! 
//! Core pairlist selection system using biomimetic parasitic organisms
//! integrated with QADO quantum memory and CQGS compliance.

pub mod manager;
pub mod selection_engine;
pub mod biomimetic_orchestra;
pub mod simd_scoring;
pub mod simd_pair_scorer;
pub mod quantum_integration;
pub mod whale_detection;
pub mod zombie_detection;
pub mod mycelial_analysis;
pub mod resource_handlers;

pub use manager::ParasiticPairlistManager;
pub use selection_engine::ParasiticSelectionEngine;
pub use biomimetic_orchestra::BiomimeticOrchestra;
pub use simd_scoring::SimdPairScorer;
// pub use quantum_integration::ParasiticQuantumMemory; // TODO: implement when quantum types available
pub use whale_detection::{WhaleNestDetector, WhaleNest, WhaleOrder};
pub use zombie_detection::{ZombiePairDetector, ZombiePair};
pub use mycelial_analysis::{MycelialNetworkAnalyzer, CorrelationNetwork};
pub use resource_handlers::{ParasiticResourceManager, ResourceMetrics};
pub use crate::quantum::memory::{OrganismAnalysis, QuantumEnhancedAnalysis};

use serde::{Serialize, Deserialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use async_trait::async_trait;

/// Stub trait for parasitic organisms (TODO: move to organisms module when available)
#[async_trait]
pub trait ParasiticOrganism {
    fn id(&self) -> Uuid;
    fn organism_type(&self) -> &str;
    async fn analyze_pair(&self, pair: &TradingPair) -> f64;
    async fn infect_pair(&self, pair_id: &str, vulnerability: f64) -> Result<(), Box<dyn std::error::Error>>;
}

/// Stub types for missing analysis structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CQGSValidatedAnalysis {
    pub pair_id: String,
    pub validation_score: f64,
    pub compliance_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencePattern {
    pub pattern_id: String,
    pub strength: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergenceDetector {
    pub threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusVoting {
    pub required_votes: u32,
}

impl EmergenceDetector {
    pub fn new() -> Self {
        Self { threshold: 0.7 }
    }
    
    pub async fn detect_emergence_patterns(&self, _pairs: &[TradingPair]) -> Vec<EmergencePattern> {
        // Stub implementation
        vec![]
    }
}

impl ConsensusVoting {
    pub fn new() -> Self {
        Self { required_votes: 3 }
    }
}

/// More stub types for missing structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConditions {
    pub volatility: f64,
    pub liquidity: f64,
    pub trend_direction: String,
    pub trend_strength: f64,
    pub noise_level: f64,
    pub spread: f64,
}

/// Basic trading pair representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingPair {
    pub pair_id: String,
    pub base_asset: String,
    pub quote_asset: String,
    pub current_price: f64,
    pub volume_24h: f64,
    pub volatility: f64,
    pub liquidity_score: f64,
    pub last_update: DateTime<Utc>,
}

/// Trading pair selection with parasitic characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectedPair {
    pub pair_id: String,
    pub selection_score: f64,
    pub parasitic_opportunity: f64,
    pub vulnerability_score: f64,
    pub organism_votes: Vec<OrganismVote>,
    pub emergence_detected: bool,
    pub quantum_enhanced: bool,
    pub cqgs_compliance_score: f64,
    pub selection_time: DateTime<Utc>,
}

/// Individual organism vote for pair selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganismVote {
    pub organism_id: Uuid,
    pub organism_type: String,
    pub vote_score: f64,
    pub confidence: f64,
    pub strategy: String,
}

/// Parasitic pattern detected in market data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParasiticPattern {
    pub pair_id: String,
    pub host_type: HostType,
    pub vulnerability_score: f64,
    pub parasitic_opportunity: f64,
    pub resistance_level: f64,
    pub exploitation_strategy: ExploitationStrategy,
    pub last_successful_parasitism: Option<DateTime<Utc>>,
    pub emergence_patterns: Vec<EmergentBehavior>,
}

/// Types of trading hosts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HostType {
    Whale,
    AlgoTrader,
    MarketMaker,
    RetailSwarm,
    ArbitrageBot,
}

/// Exploitation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExploitationStrategy {
    Shadow,    // Follow large orders
    FrontRun,  // Anticipate predictable moves
    Arbitrage, // Cross-pair opportunities
    Leech,     // Extract from inefficiencies
    Mimic,     // Copy successful patterns
}

/// Emergent behaviors detected
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergentBehavior {
    pub behavior_type: String,
    pub strength: f64,
    pub duration_estimate: u64, // seconds
    pub profit_potential: f64,
}

/// CQGS compliance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CQGSComplianceMetrics {
    pub zero_mock_compliance: f64,   // Must be 1.0
    pub sentinel_validation: f64,    // Sentinel approval score
    pub hyperbolic_optimization: f64, // Topology efficiency
    pub neural_enhancement: f64,     // AI intelligence factor
    pub governance_score: f64,       // Overall governance compliance
}

impl CQGSComplianceMetrics {
    pub fn is_compliant(&self) -> bool {
        self.zero_mock_compliance >= 1.0 &&
        self.sentinel_validation >= 0.8 &&
        self.governance_score >= 0.9
    }
}