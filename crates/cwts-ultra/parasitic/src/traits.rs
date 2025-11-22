//! Core trait definitions for the parasitic pairlist system
//!
//! This module defines the fundamental traits and structures that enable
//! biomimetic organisms to parasitically exploit trading pair opportunities.
//! The system leverages quantum-enhanced memory and evolutionary fitness scoring.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Unique identifier for trading pairs
pub type PairId = String;

/// Unique identifier for organism instances  
pub type OrganismId = String;

/// Host identifier (whale addresses, algorithmic traders, etc.)
pub type HostId = String;

/// Timestamp in milliseconds since epoch
pub type Timestamp = u64;

/// Core trait for all parasitic organisms
///
/// Each organism implements a different parasitic strategy inspired by nature:
/// - Cuckoo: Brood parasitism (nest detection, egg laying near whales)  
/// - Wasp: Parasitoid behavior (lifecycle tracking, injection)
/// - Cordyceps: Mind control (exploit predictable algorithms)
/// - Mycelial: Network parasitism (correlation exploitation)
/// - Octopus: Adaptive camouflage (stealth strategies)
/// - Anglerfish: Lure-based attraction (honey pot strategies)
/// - Komodo: Persistent tracking (long-term wound exploitation)
/// - Tardigrade: Extreme survival (cryptobiosis during market chaos)
/// - Electric Eel: Shock tactics (market disruption for hidden liquidity)
/// - Platypus: Electroreception (subtle signal detection)
#[async_trait::async_trait]
pub trait ParasiticOrganism: Send + Sync {
    /// Unique organism type identifier
    fn organism_type(&self) -> OrganismType;

    /// Select optimal host from available trading pairs
    async fn select_host(&self, candidates: &[TradingPair]) -> Vec<HostTarget>;

    /// Infiltrate the selected host environment  
    async fn infiltrate(&mut self, host: &HostTarget)
        -> Result<InfiltrationResult, ParasiticError>;

    /// Extract value from the parasitic relationship
    async fn extract_value(
        &mut self,
        infiltration: &InfiltrationResult,
    ) -> Result<ExtractionResult, ParasiticError>;

    /// Adapt to host resistance and environmental changes
    async fn adapt_strategy(&mut self, feedback: &EvolutionaryFeedback) -> AdaptationResult;

    /// Calculate evolutionary fitness score
    fn fitness_score(&self) -> f64;

    /// Enter dormant state when conditions are unfavorable
    async fn enter_cryptobiosis(&mut self, trigger: &ThreatLevel) -> CryptobiosisState;

    /// Revive from cryptobiosis when conditions improve
    async fn revive(&mut self, conditions: &MarketConditions) -> RevivalResult;

    /// Detect if the organism has been discovered by hosts
    fn detection_risk(&self) -> f64;

    /// Generate camouflage patterns to avoid detection
    async fn generate_camouflage(&mut self, threat: &ThreatLevel) -> CamouflagePattern;
}

/// Trait for trading pairs that can be parasitized
pub trait HostPair: Send + Sync {
    /// Get pair identifier
    fn pair_id(&self) -> &PairId;

    /// Calculate vulnerability to parasitic exploitation
    fn vulnerability_score(&self) -> f64;

    /// Detect presence of exploitable hosts (whales, algorithms, etc.)
    fn detect_hosts(&self) -> Vec<HostSignature>;

    /// Get current market depth and liquidity
    fn market_depth(&self) -> MarketDepth;

    /// Check if the pair is suitable for specific organism type
    fn is_suitable_for(&self, organism: OrganismType) -> bool;

    /// Get historical parasitic success rate for this pair
    fn parasitic_history(&self) -> ParasiticHistory;

    /// Calculate resistance level to parasitic exploitation
    fn resistance_level(&self) -> f64;

    /// Get order flow patterns
    fn order_flow_patterns(&self) -> Vec<OrderFlowPattern>;

    /// Detect algorithmic trading patterns
    fn algorithmic_patterns(&self) -> Vec<AlgorithmicSignature>;

    /// Get correlation network with other pairs
    fn correlation_network(&self) -> CorrelationNetwork;
}

/// Core organism types based on biological parasites
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrganismType {
    /// Cuckoo bird - Brood parasite that exploits whale "nests"
    Cuckoo,
    /// Parasitoid wasp - Tracks host lifecycle and injects trackers
    Wasp,
    /// Cordyceps fungus - Mind controls predictable algorithms
    Cordyceps,
    /// Mycelial network - Exploits correlation networks
    Mycelial,
    /// Octopus - Adaptive camouflage and stealth
    Octopus,
    /// Anglerfish - Lure-based attraction strategies  
    Anglerfish,
    /// Komodo dragon - Persistent wound tracking
    Komodo,
    /// Tardigrade - Extreme survival in harsh conditions
    Tardigrade,
    /// Electric eel - Shock tactics for hidden liquidity
    ElectricEel,
    /// Platypus - Electroreception of subtle signals
    Platypus,
}

/// Trading pair representation with parasitic context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingPair {
    pub id: PairId,
    pub base_asset: String,
    pub quote_asset: String,
    pub volume_24h: f64,
    pub price: f64,
    pub best_bid: f64,
    pub best_ask: f64,
    pub spread: f64,
    pub market_depth: MarketDepth,
    pub order_history: Vec<OrderEvent>,
    pub volatility: f64,
    pub liquidity_score: f64,
    pub whale_activity: WhaleActivity,
    pub algorithmic_signatures: Vec<AlgorithmicSignature>,
    pub correlation_coefficients: HashMap<PairId, f64>,
    pub parasitic_metadata: ParasiticMetadata,
}

/// Host target for parasitic exploitation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostTarget {
    pub pair_id: PairId,
    pub host_type: HostType,
    pub host_id: HostId,
    pub vulnerability_score: f64,
    pub opportunity_value: f64,
    pub resistance_level: f64,
    pub detection_risk: f64,
    pub exploitation_window: ExploitationWindow,
    pub entry_points: Vec<EntryPoint>,
    pub camouflage_requirements: CamouflageRequirements,
}

/// Types of hosts that can be parasitized
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HostType {
    /// Large traders with predictable patterns
    Whale,
    /// Algorithmic trading systems
    AlgorithmicTrader,
    /// Market making algorithms
    MarketMaker,
    /// High-frequency trading systems
    HFTSystem,
    /// Arbitrage bots
    ArbitrageBot,
    /// Liquidity providers
    LiquidityProvider,
    /// Retail trader clusters
    RetailCluster,
}

/// Result of infiltration attempt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfiltrationResult {
    pub success: bool,
    pub organism_id: OrganismId,
    pub host_target: HostTarget,
    pub infiltration_depth: f64,
    pub stealth_level: f64,
    pub resource_access: Vec<ResourceAccess>,
    pub monitoring_points: Vec<MonitoringPoint>,
    pub extraction_opportunities: Vec<ExtractionOpportunity>,
    pub threat_level: ThreatLevel,
    pub timestamp: Timestamp,
}

/// Result of value extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResult {
    pub profit: f64,
    pub efficiency: f64,
    pub risk_taken: f64,
    pub host_damage: f64,
    pub detection_events: Vec<DetectionEvent>,
    pub adaptation_triggers: Vec<AdaptationTrigger>,
    pub timestamp: Timestamp,
}

/// Evolutionary feedback for organism adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionaryFeedback {
    pub success_rate: f64,
    pub profit_efficiency: f64,
    pub detection_frequency: f64,
    pub host_resistance_evolution: f64,
    pub environmental_pressure: f64,
    pub competitive_pressure: f64,
    pub resource_availability: f64,
    pub learning_signals: Vec<LearningSignal>,
}

/// Adaptation result after evolutionary pressure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationResult {
    pub strategy_mutations: Vec<StrategyMutation>,
    pub new_capabilities: Vec<Capability>,
    pub obsolete_behaviors: Vec<Behavior>,
    pub fitness_change: f64,
    pub energy_cost: f64,
    pub adaptation_time: u64,
}

/// Market depth information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDepth {
    pub bids: Vec<OrderLevel>,
    pub asks: Vec<OrderLevel>,
    pub total_bid_volume: f64,
    pub total_ask_volume: f64,
    pub depth_score: f64,
    pub liquidity_gaps: Vec<LiquidityGap>,
}

/// Order level in market depth
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderLevel {
    pub price: f64,
    pub volume: f64,
    pub order_count: u32,
    pub whale_indicator: bool,
    pub algorithmic_signature: Option<AlgorithmicSignature>,
}

/// Whale activity detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleActivity {
    pub whale_addresses: Vec<HostId>,
    pub activity_score: f64,
    pub movement_patterns: Vec<MovementPattern>,
    pub nest_locations: Vec<NestLocation>,
    pub vulnerability_windows: Vec<VulnerabilityWindow>,
}

/// Algorithmic signature detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmicSignature {
    pub algorithm_type: AlgorithmType,
    pub predictability_score: f64,
    pub pattern_strength: f64,
    pub execution_timing: Vec<ExecutionTiming>,
    pub control_points: Vec<ControlPoint>,
    pub exploitation_vectors: Vec<ExploitationVector>,
}

/// Types of detected algorithms
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlgorithmType {
    TWAP,
    VWAP,
    Iceberg,
    Stop,
    Momentum,
    MeanReversion,
    Arbitrage,
    MarketMaking,
    Unknown,
}

/// Parasitic metadata for pairs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParasiticMetadata {
    pub last_parasitism: Option<Timestamp>,
    pub success_history: Vec<ParasiticSuccess>,
    pub resistance_evolution: f64,
    pub organism_preferences: HashMap<OrganismType, f64>,
    pub vulnerability_trends: Vec<VulnerabilityTrend>,
    pub camouflage_effectiveness: HashMap<CamouflageType, f64>,
}

/// Historical parasitic success data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParasiticHistory {
    pub total_attempts: u64,
    pub successful_parasitism: u64,
    pub average_profit: f64,
    pub detection_rate: f64,
    pub host_adaptation_rate: f64,
    pub organism_rankings: HashMap<OrganismType, f64>,
}

/// Cryptobiosis state for extreme survival
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptobiosisState {
    pub is_dormant: bool,
    pub trigger_reason: String,
    pub suspended_activities: Vec<Activity>,
    pub revival_conditions: Vec<RevivalCondition>,
    pub energy_conservation: f64,
    pub duration_estimate: u64,
}

/// Market conditions for revival assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConditions {
    pub volatility: f64,
    pub liquidity: f64,
    pub threat_level: ThreatLevel,
    pub opportunity_density: f64,
    pub competition_intensity: f64,
    pub regulatory_pressure: f64,
}

/// Threat level assessment
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ThreatLevel {
    None,
    Low,
    Medium,
    High,
    Critical,
    Extinction,
}

/// Camouflage pattern for stealth
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CamouflagePattern {
    pub pattern_type: CamouflageType,
    pub effectiveness: f64,
    pub energy_cost: f64,
    pub duration: u64,
    pub detection_resistance: f64,
    pub behavioral_modifications: Vec<BehaviorModification>,
}

/// Types of camouflage
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CamouflageType {
    Mimicry,    // Mimic other traders
    Background, // Blend with market noise
    Disruption, // Break up recognizable patterns
    Motion,     // Dynamic movement patterns
    Counter,    // Counter-surveillance
}

/// Evolutionary fitness scoring system
pub trait FitnessScorer: Send + Sync {
    /// Calculate overall fitness score
    fn calculate_fitness(
        &self,
        organism: &dyn ParasiticOrganism,
        history: &ParasiticHistory,
    ) -> f64;

    /// Score survival ability  
    fn survival_score(&self, adaptations: &[AdaptationResult]) -> f64;

    /// Score reproduction success (strategy replication)
    fn reproduction_score(&self, success_rate: f64, profit_efficiency: f64) -> f64;

    /// Score resource extraction efficiency
    fn extraction_efficiency_score(&self, extractions: &[ExtractionResult]) -> f64;

    /// Score stealth and camouflage effectiveness
    fn stealth_score(&self, detection_events: &[DetectionEvent]) -> f64;

    /// Score adaptation speed
    fn adaptation_speed_score(&self, adaptations: &[AdaptationResult]) -> f64;
}

/// Registry system for managing organism instances
pub trait OrganismRegistry: Send + Sync {
    /// Register a new organism instance
    async fn register_organism(&mut self, organism: Box<dyn ParasiticOrganism>) -> OrganismId;

    /// Get organism by ID
    async fn get_organism(&self, id: &OrganismId) -> Option<Arc<RwLock<dyn ParasiticOrganism>>>;

    /// Get all organisms of a specific type
    async fn get_organisms_by_type(
        &self,
        organism_type: OrganismType,
    ) -> Vec<Arc<RwLock<dyn ParasiticOrganism>>>;

    /// Remove organism (extinction)
    async fn remove_organism(&mut self, id: &OrganismId) -> bool;

    /// Get organism performance metrics
    async fn get_organism_metrics(&self, id: &OrganismId) -> Option<OrganismMetrics>;

    /// Spawn new organism with mutations
    async fn spawn_mutated_organism(
        &mut self,
        parent_id: &OrganismId,
        mutations: &[StrategyMutation],
    ) -> Result<OrganismId, ParasiticError>;

    /// Get population statistics
    async fn get_population_stats(&self) -> PopulationStatistics;

    /// Evolution pressure selection
    async fn evolutionary_selection(&mut self, pressure: &EvolutionaryPressure) -> Vec<OrganismId>;
}

// Supporting types

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganismMetrics {
    pub organism_id: OrganismId,
    pub organism_type: OrganismType,
    pub fitness_score: f64,
    pub total_extractions: u64,
    pub total_profit: f64,
    pub average_efficiency: f64,
    pub detection_rate: f64,
    pub adaptation_count: u32,
    pub cryptobiosis_events: u32,
    pub active_time_percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationStatistics {
    pub total_organisms: u64,
    pub organisms_by_type: HashMap<OrganismType, u64>,
    pub average_fitness: f64,
    pub extinction_rate: f64,
    pub adaptation_rate: f64,
    pub diversity_index: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionaryPressure {
    pub selection_intensity: f64,
    pub mutation_rate: f64,
    pub environmental_stress: f64,
    pub resource_scarcity: f64,
    pub competitive_pressure: f64,
}

/// Error types for parasitic operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParasiticError {
    InfiltrationFailed(String),
    ExtractionFailed(String),
    DetectionEvent(DetectionEvent),
    HostResistance(String),
    ResourceExhaustion(String),
    AdaptationFailure(String),
    CryptobiosisRequired(ThreatLevel),
    EvolutionaryFailure(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionEvent {
    pub timestamp: Timestamp,
    pub host_id: HostId,
    pub detection_method: String,
    pub confidence: f64,
    pub countermeasures_deployed: Vec<String>,
}

// Additional supporting structures (trimmed for brevity)

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExploitationWindow {
    pub start_time: Timestamp,
    pub end_time: Timestamp,
    pub confidence: f64,
    pub opportunity_value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntryPoint {
    pub price_level: f64,
    pub entry_method: String,
    pub stealth_requirement: f64,
    pub success_probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CamouflageRequirements {
    pub min_stealth_level: f64,
    pub required_patterns: Vec<CamouflageType>,
    pub energy_budget: f64,
    pub duration_requirement: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAccess {
    pub resource_type: String,
    pub access_level: f64,
    pub extraction_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringPoint {
    pub location: String,
    pub sensitivity: f64,
    pub data_quality: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionOpportunity {
    pub opportunity_type: String,
    pub value_estimate: f64,
    pub risk_level: f64,
    pub time_window: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationTrigger {
    pub trigger_type: String,
    pub intensity: f64,
    pub required_response: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningSignal {
    pub signal_type: String,
    pub strength: f64,
    pub data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyMutation {
    pub mutation_type: String,
    pub parameters: HashMap<String, f64>,
    pub expected_impact: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capability {
    pub name: String,
    pub effectiveness: f64,
    pub energy_cost: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Behavior {
    pub name: String,
    pub frequency: f64,
    pub effectiveness: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostSignature {
    pub host_id: HostId,
    pub host_type: HostType,
    pub strength: f64,
    pub patterns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderFlowPattern {
    pub pattern_type: String,
    pub frequency: f64,
    pub predictability: f64,
    pub exploitation_potential: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationNetwork {
    pub correlations: HashMap<PairId, f64>,
    pub network_depth: u32,
    pub hub_strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderEvent {
    pub timestamp: Timestamp,
    pub order_type: String,
    pub price: f64,
    pub volume: f64,
    pub side: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityGap {
    pub price_range: (f64, f64),
    pub gap_size: f64,
    pub exploitation_value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MovementPattern {
    pub pattern_type: String,
    pub frequency: f64,
    pub amplitude: f64,
    pub predictability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NestLocation {
    pub price_level: f64,
    pub size: f64,
    pub vulnerability: f64,
    pub activity_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulnerabilityWindow {
    pub start_time: Timestamp,
    pub duration: u64,
    pub vulnerability_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTiming {
    pub expected_time: Timestamp,
    pub confidence: f64,
    pub trigger_conditions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlPoint {
    pub price: f64,
    pub timing: Timestamp,
    pub control_strength: f64,
    pub exploitation_method: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExploitationVector {
    pub vector_type: String,
    pub effectiveness: f64,
    pub risk_level: f64,
    pub resource_requirement: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParasiticSuccess {
    pub timestamp: Timestamp,
    pub organism_type: OrganismType,
    pub profit: f64,
    pub efficiency: f64,
    pub stealth_maintained: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulnerabilityTrend {
    pub timestamp: Timestamp,
    pub vulnerability_score: f64,
    pub trend_direction: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Activity {
    pub activity_type: String,
    pub energy_consumption: f64,
    pub importance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevivalCondition {
    pub condition_type: String,
    pub threshold: f64,
    pub current_value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevivalResult {
    pub success: bool,
    pub energy_recovered: f64,
    pub capabilities_restored: Vec<String>,
    pub time_to_full_activity: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorModification {
    pub behavior_name: String,
    pub modification_type: String,
    pub intensity: f64,
}

impl Default for ThreatLevel {
    fn default() -> Self {
        ThreatLevel::None
    }
}

impl OrganismType {
    /// Get the natural parasitic strategy for this organism type
    pub fn default_strategy(&self) -> String {
        match self {
            OrganismType::Cuckoo => "brood_parasitism".to_string(),
            OrganismType::Wasp => "parasitoid_injection".to_string(),
            OrganismType::Cordyceps => "mind_control".to_string(),
            OrganismType::Mycelial => "network_exploitation".to_string(),
            OrganismType::Octopus => "adaptive_camouflage".to_string(),
            OrganismType::Anglerfish => "lure_attraction".to_string(),
            OrganismType::Komodo => "persistent_tracking".to_string(),
            OrganismType::Tardigrade => "extreme_survival".to_string(),
            OrganismType::ElectricEel => "shock_disruption".to_string(),
            OrganismType::Platypus => "electroreception".to_string(),
        }
    }

    /// Get the preferred host types for this organism
    pub fn preferred_hosts(&self) -> Vec<HostType> {
        match self {
            OrganismType::Cuckoo => vec![HostType::Whale, HostType::LiquidityProvider],
            OrganismType::Wasp => vec![HostType::AlgorithmicTrader, HostType::HFTSystem],
            OrganismType::Cordyceps => vec![HostType::AlgorithmicTrader, HostType::ArbitrageBot],
            OrganismType::Mycelial => vec![HostType::MarketMaker, HostType::LiquidityProvider],
            OrganismType::Octopus => vec![HostType::HFTSystem, HostType::AlgorithmicTrader],
            OrganismType::Anglerfish => vec![HostType::RetailCluster, HostType::AlgorithmicTrader],
            OrganismType::Komodo => vec![HostType::Whale, HostType::MarketMaker],
            OrganismType::Tardigrade => vec![], // Can parasitize any host type
            OrganismType::ElectricEel => vec![HostType::HFTSystem, HostType::MarketMaker],
            OrganismType::Platypus => vec![HostType::AlgorithmicTrader, HostType::HFTSystem],
        }
    }
}

impl ThreatLevel {
    /// Check if threat level requires cryptobiosis
    pub fn requires_cryptobiosis(&self) -> bool {
        matches!(self, ThreatLevel::Critical | ThreatLevel::Extinction)
    }

    /// Get numerical threat value for calculations
    pub fn as_f64(&self) -> f64 {
        match self {
            ThreatLevel::None => 0.0,
            ThreatLevel::Low => 0.2,
            ThreatLevel::Medium => 0.4,
            ThreatLevel::High => 0.6,
            ThreatLevel::Critical => 0.8,
            ThreatLevel::Extinction => 1.0,
        }
    }
}

impl TradingPair {
    /// Check if pair is suitable for parasitic exploitation
    pub fn is_parasitable(&self) -> bool {
        self.liquidity_score > 0.5
            && self.volatility > 0.01
            && !self.whale_activity.whale_addresses.is_empty()
    }

    /// Get the most promising organism types for this pair
    pub fn optimal_organism_types(&self) -> Vec<OrganismType> {
        let mut types = Vec::new();

        if !self.whale_activity.whale_addresses.is_empty() {
            types.push(OrganismType::Cuckoo);
            types.push(OrganismType::Komodo);
        }

        if !self.algorithmic_signatures.is_empty() {
            types.push(OrganismType::Wasp);
            types.push(OrganismType::Cordyceps);
            types.push(OrganismType::Platypus);
        }

        if self.correlation_coefficients.len() > 5 {
            types.push(OrganismType::Mycelial);
        }

        if self.volatility > 0.05 {
            types.push(OrganismType::ElectricEel);
        }

        // Always consider adaptive organisms
        types.push(OrganismType::Octopus);
        types.push(OrganismType::Tardigrade);

        types
    }
}
