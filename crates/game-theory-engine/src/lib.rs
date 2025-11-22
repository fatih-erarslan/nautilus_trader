// Game Theory Engine with Machiavellian Tactics and Nash Equilibrium Solving
// Copyright (c) 2025 TENGRI Trading Swarm

use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use tracing::{info, warn, error, debug};

pub mod nash_solver;
pub mod machiavellian_tactics;
pub mod coalition_games;
pub mod mechanism_design;
pub mod auction_theory;
pub mod behavioral_models;
pub mod evolutionary_games;
pub mod game_tree;

#[cfg(feature = "python")]
pub mod python_bindings_simple;

pub use nash_solver::*;
pub use machiavellian_tactics::*;
pub use coalition_games::*;
pub use mechanism_design::*;
pub use auction_theory::*;
pub use behavioral_models::*;
pub use evolutionary_games::*;
pub use game_tree::*;

// use market_regime_detector::MarketRegime; // Commented out - no such crate exists

// Define MarketRegime locally for now
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketRegime {
    LowVolatility,
    HighVolatility,
    Trending,
    Sideways,
    Crisis,
}

/// Types of games in financial markets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GameType {
    // Classic game theory
    PrisonersDilemma,
    ChickenGame,
    StagHunt,
    BattleOfSexes,
    MatchingPennies,
    RockPaperScissors,
    
    // Auction games
    EnglishAuction,
    DutchAuction,
    SealedBidAuction,
    VickreyAuction,
    DoubleAuction,
    
    // Trading games
    MarketMaking,
    OrderBookGame,
    LiquidityProvision,
    ArbitrageGame,
    WhaleBattile,
    FlashCrashGame,
    
    // Cooperative games
    CoalitionFormation,
    CostSharing,
    VotingGame,
    Bargaining,
    
    // Evolutionary games
    HawkDove,
    TitForTat,
    Generous,
    Grudger,
    
    // Information games
    SignalingGame,
    ScreeningGame,
    CheapTalk,
    
    // Network games
    DiffusionGame,
    ContagionGame,
    NetworkFormation,
    
    // Mechanism design
    OptimalAuction,
    IncentiveDesign,
    ContractTheory,
}

impl GameType {
    /// Get the typical number of players for this game type
    pub fn typical_player_count(&self) -> (usize, usize) {
        match self {
            GameType::PrisonersDilemma => (2, 2),
            GameType::ChickenGame => (2, 2),
            GameType::DoubleAuction => (2, 1000),
            GameType::OrderBookGame => (10, 10000),
            GameType::CoalitionFormation => (3, 100),
            GameType::WhaleBattile => (2, 10),
            _ => (2, 10),
        }
    }
    
    /// Check if this game type is zero-sum
    pub fn is_zero_sum(&self) -> bool {
        matches!(self, 
            GameType::MatchingPennies |
            GameType::RockPaperScissors |
            GameType::WhaleBattile
        )
    }
    
    /// Check if this game supports coalition formation
    pub fn supports_coalitions(&self) -> bool {
        matches!(self,
            GameType::CoalitionFormation |
            GameType::CostSharing |
            GameType::VotingGame |
            GameType::Bargaining
        )
    }
    
    /// Get the information structure
    pub fn information_structure(&self) -> InformationStructure {
        match self {
            GameType::SealedBidAuction => InformationStructure::Private,
            GameType::EnglishAuction => InformationStructure::Public,
            GameType::SignalingGame => InformationStructure::Asymmetric,
            GameType::OrderBookGame => InformationStructure::Partial,
            _ => InformationStructure::Complete,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InformationStructure {
    Complete,       // All information is public
    Incomplete,     // Some private information
    Perfect,        // All history is observable
    Imperfect,      // Some actions are unobservable
    Private,        // Each player has private information
    Public,         // All information is public
    Asymmetric,     // Different information across players
    Partial,        // Limited observability
}

/// Player types in the trading ecosystem
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlayerType {
    // Individual traders
    RetailTrader,
    DayTrader,
    SwingTrader,
    Scalper,
    
    // Institutional players
    HedgeFund,
    MutualFund,
    PensionFund,
    InsuranceCompany,
    Bank,
    
    // Market makers
    DesignatedMarketMaker,
    AuthorizedParticipant,
    ElectronicMarketMaker,
    
    // High-frequency traders
    HFTArbitrager,
    HFTMarketMaker,
    HFTMomentum,
    
    // Special actors
    Whale,              // Large position holder
    Manipulator,        // Market manipulator
    CentralBank,        // Central bank intervention
    Regulator,          // Regulatory authority
    
    // Algorithmic entities
    AITrader,           // AI-driven trader
    QuantumTrader,      // Quantum-enhanced trader
    SwarmTrader,        // Our swarm system
}

impl PlayerType {
    /// Get the typical capital size for this player type
    pub fn typical_capital(&self) -> (f64, f64) {
        match self {
            PlayerType::RetailTrader => (1000.0, 100_000.0),
            PlayerType::DayTrader => (10_000.0, 1_000_000.0),
            PlayerType::HedgeFund => (100_000_000.0, 10_000_000_000.0),
            PlayerType::Whale => (1_000_000_000.0, 100_000_000_000.0),
            PlayerType::CentralBank => (1_000_000_000_000.0, 10_000_000_000_000.0),
            PlayerType::SwarmTrader => (1_000_000.0, 1_000_000_000.0),
            _ => (1_000_000.0, 100_000_000.0),
        }
    }
    
    /// Get risk tolerance
    pub fn risk_tolerance(&self) -> f64 {
        match self {
            PlayerType::Scalper => 0.1,
            PlayerType::RetailTrader => 0.3,
            PlayerType::PensionFund => 0.2,
            PlayerType::HedgeFund => 0.8,
            PlayerType::Whale => 0.6,
            PlayerType::Manipulator => 0.9,
            PlayerType::SwarmTrader => 0.7,
            _ => 0.5,
        }
    }
    
    /// Get sophistication level
    pub fn sophistication_level(&self) -> SophisticationLevel {
        match self {
            PlayerType::RetailTrader => SophisticationLevel::Basic,
            PlayerType::HFTArbitrager => SophisticationLevel::Expert,
            PlayerType::QuantumTrader => SophisticationLevel::Advanced,
            PlayerType::SwarmTrader => SophisticationLevel::Expert,
            PlayerType::AITrader => SophisticationLevel::Advanced,
            _ => SophisticationLevel::Intermediate,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SophisticationLevel {
    Basic,          // Limited strategy repertoire
    Intermediate,   // Standard strategies
    Advanced,       // Complex strategies
    Expert,         // Cutting-edge strategies
}

/// Game state representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameState {
    pub game_type: GameType,
    pub players: Vec<Player>,
    pub market_context: MarketContext,
    pub information_sets: HashMap<String, InformationSet>,
    pub action_history: Vec<GameAction>,
    pub current_round: u32,
    pub payoff_matrix: Option<PayoffMatrix>,
    pub nash_equilibria: Vec<NashEquilibrium>,
    pub nash_equilibrium_found: bool,
    pub dominant_strategies: HashMap<String, Strategy>,
    pub cooperation_level: f64,
    pub competition_intensity: f64,
}

/// Player in the game
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Player {
    pub id: String,
    pub player_type: PlayerType,
    pub capital: f64,
    pub risk_tolerance: f64,
    pub sophistication: SophisticationLevel,
    pub current_position: f64,
    pub strategy: Strategy,
    pub private_information: HashMap<String, f64>,
    pub reputation: f64,
    pub cooperation_history: Vec<bool>,
}

/// Market context for the game
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketContext {
    pub regime: MarketRegime,
    pub volatility: f64,
    pub liquidity: f64,
    pub volume: f64,
    pub spread: f64,
    pub market_impact: f64,
    pub information_asymmetry: f64,
    pub regulatory_environment: RegulatoryEnvironment,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulatoryEnvironment {
    pub short_selling_allowed: bool,
    pub position_limits: Option<f64>,
    pub circuit_breakers: bool,
    pub market_making_obligations: bool,
    pub transparency_requirements: TransparencyLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransparencyLevel {
    Full,           // All information public
    Partial,        // Some information delayed
    Limited,        // Minimal disclosure
    Dark,           // No transparency
}

/// Information set for a player
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationSet {
    pub player_id: String,
    pub observable_actions: Vec<GameAction>,
    pub private_signals: HashMap<String, f64>,
    pub beliefs: HashMap<String, f64>,
    pub uncertainty_level: f64,
}

/// Action in the game
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameAction {
    pub player_id: String,
    pub action_type: ActionType,
    pub parameters: HashMap<String, f64>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub visibility: ActionVisibility,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionType {
    // Trading actions
    Buy,
    Sell,
    Hold,
    Cancel,
    Modify,
    
    // Market making actions
    Quote,
    Spread,
    Inventory,
    
    // Strategic actions
    Signal,
    Bluff,
    Cooperate,
    Defect,
    Punish,
    Forgive,
    
    // Information actions
    Research,
    Monitor,
    Share,
    Hide,
    
    // Manipulation actions
    Spoof,
    Layer,
    Wash,
    Paint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionVisibility {
    Public,         // Visible to all players
    Private,        // Visible only to the player
    Delayed,        // Visible after delay
    Noisy,          // Visible with noise
    Partial,        // Partially observable
}

/// Strategy representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Strategy {
    pub name: String,
    pub strategy_type: StrategyType,
    pub parameters: HashMap<String, f64>,
    pub conditions: Vec<StrategyCondition>,
    pub actions: Vec<StrategyAction>,
    pub expected_payoff: f64,
    pub risk_level: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StrategyType {
    // Pure strategies
    Pure,
    
    // Mixed strategies
    Mixed,
    
    // Behavioral strategies
    TitForTat,
    Generous,
    Grudger,
    Random,
    
    // Sophisticated strategies
    Machiavellian,
    GameTheoretic,
    Evolutionary,
    Adaptive,
    
    // Learning strategies
    Imitation,
    Reinforcement,
    Belief,
    
    // Cooperative strategies
    Reciprocal,
    Altruistic,
    Conditional,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyCondition {
    pub condition_type: ConditionType,
    pub threshold: f64,
    pub variables: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConditionType {
    MarketRegime,
    Price,
    Volume,
    Volatility,
    Position,
    Profit,
    Loss,
    Time,
    OpponentAction,
    Information,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyAction {
    pub action: ActionType,
    pub probability: f64,
    pub parameters: HashMap<String, f64>,
}

/// Payoff matrix for normal form games
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PayoffMatrix {
    pub players: Vec<String>,
    pub strategies: HashMap<String, Vec<String>>,
    pub payoffs: HashMap<String, f64>,
    pub dimension: Vec<usize>,
}

/// Nash equilibrium solution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NashEquilibrium {
    pub equilibrium_type: EquilibriumType,
    pub strategies: HashMap<String, MixedStrategy>,
    pub payoffs: HashMap<String, f64>,
    pub stability: f64,
    pub uniqueness: bool,
    pub efficiency: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EquilibriumType {
    Pure,           // Pure strategy Nash equilibrium
    Mixed,          // Mixed strategy Nash equilibrium
    Correlated,     // Correlated equilibrium
    Evolutionary,   // Evolutionarily stable strategy
    Trembling,      // Trembling hand perfect
    Subgame,        // Subgame perfect
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedStrategy {
    pub pure_strategies: Vec<String>,
    pub probabilities: Vec<f64>,
    pub expected_payoff: f64,
}

/// Configuration for the game theory engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameTheoryConfig {
    pub enable_deception: bool,
    pub machiavellian_intensity: f64,
    pub cooperation_threshold: f64,
    pub adversarial_modeling: bool,
    pub nash_solver_tolerance: f64,
    pub max_iterations: u32,
    pub learning_rate: f64,
    pub reputation_decay: f64,
}

/// Core trait for game theory engines
#[async_trait]
pub trait GameTheoryEngine: Send + Sync {
    async fn analyze_market_game(&mut self, 
                                market_data: &MarketData, 
                                regime: &MarketRegime,
                                warnings: &[EarlyWarning]) -> Result<GameState>;
    
    async fn solve_nash_equilibrium(&self, game_state: &GameState) -> Result<Vec<NashEquilibrium>>;
    
    async fn generate_machiavellian_strategy(&self, 
                                           game_state: &GameState,
                                           player_id: &str) -> Result<Strategy>;
    
    async fn predict_opponent_actions(&self, 
                                    game_state: &GameState,
                                    opponent_id: &str) -> Result<Vec<(ActionType, f64)>>;
    
    async fn update_beliefs(&mut self, game_state: &GameState, new_action: &GameAction) -> Result<()>;
}

// Placeholder types for external dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub symbol: String,
    pub price: f64,
    pub volume: f64,
    pub volatility: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyWarning {
    pub warning_type: String,
    pub severity: f64,
    pub description: String,
}

/// Error types for game theory operations
#[derive(thiserror::Error, Debug)]
pub enum GameTheoryError {
    #[error("Nash equilibrium not found: {0}")]
    NashEquilibriumNotFound(String),
    
    #[error("Invalid game configuration: {0}")]
    InvalidGameConfiguration(String),
    
    #[error("Player strategy error: {0}")]
    PlayerStrategyError(String),
    
    #[error("Payoff calculation error: {0}")]
    PayoffCalculationError(String),
    
    #[error("Information set error: {0}")]
    InformationSetError(String),
    
    #[error("Mechanism design error: {0}")]
    MechanismDesignError(String),
}

pub type GameTheoryResult<T> = Result<T, GameTheoryError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_game_type_properties() {
        assert!(GameType::MatchingPennies.is_zero_sum());
        assert!(!GameType::PrisonersDilemma.is_zero_sum());
        assert!(GameType::CoalitionFormation.supports_coalitions());
        assert_eq!(GameType::PrisonersDilemma.typical_player_count(), (2, 2));
    }

    #[test]
    fn test_player_type_properties() {
        let whale_capital = PlayerType::Whale.typical_capital();
        let retail_capital = PlayerType::RetailTrader.typical_capital();
        assert!(whale_capital.0 > retail_capital.1);
        
        assert_eq!(PlayerType::HFTArbitrager.sophistication_level(), SophisticationLevel::Expert);
        assert_eq!(PlayerType::RetailTrader.sophistication_level(), SophisticationLevel::Basic);
    }

    #[test]
    fn test_strategy_serialization() {
        let strategy = Strategy {
            name: "Test Strategy".to_string(),
            strategy_type: StrategyType::Machiavellian,
            parameters: HashMap::new(),
            conditions: vec![],
            actions: vec![],
            expected_payoff: 0.75,
            risk_level: 0.3,
        };

        let serialized = serde_json::to_string(&strategy).expect("Serialization failed");
        let deserialized: Strategy = serde_json::from_str(&serialized).expect("Deserialization failed");
        
        assert_eq!(strategy.name, deserialized.name);
        assert_eq!(strategy.strategy_type, deserialized.strategy_type);
    }

    #[test]
    fn test_information_structure() {
        assert_eq!(GameType::SealedBidAuction.information_structure(), InformationStructure::Private);
        assert_eq!(GameType::EnglishAuction.information_structure(), InformationStructure::Public);
    }

    #[tokio::test]
    async fn test_game_state_creation() {
        let game_state = GameState {
            game_type: GameType::PrisonersDilemma,
            players: vec![],
            market_context: MarketContext {
                regime: MarketRegime::LowVolatility,
                volatility: 0.2,
                liquidity: 1000000.0,
                volume: 500000.0,
                spread: 0.01,
                market_impact: 0.001,
                information_asymmetry: 0.3,
                regulatory_environment: RegulatoryEnvironment {
                    short_selling_allowed: true,
                    position_limits: None,
                    circuit_breakers: true,
                    market_making_obligations: false,
                    transparency_requirements: TransparencyLevel::Partial,
                },
            },
            information_sets: HashMap::new(),
            action_history: vec![],
            current_round: 1,
            payoff_matrix: None,
            nash_equilibria: vec![],
            nash_equilibrium_found: false,
            dominant_strategies: HashMap::new(),
            cooperation_level: 0.5,
            competition_intensity: 0.7,
        };

        assert_eq!(game_state.game_type, GameType::PrisonersDilemma);
        assert_eq!(game_state.current_round, 1);
        assert!(!game_state.nash_equilibrium_found);
    }
}