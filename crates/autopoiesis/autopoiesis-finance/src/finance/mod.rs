//! Financial Markets Domain
//! 
//! Implementation of autopoietic financial systems integrating:
//! - Bateson-inspired market cognition (market_mind)
//! - Prigogine-based market thermodynamics (dissipative_trading)
//! - Strogatz synchronization for traders (sync_traders)
//! - Grinberg consciousness-driven markets (syntergic_market)
//!
//! This module demonstrates how financial markets exhibit autopoietic properties
//! through self-organizing behavior, consciousness effects, and emergent intelligence.

pub mod market_mind;
pub mod dissipative_trading;
pub mod sync_traders;
pub mod syntergic_market;

// Re-export key types for public API
pub use market_mind::{
    MarketMind,
    MarketCognitionLevel,
    PatternRecognition,
    MarketLearning,
};

pub use dissipative_trading::{
    DissipativeMarket,
    MarketThermodynamics,
    TradingBifurcation,
    MarketEntropy,
};

pub use sync_traders::{
    TraderSynchronization,
    TraderOscillator,
    KuramotoTraders,
    SyncMetrics,
};

pub use syntergic_market::{
    SyntergicMarket,
    MarketConsciousness,
    CollectiveTrading,
    ConsciousnessPrice,
};

use crate::prelude::*;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Financial market symbol
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct Symbol(pub String);

impl Symbol {
    pub fn new(s: &str) -> Self {
        Symbol(s.to_uppercase())
    }
}

impl std::fmt::Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Market price with timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketPrice {
    pub symbol: Symbol,
    pub price: f64,
    pub volume: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub bid: f64,
    pub ask: f64,
}

/// Order in the financial market
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketOrder {
    pub id: uuid::Uuid,
    pub symbol: Symbol,
    pub side: OrderSide,
    pub quantity: f64,
    pub price: Option<f64>, // None for market orders
    pub order_type: OrderType,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub trader_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
}

/// Trade execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub id: uuid::Uuid,
    pub symbol: Symbol,
    pub price: f64,
    pub quantity: f64,
    pub buyer_id: String,
    pub seller_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub market_impact: f64,
}

/// Integrated financial market system
/// Combines all autopoietic financial components
#[derive(Debug)]
pub struct AutopoieticMarket {
    /// Market mind for cognitive pattern recognition
    pub market_mind: MarketMind,
    
    /// Dissipative trading dynamics
    pub dissipative_market: DissipativeMarket,
    
    /// Trader synchronization system
    pub trader_sync: TraderSynchronization,
    
    /// Syntergic consciousness layer
    pub syntergic_market: SyntergicMarket,
    
    /// Active symbols
    pub symbols: Vec<Symbol>,
    
    /// Market state
    pub market_state: MarketState,
    
    /// Integration parameters
    pub integration_strength: f64,
}

/// Complete market state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketState {
    pub prices: HashMap<Symbol, MarketPrice>,
    pub recent_trades: Vec<Trade>,
    pub order_book: HashMap<Symbol, OrderBook>,
    pub market_sentiment: f64,
    pub volatility: f64,
    pub volume_profile: VolumeProfile,
    pub regime: MarketRegime,
}

/// Order book for a symbol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub bids: Vec<(f64, f64)>, // (price, quantity)
    pub asks: Vec<(f64, f64)>,
    pub spread: f64,
    pub depth: f64,
}

/// Volume profile analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeProfile {
    pub total_volume: f64,
    pub volume_weighted_price: f64,
    pub volume_distribution: HashMap<String, f64>, // price level -> volume
    pub concentration: f64,
}

/// Market regime classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketRegime {
    Trending { direction: f64, strength: f64 },
    Ranging { center: f64, bandwidth: f64 },
    Volatile { intensity: f64 },
    Crisis { severity: f64 },
    Syntergic { coherence: f64 },
}

impl AutopoieticMarket {
    /// Create new integrated market system
    pub fn new(symbols: Vec<Symbol>) -> Self {
        let dimensions = (10, 10, 10); // 3D market space
        
        Self {
            market_mind: MarketMind::new(symbols.clone()),
            dissipative_market: DissipativeMarket::new(symbols.clone()),
            trader_sync: TraderSynchronization::new(100), // 100 traders
            syntergic_market: SyntergicMarket::new(dimensions, symbols.clone()),
            symbols: symbols.clone(),
            market_state: MarketState::new(symbols),
            integration_strength: 1.0,
        }
    }
    
    /// Initialize market with coherent autopoietic state
    pub fn initialize_autopoietic_market(&mut self) {
        println!("🏦 Initializing autopoietic financial market...");
        
        // Initialize cognitive layer
        self.market_mind.initialize_market_cognition();
        
        // Initialize thermodynamic layer
        self.dissipative_market.initialize_far_from_equilibrium();
        
        // Initialize synchronization layer
        self.trader_sync.initialize_trader_oscillators();
        
        // Initialize consciousness layer
        self.syntergic_market.initialize_market_consciousness();
        
        println!("✅ Autopoietic market system ready");
    }
    
    /// Process complete market cycle
    pub fn process_market_cycle(&mut self, dt: f64, market_events: Vec<MarketEvent>) {
        // 1. Market mind processes information and learns patterns
        let cognitive_insights = self.market_mind.process_market_information(&self.market_state, dt);
        
        // 2. Dissipative dynamics maintain far-from-equilibrium state
        let thermodynamic_forces = self.dissipative_market.evolve_thermodynamics(dt, &market_events);
        
        // 3. Trader synchronization creates emergent coordination
        let sync_effects = self.trader_sync.update_synchronization(dt, &self.market_state);
        
        // 4. Syntergic consciousness influences market reality
        let consciousness_effects = self.syntergic_market.process_collective_consciousness(
            dt, &cognitive_insights, &sync_effects
        );
        
        // 5. Integrate all effects into unified market evolution
        self.integrate_market_forces(
            cognitive_insights,
            thermodynamic_forces,
            sync_effects,
            consciousness_effects,
            dt
        );
        
        // 6. Update market state
        self.update_market_state(dt);
        
        // 7. Check for emergent autopoietic behavior
        if self.is_autopoietic_behavior_emerged() {
            self.handle_autopoietic_emergence();
        }
    }
    
    /// Integrate all market forces into unified evolution
    fn integrate_market_forces(&mut self, 
                              cognitive: CognitiveInsights,
                              thermodynamic: ThermodynamicForces,
                              synchronization: SyncEffects,
                              consciousness: ConsciousnessEffects,
                              dt: f64) {
        // Weight the different influences
        let cognitive_weight = 0.25;
        let thermodynamic_weight = 0.30;
        let sync_weight = 0.20;
        let consciousness_weight = 0.25;
        
        // Apply integrated forces to market state
        for symbol in &self.symbols.clone() {
            if let Some(price) = self.market_state.prices.get_mut(symbol) {
                let total_force = 
                    cognitive_weight * cognitive.price_influences.get(symbol).unwrap_or(&0.0) +
                    thermodynamic_weight * thermodynamic.energy_flows.get(symbol).unwrap_or(&0.0) +
                    sync_weight * synchronization.coordination_effects.get(symbol).unwrap_or(&0.0) +
                    consciousness_weight * consciousness.reality_modifications.get(symbol).unwrap_or(&0.0);
                
                // Apply force to price with nonlinear dynamics
                let price_change = total_force * dt * (1.0 + 0.1 * (total_force * 10.0).sin());
                price.price *= (1.0 + price_change).max(0.1); // Prevent negative prices
                price.timestamp = chrono::Utc::now();
            }
        }
    }
    
    /// Update complete market state
    fn update_market_state(&mut self, dt: f64) {
        // Update volatility based on recent price movements
        self.market_state.volatility = self.calculate_realized_volatility();
        
        // Update market sentiment from consciousness layer
        self.market_state.market_sentiment = self.syntergic_market.get_collective_sentiment();
        
        // Classify current market regime
        self.market_state.regime = self.classify_market_regime();
        
        // Update volume profile
        self.market_state.volume_profile = self.calculate_volume_profile();
    }
    
    /// Check if autopoietic behavior has emerged
    fn is_autopoietic_behavior_emerged(&self) -> bool {
        let cognitive_active = self.market_mind.is_learning_actively();
        let thermodynamic_stable = self.dissipative_market.is_far_from_equilibrium();
        let sync_achieved = self.trader_sync.get_order_parameter() > 0.5;
        let consciousness_coherent = self.syntergic_market.is_consciousness_coherent();
        
        cognitive_active && thermodynamic_stable && sync_achieved && consciousness_coherent
    }
    
    /// Handle emergence of autopoietic market behavior
    fn handle_autopoietic_emergence(&mut self) {
        println!("🚀 AUTOPOIETIC MARKET BEHAVIOR EMERGED!");
        println!("   - Market demonstrates self-creation and self-maintenance");
        println!("   - Cognitive patterns recognized and learned");
        println!("   - Thermodynamic stability maintained far from equilibrium");
        println!("   - Trader synchronization achieved");
        println!("   - Collective consciousness coherent");
        
        // Enhance integration when autopoiesis emerges
        self.integration_strength = (self.integration_strength * 1.2).min(3.0);
        
        // Could trigger additional behaviors:
        // - Enhanced pattern recognition
        // - Increased market efficiency
        // - Stronger crisis resilience
        // - Better price discovery
    }
    
    /// Calculate realized volatility
    fn calculate_realized_volatility(&self) -> f64 {
        // Simplified volatility calculation
        let returns: Vec<f64> = self.market_state.recent_trades
            .windows(2)
            .map(|trades| (trades[1].price / trades[0].price).ln())
            .collect();
        
        if returns.len() < 2 {
            return 0.1; // Default volatility
        }
        
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;
            
        variance.sqrt() * (252.0_f64).sqrt() // Annualized
    }
    
    /// Classify current market regime
    fn classify_market_regime(&self) -> MarketRegime {
        let volatility = self.market_state.volatility;
        let sentiment = self.market_state.market_sentiment;
        let consciousness_coherence = self.syntergic_market.get_coherence_level();
        
        if consciousness_coherence > 0.8 {
            MarketRegime::Syntergic { coherence: consciousness_coherence }
        } else if volatility > 0.5 {
            MarketRegime::Crisis { severity: volatility }
        } else if sentiment.abs() > 0.7 {
            MarketRegime::Trending { 
                direction: sentiment.signum(), 
                strength: sentiment.abs() 
            }
        } else if volatility < 0.1 {
            MarketRegime::Ranging { 
                center: self.calculate_average_price(), 
                bandwidth: volatility 
            }
        } else {
            MarketRegime::Volatile { intensity: volatility }
        }
    }
    
    /// Calculate volume profile
    fn calculate_volume_profile(&self) -> VolumeProfile {
        let total_volume: f64 = self.market_state.recent_trades
            .iter()
            .map(|t| t.quantity)
            .sum();
            
        let volume_weighted_price = if total_volume > 0.0 {
            self.market_state.recent_trades
                .iter()
                .map(|t| t.price * t.quantity)
                .sum::<f64>() / total_volume
        } else {
            self.calculate_average_price()
        };
        
        VolumeProfile {
            total_volume,
            volume_weighted_price,
            volume_distribution: HashMap::new(), // Simplified
            concentration: self.calculate_volume_concentration(),
        }
    }
    
    /// Get average price across all symbols
    fn calculate_average_price(&self) -> f64 {
        if self.market_state.prices.is_empty() {
            return 100.0; // Default price
        }
        
        self.market_state.prices
            .values()
            .map(|p| p.price)
            .sum::<f64>() / self.market_state.prices.len() as f64
    }
    
    /// Calculate volume concentration metric
    fn calculate_volume_concentration(&self) -> f64 {
        // Herfindahl-Hirschman Index for volume concentration
        let total_volume: f64 = self.market_state.recent_trades
            .iter()
            .map(|t| t.quantity)
            .sum();
            
        if total_volume == 0.0 {
            return 0.0;
        }
        
        self.market_state.recent_trades
            .iter()
            .map(|t| (t.quantity / total_volume).powi(2))
            .sum()
    }
    
    /// Get complete market analytics
    pub fn get_market_analytics(&self) -> MarketAnalytics {
        MarketAnalytics {
            cognitive_level: self.market_mind.get_cognition_level(),
            thermodynamic_health: self.dissipative_market.get_entropy_production(),
            synchronization_strength: self.trader_sync.get_order_parameter(),
            consciousness_coherence: self.syntergic_market.get_coherence_level(),
            autopoietic_score: self.calculate_autopoietic_score(),
            market_state: self.market_state.clone(),
            integration_strength: self.integration_strength,
        }
    }
    
    /// Calculate overall autopoietic score
    fn calculate_autopoietic_score(&self) -> f64 {
        let cognitive = self.market_mind.get_cognition_level();
        let thermodynamic = if self.dissipative_market.is_far_from_equilibrium() { 1.0 } else { 0.0 };
        let synchronization = self.trader_sync.get_order_parameter();
        let consciousness = self.syntergic_market.get_coherence_level();
        
        (cognitive * thermodynamic * synchronization * consciousness).powf(0.25) * self.integration_strength
    }
}

/// Market event that can influence dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketEvent {
    News { content: String, sentiment: f64, impact: f64 },
    RegulationChange { description: String, effect: f64 },
    TechnicalBreakout { symbol: Symbol, direction: f64, strength: f64 },
    LiquidityShock { symbol: Symbol, impact: f64 },
    ConsciousnessShift { coherence_change: f64, narrative: String },
}

/// Complete market analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketAnalytics {
    pub cognitive_level: f64,
    pub thermodynamic_health: f64,
    pub synchronization_strength: f64,
    pub consciousness_coherence: f64,
    pub autopoietic_score: f64,
    pub market_state: MarketState,
    pub integration_strength: f64,
}

/// Placeholder types for integration (defined in respective modules)
#[derive(Debug, Clone)]
pub struct CognitiveInsights {
    pub price_influences: HashMap<Symbol, f64>,
    pub pattern_strength: f64,
    pub learning_rate: f64,
}

#[derive(Debug, Clone)]
pub struct ThermodynamicForces {
    pub energy_flows: HashMap<Symbol, f64>,
    pub entropy_production: f64,
    pub bifurcation_proximity: f64,
}

#[derive(Debug, Clone)]
pub struct SyncEffects {
    pub coordination_effects: HashMap<Symbol, f64>,
    pub order_parameter: f64,
    pub cluster_formation: Vec<Vec<usize>>,
}

#[derive(Debug, Clone)]
pub struct ConsciousnessEffects {
    pub reality_modifications: HashMap<Symbol, f64>,
    pub collective_belief: f64,
    pub narrative_coherence: f64,
}

impl MarketState {
    /// Create new market state
    fn new(symbols: Vec<Symbol>) -> Self {
        let mut prices = HashMap::new();
        let mut order_books = HashMap::new();
        
        for symbol in &symbols {
            prices.insert(symbol.clone(), MarketPrice {
                symbol: symbol.clone(),
                price: 100.0, // Default starting price
                volume: 0.0,
                timestamp: chrono::Utc::now(),
                bid: 99.5,
                ask: 100.5,
            });
            
            order_books.insert(symbol.clone(), OrderBook {
                bids: vec![(99.5, 1000.0), (99.0, 2000.0)],
                asks: vec![(100.5, 1000.0), (101.0, 2000.0)],
                spread: 1.0,
                depth: 3000.0,
            });
        }
        
        Self {
            prices,
            recent_trades: Vec::new(),
            order_book: order_books,
            market_sentiment: 0.0,
            volatility: 0.1,
            volume_profile: VolumeProfile {
                total_volume: 0.0,
                volume_weighted_price: 100.0,
                volume_distribution: HashMap::new(),
                concentration: 0.0,
            },
            regime: MarketRegime::Ranging { center: 100.0, bandwidth: 0.1 },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_autopoietic_market_creation() {
        let symbols = vec![Symbol::new("BTCUSD"), Symbol::new("ETHUSD")];
        let market = AutopoieticMarket::new(symbols.clone());
        
        assert_eq!(market.symbols.len(), 2);
        assert_eq!(market.integration_strength, 1.0);
    }
    
    #[test]
    fn test_market_initialization() {
        let symbols = vec![Symbol::new("BTCUSD")];
        let mut market = AutopoieticMarket::new(symbols);
        
        market.initialize_autopoietic_market();
        // Market should be initialized without panicking
    }
    
    #[test] 
    fn test_symbol_creation() {
        let symbol = Symbol::new("btcusd");
        assert_eq!(symbol.0, "BTCUSD");
        assert_eq!(format!("{}", symbol), "BTCUSD");
    }
    
    #[test]
    fn test_market_state_creation() {
        let symbols = vec![Symbol::new("BTCUSD"), Symbol::new("ETHUSD")];
        let state = MarketState::new(symbols.clone());
        
        assert_eq!(state.prices.len(), 2);
        assert_eq!(state.order_book.len(), 2);
        assert!(state.prices.contains_key(&Symbol::new("BTCUSD")));
        assert!(state.prices.contains_key(&Symbol::new("ETHUSD")));
    }
}