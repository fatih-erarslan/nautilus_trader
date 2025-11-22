//! Test data generators and fixtures for autopoiesis testing
//! Provides realistic test data for various system components

use autopoiesis::core::*;
use autopoiesis::consciousness::*; 
use autopoiesis::emergence::*;
use autopoiesis::domains::finance::*;
use rust_decimal::Decimal;
use std::collections::{HashMap, VecDeque};
use rand::{Rng, SeedableRng};
use rand_distr::{Normal, Distribution};
use chrono::{DateTime, Utc, Duration as ChronoDuration};

/// Seeded random number generator for reproducible tests
pub struct SeededRng {
    rng: rand::rngs::StdRng,
}

impl SeededRng {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }
    
    pub fn gen_range<T, R>(&mut self, range: R) -> T
    where
        T: rand::distributions::uniform::SampleUniform,
        R: rand::distributions::uniform::SampleRange<T>,
    {
        self.rng.gen_range(range)
    }
    
    pub fn gen<T>(&mut self) -> T
    where
        rand::distributions::Standard: rand::distributions::Distribution<T>,
    {
        self.rng.gen()
    }
}

/// Market data generator for financial testing
pub struct MarketDataGenerator {
    rng: SeededRng,
    base_prices: HashMap<String, f64>,
    volatilities: HashMap<String, f64>,
    correlations: HashMap<(String, String), f64>,
}

impl MarketDataGenerator {
    pub fn new(seed: u64) -> Self {
        let mut generator = Self {
            rng: SeededRng::new(seed),
            base_prices: HashMap::new(),
            volatilities: HashMap::new(),
            correlations: HashMap::new(),
        };
        
        // Initialize with common crypto pairs
        generator.add_asset("BTC", 45000.0, 0.04); // 4% daily volatility
        generator.add_asset("ETH", 3000.0, 0.05);  // 5% daily volatility
        generator.add_asset("SOL", 100.0, 0.08);   // 8% daily volatility
        generator.add_asset("ADA", 0.50, 0.06);    // 6% daily volatility
        generator.add_asset("DOT", 25.0, 0.07);    // 7% daily volatility
        
        // Add some correlations
        generator.add_correlation("BTC", "ETH", 0.7);
        generator.add_correlation("ETH", "SOL", 0.6);
        generator.add_correlation("SOL", "ADA", 0.4);
        
        generator
    }
    
    pub fn add_asset(&mut self, symbol: &str, base_price: f64, volatility: f64) {
        self.base_prices.insert(symbol.to_string(), base_price);
        self.volatilities.insert(symbol.to_string(), volatility);
    }
    
    pub fn add_correlation(&mut self, symbol1: &str, symbol2: &str, correlation: f64) {
        self.correlations.insert((symbol1.to_string(), symbol2.to_string()), correlation);
        self.correlations.insert((symbol2.to_string(), symbol1.to_string()), correlation);
    }
    
    /// Generate realistic price series using geometric Brownian motion
    pub fn generate_price_series(&mut self, symbol: &str, steps: usize, dt: f64) -> Vec<f64> {
        let base_price = self.base_prices.get(symbol).unwrap_or(&1000.0);
        let volatility = self.volatilities.get(symbol).unwrap_or(&0.02);
        
        let mut prices = vec![*base_price];
        let drift = 0.0001; // Small positive drift (0.01% per step)
        
        let normal = Normal::new(0.0, 1.0).unwrap();
        
        for _ in 1..steps {
            let last_price = prices.last().unwrap();
            let random_shock = normal.sample(&mut self.rng.rng);
            
            // Geometric Brownian Motion: dS = S * (μdt + σdW)
            let price_change = last_price * (drift * dt + volatility * (dt.sqrt()) * random_shock);
            let new_price = last_price + price_change;
            
            // Prevent negative prices
            prices.push(new_price.max(0.01));
        }
        
        prices
    }
    
    /// Generate correlated price series for multiple assets
    pub fn generate_correlated_series(&mut self, symbols: &[&str], steps: usize, dt: f64) -> HashMap<String, Vec<f64>> {
        let mut series = HashMap::new();
        let mut shocks = HashMap::new();
        
        let normal = Normal::new(0.0, 1.0).unwrap();
        
        // Generate independent shocks for each asset
        for symbol in symbols {
            let mut asset_shocks = Vec::new();
            for _ in 0..steps {
                asset_shocks.push(normal.sample(&mut self.rng.rng));
            }
            shocks.insert(symbol.to_string(), asset_shocks);
        }
        
        // Apply correlations
        for (i, &symbol1) in symbols.iter().enumerate() {
            for (j, &symbol2) in symbols.iter().enumerate() {
                if i != j {
                    if let Some(&correlation) = self.correlations.get(&(symbol1.to_string(), symbol2.to_string())) {
                        if let (Some(shocks1), Some(shocks2)) = (shocks.get_mut(symbol1), shocks.get(symbol2)) {
                            // Apply correlation using Cholesky decomposition (simplified)
                            for k in 0..steps {
                                shocks1[k] = shocks1[k] * (1.0 - correlation).sqrt() + shocks2[k] * correlation.sqrt();
                            }
                        }
                    }
                }
            }
        }
        
        // Generate price series using correlated shocks
        for &symbol in symbols {
            let base_price = self.base_prices.get(symbol).unwrap_or(&1000.0);
            let volatility = self.volatilities.get(symbol).unwrap_or(&0.02);
            let asset_shocks = shocks.get(symbol).unwrap();
            
            let mut prices = vec![*base_price];
            let drift = 0.0001;
            
            for i in 1..steps {
                let last_price = prices.last().unwrap();
                let shock = asset_shocks[i];
                
                let price_change = last_price * (drift * dt + volatility * (dt.sqrt()) * shock);
                let new_price = last_price + price_change;
                
                prices.push(new_price.max(0.01));
            }
            
            series.insert(symbol.to_string(), prices);
        }
        
        series
    }
    
    /// Generate order book data
    pub fn generate_order_book(&mut self, symbol: &str, mid_price: f64, depth: usize) -> OrderBook {
        let spread_pct = self.rng.gen_range(0.001..0.01); // 0.1% to 1% spread
        let spread = mid_price * spread_pct;
        
        let mut bids = Vec::new();
        let mut asks = Vec::new();
        
        // Generate bids (buy orders)
        for i in 0..depth {
            let price_offset = (i + 1) as f64 * spread / (depth as f64 * 2.0);
            let price = mid_price - price_offset;
            let quantity = self.rng.gen_range(0.1..10.0); // Random quantity
            
            bids.push(OrderBookLevel {
                price: Decimal::from_f64_retain(price).unwrap(),
                quantity: Decimal::from_f64_retain(quantity).unwrap(),
            });
        }
        
        // Generate asks (sell orders)
        for i in 0..depth {
            let price_offset = (i + 1) as f64 * spread / (depth as f64 * 2.0);
            let price = mid_price + price_offset;
            let quantity = self.rng.gen_range(0.1..10.0);
            
            asks.push(OrderBookLevel {
                price: Decimal::from_f64_retain(price).unwrap(),
                quantity: Decimal::from_f64_retain(quantity).unwrap(),
            });
        }
        
        OrderBook {
            symbol: symbol.to_string(),
            bids,
            asks,
            timestamp: Utc::now(),
        }
    }
    
    /// Generate trade data
    pub fn generate_trades(&mut self, symbol: &str, price_series: &[f64], trade_frequency: f64) -> Vec<Trade> {
        let mut trades = Vec::new();
        let mut current_time = Utc::now();
        
        for (i, &price) in price_series.iter().enumerate() {
            // Decide if a trade occurs (based on frequency)
            if self.rng.gen::<f64>() < trade_frequency {
                let quantity = self.rng.gen_range(0.01..5.0);
                let side = if self.rng.gen::<bool>() { TradeSide::Buy } else { TradeSide::Sell };
                
                trades.push(Trade {
                    id: format!("trade_{}_{}", symbol, trades.len()),
                    symbol: symbol.to_string(),
                    price: Decimal::from_f64_retain(price).unwrap(),
                    quantity: Decimal::from_f64_retain(quantity).unwrap(),
                    side,
                    timestamp: current_time,
                });
            }
            
            // Advance time (assuming each price point is 1 minute)
            current_time += ChronoDuration::minutes(1);
        }
        
        trades
    }
}

/// Supporting data structures for market data
#[derive(Clone, Debug)]
pub struct OrderBook {
    pub symbol: String,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Clone, Debug)]
pub struct OrderBookLevel {
    pub price: Decimal,
    pub quantity: Decimal,
}

#[derive(Clone, Debug)]
pub struct Trade {
    pub id: String,
    pub symbol: String,
    pub price: Decimal,
    pub quantity: Decimal,
    pub side: TradeSide,
    pub timestamp: DateTime<Utc>,
}

#[derive(Clone, Debug)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Consciousness data generator
pub struct ConsciousnessDataGenerator {
    rng: SeededRng,
}

impl ConsciousnessDataGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: SeededRng::new(seed),
        }
    }
    
    /// Generate realistic consciousness state evolution
    pub fn generate_consciousness_evolution(&mut self, dimensions: (usize, usize, usize), 
                                          steps: usize, base_frequency: f64) -> Vec<ConsciousnessState> {
        let mut system = ConsciousnessSystem::new(dimensions, base_frequency);
        system.initialize_coherent_consciousness();
        
        let mut states = Vec::new();
        
        for step in 0..steps {
            // Add some randomness to the evolution
            if step % 10 == 0 && self.rng.gen::<f64>() < 0.3 {
                // Occasionally apply stimulus
                let pos = (
                    self.rng.gen_range(0..dimensions.0),
                    self.rng.gen_range(0..dimensions.1),
                    self.rng.gen_range(0..dimensions.2),
                );
                let intensity = self.rng.gen_range(0.1..2.0);
                let frequency = base_frequency + self.rng.gen_range(-10.0..10.0);
                
                system.apply_stimulus(pos, intensity, frequency, StimulusModality::Neutral);
            }
            
            // Process consciousness cycle
            system.process_consciousness_cycle(0.01, None);
            
            // Record state every few steps
            if step % 5 == 0 {
                states.push(system.get_consciousness_state());
            }
        }
        
        states
    }
    
    /// Generate varied consciousness parameters for testing
    pub fn generate_test_parameters(&mut self) -> Vec<(usize, usize, usize, f64)> {
        vec![
            (2, 2, 2, 20.0),
            (3, 3, 3, 40.0),
            (4, 4, 4, 60.0),
            (5, 5, 5, 80.0),
            (3, 4, 5, 45.0), // Non-cubic dimensions
            (6, 3, 4, 35.0),
        ]
    }
}

/// Emergence data generator
pub struct EmergenceDataGenerator {
    rng: SeededRng,
}

impl EmergenceDataGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: SeededRng::new(seed),
        }
    }
    
    /// Generate realistic emergence history with various patterns
    pub fn generate_emergence_history(&mut self, length: usize, pattern_type: EmergencePattern) -> EmergenceHistory {
        let mut history = EmergenceHistory {
            metrics_history: VecDeque::new(),
            phase_trajectories: VecDeque::new(),
            avalanche_events: VecDeque::new(),
            fitness_evolution: VecDeque::new(),
            lattice_states: VecDeque::new(),
        };
        
        for i in 0..length {
            let t = i as f64;
            let (complexity, coherence) = match pattern_type {
                EmergencePattern::Periodic => {
                    let complexity = 0.5 + 0.3 * (t * 0.1).sin();
                    let coherence = 0.5 + 0.3 * (t * 0.1).cos();
                    (complexity, coherence)
                },
                EmergencePattern::Chaotic => {
                    // Logistic map for chaotic behavior
                    let r = 3.8; // Chaotic parameter
                    let mut x = 0.5; // Initial condition
                    for _ in 0..i % 100 {
                        x = r * x * (1.0 - x);
                    }
                    let complexity = x;
                    let coherence = 1.0 - x; // Inverse relationship
                    (complexity, coherence)
                },
                EmergencePattern::PhaseTransition => {
                    // Sudden transition around midpoint
                    let transition_point = length as f64 / 2.0;
                    let complexity = if t < transition_point { 0.3 } else { 0.8 };
                    let coherence = if t < transition_point { 0.2 } else { 0.9 };
                    (complexity, coherence)
                },
                EmergencePattern::SelfOrganizing => {
                    // Gradual increase with noise
                    let base_complexity = (t / length as f64).min(1.0);
                    let noise = self.rng.gen_range(-0.1..0.1);
                    let complexity = (base_complexity + noise).clamp(0.0, 1.0);
                    let coherence = (base_complexity * 0.8 + noise * 0.5).clamp(0.0, 1.0);
                    (complexity, coherence)
                },
                EmergencePattern::Random => {
                    let complexity = self.rng.gen_range(0.0..1.0);
                    let coherence = self.rng.gen_range(0.0..1.0);
                    (complexity, coherence)
                },
            };
            
            let metrics = SystemMetrics {
                timestamp: t,
                system_size: 100 + (complexity * 100.0) as usize,
                total_energy: 1000.0 + complexity * 500.0 + self.rng.gen_range(-50.0..50.0),
                entropy: 100.0 - coherence * 50.0 + self.rng.gen_range(-10.0..10.0),
                information: complexity,
                complexity,
                coherence,
                coupling: (complexity + coherence) / 2.0,
            };
            
            history.metrics_history.push_back(metrics);
        }
        
        history
    }
    
    /// Generate emergence events for testing detection algorithms
    pub fn generate_emergence_events(&mut self, count: usize) -> Vec<EmergenceEvent> {
        let mut events = Vec::new();
        
        for i in 0..count {
            let event_type = match self.rng.gen_range(0..4) {
                0 => EmergenceType::SelfOrganization,
                1 => EmergenceType::PhaseTransition,
                2 => EmergenceType::CriticalTransition,
                _ => EmergenceType::Synchronization,
            };
            
            events.push(EmergenceEvent {
                id: format!("event_{}", i),
                event_type,
                timestamp: i as f64 * 10.0,
                magnitude: self.rng.gen_range(0.1..1.0),
                duration: self.rng.gen_range(1.0..20.0),
                confidence: self.rng.gen_range(0.5..1.0),
                affected_components: (1..10).map(|j| format!("component_{}", j)).collect(),
            });
        }
        
        events
    }
}

#[derive(Clone, Debug)]
pub enum EmergencePattern {
    Periodic,
    Chaotic,
    PhaseTransition,
    SelfOrganizing,
    Random,
}

#[derive(Clone, Debug)]
pub struct EmergenceEvent {
    pub id: String,
    pub event_type: EmergenceType,
    pub timestamp: f64,
    pub magnitude: f64,
    pub duration: f64,
    pub confidence: f64,
    pub affected_components: Vec<String>,
}

/// Portfolio data generator
pub struct PortfolioDataGenerator {
    rng: SeededRng,
}

impl PortfolioDataGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: SeededRng::new(seed),
        }
    }
    
    /// Generate realistic portfolio compositions
    pub fn generate_portfolio(&mut self, portfolio_type: PortfolioType) -> HashMap<String, Decimal> {
        let mut portfolio = HashMap::new();
        
        match portfolio_type {
            PortfolioType::Conservative => {
                portfolio.insert("USD".to_string(), Decimal::from_f64_retain(50000.0).unwrap());
                portfolio.insert("BTC".to_string(), Decimal::from_f64_retain(0.5).unwrap());
                portfolio.insert("ETH".to_string(), Decimal::from_f64_retain(5.0).unwrap());
            },
            PortfolioType::Balanced => {
                portfolio.insert("USD".to_string(), Decimal::from_f64_retain(25000.0).unwrap());
                portfolio.insert("BTC".to_string(), Decimal::from_f64_retain(1.0).unwrap());
                portfolio.insert("ETH".to_string(), Decimal::from_f64_retain(10.0).unwrap());
                portfolio.insert("SOL".to_string(), Decimal::from_f64_retain(50.0).unwrap());
                portfolio.insert("ADA".to_string(), Decimal::from_f64_retain(1000.0).unwrap());
            },
            PortfolioType::Aggressive => {
                portfolio.insert("USD".to_string(), Decimal::from_f64_retain(10000.0).unwrap());
                portfolio.insert("BTC".to_string(), Decimal::from_f64_retain(2.0).unwrap());
                portfolio.insert("ETH".to_string(), Decimal::from_f64_retain(20.0).unwrap());
                portfolio.insert("SOL".to_string(), Decimal::from_f64_retain(200.0).unwrap());
                portfolio.insert("ADA".to_string(), Decimal::from_f64_retain(5000.0).unwrap());
                portfolio.insert("DOT".to_string(), Decimal::from_f64_retain(100.0).unwrap());
            },
            PortfolioType::Random => {
                let assets = vec!["BTC", "ETH", "SOL", "ADA", "DOT"];
                portfolio.insert("USD".to_string(), Decimal::from_f64_retain(self.rng.gen_range(5000.0..50000.0)).unwrap());
                
                for asset in assets {
                    let quantity = match asset {
                        "BTC" => self.rng.gen_range(0.1..5.0),
                        "ETH" => self.rng.gen_range(1.0..50.0),
                        _ => self.rng.gen_range(10.0..1000.0),
                    };
                    portfolio.insert(asset.to_string(), Decimal::from_f64_retain(quantity).unwrap());
                }
            },
        }
        
        portfolio
    }
    
    /// Generate performance history for a portfolio
    pub fn generate_performance_history(&mut self, initial_value: f64, days: usize, volatility: f64) -> Vec<f64> {
        let mut values = vec![initial_value];
        let normal = Normal::new(0.0, 1.0).unwrap();
        
        for _ in 1..days {
            let last_value = values.last().unwrap();
            let daily_return = normal.sample(&mut self.rng.rng) * volatility;
            let new_value = last_value * (1.0 + daily_return);
            values.push(new_value.max(initial_value * 0.1)); // Prevent total collapse
        }
        
        values
    }
}

#[derive(Clone, Debug)]
pub enum PortfolioType {
    Conservative,
    Balanced,
    Aggressive,
    Random,
}

/// Test scenario generator
pub struct TestScenarioGenerator {
    market_gen: MarketDataGenerator,
    consciousness_gen: ConsciousnessDataGenerator,
    emergence_gen: EmergenceDataGenerator,
    portfolio_gen: PortfolioDataGenerator,
}

impl TestScenarioGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            market_gen: MarketDataGenerator::new(seed),
            consciousness_gen: ConsciousnessDataGenerator::new(seed + 1),
            emergence_gen: EmergenceDataGenerator::new(seed + 2),
            portfolio_gen: PortfolioDataGenerator::new(seed + 3),
        }
    }
    
    /// Generate complete test scenario
    pub fn generate_scenario(&mut self, scenario_type: ScenarioType) -> TestScenario {
        match scenario_type {
            ScenarioType::MarketCrash => self.generate_market_crash_scenario(),
            ScenarioType::BullRun => self.generate_bull_run_scenario(),
            ScenarioType::Sideways => self.generate_sideways_scenario(),
            ScenarioType::HighVolatility => self.generate_high_volatility_scenario(),
            ScenarioType::EmergenceTesting => self.generate_emergence_testing_scenario(),
            ScenarioType::ConsciousnessEvolution => self.generate_consciousness_evolution_scenario(),
        }
    }
    
    fn generate_market_crash_scenario(&mut self) -> TestScenario {
        // Simulate market crash with high correlations
        self.market_gen.add_correlation("BTC", "ETH", 0.95);
        self.market_gen.add_correlation("ETH", "SOL", 0.9);
        
        let symbols = vec!["BTC", "ETH", "SOL", "ADA"];
        let crash_series = self.market_gen.generate_correlated_series(&symbols, 100, 1.0/24.0);
        
        // Apply crash (30% drop over 10 periods)
        let mut crashed_series = HashMap::new();
        for (symbol, mut prices) in crash_series {
            for i in 50..60 {
                if i < prices.len() {
                    prices[i] *= 1.0 - 0.03 * (i - 49) as f64; // 3% drop per period
                }
            }
            crashed_series.insert(symbol, prices);
        }
        
        TestScenario {
            name: "Market Crash".to_string(),
            description: "Simulates a coordinated market crash with high correlations".to_string(),
            market_data: crashed_series,
            portfolios: vec![
                self.portfolio_gen.generate_portfolio(PortfolioType::Conservative),
                self.portfolio_gen.generate_portfolio(PortfolioType::Balanced),
                self.portfolio_gen.generate_portfolio(PortfolioType::Aggressive),
            ],
            emergence_history: self.emergence_gen.generate_emergence_history(100, EmergencePattern::PhaseTransition),
            consciousness_states: vec![], // Not relevant for this scenario
            expected_behaviors: vec![
                "Portfolio values should decline".to_string(),
                "Risk metrics should spike".to_string(),
                "Correlation should increase".to_string(),
            ],
        }
    }
    
    fn generate_bull_run_scenario(&mut self) -> TestScenario {
        let symbols = vec!["BTC", "ETH", "SOL", "ADA"];
        let mut bull_series = self.market_gen.generate_correlated_series(&symbols, 200, 1.0/24.0);
        
        // Apply bull run (gradual increase)
        for (_, prices) in bull_series.iter_mut() {
            for i in 1..prices.len() {
                prices[i] *= 1.0 + 0.005; // 0.5% increase per period
            }
        }
        
        TestScenario {
            name: "Bull Run".to_string(),
            description: "Simulates sustained upward market movement".to_string(),
            market_data: bull_series,
            portfolios: vec![self.portfolio_gen.generate_portfolio(PortfolioType::Balanced)],
            emergence_history: self.emergence_gen.generate_emergence_history(200, EmergencePattern::SelfOrganizing),
            consciousness_states: vec![],
            expected_behaviors: vec![
                "Portfolio values should increase".to_string(),
                "Volatility should be moderate".to_string(),
                "Risk-adjusted returns should be positive".to_string(),
            ],
        }
    }
    
    fn generate_sideways_scenario(&mut self) -> TestScenario {
        let symbols = vec!["BTC", "ETH"];
        let sideways_series = self.market_gen.generate_correlated_series(&symbols, 300, 1.0/24.0);
        
        TestScenario {
            name: "Sideways Market".to_string(),
            description: "Simulates range-bound market with no clear trend".to_string(),
            market_data: sideways_series,
            portfolios: vec![self.portfolio_gen.generate_portfolio(PortfolioType::Conservative)],
            emergence_history: self.emergence_gen.generate_emergence_history(300, EmergencePattern::Periodic),
            consciousness_states: vec![],
            expected_behaviors: vec![
                "Portfolio values should remain stable".to_string(),
                "Mean reversion strategies should work".to_string(),
                "Low directional momentum".to_string(),
            ],
        }
    }
    
    fn generate_high_volatility_scenario(&mut self) -> TestScenario {
        // Increase volatilities for high volatility scenario
        self.market_gen.add_asset("BTC", 45000.0, 0.1); // 10% daily volatility
        self.market_gen.add_asset("ETH", 3000.0, 0.12); // 12% daily volatility
        
        let symbols = vec!["BTC", "ETH"];
        let volatile_series = self.market_gen.generate_correlated_series(&symbols, 150, 1.0/24.0);
        
        TestScenario {
            name: "High Volatility".to_string(),
            description: "Simulates extremely volatile market conditions".to_string(),
            market_data: volatile_series,
            portfolios: vec![self.portfolio_gen.generate_portfolio(PortfolioType::Random)],
            emergence_history: self.emergence_gen.generate_emergence_history(150, EmergencePattern::Chaotic),
            consciousness_states: vec![],
            expected_behaviors: vec![
                "High price swings".to_string(),
                "Increased risk metrics".to_string(),
                "Challenging for risk management".to_string(),
            ],
        }
    }
    
    fn generate_emergence_testing_scenario(&mut self) -> TestScenario {
        TestScenario {
            name: "Emergence Testing".to_string(),
            description: "Focused on testing emergence detection algorithms".to_string(),
            market_data: HashMap::new(),
            portfolios: vec![],
            emergence_history: self.emergence_gen.generate_emergence_history(500, EmergencePattern::PhaseTransition),
            consciousness_states: vec![],
            expected_behaviors: vec![
                "Phase transition should be detected".to_string(),
                "Emergence score should spike".to_string(),
                "Patterns should be identified".to_string(),
            ],
        }
    }
    
    fn generate_consciousness_evolution_scenario(&mut self) -> TestScenario {
        let consciousness_states = self.consciousness_gen.generate_consciousness_evolution(
            (4, 4, 4), 200, 40.0
        );
        
        TestScenario {
            name: "Consciousness Evolution".to_string(),
            description: "Tests consciousness system evolution and integration".to_string(),
            market_data: HashMap::new(),
            portfolios: vec![],
            emergence_history: EmergenceHistory {
                metrics_history: VecDeque::new(),
                phase_trajectories: VecDeque::new(),
                avalanche_events: VecDeque::new(),
                fitness_evolution: VecDeque::new(),
                lattice_states: VecDeque::new(),
            },
            consciousness_states,
            expected_behaviors: vec![
                "Consciousness should emerge periodically".to_string(),
                "Integration metrics should evolve".to_string(),
                "System should maintain stability".to_string(),
            ],
        }
    }
}

#[derive(Clone, Debug)]
pub enum ScenarioType {
    MarketCrash,
    BullRun,
    Sideways,
    HighVolatility,
    EmergenceTesting,
    ConsciousnessEvolution,
}

#[derive(Clone, Debug)]
pub struct TestScenario {
    pub name: String,
    pub description: String,
    pub market_data: HashMap<String, Vec<f64>>,
    pub portfolios: Vec<HashMap<String, Decimal>>,
    pub emergence_history: EmergenceHistory,
    pub consciousness_states: Vec<ConsciousnessState>,
    pub expected_behaviors: Vec<String>,
}

#[cfg(test)]
mod generator_tests {
    use super::*;
    
    #[test]
    fn test_market_data_generator() {
        let mut generator = MarketDataGenerator::new(42);
        let prices = generator.generate_price_series("BTC", 100, 1.0/24.0);
        
        assert_eq!(prices.len(), 100);
        assert!(prices.iter().all(|&p| p > 0.0));
        
        // Prices should show some variation
        let initial_price = prices[0];
        let final_price = *prices.last().unwrap();
        assert!((final_price - initial_price).abs() / initial_price > 0.01); // At least 1% change
    }
    
    #[test]
    fn test_consciousness_data_generator() {
        let mut generator = ConsciousnessDataGenerator::new(123);
        let states = generator.generate_consciousness_evolution((3, 3, 3), 50, 40.0);
        
        assert!(states.len() > 0);
        
        // States should have valid metrics
        for state in &states {
            assert!(state.integration_metrics.overall_integration >= 0.0);
            assert!(state.field_state.consciousness_level >= 0.0);
        }
    }
    
    #[test]
    fn test_emergence_data_generator() {
        let mut generator = EmergenceDataGenerator::new(456);
        let history = generator.generate_emergence_history(100, EmergencePattern::Periodic);
        
        assert_eq!(history.metrics_history.len(), 100);
        
        // Check that periodic pattern is present
        let complexities: Vec<f64> = history.metrics_history.iter()
            .map(|m| m.complexity)
            .collect();
        
        // Simple test: should have some variation
        let min_complexity = complexities.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_complexity = complexities.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        assert!(max_complexity - min_complexity > 0.1);
    }
    
    #[test]
    fn test_portfolio_generator() {
        let mut generator = PortfolioDataGenerator::new(789);
        let portfolio = generator.generate_portfolio(PortfolioType::Balanced);
        
        assert!(portfolio.contains_key("USD"));
        assert!(portfolio.contains_key("BTC"));
        assert!(portfolio.contains_key("ETH"));
        
        // All quantities should be positive
        for (_, &quantity) in &portfolio {
            assert!(quantity > Decimal::ZERO);
        }
    }
    
    #[test]
    fn test_scenario_generator() {
        let mut generator = TestScenarioGenerator::new(999);
        let scenario = generator.generate_scenario(ScenarioType::MarketCrash);
        
        assert_eq!(scenario.name, "Market Crash");
        assert!(!scenario.market_data.is_empty());
        assert!(!scenario.portfolios.is_empty());
        assert!(!scenario.expected_behaviors.is_empty());
    }
}