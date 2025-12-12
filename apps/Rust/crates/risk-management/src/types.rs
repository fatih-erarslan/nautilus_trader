//! Common types for the risk management system

use std::collections::HashMap;
use std::time::Duration;
use chrono::{DateTime, Utc};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Portfolio representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Portfolio {
    /// Portfolio ID
    pub id: Uuid,
    /// Asset positions
    pub positions: Vec<Position>,
    /// Available assets
    pub assets: Vec<Asset>,
    /// Historical returns
    pub returns: Vec<f64>,
    /// Target returns
    pub targets: Vec<f64>,
    /// Market data
    pub market_data: MarketData,
    /// Portfolio value
    pub total_value: f64,
    /// Cash position
    pub cash: f64,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl Default for Portfolio {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            positions: Vec::new(),
            assets: Vec::new(),
            returns: Vec::new(),
            targets: Vec::new(),
            market_data: MarketData::default(),
            total_value: 100_000.0,
            cash: 10_000.0,
            timestamp: Utc::now(),
        }
    }
}

/// Asset position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Asset symbol
    pub symbol: String,
    /// Quantity held
    pub quantity: f64,
    /// Current price
    pub price: f64,
    /// Market value
    pub market_value: f64,
    /// Weight in portfolio
    pub weight: f64,
    /// P&L
    pub pnl: f64,
    /// Entry price
    pub entry_price: f64,
    /// Entry timestamp
    pub entry_time: DateTime<Utc>,
}

/// Asset information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Asset {
    /// Asset symbol
    pub symbol: String,
    /// Asset name
    pub name: String,
    /// Asset class
    pub asset_class: AssetClass,
    /// Current price
    pub price: f64,
    /// Volatility
    pub volatility: f64,
    /// Beta
    pub beta: f64,
    /// Expected return
    pub expected_return: f64,
    /// Liquidity score
    pub liquidity_score: f64,
}

/// Asset classes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssetClass {
    Equity,
    Bond,
    Currency,
    Commodity,
    Crypto,
    Derivative,
    Alternative,
}

/// Execution strategy types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionStrategy {
    /// Quantum-enhanced execution plan
    Quantum(QuantumExecutionPlan),
    /// Time-Weighted Average Price
    Twap(TwapStrategy),
    /// Volume-Weighted Average Price
    Vwap(VwapStrategy),
}

/// Quantum execution plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumExecutionPlan {
    pub expected_cost: f64,
    pub cost_variance: f64,
    pub quantum_advantage: f64,
    pub execution_time: Duration,
}

/// TWAP strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwapStrategy {
    pub expected_cost: f64,
    pub cost_variance: f64,
    pub time_intervals: u32,
    pub execution_duration: Duration,
}

/// VWAP strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VwapStrategy {
    pub expected_cost: f64,
    pub cost_variance: f64,
    pub volume_profile: Vec<f64>,
    pub execution_duration: Duration,
}

/// Routing strategy for message delivery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Direct delivery to specific agent
    DirectDelivery,
    /// Broadcast to all agents
    Broadcast,
}

/// Execution risk metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRiskMetrics {
    pub cost_variance: f64,
    pub execution_shortfall_risk: f64,
    pub timing_risk: f64,
}

/// Market data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    /// Price data
    pub prices: HashMap<String, Vec<f64>>,
    /// Return data
    pub returns: HashMap<String, Vec<f64>>,
    /// Volatility data
    pub volatilities: HashMap<String, f64>,
    /// Correlation matrix
    pub correlation_matrix: Array2<f64>,
    /// Timestamps
    pub timestamps: Vec<DateTime<Utc>>,
}

impl Default for MarketData {
    fn default() -> Self {
        Self {
            prices: HashMap::new(),
            returns: HashMap::new(),
            volatilities: HashMap::new(),
            correlation_matrix: Array2::zeros((0, 0)),
            timestamps: Vec::new(),
        }
    }
}

/// Trading signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    /// Asset symbol
    pub symbol: String,
    /// Signal direction
    pub direction: SignalDirection,
    /// Signal strength (0-1)
    pub strength: f64,
    /// Confidence level
    pub confidence: f64,
    /// Expected return
    pub expected_return: f64,
    /// Expected volatility
    pub expected_volatility: f64,
    /// Time horizon
    pub time_horizon: Duration,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Signal direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalDirection {
    Buy,
    Sell,
    Hold,
}

/// Position sizes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionSizes {
    /// Position sizes by symbol
    pub sizes: HashMap<String, f64>,
    /// Total allocation
    pub total_allocation: f64,
    /// Risk budget used
    pub risk_budget_used: f64,
    /// Kelly fractions
    pub kelly_fractions: HashMap<String, f64>,
}

/// Portfolio constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioConstraints {
    /// Minimum weights
    pub min_weights: HashMap<String, f64>,
    /// Maximum weights
    pub max_weights: HashMap<String, f64>,
    /// Maximum turnover
    pub max_turnover: f64,
    /// Maximum risk
    pub max_risk: f64,
    /// Target return
    pub target_return: Option<f64>,
    /// Sector constraints
    pub sector_constraints: HashMap<String, f64>,
}

/// Optimized portfolio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedPortfolio {
    /// Optimal weights
    pub weights: HashMap<String, f64>,
    /// Expected return
    pub expected_return: f64,
    /// Expected risk
    pub expected_risk: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Optimization objective value
    pub objective_value: f64,
    /// Convergence status
    pub converged: bool,
}

/// Stress scenario
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressScenario {
    /// Scenario name
    pub name: String,
    /// Scenario description
    pub description: String,
    /// Asset shocks
    pub asset_shocks: HashMap<String, f64>,
    /// Volatility multipliers
    pub volatility_multipliers: HashMap<String, f64>,
    /// Correlation shifts
    pub correlation_shifts: Array2<f64>,
    /// Liquidity impacts
    pub liquidity_impacts: HashMap<String, f64>,
    /// Scenario probability
    pub probability: f64,
}

/// Real-time risk metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeRiskMetrics {
    /// Portfolio VaR
    pub portfolio_var: f64,
    /// Portfolio CVaR
    pub portfolio_cvar: f64,
    /// Current drawdown
    pub current_drawdown: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Portfolio volatility
    pub portfolio_volatility: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Beta
    pub beta: f64,
    /// Tracking error
    pub tracking_error: f64,
    /// Concentration risk
    pub concentration_risk: f64,
    /// Liquidity risk
    pub liquidity_risk: f64,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Risk limit breach
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimitBreach {
    /// Breach type
    pub breach_type: BreachType,
    /// Current value
    pub current_value: f64,
    /// Limit value
    pub limit_value: f64,
    /// Severity
    pub severity: BreachSeverity,
    /// Asset or portfolio affected
    pub affected_entity: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Recommended action
    pub recommended_action: String,
}

/// Types of risk limit breaches
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BreachType {
    VaR,
    CVaR,
    Drawdown,
    Leverage,
    Concentration,
    Liquidity,
    Volatility,
    Correlation,
}

/// Breach severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BreachSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Reporting periods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportingPeriod {
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Annual,
}

/// Risk-adjusted metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAdjustedMetrics {
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Calmar ratio
    pub calmar_ratio: f64,
    /// Information ratio
    pub information_ratio: f64,
    /// Treynor ratio
    pub treynor_ratio: f64,
    /// Jensen's alpha
    pub jensen_alpha: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Volatility
    pub volatility: f64,
    /// Skewness
    pub skewness: f64,
    /// Kurtosis
    pub kurtosis: f64,
    /// VaR (1%, 5%, 10%)
    pub var_levels: HashMap<String, f64>,
    /// CVaR (1%, 5%, 10%)
    pub cvar_levels: HashMap<String, f64>,
}

/// Monte Carlo simulation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloResults {
    /// Simulated portfolio values
    pub portfolio_values: Vec<f64>,
    /// Simulated returns
    pub returns: Vec<f64>,
    /// Value-at-Risk estimates
    pub var_estimates: HashMap<String, f64>,
    /// Conditional VaR estimates
    pub cvar_estimates: HashMap<String, f64>,
    /// Probability of loss
    pub probability_of_loss: f64,
    /// Expected shortfall
    pub expected_shortfall: f64,
    /// Simulation parameters
    pub simulation_params: SimulationParams,
}

/// Simulation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationParams {
    /// Number of simulations
    pub num_simulations: usize,
    /// Time horizon
    pub time_horizon: Duration,
    /// Random seed
    pub seed: Option<u64>,
    /// Simulation method
    pub method: SimulationMethod,
}

/// Simulation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SimulationMethod {
    MonteCarlo,
    QuasiMonteCarlo,
    LatinHyperCube,
    Sobol,
    Halton,
}

impl Portfolio {
    /// Calculate portfolio value
    pub fn calculate_value(&self) -> f64 {
        self.positions.iter().map(|p| p.market_value).sum::<f64>() + self.cash
    }
    
    /// Calculate portfolio weights
    pub fn calculate_weights(&self) -> HashMap<String, f64> {
        let total_value = self.calculate_value();
        self.positions
            .iter()
            .map(|p| (p.symbol.clone(), p.market_value / total_value))
            .collect()
    }
    
    /// Get position by symbol
    pub fn get_position(&self, symbol: &str) -> Option<&Position> {
        self.positions.iter().find(|p| p.symbol == symbol)
    }
    
    /// Update position
    pub fn update_position(&mut self, symbol: &str, quantity: f64, price: f64) {
        if let Some(position) = self.positions.iter_mut().find(|p| p.symbol == symbol) {
            position.quantity = quantity;
            position.price = price;
            position.market_value = quantity * price;
            position.pnl = (price - position.entry_price) * quantity;
        }
    }
    
    /// Add new position
    pub fn add_position(&mut self, symbol: String, quantity: f64, price: f64) {
        let position = Position {
            symbol,
            quantity,
            price,
            market_value: quantity * price,
            weight: 0.0, // Will be calculated later
            pnl: 0.0,
            entry_price: price,
            entry_time: Utc::now(),
        };
        self.positions.push(position);
    }
    
    /// Remove position
    pub fn remove_position(&mut self, symbol: &str) {
        self.positions.retain(|p| p.symbol != symbol);
    }
    
    /// Calculate portfolio beta
    pub fn calculate_beta(&self) -> f64 {
        let weights = self.calculate_weights();
        let mut portfolio_beta = 0.0;
        
        for asset in &self.assets {
            if let Some(&weight) = weights.get(&asset.symbol) {
                portfolio_beta += weight * asset.beta;
            }
        }
        
        portfolio_beta
    }
    
    /// Calculate portfolio expected return
    pub fn calculate_expected_return(&self) -> f64 {
        let weights = self.calculate_weights();
        let mut expected_return = 0.0;
        
        for asset in &self.assets {
            if let Some(&weight) = weights.get(&asset.symbol) {
                expected_return += weight * asset.expected_return;
            }
        }
        
        expected_return
    }
    
    /// Calculate portfolio volatility
    pub fn calculate_volatility(&self) -> f64 {
        let weights = self.calculate_weights();
        let mut portfolio_variance = 0.0;
        
        // Individual asset variances
        for asset in &self.assets {
            if let Some(&weight) = weights.get(&asset.symbol) {
                portfolio_variance += weight.powi(2) * asset.volatility.powi(2);
            }
        }
        
        // Covariance terms (simplified - would need full covariance matrix)
        for i in 0..self.assets.len() {
            for j in i + 1..self.assets.len() {
                let asset_i = &self.assets[i];
                let asset_j = &self.assets[j];
                
                if let (Some(&weight_i), Some(&weight_j)) = 
                    (weights.get(&asset_i.symbol), weights.get(&asset_j.symbol)) {
                    // Simplified correlation assumption (would use actual correlation matrix)
                    let correlation = 0.3; // Placeholder
                    portfolio_variance += 2.0 * weight_i * weight_j * 
                        asset_i.volatility * asset_j.volatility * correlation;
                }
            }
        }
        
        portfolio_variance.sqrt()
    }
}

// ============ EPSILON-2: ADDITIONAL MISSING TYPES ============

/// Regime monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeMonitoringConfig {
    pub lookback_periods: u32,
    pub volatility_threshold: f64,
    pub trend_strength_threshold: f64,
    pub update_frequency_ms: u64,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig<T> {
    pub benchmark: String,
    pub measurement_window: Duration,
    pub risk_free_rate: f64,
    pub parameters: T,
}

/// Strategy parameter adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyParameterAdaptation {
    pub parameter_name: String,
    pub current_value: f64,
    pub target_value: f64,
    pub adaptation_rate: f64,
    pub bounds: (f64, f64),
}

/// Real-time liquidity update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeLiquidityUpdate {
    pub symbol: String,
    pub bid_size: f64,
    pub ask_size: f64,
    pub spread: f64,
    pub timestamp: DateTime<Utc>,
}

/// Real-time correlation update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeCorrelationUpdate {
    pub asset_pair: (String, String),
    pub correlation: f64,
    pub rolling_window: u32,
    pub timestamp: DateTime<Utc>,
}

/// Temperature scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureScalingConfig {
    pub initial_temperature: f64,
    pub target_temperature: f64,
    pub cooling_rate: f64,
    pub min_temperature: f64,
}

// ============ GAMMA-3: LIQUIDITY RISK MISSING TYPES ============

/// Quantum liquidity estimator
#[derive(Debug, Clone)]
pub struct QuantumLiquidityEstimator {
    pub quantum_state: Vec<f64>,
    pub estimation_confidence: f64,
    pub coherence_time_ms: u64,
}

/// Market impact analyzer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketImpactAnalyzer {
    pub impact_models: Vec<String>,
    pub historical_data_window: Duration,
    pub prediction_accuracy: f64,
}

/// Liquidity risk calculator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityRiskCalculator {
    pub risk_metrics: Vec<String>,
    pub calculation_method: String,
    pub threshold_values: HashMap<String, f64>,
}

// ============ GAMMA-4: CORRELATION ANALYSIS MISSING TYPES ============

/// Quantum correlation detector
#[derive(Debug, Clone)]
pub struct QuantumCorrelationDetector {
    pub entanglement_threshold: f64,
    pub measurement_basis: Vec<String>,
    pub correlation_matrix: Array2<f64>,
}

/// Regime change detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeChangeDetector {
    pub detection_algorithms: Vec<String>,
    pub sensitivity: f64,
    pub lookback_window: Duration,
}

/// Copula analyzer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopulaAnalyzer {
    pub copula_type: String,
    pub parameters: HashMap<String, f64>,
    pub goodness_of_fit: f64,
}

// ============ GAMMA-8: FINAL MISSING TYPES BATCH ============

/// Multi-timeframe market data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTimeframeMarketData {
    pub symbol: String,
    pub timeframes: HashMap<String, Vec<MarketTick>>,
    pub correlation_matrix: Array2<f64>,
}

/// Market tick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketTick {
    pub timestamp: DateTime<Utc>,
    pub price: f64,
    pub volume: f64,
    pub bid: f64,
    pub ask: f64,
}

/// Historical calibration data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalCalibrationData {
    pub training_period: Duration,
    pub validation_period: Duration,
    pub market_data: Vec<MarketTick>,
    pub performance_metrics: HashMap<String, f64>,
}

/// Execution optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionOptimizationResult {
    pub optimal_strategy: String,
    pub expected_cost: f64,
    pub risk_adjusted_return: f64,
    pub confidence_interval: (f64, f64),
}

// ============ OMEGA-3: COORDINATION MISSING TYPES ============

/// Coordination metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationMetrics {
    pub agent_count: u32,
    pub message_rate: f64,
    pub coordination_latency_us: u64,
    pub success_rate: f64,
}

/// Quantum consensus engine
#[derive(Debug, Clone)]
pub struct QuantumConsensusEngine {
    pub consensus_algorithm: String,
    pub participant_count: u32,
    pub quantum_entanglement_factor: f64,
}

/// Swarm load balancer
#[derive(Debug, Clone)]
pub struct SwarmLoadBalancer {
    pub balancing_strategy: String,
    pub node_weights: HashMap<String, f64>,
    pub current_load: f64,
}