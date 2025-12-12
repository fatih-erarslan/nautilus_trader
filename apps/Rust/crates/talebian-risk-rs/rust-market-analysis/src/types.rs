//! Core types and data structures for market analysis

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Market data structure containing OHLCV and additional metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub prices: Vec<f64>,
    pub volumes: Vec<f64>,
    pub trades: Vec<Trade>,
    pub order_book: Option<OrderBook>,
    pub timeframe: Timeframe,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl MarketData {
    pub fn new(symbol: String, timeframe: Timeframe) -> Self {
        Self {
            symbol,
            timestamp: Utc::now(),
            prices: Vec::new(),
            volumes: Vec::new(),
            trades: Vec::new(),
            order_book: None,
            timeframe,
            metadata: HashMap::new(),
        }
    }
    
    pub fn mock_data() -> Self {
        let mut data = Self::new("BTCUSDT".to_string(), Timeframe::OneMinute);
        data.prices = (0..100).map(|i| 50000.0 + (i as f64 * 10.0)).collect();
        data.volumes = (0..100).map(|i| 100.0 + (i as f64 * 5.0)).collect();
        data
    }
    
    pub fn add_trade(&mut self, trade: Trade) {
        self.trades.push(trade);
    }
    
    pub fn set_order_book(&mut self, order_book: OrderBook) {
        self.order_book = Some(order_book);
    }
    
    pub fn get_returns(&self) -> Vec<f64> {
        self.prices.windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect()
    }
}

/// Individual trade data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub price: f64,
    pub quantity: f64,
    pub side: TradeSide,
    pub trade_type: TradeType,
}

/// Trade side (buy/sell)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Trade type classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradeType {
    Market,
    Limit,
    Stop,
    Whale,      // Large institutional trades
    Retail,     // Small retail trades
    Unknown,
}

/// Order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub timestamp: DateTime<Utc>,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
    pub sequence: u64,
}

/// Order book level (price, quantity)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: f64,
    pub quantity: f64,
    pub order_count: Option<u32>,
}

/// Time frame for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Timeframe {
    OneSecond,
    FiveSeconds,
    FifteenSeconds,
    ThirtySeconds,
    OneMinute,
    FiveMinutes,
    FifteenMinutes,
    ThirtyMinutes,
    OneHour,
    FourHours,
    OneDay,
    OneWeek,
}

impl Timeframe {
    pub fn to_seconds(&self) -> u32 {
        match self {
            Timeframe::OneSecond => 1,
            Timeframe::FiveSeconds => 5,
            Timeframe::FifteenSeconds => 15,
            Timeframe::ThirtySeconds => 30,
            Timeframe::OneMinute => 60,
            Timeframe::FiveMinutes => 300,
            Timeframe::FifteenMinutes => 900,
            Timeframe::ThirtyMinutes => 1800,
            Timeframe::OneHour => 3600,
            Timeframe::FourHours => 14400,
            Timeframe::OneDay => 86400,
            Timeframe::OneWeek => 604800,
        }
    }
}

/// Complete market analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketAnalysis {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub whale_signals: Vec<WhaleSignal>,
    pub regime_info: RegimeInfo,
    pub patterns: Vec<Pattern>,
    pub predictions: Predictions,
    pub microstructure: MicrostructureAnalysis,
    pub confidence_score: f64,
    pub risk_metrics: RiskMetrics,
}

/// Whale activity signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleSignal {
    pub signal_type: WhaleSignalType,
    pub strength: f64,
    pub confidence: f64,
    pub volume_profile: VolumeProfile,
    pub order_flow: OrderFlowImbalance,
    pub impact_analysis: PriceImpactAnalysis,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WhaleSignalType {
    Accumulation,
    Distribution,
    Manipulation,
    Breakout,
    Support,
    Resistance,
}

/// Volume profile analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeProfile {
    pub value_area_high: f64,
    pub value_area_low: f64,
    pub point_of_control: f64,
    pub volume_by_price: Vec<(f64, f64)>, // (price, volume)
    pub volume_delta: f64,
    pub cumulative_volume_delta: f64,
}

/// Order flow imbalance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderFlowImbalance {
    pub buy_pressure: f64,
    pub sell_pressure: f64,
    pub imbalance_ratio: f64,
    pub aggressive_buy_ratio: f64,
    pub aggressive_sell_ratio: f64,
    pub order_size_distribution: OrderSizeDistribution,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderSizeDistribution {
    pub small_orders: f64,    // < 1 BTC equivalent
    pub medium_orders: f64,   // 1-10 BTC equivalent
    pub large_orders: f64,    // 10-100 BTC equivalent
    pub whale_orders: f64,    // > 100 BTC equivalent
}

/// Price impact analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceImpactAnalysis {
    pub immediate_impact: f64,
    pub delayed_impact: f64,
    pub recovery_time: Option<chrono::Duration>,
    pub market_depth: f64,
    pub slippage_estimate: f64,
}

/// Market regime information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeInfo {
    pub current_regime: MarketRegime,
    pub previous_regime: Option<MarketRegime>,
    pub confidence: f64,
    pub regime_duration: chrono::Duration,
    pub transition_probability: HashMap<MarketRegime, f64>,
    pub volatility_regime: VolatilityRegime,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum MarketRegime {
    Bull,
    Bear,
    Sideways,
    HighVolatility,
    LowVolatility,
    TrendingUp,
    TrendingDown,
    Accumulation,
    Distribution,
    Breakout,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VolatilityRegime {
    Low,
    Medium,
    High,
    Extreme,
}

/// Detected patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    pub pattern_type: PatternType,
    pub confidence: f64,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub price_levels: Vec<f64>,
    pub volume_confirmation: bool,
    pub breakout_target: Option<f64>,
    pub stop_loss: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    // Classical patterns
    HeadAndShoulders,
    InverseHeadAndShoulders,
    DoubleTop,
    DoubleBottom,
    TripleTop,
    TripleBottom,
    Cup,
    Handle,
    Flag,
    Pennant,
    Triangle,
    Wedge,
    
    // Support/Resistance
    Support,
    Resistance,
    Breakout,
    Breakdown,
    
    // Volume patterns
    VolumeSpike,
    VolumeClimaxTop,
    VolumeClimaxBottom,
    
    // Microstructure patterns
    OrderBookImbalance,
    HiddenLiquidity,
    WallBreach,
}

/// Prediction results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Predictions {
    pub short_term: Vec<PricePrediction>,  // 1-60 minutes
    pub medium_term: Vec<PricePrediction>, // 1-24 hours
    pub volatility_forecast: VolatilityForecast,
    pub trend_probability: TrendProbability,
    pub liquidity_forecast: LiquidityForecast,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricePrediction {
    pub horizon: chrono::Duration,
    pub predicted_price: f64,
    pub confidence_interval: (f64, f64),
    pub probability_distribution: Vec<(f64, f64)>, // (price, probability)
    pub model_uncertainty: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityForecast {
    pub current_volatility: f64,
    pub forecasted_volatility: Vec<(chrono::Duration, f64)>,
    pub volatility_regime_change_probability: f64,
    pub garch_parameters: Option<GarchParameters>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GarchParameters {
    pub omega: f64,
    pub alpha: f64,
    pub beta: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendProbability {
    pub uptrend_probability: f64,
    pub downtrend_probability: f64,
    pub sideways_probability: f64,
    pub trend_strength: f64,
    pub momentum_indicators: MomentumIndicators,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MomentumIndicators {
    pub rsi: f64,
    pub macd: f64,
    pub macd_signal: f64,
    pub stochastic: f64,
    pub williams_r: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityForecast {
    pub current_liquidity: f64,
    pub forecasted_liquidity: Vec<(chrono::Duration, f64)>,
    pub market_impact_cost: f64,
    pub optimal_execution_window: Option<chrono::Duration>,
}

/// Market microstructure analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicrostructureAnalysis {
    pub bid_ask_spread: f64,
    pub market_depth: MarketDepth,
    pub order_flow: DetailedOrderFlow,
    pub liquidity_metrics: LiquidityMetrics,
    pub market_efficiency: MarketEfficiency,
    pub trading_intensity: TradingIntensity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDepth {
    pub total_bid_depth: f64,
    pub total_ask_depth: f64,
    pub depth_imbalance: f64,
    pub depth_by_level: Vec<(u32, f64, f64)>, // (level, bid_depth, ask_depth)
    pub market_resilience: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedOrderFlow {
    pub buy_volume: f64,
    pub sell_volume: f64,
    pub net_flow: f64,
    pub aggressive_ratio: f64,
    pub order_arrival_rate: f64,
    pub cancellation_rate: f64,
    pub fill_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityMetrics {
    pub amihud_illiquidity: f64,
    pub roll_spread: f64,
    pub effective_spread: f64,
    pub realized_spread: f64,
    pub price_impact: f64,
    pub kyle_lambda: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketEfficiency {
    pub variance_ratio: f64,
    pub hurst_exponent: f64,
    pub autocorrelation: Vec<f64>,
    pub information_share: f64,
    pub price_discovery_metrics: PriceDiscoveryMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceDiscoveryMetrics {
    pub information_share: f64,
    pub component_share: f64,
    pub hasbrouck_info_share: f64,
    pub gonzalo_granger_metric: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingIntensity {
    pub trades_per_minute: f64,
    pub volume_per_minute: f64,
    pub average_trade_size: f64,
    pub trade_size_variance: f64,
    pub intensity_clustering: f64,
}

/// Risk metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    pub value_at_risk_95: f64,
    pub expected_shortfall_95: f64,
    pub maximum_drawdown: f64,
    pub volatility_regime: String,
    pub tail_ratio: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

/// Market state tracking
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MarketState {
    pub current_regime: Option<MarketRegime>,
    pub volatility_regime: Option<VolatilityRegime>,
    pub last_whale_activity: Option<DateTime<Utc>>,
    pub active_patterns: Vec<Pattern>,
    pub trend_strength: f64,
    pub market_stress_level: f64,
}

/// Performance metrics for the analyzer
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_analyses: u64,
    pub average_processing_time: f64,
    pub cache_hit_rate: f64,
    pub prediction_accuracy: f64,
    pub whale_detection_accuracy: f64,
    pub pattern_recognition_accuracy: f64,
    pub regime_detection_accuracy: f64,
    pub last_analysis_time: Option<DateTime<Utc>>,
    pub high_confidence_analyses: u64,
}

/// Analysis signals for event system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalysisSignal {
    WhaleActivity {
        symbol: String,
        signal: WhaleSignal,
        timestamp: DateTime<Utc>,
    },
    RegimeChange {
        symbol: String,
        old_regime: Option<MarketRegime>,
        new_regime: MarketRegime,
        confidence: f64,
        timestamp: DateTime<Utc>,
    },
    PatternDetected {
        symbol: String,
        pattern: Pattern,
        timestamp: DateTime<Utc>,
    },
    RiskAlert {
        symbol: String,
        risk_type: RiskAlertType,
        severity: f64,
        timestamp: DateTime<Utc>,
    },
    LiquidityAlert {
        symbol: String,
        liquidity_level: f64,
        threshold: f64,
        timestamp: DateTime<Utc>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskAlertType {
    HighVolatility,
    LiquidityCrisis,
    TailRisk,
    DrawdownWarning,
    VolatilityRegimeChange,
}

/// Model feedback for online learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFeedback {
    pub whale_feedback: WhaleFeedback,
    pub regime_feedback: RegimeFeedback,
    pub pattern_feedback: PatternFeedback,
    pub prediction_feedback: PredictionFeedback,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleFeedback {
    pub true_positives: u32,
    pub false_positives: u32,
    pub false_negatives: u32,
    pub parameter_adjustments: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeFeedback {
    pub correct_classifications: u32,
    pub incorrect_classifications: u32,
    pub regime_transition_accuracy: f64,
    pub model_updates: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternFeedback {
    pub pattern_accuracy: HashMap<PatternType, f64>,
    pub breakout_success_rate: f64,
    pub false_pattern_rate: f64,
    pub weight_adjustments: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionFeedback {
    pub price_prediction_accuracy: f64,
    pub volatility_prediction_accuracy: f64,
    pub directional_accuracy: f64,
    pub model_performance_metrics: HashMap<String, f64>,
}