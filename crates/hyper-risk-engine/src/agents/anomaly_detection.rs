//! Anomaly detection agent for identifying market irregularities.
//!
//! Operates in the medium path (<1ms) using statistical methods to detect
//! anomalous market behavior, price movements, and trading patterns.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

use parking_lot::RwLock;

use crate::core::types::{
    MarketRegime, Portfolio, Price, RiskDecision, Symbol, Timestamp,
};
use crate::core::error::Result;

use super::base::{Agent, AgentConfig, AgentId, AgentStats, AgentStatus};

/// Configuration for the anomaly detection agent.
#[derive(Debug, Clone)]
pub struct AnomalyDetectionConfig {
    /// Base agent configuration.
    pub base: AgentConfig,
    /// Z-score threshold for anomaly detection.
    pub zscore_threshold: f64,
    /// Lookback window for statistics calculation.
    pub lookback_window: usize,
    /// Minimum observations before detecting anomalies.
    pub min_observations: usize,
    /// Enable volume anomaly detection.
    pub detect_volume_anomalies: bool,
    /// Enable price spike detection.
    pub detect_price_spikes: bool,
}

impl Default for AnomalyDetectionConfig {
    fn default() -> Self {
        Self {
            base: AgentConfig {
                name: "anomaly_detection_agent".to_string(),
                enabled: true,
                priority: 3,
                max_latency_us: 1000, // 1ms
                verbose: false,
            },
            zscore_threshold: 3.0,
            lookback_window: 100,
            min_observations: 20,
            detect_volume_anomalies: true,
            detect_price_spikes: true,
        }
    }
}

/// Type of detected anomaly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalyType {
    /// Price spike anomaly.
    PriceSpike,
    /// Volume surge anomaly.
    VolumeSurge,
    /// Spread widening anomaly.
    SpreadWidening,
    /// Correlation breakdown.
    CorrelationBreakdown,
    /// Liquidity drought.
    LiquidityDrought,
}

/// Detected market anomaly.
#[derive(Debug, Clone)]
pub struct Anomaly {
    /// Symbol affected by anomaly.
    pub symbol: Symbol,
    /// Type of anomaly detected.
    pub anomaly_type: AnomalyType,
    /// Z-score of the anomalous observation.
    pub zscore: f64,
    /// Current value.
    pub current_value: f64,
    /// Expected value (mean).
    pub expected_value: f64,
    /// Standard deviation.
    pub std_dev: f64,
    /// Detection timestamp.
    pub detected_at: Timestamp,
    /// Severity score (0.0 to 1.0).
    pub severity: f64,
}

/// Running statistics calculator.
#[derive(Debug)]
struct RunningStats {
    values: VecDeque<f64>,
    sum: f64,
    sum_sq: f64,
    window_size: usize,
}

impl RunningStats {
    fn new(window_size: usize) -> Self {
        Self {
            values: VecDeque::with_capacity(window_size),
            sum: 0.0,
            sum_sq: 0.0,
            window_size,
        }
    }

    fn push(&mut self, value: f64) {
        if self.values.len() >= self.window_size {
            if let Some(old) = self.values.pop_front() {
                self.sum -= old;
                self.sum_sq -= old * old;
            }
        }
        self.values.push_back(value);
        self.sum += value;
        self.sum_sq += value * value;
    }

    fn mean(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.sum / self.values.len() as f64
    }

    fn std_dev(&self) -> f64 {
        if self.values.len() < 2 {
            return 0.0;
        }
        let n = self.values.len() as f64;
        let variance = (self.sum_sq / n) - (self.sum / n).powi(2);
        variance.max(0.0).sqrt()
    }

    fn zscore(&self, value: f64) -> f64 {
        let std = self.std_dev();
        if std < 1e-10 {
            return 0.0;
        }
        (value - self.mean()) / std
    }

    fn count(&self) -> usize {
        self.values.len()
    }
}

/// Anomaly detection agent.
#[derive(Debug)]
pub struct AnomalyDetectionAgent {
    config: AnomalyDetectionConfig,
    status: AtomicU8,
    stats: AgentStats,
    /// Price statistics by symbol.
    price_stats: RwLock<std::collections::HashMap<Symbol, RunningStats>>,
    /// Volume statistics by symbol.
    volume_stats: RwLock<std::collections::HashMap<Symbol, RunningStats>>,
    /// Detected anomalies.
    anomalies: RwLock<Vec<Anomaly>>,
}

impl AnomalyDetectionAgent {
    /// Create a new anomaly detection agent.
    pub fn new(config: AnomalyDetectionConfig) -> Self {
        Self {
            config,
            status: AtomicU8::new(AgentStatus::Idle as u8),
            stats: AgentStats::new(),
            price_stats: RwLock::new(std::collections::HashMap::new()),
            volume_stats: RwLock::new(std::collections::HashMap::new()),
            anomalies: RwLock::new(Vec::new()),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(AnomalyDetectionConfig::default())
    }

    /// Record a price observation and check for anomalies.
    pub fn record_price(&self, symbol: &Symbol, price: Price) -> Option<Anomaly> {
        let price_f64 = price.as_f64();
        let mut stats = self.price_stats.write();

        let running = stats
            .entry(symbol.clone())
            .or_insert_with(|| RunningStats::new(self.config.lookback_window));

        let anomaly = if running.count() >= self.config.min_observations {
            let zscore = running.zscore(price_f64);
            if zscore.abs() >= self.config.zscore_threshold && self.config.detect_price_spikes {
                Some(Anomaly {
                    symbol: symbol.clone(),
                    anomaly_type: AnomalyType::PriceSpike,
                    zscore,
                    current_value: price_f64,
                    expected_value: running.mean(),
                    std_dev: running.std_dev(),
                    detected_at: Timestamp::now(),
                    severity: (zscore.abs() / 5.0).min(1.0),
                })
            } else {
                None
            }
        } else {
            None
        };

        running.push(price_f64);

        if let Some(ref a) = anomaly {
            self.anomalies.write().push(a.clone());
        }

        anomaly
    }

    /// Record a volume observation and check for anomalies.
    pub fn record_volume(&self, symbol: &Symbol, volume: f64) -> Option<Anomaly> {
        let mut stats = self.volume_stats.write();

        let running = stats
            .entry(symbol.clone())
            .or_insert_with(|| RunningStats::new(self.config.lookback_window));

        let anomaly = if running.count() >= self.config.min_observations {
            let zscore = running.zscore(volume);
            if zscore >= self.config.zscore_threshold && self.config.detect_volume_anomalies {
                Some(Anomaly {
                    symbol: symbol.clone(),
                    anomaly_type: AnomalyType::VolumeSurge,
                    zscore,
                    current_value: volume,
                    expected_value: running.mean(),
                    std_dev: running.std_dev(),
                    detected_at: Timestamp::now(),
                    severity: (zscore / 5.0).min(1.0),
                })
            } else {
                None
            }
        } else {
            None
        };

        running.push(volume);

        if let Some(ref a) = anomaly {
            self.anomalies.write().push(a.clone());
        }

        anomaly
    }

    /// Get recent anomalies.
    pub fn get_anomalies(&self) -> Vec<Anomaly> {
        self.anomalies.read().clone()
    }

    /// Clear anomaly history.
    pub fn clear_anomalies(&self) {
        self.anomalies.write().clear();
    }

    /// Convert u8 to AgentStatus.
    fn status_from_u8(value: u8) -> AgentStatus {
        match value {
            0 => AgentStatus::Idle,
            1 => AgentStatus::Processing,
            2 => AgentStatus::Paused,
            3 => AgentStatus::Error,
            4 => AgentStatus::ShuttingDown,
            _ => AgentStatus::Error,
        }
    }
}

impl Agent for AnomalyDetectionAgent {
    fn id(&self) -> AgentId {
        AgentId::new(&self.config.base.name)
    }

    fn status(&self) -> AgentStatus {
        Self::status_from_u8(self.status.load(Ordering::Relaxed))
    }

    fn process(&self, portfolio: &Portfolio, _regime: MarketRegime) -> Result<Option<RiskDecision>> {
        let start = Instant::now();
        self.status.store(AgentStatus::Processing as u8, Ordering::Relaxed);

        // Process portfolio positions for anomaly detection
        for position in &portfolio.positions {
            self.record_price(&position.symbol, position.current_price);
        }

        let latency_ns = start.elapsed().as_nanos() as u64;
        self.stats.record_cycle(latency_ns);
        self.status.store(AgentStatus::Idle as u8, Ordering::Relaxed);

        Ok(None)
    }

    fn start(&self) -> Result<()> {
        self.status.store(AgentStatus::Idle as u8, Ordering::Relaxed);
        Ok(())
    }

    fn stop(&self) -> Result<()> {
        self.status.store(AgentStatus::ShuttingDown as u8, Ordering::Relaxed);
        Ok(())
    }

    fn pause(&self) {
        self.status.store(AgentStatus::Paused as u8, Ordering::Relaxed);
    }

    fn resume(&self) {
        self.status.store(AgentStatus::Idle as u8, Ordering::Relaxed);
    }

    fn process_count(&self) -> u64 {
        self.stats.cycles.load(Ordering::Relaxed)
    }

    fn avg_latency_ns(&self) -> u64 {
        self.stats.avg_latency_ns()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anomaly_detection_agent_creation() {
        let agent = AnomalyDetectionAgent::with_defaults();
        assert_eq!(agent.status(), AgentStatus::Idle);
        assert_eq!(agent.process_count(), 0);
    }

    #[test]
    fn test_price_anomaly_detection() {
        let mut config = AnomalyDetectionConfig::default();
        config.min_observations = 5;
        config.zscore_threshold = 2.0;

        let agent = AnomalyDetectionAgent::new(config);
        let symbol = Symbol::new("TEST");

        // Build up history with varied prices (creates std_dev > 0)
        let base_prices = [99.0, 101.0, 100.5, 99.5, 100.0, 100.2, 99.8, 100.1, 99.9, 100.3];
        for price in base_prices {
            agent.record_price(&symbol, Price::from_f64(price));
        }

        // Introduce a spike (significantly outside normal range)
        let anomaly = agent.record_price(&symbol, Price::from_f64(150.0));
        assert!(anomaly.is_some());

        let a = anomaly.unwrap();
        assert_eq!(a.anomaly_type, AnomalyType::PriceSpike);
        assert!(a.zscore > 2.0);
    }

    #[test]
    fn test_volume_anomaly_detection() {
        let mut config = AnomalyDetectionConfig::default();
        config.min_observations = 5;
        config.zscore_threshold = 2.0;

        let agent = AnomalyDetectionAgent::new(config);
        let symbol = Symbol::new("TEST");

        // Build up history with varied volume (creates std_dev > 0)
        let base_volumes = [990.0, 1010.0, 1005.0, 995.0, 1000.0, 1002.0, 998.0, 1001.0, 999.0, 1003.0];
        for volume in base_volumes {
            agent.record_volume(&symbol, volume);
        }

        // Introduce a volume surge (significantly outside normal range)
        let anomaly = agent.record_volume(&symbol, 5000.0);
        assert!(anomaly.is_some());

        let a = anomaly.unwrap();
        assert_eq!(a.anomaly_type, AnomalyType::VolumeSurge);
    }

    #[test]
    fn test_agent_lifecycle() {
        let agent = AnomalyDetectionAgent::with_defaults();

        agent.start().unwrap();
        assert_eq!(agent.status(), AgentStatus::Idle);

        agent.pause();
        assert_eq!(agent.status(), AgentStatus::Paused);

        agent.resume();
        assert_eq!(agent.status(), AgentStatus::Idle);

        agent.stop().unwrap();
        assert_eq!(agent.status(), AgentStatus::ShuttingDown);
    }
}
