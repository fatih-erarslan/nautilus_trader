//! Expert systems for hedge algorithms

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::RwLock;
use crate::{HedgeError, HedgeConfig, MarketData};
use crate::utils::Expert;

/// Expert system manager
pub struct ExpertSystem {
    /// Registered experts
    experts: Arc<RwLock<HashMap<String, Box<dyn Expert + Send + Sync>>>>,
    /// Expert performance tracking
    performance_tracker: Arc<RwLock<ExpertPerformanceTracker>>,
    /// Expert ensemble
    ensemble: Arc<RwLock<ExpertEnsemble>>,
    /// Configuration
    config: HedgeConfig,
}

impl ExpertSystem {
    /// Create new expert system
    pub fn new(config: HedgeConfig) -> Self {
        Self {
            experts: Arc::new(RwLock::new(HashMap::new())),
            performance_tracker: Arc::new(RwLock::new(ExpertPerformanceTracker::new(config.clone()))),
            ensemble: Arc::new(RwLock::new(ExpertEnsemble::new(config.clone()))),
            config,
        }
    }
    
    /// Register expert
    pub fn register_expert(&self, name: String, expert: Box<dyn Expert + Send + Sync>) -> Result<(), HedgeError> {
        let mut experts = self.experts.write().map_err(|_| HedgeError::expert("Failed to acquire lock"))?;
        
        if experts.len() >= self.config.expert_config.max_experts {
            return Err(HedgeError::expert("Maximum number of experts reached"));
        }
        
        experts.insert(name.clone(), expert);
        
        // Initialize performance tracking
        self.performance_tracker.write().map_err(|_| HedgeError::expert("Failed to acquire lock"))?.initialize_expert(&name)?;
        
        // Add to ensemble
        self.ensemble.write().map_err(|_| HedgeError::expert("Failed to acquire lock"))?.add_expert(&name)?;
        
        Ok(())
    }
    
    /// Remove expert
    pub fn remove_expert(&self, name: &str) -> Result<(), HedgeError> {
        let mut experts = self.experts.write().map_err(|_| HedgeError::expert("Failed to acquire lock"))?;
        
        if experts.remove(name).is_none() {
            return Err(HedgeError::expert(format!("Expert {} not found", name)));
        }
        
        // Remove from performance tracking
        self.performance_tracker.write().map_err(|_| HedgeError::expert("Failed to acquire lock"))?.remove_expert(name)?;
        
        // Remove from ensemble
        self.ensemble.write().map_err(|_| HedgeError::expert("Failed to acquire lock"))?.remove_expert(name)?;
        
        Ok(())
    }
    
    /// Update all experts with market data
    pub fn update_all(&self, market_data: &MarketData) -> Result<(), HedgeError> {
        let experts = self.experts.read().map_err(|_| HedgeError::expert("Failed to acquire lock"))?;
        
        for (name, expert) in experts.iter() {
            if let Err(e) = expert.update(market_data) {
                tracing::warn!("Expert {} update failed: {}", name, e);
            }
        }
        
        Ok(())
    }
    
    /// Get predictions from all experts
    pub fn get_predictions(&self) -> Result<HashMap<String, f64>, HedgeError> {
        let experts = self.experts.read().map_err(|_| HedgeError::expert("Failed to acquire lock"))?;
        let mut predictions = HashMap::new();
        
        for (name, expert) in experts.iter() {
            match expert.get_signal() {
                Ok(signal) => {
                    predictions.insert(name.clone(), signal);
                }
                Err(e) => {
                    tracing::warn!("Expert {} signal failed: {}", name, e);
                    predictions.insert(name.clone(), 0.0);
                }
            }
        }
        
        Ok(predictions)
    }
    
    /// Get expert confidences
    pub fn get_confidences(&self) -> Result<HashMap<String, f64>, HedgeError> {
        let experts = self.experts.read().map_err(|_| HedgeError::expert("Failed to acquire lock"))?;
        let mut confidences = HashMap::new();
        
        for (name, expert) in experts.iter() {
            confidences.insert(name.clone(), expert.get_confidence());
        }
        
        Ok(confidences)
    }
    
    /// Update expert performance
    pub fn update_performance(&self, actual_return: f64) -> Result<(), HedgeError> {
        let predictions = self.get_predictions()?;
        
        for (name, prediction) in predictions {
            let performance = -(prediction - actual_return).powi(2); // Negative squared error
            self.performance_tracker.write().map_err(|_| HedgeError::expert("Failed to acquire lock"))?.update_performance(&name, performance)?;
        }
        
        Ok(())
    }
    
    /// Prune poor performing experts
    pub fn prune_experts(&self) -> Result<Vec<String>, HedgeError> {
        let mut pruned_experts: Vec<String> = Vec::new();
        let performance_tracker = self.performance_tracker.read().map_err(|_| HedgeError::expert("Failed to acquire lock"))?;
        
        let expert_names: Vec<String> = self.experts.read().map_err(|_| HedgeError::expert("Failed to acquire lock"))?.keys().cloned().collect();
        
        for name in expert_names {
            if let Some(performance) = performance_tracker.get_performance(&name) {
                if performance < self.config.expert_config.pruning_threshold {
                    self.remove_expert(&name)?;
                    pruned_experts.push(name);
                }
            }
        }
        
        Ok(pruned_experts)
    }
    
    /// Get ensemble prediction
    pub fn get_ensemble_prediction(&self) -> Result<f64, HedgeError> {
        let predictions = self.get_predictions()?;
        self.ensemble.read().map_err(|_| HedgeError::expert("Failed to acquire lock"))?.get_ensemble_prediction(&predictions)
    }
    
    /// Get expert rankings
    pub fn get_expert_rankings(&self) -> Result<Vec<(String, f64)>, HedgeError> {
        self.performance_tracker.read().map_err(|_| HedgeError::expert("Failed to acquire lock"))?.get_rankings()
    }
    
    /// Get expert statistics
    pub fn get_expert_statistics(&self) -> Result<HashMap<String, ExpertStats>, HedgeError> {
        self.performance_tracker.read().map_err(|_| HedgeError::expert("Failed to acquire lock"))?.get_statistics()
    }
}

/// Expert performance tracker
#[derive(Debug, Clone)]
pub struct ExpertPerformanceTracker {
    /// Expert performance history
    performance_history: HashMap<String, VecDeque<f64>>,
    /// Expert statistics
    statistics: HashMap<String, ExpertStats>,
    /// Configuration
    config: HedgeConfig,
}

impl ExpertPerformanceTracker {
    /// Create new performance tracker
    pub fn new(config: HedgeConfig) -> Self {
        Self {
            performance_history: HashMap::new(),
            statistics: HashMap::new(),
            config,
        }
    }
    
    /// Initialize expert tracking
    pub fn initialize_expert(&mut self, name: &str) -> Result<(), HedgeError> {
        self.performance_history.insert(name.to_string(), VecDeque::new());
        self.statistics.insert(name.to_string(), ExpertStats::new());
        Ok(())
    }
    
    /// Remove expert tracking
    pub fn remove_expert(&mut self, name: &str) -> Result<(), HedgeError> {
        self.performance_history.remove(name);
        self.statistics.remove(name);
        Ok(())
    }
    
    /// Update expert performance
    pub fn update_performance(&mut self, name: &str, performance: f64) -> Result<(), HedgeError> {
        let history = self.performance_history.get_mut(name)
            .ok_or_else(|| HedgeError::expert(format!("Expert {} not found", name)))?;
        
        history.push_back(performance);
        
        // Keep only recent history
        if history.len() > self.config.expert_config.evaluation_window {
            history.pop_front();
        }
        
        // Update statistics
        self.update_statistics(name)?;
        
        Ok(())
    }
    
    /// Update expert statistics
    fn update_statistics(&mut self, name: &str) -> Result<(), HedgeError> {
        let history = self.performance_history.get(name)
            .ok_or_else(|| HedgeError::expert(format!("Expert {} not found", name)))?;
        
        let stats = self.statistics.get_mut(name)
            .ok_or_else(|| HedgeError::expert(format!("Expert {} not found", name)))?;
        
        if history.is_empty() {
            return Ok(());
        }
        
        let performances: Vec<f64> = history.iter().copied().collect();
        
        // Calculate statistics
        stats.total_predictions = performances.len();
        stats.average_performance = performances.iter().sum::<f64>() / performances.len() as f64;
        stats.best_performance = performances.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        stats.worst_performance = performances.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        // Calculate variance
        let variance = performances.iter()
            .map(|p| (p - stats.average_performance).powi(2))
            .sum::<f64>() / performances.len() as f64;
        stats.performance_volatility = variance.sqrt();
        
        // Calculate consistency (lower volatility = higher consistency)
        stats.consistency = 1.0 / (1.0 + stats.performance_volatility);
        
        // Calculate hit rate (percentage of positive performances)
        stats.hit_rate = performances.iter()
            .filter(|&&p| p > 0.0)
            .count() as f64 / performances.len() as f64;
        
        // Calculate Sharpe ratio
        stats.sharpe_ratio = if stats.performance_volatility > 0.0 {
            stats.average_performance / stats.performance_volatility
        } else {
            0.0
        };
        
        // Calculate maximum drawdown
        let mut peak = f64::NEG_INFINITY;
        let mut max_drawdown = 0.0;
        let mut cumulative_performance = 0.0;
        
        for &performance in &performances {
            cumulative_performance += performance;
            if cumulative_performance > peak {
                peak = cumulative_performance;
            }
            let drawdown = peak - cumulative_performance;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }
        
        stats.max_drawdown = max_drawdown;
        
        // Calculate overall score
        stats.overall_score = stats.average_performance * stats.consistency * stats.hit_rate;
        
        stats.last_updated = chrono::Utc::now();
        
        Ok(())
    }
    
    /// Get expert performance
    pub fn get_performance(&self, name: &str) -> Option<f64> {
        self.statistics.get(name).map(|stats| stats.overall_score)
    }
    
    /// Get expert rankings
    pub fn get_rankings(&self) -> Result<Vec<(String, f64)>, HedgeError> {
        let mut rankings: Vec<(String, f64)> = self.statistics.iter()
            .map(|(name, stats)| (name.clone(), stats.overall_score))
            .collect();
        
        rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        Ok(rankings)
    }
    
    /// Get expert statistics
    pub fn get_statistics(&self) -> Result<HashMap<String, ExpertStats>, HedgeError> {
        Ok(self.statistics.clone())
    }
}

/// Expert statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExpertStats {
    /// Total predictions made
    pub total_predictions: usize,
    /// Average performance
    pub average_performance: f64,
    /// Best performance
    pub best_performance: f64,
    /// Worst performance
    pub worst_performance: f64,
    /// Performance volatility
    pub performance_volatility: f64,
    /// Consistency score
    pub consistency: f64,
    /// Hit rate
    pub hit_rate: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Overall score
    pub overall_score: f64,
    /// Last updated
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

impl ExpertStats {
    /// Create new expert stats
    pub fn new() -> Self {
        Self {
            total_predictions: 0,
            average_performance: 0.0,
            best_performance: 0.0,
            worst_performance: 0.0,
            performance_volatility: 0.0,
            consistency: 0.0,
            hit_rate: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            overall_score: 0.0,
            last_updated: chrono::Utc::now(),
        }
    }
}

/// Expert ensemble
#[derive(Debug, Clone)]
pub struct ExpertEnsemble {
    /// Expert weights
    weights: HashMap<String, f64>,
    /// Ensemble method
    method: EnsembleMethod,
    /// Configuration
    config: HedgeConfig,
}

impl ExpertEnsemble {
    /// Create new expert ensemble
    pub fn new(config: HedgeConfig) -> Self {
        Self {
            weights: HashMap::new(),
            method: EnsembleMethod::WeightedAverage,
            config,
        }
    }
    
    /// Add expert to ensemble
    pub fn add_expert(&mut self, name: &str) -> Result<(), HedgeError> {
        self.weights.insert(name.to_string(), 1.0);
        self.normalize_weights()?;
        Ok(())
    }
    
    /// Remove expert from ensemble
    pub fn remove_expert(&mut self, name: &str) -> Result<(), HedgeError> {
        self.weights.remove(name);
        self.normalize_weights()?;
        Ok(())
    }
    
    /// Update expert weights
    pub fn update_weights(&mut self, performance: &HashMap<String, f64>) -> Result<(), HedgeError> {
        for (name, perf) in performance {
            if let Some(weight) = self.weights.get_mut(name) {
                *weight = perf.max(0.0); // Only positive weights
            }
        }
        
        self.normalize_weights()?;
        Ok(())
    }
    
    /// Normalize weights
    fn normalize_weights(&mut self) -> Result<(), HedgeError> {
        let total_weight: f64 = self.weights.values().sum();
        
        if total_weight > 0.0 {
            for weight in self.weights.values_mut() {
                *weight /= total_weight;
            }
        }
        
        Ok(())
    }
    
    /// Get ensemble prediction
    pub fn get_ensemble_prediction(&self, predictions: &HashMap<String, f64>) -> Result<f64, HedgeError> {
        match self.method {
            EnsembleMethod::WeightedAverage => {
                let mut weighted_sum = 0.0;
                let mut total_weight = 0.0;
                
                for (name, prediction) in predictions {
                    if let Some(weight) = self.weights.get(name) {
                        weighted_sum += prediction * weight;
                        total_weight += weight;
                    }
                }
                
                if total_weight > 0.0 {
                    Ok(weighted_sum / total_weight)
                } else {
                    Ok(0.0)
                }
            }
            EnsembleMethod::Median => {
                let mut values: Vec<f64> = predictions.values().copied().collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                
                if values.is_empty() {
                    Ok(0.0)
                } else if values.len() % 2 == 0 {
                    let mid = values.len() / 2;
                    Ok((values[mid - 1] + values[mid]) / 2.0)
                } else {
                    Ok(values[values.len() / 2])
                }
            }
            EnsembleMethod::BestExpert => {
                let mut best_prediction = 0.0;
                let mut best_weight = 0.0;
                
                for (name, prediction) in predictions {
                    if let Some(weight) = self.weights.get(name) {
                        if *weight > best_weight {
                            best_weight = *weight;
                            best_prediction = *prediction;
                        }
                    }
                }
                
                Ok(best_prediction)
            }
            EnsembleMethod::Stacking => {
                // Simplified stacking - would need meta-learner in practice
                let predictions_vec: Vec<f64> = predictions.values().copied().collect();
                let mean = predictions_vec.iter().sum::<f64>() / predictions_vec.len() as f64;
                Ok(mean)
            }
        }
    }
}

/// Ensemble method
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum EnsembleMethod {
    WeightedAverage,
    Median,
    BestExpert,
    Stacking,
}

/// Trend following expert
#[derive(Debug, Clone)]
pub struct TrendFollowingExpert {
    /// Expert name
    name: String,
    /// Price history
    price_history: VecDeque<f64>,
    /// Moving average period
    ma_period: usize,
    /// Trend threshold
    trend_threshold: f64,
    /// Current signal
    current_signal: f64,
    /// Confidence
    confidence: f64,
}

impl TrendFollowingExpert {
    /// Create new trend following expert
    pub fn new(name: String, ma_period: usize, trend_threshold: f64) -> Self {
        Self {
            name,
            price_history: VecDeque::new(),
            ma_period,
            trend_threshold,
            current_signal: 0.0,
            confidence: 0.0,
        }
    }
    
    /// Calculate moving average
    fn calculate_ma(&self) -> Option<f64> {
        if self.price_history.len() < self.ma_period {
            return None;
        }
        
        let sum: f64 = self.price_history.iter().rev().take(self.ma_period).sum();
        Some(sum / self.ma_period as f64)
    }
    
    /// Calculate trend strength
    fn calculate_trend_strength(&self) -> f64 {
        if self.price_history.len() < 2 {
            return 0.0;
        }
        
        let current_price = *self.price_history.back().unwrap();
        let previous_price = self.price_history[self.price_history.len() - 2];
        
        (current_price - previous_price) / previous_price
    }
}

impl Expert for TrendFollowingExpert {
    fn update(&self, _market_data: &MarketData) -> Result<(), HedgeError> {
        // This would need to be mutable in practice
        // For now, we'll simulate the update
        Ok(())
    }
    
    fn get_signal(&self) -> Result<f64, HedgeError> {
        Ok(self.current_signal)
    }
    
    fn get_name(&self) -> &str {
        &self.name
    }
    
    fn get_confidence(&self) -> f64 {
        self.confidence
    }
}

/// Mean reversion expert
#[derive(Debug, Clone)]
pub struct MeanReversionExpert {
    /// Expert name
    name: String,
    /// Price history
    price_history: VecDeque<f64>,
    /// Lookback period
    lookback_period: usize,
    /// Reversion threshold
    reversion_threshold: f64,
    /// Current signal
    current_signal: f64,
    /// Confidence
    confidence: f64,
}

impl MeanReversionExpert {
    /// Create new mean reversion expert
    pub fn new(name: String, lookback_period: usize, reversion_threshold: f64) -> Self {
        Self {
            name,
            price_history: VecDeque::new(),
            lookback_period,
            reversion_threshold,
            current_signal: 0.0,
            confidence: 0.0,
        }
    }
    
    /// Calculate z-score
    fn calculate_zscore(&self) -> Option<f64> {
        if self.price_history.len() < self.lookback_period {
            return None;
        }
        
        let recent_prices: Vec<f64> = self.price_history.iter()
            .rev()
            .take(self.lookback_period)
            .copied()
            .collect();
        
        let mean = recent_prices.iter().sum::<f64>() / recent_prices.len() as f64;
        let variance = recent_prices.iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>() / recent_prices.len() as f64;
        
        let std_dev = variance.sqrt();
        
        if std_dev > 0.0 {
            let current_price = *self.price_history.back().unwrap();
            Some((current_price - mean) / std_dev)
        } else {
            None
        }
    }
}

impl Expert for MeanReversionExpert {
    fn update(&self, _market_data: &MarketData) -> Result<(), HedgeError> {
        // This would need to be mutable in practice
        Ok(())
    }
    
    fn get_signal(&self) -> Result<f64, HedgeError> {
        Ok(self.current_signal)
    }
    
    fn get_name(&self) -> &str {
        &self.name
    }
    
    fn get_confidence(&self) -> f64 {
        self.confidence
    }
}

/// Volatility expert
#[derive(Debug, Clone)]
pub struct VolatilityExpert {
    /// Expert name
    name: String,
    /// Return history
    return_history: VecDeque<f64>,
    /// Volatility window
    volatility_window: usize,
    /// Current signal
    current_signal: f64,
    /// Confidence
    confidence: f64,
}

impl VolatilityExpert {
    /// Create new volatility expert
    pub fn new(name: String, volatility_window: usize) -> Self {
        Self {
            name,
            return_history: VecDeque::new(),
            volatility_window,
            current_signal: 0.0,
            confidence: 0.0,
        }
    }
    
    /// Calculate volatility
    fn calculate_volatility(&self) -> Option<f64> {
        if self.return_history.len() < self.volatility_window {
            return None;
        }
        
        let recent_returns: Vec<f64> = self.return_history.iter()
            .rev()
            .take(self.volatility_window)
            .copied()
            .collect();
        
        let mean = recent_returns.iter().sum::<f64>() / recent_returns.len() as f64;
        let variance = recent_returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / recent_returns.len() as f64;
        
        Some(variance.sqrt())
    }
}

impl Expert for VolatilityExpert {
    fn update(&self, _market_data: &MarketData) -> Result<(), HedgeError> {
        // This would need to be mutable in practice
        Ok(())
    }
    
    fn get_signal(&self) -> Result<f64, HedgeError> {
        Ok(self.current_signal)
    }
    
    fn get_name(&self) -> &str {
        &self.name
    }
    
    fn get_confidence(&self) -> f64 {
        self.confidence
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expert_system_creation() {
        let config = HedgeConfig::default();
        let expert_system = ExpertSystem::new(config);
        
        assert_eq!(expert_system.experts.read().len(), 0);
    }

    #[test]
    fn test_expert_performance_tracker() {
        let config = HedgeConfig::default();
        let mut tracker = ExpertPerformanceTracker::new(config);
        
        tracker.initialize_expert("test_expert").unwrap();
        tracker.update_performance("test_expert", 0.05).unwrap();
        
        let performance = tracker.get_performance("test_expert").unwrap();
        assert!(performance > 0.0);
    }

    #[test]
    fn test_expert_ensemble() {
        let config = HedgeConfig::default();
        let mut ensemble = ExpertEnsemble::new(config);
        
        ensemble.add_expert("expert1").unwrap();
        ensemble.add_expert("expert2").unwrap();
        
        let mut predictions = HashMap::new();
        predictions.insert("expert1".to_string(), 0.05);
        predictions.insert("expert2".to_string(), -0.02);
        
        let ensemble_prediction = ensemble.get_ensemble_prediction(&predictions).unwrap();
        assert!(ensemble_prediction.abs() < 0.1);
    }

    #[test]
    fn test_trend_following_expert() {
        let expert = TrendFollowingExpert::new("trend_expert".to_string(), 10, 0.02);
        
        assert_eq!(expert.get_name(), "trend_expert");
        assert_eq!(expert.get_confidence(), 0.0);
        
        let signal = expert.get_signal().unwrap();
        assert_eq!(signal, 0.0);
    }
}