//! Main Panarchy analyzer implementation

use crate::pcr::{calculate_pcr_batch, calculate_pcr_components, calculate_adx, FastPCRCalculator};
use crate::phase::{PhaseIdentifier, calculate_regime_score, identify_phases_batch};
use crate::types::*;
use crate::{PanarchyError, Result};
use std::time::Instant;

/// Main Panarchy analyzer with sub-microsecond performance
pub struct PanarchyAnalyzer {
    params: PanarchyParameters,
    phase_identifier: PhaseIdentifier,
    pcr_calculator: Option<FastPCRCalculator>,
    cache_enabled: bool,
}

impl PanarchyAnalyzer {
    /// Create a new analyzer with default parameters
    pub fn new() -> Self {
        Self::with_params(PanarchyParameters::default())
    }
    
    /// Create a new analyzer with custom parameters
    pub fn with_params(params: PanarchyParameters) -> Self {
        Self {
            phase_identifier: PhaseIdentifier::new(params.clone()),
            params,
            pcr_calculator: None,
            cache_enabled: true,
        }
    }
    
    /// Enable or disable caching
    pub fn set_cache_enabled(&mut self, enabled: bool) {
        self.cache_enabled = enabled;
        if !enabled {
            self.pcr_calculator = None;
        }
    }
    
    /// Main analysis entry point - compatible with CDFA server
    pub fn analyze(&mut self, prices: &[f64], volumes: &[f64]) -> Result<PanarchyResult> {
        let start = Instant::now();
        
        // Validate inputs
        if prices.is_empty() {
            return Err(PanarchyError::InsufficientData {
                required: 1,
                actual: 0,
            });
        }
        
        if prices.len() != volumes.len() {
            return Err(PanarchyError::InvalidParameters {
                message: "Prices and volumes must have the same length".to_string(),
            });
        }
        
        // Convert to market data
        let market_data: Vec<MarketData> = prices.iter()
            .zip(volumes.iter())
            .map(|(&price, &volume)| MarketData {
                open: price,
                high: price,
                low: price,
                close: price,
                volume,
            })
            .collect();
        
        // Perform analysis
        let result = self.analyze_market_data(&market_data)?;
        
        let computation_time_ns = start.elapsed().as_nanos() as u64;
        
        Ok(PanarchyResult {
            computation_time_ns,
            ..result
        })
    }
    
    /// Analyze market data with full OHLCV information
    pub fn analyze_market_data(&mut self, market_data: &[MarketData]) -> Result<PanarchyResult> {
        let n = market_data.len();
        let period = self.params.adx_period;
        
        if n < period + 1 {
            return Err(PanarchyError::InsufficientData {
                required: period + 1,
                actual: n,
            });
        }
        
        // Calculate PCR components
        let pcr_components = calculate_pcr_batch(market_data, period, self.params.autocorr_lag)?;
        
        // Get latest PCR
        let latest_pcr = pcr_components.last()
            .ok_or_else(|| PanarchyError::ComputationError {
                message: "Failed to calculate PCR components".to_string(),
            })?;
        
        // Identify phase
        let phase = self.phase_identifier.identify_phase(latest_pcr);
        
        // Get phase scores
        let mut phase_scores = latest_pcr.phase_scores();
        phase_scores.normalize();
        
        // Calculate regime score
        let regime_score = self.calculate_full_regime_score(
            market_data,
            &phase,
            &phase_scores,
        )?;
        
        // Calculate signal (0-1 range)
        let signal = regime_score / 100.0;
        
        // Calculate confidence based on phase stability
        let confidence = self.calculate_confidence(&phase, &phase_scores);
        
        Ok(PanarchyResult {
            phase,
            pcr: *latest_pcr,
            phase_scores,
            regime_score,
            confidence,
            signal,
            data_points: n,
            computation_time_ns: 0, // Will be set by caller
        })
    }
    
    /// Calculate full regime score with all indicators
    fn calculate_full_regime_score(
        &self,
        market_data: &[MarketData],
        phase: &MarketPhase,
        phase_scores: &PhaseScores,
    ) -> Result<f64> {
        // Extract price arrays
        let closes: Vec<f64> = market_data.iter().map(|d| d.close).collect();
        let highs: Vec<f64> = market_data.iter().map(|d| d.high).collect();
        let lows: Vec<f64> = market_data.iter().map(|d| d.low).collect();
        
        // Calculate ADX
        let adx_values = calculate_adx(&highs, &lows, &closes, self.params.adx_period)?;
        let latest_adx = adx_values.last().copied().unwrap_or(25.0);
        
        // Calculate volatility regime (simplified)
        let returns: Vec<f64> = closes.windows(2)
            .map(|w| if w[0] > 0.0 { (w[1] - w[0]) / w[0] } else { 0.0 })
            .collect();
        
        let volatility = if returns.len() >= self.params.adx_period {
            let recent_returns = &returns[returns.len() - self.params.adx_period..];
            let mean = recent_returns.iter().sum::<f64>() / recent_returns.len() as f64;
            let variance = recent_returns.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / recent_returns.len() as f64;
            variance.sqrt()
        } else {
            0.5
        };
        
        // Default values for missing indicators
        let soc_regime = "normal";
        let soc_fragility = 0.5;
        
        Ok(calculate_regime_score(
            *phase,
            phase_scores,
            soc_regime,
            volatility,
            soc_fragility,
            latest_adx,
        ))
    }
    
    /// Calculate confidence based on phase stability and score distribution
    fn calculate_confidence(&self, phase: &MarketPhase, phase_scores: &PhaseScores) -> f64 {
        // Base confidence from phase
        let phase_confidence = match phase {
            MarketPhase::Growth => 0.8,
            MarketPhase::Conservation => 0.9,
            MarketPhase::Release => 0.7,
            MarketPhase::Reorganization => 0.6,
            MarketPhase::Unknown => 0.5,
        };
        
        // Calculate score stability (how dominant is the winning phase)
        let scores = [
            phase_scores.growth,
            phase_scores.conservation,
            phase_scores.release,
            phase_scores.reorganization,
        ];
        
        let max_score = scores.iter().cloned().fold(0.0, f64::max);
        let second_max = scores.iter()
            .filter(|&&s| s < max_score - 1e-10)
            .cloned()
            .fold(0.0, f64::max);
        
        let dominance = if max_score > 0.0 {
            (max_score - second_max) / max_score
        } else {
            0.0
        };
        
        // Combine phase confidence with dominance
        let confidence = phase_confidence * 0.6 + dominance * 0.4;
        confidence.clamp(0.1, 1.0)
    }
    
    /// Calculate PCR components for a time series
    pub fn calculate_pcr(&self, prices: &[f64], period: usize) -> Result<Vec<PCRComponents>> {
        // Calculate returns
        let mut returns = vec![0.0];
        for i in 1..prices.len() {
            if prices[i - 1] > 0.0 {
                returns.push((prices[i] / prices[i - 1]).ln());
            } else {
                returns.push(0.0);
            }
        }
        
        // Simple volatility calculation
        let mut volatilities = vec![0.0; prices.len()];
        for i in period..prices.len() {
            let window = &returns[i - period..i];
            let mean = window.iter().sum::<f64>() / period as f64;
            let variance = window.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / period as f64;
            volatilities[i] = variance.sqrt();
        }
        
        // Normalize volatilities
        let max_vol = volatilities.iter().cloned().fold(0.0, f64::max);
        if max_vol > 0.0 {
            for vol in &mut volatilities {
                *vol /= max_vol;
            }
        }
        
        calculate_pcr_components(&prices, &returns, &volatilities, period, self.params.autocorr_lag)
    }
    
    /// Identify regime for a series of PCR components
    pub fn identify_regime(&mut self, pcr_components: &[PCRComponents]) -> Vec<MarketPhase> {
        pcr_components.iter()
            .map(|pcr| self.phase_identifier.identify_phase(pcr))
            .collect()
    }
    
    /// Get current parameters
    pub fn params(&self) -> &PanarchyParameters {
        &self.params
    }
    
    /// Update parameters
    pub fn set_params(&mut self, params: PanarchyParameters) {
        self.params = params.clone();
        self.phase_identifier = PhaseIdentifier::new(params);
        self.pcr_calculator = None; // Reset calculator with new params
    }
}

impl Default for PanarchyAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch analyzer for processing multiple time series efficiently
pub struct BatchPanarchyAnalyzer {
    analyzers: Vec<PanarchyAnalyzer>,
    params: PanarchyParameters,
}

impl BatchPanarchyAnalyzer {
    pub fn new(batch_size: usize) -> Self {
        let params = PanarchyParameters::default();
        let analyzers = (0..batch_size)
            .map(|_| PanarchyAnalyzer::with_params(params.clone()))
            .collect();
        
        Self { analyzers, params }
    }
    
    pub fn analyze_batch(
        &mut self,
        price_series: &[Vec<f64>],
        volume_series: &[Vec<f64>],
    ) -> Vec<Result<PanarchyResult>> {
        price_series.iter()
            .zip(volume_series.iter())
            .zip(self.analyzers.iter_mut())
            .map(|((prices, volumes), analyzer)| {
                analyzer.analyze(prices, volumes)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_analyzer_creation() {
        let analyzer = PanarchyAnalyzer::new();
        assert_eq!(analyzer.params().adx_period, 14);
    }
    
    #[test]
    fn test_basic_analysis() {
        let mut analyzer = PanarchyAnalyzer::new();
        
        // Generate test data
        let mut prices = vec![];
        let mut volumes = vec![];
        
        for i in 0..100 {
            let price = 100.0 + (i as f64 * 0.1).sin() * 10.0;
            prices.push(price);
            volumes.push(1000.0 + (i as f64 * 0.2).cos() * 100.0);
        }
        
        let result = analyzer.analyze(&prices, &volumes).unwrap();
        
        assert!(result.signal >= 0.0 && result.signal <= 1.0);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert_eq!(result.data_points, 100);
    }
    
    #[test]
    fn test_insufficient_data() {
        let mut analyzer = PanarchyAnalyzer::new();
        let prices = vec![100.0, 101.0];
        let volumes = vec![1000.0, 1100.0];
        
        let result = analyzer.analyze(&prices, &volumes);
        assert!(matches!(result, Err(PanarchyError::InsufficientData { .. })));
    }
    
    #[test]
    fn test_batch_analyzer() {
        let mut batch_analyzer = BatchPanarchyAnalyzer::new(3);
        
        let price_series = vec![
            (0..50).map(|i| 100.0 + i as f64).collect(),
            (0..50).map(|i| 200.0 - i as f64).collect(),
            (0..50).map(|i| 150.0 + (i as f64 * 0.1).sin() * 10.0).collect(),
        ];
        
        let volume_series = vec![
            vec![1000.0; 50],
            vec![2000.0; 50],
            vec![1500.0; 50],
        ];
        
        let results = batch_analyzer.analyze_batch(&price_series, &volume_series);
        assert_eq!(results.len(), 3);
        
        for result in results {
            assert!(result.is_ok());
        }
    }
}