//! Correlation Analysis Module
//!
//! Advanced correlation analysis for market factors and quantum-enhanced detection.

use crate::core::{QarResult, FactorMap, StandardFactors};
use crate::error::QarError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Correlation analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationResult {
    /// Correlation score (0.0 to 1.0)
    pub score: f64,
    /// Confidence in correlation assessment
    pub confidence: f64,
    /// Pairwise correlations
    pub pairwise_correlations: HashMap<String, f64>,
    /// Principal components
    pub principal_components: PrincipalComponents,
    /// Correlation regime
    pub regime: CorrelationRegime,
    /// Dynamic correlations
    pub dynamic_correlations: DynamicCorrelations,
}

/// Principal component analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrincipalComponents {
    /// Eigenvalues
    pub eigenvalues: Vec<f64>,
    /// Explained variance ratios
    pub explained_variance: Vec<f64>,
    /// Component loadings
    pub loadings: HashMap<String, Vec<f64>>,
    /// First principal component score
    pub pc1_score: f64,
}

/// Correlation regime enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CorrelationRegime {
    HighCorrelation,    // Strong correlations across factors
    LowCorrelation,     // Weak correlations, independent factors
    MixedCorrelation,   // Some strong, some weak correlations
    Transitional,       // Changing correlation structure
}

/// Dynamic correlation tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicCorrelations {
    /// Rolling correlation windows
    pub rolling_correlations: HashMap<String, Vec<f64>>,
    /// Correlation stability
    pub stability: f64,
    /// Breakpoint detection
    pub breakpoints: Vec<CorrelationBreakpoint>,
}

/// Correlation breakpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationBreakpoint {
    /// Timestamp of breakpoint
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Factor pair affected
    pub factor_pair: String,
    /// Correlation before break
    pub correlation_before: f64,
    /// Correlation after break
    pub correlation_after: f64,
    /// Breakpoint significance
    pub significance: f64,
}

/// Correlation analyzer
pub struct CorrelationAnalyzer {
    config: super::AnalysisConfig,
    correlation_params: CorrelationParameters,
    factor_history: HashMap<String, Vec<f64>>,
    history: Vec<CorrelationResult>,
}

/// Correlation analysis parameters
#[derive(Debug, Clone)]
pub struct CorrelationParameters {
    /// Rolling window size for dynamic correlations
    pub rolling_window: usize,
    /// Minimum correlation threshold for significance
    pub min_correlation_threshold: f64,
    /// PCA variance threshold
    pub pca_variance_threshold: f64,
    /// Breakpoint detection sensitivity
    pub breakpoint_sensitivity: f64,
}

impl Default for CorrelationParameters {
    fn default() -> Self {
        Self {
            rolling_window: 20,
            min_correlation_threshold: 0.3,
            pca_variance_threshold: 0.95,
            breakpoint_sensitivity: 0.1,
        }
    }
}

impl CorrelationAnalyzer {
    /// Create a new correlation analyzer
    pub fn new(config: super::AnalysisConfig) -> QarResult<Self> {
        Ok(Self {
            config,
            correlation_params: CorrelationParameters::default(),
            factor_history: HashMap::new(),
            history: Vec::new(),
        })
    }

    /// Analyze correlations from market factors
    pub async fn analyze(&mut self, factors: &FactorMap) -> QarResult<CorrelationResult> {
        // Update factor history
        self.update_factor_history(factors)?;

        // Calculate pairwise correlations
        let pairwise_correlations = self.calculate_pairwise_correlations()?;
        
        // Perform PCA
        let principal_components = self.perform_pca()?;
        
        // Determine correlation regime
        let regime = self.determine_correlation_regime(&pairwise_correlations);
        
        // Calculate dynamic correlations
        let dynamic_correlations = self.calculate_dynamic_correlations()?;
        
        // Calculate overall score and confidence
        let score = self.calculate_correlation_score(&pairwise_correlations);
        let confidence = self.calculate_correlation_confidence(&pairwise_correlations, &dynamic_correlations);

        let result = CorrelationResult {
            score,
            confidence,
            pairwise_correlations,
            principal_components,
            regime,
            dynamic_correlations,
        };

        // Store in history
        self.add_to_history(result.clone());

        Ok(result)
    }

    /// Update factor history with new data
    fn update_factor_history(&mut self, factors: &FactorMap) -> QarResult<()> {
        for factor in &[
            StandardFactors::Trend,
            StandardFactors::Momentum,
            StandardFactors::Volatility,
            StandardFactors::Volume,
            StandardFactors::Sentiment,
            StandardFactors::Liquidity,
            StandardFactors::Risk,
            StandardFactors::Efficiency,
        ] {
            let factor_name = factor.to_string();
            let value = factors.get_factor(factor)?;
            
            self.factor_history
                .entry(factor_name)
                .or_insert_with(Vec::new)
                .push(value);
        }

        // Maintain maximum history length
        for values in self.factor_history.values_mut() {
            if values.len() > self.config.max_history {
                values.remove(0);
            }
        }

        Ok(())
    }

    /// Calculate pairwise correlations between factors
    fn calculate_pairwise_correlations(&self) -> QarResult<HashMap<String, f64>> {
        let mut correlations = HashMap::new();
        
        let factor_names: Vec<String> = self.factor_history.keys().cloned().collect();
        
        for i in 0..factor_names.len() {
            for j in (i + 1)..factor_names.len() {
                let factor1 = &factor_names[i];
                let factor2 = &factor_names[j];
                
                if let (Some(values1), Some(values2)) = (
                    self.factor_history.get(factor1),
                    self.factor_history.get(factor2),
                ) {
                    let correlation = self.calculate_correlation(values1, values2)?;
                    let pair_key = format!("{}_{}", factor1, factor2);
                    correlations.insert(pair_key, correlation);
                }
            }
        }

        Ok(correlations)
    }

    /// Calculate correlation between two series
    fn calculate_correlation(&self, series1: &[f64], series2: &[f64]) -> QarResult<f64> {
        if series1.len() != series2.len() || series1.len() < 2 {
            return Ok(0.0);
        }

        let n = series1.len() as f64;
        let mean1 = series1.iter().sum::<f64>() / n;
        let mean2 = series2.iter().sum::<f64>() / n;

        let numerator: f64 = series1.iter().zip(series2.iter())
            .map(|(x1, x2)| (x1 - mean1) * (x2 - mean2))
            .sum();

        let sum_sq1: f64 = series1.iter().map(|x| (x - mean1).powi(2)).sum();
        let sum_sq2: f64 = series2.iter().map(|x| (x - mean2).powi(2)).sum();

        let denominator = (sum_sq1 * sum_sq2).sqrt();

        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    /// Perform Principal Component Analysis
    fn perform_pca(&self) -> QarResult<PrincipalComponents> {
        if self.factor_history.is_empty() {
            return Ok(PrincipalComponents {
                eigenvalues: Vec::new(),
                explained_variance: Vec::new(),
                loadings: HashMap::new(),
                pc1_score: 0.0,
            });
        }

        // Create data matrix
        let factor_names: Vec<String> = self.factor_history.keys().cloned().collect();
        let n_factors = factor_names.len();
        let n_observations = self.factor_history.values().next().unwrap().len();

        if n_observations < 2 || n_factors < 2 {
            return Ok(PrincipalComponents {
                eigenvalues: Vec::new(),
                explained_variance: Vec::new(),
                loadings: HashMap::new(),
                pc1_score: 0.0,
            });
        }

        // Calculate covariance matrix (simplified implementation)
        let mut covariance_matrix = vec![vec![0.0; n_factors]; n_factors];
        
        for i in 0..n_factors {
            for j in 0..n_factors {
                let series1 = &self.factor_history[&factor_names[i]];
                let series2 = &self.factor_history[&factor_names[j]];
                
                if i == j {
                    covariance_matrix[i][j] = self.calculate_variance(series1);
                } else {
                    covariance_matrix[i][j] = self.calculate_covariance(series1, series2);
                }
            }
        }

        // Simplified eigenvalue calculation (using trace and determinant for 2x2 case)
        let eigenvalues = if n_factors == 2 {
            self.calculate_2x2_eigenvalues(&covariance_matrix)
        } else {
            // For larger matrices, use simplified approach
            let trace = (0..n_factors).map(|i| covariance_matrix[i][i]).sum::<f64>();
            vec![trace / n_factors as f64; n_factors] // Equal eigenvalues approximation
        };

        // Calculate explained variance
        let total_variance: f64 = eigenvalues.iter().sum();
        let explained_variance: Vec<f64> = if total_variance > 0.0 {
            eigenvalues.iter().map(|&ev| ev / total_variance).collect()
        } else {
            vec![1.0 / n_factors as f64; n_factors]
        };

        // Simplified loadings calculation
        let mut loadings = HashMap::new();
        for (i, factor_name) in factor_names.iter().enumerate() {
            let loading = if !eigenvalues.is_empty() {
                (eigenvalues[0] / total_variance).sqrt()
            } else {
                1.0 / (n_factors as f64).sqrt()
            };
            loadings.insert(factor_name.clone(), vec![loading]);
        }

        let pc1_score = explained_variance.first().copied().unwrap_or(0.0);

        Ok(PrincipalComponents {
            eigenvalues,
            explained_variance,
            loadings,
            pc1_score,
        })
    }

    /// Calculate variance of a series
    fn calculate_variance(&self, series: &[f64]) -> f64 {
        if series.len() < 2 {
            return 0.0;
        }

        let mean = series.iter().sum::<f64>() / series.len() as f64;
        let variance = series.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (series.len() - 1) as f64;
        
        variance
    }

    /// Calculate covariance between two series
    fn calculate_covariance(&self, series1: &[f64], series2: &[f64]) -> f64 {
        if series1.len() != series2.len() || series1.len() < 2 {
            return 0.0;
        }

        let n = series1.len() as f64;
        let mean1 = series1.iter().sum::<f64>() / n;
        let mean2 = series2.iter().sum::<f64>() / n;

        let covariance = series1.iter().zip(series2.iter())
            .map(|(x1, x2)| (x1 - mean1) * (x2 - mean2))
            .sum::<f64>() / (n - 1.0);

        covariance
    }

    /// Calculate eigenvalues for 2x2 matrix
    fn calculate_2x2_eigenvalues(&self, matrix: &[Vec<f64>]) -> Vec<f64> {
        if matrix.len() != 2 || matrix[0].len() != 2 {
            return vec![0.0, 0.0];
        }

        let a = matrix[0][0];
        let b = matrix[0][1];
        let c = matrix[1][0];
        let d = matrix[1][1];

        let trace = a + d;
        let determinant = a * d - b * c;
        let discriminant = trace * trace - 4.0 * determinant;

        if discriminant < 0.0 {
            vec![trace / 2.0, trace / 2.0]
        } else {
            let sqrt_discriminant = discriminant.sqrt();
            let lambda1 = (trace + sqrt_discriminant) / 2.0;
            let lambda2 = (trace - sqrt_discriminant) / 2.0;
            vec![lambda1.max(lambda2), lambda1.min(lambda2)]
        }
    }

    /// Determine correlation regime
    fn determine_correlation_regime(&self, correlations: &HashMap<String, f64>) -> CorrelationRegime {
        if correlations.is_empty() {
            return CorrelationRegime::LowCorrelation;
        }

        let abs_correlations: Vec<f64> = correlations.values().map(|&c| c.abs()).collect();
        let avg_correlation = abs_correlations.iter().sum::<f64>() / abs_correlations.len() as f64;
        
        let high_corr_count = abs_correlations.iter().filter(|&&c| c > 0.7).count();
        let low_corr_count = abs_correlations.iter().filter(|&&c| c < 0.3).count();
        let total_correlations = abs_correlations.len();

        if high_corr_count > total_correlations / 2 {
            CorrelationRegime::HighCorrelation
        } else if low_corr_count > total_correlations / 2 {
            CorrelationRegime::LowCorrelation
        } else if avg_correlation > 0.5 {
            CorrelationRegime::MixedCorrelation
        } else {
            CorrelationRegime::Transitional
        }
    }

    /// Calculate dynamic correlations
    fn calculate_dynamic_correlations(&self) -> QarResult<DynamicCorrelations> {
        let mut rolling_correlations = HashMap::new();
        let mut breakpoints = Vec::new();

        let factor_names: Vec<String> = self.factor_history.keys().cloned().collect();
        
        for i in 0..factor_names.len() {
            for j in (i + 1)..factor_names.len() {
                let factor1 = &factor_names[i];
                let factor2 = &factor_names[j];
                let pair_key = format!("{}_{}", factor1, factor2);
                
                if let (Some(values1), Some(values2)) = (
                    self.factor_history.get(factor1),
                    self.factor_history.get(factor2),
                ) {
                    let rolling_corrs = self.calculate_rolling_correlations(values1, values2)?;
                    
                    // Detect breakpoints in correlation
                    let pair_breakpoints = self.detect_correlation_breakpoints(&rolling_corrs, &pair_key);
                    breakpoints.extend(pair_breakpoints);
                    
                    rolling_correlations.insert(pair_key, rolling_corrs);
                }
            }
        }

        // Calculate overall stability
        let stability = self.calculate_correlation_stability(&rolling_correlations);

        Ok(DynamicCorrelations {
            rolling_correlations,
            stability,
            breakpoints,
        })
    }

    /// Calculate rolling correlations
    fn calculate_rolling_correlations(&self, series1: &[f64], series2: &[f64]) -> QarResult<Vec<f64>> {
        let window_size = self.correlation_params.rolling_window;
        let mut rolling_corrs = Vec::new();

        if series1.len() < window_size || series2.len() < window_size {
            return Ok(rolling_corrs);
        }

        for i in window_size..=series1.len() {
            let window1 = &series1[i - window_size..i];
            let window2 = &series2[i - window_size..i];
            let correlation = self.calculate_correlation(window1, window2)?;
            rolling_corrs.push(correlation);
        }

        Ok(rolling_corrs)
    }

    /// Detect correlation breakpoints
    fn detect_correlation_breakpoints(&self, rolling_corrs: &[f64], pair_key: &str) -> Vec<CorrelationBreakpoint> {
        let mut breakpoints = Vec::new();
        
        if rolling_corrs.len() < 10 {
            return breakpoints;
        }

        let sensitivity = self.correlation_params.breakpoint_sensitivity;
        
        for i in 5..(rolling_corrs.len() - 5) {
            let before_window = &rolling_corrs[i - 5..i];
            let after_window = &rolling_corrs[i..i + 5];
            
            let before_mean = before_window.iter().sum::<f64>() / before_window.len() as f64;
            let after_mean = after_window.iter().sum::<f64>() / after_window.len() as f64;
            
            let change = (after_mean - before_mean).abs();
            
            if change > sensitivity {
                breakpoints.push(CorrelationBreakpoint {
                    timestamp: chrono::Utc::now(),
                    factor_pair: pair_key.to_string(),
                    correlation_before: before_mean,
                    correlation_after: after_mean,
                    significance: change,
                });
            }
        }

        breakpoints
    }

    /// Calculate correlation stability
    fn calculate_correlation_stability(&self, rolling_correlations: &HashMap<String, Vec<f64>>) -> f64 {
        if rolling_correlations.is_empty() {
            return 1.0; // Perfect stability if no data
        }

        let mut stability_scores = Vec::new();

        for rolling_corrs in rolling_correlations.values() {
            if rolling_corrs.len() < 2 {
                continue;
            }

            // Calculate coefficient of variation as stability measure
            let mean = rolling_corrs.iter().sum::<f64>() / rolling_corrs.len() as f64;
            let variance = rolling_corrs.iter()
                .map(|c| (c - mean).powi(2))
                .sum::<f64>() / rolling_corrs.len() as f64;
            
            let std_dev = variance.sqrt();
            let coefficient_of_variation = if mean.abs() > 1e-10 {
                std_dev / mean.abs()
            } else {
                0.0
            };
            
            // Convert to stability score (lower CV = higher stability)
            let stability = 1.0 / (1.0 + coefficient_of_variation);
            stability_scores.push(stability);
        }

        if stability_scores.is_empty() {
            1.0
        } else {
            stability_scores.iter().sum::<f64>() / stability_scores.len() as f64
        }
    }

    /// Calculate overall correlation score
    fn calculate_correlation_score(&self, correlations: &HashMap<String, f64>) -> f64 {
        if correlations.is_empty() {
            return 0.0;
        }

        // Calculate average absolute correlation
        let abs_correlations: Vec<f64> = correlations.values().map(|&c| c.abs()).collect();
        abs_correlations.iter().sum::<f64>() / abs_correlations.len() as f64
    }

    /// Calculate confidence in correlation assessment
    fn calculate_correlation_confidence(&self, correlations: &HashMap<String, f64>, dynamic_corrs: &DynamicCorrelations) -> f64 {
        let mut confidence_factors = Vec::new();

        // Sample size confidence
        let min_sample_size = self.factor_history.values()
            .map(|v| v.len())
            .min()
            .unwrap_or(0);
        
        let sample_confidence = if min_sample_size < 10 {
            0.3
        } else if min_sample_size < 30 {
            0.6
        } else {
            0.9
        };
        confidence_factors.push(sample_confidence);

        // Stability confidence
        confidence_factors.push(dynamic_corrs.stability);

        // Significance confidence (based on number of significant correlations)
        if !correlations.is_empty() {
            let significant_corrs = correlations.values()
                .filter(|&&c| c.abs() > self.correlation_params.min_correlation_threshold)
                .count();
            let significance_confidence = significant_corrs as f64 / correlations.len() as f64;
            confidence_factors.push(significance_confidence);
        }

        // Calculate overall confidence
        if confidence_factors.is_empty() {
            0.5
        } else {
            confidence_factors.iter().sum::<f64>() / confidence_factors.len() as f64
        }
    }

    fn add_to_history(&mut self, result: CorrelationResult) {
        self.history.push(result);
        
        if self.history.len() > self.config.max_history {
            self.history.remove(0);
        }
    }

    /// Get analysis history
    pub fn get_history(&self) -> &[CorrelationResult] {
        &self.history
    }

    /// Get latest analysis
    pub fn get_latest(&self) -> Option<&CorrelationResult> {
        self.history.last()
    }

    /// Get correlation parameters
    pub fn get_parameters(&self) -> &CorrelationParameters {
        &self.correlation_params
    }

    /// Update correlation parameters
    pub fn update_parameters(&mut self, params: CorrelationParameters) {
        self.correlation_params = params;
    }

    /// Get factor history
    pub fn get_factor_history(&self) -> &HashMap<String, Vec<f64>> {
        &self.factor_history
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::StandardFactors;

    #[tokio::test]
    async fn test_correlation_analyzer() {
        let config = super::super::AnalysisConfig::default();
        let mut analyzer = CorrelationAnalyzer::new(config).unwrap();

        // Add multiple data points to build history
        for i in 0..10 {
            let mut factors = std::collections::HashMap::new();
            factors.insert(StandardFactors::Trend.to_string(), 0.5 + i as f64 * 0.05);
            factors.insert(StandardFactors::Momentum.to_string(), 0.6 + i as f64 * 0.03);
            factors.insert(StandardFactors::Volatility.to_string(), 0.4 - i as f64 * 0.02);
            factors.insert(StandardFactors::Volume.to_string(), 0.7 + i as f64 * 0.01);
            
            let factor_map = FactorMap::new(factors).unwrap();
            let result = analyzer.analyze(&factor_map).await;
            assert!(result.is_ok());
        }

        let latest = analyzer.get_latest().unwrap();
        assert!(latest.score >= 0.0 && latest.score <= 1.0);
        assert!(latest.confidence >= 0.0 && latest.confidence <= 1.0);
        assert!(!latest.pairwise_correlations.is_empty());
    }

    #[test]
    fn test_correlation_calculation() {
        let config = super::super::AnalysisConfig::default();
        let analyzer = CorrelationAnalyzer::new(config).unwrap();
        
        // Perfect positive correlation
        let series1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let series2 = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let correlation = analyzer.calculate_correlation(&series1, &series2).unwrap();
        assert!((correlation - 1.0).abs() < 0.01);

        // Perfect negative correlation
        let series3 = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let correlation = analyzer.calculate_correlation(&series1, &series3).unwrap();
        assert!((correlation + 1.0).abs() < 0.01);

        // No correlation
        let series4 = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let correlation = analyzer.calculate_correlation(&series1, &series4).unwrap();
        assert!(correlation.abs() < 0.5);
    }

    #[test]
    fn test_variance_calculation() {
        let config = super::super::AnalysisConfig::default();
        let analyzer = CorrelationAnalyzer::new(config).unwrap();
        
        let constant_series = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let variance = analyzer.calculate_variance(&constant_series);
        assert!(variance.abs() < 0.01);

        let variable_series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let variance = analyzer.calculate_variance(&variable_series);
        assert!(variance > 0.0);
    }

    #[test]
    fn test_2x2_eigenvalue_calculation() {
        let config = super::super::AnalysisConfig::default();
        let analyzer = CorrelationAnalyzer::new(config).unwrap();
        
        let matrix = vec![
            vec![4.0, 2.0],
            vec![2.0, 3.0]
        ];
        
        let eigenvalues = analyzer.calculate_2x2_eigenvalues(&matrix);
        assert_eq!(eigenvalues.len(), 2);
        assert!(eigenvalues[0] >= eigenvalues[1]); // Should be sorted
        
        // For this matrix, eigenvalues should be approximately 5.828 and 1.172
        assert!(eigenvalues[0] > 5.0 && eigenvalues[0] < 6.0);
        assert!(eigenvalues[1] > 1.0 && eigenvalues[1] < 2.0);
    }

    #[test]
    fn test_correlation_regime_determination() {
        let config = super::super::AnalysisConfig::default();
        let analyzer = CorrelationAnalyzer::new(config).unwrap();
        
        // High correlation case
        let mut high_corr = HashMap::new();
        high_corr.insert("pair1".to_string(), 0.8);
        high_corr.insert("pair2".to_string(), 0.9);
        high_corr.insert("pair3".to_string(), 0.7);
        
        let regime = analyzer.determine_correlation_regime(&high_corr);
        assert!(matches!(regime, CorrelationRegime::HighCorrelation));

        // Low correlation case
        let mut low_corr = HashMap::new();
        low_corr.insert("pair1".to_string(), 0.1);
        low_corr.insert("pair2".to_string(), 0.2);
        low_corr.insert("pair3".to_string(), 0.15);
        
        let regime = analyzer.determine_correlation_regime(&low_corr);
        assert!(matches!(regime, CorrelationRegime::LowCorrelation));
    }

    #[test]
    fn test_rolling_correlation_calculation() {
        let config = super::super::AnalysisConfig::default();
        let mut analyzer = CorrelationAnalyzer::new(config).unwrap();
        analyzer.correlation_params.rolling_window = 3;
        
        let series1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let series2 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        
        let rolling_corrs = analyzer.calculate_rolling_correlations(&series1, &series2).unwrap();
        assert!(!rolling_corrs.is_empty());
        
        // Should be perfect correlation in all windows
        for &corr in &rolling_corrs {
            assert!((corr - 1.0).abs() < 0.01);
        }
    }
}