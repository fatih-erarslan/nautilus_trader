//! Antifragility Detection and Measurement
//! 
//! Implementation of Nassim Taleb's concept of antifragility in financial markets.
//! Antifragile systems benefit from volatility and disorder rather than just surviving them.

use crate::market_data::MarketData;
use crate::utils::Statistics;
use anyhow::Result;

pub struct AntifragilityAnalyzer {
    lookback_period: usize,
    volatility_threshold: f64,
    convexity_threshold: f64,
}

impl AntifragilityAnalyzer {
    pub fn new() -> Self {
        Self {
            lookback_period: 50,
            volatility_threshold: 0.02,
            convexity_threshold: 0.1,
        }
    }

    pub fn calculate_antifragility(&self, data: &MarketData) -> Result<f64> {
        if data.len() < self.lookback_period {
            return Ok(0.0);
        }

        // Core antifragility components
        let volatility_benefit = self.calculate_volatility_benefit(data)?;
        let stress_resistance = self.calculate_stress_resistance(data)?;
        let convexity_score = self.calculate_convexity(data)?;
        let adaptation_speed = self.calculate_adaptation_speed(data)?;
        let tail_benefit = self.calculate_tail_benefit(data)?;

        // Weighted combination
        let antifragility_score = (
            volatility_benefit * 0.25 +
            stress_resistance * 0.20 +
            convexity_score * 0.25 +
            adaptation_speed * 0.15 +
            tail_benefit * 0.15
        ).clamp(0.0, 1.0);

        Ok(antifragility_score)
    }

    /// Measures how much the asset benefits from increased volatility
    fn calculate_volatility_benefit(&self, data: &MarketData) -> Result<f64> {
        let returns = data.returns();
        let volatilities = data.volatility(20);
        
        if returns.len() < 40 || volatilities.is_empty() {
            return Ok(0.0);
        }

        // Calculate correlation between volatility and subsequent returns
        let mut vol_return_pairs = Vec::new();
        
        for i in 0..(volatilities.len().min(returns.len() - 20)) {
            let vol = volatilities[i];
            let future_return = returns[i + 20]; // 20-period forward return
            vol_return_pairs.push((vol, future_return));
        }

        let correlation = self.calculate_correlation(&vol_return_pairs);
        
        // Positive correlation indicates antifragility (benefits from volatility)
        Ok((correlation + 1.0) / 2.0) // Normalize to 0-1
    }

    /// Measures resistance to stress events and ability to recover
    fn calculate_stress_resistance(&self, data: &MarketData) -> Result<f64> {
        let returns = data.returns();
        
        if returns.len() < 50 {
            return Ok(0.0);
        }

        // Identify stress events (large negative returns)
        let stress_threshold = Statistics::percentile(&returns, 10.0); // Bottom 10% of returns
        let mut recovery_scores = Vec::new();

        for (i, &ret) in returns.iter().enumerate() {
            if ret < stress_threshold && i + 20 < returns.len() {
                // Measure recovery over next 20 periods
                let recovery_returns: f64 = returns[i+1..i+21].iter().sum();
                let recovery_score = if recovery_returns > -ret {
                    1.0 // Full recovery or better
                } else {
                    (recovery_returns + ret.abs()) / ret.abs()
                };
                recovery_scores.push(recovery_score.max(0.0));
            }
        }

        if recovery_scores.is_empty() {
            Ok(0.5) // Neutral if no stress events
        } else {
            Ok(recovery_scores.iter().sum::<f64>() / recovery_scores.len() as f64)
        }
    }

    /// Measures convexity (non-linear response to changes)
    fn calculate_convexity(&self, data: &MarketData) -> Result<f64> {
        let returns = data.returns();
        
        if returns.len() < 30 {
            return Ok(0.0);
        }

        // Calculate second derivative approximation
        let mut convexities = Vec::new();
        
        for i in 2..returns.len() {
            let second_derivative = returns[i] - 2.0 * returns[i-1] + returns[i-2];
            convexities.push(second_derivative.abs());
        }

        // Higher convexity indicates more non-linear (antifragile) behavior
        let avg_convexity = convexities.iter().sum::<f64>() / convexities.len() as f64;
        
        // Normalize based on threshold
        Ok((avg_convexity / self.convexity_threshold).min(1.0))
    }

    /// Measures speed of adaptation to new conditions
    fn calculate_adaptation_speed(&self, data: &MarketData) -> Result<f64> {
        let prices = &data.prices;
        
        if prices.len() < 40 {
            return Ok(0.0);
        }

        // Calculate regime change detection and adaptation
        let mut adaptation_scores = Vec::new();
        
        // Use rolling window to detect regime changes
        for i in 20..(prices.len() - 20) {
            let before_mean = prices[i-20..i].iter().sum::<f64>() / 20.0;
            let after_mean = prices[i..i+20].iter().sum::<f64>() / 20.0;
            
            let regime_change = (after_mean - before_mean).abs() / before_mean;
            
            if regime_change > 0.05 { // 5% change threshold
                // Measure how quickly price adapts to new level
                let mut adaptation_time = 20.0;
                let target_level = after_mean;
                
                for j in 1..20 {
                    if (prices[i + j] - target_level).abs() / target_level < 0.02 {
                        adaptation_time = j as f64;
                        break;
                    }
                }
                
                // Faster adaptation = higher score
                adaptation_scores.push(1.0 - (adaptation_time / 20.0));
            }
        }

        if adaptation_scores.is_empty() {
            Ok(0.5)
        } else {
            Ok(adaptation_scores.iter().sum::<f64>() / adaptation_scores.len() as f64)
        }
    }

    /// Measures benefit from tail events (extreme outcomes)
    fn calculate_tail_benefit(&self, data: &MarketData) -> Result<f64> {
        let returns = data.returns();
        
        if returns.len() < 100 {
            return Ok(0.0);
        }

        // Identify tail events (extreme returns)
        let left_tail = Statistics::percentile(&returns, 5.0);   // Bottom 5%
        let right_tail = Statistics::percentile(&returns, 95.0); // Top 5%
        
        let mut tail_benefits = Vec::new();
        
        for (i, &ret) in returns.iter().enumerate() {
            if (ret < left_tail || ret > right_tail) && i + 10 < returns.len() {
                // Measure subsequent performance after tail event
                let post_tail_returns: f64 = returns[i+1..i+11].iter().sum();
                
                // Antifragile assets should benefit from tail events
                let benefit_score = if ret > 0.0 {
                    post_tail_returns.max(0.0) / ret.abs()
                } else {
                    (-post_tail_returns).max(0.0) / ret.abs()
                };
                
                tail_benefits.push(benefit_score.min(2.0)); // Cap at 2x
            }
        }

        if tail_benefits.is_empty() {
            Ok(0.5)
        } else {
            Ok((tail_benefits.iter().sum::<f64>() / tail_benefits.len() as f64).min(1.0))
        }
    }

    fn calculate_correlation(&self, pairs: &[(f64, f64)]) -> f64 {
        if pairs.len() < 10 {
            return 0.0;
        }

        let n = pairs.len() as f64;
        let sum_x: f64 = pairs.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = pairs.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = pairs.iter().map(|(x, y)| x * y).sum();
        let sum_x2: f64 = pairs.iter().map(|(x, _)| x * x).sum();
        let sum_y2: f64 = pairs.iter().map(|(_, y)| y * y).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
}

#[derive(Debug, Clone)]
pub struct AntifragilityMetrics {
    pub overall_score: f64,
    pub volatility_benefit: f64,
    pub stress_resistance: f64,
    pub convexity: f64,
    pub adaptation_speed: f64,
    pub tail_benefit: f64,
    pub confidence: f64,
}

impl Default for AntifragilityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}