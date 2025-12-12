//! Pareto distribution implementation

use crate::distributions::{FatTailDistribution, DistributionMoments};
use crate::error::TalebianResult as Result;
use serde::{Deserialize, Serialize};

/// Pareto distribution for modeling power law behaviors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoDistribution {
    /// Shape parameter (alpha)
    pub alpha: f64,
    /// Scale parameter (minimum value)
    pub x_min: f64,
}

impl ParetoDistribution {
    /// Create a new Pareto distribution
    pub fn new(alpha: f64, x_min: f64) -> Result<Self> {
        if alpha <= 0.0 || x_min <= 0.0 {
            return Err(crate::error::TalebianError::distribution(
                "Alpha and x_min must be positive"
            ));
        }
        
        Ok(Self { alpha, x_min })
    }
}

impl FatTailDistribution for ParetoDistribution {
    fn name(&self) -> &'static str {
        "Pareto"
    }
    
    fn pdf(&self, x: f64) -> f64 {
        if x < self.x_min {
            0.0
        } else {
            self.alpha * self.x_min.powf(self.alpha) / x.powf(self.alpha + 1.0)
        }
    }
    
    fn cdf(&self, x: f64) -> f64 {
        if x < self.x_min {
            0.0
        } else {
            1.0 - (self.x_min / x).powf(self.alpha)
        }
    }
    
    fn quantile(&self, p: f64) -> Result<f64> {
        if p < 0.0 || p > 1.0 {
            return Err(crate::error::TalebianError::distribution(
                "Probability must be between 0 and 1"
            ));
        }
        
        Ok(self.x_min * (1.0 - p).powf(-1.0 / self.alpha))
    }
    
    fn sample(&self, n: usize) -> Result<Vec<f64>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut samples = Vec::with_capacity(n);
        
        for _ in 0..n {
            let u: f64 = rng.gen();
            samples.push(self.quantile(u)?);
        }
        
        Ok(samples)
    }
    
    fn moments(&self) -> Result<DistributionMoments> {
        let mut moments = DistributionMoments::new();
        
        // Mean exists only if alpha > 1
        if self.alpha > 1.0 {
            let mean = self.alpha * self.x_min / (self.alpha - 1.0);
            moments.set_mean(mean);
        }
        
        // Variance exists only if alpha > 2
        if self.alpha > 2.0 {
            let mean = self.alpha * self.x_min / (self.alpha - 1.0);
            let variance = (self.alpha * self.x_min * self.x_min) / 
                          ((self.alpha - 1.0).powi(2) * (self.alpha - 2.0));
            moments.set_variance(variance);
        }
        
        // Skewness exists only if alpha > 3
        if self.alpha > 3.0 {
            let skewness = 2.0 * (1.0 + self.alpha) / (self.alpha - 3.0) * 
                          ((self.alpha - 2.0) / self.alpha).sqrt();
            moments.set_skewness(skewness);
        }
        
        // Kurtosis exists only if alpha > 4
        if self.alpha > 4.0 {
            let kurtosis = 6.0 * (self.alpha.powi(3) + self.alpha.powi(2) - 6.0 * self.alpha - 2.0) /
                          (self.alpha * (self.alpha - 3.0) * (self.alpha - 4.0));
            moments.set_kurtosis(kurtosis);
        }
        
        Ok(moments)
    }
    
    fn cvar(&self, confidence: f64) -> Result<f64> {
        let var = self.var(confidence)?;
        let alpha = 1.0 - confidence;
        
        // For Pareto distribution, CVaR has a closed form
        if self.alpha > 1.0 {
            Ok(var * self.alpha / (self.alpha - 1.0) / alpha)
        } else {
            Err(crate::error::TalebianError::distribution(
                "CVaR is infinite for alpha <= 1"
            ))
        }
    }
    
    fn tail_index(&self) -> Result<f64> {
        Ok(1.0 / self.alpha)
    }
    
    fn expected_shortfall(&self, threshold: f64) -> Result<f64> {
        if threshold < self.x_min {
            return Ok(0.0);
        }
        
        if self.alpha > 1.0 {
            Ok(self.alpha * threshold / (self.alpha - 1.0))
        } else {
            Err(crate::error::TalebianError::distribution(
                "Expected shortfall is infinite for alpha <= 1"
            ))
        }
    }
    
    fn fit(&mut self, data: &[f64]) -> Result<()> {
        if data.is_empty() {
            return Err(crate::error::TalebianError::distribution(
                "Cannot fit distribution to empty data"
            ));
        }
        
        // Use MLE to estimate parameters
        self.x_min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        let n = data.len() as f64;
        let sum_log = data.iter().map(|&x| (x / self.x_min).ln()).sum::<f64>();
        
        self.alpha = n / sum_log;
        
        Ok(())
    }
    
    fn log_likelihood(&self, data: &[f64]) -> Result<f64> {
        let mut ll = 0.0;
        let n = data.len() as f64;
        
        for &x in data {
            if x < self.x_min {
                return Ok(f64::NEG_INFINITY);
            }
            ll += self.pdf(x).ln();
        }
        
        Ok(ll)
    }
    
    fn parameter_count(&self) -> usize {
        2
    }
    
    fn parameters(&self) -> Vec<f64> {
        vec![self.alpha, self.x_min]
    }
    
    fn set_parameters(&mut self, params: &[f64]) -> Result<()> {
        if params.len() != 2 {
            return Err(crate::error::TalebianError::distribution(
                "Pareto distribution requires exactly 2 parameters"
            ));
        }
        
        self.alpha = params[0];
        self.x_min = params[1];
        self.validate_parameters()
    }
    
    fn validate_parameters(&self) -> Result<()> {
        if self.alpha <= 0.0 {
            return Err(crate::error::TalebianError::distribution(
                "Alpha parameter must be positive"
            ));
        }
        
        if self.x_min <= 0.0 {
            return Err(crate::error::TalebianError::distribution(
                "X_min parameter must be positive"
            ));
        }
        
        Ok(())
    }
    
    fn has_finite_mean(&self) -> bool {
        self.alpha > 1.0
    }
    
    fn has_finite_variance(&self) -> bool {
        self.alpha > 2.0
    }
    
    fn has_finite_higher_moments(&self) -> bool {
        self.alpha > 4.0
    }
}