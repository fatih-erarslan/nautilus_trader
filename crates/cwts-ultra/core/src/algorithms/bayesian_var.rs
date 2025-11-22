use ndarray::{Array1, Array2};

#[derive(Debug, Clone)]
pub struct BayesianVaR {
    confidence_level: f64,
    time_horizon: usize,
    posterior_samples: Vec<f64>,
}

impl BayesianVaR {
    pub fn new(confidence_level: f64, time_horizon: usize) -> Self {
        Self {
            confidence_level,
            time_horizon,
            posterior_samples: Vec::new(),
        }
    }
    
    pub fn calculate(&self, returns: &Array1<f64>) -> f64 {
        let mut sorted = returns.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let index = ((1.0 - self.confidence_level) * sorted.len() as f64) as usize;
        sorted[index.min(sorted.len() - 1)]
    }
    
    pub fn update_posterior(&mut self, new_data: &Array1<f64>) {
        self.posterior_samples.extend(new_data.iter().copied());
    }
    
    pub fn get_risk_metrics(&self) -> RiskMetrics {
        RiskMetrics {
            var: self.calculate(&Array1::from(self.posterior_samples.clone())),
            confidence_level: self.confidence_level,
            samples: self.posterior_samples.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RiskMetrics {
    pub var: f64,
    pub confidence_level: f64,
    pub samples: usize,
}