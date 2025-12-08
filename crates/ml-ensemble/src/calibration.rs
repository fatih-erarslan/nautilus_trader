use crate::types::{CalibrationConfig, MarketCondition};
use anyhow::Result;
use ndarray::{Array1, Array2};
use std::collections::VecDeque;
use tracing::debug;

/// Confidence calibration for ensemble predictions
pub struct ConfidenceCalibrator {
    config: CalibrationConfig,
    calibration_data: VecDeque<CalibrationSample>,
    isotonic_model: Option<IsotonicRegression>,
    platt_model: Option<PlattScaling>,
}

#[derive(Debug, Clone)]
struct CalibrationSample {
    raw_confidence: f64,
    actual_outcome: f64,
    market_condition: MarketCondition,
}

impl ConfidenceCalibrator {
    /// Create new calibrator
    pub fn new(config: CalibrationConfig) -> Self {
        Self {
            config,
            calibration_data: VecDeque::new(),
            isotonic_model: None,
            platt_model: None,
        }
    }
    
    /// Calibrate confidence score
    pub fn calibrate_confidence(
        &self,
        raw_confidence: f64,
        market_condition: MarketCondition,
    ) -> Result<f64> {
        // Apply calibration if models are trained
        let calibrated = if self.config.isotonic_regression && self.isotonic_model.is_some() {
            self.isotonic_model.as_ref().unwrap().predict(raw_confidence)
        } else if self.config.platt_scaling && self.platt_model.is_some() {
            self.platt_model.as_ref().unwrap().predict(raw_confidence)
        } else {
            raw_confidence
        };
        
        // Apply market condition adjustment
        let adjusted = self.adjust_for_market_condition(calibrated, market_condition);
        
        // Ensure confidence is in [0, 1]
        Ok(adjusted.clamp(0.0, 1.0))
    }
    
    /// Update calibration with new observation
    pub fn update(&mut self, actual_outcome: f64) -> Result<()> {
        // Note: In production, we'd store the raw confidence from the prediction
        // For now, we'll skip the update if we don't have the matching prediction
        
        // Retrain calibration models if we have enough data
        if self.calibration_data.len() >= self.config.min_samples {
            if self.config.isotonic_regression {
                self.train_isotonic_regression()?;
            }
            if self.config.platt_scaling {
                self.train_platt_scaling()?;
            }
        }
        
        Ok(())
    }
    
    /// Add calibration sample
    pub fn add_sample(
        &mut self,
        raw_confidence: f64,
        actual_outcome: f64,
        market_condition: MarketCondition,
    ) {
        let sample = CalibrationSample {
            raw_confidence,
            actual_outcome,
            market_condition,
        };
        
        self.calibration_data.push_back(sample);
        
        // Limit data size
        if self.calibration_data.len() > self.config.window_size {
            self.calibration_data.pop_front();
        }
    }
    
    /// Adjust confidence based on market condition
    fn adjust_for_market_condition(
        &self,
        confidence: f64,
        market_condition: MarketCondition,
    ) -> f64 {
        // Market-specific adjustments based on historical performance
        match market_condition {
            MarketCondition::HighVolatility => confidence * 0.9, // Less confident in volatile markets
            MarketCondition::Anomalous => confidence * 0.8,     // Much less confident in anomalies
            MarketCondition::Trending => confidence * 1.05,      // Slightly more confident in trends
            MarketCondition::Ranging => confidence * 0.95,       // Slightly less confident in ranges
            _ => confidence,
        }
    }
    
    /// Train isotonic regression model
    fn train_isotonic_regression(&mut self) -> Result<()> {
        let n = self.calibration_data.len();
        if n < self.config.min_samples {
            return Ok(());
        }
        
        let mut data: Vec<(f64, f64)> = self.calibration_data
            .iter()
            .map(|s| (s.raw_confidence, s.actual_outcome))
            .collect();
        
        // Sort by raw confidence
        data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        let isotonic = IsotonicRegression::fit(&data)?;
        self.isotonic_model = Some(isotonic);
        
        debug!("Trained isotonic regression with {} samples", n);
        Ok(())
    }
    
    /// Train Platt scaling model
    fn train_platt_scaling(&mut self) -> Result<()> {
        let n = self.calibration_data.len();
        if n < self.config.min_samples {
            return Ok(());
        }
        
        let x: Vec<f64> = self.calibration_data.iter().map(|s| s.raw_confidence).collect();
        let y: Vec<f64> = self.calibration_data.iter().map(|s| s.actual_outcome).collect();
        
        let platt = PlattScaling::fit(&x, &y)?;
        self.platt_model = Some(platt);
        
        debug!("Trained Platt scaling with {} samples", n);
        Ok(())
    }
}

/// Isotonic regression for calibration
struct IsotonicRegression {
    x_values: Vec<f64>,
    y_values: Vec<f64>,
}

impl IsotonicRegression {
    /// Fit isotonic regression
    fn fit(data: &[(f64, f64)]) -> Result<Self> {
        let n = data.len();
        let mut y_values: Vec<f64> = data.iter().map(|(_, y)| *y).collect();
        
        // Pool adjacent violators algorithm
        let mut i = 0;
        while i < n - 1 {
            if y_values[i] > y_values[i + 1] {
                // Find violating sequence
                let mut j = i + 1;
                let mut sum = y_values[i] + y_values[j];
                let mut count = 2;
                
                while j < n - 1 && y_values[j] >= y_values[j + 1] {
                    j += 1;
                    sum += y_values[j];
                    count += 1;
                }
                
                // Pool values
                let pooled = sum / count as f64;
                for k in i..=j {
                    y_values[k] = pooled;
                }
                
                // Restart from beginning
                i = 0;
            } else {
                i += 1;
            }
        }
        
        let x_values: Vec<f64> = data.iter().map(|(x, _)| *x).collect();
        
        Ok(Self { x_values, y_values })
    }
    
    /// Predict calibrated confidence
    fn predict(&self, x: f64) -> f64 {
        // Linear interpolation
        if x <= self.x_values[0] {
            return self.y_values[0];
        }
        if x >= self.x_values[self.x_values.len() - 1] {
            return self.y_values[self.y_values.len() - 1];
        }
        
        // Find interval
        for i in 0..self.x_values.len() - 1 {
            if x >= self.x_values[i] && x <= self.x_values[i + 1] {
                let t = (x - self.x_values[i]) / (self.x_values[i + 1] - self.x_values[i]);
                return self.y_values[i] * (1.0 - t) + self.y_values[i + 1] * t;
            }
        }
        
        self.y_values[self.y_values.len() - 1]
    }
}

/// Platt scaling for calibration
struct PlattScaling {
    a: f64,
    b: f64,
}

impl PlattScaling {
    /// Fit Platt scaling (sigmoid calibration)
    fn fit(x: &[f64], y: &[f64]) -> Result<Self> {
        // Simple logistic regression
        // In production, use proper optimization
        let n = x.len() as f64;
        
        // Calculate means
        let x_mean = x.iter().sum::<f64>() / n;
        let y_mean = y.iter().sum::<f64>() / n;
        
        // Calculate covariance and variance
        let mut cov = 0.0;
        let mut var = 0.0;
        
        for i in 0..x.len() {
            cov += (x[i] - x_mean) * (y[i] - y_mean);
            var += (x[i] - x_mean) * (x[i] - x_mean);
        }
        
        let a = if var > 0.0 { -cov / var } else { -1.0 };
        let b = -a * x_mean;
        
        Ok(Self { a, b })
    }
    
    /// Predict calibrated confidence
    fn predict(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-self.a * x - self.b).exp())
    }
}