//! Regression Models using Quantum Annealing
//!
//! This module implements various regression models that use quantum annealing
//! for parameter optimization, providing better global optimization compared
//! to traditional gradient-based methods.

use crate::core::*;
use crate::error::*;
use crate::annealing::QuantumAnnealer;
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

/// Linear regression model using quantum annealing
#[derive(Debug, Clone)]
pub struct QuantumLinearRegression {
    /// Model coefficients
    pub coefficients: Option<Vec<f64>>,
    /// Intercept term
    pub intercept: Option<f64>,
    /// Training metrics
    pub metrics: HashMap<String, f64>,
    /// Annealing configuration
    config: QuantumAnnealingConfig,
}

impl QuantumLinearRegression {
    /// Create a new quantum linear regression model
    pub fn new(config: QuantumAnnealingConfig) -> Self {
        Self {
            coefficients: None,
            intercept: None,
            metrics: HashMap::new(),
            config,
        }
    }
    
    /// Train the model on the given problem
    pub fn fit(&mut self, problem: &RegressionProblem) -> QarResult<()> {
        let num_features = problem.feature_dimension();
        let num_params = num_features + 1; // +1 for intercept
        
        // Initial parameters (small random values)
        let initial_params: Vec<f64> = (0..num_params)
            .map(|_| rand::random::<f64>() * 0.1 - 0.05)
            .collect();
        
        // Create cost function (mean squared error)
        let cost_function = |params: &[f64]| -> f64 {
            let coefficients = &params[0..num_features];
            let intercept = params[num_features];
            
            let mut total_error = 0.0;
            for (features, &target) in problem.features.iter().zip(problem.targets.iter()) {
                let prediction = features.iter()
                    .zip(coefficients.iter())
                    .map(|(&x, &w)| x * w)
                    .sum::<f64>() + intercept;
                
                let error = prediction - target;
                total_error += error * error;
            }
            
            total_error / problem.num_samples() as f64 // MSE
        };
        
        // Optimize using quantum annealing
        let mut annealer = QuantumAnnealer::new(self.config.clone())?;
        let result = annealer.optimize(cost_function, initial_params)?;
        
        // Extract coefficients and intercept
        self.coefficients = Some(result.optimal_parameters[0..num_features].to_vec());
        self.intercept = Some(result.optimal_parameters[num_features]);
        
        // Store metrics
        self.metrics.insert("mse".to_string(), result.optimal_energy);
        self.metrics.insert("rmse".to_string(), result.optimal_energy.sqrt());
        self.metrics.insert("iterations".to_string(), result.iterations as f64);
        self.metrics.insert("acceptance_rate".to_string(), result.acceptance_rate);
        
        // Calculate R-squared
        let r_squared = self.calculate_r_squared(problem)?;
        self.metrics.insert("r_squared".to_string(), r_squared);
        
        Ok(())
    }
    
    /// Make predictions on new data
    pub fn predict(&self, features: &[Vec<f64>]) -> QarResult<Vec<f64>> {
        let coefficients = self.coefficients.as_ref()
            .ok_or_else(|| QarError::RegressionError("Model not trained".to_string()))?;
        let intercept = self.intercept
            .ok_or_else(|| QarError::RegressionError("Model not trained".to_string()))?;
        
        let predictions: Vec<f64> = features.iter()
            .map(|feature_vec| {
                feature_vec.iter()
                    .zip(coefficients.iter())
                    .map(|(&x, &w)| x * w)
                    .sum::<f64>() + intercept
            })
            .collect();
        
        Ok(predictions)
    }
    
    /// Calculate R-squared metric
    fn calculate_r_squared(&self, problem: &RegressionProblem) -> QarResult<f64> {
        let predictions = self.predict(&problem.features)?;
        
        let mean_target = problem.targets.iter().sum::<f64>() / problem.targets.len() as f64;
        
        let ss_res: f64 = problem.targets.iter()
            .zip(predictions.iter())
            .map(|(&actual, &predicted)| (actual - predicted).powi(2))
            .sum();
        
        let ss_tot: f64 = problem.targets.iter()
            .map(|&target| (target - mean_target).powi(2))
            .sum();
        
        if ss_tot == 0.0 {
            Ok(1.0) // Perfect fit when all targets are the same
        } else {
            Ok(1.0 - ss_res / ss_tot)
        }
    }
}

/// Non-linear regression using quantum annealing
pub struct QuantumNonLinearRegression {
    /// Model parameters
    pub parameters: Option<Vec<f64>>,
    /// Basis functions used
    basis_functions: Vec<Box<dyn Fn(&[f64], &[f64]) -> f64 + Send + Sync>>,
    /// Training metrics
    pub metrics: HashMap<String, f64>,
    /// Annealing configuration
    config: QuantumAnnealingConfig,
}

impl QuantumNonLinearRegression {
    /// Create a new quantum non-linear regression model
    pub fn new(config: QuantumAnnealingConfig) -> Self {
        Self {
            parameters: None,
            basis_functions: Vec::new(),
            metrics: HashMap::new(),
            config,
        }
    }
    
    /// Add a polynomial basis function
    pub fn add_polynomial_basis(&mut self, degree: usize) {
        for d in 1..=degree {
            let degree_copy = d;
            self.basis_functions.push(Box::new(move |features: &[f64], _params: &[f64]| {
                features.iter().map(|&x| x.powi(degree_copy as i32)).sum()
            }));
        }
    }
    
    /// Add a Gaussian radial basis function
    pub fn add_gaussian_basis(&mut self, center: Vec<f64>, width: f64) {
        self.basis_functions.push(Box::new(move |features: &[f64], _params: &[f64]| {
            let distance_sq: f64 = features.iter()
                .zip(center.iter())
                .map(|(&x, &c)| (x - c).powi(2))
                .sum();
            (-distance_sq / (2.0 * width * width)).exp()
        }));
    }
    
    /// Train the model
    pub fn fit(&mut self, problem: &RegressionProblem) -> QarResult<()> {
        if self.basis_functions.is_empty() {
            return Err(QarError::RegressionError("No basis functions defined".to_string()));
        }
        
        let num_params = self.basis_functions.len() + 1; // +1 for intercept
        
        // Initial parameters
        let initial_params: Vec<f64> = (0..num_params)
            .map(|_| rand::random::<f64>() * 0.1 - 0.05)
            .collect();
        
        // Cost function
        let cost_function = |params: &[f64]| -> f64 {
            let weights = &params[0..self.basis_functions.len()];
            let intercept = params[self.basis_functions.len()];
            
            let mut total_error = 0.0;
            for (features, &target) in problem.features.iter().zip(problem.targets.iter()) {
                let mut prediction = intercept;
                
                for (i, basis_fn) in self.basis_functions.iter().enumerate() {
                    prediction += weights[i] * basis_fn(features, params);
                }
                
                let error = prediction - target;
                total_error += error * error;
            }
            
            total_error / problem.num_samples() as f64
        };
        
        // Optimize
        let mut annealer = QuantumAnnealer::new(self.config.clone())?;
        let result = annealer.optimize(cost_function, initial_params)?;
        
        self.parameters = Some(result.optimal_parameters);
        
        // Store metrics
        self.metrics.insert("mse".to_string(), result.optimal_energy);
        self.metrics.insert("rmse".to_string(), result.optimal_energy.sqrt());
        self.metrics.insert("iterations".to_string(), result.iterations as f64);
        
        Ok(())
    }
    
    /// Make predictions
    pub fn predict(&self, features: &[Vec<f64>]) -> QarResult<Vec<f64>> {
        let params = self.parameters.as_ref()
            .ok_or_else(|| QarError::RegressionError("Model not trained".to_string()))?;
        
        let weights = &params[0..self.basis_functions.len()];
        let intercept = params[self.basis_functions.len()];
        
        let predictions: Vec<f64> = features.iter()
            .map(|feature_vec| {
                let mut prediction = intercept;
                
                for (i, basis_fn) in self.basis_functions.iter().enumerate() {
                    prediction += weights[i] * basis_fn(feature_vec, params);
                }
                
                prediction
            })
            .collect();
        
        Ok(predictions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_quantum_linear_regression() {
        // Create simple linear problem: y = 2x + 1
        let features = vec![
            vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]
        ];
        let targets = vec![3.0, 5.0, 7.0, 9.0, 11.0];
        
        let problem = RegressionProblem::new(features, targets).unwrap();
        
        let config = QuantumAnnealingConfig {
            num_steps: 500,
            num_chains: 2,
            seed: Some(42),
            ..Default::default()
        };
        
        let mut model = QuantumLinearRegression::new(config);
        let result = model.fit(&problem);
        
        assert!(result.is_ok());
        assert!(model.coefficients.is_some());
        assert!(model.intercept.is_some());
        
        // Coefficients should be close to [2.0] and intercept close to 1.0
        let coef = model.coefficients.unwrap();
        let intercept = model.intercept.unwrap();
        
        assert_relative_eq!(coef[0], 2.0, epsilon = 0.5);
        assert_relative_eq!(intercept, 1.0, epsilon = 0.5);
        
        // Test prediction
        let test_features = vec![vec![6.0]];
        let predictions = model.predict(&test_features).unwrap();
        assert_relative_eq!(predictions[0], 13.0, epsilon = 1.0); // Should be close to 2*6 + 1 = 13
    }

    #[test]
    fn test_quantum_nonlinear_regression() {
        let config = QuantumAnnealingConfig {
            num_steps: 200,
            seed: Some(42),
            ..Default::default()
        };
        
        let mut model = QuantumNonLinearRegression::new(config);
        model.add_polynomial_basis(2); // Add x and x^2 terms
        
        // Create quadratic problem: y = x^2 + 2x + 1
        let features = vec![
            vec![1.0], vec![2.0], vec![3.0], vec![4.0]
        ];
        let targets = vec![4.0, 9.0, 16.0, 25.0]; // 1+2+1=4, 4+4+1=9, 9+6+1=16, 16+8+1=25
        
        let problem = RegressionProblem::new(features, targets).unwrap();
        
        let result = model.fit(&problem);
        assert!(result.is_ok());
        assert!(model.parameters.is_some());
        
        // Test prediction
        let test_features = vec![vec![5.0]];
        let predictions = model.predict(&test_features);
        assert!(predictions.is_ok());
    }

    #[test]
    fn test_model_without_training() {
        let config = QuantumAnnealingConfig::default();
        let model = QuantumLinearRegression::new(config);
        
        let test_features = vec![vec![1.0]];
        let result = model.predict(&test_features);
        
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), QarError::RegressionError(_)));
    }

    #[test]
    fn test_r_squared_calculation() {
        let features = vec![
            vec![1.0], vec![2.0], vec![3.0]
        ];
        let targets = vec![2.0, 4.0, 6.0]; // Perfect linear relationship: y = 2x
        
        let problem = RegressionProblem::new(features, targets).unwrap();
        
        let config = QuantumAnnealingConfig {
            num_steps: 300,
            seed: Some(42),
            ..Default::default()
        };
        
        let mut model = QuantumLinearRegression::new(config);
        model.fit(&problem).unwrap();
        
        // R-squared should be close to 1.0 for perfect linear relationship
        let r_squared = model.metrics.get("r_squared").unwrap();
        assert!(*r_squared > 0.9);
    }
}