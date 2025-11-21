//! Mock Contracts for London School TDD
//!
//! This module defines mock contracts and test doubles following the London School
//! mockist approach. Focuses on interaction testing and behavior verification.

use crate::{
    types::{AlignedVec, CalibrationScores, Confidence, Temperature, AtsCpVariant, AtsCpResult},
    config::{AtsCpConfig, QuantileMethod},
    error::{AtsCoreError, Result},
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Mock call recording for behavior verification
#[derive(Debug, Clone, PartialEq)]
pub struct MockCall {
    pub method_name: String,
    pub parameters: Vec<serde_json::Value>,
    pub return_value: Option<serde_json::Value>,
    pub execution_time: Duration,
    pub call_order: u64,
}

/// Mock expectation for contract verification
#[derive(Debug, Clone)]
pub struct MockExpectation {
    pub method_name: String,
    pub expected_parameters: Option<Vec<serde_json::Value>>,
    pub return_value: serde_json::Value,
    pub call_count: usize,
    pub max_calls: Option<usize>,
}

/// Mock registry for managing test doubles
pub struct MockRegistry {
    mocks: HashMap<String, Box<dyn MockBehavior + Send + Sync>>,
    call_history: Arc<Mutex<Vec<MockCall>>>,
    expectations: Arc<Mutex<Vec<MockExpectation>>>,
    call_counter: Arc<Mutex<u64>>,
}

/// Trait for mock behavior implementation
pub trait MockBehavior {
    fn handle_call(
        &mut self,
        method: &str,
        params: &[serde_json::Value],
    ) -> Result<serde_json::Value>;
    
    fn verify_expectations(&self) -> Result<Vec<String>>;
    fn reset(&mut self);
}

impl MockRegistry {
    pub fn new() -> Self {
        Self {
            mocks: HashMap::new(),
            call_history: Arc::new(Mutex::new(Vec::new())),
            expectations: Arc::new(Mutex::new(Vec::new())),
            call_counter: Arc::new(Mutex::new(0)),
        }
    }
    
    /// Register mock contracts for conformal predictor
    pub fn register_conformal_predictor_mocks(&mut self) -> Result<()> {
        let mock = Box::new(ConformalPredictorMock::new());
        self.mocks.insert("ConformalPredictor".to_string(), mock);
        println!("✅ Registered ConformalPredictor mock contracts");
        Ok(())
    }
    
    /// Register mock contracts for temperature scaler
    pub fn register_temperature_scaler_mocks(&mut self) -> Result<()> {
        let mock = Box::new(TemperatureScalerMock::new());
        self.mocks.insert("TemperatureScaler".to_string(), mock);
        println!("✅ Registered TemperatureScaler mock contracts");
        Ok(())
    }
    
    /// Register mock contracts for quantile computer
    pub fn register_quantile_computer_mocks(&mut self) -> Result<()> {
        let mock = Box::new(QuantileComputerMock::new());
        self.mocks.insert("QuantileComputer".to_string(), mock);
        println!("✅ Registered QuantileComputer mock contracts");
        Ok(())
    }
    
    /// Record mock call for behavior verification
    pub fn record_call(
        &self,
        method_name: String,
        parameters: Vec<serde_json::Value>,
        return_value: Option<serde_json::Value>,
        execution_time: Duration,
    ) -> Result<()> {
        let mut history = self.call_history.lock()
            .map_err(|_| AtsCoreError::concurrency("call_history", "lock failed"))?;
        
        let mut counter = self.call_counter.lock()
            .map_err(|_| AtsCoreError::concurrency("call_counter", "lock failed"))?;
        
        *counter += 1;
        
        let call = MockCall {
            method_name,
            parameters,
            return_value,
            execution_time,
            call_order: *counter,
        };
        
        history.push(call);
        Ok(())
    }
    
    /// Verify all mock expectations
    pub fn verify_all_expectations(&self) -> Result<Vec<String>> {
        let mut violations = Vec::new();
        
        for (name, mock) in &self.mocks {
            match mock.verify_expectations() {
                Ok(mock_violations) => violations.extend(mock_violations),
                Err(e) => violations.push(format!("Mock {} verification failed: {}", name, e)),
            }
        }
        
        Ok(violations)
    }
    
    /// Get call history for verification
    pub fn get_call_history(&self) -> Result<Vec<MockCall>> {
        let history = self.call_history.lock()
            .map_err(|_| AtsCoreError::concurrency("call_history", "lock failed"))?;
        
        Ok(history.clone())
    }
    
    /// Verify interaction sequence
    pub fn verify_interaction_sequence(&self, expected_sequence: &[&str]) -> Result<bool> {
        let history = self.get_call_history()?;
        
        if history.len() != expected_sequence.len() {
            return Ok(false);
        }
        
        for (i, call) in history.iter().enumerate() {
            if call.method_name != expected_sequence[i] {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    /// Reset all mocks
    pub fn reset_all(&mut self) {
        for mock in self.mocks.values_mut() {
            mock.reset();
        }
        
        if let Ok(mut history) = self.call_history.lock() {
            history.clear();
        }
        
        if let Ok(mut expectations) = self.expectations.lock() {
            expectations.clear();
        }
        
        if let Ok(mut counter) = self.call_counter.lock() {
            *counter = 0;
        }
    }
}

/// Mock implementation for ConformalPredictor
pub struct ConformalPredictorMock {
    expectations: Vec<MockExpectation>,
    call_counts: HashMap<String, usize>,
}

impl ConformalPredictorMock {
    pub fn new() -> Self {
        Self {
            expectations: Vec::new(),
            call_counts: HashMap::new(),
        }
    }
    
    /// Set expectation for predict method
    pub fn expect_predict(
        &mut self,
        expected_predictions: Vec<f64>,
        expected_calibration: Vec<f64>,
        return_intervals: Vec<(f64, f64)>,
    ) {
        let expectation = MockExpectation {
            method_name: "predict".to_string(),
            expected_parameters: Some(vec![
                serde_json::to_value(&expected_predictions).unwrap(),
                serde_json::to_value(&expected_calibration).unwrap(),
            ]),
            return_value: serde_json::to_value(&return_intervals).unwrap(),
            call_count: 0,
            max_calls: Some(1),
        };
        self.expectations.push(expectation);
    }
    
    /// Set expectation for ats_cp_predict method
    pub fn expect_ats_cp_predict(
        &mut self,
        expected_logits: Vec<f64>,
        expected_calibration_logits: Vec<Vec<f64>>,
        expected_labels: Vec<usize>,
        confidence: f64,
        variant: AtsCpVariant,
        return_result: AtsCpResult,
    ) {
        let expectation = MockExpectation {
            method_name: "ats_cp_predict".to_string(),
            expected_parameters: Some(vec![
                serde_json::to_value(&expected_logits).unwrap(),
                serde_json::to_value(&expected_calibration_logits).unwrap(),
                serde_json::to_value(&expected_labels).unwrap(),
                serde_json::to_value(&confidence).unwrap(),
                serde_json::to_value(&variant).unwrap(),
            ]),
            return_value: serde_json::to_value(&return_result).unwrap(),
            call_count: 0,
            max_calls: Some(1),
        };
        self.expectations.push(expectation);
    }
}

impl MockBehavior for ConformalPredictorMock {
    fn handle_call(
        &mut self,
        method: &str,
        params: &[serde_json::Value],
    ) -> Result<serde_json::Value> {
        *self.call_counts.entry(method.to_string()).or_insert(0) += 1;
        
        // Find matching expectation
        for expectation in &mut self.expectations {
            if expectation.method_name == method {
                if let Some(ref expected_params) = expectation.expected_parameters {
                    if params != expected_params.as_slice() {
                        continue;
                    }
                }
                
                expectation.call_count += 1;
                
                if let Some(max_calls) = expectation.max_calls {
                    if expectation.call_count > max_calls {
                        return Err(AtsCoreError::validation(
                            "mock_expectation",
                            &format!("Method {} called too many times", method),
                        ));
                    }
                }
                
                return Ok(expectation.return_value.clone());
            }
        }
        
        // Default behavior for undefined methods
        match method {
            "predict" => Ok(serde_json::json!(vec![(0.0, 1.0)])),
            "ats_cp_predict" => Ok(serde_json::json!({
                "conformal_set": [0],
                "calibrated_probabilities": [0.5, 0.3, 0.2],
                "optimal_temperature": 1.0,
                "quantile_threshold": 0.1,
                "coverage_guarantee": 0.95,
                "execution_time_ns": 1000,
                "variant": "GQ"
            })),
            _ => Err(AtsCoreError::validation("mock_method", &format!("Unknown method: {}", method))),
        }
    }
    
    fn verify_expectations(&self) -> Result<Vec<String>> {
        let mut violations = Vec::new();
        
        for expectation in &self.expectations {
            if expectation.call_count == 0 {
                violations.push(format!(
                    "Expected method '{}' was never called",
                    expectation.method_name
                ));
            }
            
            if let Some(max_calls) = expectation.max_calls {
                if expectation.call_count > max_calls {
                    violations.push(format!(
                        "Method '{}' was called {} times, expected at most {}",
                        expectation.method_name,
                        expectation.call_count,
                        max_calls
                    ));
                }
            }
        }
        
        Ok(violations)
    }
    
    fn reset(&mut self) {
        self.expectations.clear();
        self.call_counts.clear();
    }
}

/// Mock implementation for TemperatureScaler
pub struct TemperatureScalerMock {
    expectations: Vec<MockExpectation>,
    call_counts: HashMap<String, usize>,
}

impl TemperatureScalerMock {
    pub fn new() -> Self {
        Self {
            expectations: Vec::new(),
            call_counts: HashMap::new(),
        }
    }
    
    /// Set expectation for scale_logits method
    pub fn expect_scale_logits(
        &mut self,
        expected_logits: Vec<f64>,
        expected_temperature: f64,
        return_scaled: Vec<f64>,
    ) {
        let expectation = MockExpectation {
            method_name: "scale_logits".to_string(),
            expected_parameters: Some(vec![
                serde_json::to_value(&expected_logits).unwrap(),
                serde_json::to_value(&expected_temperature).unwrap(),
            ]),
            return_value: serde_json::to_value(&return_scaled).unwrap(),
            call_count: 0,
            max_calls: Some(1),
        };
        self.expectations.push(expectation);
    }
}

impl MockBehavior for TemperatureScalerMock {
    fn handle_call(
        &mut self,
        method: &str,
        params: &[serde_json::Value],
    ) -> Result<serde_json::Value> {
        *self.call_counts.entry(method.to_string()).or_insert(0) += 1;
        
        // Find matching expectation
        for expectation in &mut self.expectations {
            if expectation.method_name == method {
                if let Some(ref expected_params) = expectation.expected_parameters {
                    if params != expected_params.as_slice() {
                        continue;
                    }
                }
                
                expectation.call_count += 1;
                return Ok(expectation.return_value.clone());
            }
        }
        
        // Default behavior
        match method {
            "scale_logits" => Ok(serde_json::json!(vec![0.5, 0.3, 0.2])),
            "compute_optimal_temperature" => Ok(serde_json::json!(1.0)),
            _ => Err(AtsCoreError::validation("mock_method", &format!("Unknown method: {}", method))),
        }
    }
    
    fn verify_expectations(&self) -> Result<Vec<String>> {
        let mut violations = Vec::new();
        
        for expectation in &self.expectations {
            if expectation.call_count == 0 {
                violations.push(format!(
                    "Expected method '{}' was never called",
                    expectation.method_name
                ));
            }
        }
        
        Ok(violations)
    }
    
    fn reset(&mut self) {
        self.expectations.clear();
        self.call_counts.clear();
    }
}

/// Mock implementation for QuantileComputer
pub struct QuantileComputerMock {
    expectations: Vec<MockExpectation>,
    call_counts: HashMap<String, usize>,
}

impl QuantileComputerMock {
    pub fn new() -> Self {
        Self {
            expectations: Vec::new(),
            call_counts: HashMap::new(),
        }
    }
    
    /// Set expectation for compute_quantile method
    pub fn expect_compute_quantile(
        &mut self,
        expected_data: Vec<f64>,
        expected_confidence: f64,
        expected_method: QuantileMethod,
        return_quantile: f64,
    ) {
        let expectation = MockExpectation {
            method_name: "compute_quantile".to_string(),
            expected_parameters: Some(vec![
                serde_json::to_value(&expected_data).unwrap(),
                serde_json::to_value(&expected_confidence).unwrap(),
                serde_json::to_value(&expected_method).unwrap(),
            ]),
            return_value: serde_json::to_value(&return_quantile).unwrap(),
            call_count: 0,
            max_calls: Some(1),
        };
        self.expectations.push(expectation);
    }
}

impl MockBehavior for QuantileComputerMock {
    fn handle_call(
        &mut self,
        method: &str,
        params: &[serde_json::Value],
    ) -> Result<serde_json::Value> {
        *self.call_counts.entry(method.to_string()).or_insert(0) += 1;
        
        // Find matching expectation
        for expectation in &mut self.expectations {
            if expectation.method_name == method {
                if let Some(ref expected_params) = expectation.expected_parameters {
                    if params != expected_params.as_slice() {
                        continue;
                    }
                }
                
                expectation.call_count += 1;
                return Ok(expectation.return_value.clone());
            }
        }
        
        // Default behavior
        match method {
            "compute_quantile" => Ok(serde_json::json!(0.95)),
            "compute_quantile_linear" => Ok(serde_json::json!(0.95)),
            "compute_quantile_nearest" => Ok(serde_json::json!(0.95)),
            _ => Err(AtsCoreError::validation("mock_method", &format!("Unknown method: {}", method))),
        }
    }
    
    fn verify_expectations(&self) -> Result<Vec<String>> {
        let mut violations = Vec::new();
        
        for expectation in &self.expectations {
            if expectation.call_count == 0 {
                violations.push(format!(
                    "Expected method '{}' was never called",
                    expectation.method_name
                ));
            }
        }
        
        Ok(violations)
    }
    
    fn reset(&mut self) {
        self.expectations.clear();
        self.call_counts.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mock_registry_creation() {
        let registry = MockRegistry::new();
        assert_eq!(registry.mocks.len(), 0);
    }
    
    #[test]
    fn test_conformal_predictor_mock() {
        let mut mock = ConformalPredictorMock::new();
        
        // Set expectation
        mock.expect_predict(
            vec![1.0, 2.0, 3.0],
            vec![0.1, 0.2, 0.3],
            vec![(0.9, 1.1), (1.9, 2.1), (2.9, 3.1)],
        );
        
        // Verify expectation handling
        let params = vec![
            serde_json::json!([1.0, 2.0, 3.0]),
            serde_json::json!([0.1, 0.2, 0.3]),
        ];
        
        let result = mock.handle_call("predict", &params);
        assert!(result.is_ok());
        
        // Verify expectations
        let violations = mock.verify_expectations().unwrap();
        assert_eq!(violations.len(), 0);
    }
    
    #[test]
    fn test_mock_call_recording() {
        let registry = MockRegistry::new();
        
        let result = registry.record_call(
            "test_method".to_string(),
            vec![serde_json::json!("param1")],
            Some(serde_json::json!("result")),
            Duration::from_millis(1),
        );
        
        assert!(result.is_ok());
        
        let history = registry.get_call_history().unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].method_name, "test_method");
    }
}