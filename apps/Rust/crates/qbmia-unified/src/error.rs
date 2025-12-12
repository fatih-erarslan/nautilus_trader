//! Error types for QBMIA Unified
//! 
//! TENGRI-compliant error handling with strict validation

use thiserror::Error;

/// Main error type for QBMIA operations
#[derive(Error, Debug)]
pub enum QBMIAError {
    /// Hardware-related errors
    #[error("Hardware error: {0}")]
    Hardware(String),
    
    /// Invalid input data
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    /// Network/API errors
    #[error("Network error: {0}")]
    NetworkError(String),
    
    /// Parsing errors
    #[error("Parsing error: {0}")]
    Parsing(String),
    
    /// Numerical computation errors
    #[error("Numerical error: {0}")]
    Numerical(String),
    
    /// Quantum simulation errors
    #[error("Quantum simulation error: {0}")]
    QuantumSimulation(String),
    
    /// Biological processing errors
    #[error("Biological processing error: {0}")]
    BiologicalProcessing(String),
    
    /// TENGRI compliance violations
    #[error("TENGRI violation: {0}")]
    TengriViolation(String),
}

impl QBMIAError {
    /// Create hardware error
    pub fn hardware(msg: impl Into<String>) -> Self {
        Self::Hardware(msg.into())
    }
    
    /// Create invalid input error
    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }
    
    /// Create network error
    pub fn network_error(msg: impl Into<String>) -> Self {
        Self::NetworkError(msg.into())
    }
    
    /// Create parsing error
    pub fn parsing(msg: impl Into<String>) -> Self {
        Self::Parsing(msg.into())
    }
    
    /// Create numerical error
    pub fn numerical(msg: impl Into<String>) -> Self {
        Self::Numerical(msg.into())
    }
    
    /// Create quantum simulation error
    pub fn quantum_simulation(msg: impl Into<String>) -> Self {
        Self::QuantumSimulation(msg.into())
    }
    
    /// Create biological processing error
    pub fn biological_processing(msg: impl Into<String>) -> Self {
        Self::BiologicalProcessing(msg.into())
    }
    
    /// Create TENGRI violation error
    pub fn tengri_violation(msg: impl Into<String>) -> Self {
        Self::TengriViolation(msg.into())
    }
}

/// Result type for QBMIA operations
pub type Result<T> = std::result::Result<T, QBMIAError>;

/// Validation trait for TENGRI compliance
pub trait TengriCompliant {
    /// Validate that no mock data is used
    fn validate_no_mock_data(&self) -> Result<()>;
    
    /// Validate that real data sources are used
    fn validate_real_data_sources(&self) -> Result<()>;
    
    /// Validate GPU-only quantum computation
    fn validate_gpu_quantum(&self) -> Result<()>;
}

/// Mock data detection utility
pub struct MockDataDetector;

impl MockDataDetector {
    /// Check if data appears to be mock/synthetic
    pub fn is_mock_data(data: &[f64]) -> bool {
        if data.is_empty() {
            return true;
        }
        
        // Check for patterns that suggest mock data
        
        // 1. Perfect mathematical sequences
        if Self::is_arithmetic_sequence(data) || Self::is_geometric_sequence(data) {
            return true;
        }
        
        // 2. Unrealistic uniformity
        if Self::is_too_uniform(data) {
            return true;
        }
        
        // 3. Obvious test values
        if Self::contains_test_values(data) {
            return true;
        }
        
        false
    }
    
    /// Check if data is an arithmetic sequence
    fn is_arithmetic_sequence(data: &[f64]) -> bool {
        if data.len() < 3 {
            return false;
        }
        
        let diff = data[1] - data[0];
        for i in 2..data.len() {
            if (data[i] - data[i-1] - diff).abs() > 1e-10 {
                return false;
            }
        }
        true
    }
    
    /// Check if data is a geometric sequence
    fn is_geometric_sequence(data: &[f64]) -> bool {
        if data.len() < 3 || data.iter().any(|&x| x.abs() < 1e-10) {
            return false;
        }
        
        let ratio = data[1] / data[0];
        for i in 2..data.len() {
            if (data[i] / data[i-1] - ratio).abs() > 1e-6 {
                return false;
            }
        }
        true
    }
    
    /// Check if data is too uniform (suggesting artificial generation)
    fn is_too_uniform(data: &[f64]) -> bool {
        if data.len() < 10 {
            return false;
        }
        
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        let std_dev = variance.sqrt();
        
        // If standard deviation is too small relative to mean, it might be artificial
        if mean.abs() > 1e-6 {
            let cv = std_dev / mean.abs(); // Coefficient of variation
            if cv < 0.001 {
                return true; // Too uniform
            }
        }
        
        false
    }
    
    /// Check for obvious test values
    fn contains_test_values(data: &[f64]) -> bool {
        for &value in data {
            // Common test values
            if (value - 42.0).abs() < 1e-10 ||
               (value - 123.456).abs() < 1e-6 ||
               (value - 100.0).abs() < 1e-10 ||
               (value - 1000.0).abs() < 1e-10 {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mock_data_detection() {
        // Test arithmetic sequence
        let arithmetic = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(MockDataDetector::is_mock_data(&arithmetic));
        
        // Test geometric sequence
        let geometric = vec![1.0, 2.0, 4.0, 8.0, 16.0];
        assert!(MockDataDetector::is_mock_data(&geometric));
        
        // Test realistic data (should not be flagged as mock)
        let realistic = vec![45000.1, 45123.7, 44987.3, 45234.8, 45089.2];
        assert!(!MockDataDetector::is_mock_data(&realistic));
        
        // Test test values
        let test_values = vec![42.0, 100.0, 123.456];
        assert!(MockDataDetector::is_mock_data(&test_values));
    }
    
    #[test]
    fn test_error_creation() {
        let error = QBMIAError::tengri_violation("Mock data detected");
        assert!(matches!(error, QBMIAError::TengriViolation(_)));
        
        let error = QBMIAError::hardware("CUDA not available");
        assert!(matches!(error, QBMIAError::Hardware(_)));
    }
}