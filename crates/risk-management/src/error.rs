//! Error handling for the risk management system

use std::fmt;
use thiserror::Error;

/// Risk management errors
#[derive(Error, Debug)]
pub enum RiskError {
    #[error("VaR calculation error: {0}")]
    VarCalculation(String),
    
    #[error("CVaR calculation error: {0}")]
    CvarCalculation(String),
    
    #[error("Stress testing error: {0}")]
    StressTesting(String),
    
    #[error("Position sizing error: {0}")]
    PositionSizing(String),
    
    #[error("Portfolio optimization error: {0}")]
    PortfolioOptimization(String),
    
    #[error("Real-time monitoring error: {0}")]
    RealTimeMonitoring(String),
    
    #[error("Compliance error: {0}")]
    Compliance(String),
    
    #[error("Risk metrics calculation error: {0}")]
    RiskMetrics(String),
    
    #[error("Correlation analysis error: {0}")]
    CorrelationAnalysis(String),
    
    #[error("Monte Carlo simulation error: {0}")]
    MonteCarloSimulation(String),
    
    #[error("GPU acceleration error: {0}")]
    GpuAcceleration(String),
    
    #[error("Quantum uncertainty error: {0}")]
    QuantumUncertainty(String),
    
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    #[error("Data validation error: {0}")]
    DataValidation(String),
    
    #[error("Mathematical error: {0}")]
    Mathematical(String),
    
    #[error("Numerical instability: {0}")]
    NumericalInstability(String),
    
    #[error("Convergence error: {0}")]
    Convergence(String),
    
    #[error("Matrix operation error: {0}")]
    MatrixOperation(String),
    
    #[error("Optimization error: {0}")]
    Optimization(String),
    
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
    
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    #[error("Risk limit breach: {0}")]
    RiskLimitBreach(String),
    
    #[error("Performance constraint violation: {0}")]
    PerformanceConstraintViolation(String),
    
    #[error("Memory allocation error: {0}")]
    MemoryAllocation(String),
    
    #[error("Serialization error: {0}")]
    Serialization(String),
    
    #[error("Database error: {0}")]
    Database(String),
    
    #[error("Network error: {0}")]
    Network(String),
    
    #[error("Timeout error: {0}")]
    Timeout(String),
    
    #[error("Quantum circuit error: {0}")]
    QuantumCircuit(String),
    
    #[error("Copula model error: {0}")]
    CopulaModel(String),
    
    #[error("Kelly criterion error: {0}")]
    KellyCriterion(String),
    
    #[error("Regulatory compliance error: {0}")]
    RegulatoryCompliance(String),
    
    #[error("Reporting error: {0}")]
    Reporting(String),
    
    #[error("Audit trail error: {0}")]
    AuditTrail(String),
    
    #[error("Internal error: {0}")]
    Internal(String),
    
    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl RiskError {
    /// Create a VaR calculation error
    pub fn var_calculation<T: fmt::Display>(msg: T) -> Self {
        RiskError::VarCalculation(msg.to_string())
    }
    
    /// Create a CVaR calculation error
    pub fn cvar_calculation<T: fmt::Display>(msg: T) -> Self {
        RiskError::CvarCalculation(msg.to_string())
    }
    
    /// Create a stress testing error
    pub fn stress_testing<T: fmt::Display>(msg: T) -> Self {
        RiskError::StressTesting(msg.to_string())
    }
    
    /// Create a position sizing error
    pub fn position_sizing<T: fmt::Display>(msg: T) -> Self {
        RiskError::PositionSizing(msg.to_string())
    }
    
    /// Create a portfolio optimization error
    pub fn portfolio_optimization<T: fmt::Display>(msg: T) -> Self {
        RiskError::PortfolioOptimization(msg.to_string())
    }
    
    /// Create a real-time monitoring error
    pub fn real_time_monitoring<T: fmt::Display>(msg: T) -> Self {
        RiskError::RealTimeMonitoring(msg.to_string())
    }
    
    /// Create a compliance error
    pub fn compliance<T: fmt::Display>(msg: T) -> Self {
        RiskError::Compliance(msg.to_string())
    }
    
    /// Create a risk metrics error
    pub fn risk_metrics<T: fmt::Display>(msg: T) -> Self {
        RiskError::RiskMetrics(msg.to_string())
    }
    
    /// Create a correlation analysis error
    pub fn correlation_analysis<T: fmt::Display>(msg: T) -> Self {
        RiskError::CorrelationAnalysis(msg.to_string())
    }
    
    /// Create a Monte Carlo simulation error
    pub fn monte_carlo_simulation<T: fmt::Display>(msg: T) -> Self {
        RiskError::MonteCarloSimulation(msg.to_string())
    }
    
    /// Create a GPU acceleration error
    pub fn gpu_acceleration<T: fmt::Display>(msg: T) -> Self {
        RiskError::GpuAcceleration(msg.to_string())
    }
    
    /// Create a quantum uncertainty error
    pub fn quantum_uncertainty<T: fmt::Display>(msg: T) -> Self {
        RiskError::QuantumUncertainty(msg.to_string())
    }
    
    /// Create a configuration error
    pub fn configuration<T: fmt::Display>(msg: T) -> Self {
        RiskError::Configuration(msg.to_string())
    }
    
    /// Create a data validation error
    pub fn data_validation<T: fmt::Display>(msg: T) -> Self {
        RiskError::DataValidation(msg.to_string())
    }
    
    /// Create a mathematical error
    pub fn mathematical<T: fmt::Display>(msg: T) -> Self {
        RiskError::Mathematical(msg.to_string())
    }
    
    /// Create a numerical instability error
    pub fn numerical_instability<T: fmt::Display>(msg: T) -> Self {
        RiskError::NumericalInstability(msg.to_string())
    }
    
    /// Create a convergence error
    pub fn convergence<T: fmt::Display>(msg: T) -> Self {
        RiskError::Convergence(msg.to_string())
    }
    
    /// Create a matrix operation error
    pub fn matrix_operation<T: fmt::Display>(msg: T) -> Self {
        RiskError::MatrixOperation(msg.to_string())
    }
    
    /// Create an optimization error
    pub fn optimization<T: fmt::Display>(msg: T) -> Self {
        RiskError::Optimization(msg.to_string())
    }
    
    /// Create an insufficient data error
    pub fn insufficient_data<T: fmt::Display>(msg: T) -> Self {
        RiskError::InsufficientData(msg.to_string())
    }
    
    /// Create an invalid parameter error
    pub fn invalid_parameter<T: fmt::Display>(msg: T) -> Self {
        RiskError::InvalidParameter(msg.to_string())
    }
    
    /// Create a risk limit breach error
    pub fn risk_limit_breach<T: fmt::Display>(msg: T) -> Self {
        RiskError::RiskLimitBreach(msg.to_string())
    }
    
    /// Create a performance constraint violation error
    pub fn performance_constraint_violation<T: fmt::Display>(msg: T) -> Self {
        RiskError::PerformanceConstraintViolation(msg.to_string())
    }
    
    /// Create a memory allocation error
    pub fn memory_allocation<T: fmt::Display>(msg: T) -> Self {
        RiskError::MemoryAllocation(msg.to_string())
    }
    
    /// Create a serialization error
    pub fn serialization<T: fmt::Display>(msg: T) -> Self {
        RiskError::Serialization(msg.to_string())
    }
    
    /// Create a database error
    pub fn database<T: fmt::Display>(msg: T) -> Self {
        RiskError::Database(msg.to_string())
    }
    
    /// Create a network error
    pub fn network<T: fmt::Display>(msg: T) -> Self {
        RiskError::Network(msg.to_string())
    }
    
    /// Create a timeout error
    pub fn timeout<T: fmt::Display>(msg: T) -> Self {
        RiskError::Timeout(msg.to_string())
    }
    
    /// Create a quantum circuit error
    pub fn quantum_circuit<T: fmt::Display>(msg: T) -> Self {
        RiskError::QuantumCircuit(msg.to_string())
    }
    
    /// Create a copula model error
    pub fn copula_model<T: fmt::Display>(msg: T) -> Self {
        RiskError::CopulaModel(msg.to_string())
    }
    
    /// Create a Kelly criterion error
    pub fn kelly_criterion<T: fmt::Display>(msg: T) -> Self {
        RiskError::KellyCriterion(msg.to_string())
    }
    
    /// Create a regulatory compliance error
    pub fn regulatory_compliance<T: fmt::Display>(msg: T) -> Self {
        RiskError::RegulatoryCompliance(msg.to_string())
    }
    
    /// Create a reporting error
    pub fn reporting<T: fmt::Display>(msg: T) -> Self {
        RiskError::Reporting(msg.to_string())
    }
    
    /// Create an audit trail error
    pub fn audit_trail<T: fmt::Display>(msg: T) -> Self {
        RiskError::AuditTrail(msg.to_string())
    }
    
    /// Create an internal error
    pub fn internal<T: fmt::Display>(msg: T) -> Self {
        RiskError::Internal(msg.to_string())
    }
    
    /// Create an unknown error
    pub fn unknown<T: fmt::Display>(msg: T) -> Self {
        RiskError::Unknown(msg.to_string())
    }
}

/// Result type for risk management operations
pub type RiskResult<T> = Result<T, RiskError>;

/// Error context for debugging
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub operation: String,
    pub component: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub additional_info: std::collections::HashMap<String, String>,
}

impl ErrorContext {
    pub fn new(operation: &str, component: &str) -> Self {
        Self {
            operation: operation.to_string(),
            component: component.to_string(),
            timestamp: chrono::Utc::now(),
            additional_info: std::collections::HashMap::new(),
        }
    }
    
    pub fn with_info<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: fmt::Display,
    {
        self.additional_info.insert(key.into(), value.to_string());
        self
    }
}

/// Trait for operations that can provide error context
pub trait WithErrorContext<T> {
    fn with_context(self, context: ErrorContext) -> RiskResult<T>;
}

impl<T, E> WithErrorContext<T> for Result<T, E>
where
    E: fmt::Display,
{
    fn with_context(self, context: ErrorContext) -> RiskResult<T> {
        match self {
            Ok(value) => Ok(value),
            Err(error) => {
                let error_msg = format!(
                    "Operation '{}' failed in component '{}' at {}: {} (Additional info: {:?})",
                    context.operation,
                    context.component,
                    context.timestamp,
                    error,
                    context.additional_info
                );
                Err(RiskError::internal(error_msg))
            }
        }
    }
}

/// Macro for creating error contexts
#[macro_export]
macro_rules! error_context {
    ($operation:expr, $component:expr) => {
        ErrorContext::new($operation, $component)
    };
    ($operation:expr, $component:expr, $($key:expr => $value:expr),*) => {
        {
            let mut context = ErrorContext::new($operation, $component);
            $(
                context.additional_info.insert($key.to_string(), $value.to_string());
            )*
            context
        }
    };
}

/// Macro for wrapping operations with error context
#[macro_export]
macro_rules! with_context {
    ($operation:expr, $context:expr) => {
        $operation.with_context($context)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = RiskError::var_calculation("Test error");
        assert!(matches!(error, RiskError::VarCalculation(_)));
    }

    #[test]
    fn test_error_context() {
        let context = ErrorContext::new("test_operation", "test_component")
            .with_info("param1", "value1")
            .with_info("param2", 42);
        
        assert_eq!(context.operation, "test_operation");
        assert_eq!(context.component, "test_component");
        assert_eq!(context.additional_info.len(), 2);
    }

    #[test]
    fn test_with_error_context() {
        let result: Result<i32, &str> = Err("test error");
        let context = ErrorContext::new("test_op", "test_comp");
        
        let risk_result = result.with_context(context);
        assert!(risk_result.is_err());
        
        if let Err(RiskError::Internal(msg)) = risk_result {
            assert!(msg.contains("test_op"));
            assert!(msg.contains("test_comp"));
            assert!(msg.contains("test error"));
        } else {
            panic!("Expected Internal error");
        }
    }
}