//! Formal verification system for HyperPhysics
//! 
//! This crate provides enterprise-grade formal verification using:
//! - Z3 SMT solver for property verification
//! - Property-based testing with QuickCheck/PropTest
//! - Runtime invariant checking
//! - Mathematical proof validation
//!
//! # Architecture
//! 
//! The verification system operates on multiple levels:
//! 1. **Static Verification**: Z3 SMT proofs of mathematical properties
//! 2. **Dynamic Testing**: Property-based testing with generated inputs
//! 3. **Runtime Checking**: Invariant validation during execution
//! 4. **Proof Pipeline**: Automated verification workflow

pub mod z3_verifier;
pub mod property_testing;
pub mod invariant_checker;
pub mod proof_pipeline;
pub mod mathematical_properties;

pub use z3_verifier::*;
pub use property_testing::*;
pub use invariant_checker::*;
pub use proof_pipeline::*;

use thiserror::Error;

/// Verification system errors
#[derive(Error, Debug)]
pub enum VerificationError {
    #[error("Z3 solver error: {0}")]
    Z3Error(String),
    
    #[error("Property test failed: {0}")]
    PropertyTestFailed(String),
    
    #[error("Invariant violation: {0}")]
    InvariantViolation(String),
    
    #[error("Mathematical proof failed: {0}")]
    ProofFailed(String),
    
    #[error("Verification timeout: {0}")]
    Timeout(String),
}

pub type VerificationResult<T> = Result<T, VerificationError>;

/// Verification report containing all test results
#[derive(Debug, Clone, serde::Serialize)]
pub struct VerificationReport {
    pub z3_proofs: Vec<ProofResult>,
    pub property_tests: Vec<PropertyTestResult>,
    pub invariant_checks: Vec<InvariantResult>,
    pub overall_status: VerificationStatus,
    pub timestamp: std::time::SystemTime,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ProofResult {
    pub property_name: String,
    pub status: ProofStatus,
    pub proof_time_ms: u64,
    pub details: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PropertyTestResult {
    pub test_name: String,
    pub status: TestStatus,
    pub test_cases: u32,
    pub failures: u32,
    pub details: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct InvariantResult {
    pub invariant_name: String,
    pub status: InvariantStatus,
    pub violations: u32,
    pub details: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub enum VerificationStatus {
    Passed,
    Failed,
    Partial,
    Timeout,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub enum ProofStatus {
    Proven,
    Disproven,
    Unknown,
    Timeout,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub enum TestStatus {
    Passed,
    Failed,
    Timeout,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub enum InvariantStatus {
    Satisfied,
    Violated,
    Unknown,
}

impl VerificationReport {
    pub fn new() -> Self {
        Self {
            z3_proofs: Vec::new(),
            property_tests: Vec::new(),
            invariant_checks: Vec::new(),
            overall_status: VerificationStatus::Passed,
            timestamp: std::time::SystemTime::now(),
        }
    }
    
    pub fn add_proof_result(&mut self, result: ProofResult) {
        if result.status != ProofStatus::Proven {
            self.overall_status = VerificationStatus::Failed;
        }
        self.z3_proofs.push(result);
    }
    
    pub fn add_property_test(&mut self, result: PropertyTestResult) {
        if result.status != TestStatus::Passed {
            self.overall_status = VerificationStatus::Failed;
        }
        self.property_tests.push(result);
    }
    
    pub fn add_invariant_check(&mut self, result: InvariantResult) {
        if result.status != InvariantStatus::Satisfied {
            self.overall_status = VerificationStatus::Failed;
        }
        self.invariant_checks.push(result);
    }
    
    pub fn is_fully_verified(&self) -> bool {
        self.overall_status == VerificationStatus::Passed
    }
}

impl Default for VerificationReport {
    fn default() -> Self {
        Self::new()
    }
}
