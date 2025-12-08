//! Security Test Module
//! 
//! This module contains comprehensive security tests for the Talebian Risk Management System.
//! All tests are designed to prevent capital loss and ensure system integrity under adversarial conditions.

pub mod test_malicious_input_protection;
pub mod test_fuzzing_comprehensive;
pub mod test_financial_invariants_security;
pub mod test_end_to_end_security_workflows;

/// Re-export security test functions for easy access
pub use test_malicious_input_protection::*;
pub use test_fuzzing_comprehensive::*;
pub use test_financial_invariants_security::*;
pub use test_end_to_end_security_workflows::*;