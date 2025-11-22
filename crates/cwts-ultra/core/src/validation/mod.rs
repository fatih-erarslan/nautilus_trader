//! Data validation and integrity verification
//!
//! Provides cryptographic validation of market data to ensure
//! integrity, prevent replay attacks, and validate data quality.

pub mod authentic_data_processor;
pub mod autopoiesis_theory;
pub mod crypto_validator;
pub mod formal_verification;
pub mod ieee754_arithmetic;
pub mod mathematical_proofs;
pub mod performance_benchmarks;
pub mod root_cause_analyzer;
pub mod scientific_protocols;

pub use crypto_validator::{CryptographicDataValidator, ValidationError, ValidationStats};

pub use ieee754_arithmetic::FinancialCalculator;
pub use autopoiesis_theory::AutopoieticSystem;
pub use ieee754_arithmetic::ArithmeticError;
pub use mathematical_proofs::MathematicalValidator;

// Re-export commonly used types (note: some types not exported from modules)
// pub use ieee754_arithmetic::{ArithmeticResult, OverflowBehavior, SafeArithmetic};

pub use scientific_protocols::ValidationCriteria;
// pub use formal_constraints::ConstraintSolver; // Module not declared
