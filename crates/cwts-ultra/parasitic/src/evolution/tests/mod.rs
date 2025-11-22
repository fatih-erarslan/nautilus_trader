//! Test modules for evolution engine components
//! All tests follow TDD methodology with ZERO mocks policy

#[cfg(test)]
pub mod genetic_algorithm_tests;

#[cfg(test)]  
pub mod fitness_evaluator_tests;

#[cfg(test)]
pub mod mutation_engine_tests;

// Integration tests for the complete evolution engine
#[cfg(test)]
pub mod evolution_integration_tests;

// Performance benchmarks
#[cfg(test)]
pub mod performance_benchmarks;