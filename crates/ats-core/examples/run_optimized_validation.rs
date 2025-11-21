//! Example demonstrating the optimized conformal prediction implementation
//!
//! This example runs validation tests and benchmarks to verify that the
//! optimization goals are met.

use ats_core::conformal_optimized_standalone_test;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 ATS-Core Optimized Conformal Prediction Validation");
    println!("====================================================");
    
    // Run comprehensive validation
    let report = conformal_optimized_standalone_test::run_validation()?;
    
    // The report summary is already printed by the validation function
    
    println!("\n💡 Next Steps:");
    println!("- Run full benchmark suite with: cargo bench optimized_conformal_benchmarks");
    println!("- Generate performance report with: cargo run --bin performance_comparison_report");
    println!("- Profile with: cargo bench --bench optimized_conformal_benchmarks --profile=release");
    
    Ok(())
}