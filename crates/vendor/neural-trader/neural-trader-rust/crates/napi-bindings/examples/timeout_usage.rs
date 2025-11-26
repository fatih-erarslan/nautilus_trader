//! Example demonstrating timeout and resource limit usage in NAPI bindings
//!
//! This example shows how to use the timeout wrapper and validation functions
//! to protect async operations and validate inputs.

use anyhow::{anyhow, Result};
use tokio::time::{sleep, Duration};

// Simulated imports (these would come from the actual crate)
// In practice: use neural_trader_backend::utils::*;

// For demonstration, we'll define simplified versions
async fn with_timeout<F, T>(
    future: F,
    seconds: u64,
    operation: &str,
) -> Result<T>
where
    F: std::future::Future<Output = Result<T>>,
{
    match tokio::time::timeout(Duration::from_secs(seconds), future).await {
        Ok(result) => result,
        Err(_) => Err(anyhow!("{} timed out after {}s", operation, seconds)),
    }
}

// Example: Neural training with timeout
async fn neural_train_example() -> Result<String> {
    println!("Starting neural training with timeout...");

    // Wrap potentially long-running operation with timeout
    let result = with_timeout(
        async {
            println!("  Training neural network...");
            sleep(Duration::from_secs(2)).await;
            Ok("Training completed successfully".to_string())
        },
        300, // 5 minute timeout
        "neural_train"
    ).await?;

    println!("  {}", result);
    Ok(result)
}

// Example: Trading operation with timeout
async fn execute_trade_example() -> Result<String> {
    println!("Executing trade with timeout...");

    let result = with_timeout(
        async {
            println!("  Placing order...");
            sleep(Duration::from_millis(500)).await;
            Ok("Order executed".to_string())
        },
        30, // 30 second timeout
        "execute_trade"
    ).await?;

    println!("  {}", result);
    Ok(result)
}

// Example: Operation that times out
async fn slow_operation_example() -> Result<String> {
    println!("Starting slow operation (will timeout)...");

    let result = with_timeout(
        async {
            println!("  Starting slow computation...");
            sleep(Duration::from_secs(5)).await;
            Ok("This will never complete".to_string())
        },
        2, // 2 second timeout
        "slow_operation"
    ).await;

    match result {
        Ok(_) => println!("  Operation completed"),
        Err(e) => println!("  Operation failed: {}", e),
    }

    result
}

// Example: Validation functions
fn validation_examples() {
    println!("\nValidation Examples:");

    // JSON size validation
    let large_json = "x".repeat(1_000_000);
    let small_json = r#"{"key": "value"}"#;

    println!("  Validating JSON sizes:");
    println!("    Small JSON ({}): {}", small_json.len(),
             if small_json.len() <= 1_000_000 { "✓ OK" } else { "✗ Too large" });
    println!("    Large JSON ({}): {}", large_json.len(),
             if large_json.len() <= 1_000_000 { "✓ OK" } else { "✗ Too large" });

    // Array length validation
    println!("\n  Validating array lengths:");
    println!("    Array[100]: {}", if 100 <= 10_000 { "✓ OK" } else { "✗ Too large" });
    println!("    Array[10001]: {}", if 10_001 <= 10_000 { "✓ OK" } else { "✗ Too large" });

    // Numeric validations
    println!("\n  Validating numeric values:");
    println!("    Positive(1.0): {}", if 1.0 > 0.0 { "✓ OK" } else { "✗ Invalid" });
    println!("    Positive(0.0): {}", if 0.0 > 0.0 { "✓ OK" } else { "✗ Invalid" });
    println!("    Percentage(0.5): {}", if (0.0..=1.0).contains(&0.5) { "✓ OK" } else { "✗ Invalid" });
    println!("    Percentage(1.5): {}", if (0.0..=1.0).contains(&1.5) { "✓ OK" } else { "✗ Invalid" });
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Timeout and Resource Limit Examples ===\n");

    // Run examples
    neural_train_example().await?;
    println!();

    execute_trade_example().await?;
    println!();

    let _ = slow_operation_example().await; // Expected to fail

    validation_examples();

    println!("\n=== Summary ===");
    println!("✓ Timeouts protect against hung operations");
    println!("✓ Validation prevents invalid inputs");
    println!("✓ Clear error messages for debugging");
    println!("✓ Consistent patterns across all functions");

    Ok(())
}
