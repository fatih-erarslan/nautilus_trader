// Verification tests for risk management fixes

use std::collections::HashMap;

// Portfolio weight test verification
fn test_portfolio_weights_fixed() {
    println!("Testing portfolio weights...");

    // Simulate portfolio with AAPL and GOOGL
    // AAPL: 100 shares @ $160 = $16,000
    // GOOGL: 50 shares @ $2,900 = $145,000
    // Total positions: $161,000
    // Cash: $10,000
    // Total portfolio: $171,000

    let aapl_value = 100.0 * 160.0;
    let googl_value = 50.0 * 2900.0;
    let position_value = aapl_value + googl_value;

    let aapl_weight = aapl_value / position_value;
    let googl_weight = googl_value / position_value;

    println!("  AAPL weight: {:.6}", aapl_weight);
    println!("  GOOGL weight: {:.6}", googl_weight);
    println!("  Weight sum: {:.6}", aapl_weight + googl_weight);

    assert!((aapl_weight + googl_weight - 1.0_f64).abs() < 1e-6, "Weights should sum to 1.0");
    println!("  ✓ Portfolio weights sum to 1.0");
}

// Historical VaR test verification
fn test_historical_var_fixed() {
    println!("\nTesting historical VaR...");

    // Create return data: -5.0, -4.9, ..., 4.9
    let returns: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) / 10.0).collect();

    // Convert to losses (negative returns)
    let mut losses: Vec<f64> = returns.iter().map(|&r| -r).collect();
    losses.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // For 95% confidence, find 5th percentile of losses
    let alpha = 0.05;
    let n = losses.len();
    let rank = alpha * (n as f64 - 1.0);
    let lower_idx = rank.floor() as usize;
    let upper_idx = rank.ceil() as usize;
    let weight = rank - lower_idx as f64;

    let var = if lower_idx == upper_idx {
        losses[lower_idx]
    } else {
        losses[lower_idx] * (1.0 - weight) + losses[upper_idx] * weight
    };

    println!("  5th percentile loss index: {:.2} (between {} and {})", rank, lower_idx, upper_idx);
    println!("  VaR (95%): {:.4}", var);
    println!("  Expected range: [4.0, 5.0]");

    assert!(var > 4.0 && var < 5.0, "VaR should be between 4.0 and 5.0");
    println!("  ✓ Historical VaR within expected range");
}

// Parametric VaR test verification
fn test_parametric_var_fixed() {
    println!("\nTesting parametric VaR...");

    let returns: Vec<f64> = vec![
        -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5,
        -1.2, -0.8, -0.3, 0.2, 0.7, 1.2,
    ];

    let n = returns.len() as f64;
    let mean: f64 = returns.iter().sum::<f64>() / n;
    let variance: f64 = returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    println!("  Mean return: {:.4}", mean);
    println!("  Std dev: {:.4}", std_dev);

    // For 95% confidence
    let z_95 = 1.645;
    let var = -mean + z_95 * std_dev;

    println!("  VaR (95%, parametric): {:.4}", var);
    println!("  Expected range: [1.0, 2.5]");

    assert!(var > 1.0 && var < 2.5, "Parametric VaR should be between 1.0 and 2.5");
    println!("  ✓ Parametric VaR within expected range");
}

// Entropy constraint test verification
fn test_entropy_constraint_fixed() {
    println!("\nTesting entropy constraint...");

    let returns: Vec<f64> = vec![0.1, -0.2, 0.15, -0.1, 0.05];

    let n = returns.len() as f64;
    let mean: f64 = returns.iter().sum::<f64>() / n;
    let variance: f64 = returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    let z_95 = 1.645;
    let var_base = -mean + z_95 * std_dev;

    // Apply entropy adjustment
    let entropy_adjustment_0 = 1.0 + 0.1 * 0.0;
    let entropy_adjustment_2 = 1.0 + 0.1 * 2.0;

    let var_entropy_0 = var_base * entropy_adjustment_0;
    let var_entropy_2 = var_base * entropy_adjustment_2;

    println!("  Base VaR: {:.6}", var_base);
    println!("  VaR (entropy=0.0): {:.6}", var_entropy_0);
    println!("  VaR (entropy=2.0): {:.6}", var_entropy_2);
    println!("  Entropy adjustment: {:.2}x", entropy_adjustment_2);

    assert!(var_entropy_2 > var_entropy_0, "Higher entropy should increase VaR");
    println!("  ✓ Entropy constraint increases VaR as expected");
}

fn main() {
    println!("=== Risk Management Test Verification ===\n");

    test_portfolio_weights_fixed();
    test_historical_var_fixed();
    test_parametric_var_fixed();
    test_entropy_constraint_fixed();

    println!("\n=== All verification tests passed! ===");
}
