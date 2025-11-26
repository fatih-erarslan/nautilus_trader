/*!
 * NAPI-RS bindings for performance-critical dynamic pricing operations
 */

#![deny(clippy::all)]

use napi::bindgen_prelude::*;
use napi_derive::napi;
use ndarray::{Array1, Array2};

/// Fast elasticity calculation using vectorized operations
#[napi]
pub fn calculate_elasticity_fast(prices: Vec<f64>, demands: Vec<f64>) -> Result<f64> {
    if prices.len() != demands.len() || prices.len() < 2 {
        return Err(Error::from_reason("Invalid input lengths"));
    }

    let n = prices.len() as f64;

    // Convert to log space
    let log_prices: Vec<f64> = prices.iter().map(|&p| p.ln()).collect();
    let log_demands: Vec<f64> = demands.iter().map(|&d| d.ln()).collect();

    // Calculate means
    let mean_log_p: f64 = log_prices.iter().sum::<f64>() / n;
    let mean_log_d: f64 = log_demands.iter().sum::<f64>() / n;

    // Calculate covariance and variance
    let mut cov = 0.0;
    let mut var = 0.0;

    for i in 0..prices.len() {
        let diff_p = log_prices[i] - mean_log_p;
        let diff_d = log_demands[i] - mean_log_d;
        cov += diff_p * diff_d;
        var += diff_p * diff_p;
    }

    if var < 1e-10 {
        return Err(Error::from_reason("Zero variance in prices"));
    }

    // Elasticity = cov(log(P), log(D)) / var(log(P))
    Ok(cov / var)
}

/// Batch demand prediction using elasticity
#[napi]
pub fn predict_demand_batch(
    prices: Vec<f64>,
    base_price: f64,
    base_demand: f64,
    elasticity: f64,
) -> Result<Vec<f64>> {
    let demands: Vec<f64> = prices
        .iter()
        .map(|&price| {
            let price_change = (price - base_price) / base_price;
            let demand_change = elasticity * price_change;
            (base_demand * (1.0 + demand_change)).max(0.0)
        })
        .collect();

    Ok(demands)
}

/// Fast Q-value update for reinforcement learning
#[napi]
pub fn q_learning_update_batch(
    q_values: Vec<f64>,
    rewards: Vec<f64>,
    next_q_values: Vec<f64>,
    learning_rate: f64,
    discount_factor: f64,
) -> Result<Vec<f64>> {
    if q_values.len() != rewards.len() || q_values.len() != next_q_values.len() {
        return Err(Error::from_reason("Mismatched input lengths"));
    }

    let updated: Vec<f64> = q_values
        .iter()
        .zip(rewards.iter())
        .zip(next_q_values.iter())
        .map(|((&q, &r), &next_q)| {
            q + learning_rate * (r + discount_factor * next_q - q)
        })
        .collect();

    Ok(updated)
}

/// Calculate revenue optimization score
#[napi]
pub fn calculate_revenue_scores(
    prices: Vec<f64>,
    demands: Vec<f64>,
) -> Result<Vec<f64>> {
    if prices.len() != demands.len() {
        return Err(Error::from_reason("Mismatched input lengths"));
    }

    let revenues: Vec<f64> = prices
        .iter()
        .zip(demands.iter())
        .map(|(&p, &d)| p * d)
        .collect();

    Ok(revenues)
}

/// Fast competitive position analysis
#[napi]
pub struct CompetitiveMetrics {
    pub avg_price: f64,
    pub min_price: f64,
    pub max_price: f64,
    pub dispersion: f64,
}

#[napi]
pub fn analyze_competition_fast(prices: Vec<f64>) -> Result<CompetitiveMetrics> {
    if prices.is_empty() {
        return Err(Error::from_reason("Empty price list"));
    }

    let n = prices.len() as f64;
    let avg = prices.iter().sum::<f64>() / n;
    let min = prices.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Calculate standard deviation
    let variance: f64 = prices.iter().map(|&p| (p - avg).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();
    let dispersion = std_dev / avg;

    Ok(CompetitiveMetrics {
        avg_price: avg,
        min_price: min,
        max_price: max,
        dispersion,
    })
}

/// Vectorized conformal prediction intervals
#[napi]
pub fn conformal_intervals(
    predictions: Vec<f64>,
    calibration_scores: Vec<f64>,
    alpha: f64,
) -> Result<Vec<(f64, f64, f64)>> {
    if calibration_scores.is_empty() {
        return Err(Error::from_reason("Empty calibration scores"));
    }

    // Calculate quantile
    let mut sorted_scores = calibration_scores.clone();
    sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted_scores.len();
    let quantile_idx = ((n as f64 + 1.0) * (1.0 - alpha)).ceil() as usize - 1;
    let quantile_idx = quantile_idx.min(n - 1);
    let conformal_score = sorted_scores[quantile_idx];

    // Create intervals for each prediction
    let intervals: Vec<(f64, f64, f64)> = predictions
        .iter()
        .map(|&pred| {
            let lower = (pred - conformal_score).max(0.0);
            let upper = pred + conformal_score;
            (pred, lower, upper)
        })
        .collect();

    Ok(intervals)
}

/// Multi-armed bandit UCB calculation
#[napi]
pub fn calculate_ucb_scores(
    avg_rewards: Vec<f64>,
    counts: Vec<u32>,
    total_pulls: u32,
    exploration_constant: f64,
) -> Result<Vec<f64>> {
    if avg_rewards.len() != counts.len() {
        return Err(Error::from_reason("Mismatched input lengths"));
    }

    let log_total = (total_pulls as f64).ln();

    let ucb_scores: Vec<f64> = avg_rewards
        .iter()
        .zip(counts.iter())
        .map(|(&avg, &count)| {
            if count == 0 {
                return f64::INFINITY;
            }
            let exploration_bonus = exploration_constant * (log_total / count as f64).sqrt();
            avg + exploration_bonus
        })
        .collect();

    Ok(ucb_scores)
}

/// Fast price optimization using gradient ascent
#[napi]
pub fn optimize_price_gradient(
    initial_price: f64,
    base_demand: f64,
    elasticity: f64,
    iterations: u32,
    learning_rate: f64,
) -> Result<f64> {
    let mut price = initial_price;

    for _ in 0..iterations {
        // Revenue = price * demand
        // demand = base_demand * (price / initial_price)^elasticity
        let demand = base_demand * (price / initial_price).powf(elasticity);

        // Gradient of revenue w.r.t. price
        // d(Revenue)/d(price) = demand + price * d(demand)/d(price)
        let demand_gradient = base_demand * elasticity * (price / initial_price).powf(elasticity - 1.0) / initial_price;
        let revenue_gradient = demand + price * demand_gradient;

        // Gradient ascent
        price += learning_rate * revenue_gradient;

        // Clamp to reasonable bounds
        price = price.max(initial_price * 0.5).min(initial_price * 2.0);
    }

    Ok(price)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elasticity_calculation() {
        let prices = vec![90.0, 95.0, 100.0, 105.0, 110.0];
        let demands = vec![120.0, 110.0, 100.0, 90.0, 80.0];

        let elasticity = calculate_elasticity_fast(prices, demands).unwrap();

        // Should be negative (inverse relationship)
        assert!(elasticity < 0.0);
        // Should be elastic (magnitude > 1)
        assert!(elasticity.abs() > 0.5);
    }

    #[test]
    fn test_demand_prediction() {
        let prices = vec![90.0, 100.0, 110.0];
        let predictions = predict_demand_batch(prices, 100.0, 100.0, -1.5).unwrap();

        // Lower price should give higher demand
        assert!(predictions[0] > predictions[1]);
        assert!(predictions[1] > predictions[2]);
    }
}
