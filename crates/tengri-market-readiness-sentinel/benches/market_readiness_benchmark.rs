//! Market Readiness Sentinel Benchmarks
//!
//! Performance benchmarks for market readiness validation components:
//! - Volatility calculations (GARCH, realized, historical)
//! - Regime detection algorithms (HMM, ensemble)
//! - Validation status aggregation
//! - Risk metrics computation

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::collections::VecDeque;
use std::f64;

/// Simulate returns data for volatility benchmarks
fn generate_returns(n: usize) -> Vec<f64> {
    // Geometric Brownian motion style returns
    let mut returns = Vec::with_capacity(n);
    let mut seed = 42u64;

    for _ in 0..n {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = ((seed >> 33) as f64) / (1u64 << 31) as f64;
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = ((seed >> 33) as f64) / (1u64 << 31) as f64;

        // Box-Muller transform for normal distribution
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        returns.push(z * 0.02); // 2% daily volatility
    }
    returns
}

/// Calculate realized volatility (annualized)
fn calculate_realized_volatility(returns: &[f64]) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>() / (returns.len() - 1) as f64;

    variance.sqrt() * (252.0_f64).sqrt() // Annualize
}

/// Calculate historical volatility with rolling window
fn calculate_historical_volatility(returns: &[f64], window: usize) -> Vec<f64> {
    if returns.len() < window {
        return vec![];
    }

    let mut vols = Vec::with_capacity(returns.len() - window + 1);

    for i in 0..=(returns.len() - window) {
        let window_returns = &returns[i..i + window];
        vols.push(calculate_realized_volatility(window_returns));
    }

    vols
}

/// GARCH(1,1) volatility forecast
fn garch_forecast(returns: &[f64], omega: f64, alpha: f64, beta: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    // Initialize variance with sample variance
    let mut variance = returns.iter().map(|r| r.powi(2)).sum::<f64>() / returns.len() as f64;

    // Iterate GARCH equation
    for &r in returns {
        variance = omega + alpha * r.powi(2) + beta * variance;
    }

    variance.sqrt()
}

/// Volatility of volatility calculation
fn calculate_vol_of_vol(returns: &[f64], window: usize) -> f64 {
    let vols = calculate_historical_volatility(returns, window);

    if vols.len() < 2 {
        return 0.0;
    }

    let mean_vol = vols.iter().sum::<f64>() / vols.len() as f64;
    let variance = vols.iter()
        .map(|v| (v - mean_vol).powi(2))
        .sum::<f64>() / (vols.len() - 1) as f64;

    variance.sqrt()
}

/// Regime detection using simple threshold-based classification
#[derive(Debug, Clone, Copy, PartialEq)]
enum MarketRegime {
    Calm,
    Normal,
    Volatile,
    Crisis,
}

fn detect_regime(volatility: f64, correlation: f64, volume_ratio: f64) -> (MarketRegime, f64) {
    // Ensemble scoring for regime detection
    let mut scores = [0.0f64; 4]; // Calm, Normal, Volatile, Crisis

    // Volatility-based scoring
    if volatility < 0.1 {
        scores[0] += 0.4;
        scores[1] += 0.3;
    } else if volatility < 0.2 {
        scores[1] += 0.4;
        scores[2] += 0.3;
    } else if volatility < 0.35 {
        scores[2] += 0.4;
        scores[3] += 0.3;
    } else {
        scores[3] += 0.5;
        scores[2] += 0.2;
    }

    // Correlation-based scoring
    if correlation > 0.8 {
        scores[3] += 0.3;
    } else if correlation > 0.6 {
        scores[2] += 0.2;
        scores[1] += 0.1;
    } else {
        scores[0] += 0.2;
        scores[1] += 0.2;
    }

    // Volume-based scoring
    if volume_ratio > 2.0 {
        scores[3] += 0.2;
        scores[2] += 0.1;
    } else if volume_ratio > 1.5 {
        scores[2] += 0.2;
    } else if volume_ratio < 0.7 {
        scores[0] += 0.2;
    } else {
        scores[1] += 0.2;
    }

    // Find highest scoring regime
    let (idx, &confidence) = scores.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();

    let regime = match idx {
        0 => MarketRegime::Calm,
        1 => MarketRegime::Normal,
        2 => MarketRegime::Volatile,
        _ => MarketRegime::Crisis,
    };

    (regime, confidence)
}

/// Hidden Markov Model transition probability calculation
fn hmm_transition_probability(
    current_regime: usize,
    observations: &[f64],
    transition_matrix: &[[f64; 4]; 4],
    emission_means: &[f64; 4],
    emission_vars: &[f64; 4],
) -> [f64; 4] {
    let mut state_probs = [0.0f64; 4];
    state_probs[current_regime] = 1.0;

    for &obs in observations {
        let mut new_probs = [0.0f64; 4];

        for next_state in 0..4 {
            for prev_state in 0..4 {
                // Transition probability
                let trans_prob = transition_matrix[prev_state][next_state];

                // Emission probability (Gaussian)
                let mean = emission_means[next_state];
                let var = emission_vars[next_state];
                let diff = obs - mean;
                let emission_prob = (-diff * diff / (2.0 * var)).exp() / (2.0 * std::f64::consts::PI * var).sqrt();

                new_probs[next_state] += state_probs[prev_state] * trans_prob * emission_prob;
            }
        }

        // Normalize
        let sum: f64 = new_probs.iter().sum();
        if sum > 0.0 {
            for p in &mut new_probs {
                *p /= sum;
            }
        }

        state_probs = new_probs;
    }

    state_probs
}

/// VaR (Value at Risk) calculation using historical simulation
fn calculate_var(returns: &[f64], confidence: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let idx = ((1.0 - confidence) * sorted.len() as f64).floor() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Expected Shortfall (CVaR) calculation
fn calculate_expected_shortfall(returns: &[f64], confidence: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let var = calculate_var(returns, confidence);
    let tail_returns: Vec<f64> = returns.iter()
        .filter(|&&r| r <= var)
        .cloned()
        .collect();

    if tail_returns.is_empty() {
        return var;
    }

    tail_returns.iter().sum::<f64>() / tail_returns.len() as f64
}

/// Market impact estimation using Almgren-Chriss model
fn estimate_market_impact(
    volume: f64,
    avg_daily_volume: f64,
    volatility: f64,
    spread: f64,
) -> f64 {
    let participation_rate = volume / avg_daily_volume;

    // Temporary impact (linear)
    let temp_impact = spread * 0.5 + volatility * participation_rate.sqrt() * 0.1;

    // Permanent impact (square root)
    let perm_impact = volatility * participation_rate.sqrt() * 0.05;

    temp_impact + perm_impact
}

/// Validation status aggregation
#[derive(Debug, Clone, Copy)]
enum ValidationStatus {
    Passed,
    Warning,
    Failed,
}

fn aggregate_validations(statuses: &[(ValidationStatus, f64)]) -> (ValidationStatus, f64) {
    if statuses.is_empty() {
        return (ValidationStatus::Passed, 1.0);
    }

    let mut failed_count = 0;
    let mut warning_count = 0;
    let mut total_confidence = 0.0;

    for (status, confidence) in statuses {
        match status {
            ValidationStatus::Failed => failed_count += 1,
            ValidationStatus::Warning => warning_count += 1,
            ValidationStatus::Passed => {}
        }
        total_confidence += confidence;
    }

    let avg_confidence = total_confidence / statuses.len() as f64;

    let overall_status = if failed_count > 0 {
        ValidationStatus::Failed
    } else if warning_count > 0 {
        ValidationStatus::Warning
    } else {
        ValidationStatus::Passed
    };

    (overall_status, avg_confidence)
}

// ============== BENCHMARKS ==============

fn bench_volatility_calculations(c: &mut Criterion) {
    let mut group = c.benchmark_group("volatility");

    for size in [100, 252, 504, 1000] {
        let returns = generate_returns(size);

        group.throughput(Throughput::Elements(size as u64));

        // Realized volatility
        group.bench_with_input(
            BenchmarkId::new("realized", size),
            &returns,
            |b, data| {
                b.iter(|| black_box(calculate_realized_volatility(data)));
            },
        );

        // Historical volatility (20-day window)
        group.bench_with_input(
            BenchmarkId::new("historical_20d", size),
            &returns,
            |b, data| {
                b.iter(|| black_box(calculate_historical_volatility(data, 20)));
            },
        );

        // GARCH forecast
        group.bench_with_input(
            BenchmarkId::new("garch", size),
            &returns,
            |b, data| {
                b.iter(|| black_box(garch_forecast(data, 0.01, 0.1, 0.8)));
            },
        );

        // Vol of vol
        group.bench_with_input(
            BenchmarkId::new("vol_of_vol", size),
            &returns,
            |b, data| {
                b.iter(|| black_box(calculate_vol_of_vol(data, 20)));
            },
        );
    }

    group.finish();
}

fn bench_regime_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("regime_detection");

    // Generate test scenarios
    let scenarios: Vec<(f64, f64, f64)> = vec![
        (0.08, 0.4, 0.9),   // Calm
        (0.15, 0.5, 1.0),   // Normal
        (0.28, 0.7, 1.6),   // Volatile
        (0.45, 0.85, 2.5),  // Crisis
    ];

    // Simple regime detection
    group.bench_function("threshold_based", |b| {
        b.iter(|| {
            for &(vol, corr, volume) in &scenarios {
                black_box(detect_regime(vol, corr, volume));
            }
        });
    });

    // HMM-based detection
    let transition_matrix = [
        [0.7, 0.2, 0.08, 0.02],
        [0.2, 0.6, 0.15, 0.05],
        [0.05, 0.15, 0.6, 0.2],
        [0.02, 0.08, 0.2, 0.7],
    ];
    let emission_means = [0.08, 0.15, 0.25, 0.4];
    let emission_vars = [0.01, 0.02, 0.04, 0.08];

    for obs_count in [10, 50, 100] {
        let observations: Vec<f64> = (0..obs_count)
            .map(|i| 0.15 + 0.02 * (i as f64 * 0.1).sin())
            .collect();

        group.bench_with_input(
            BenchmarkId::new("hmm", obs_count),
            &observations,
            |b, obs| {
                b.iter(|| {
                    black_box(hmm_transition_probability(
                        1, // Start in Normal regime
                        obs,
                        &transition_matrix,
                        &emission_means,
                        &emission_vars,
                    ))
                });
            },
        );
    }

    group.finish();
}

fn bench_risk_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("risk_metrics");

    for size in [100, 252, 1000] {
        let returns = generate_returns(size);

        group.throughput(Throughput::Elements(size as u64));

        // VaR at 95%
        group.bench_with_input(
            BenchmarkId::new("var_95", size),
            &returns,
            |b, data| {
                b.iter(|| black_box(calculate_var(data, 0.95)));
            },
        );

        // VaR at 99%
        group.bench_with_input(
            BenchmarkId::new("var_99", size),
            &returns,
            |b, data| {
                b.iter(|| black_box(calculate_var(data, 0.99)));
            },
        );

        // Expected Shortfall
        group.bench_with_input(
            BenchmarkId::new("expected_shortfall", size),
            &returns,
            |b, data| {
                b.iter(|| black_box(calculate_expected_shortfall(data, 0.95)));
            },
        );
    }

    group.finish();
}

fn bench_market_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("market_impact");

    // Different trade sizes
    let scenarios: Vec<(f64, f64, f64, f64)> = vec![
        (10000.0, 1000000.0, 0.15, 0.001),    // Small trade
        (100000.0, 1000000.0, 0.15, 0.001),   // Medium trade
        (500000.0, 1000000.0, 0.15, 0.001),   // Large trade
        (100000.0, 1000000.0, 0.35, 0.002),   // High volatility
    ];

    group.bench_function("almgren_chriss", |b| {
        b.iter(|| {
            for &(vol, adv, volatility, spread) in &scenarios {
                black_box(estimate_market_impact(vol, adv, volatility, spread));
            }
        });
    });

    group.finish();
}

fn bench_validation_aggregation(c: &mut Criterion) {
    let mut group = c.benchmark_group("validation");

    for count in [5, 10, 20, 50] {
        // Generate mixed validation results
        let validations: Vec<(ValidationStatus, f64)> = (0..count)
            .map(|i| {
                let status = match i % 10 {
                    0 => ValidationStatus::Failed,
                    1..=3 => ValidationStatus::Warning,
                    _ => ValidationStatus::Passed,
                };
                let confidence = 0.7 + (i as f64 % 30.0) * 0.01;
                (status, confidence)
            })
            .collect();

        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(
            BenchmarkId::new("aggregate", count),
            &validations,
            |b, data| {
                b.iter(|| black_box(aggregate_validations(data)));
            },
        );
    }

    group.finish();
}

fn bench_rolling_calculations(c: &mut Criterion) {
    let mut group = c.benchmark_group("rolling");

    let returns = generate_returns(1000);

    // Rolling z-score calculation
    group.bench_function("z_score_252d", |b| {
        b.iter(|| {
            let mut z_scores = Vec::with_capacity(returns.len() - 252);
            let mut window: VecDeque<f64> = VecDeque::with_capacity(252);

            for &r in &returns {
                window.push_back(r);
                if window.len() > 252 {
                    window.pop_front();
                }

                if window.len() == 252 {
                    let mean = window.iter().sum::<f64>() / 252.0;
                    let variance = window.iter()
                        .map(|x| (x - mean).powi(2))
                        .sum::<f64>() / 251.0;
                    let std = variance.sqrt();

                    if std > 0.0 {
                        z_scores.push((r - mean) / std);
                    }
                }
            }
            black_box(z_scores)
        });
    });

    // Rolling percentile rank
    group.bench_function("percentile_252d", |b| {
        b.iter(|| {
            let mut percentiles = Vec::with_capacity(returns.len() - 252);
            let mut window: Vec<f64> = Vec::with_capacity(252);

            for &r in &returns {
                if window.len() >= 252 {
                    window.remove(0);
                }
                window.push(r);

                if window.len() == 252 {
                    let mut sorted = window.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

                    let pos = sorted.iter().position(|&x| x >= r).unwrap_or(252);
                    percentiles.push(pos as f64 / 252.0 * 100.0);
                }
            }
            black_box(percentiles)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_volatility_calculations,
    bench_regime_detection,
    bench_risk_metrics,
    bench_market_impact,
    bench_validation_aggregation,
    bench_rolling_calculations,
);

criterion_main!(benches);
