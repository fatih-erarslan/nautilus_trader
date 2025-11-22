#pragma once

#include <cstddef>

extern "C" {

/**
 * C++ High-Performance Probabilistic Computing Library
 * Header file for CWTS probabilistic algorithms
 */

// Forward declaration
struct ProbabilisticComputeEngine;
struct HeavyTailParams;
struct BenchmarkResults;
struct KalmanState;
struct RegimeSwitchingModel;

// Heavy-tail distribution parameters
struct HeavyTailParams {
    double degrees_of_freedom;
    double location;
    double scale;
    double tail_index;
    double log_likelihood;
};

// Performance benchmark results
struct BenchmarkResults {
    double monte_carlo_duration_ms;
    double fft_duration_ms;
    double matrix_mult_duration_ms;
    double heavy_tail_estimation_ms;
    double total_duration_ms;
};

// Kalman filter state
struct KalmanState {
    double state_estimate;
    double error_covariance;
    double process_noise;
    double measurement_noise;
};

// Regime switching model
struct RegimeSwitchingModel {
    static const int NUM_REGIMES = 4;
    double transition_matrix[NUM_REGIMES][NUM_REGIMES];
    double emission_params[NUM_REGIMES][2]; // [mean, variance] for each regime
    double state_probabilities[NUM_REGIMES];
    double log_likelihood;
};

// Engine management
ProbabilisticComputeEngine* create_compute_engine(size_t buffer_size);
void destroy_compute_engine(ProbabilisticComputeEngine* engine);

// Monte Carlo simulations
void simd_monte_carlo_var(
    double* returns_out,
    double portfolio_value,
    double mean_return,
    double volatility,
    size_t iterations,
    unsigned int seed
);

void antithetic_monte_carlo_var(
    double* returns_out,
    double portfolio_value,
    double mean_return,
    double volatility,
    size_t iterations,
    unsigned int seed
);

void control_variates_monte_carlo(
    double* adjusted_returns,
    const double* raw_returns,
    double control_mean,
    double beta_coefficient,
    size_t size
);

// Heavy-tail distribution analysis
HeavyTailParams estimate_heavy_tail_params(const double* data, size_t size);

// FFT-based option pricing
void fft_option_pricing(
    double* option_prices,
    double spot_price,
    double risk_free_rate,
    double dividend_yield,
    double volatility,
    double time_to_expiry,
    const double* strike_prices,
    size_t num_strikes,
    int option_type
);

// Matrix operations
void parallel_matrix_multiply(
    double* result,
    const double* matrix_a,
    const double* matrix_b,
    size_t rows_a,
    size_t cols_a,
    size_t cols_b
);

// Filtering and estimation
void kalman_filter_update(
    KalmanState* state,
    double observation,
    double control_input
);

void update_regime_probabilities(
    RegimeSwitchingModel* model,
    double new_observation
);

// Performance benchmarking
BenchmarkResults run_performance_benchmark(size_t problem_size);

// C interface functions
void compute_monte_carlo_var_c(
    double* returns,
    double portfolio_value,
    double mean_return,
    double volatility,
    size_t iterations
);

HeavyTailParams estimate_tail_params_c(const double* data, size_t size);
BenchmarkResults benchmark_c(size_t problem_size);

} // extern "C"