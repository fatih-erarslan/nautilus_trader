/// Black-Scholes validation tests using academic benchmarks
///
/// These tests validate our implementation against known values from:
/// - Hull, J. (2018). "Options, Futures, and Other Derivatives" (10th ed.)
/// - Haug, E. G. (2007). "The Complete Guide to Option Pricing Formulas" (2nd ed.)

use hyperphysics_finance::risk::{OptionParams, calculate_black_scholes, calculate_put_greeks};
use approx::assert_relative_eq;

/// Hull (2018), Example 15.6, Page 338
/// European call option with S=42, K=40, r=0.1, σ=0.2, T=0.5
#[test]
fn test_hull_example_15_6() {
    let params = OptionParams {
        spot: 42.0,
        strike: 40.0,
        rate: 0.10,
        volatility: 0.20,
        time_to_maturity: 0.5,
    };

    let (call_price, greeks) = calculate_black_scholes(&params).unwrap();

    // Hull gives C = 4.76 (rounded)
    assert_relative_eq!(call_price, 4.76, epsilon = 0.01);

    // Delta should be positive and less than 1
    assert!(greeks.delta > 0.7 && greeks.delta < 0.9);
}

/// Hull (2018), Example 19.1, Page 406
/// At-the-money option pricing
#[test]
fn test_hull_example_19_1() {
    let params = OptionParams {
        spot: 49.0,
        strike: 50.0,
        rate: 0.05,
        volatility: 0.20,
        time_to_maturity: 0.3846,  // 20 weeks
    };

    let (call_price, greeks) = calculate_black_scholes(&params).unwrap();

    // Hull gives C ≈ 2.40
    assert_relative_eq!(call_price, 2.40, epsilon = 0.05);

    // Delta ≈ 0.522
    assert_relative_eq!(greeks.delta, 0.522, epsilon = 0.01);
}

/// Put-call parity validation
/// C - P = S - K·e^(-rT)
#[test]
fn test_put_call_parity_validation() {
    let test_cases = vec![
        // (S, K, r, σ, T)
        (100.0, 100.0, 0.05, 0.20, 1.0),
        (50.0, 55.0, 0.08, 0.25, 0.75),
        (200.0, 180.0, 0.03, 0.15, 2.0),
        (75.0, 75.0, 0.06, 0.30, 0.5),
    ];

    for (spot, strike, rate, vol, time) in test_cases {
        let params = OptionParams {
            spot,
            strike,
            rate,
            volatility: vol,
            time_to_maturity: time,
        };

        let (call_price, _) = calculate_black_scholes(&params).unwrap();
        let (put_price, _) = calculate_put_greeks(&params).unwrap();

        let discount_factor = (-rate * time).exp();
        let pv_strike = strike * discount_factor;

        // Put-call parity: C - P = S - K·e^(-rT)
        let lhs = call_price - put_price;
        let rhs = spot - pv_strike;

        assert_relative_eq!(lhs, rhs, epsilon = 1e-10);
    }
}

/// Test Greeks relationships
#[test]
fn test_greeks_relationships() {
    let params = OptionParams {
        spot: 100.0,
        strike: 100.0,
        rate: 0.05,
        volatility: 0.20,
        time_to_maturity: 1.0,
    };

    let (_, call_greeks) = calculate_black_scholes(&params).unwrap();
    let (_, put_greeks) = calculate_put_greeks(&params).unwrap();

    // Delta relationship: Δ_put = Δ_call - 1
    assert_relative_eq!(put_greeks.delta, call_greeks.delta - 1.0, epsilon = 1e-10);

    // Gamma is equal for calls and puts
    assert_relative_eq!(call_greeks.gamma, put_greeks.gamma, epsilon = 1e-10);

    // Vega is equal for calls and puts
    assert_relative_eq!(call_greeks.vega, put_greeks.vega, epsilon = 1e-10);

    // All greeks should be finite
    assert!(call_greeks.delta.is_finite());
    assert!(call_greeks.gamma.is_finite());
    assert!(call_greeks.vega.is_finite());
    assert!(call_greeks.theta.is_finite());
    assert!(call_greeks.rho.is_finite());
}

/// Test deep in-the-money call (S >> K)
#[test]
fn test_deep_itm_call() {
    let params = OptionParams {
        spot: 150.0,
        strike: 100.0,
        rate: 0.05,
        volatility: 0.20,
        time_to_maturity: 1.0,
    };

    let (call_price, greeks) = calculate_black_scholes(&params).unwrap();

    // Delta should approach 1.0 for deep ITM calls
    assert!(greeks.delta > 0.95);

    // Gamma should be low (option deep in money)
    assert!(greeks.gamma < 0.01);

    // Price should be approximately S - K·e^(-rT)
    let pv_strike = params.strike * (-params.rate * params.time_to_maturity).exp();
    let intrinsic = params.spot - pv_strike;
    assert_relative_eq!(call_price, intrinsic, epsilon = 5.0);
}

/// Test deep out-of-the-money call (S << K)
#[test]
fn test_deep_otm_call() {
    let params = OptionParams {
        spot: 50.0,
        strike: 100.0,
        rate: 0.05,
        volatility: 0.20,
        time_to_maturity: 1.0,
    };

    let (call_price, greeks) = calculate_black_scholes(&params).unwrap();

    // Delta should approach 0 for deep OTM calls
    assert!(greeks.delta < 0.1);

    // Price should be close to 0
    assert!(call_price < 1.0);
}

/// Test option value increases with volatility
#[test]
fn test_volatility_monotonicity() {
    let base_params = OptionParams {
        spot: 100.0,
        strike: 100.0,
        rate: 0.05,
        volatility: 0.10,
        time_to_maturity: 1.0,
    };

    let volatilities = vec![0.10, 0.15, 0.20, 0.25, 0.30];
    let mut previous_price = 0.0;

    for &vol in &volatilities {
        let params = OptionParams {
            volatility: vol,
            ..base_params
        };

        let (price, greeks) = calculate_black_scholes(&params).unwrap();

        // Price should increase with volatility
        assert!(price > previous_price);
        previous_price = price;

        // Vega should be positive
        assert!(greeks.vega > 0.0);
    }
}

/// Test time decay (theta)
#[test]
fn test_time_decay() {
    let times = vec![2.0, 1.5, 1.0, 0.5, 0.25];
    let mut previous_price = f64::INFINITY;

    for &time in &times {
        let params = OptionParams {
            spot: 100.0,
            strike: 100.0,
            rate: 0.05,
            volatility: 0.20,
            time_to_maturity: time,
        };

        let (price, greeks) = calculate_black_scholes(&params).unwrap();

        // Price should decrease as time approaches expiry
        assert!(price < previous_price);
        previous_price = price;

        // Theta should be negative (time decay)
        assert!(greeks.theta < 0.0);
    }
}

/// Test interest rate sensitivity (rho)
#[test]
fn test_interest_rate_sensitivity() {
    let rates = vec![0.02, 0.04, 0.06, 0.08, 0.10];
    let mut previous_call_price = 0.0;

    for &rate in &rates {
        let params = OptionParams {
            spot: 100.0,
            strike: 100.0,
            rate,
            volatility: 0.20,
            time_to_maturity: 1.0,
        };

        let (call_price, greeks) = calculate_black_scholes(&params).unwrap();

        // Call price increases with interest rate
        assert!(call_price > previous_call_price);
        previous_call_price = call_price;

        // Rho should be positive for calls
        assert!(greeks.rho > 0.0);
    }
}
