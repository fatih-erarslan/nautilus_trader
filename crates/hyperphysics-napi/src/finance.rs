//! Finance module - Black-Scholes and options pricing
//!
//! Zero-overhead bindings for financial computations.

use napi::bindgen_prelude::*;
use napi_derive::napi;
use crate::OptionPriceResult;

/// Calculate Black-Scholes option price and Greeks
pub fn calculate_option_price_internal(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    time_to_maturity: f64,
) -> Result<OptionPriceResult> {
    // Validate inputs
    if spot <= 0.0 || strike <= 0.0 {
        return Err(Error::new(Status::InvalidArg, "Spot and strike must be positive"));
    }
    if volatility <= 0.0 {
        return Err(Error::new(Status::InvalidArg, "Volatility must be positive"));
    }
    if time_to_maturity <= 0.0 {
        return Err(Error::new(Status::InvalidArg, "Time to maturity must be positive"));
    }

    use hyperphysics_finance::{calculate_black_scholes, OptionParams};

    let params = OptionParams {
        spot,
        strike,
        rate,
        volatility,
        time_to_maturity,
    };

    let (call_price, greeks) = calculate_black_scholes(&params)
        .map_err(|e| Error::new(Status::GenericFailure, format!("Calculation failed: {:?}", e)))?;

    // Put-call parity: P = C - S + K*e^(-rT)
    let put_price = call_price - spot + strike * (-rate * time_to_maturity).exp();

    Ok(OptionPriceResult {
        call_price,
        put_price,
        delta: greeks.delta,
        gamma: greeks.gamma,
        vega: greeks.vega,
        theta: greeks.theta,
        rho: greeks.rho,
    })
}

/// Black-Scholes option pricer
#[napi]
pub struct BlackScholes;

#[napi]
impl BlackScholes {
    /// Calculate call option price
    #[napi]
    pub fn call_price(
        spot: f64,
        strike: f64,
        rate: f64,
        volatility: f64,
        time_to_maturity: f64,
    ) -> Result<f64> {
        let result = calculate_option_price_internal(spot, strike, rate, volatility, time_to_maturity)?;
        Ok(result.call_price)
    }

    /// Calculate put option price
    #[napi]
    pub fn put_price(
        spot: f64,
        strike: f64,
        rate: f64,
        volatility: f64,
        time_to_maturity: f64,
    ) -> Result<f64> {
        let result = calculate_option_price_internal(spot, strike, rate, volatility, time_to_maturity)?;
        Ok(result.put_price)
    }

    /// Calculate all Greeks
    #[napi]
    pub fn greeks(
        spot: f64,
        strike: f64,
        rate: f64,
        volatility: f64,
        time_to_maturity: f64,
    ) -> Result<OptionPriceResult> {
        calculate_option_price_internal(spot, strike, rate, volatility, time_to_maturity)
    }
}
