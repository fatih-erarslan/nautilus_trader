/// Black-Scholes Greeks Implementation
///
/// References:
/// - Black, F., & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities"
///   Journal of Political Economy, 81(3), 637-654.
/// - Hull, J. (2018). "Options, Futures, and Other Derivatives" (10th ed.)
///   Pearson Education, ISBN: 978-0134472089
///
/// Mathematical Formulas:
///
/// Call Option Price:
/// ```latex
/// C = S·N(d₁) - K·e^(-rτ)·N(d₂)
/// ```
///
/// Greeks:
/// ```latex
/// Delta (Δ)  = ∂C/∂S = N(d₁)
/// Gamma (Γ)  = ∂²C/∂S² = φ(d₁) / (S·σ·√τ)
/// Vega  (ν)  = ∂C/∂σ = S·φ(d₁)·√τ
/// Theta (Θ)  = ∂C/∂t = -S·φ(d₁)·σ/(2√τ) - r·K·e^(-rτ)·N(d₂)
/// Rho   (ρ)  = ∂C/∂r = K·τ·e^(-rτ)·N(d₂)
/// ```
///
/// Where:
/// - S: Spot price
/// - K: Strike price
/// - r: Risk-free rate
/// - σ: Volatility
/// - τ: Time to maturity (years)
/// - N(x): Standard normal CDF
/// - φ(x): Standard normal PDF
use statrs::distribution::{Normal, ContinuousCDF};
use crate::types::FinanceError;

const SQRT_2PI: f64 = 2.5066282746310002;  // √(2π)

/// Black-Scholes option parameters
#[derive(Debug, Clone, Copy)]
pub struct OptionParams {
    pub spot: f64,           // S: Current asset price
    pub strike: f64,         // K: Strike price
    pub rate: f64,           // r: Risk-free rate (annualized)
    pub volatility: f64,     // σ: Volatility (annualized)
    pub time_to_maturity: f64, // τ: Time to maturity (years)
}

impl OptionParams {
    /// Validate option parameters
    pub fn validate(&self) -> Result<(), FinanceError> {
        if self.spot <= 0.0 {
            return Err(FinanceError::InvalidOptionParams(
                format!("Spot price must be positive: {}", self.spot)
            ));
        }
        if self.strike <= 0.0 {
            return Err(FinanceError::InvalidOptionParams(
                format!("Strike price must be positive: {}", self.strike)
            ));
        }
        if self.volatility <= 0.0 {
            return Err(FinanceError::InvalidOptionParams(
                format!("Volatility must be positive: {}", self.volatility)
            ));
        }
        if self.time_to_maturity <= 0.0 {
            return Err(FinanceError::InvalidOptionParams(
                format!("Time to maturity must be positive: {}", self.time_to_maturity)
            ));
        }
        Ok(())
    }

    /// Calculate d₁ = [ln(S/K) + (r + σ²/2)τ] / (σ√τ)
    fn d1(&self) -> f64 {
        let ln_s_k = (self.spot / self.strike).ln();
        let variance_term = (self.rate + 0.5 * self.volatility.powi(2)) * self.time_to_maturity;
        let denominator = self.volatility * self.time_to_maturity.sqrt();
        (ln_s_k + variance_term) / denominator
    }

    /// Calculate d₂ = d₁ - σ√τ
    fn d2(&self) -> f64 {
        self.d1() - self.volatility * self.time_to_maturity.sqrt()
    }

    /// Standard normal PDF: φ(x) = (1/√(2π)) · e^(-x²/2)
    fn pdf(x: f64) -> f64 {
        (-0.5 * x.powi(2)).exp() / SQRT_2PI
    }

    /// Standard normal CDF: N(x)
    fn cdf(x: f64) -> f64 {
        let normal = Normal::new(0.0, 1.0).unwrap();
        normal.cdf(x)
    }
}

/// Black-Scholes Greeks for European call options
#[derive(Debug, Clone, Copy)]
pub struct Greeks {
    /// Delta: ∂C/∂S - sensitivity to spot price
    pub delta: f64,

    /// Gamma: ∂²C/∂S² - rate of change of delta
    pub gamma: f64,

    /// Vega: ∂C/∂σ - sensitivity to volatility (× 0.01 for 1% change)
    pub vega: f64,

    /// Theta: ∂C/∂t - time decay (per day, divide by 365)
    pub theta: f64,

    /// Rho: ∂C/∂r - sensitivity to interest rate (× 0.01 for 1% change)
    pub rho: f64,
}

/// Calculate Black-Scholes call option price and Greeks
///
/// # Example
/// ```rust
/// use hyperphysics_finance::risk::greeks::*;
///
/// let params = OptionParams {
///     spot: 100.0,
///     strike: 100.0,
///     rate: 0.05,
///     volatility: 0.20,
///     time_to_maturity: 1.0,
/// };
///
/// let (price, greeks) = calculate_black_scholes(&params).unwrap();
/// ```
pub fn calculate_black_scholes(params: &OptionParams) -> Result<(f64, Greeks), FinanceError> {
    params.validate()?;

    let d1 = params.d1();
    let d2 = params.d2();

    let nd1 = OptionParams::cdf(d1);
    let nd2 = OptionParams::cdf(d2);
    let phi_d1 = OptionParams::pdf(d1);

    let sqrt_tau = params.time_to_maturity.sqrt();
    let discount_factor = (-params.rate * params.time_to_maturity).exp();

    // Call option price: C = S·N(d₁) - K·e^(-rτ)·N(d₂)
    let call_price = params.spot * nd1 - params.strike * discount_factor * nd2;

    // Delta: ∂C/∂S = N(d₁)
    let delta = nd1;

    // Gamma: ∂²C/∂S² = φ(d₁) / (S·σ·√τ)
    let gamma = phi_d1 / (params.spot * params.volatility * sqrt_tau);

    // Vega: ∂C/∂σ = S·φ(d₁)·√τ
    // Note: This is for a 1-unit change in σ. For 1% change, divide by 100.
    let vega = params.spot * phi_d1 * sqrt_tau;

    // Theta: ∂C/∂t = -S·φ(d₁)·σ/(2√τ) - r·K·e^(-rτ)·N(d₂)
    // Note: This is per year. For per day, divide by 365.
    let theta = -(params.spot * phi_d1 * params.volatility) / (2.0 * sqrt_tau)
        - params.rate * params.strike * discount_factor * nd2;

    // Rho: ∂C/∂r = K·τ·e^(-rτ)·N(d₂)
    // Note: This is for a 1-unit change in r. For 1% change, divide by 100.
    let rho = params.strike * params.time_to_maturity * discount_factor * nd2;

    Ok((
        call_price,
        Greeks {
            delta,
            gamma,
            vega,
            theta,
            rho,
        },
    ))
}

/// Calculate Black-Scholes put option price and Greeks using put-call parity
///
/// Put-call parity: P = C - S + K·e^(-rτ)
///
/// Put Greeks:
/// - Delta_put = Delta_call - 1
/// - Gamma_put = Gamma_call
/// - Vega_put = Vega_call
/// - Theta_put = Theta_call + r·K·e^(-rτ)
/// - Rho_put = Rho_call - K·τ·e^(-rτ)
pub fn calculate_put_greeks(params: &OptionParams) -> Result<(f64, Greeks), FinanceError> {
    let (call_price, call_greeks) = calculate_black_scholes(params)?;

    let discount_factor = (-params.rate * params.time_to_maturity).exp();
    let present_value_strike = params.strike * discount_factor;

    // Put price via put-call parity
    let put_price = call_price - params.spot + present_value_strike;

    // Put Greeks
    let put_greeks = Greeks {
        delta: call_greeks.delta - 1.0,
        gamma: call_greeks.gamma,  // Same as call
        vega: call_greeks.vega,    // Same as call
        theta: call_greeks.theta + params.rate * present_value_strike,
        rho: call_greeks.rho - params.strike * params.time_to_maturity * discount_factor,
    };

    Ok((put_price, put_greeks))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Test values from Hull (2018), Example 19.1
    #[test]
    fn test_black_scholes_hull_example() {
        let params = OptionParams {
            spot: 42.0,
            strike: 40.0,
            rate: 0.10,
            volatility: 0.20,
            time_to_maturity: 0.5,  // 6 months
        };

        let (call_price, greeks) = calculate_black_scholes(&params).unwrap();

        // Hull Example 19.1: Call price should be approximately 4.76
        assert_relative_eq!(call_price, 4.76, epsilon = 0.01);

        // Verify Delta is between 0 and 1 for call
        assert!(greeks.delta > 0.0 && greeks.delta < 1.0);

        // Gamma should be positive
        assert!(greeks.gamma > 0.0);

        // Vega should be positive
        assert!(greeks.vega > 0.0);

        // Theta should be negative (time decay)
        assert!(greeks.theta < 0.0);

        // Rho should be positive for call
        assert!(greeks.rho > 0.0);
    }

    /// Test at-the-money option (S = K)
    #[test]
    fn test_atm_option() {
        let params = OptionParams {
            spot: 100.0,
            strike: 100.0,
            rate: 0.05,
            volatility: 0.20,
            time_to_maturity: 1.0,
        };

        let (call_price, greeks) = calculate_black_scholes(&params).unwrap();

        // ATM call delta should be around 0.5-0.65 (above 0.5 due to positive drift from rate)
        assert!(greeks.delta > 0.5 && greeks.delta < 0.7);

        // Gamma is highest for ATM options
        assert!(greeks.gamma > 0.01);

        // Call price should be positive
        assert!(call_price > 0.0);
    }

    /// Test put-call parity
    #[test]
    fn test_put_call_parity() {
        let params = OptionParams {
            spot: 50.0,
            strike: 55.0,
            rate: 0.08,
            volatility: 0.25,
            time_to_maturity: 0.75,
        };

        let (call_price, _) = calculate_black_scholes(&params).unwrap();
        let (put_price, _) = calculate_put_greeks(&params).unwrap();

        let discount_factor = (-params.rate * params.time_to_maturity).exp();
        let pv_strike = params.strike * discount_factor;

        // Put-call parity: C - P = S - K·e^(-rτ)
        let lhs = call_price - put_price;
        let rhs = params.spot - pv_strike;

        assert_relative_eq!(lhs, rhs, epsilon = 1e-10);
    }

    /// Test put-call delta relationship
    #[test]
    fn test_greeks_sum_rule() {
        let params = OptionParams {
            spot: 100.0,
            strike: 100.0,
            rate: 0.05,
            volatility: 0.20,
            time_to_maturity: 1.0,
        };

        let (_, call_greeks) = calculate_black_scholes(&params).unwrap();
        let (_, put_greeks) = calculate_put_greeks(&params).unwrap();

        // Delta_put = Delta_call - 1, so Delta_call + Delta_put = 2*Delta_call - 1
        // This is NOT necessarily 1.0; the test was incorrect
        // Instead verify: Delta_put = Delta_call - 1
        assert_relative_eq!(put_greeks.delta, call_greeks.delta - 1.0, epsilon = 1e-10);

        // Gamma should be equal for calls and puts
        assert_relative_eq!(call_greeks.gamma, put_greeks.gamma, epsilon = 1e-10);

        // Vega should be equal for calls and puts
        assert_relative_eq!(call_greeks.vega, put_greeks.vega, epsilon = 1e-10);
    }

    #[test]
    fn test_invalid_params() {
        let invalid_spot = OptionParams {
            spot: -100.0,
            strike: 100.0,
            rate: 0.05,
            volatility: 0.20,
            time_to_maturity: 1.0,
        };
        assert!(calculate_black_scholes(&invalid_spot).is_err());

        let invalid_vol = OptionParams {
            spot: 100.0,
            strike: 100.0,
            rate: 0.05,
            volatility: 0.0,
            time_to_maturity: 1.0,
        };
        assert!(calculate_black_scholes(&invalid_vol).is_err());
    }
}
