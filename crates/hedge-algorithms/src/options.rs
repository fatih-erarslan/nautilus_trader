//! Options hedging algorithms and Greeks calculation

use crate::{HedgeError, HedgeConfig, MarketData};
use statrs::distribution::{Normal, ContinuousCDF};

/// Options hedging manager
#[derive(Debug, Clone)]
pub struct OptionsHedger {
    /// Risk-free rate
    pub risk_free_rate: f64,
    /// Dividend yield
    pub dividend_yield: f64,
    /// Configuration
    config: HedgeConfig,
}

impl OptionsHedger {
    /// Create new options hedger
    pub fn new(config: HedgeConfig) -> Self {
        Self {
            risk_free_rate: config.options_config.bs_params.risk_free_rate,
            dividend_yield: config.options_config.bs_params.dividend_yield,
            config,
        }
    }
    
    /// Calculate Black-Scholes option price
    pub fn black_scholes_price(
        &self,
        spot: f64,
        strike: f64,
        time_to_expiry: f64,
        volatility: f64,
        option_type: OptionType,
    ) -> Result<f64, HedgeError> {
        let d1 = self.calculate_d1(spot, strike, time_to_expiry, volatility)?;
        let d2 = d1 - volatility * time_to_expiry.sqrt();
        
        let normal = Normal::new(0.0, 1.0).map_err(|e| HedgeError::options(e.to_string()))?;
        
        match option_type {
            OptionType::Call => {
                let call_price = spot * (-self.dividend_yield * time_to_expiry).exp() * normal.cdf(d1)
                    - strike * (-self.risk_free_rate * time_to_expiry).exp() * normal.cdf(d2);
                Ok(call_price)
            }
            OptionType::Put => {
                let put_price = strike * (-self.risk_free_rate * time_to_expiry).exp() * normal.cdf(-d2)
                    - spot * (-self.dividend_yield * time_to_expiry).exp() * normal.cdf(-d1);
                Ok(put_price)
            }
        }
    }
    
    /// Calculate Delta
    pub fn calculate_delta(
        &self,
        spot: f64,
        strike: f64,
        time_to_expiry: f64,
        volatility: f64,
        option_type: OptionType,
    ) -> Result<f64, HedgeError> {
        let d1 = self.calculate_d1(spot, strike, time_to_expiry, volatility)?;
        let normal = Normal::new(0.0, 1.0).map_err(|e| HedgeError::options(e.to_string()))?;
        
        match option_type {
            OptionType::Call => {
                Ok((-self.dividend_yield * time_to_expiry).exp() * normal.cdf(d1))
            }
            OptionType::Put => {
                Ok((-self.dividend_yield * time_to_expiry).exp() * (normal.cdf(d1) - 1.0))
            }
        }
    }
    
    /// Calculate Gamma
    pub fn calculate_gamma(
        &self,
        spot: f64,
        strike: f64,
        time_to_expiry: f64,
        volatility: f64,
    ) -> Result<f64, HedgeError> {
        let d1 = self.calculate_d1(spot, strike, time_to_expiry, volatility)?;
        let normal = Normal::new(0.0, 1.0).map_err(|e| HedgeError::options(e.to_string()))?;
        
        let gamma = (-self.dividend_yield * time_to_expiry).exp()
            * normal.pdf(d1)
            / (spot * volatility * time_to_expiry.sqrt());
        
        Ok(gamma)
    }
    
    /// Calculate Theta
    pub fn calculate_theta(
        &self,
        spot: f64,
        strike: f64,
        time_to_expiry: f64,
        volatility: f64,
        option_type: OptionType,
    ) -> Result<f64, HedgeError> {
        let d1 = self.calculate_d1(spot, strike, time_to_expiry, volatility)?;
        let d2 = d1 - volatility * time_to_expiry.sqrt();
        let normal = Normal::new(0.0, 1.0).map_err(|e| HedgeError::options(e.to_string()))?;
        
        let term1 = -spot * normal.pdf(d1) * volatility * (-self.dividend_yield * time_to_expiry).exp()
            / (2.0 * time_to_expiry.sqrt());
        
        match option_type {
            OptionType::Call => {
                let term2 = self.dividend_yield * spot * normal.cdf(d1) * (-self.dividend_yield * time_to_expiry).exp();
                let term3 = -self.risk_free_rate * strike * normal.cdf(d2) * (-self.risk_free_rate * time_to_expiry).exp();
                Ok(term1 + term2 + term3)
            }
            OptionType::Put => {
                let term2 = -self.dividend_yield * spot * normal.cdf(-d1) * (-self.dividend_yield * time_to_expiry).exp();
                let term3 = self.risk_free_rate * strike * normal.cdf(-d2) * (-self.risk_free_rate * time_to_expiry).exp();
                Ok(term1 + term2 + term3)
            }
        }
    }
    
    /// Calculate Vega
    pub fn calculate_vega(
        &self,
        spot: f64,
        strike: f64,
        time_to_expiry: f64,
        volatility: f64,
    ) -> Result<f64, HedgeError> {
        let d1 = self.calculate_d1(spot, strike, time_to_expiry, volatility)?;
        let normal = Normal::new(0.0, 1.0).map_err(|e| HedgeError::options(e.to_string()))?;
        
        let vega = spot * time_to_expiry.sqrt() * normal.pdf(d1) * (-self.dividend_yield * time_to_expiry).exp();
        Ok(vega)
    }
    
    /// Calculate Rho
    pub fn calculate_rho(
        &self,
        spot: f64,
        strike: f64,
        time_to_expiry: f64,
        volatility: f64,
        option_type: OptionType,
    ) -> Result<f64, HedgeError> {
        let d1 = self.calculate_d1(spot, strike, time_to_expiry, volatility)?;
        let d2 = d1 - volatility * time_to_expiry.sqrt();
        let normal = Normal::new(0.0, 1.0).map_err(|e| HedgeError::options(e.to_string()))?;
        
        match option_type {
            OptionType::Call => {
                let rho = strike * time_to_expiry * normal.cdf(d2) * (-self.risk_free_rate * time_to_expiry).exp();
                Ok(rho)
            }
            OptionType::Put => {
                let rho = -strike * time_to_expiry * normal.cdf(-d2) * (-self.risk_free_rate * time_to_expiry).exp();
                Ok(rho)
            }
        }
    }
    
    /// Calculate all Greeks
    pub fn calculate_greeks(
        &self,
        spot: f64,
        strike: f64,
        time_to_expiry: f64,
        volatility: f64,
        option_type: OptionType,
    ) -> Result<Greeks, HedgeError> {
        Ok(Greeks {
            delta: self.calculate_delta(spot, strike, time_to_expiry, volatility, option_type)?,
            gamma: self.calculate_gamma(spot, strike, time_to_expiry, volatility)?,
            theta: self.calculate_theta(spot, strike, time_to_expiry, volatility, option_type)?,
            vega: self.calculate_vega(spot, strike, time_to_expiry, volatility)?,
            rho: self.calculate_rho(spot, strike, time_to_expiry, volatility, option_type)?,
        })
    }
    
    /// Calculate d1 parameter
    fn calculate_d1(
        &self,
        spot: f64,
        strike: f64,
        time_to_expiry: f64,
        volatility: f64,
    ) -> Result<f64, HedgeError> {
        if time_to_expiry <= 0.0 || volatility <= 0.0 || spot <= 0.0 || strike <= 0.0 {
            return Err(HedgeError::options("Invalid parameters for d1 calculation"));
        }
        
        let d1 = ((spot / strike).ln() + (self.risk_free_rate - self.dividend_yield + 0.5 * volatility.powi(2)) * time_to_expiry)
            / (volatility * time_to_expiry.sqrt());
        
        Ok(d1)
    }
    
    /// Calculate hedge ratio
    pub fn calculate_hedge_ratio(
        &self,
        greeks: &Greeks,
        position_size: f64,
    ) -> Result<f64, HedgeError> {
        // Delta hedge ratio
        let hedge_ratio = -greeks.delta * position_size;
        Ok(hedge_ratio)
    }
    
    /// Calculate implied volatility (simplified Newton-Raphson)
    pub fn calculate_implied_volatility(
        &self,
        market_price: f64,
        spot: f64,
        strike: f64,
        time_to_expiry: f64,
        option_type: OptionType,
    ) -> Result<f64, HedgeError> {
        let mut volatility = 0.2; // Initial guess
        let max_iterations = 100;
        let tolerance = 1e-6;
        
        for _ in 0..max_iterations {
            let price = self.black_scholes_price(spot, strike, time_to_expiry, volatility, option_type)?;
            let vega = self.calculate_vega(spot, strike, time_to_expiry, volatility)?;
            
            if vega.abs() < tolerance {
                break;
            }
            
            let price_diff = price - market_price;
            
            if price_diff.abs() < tolerance {
                break;
            }
            
            volatility -= price_diff / vega;
            volatility = volatility.max(0.001).min(5.0); // Clamp volatility
        }
        
        Ok(volatility)
    }
}

/// Option type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptionType {
    Call,
    Put,
}

/// Greeks structure
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Greeks {
    /// Delta
    pub delta: f64,
    /// Gamma
    pub gamma: f64,
    /// Theta
    pub theta: f64,
    /// Vega
    pub vega: f64,
    /// Rho
    pub rho: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_black_scholes_call() {
        let config = HedgeConfig::default();
        let hedger = OptionsHedger::new(config);
        
        let price = hedger.black_scholes_price(100.0, 100.0, 1.0, 0.2, OptionType::Call).unwrap();
        assert!(price > 0.0);
        assert!(price < 100.0);
    }
    
    #[test]
    fn test_greeks_calculation() {
        let config = HedgeConfig::default();
        let hedger = OptionsHedger::new(config);
        
        let greeks = hedger.calculate_greeks(100.0, 100.0, 1.0, 0.2, OptionType::Call).unwrap();
        
        assert!(greeks.delta > 0.0 && greeks.delta < 1.0);
        assert!(greeks.gamma > 0.0);
        assert!(greeks.theta < 0.0);
        assert!(greeks.vega > 0.0);
    }
}