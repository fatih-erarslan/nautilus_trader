//! Derivatives pricing and forecasting using consciousness-aware NHITS
//! 
//! This module implements sophisticated derivatives pricing models that leverage
//! NHITS predictions enhanced with consciousness mechanisms for options, futures,
//! and other derivative instruments pricing and risk management.

use super::*;
use ndarray::{Array1, Array2, s};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Derivatives pricing engine using consciousness-aware NHITS
#[derive(Debug)]
pub struct DerivativesPricingEngine {
    pub underlying_predictor: super::price_prediction::PricePredictor,
    pub volatility_predictor: super::volatility_modeling::VolatilityPredictor,
    pub risk_free_rate: f32,
    pub dividend_yield: f32,
    pub consciousness_adjustment_factor: f32,
    pub pricing_models: HashMap<DerivativeType, Box<dyn DerivativePricingModel>>,
    pub implied_volatility_surface: VolatilitySurface,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum DerivativeType {
    EuropeanCall,
    EuropeanPut,
    AmericanCall,
    AmericanPut,
    AsianOption,
    BarrierOption,
    BinaryOption,
    Future,
    Forward,
    Swap,
    Swaption,
    CreditDefault,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionContract {
    pub symbol: String,
    pub option_type: OptionType,
    pub strike_price: f32,
    pub expiration_timestamp: i64,
    pub underlying_price: f32,
    pub risk_free_rate: f32,
    pub dividend_yield: f32,
    pub days_to_expiration: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptionType {
    Call,
    Put,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivativePricing {
    pub contract: OptionContract,
    pub theoretical_price: f32,
    pub implied_volatility: f32,
    pub consciousness_adjusted_price: f32,
    pub greeks: Greeks,
    pub confidence_interval: (f32, f32),
    pub pricing_model: String,
    pub consciousness_factor: f32,
    pub market_regime_adjustment: f32,
    pub pricing_timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Greeks {
    pub delta: f32,      // Price sensitivity to underlying
    pub gamma: f32,      // Delta sensitivity to underlying
    pub theta: f32,      // Time decay
    pub vega: f32,       // Volatility sensitivity
    pub rho: f32,        // Interest rate sensitivity
    pub charm: f32,      // Delta decay (dDelta/dTime)
    pub vanna: f32,      // Vega sensitivity to underlying (dVega/dS)
    pub volga: f32,      // Vega sensitivity to volatility (dVega/dVol)
}

#[derive(Debug)]
pub struct VolatilitySurface {
    pub strikes: Vec<f32>,
    pub expirations: Vec<f32>,
    pub implied_volatilities: Array2<f32>,
    pub consciousness_adjustments: Array2<f32>,
}

/// Trait for derivative pricing models
pub trait DerivativePricingModel: std::fmt::Debug {
    fn price_derivative(&self, contract: &OptionContract, volatility: f32, consciousness: f32) -> f32;
    fn calculate_greeks(&self, contract: &OptionContract, volatility: f32, consciousness: f32) -> Greeks;
    fn calculate_implied_volatility(&self, contract: &OptionContract, market_price: f32, consciousness: f32) -> f32;
}

/// Black-Scholes pricing model with consciousness enhancement
#[derive(Debug)]
pub struct ConsciousnessBlackScholesModel {
    pub consciousness_volatility_adjustment: f32,
    pub consciousness_time_adjustment: f32,
}

impl DerivativePricingModel for ConsciousnessBlackScholesModel {
    fn price_derivative(&self, contract: &OptionContract, volatility: f32, consciousness: f32) -> f32 {
        // Enhanced Black-Scholes with consciousness adjustments
        let s = contract.underlying_price;
        let k = contract.strike_price;
        let t = contract.days_to_expiration / 365.0;
        let r = contract.risk_free_rate;
        let q = contract.dividend_yield;
        
        // Consciousness-adjusted parameters
        let adjusted_volatility = volatility * (1.0 + consciousness * self.consciousness_volatility_adjustment);
        let adjusted_time = t * (1.0 + consciousness * self.consciousness_time_adjustment);
        
        // Black-Scholes formula
        let d1 = (s.ln() - k.ln() + (r - q + 0.5 * adjusted_volatility.powi(2)) * adjusted_time) / 
                 (adjusted_volatility * adjusted_time.sqrt());
        let d2 = d1 - adjusted_volatility * adjusted_time.sqrt();
        
        match contract.option_type {
            OptionType::Call => {
                s * (-q * adjusted_time).exp() * self.normal_cdf(d1) - 
                k * (-r * adjusted_time).exp() * self.normal_cdf(d2)
            },
            OptionType::Put => {
                k * (-r * adjusted_time).exp() * self.normal_cdf(-d2) - 
                s * (-q * adjusted_time).exp() * self.normal_cdf(-d1)
            }
        }
    }
    
    fn calculate_greeks(&self, contract: &OptionContract, volatility: f32, consciousness: f32) -> Greeks {
        let s = contract.underlying_price;
        let k = contract.strike_price;
        let t = contract.days_to_expiration / 365.0;
        let r = contract.risk_free_rate;
        let q = contract.dividend_yield;
        
        let adjusted_volatility = volatility * (1.0 + consciousness * self.consciousness_volatility_adjustment);
        let adjusted_time = t * (1.0 + consciousness * self.consciousness_time_adjustment);
        
        let d1 = (s.ln() - k.ln() + (r - q + 0.5 * adjusted_volatility.powi(2)) * adjusted_time) / 
                 (adjusted_volatility * adjusted_time.sqrt());
        let d2 = d1 - adjusted_volatility * adjusted_time.sqrt();
        
        let nd1 = self.normal_cdf(d1);
        let nd2 = self.normal_cdf(d2);
        let npdf_d1 = self.normal_pdf(d1);
        
        let delta = match contract.option_type {
            OptionType::Call => (-q * adjusted_time).exp() * nd1,
            OptionType::Put => (-q * adjusted_time).exp() * (nd1 - 1.0),
        };
        
        let gamma = (-q * adjusted_time).exp() * npdf_d1 / (s * adjusted_volatility * adjusted_time.sqrt());
        
        let theta = match contract.option_type {
            OptionType::Call => {
                -s * npdf_d1 * adjusted_volatility * (-q * adjusted_time).exp() / (2.0 * adjusted_time.sqrt()) - 
                r * k * (-r * adjusted_time).exp() * nd2 + 
                q * s * (-q * adjusted_time).exp() * nd1
            },
            OptionType::Put => {
                -s * npdf_d1 * adjusted_volatility * (-q * adjusted_time).exp() / (2.0 * adjusted_time.sqrt()) + 
                r * k * (-r * adjusted_time).exp() * (1.0 - nd2) - 
                q * s * (-q * adjusted_time).exp() * (1.0 - nd1)
            }
        } / 365.0;  // Convert to daily theta
        
        let vega = s * (-q * adjusted_time).exp() * npdf_d1 * adjusted_time.sqrt() / 100.0;  // Per 1% vol change
        
        let rho = match contract.option_type {
            OptionType::Call => k * adjusted_time * (-r * adjusted_time).exp() * nd2 / 100.0,
            OptionType::Put => -k * adjusted_time * (-r * adjusted_time).exp() * (1.0 - nd2) / 100.0,
        };
        
        // Second-order Greeks (simplified)
        let charm = -(-q * adjusted_time).exp() * npdf_d1 * 
                   (2.0 * (r - q) * adjusted_time - d2 * adjusted_volatility * adjusted_time.sqrt()) / 
                   (2.0 * adjusted_time * adjusted_volatility * adjusted_time.sqrt()) / 365.0;
        
        let vanna = vega * d2 / adjusted_volatility;
        
        let volga = vega * d1 * d2 / adjusted_volatility;
        
        Greeks {
            delta,
            gamma,
            theta,
            vega,
            rho,
            charm,
            vanna,
            volga,
        }
    }
    
    fn calculate_implied_volatility(&self, contract: &OptionContract, market_price: f32, consciousness: f32) -> f32 {
        // Newton-Raphson method for implied volatility
        let mut vol_estimate = 0.3;  // Initial guess: 30%
        let tolerance = 1e-6;
        let max_iterations = 100;
        
        for _ in 0..max_iterations {
            let price = self.price_derivative(contract, vol_estimate, consciousness);
            let greeks = self.calculate_greeks(contract, vol_estimate, consciousness);
            
            let price_diff = price - market_price;
            if price_diff.abs() < tolerance {
                break;
            }
            
            // Newton-Raphson update
            if greeks.vega != 0.0 {
                vol_estimate -= price_diff / (greeks.vega * 100.0);  // Vega is per 1% change
            } else {
                break;
            }
            
            // Keep volatility positive and reasonable
            vol_estimate = vol_estimate.max(0.001).min(5.0);
        }
        
        vol_estimate
    }
}

impl ConsciousnessBlackScholesModel {
    pub fn new() -> Self {
        Self {
            consciousness_volatility_adjustment: 0.2,  // 20% volatility adjustment
            consciousness_time_adjustment: -0.1,       // 10% time decay adjustment
        }
    }
    
    fn normal_cdf(&self, x: f32) -> f32 {
        // Approximation of cumulative normal distribution
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;
        
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x_abs = x.abs();
        
        let t = 1.0 / (1.0 + p * x_abs);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x_abs * x_abs / 2.0).exp();
        
        0.5 * (1.0 + sign * y)
    }
    
    fn normal_pdf(&self, x: f32) -> f32 {
        // Standard normal probability density function
        (1.0 / (2.0 * std::f32::consts::PI).sqrt()) * (-0.5 * x * x).exp()
    }
}

/// Monte Carlo pricing model with consciousness enhancement
#[derive(Debug)]
pub struct ConsciousnessMonteCarloModel {
    pub num_simulations: usize,
    pub consciousness_drift_adjustment: f32,
    pub consciousness_volatility_adjustment: f32,
    pub random_seed: Option<u64>,
}

impl DerivativePricingModel for ConsciousnessMonteCarloModel {
    fn price_derivative(&self, contract: &OptionContract, volatility: f32, consciousness: f32) -> f32 {
        let s0 = contract.underlying_price;
        let k = contract.strike_price;
        let t = contract.days_to_expiration / 365.0;
        let r = contract.risk_free_rate;
        let q = contract.dividend_yield;
        
        // Consciousness adjustments
        let adjusted_drift = r - q + consciousness * self.consciousness_drift_adjustment;
        let adjusted_volatility = volatility * (1.0 + consciousness * self.consciousness_volatility_adjustment);
        
        let dt = t;  // Single time step for European options
        let mut payoffs = Vec::with_capacity(self.num_simulations);
        
        for _ in 0..self.num_simulations {
            // Generate random normal
            let z = self.sample_normal();
            
            // Simulate final stock price
            let st = s0 * ((adjusted_drift - 0.5 * adjusted_volatility.powi(2)) * dt + 
                          adjusted_volatility * dt.sqrt() * z).exp();
            
            // Calculate payoff
            let payoff = match contract.option_type {
                OptionType::Call => (st - k).max(0.0),
                OptionType::Put => (k - st).max(0.0),
            };
            
            payoffs.push(payoff);
        }
        
        // Discount expected payoff
        let expected_payoff = payoffs.iter().sum::<f32>() / self.num_simulations as f32;
        expected_payoff * (-r * t).exp()
    }
    
    fn calculate_greeks(&self, contract: &OptionContract, volatility: f32, consciousness: f32) -> Greeks {
        // Calculate Greeks using finite differences
        let h = 0.01;  // 1% bump
        
        let base_price = self.price_derivative(contract, volatility, consciousness);
        
        // Delta
        let mut up_contract = contract.clone();
        up_contract.underlying_price *= 1.0 + h;
        let up_price = self.price_derivative(&up_contract, volatility, consciousness);
        let delta = (up_price - base_price) / (contract.underlying_price * h);
        
        // Gamma
        let mut down_contract = contract.clone();
        down_contract.underlying_price *= 1.0 - h;
        let down_price = self.price_derivative(&down_contract, volatility, consciousness);
        let gamma = (up_price - 2.0 * base_price + down_price) / (contract.underlying_price * h).powi(2);
        
        // Theta
        let mut theta_contract = contract.clone();
        theta_contract.days_to_expiration -= 1.0;
        let theta_price = self.price_derivative(&theta_contract, volatility, consciousness);
        let theta = theta_price - base_price;  // Daily theta
        
        // Vega
        let vol_bump = 0.01;  // 1% volatility bump
        let vega_price = self.price_derivative(contract, volatility + vol_bump, consciousness);
        let vega = vega_price - base_price;
        
        // Rho
        let mut rho_contract = contract.clone();
        rho_contract.risk_free_rate += 0.01;  // 1% rate bump
        let rho_price = self.price_derivative(&rho_contract, volatility, consciousness);
        let rho = rho_price - base_price;
        
        Greeks {
            delta,
            gamma,
            theta,
            vega,
            rho,
            charm: 0.0,  // Simplified
            vanna: 0.0,  // Simplified
            volga: 0.0,  // Simplified
        }
    }
    
    fn calculate_implied_volatility(&self, contract: &OptionContract, market_price: f32, consciousness: f32) -> f32 {
        // Bisection method for implied volatility
        let mut vol_low = 0.001;
        let mut vol_high = 5.0;
        let tolerance = 1e-6;
        let max_iterations = 100;
        
        for _ in 0..max_iterations {
            let vol_mid = (vol_low + vol_high) / 2.0;
            let price_mid = self.price_derivative(contract, vol_mid, consciousness);
            
            if (price_mid - market_price).abs() < tolerance {
                return vol_mid;
            }
            
            if price_mid > market_price {
                vol_high = vol_mid;
            } else {
                vol_low = vol_mid;
            }
        }
        
        (vol_low + vol_high) / 2.0
    }
}

impl ConsciousnessMonteCarloModel {
    pub fn new(num_simulations: usize) -> Self {
        Self {
            num_simulations,
            consciousness_drift_adjustment: 0.1,
            consciousness_volatility_adjustment: 0.15,
            random_seed: None,
        }
    }
    
    fn sample_normal(&self) -> f32 {
        // Use proper statistical distribution from statrs crate
        use statrs::distribution::{Normal, ContinuousCDF};
        use rand::thread_rng;
        
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut rng = thread_rng();
        normal.sample(&mut rng) as f32
    }
}

impl DerivativesPricingEngine {
    pub fn new(risk_free_rate: f32, dividend_yield: f32) -> Self {
        let mut pricing_models: HashMap<DerivativeType, Box<dyn DerivativePricingModel>> = HashMap::new();
        
        // Add default pricing models
        pricing_models.insert(
            DerivativeType::EuropeanCall,
            Box::new(ConsciousnessBlackScholesModel::new())
        );
        pricing_models.insert(
            DerivativeType::EuropeanPut,
            Box::new(ConsciousnessBlackScholesModel::new())
        );
        
        Self {
            underlying_predictor: super::price_prediction::PricePredictor::new(60, 10),
            volatility_predictor: super::volatility_modeling::VolatilityPredictor::new(
                10, 60, 10, super::volatility_modeling::VolatilityType::GARCH
            ),
            risk_free_rate,
            dividend_yield,
            consciousness_adjustment_factor: 1.0,
            pricing_models,
            implied_volatility_surface: VolatilitySurface {
                strikes: Vec::new(),
                expirations: Vec::new(),
                implied_volatilities: Array2::zeros((0, 0)),
                consciousness_adjustments: Array2::zeros((0, 0)),
            },
        }
    }
    
    /// Price a derivative contract with consciousness enhancement
    pub fn price_derivative(
        &mut self,
        contract: &OptionContract,
        market_data: &Array2<f32>,
        derivative_type: DerivativeType,
    ) -> Result<DerivativePricing, String> {
        // Get volatility forecast
        let returns = utils::calculate_returns(&market_data.slice(s![.., 3]).to_vec());
        let vol_forecast = self.volatility_predictor.predict_volatility(&returns)?;
        let volatility = vol_forecast.volatility_predictions.get(0).copied().unwrap_or(0.2);
        
        // Calculate consciousness factor
        let consciousness = vol_forecast.consciousness_state * self.consciousness_adjustment_factor;
        
        // Get pricing model
        let pricing_model = self.pricing_models.get(&derivative_type)
            .ok_or_else(|| format!("No pricing model available for {:?}", derivative_type))?;
        
        // Calculate theoretical price
        let theoretical_price = pricing_model.price_derivative(contract, volatility, consciousness);
        
        // Calculate consciousness-adjusted price
        let market_regime_adjustment = self.calculate_market_regime_adjustment(market_data);
        let consciousness_adjusted_price = theoretical_price * (1.0 + consciousness * 0.1 + market_regime_adjustment);
        
        // Calculate Greeks
        let greeks = pricing_model.calculate_greeks(contract, volatility, consciousness);
        
        // Calculate confidence interval
        let confidence_interval = self.calculate_confidence_interval(
            theoretical_price,
            volatility,
            consciousness
        );
        
        Ok(DerivativePricing {
            contract: contract.clone(),
            theoretical_price,
            implied_volatility: volatility,
            consciousness_adjusted_price,
            greeks,
            confidence_interval,
            pricing_model: format!("{:?}", derivative_type),
            consciousness_factor: consciousness,
            market_regime_adjustment,
            pricing_timestamp: chrono::Utc::now().timestamp(),
        })
    }
    
    /// Calculate implied volatility with consciousness adjustment
    pub fn calculate_implied_volatility(
        &self,
        contract: &OptionContract,
        market_price: f32,
        derivative_type: DerivativeType,
        consciousness: f32,
    ) -> Result<f32, String> {
        let pricing_model = self.pricing_models.get(&derivative_type)
            .ok_or_else(|| format!("No pricing model available for {:?}", derivative_type))?;
        
        Ok(pricing_model.calculate_implied_volatility(contract, market_price, consciousness))
    }
    
    /// Build implied volatility surface
    pub fn build_volatility_surface(
        &mut self,
        option_chain: &[OptionContract],
        market_prices: &[f32],
        consciousness: f32,
    ) -> Result<(), String> {
        if option_chain.len() != market_prices.len() {
            return Err("Option chain and market prices must have same length".to_string());
        }
        
        // Extract unique strikes and expirations
        let mut strikes: Vec<f32> = option_chain.iter()
            .map(|opt| opt.strike_price)
            .collect();
        strikes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        strikes.dedup();
        
        let mut expirations: Vec<f32> = option_chain.iter()
            .map(|opt| opt.days_to_expiration)
            .collect();
        expirations.sort_by(|a, b| a.partial_cmp(b).unwrap());
        expirations.dedup();
        
        let n_strikes = strikes.len();
        let n_expirations = expirations.len();
        
        let mut implied_vols = Array2::zeros((n_strikes, n_expirations));
        let mut consciousness_adjustments = Array2::zeros((n_strikes, n_expirations));
        
        // Calculate implied volatilities for each strike/expiration combination
        for (i, option) in option_chain.iter().enumerate() {
            let strike_idx = strikes.iter().position(|&s| s == option.strike_price).unwrap();
            let exp_idx = expirations.iter().position(|&e| e == option.days_to_expiration).unwrap();
            
            let derivative_type = match option.option_type {
                OptionType::Call => DerivativeType::EuropeanCall,
                OptionType::Put => DerivativeType::EuropeanPut,
            };
            
            if let Ok(implied_vol) = self.calculate_implied_volatility(
                option, 
                market_prices[i], 
                derivative_type, 
                consciousness
            ) {
                implied_vols[[strike_idx, exp_idx]] = implied_vol;
                consciousness_adjustments[[strike_idx, exp_idx]] = consciousness;
            }
        }
        
        self.implied_volatility_surface = VolatilitySurface {
            strikes,
            expirations,
            implied_volatilities: implied_vols,
            consciousness_adjustments,
        };
        
        Ok(())
    }
    
    /// Screen options for trading opportunities
    pub fn screen_options(
        &mut self,
        option_chain: &[OptionContract],
        market_prices: &[f32],
        market_data: &Array2<f32>,
    ) -> Result<Vec<OptionOpportunity>, String> {
        let mut opportunities = Vec::new();
        
        for (i, option) in option_chain.iter().enumerate() {
            let market_price = market_prices[i];
            
            let derivative_type = match option.option_type {
                OptionType::Call => DerivativeType::EuropeanCall,
                OptionType::Put => DerivativeType::EuropeanPut,
            };
            
            if let Ok(pricing) = self.price_derivative(option, market_data, derivative_type) {
                let price_difference = pricing.consciousness_adjusted_price - market_price;
                let relative_difference = price_difference / market_price;
                
                // Look for mispriced options (>10% difference)
                if relative_difference.abs() > 0.1 {
                    let opportunity_type = if relative_difference > 0.0 {
                        OpportunityType::Undervalued
                    } else {
                        OpportunityType::Overvalued
                    };
                    
                    opportunities.push(OptionOpportunity {
                        contract: option.clone(),
                        market_price,
                        theoretical_price: pricing.consciousness_adjusted_price,
                        price_difference,
                        relative_difference,
                        opportunity_type,
                        confidence_score: pricing.consciousness_factor,
                        recommended_action: self.recommend_action(&opportunity_type, &pricing),
                    });
                }
            }
        }
        
        // Sort by absolute relative difference (best opportunities first)
        opportunities.sort_by(|a, b| {
            b.relative_difference.abs().partial_cmp(&a.relative_difference.abs()).unwrap()
        });
        
        Ok(opportunities)
    }
    
    /// Portfolio Greeks calculation
    pub fn calculate_portfolio_greeks(
        &self,
        portfolio: &[PortfolioPosition],
    ) -> PortfolioGreeks {
        let mut total_delta = 0.0;
        let mut total_gamma = 0.0;
        let mut total_theta = 0.0;
        let mut total_vega = 0.0;
        let mut total_rho = 0.0;
        
        for position in portfolio {
            total_delta += position.greeks.delta * position.quantity;
            total_gamma += position.greeks.gamma * position.quantity;
            total_theta += position.greeks.theta * position.quantity;
            total_vega += position.greeks.vega * position.quantity;
            total_rho += position.greeks.rho * position.quantity;
        }
        
        PortfolioGreeks {
            delta: total_delta,
            gamma: total_gamma,
            theta: total_theta,
            vega: total_vega,
            rho: total_rho,
        }
    }
    
    /// Risk management for derivatives portfolio
    pub fn assess_portfolio_risk(
        &self,
        portfolio: &[PortfolioPosition],
        market_scenarios: &[MarketScenario],
    ) -> PortfolioRiskAssessment {
        let portfolio_greeks = self.calculate_portfolio_greeks(portfolio);
        let mut scenario_pnls = Vec::new();
        
        for scenario in market_scenarios {
            let pnl = self.calculate_scenario_pnl(&portfolio_greeks, scenario);
            scenario_pnls.push(pnl);
        }
        
        // Calculate risk metrics
        let max_loss = scenario_pnls.iter().fold(0.0f32, |acc, &pnl| acc.min(pnl));
        let max_gain = scenario_pnls.iter().fold(0.0f32, |acc, &pnl| acc.max(pnl));
        let expected_pnl = scenario_pnls.iter().sum::<f32>() / scenario_pnls.len() as f32;
        
        // Calculate VaR (5% worst case)
        let mut sorted_pnls = scenario_pnls.clone();
        sorted_pnls.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let var_95 = sorted_pnls[(sorted_pnls.len() as f32 * 0.05) as usize];
        
        PortfolioRiskAssessment {
            portfolio_greeks,
            max_loss,
            max_gain,
            expected_pnl,
            var_95,
            scenario_pnls,
        }
    }
    
    // Private helper methods
    
    fn calculate_market_regime_adjustment(&self, market_data: &Array2<f32>) -> f32 {
        let returns = utils::calculate_returns(&market_data.slice(s![.., 3]).to_vec());
        
        if returns.len() < 20 {
            return 0.0;
        }
        
        // Calculate recent volatility vs historical
        let recent_vol = {
            let recent_returns = &returns[returns.len() - 10..];
            let mean = recent_returns.iter().sum::<f32>() / recent_returns.len() as f32;
            let variance = recent_returns.iter()
                .map(|&r| (r - mean).powi(2))
                .sum::<f32>() / (recent_returns.len() - 1) as f32;
            variance.sqrt()
        };
        
        let historical_vol = {
            let mean = returns.iter().sum::<f32>() / returns.len() as f32;
            let variance = returns.iter()
                .map(|&r| (r - mean).powi(2))
                .sum::<f32>() / (returns.len() - 1) as f32;
            variance.sqrt()
        };
        
        // Adjustment based on volatility regime
        if historical_vol > 0.0 {
            (recent_vol - historical_vol) / historical_vol * 0.1  // Up to 10% adjustment
        } else {
            0.0
        }
    }
    
    fn calculate_confidence_interval(&self, price: f32, volatility: f32, consciousness: f32) -> (f32, f32) {
        // Confidence interval based on volatility and consciousness
        let uncertainty = volatility * (1.0 - consciousness) * 0.2;  // Max 20% uncertainty
        let lower = price * (1.0 - uncertainty);
        let upper = price * (1.0 + uncertainty);
        
        (lower, upper)
    }
    
    fn recommend_action(&self, opportunity_type: &OpportunityType, pricing: &DerivativePricing) -> String {
        match opportunity_type {
            OpportunityType::Undervalued => {
                if pricing.greeks.delta.abs() > 0.5 {
                    "Strong Buy - High delta undervalued option".to_string()
                } else {
                    "Buy - Undervalued option".to_string()
                }
            },
            OpportunityType::Overvalued => {
                if pricing.greeks.theta < -0.1 {
                    "Sell - Overvalued with high time decay".to_string()
                } else {
                    "Sell - Overvalued option".to_string()
                }
            }
        }
    }
    
    fn calculate_scenario_pnl(&self, greeks: &PortfolioGreeks, scenario: &MarketScenario) -> f32 {
        // Taylor series approximation for P&L
        let delta_pnl = greeks.delta * scenario.underlying_change;
        let gamma_pnl = 0.5 * greeks.gamma * scenario.underlying_change.powi(2);
        let theta_pnl = greeks.theta * scenario.time_change;
        let vega_pnl = greeks.vega * scenario.volatility_change;
        let rho_pnl = greeks.rho * scenario.rate_change;
        
        delta_pnl + gamma_pnl + theta_pnl + vega_pnl + rho_pnl
    }
}

// Supporting data structures

#[derive(Debug, Clone)]
pub struct OptionOpportunity {
    pub contract: OptionContract,
    pub market_price: f32,
    pub theoretical_price: f32,
    pub price_difference: f32,
    pub relative_difference: f32,
    pub opportunity_type: OpportunityType,
    pub confidence_score: f32,
    pub recommended_action: String,
}

#[derive(Debug, Clone)]
pub enum OpportunityType {
    Undervalued,
    Overvalued,
}

#[derive(Debug, Clone)]
pub struct PortfolioPosition {
    pub contract: OptionContract,
    pub quantity: f32,
    pub greeks: Greeks,
}

#[derive(Debug, Clone)]
pub struct PortfolioGreeks {
    pub delta: f32,
    pub gamma: f32,
    pub theta: f32,
    pub vega: f32,
    pub rho: f32,
}

#[derive(Debug, Clone)]
pub struct MarketScenario {
    pub name: String,
    pub underlying_change: f32,
    pub volatility_change: f32,
    pub time_change: f32,
    pub rate_change: f32,
    pub probability: f32,
}

#[derive(Debug, Clone)]
pub struct PortfolioRiskAssessment {
    pub portfolio_greeks: PortfolioGreeks,
    pub max_loss: f32,
    pub max_gain: f32,
    pub expected_pnl: f32,
    pub var_95: f32,
    pub scenario_pnls: Vec<f32>,
}

/// Advanced derivatives strategies
pub mod strategies {
    use super::*;
    
    /// Straddle strategy analyzer
    pub struct StraddleAnalyzer {
        pricing_engine: DerivativesPricingEngine,
    }
    
    impl StraddleAnalyzer {
        pub fn new(risk_free_rate: f32) -> Self {
            Self {
                pricing_engine: DerivativesPricingEngine::new(risk_free_rate, 0.0),
            }
        }
        
        pub fn analyze_straddle(
            &mut self,
            underlying_price: f32,
            strike: f32,
            expiration_days: f32,
            market_data: &Array2<f32>,
        ) -> Result<StraddleAnalysis, String> {
            // Create call and put contracts
            let call_contract = OptionContract {
                symbol: "STRADDLE".to_string(),
                option_type: OptionType::Call,
                strike_price: strike,
                expiration_timestamp: chrono::Utc::now().timestamp() + (expiration_days * 86400.0) as i64,
                underlying_price,
                risk_free_rate: self.pricing_engine.risk_free_rate,
                dividend_yield: self.pricing_engine.dividend_yield,
                days_to_expiration: expiration_days,
            };
            
            let put_contract = OptionContract {
                option_type: OptionType::Put,
                ..call_contract.clone()
            };
            
            // Price both options
            let call_pricing = self.pricing_engine.price_derivative(
                &call_contract, 
                market_data, 
                DerivativeType::EuropeanCall
            )?;
            let put_pricing = self.pricing_engine.price_derivative(
                &put_contract, 
                market_data, 
                DerivativeType::EuropeanPut
            )?;
            
            let total_premium = call_pricing.consciousness_adjusted_price + put_pricing.consciousness_adjusted_price;
            
            // Calculate breakeven points
            let upper_breakeven = strike + total_premium;
            let lower_breakeven = strike - total_premium;
            
            // Calculate Greeks
            let combined_greeks = Greeks {
                delta: call_pricing.greeks.delta + put_pricing.greeks.delta,  // Should be ~0 for ATM straddle
                gamma: call_pricing.greeks.gamma + put_pricing.greeks.gamma,
                theta: call_pricing.greeks.theta + put_pricing.greeks.theta,
                vega: call_pricing.greeks.vega + put_pricing.greeks.vega,
                rho: call_pricing.greeks.rho + put_pricing.greeks.rho,
                charm: call_pricing.greeks.charm + put_pricing.greeks.charm,
                vanna: call_pricing.greeks.vanna + put_pricing.greeks.vanna,
                volga: call_pricing.greeks.volga + put_pricing.greeks.volga,
            };
            
            Ok(StraddleAnalysis {
                call_price: call_pricing.consciousness_adjusted_price,
                put_price: put_pricing.consciousness_adjusted_price,
                total_premium,
                upper_breakeven,
                lower_breakeven,
                combined_greeks,
                implied_volatility: (call_pricing.implied_volatility + put_pricing.implied_volatility) / 2.0,
                consciousness_factor: (call_pricing.consciousness_factor + put_pricing.consciousness_factor) / 2.0,
            })
        }
    }
    
    #[derive(Debug, Clone)]
    pub struct StraddleAnalysis {
        pub call_price: f32,
        pub put_price: f32,
        pub total_premium: f32,
        pub upper_breakeven: f32,
        pub lower_breakeven: f32,
        pub combined_greeks: Greeks,
        pub implied_volatility: f32,
        pub consciousness_factor: f32,
    }
    
    /// Iron Condor strategy analyzer
    pub struct IronCondorAnalyzer {
        pricing_engine: DerivativesPricingEngine,
    }
    
    impl IronCondorAnalyzer {
        pub fn new(risk_free_rate: f32) -> Self {
            Self {
                pricing_engine: DerivativesPricingEngine::new(risk_free_rate, 0.0),
            }
        }
        
        pub fn analyze_iron_condor(
            &mut self,
            underlying_price: f32,
            strikes: (f32, f32, f32, f32),  // (put_short, put_long, call_short, call_long)
            expiration_days: f32,
            market_data: &Array2<f32>,
        ) -> Result<IronCondorAnalysis, String> {
            let (put_short_strike, put_long_strike, call_short_strike, call_long_strike) = strikes;
            
            // Create contracts for each leg
            let contracts = [
                (OptionContract {
                    symbol: "IC_PUT_SHORT".to_string(),
                    option_type: OptionType::Put,
                    strike_price: put_short_strike,
                    expiration_timestamp: chrono::Utc::now().timestamp() + (expiration_days * 86400.0) as i64,
                    underlying_price,
                    risk_free_rate: self.pricing_engine.risk_free_rate,
                    dividend_yield: self.pricing_engine.dividend_yield,
                    days_to_expiration: expiration_days,
                }, -1.0),  // Short position
                
                (OptionContract {
                    symbol: "IC_PUT_LONG".to_string(),
                    option_type: OptionType::Put,
                    strike_price: put_long_strike,
                    expiration_timestamp: chrono::Utc::now().timestamp() + (expiration_days * 86400.0) as i64,
                    underlying_price,
                    risk_free_rate: self.pricing_engine.risk_free_rate,
                    dividend_yield: self.pricing_engine.dividend_yield,
                    days_to_expiration: expiration_days,
                }, 1.0),  // Long position
                
                (OptionContract {
                    symbol: "IC_CALL_SHORT".to_string(),
                    option_type: OptionType::Call,
                    strike_price: call_short_strike,
                    expiration_timestamp: chrono::Utc::now().timestamp() + (expiration_days * 86400.0) as i64,
                    underlying_price,
                    risk_free_rate: self.pricing_engine.risk_free_rate,
                    dividend_yield: self.pricing_engine.dividend_yield,
                    days_to_expiration: expiration_days,
                }, -1.0),  // Short position
                
                (OptionContract {
                    symbol: "IC_CALL_LONG".to_string(),
                    option_type: OptionType::Call,
                    strike_price: call_long_strike,
                    expiration_timestamp: chrono::Utc::now().timestamp() + (expiration_days * 86400.0) as i64,
                    underlying_price,
                    risk_free_rate: self.pricing_engine.risk_free_rate,
                    dividend_yield: self.pricing_engine.dividend_yield,
                    days_to_expiration: expiration_days,
                }, 1.0),  // Long position
            ];
            
            let mut net_premium = 0.0;
            let mut combined_greeks = Greeks {
                delta: 0.0, gamma: 0.0, theta: 0.0, vega: 0.0, rho: 0.0,
                charm: 0.0, vanna: 0.0, volga: 0.0,
            };
            
            // Price each leg and combine
            for (contract, position_size) in &contracts {
                let derivative_type = match contract.option_type {
                    OptionType::Call => DerivativeType::EuropeanCall,
                    OptionType::Put => DerivativeType::EuropeanPut,
                };
                
                let pricing = self.pricing_engine.price_derivative(contract, market_data, derivative_type)?;
                
                net_premium += pricing.consciousness_adjusted_price * position_size;
                combined_greeks.delta += pricing.greeks.delta * position_size;
                combined_greeks.gamma += pricing.greeks.gamma * position_size;
                combined_greeks.theta += pricing.greeks.theta * position_size;
                combined_greeks.vega += pricing.greeks.vega * position_size;
                combined_greeks.rho += pricing.greeks.rho * position_size;
            }
            
            // Calculate max profit/loss
            let put_width = put_short_strike - put_long_strike;
            let call_width = call_long_strike - call_short_strike;
            let max_loss = put_width.max(call_width) - net_premium;
            let max_profit = net_premium;
            
            Ok(IronCondorAnalysis {
                net_premium,
                max_profit,
                max_loss,
                profit_zone: (put_short_strike, call_short_strike),
                combined_greeks,
                pop: self.calculate_probability_of_profit(&strikes, underlying_price, expiration_days),
            })
        }
        
        fn calculate_probability_of_profit(&self, strikes: &(f32, f32, f32, f32), underlying: f32, days: f32) -> f32 {
            // Simplified POP calculation - would use more sophisticated model in practice
            let range = strikes.2 - strikes.1;  // Distance between short strikes
            let time_factor = (days / 30.0).sqrt();  // Time adjustment
            
            // Assume normal distribution and calculate probability of staying within range
            0.7 - (range / underlying * 10.0) * time_factor  // Simplified formula
        }
    }
    
    #[derive(Debug, Clone)]
    pub struct IronCondorAnalysis {
        pub net_premium: f32,
        pub max_profit: f32,
        pub max_loss: f32,
        pub profit_zone: (f32, f32),
        pub combined_greeks: Greeks,
        pub pop: f32,  // Probability of Profit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_black_scholes_pricing() {
        let model = ConsciousnessBlackScholesModel::new();
        let contract = OptionContract {
            symbol: "TEST".to_string(),
            option_type: OptionType::Call,
            strike_price: 100.0,
            expiration_timestamp: chrono::Utc::now().timestamp() + 30 * 86400,
            underlying_price: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.0,
            days_to_expiration: 30.0,
        };
        
        let price = model.price_derivative(&contract, 0.2, 0.7);
        assert!(price > 0.0);
        assert!(price < contract.underlying_price);  // Call option should be less than underlying for ATM
    }
    
    #[test]
    fn test_greeks_calculation() {
        let model = ConsciousnessBlackScholesModel::new();
        let contract = OptionContract {
            symbol: "TEST".to_string(),
            option_type: OptionType::Call,
            strike_price: 100.0,
            expiration_timestamp: chrono::Utc::now().timestamp() + 30 * 86400,
            underlying_price: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.0,
            days_to_expiration: 30.0,
        };
        
        let greeks = model.calculate_greeks(&contract, 0.2, 0.7);
        
        // ATM call delta should be around 0.5
        assert!(greeks.delta > 0.3 && greeks.delta < 0.7);
        
        // Gamma should be positive
        assert!(greeks.gamma > 0.0);
        
        // Theta should be negative (time decay)
        assert!(greeks.theta < 0.0);
        
        // Vega should be positive (call increases with volatility)
        assert!(greeks.vega > 0.0);
    }
    
    #[test]
    fn test_monte_carlo_pricing() {
        let model = ConsciousnessMonteCarloModel::new(10000);
        let contract = OptionContract {
            symbol: "TEST".to_string(),
            option_type: OptionType::Call,
            strike_price: 100.0,
            expiration_timestamp: chrono::Utc::now().timestamp() + 30 * 86400,
            underlying_price: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.0,
            days_to_expiration: 30.0,
        };
        
        let price = model.price_derivative(&contract, 0.2, 0.7);
        assert!(price > 0.0);
        assert!(price < contract.underlying_price);
    }
    
    #[test]
    fn test_derivatives_pricing_engine() {
        let mut engine = DerivativesPricingEngine::new(0.05, 0.0);
        let contract = OptionContract {
            symbol: "TEST".to_string(),
            option_type: OptionType::Call,
            strike_price: 100.0,
            expiration_timestamp: chrono::Utc::now().timestamp() + 30 * 86400,
            underlying_price: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.0,
            days_to_expiration: 30.0,
        };
        
        let market_data = Array2::zeros((100, 10));
        let result = engine.price_derivative(&contract, &market_data, DerivativeType::EuropeanCall);
        
        assert!(result.is_ok());
        let pricing = result.unwrap();
        assert!(pricing.theoretical_price > 0.0);
        assert!(pricing.consciousness_adjusted_price > 0.0);
    }
    
    #[test]
    fn test_normal_cdf() {
        let model = ConsciousnessBlackScholesModel::new();
        
        // Test standard normal CDF values
        assert!((model.normal_cdf(0.0) - 0.5).abs() < 0.01);
        assert!(model.normal_cdf(-3.0) < 0.01);
        assert!(model.normal_cdf(3.0) > 0.99);
    }
}