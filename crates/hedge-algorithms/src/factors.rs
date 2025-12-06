//! Eight-factor model implementation for hedge algorithms

use nalgebra::{DVector, DMatrix};
use std::collections::{HashMap, VecDeque};
use crate::{HedgeError, HedgeConfig, MarketData, utils::math};

/// Standard eight-factor model
pub struct StandardFactorModel {
    /// Factor names
    pub factor_names: Vec<String>,
    /// Factor loadings
    pub factor_loadings: DVector<f64>,
    /// Factor returns
    pub factor_returns: DMatrix<f64>,
    /// Factor covariance matrix
    pub factor_covariance: DMatrix<f64>,
    /// Specific risk
    pub specific_risk: f64,
    /// Market data history
    market_history: VecDeque<MarketData>,
    /// Factor calculators
    calculators: HashMap<String, Box<dyn FactorCalculator + Send + Sync>>,
    /// Configuration
    config: HedgeConfig,
    /// Time step
    time_step: usize,
}

impl StandardFactorModel {
    /// Create new standard factor model
    pub fn new(config: HedgeConfig) -> Result<Self, HedgeError> {
        let factor_names = config.factor_config.factor_names.clone();
        let num_factors = factor_names.len();
        
        let mut calculators: HashMap<String, Box<dyn FactorCalculator + Send + Sync>> = HashMap::new();
        
        // Initialize factor calculators
        calculators.insert("Volatility".to_string(), Box::new(VolatilityFactor::new()));
        calculators.insert("Momentum".to_string(), Box::new(MomentumFactor::new()));
        calculators.insert("MeanReversion".to_string(), Box::new(MeanReversionFactor::new()));
        calculators.insert("Liquidity".to_string(), Box::new(LiquidityFactor::new()));
        calculators.insert("Quality".to_string(), Box::new(QualityFactor::new()));
        calculators.insert("Growth".to_string(), Box::new(GrowthFactor::new()));
        calculators.insert("Size".to_string(), Box::new(SizeFactor::new()));
        calculators.insert("Profitability".to_string(), Box::new(ProfitabilityFactor::new()));
        
        Ok(Self {
            factor_names,
            factor_loadings: DVector::zeros(num_factors),
            factor_returns: DMatrix::zeros(num_factors, 1),
            factor_covariance: DMatrix::identity(num_factors, num_factors),
            specific_risk: 0.0,
            market_history: VecDeque::new(),
            calculators,
            config,
            time_step: 0,
        })
    }
    
    /// Update factor model with new market data
    pub fn update(&mut self, market_data: &MarketData) -> Result<(), HedgeError> {
        // Add to history
        self.market_history.push_back(market_data.clone());
        
        // Keep only recent history
        if self.market_history.len() > self.config.max_history {
            self.market_history.pop_front();
        }
        
        // Update factor loadings
        self.update_factor_loadings()?;
        
        // Update factor returns
        self.update_factor_returns()?;
        
        // Update factor covariance
        self.update_factor_covariance()?;
        
        // Update specific risk
        self.update_specific_risk()?;
        
        self.time_step += 1;
        
        Ok(())
    }
    
    /// Update factor loadings
    fn update_factor_loadings(&mut self) -> Result<(), HedgeError> {
        if self.market_history.is_empty() {
            return Ok(());
        }
        
        let mut loadings = Vec::new();
        
        for (i, factor_name) in self.factor_names.iter().enumerate() {
            if let Some(calculator) = self.calculators.get(factor_name) {
                let loading = calculator.calculate_loading(&self.market_history)?;
                loadings.push(loading * self.config.factor_config.factor_weights[i]);
            } else {
                loadings.push(0.0);
            }
        }
        
        self.factor_loadings = DVector::from_vec(loadings);
        
        Ok(())
    }
    
    /// Update factor returns
    fn update_factor_returns(&mut self) -> Result<(), HedgeError> {
        if self.market_history.len() < 2 {
            return Ok(());
        }
        
        let mut returns = Vec::new();
        
        for factor_name in &self.factor_names {
            if let Some(calculator) = self.calculators.get(factor_name) {
                let return_value = calculator.calculate_return(&self.market_history)?;
                returns.push(return_value);
            } else {
                returns.push(0.0);
            }
        }
        
        // Update factor returns matrix
        let new_returns = DVector::from_vec(returns);
        let current_cols = self.factor_returns.ncols();
        
        if current_cols == 0 {
            self.factor_returns = new_returns.clone().into();
        } else {
            // Append new column
            let mut new_matrix = DMatrix::zeros(self.factor_names.len(), current_cols + 1);
            new_matrix.columns_mut(0, current_cols).copy_from(&self.factor_returns);
            new_matrix.column_mut(current_cols).copy_from(&new_returns);
            self.factor_returns = new_matrix;
        }
        
        // Keep only recent returns
        if self.factor_returns.ncols() > self.config.max_history {
            let start_col = self.factor_returns.ncols() - self.config.max_history;
            self.factor_returns = self.factor_returns.columns(start_col, self.config.max_history).into();
        }
        
        Ok(())
    }
    
    /// Update factor covariance matrix
    fn update_factor_covariance(&mut self) -> Result<(), HedgeError> {
        if self.factor_returns.ncols() < 2 {
            return Ok(());
        }
        
        let num_factors = self.factor_names.len();
        let mut covariance = DMatrix::zeros(num_factors, num_factors);
        
        for i in 0..num_factors {
            for j in 0..num_factors {
                let factor_i_returns: Vec<f64> = self.factor_returns.row(i).iter().copied().collect();
                let factor_j_returns: Vec<f64> = self.factor_returns.row(j).iter().copied().collect();
                
                let cov = math::covariance(&factor_i_returns, &factor_j_returns)?;
                covariance[(i, j)] = cov;
            }
        }
        
        self.factor_covariance = covariance;
        
        Ok(())
    }
    
    /// Update specific risk
    fn update_specific_risk(&mut self) -> Result<(), HedgeError> {
        if self.factor_returns.ncols() < 2 {
            return Ok(());
        }
        
        // Calculate specific risk as residual variance
        let mut total_variance = 0.0;
        let mut factor_variance = 0.0;
        
        for i in 0..self.factor_names.len() {
            let factor_returns: Vec<f64> = self.factor_returns.row(i).iter().copied().collect();
            let variance = math::variance(&factor_returns)?;
            
            total_variance += variance;
            factor_variance += variance * self.factor_loadings[i].powi(2);
        }
        
        self.specific_risk = (total_variance - factor_variance).max(0.0);
        
        Ok(())
    }
    
    /// Get factor exposures
    pub fn get_exposures(&self) -> Result<DVector<f64>, HedgeError> {
        Ok(self.factor_loadings.clone())
    }
    
    /// Get factor returns
    pub fn get_factor_returns(&self) -> &DMatrix<f64> {
        &self.factor_returns
    }
    
    /// Get factor covariance matrix
    pub fn get_factor_covariance(&self) -> &DMatrix<f64> {
        &self.factor_covariance
    }
    
    /// Get specific risk
    pub fn get_specific_risk(&self) -> f64 {
        self.specific_risk
    }
    
    /// Predict expected return
    pub fn predict_return(&self, factor_expected_returns: &DVector<f64>) -> Result<f64, HedgeError> {
        if factor_expected_returns.len() != self.factor_loadings.len() {
            return Err(HedgeError::factor("Factor dimension mismatch"));
        }
        
        let expected_return = self.factor_loadings.dot(factor_expected_returns);
        Ok(expected_return)
    }
    
    /// Predict risk
    pub fn predict_risk(&self, factor_expected_returns: &DVector<f64>) -> Result<f64, HedgeError> {
        if factor_expected_returns.len() != self.factor_loadings.len() {
            return Err(HedgeError::factor("Factor dimension mismatch"));
        }
        
        let factor_risk = self.factor_loadings.transpose() * &self.factor_covariance * &self.factor_loadings;
        let total_risk = factor_risk + self.specific_risk;
        
        Ok(total_risk.sqrt())
    }
    
    /// Get factor attribution
    pub fn get_factor_attribution(&self, portfolio_return: f64) -> Result<HashMap<String, f64>, HedgeError> {
        let mut attribution = HashMap::new();
        
        for (i, factor_name) in self.factor_names.iter().enumerate() {
            let factor_contribution = self.factor_loadings[i] * portfolio_return;
            attribution.insert(factor_name.clone(), factor_contribution);
        }
        
        Ok(attribution)
    }
    
    /// Reset factor model
    pub fn reset(&mut self) -> Result<(), HedgeError> {
        self.factor_loadings = DVector::zeros(self.factor_names.len());
        self.factor_returns = DMatrix::zeros(self.factor_names.len(), 1);
        self.factor_covariance = DMatrix::identity(self.factor_names.len(), self.factor_names.len());
        self.specific_risk = 0.0;
        self.market_history.clear();
        self.time_step = 0;
        
        Ok(())
    }
}

/// Factor calculator trait
pub trait FactorCalculator {
    /// Calculate factor loading
    fn calculate_loading(&self, market_history: &VecDeque<MarketData>) -> Result<f64, HedgeError>;
    
    /// Calculate factor return
    fn calculate_return(&self, market_history: &VecDeque<MarketData>) -> Result<f64, HedgeError>;
    
    /// Get factor name
    fn get_name(&self) -> &str;
}

/// Volatility factor calculator
#[derive(Debug, Clone)]
pub struct VolatilityFactor {
    name: String,
}

impl VolatilityFactor {
    pub fn new() -> Self {
        Self {
            name: "Volatility".to_string(),
        }
    }
}

impl FactorCalculator for VolatilityFactor {
    fn calculate_loading(&self, market_history: &VecDeque<MarketData>) -> Result<f64, HedgeError> {
        if market_history.len() < 2 {
            return Ok(0.0);
        }
        
        let prices: Vec<f64> = market_history.iter().map(|d| d.close).collect();
        let returns = math::returns(&prices)?;
        let volatility = math::standard_deviation(&returns)?;
        
        Ok(volatility)
    }
    
    fn calculate_return(&self, market_history: &VecDeque<MarketData>) -> Result<f64, HedgeError> {
        if market_history.len() < 2 {
            return Ok(0.0);
        }
        
        let current_data = market_history.back().unwrap();
        let previous_data = &market_history[market_history.len() - 2];
        
        let current_volatility = (current_data.high - current_data.low) / current_data.close;
        let previous_volatility = (previous_data.high - previous_data.low) / previous_data.close;
        
        Ok(current_volatility - previous_volatility)
    }
    
    fn get_name(&self) -> &str {
        &self.name
    }
}

/// Momentum factor calculator
#[derive(Debug, Clone)]
pub struct MomentumFactor {
    name: String,
}

impl MomentumFactor {
    pub fn new() -> Self {
        Self {
            name: "Momentum".to_string(),
        }
    }
}

impl FactorCalculator for MomentumFactor {
    fn calculate_loading(&self, market_history: &VecDeque<MarketData>) -> Result<f64, HedgeError> {
        if market_history.len() < 20 {
            return Ok(0.0);
        }
        
        let prices: Vec<f64> = market_history.iter().map(|d| d.close).collect();
        let current_price = prices.last().unwrap();
        let lookback_price = prices[prices.len() - 20];
        
        let momentum = (current_price - lookback_price) / lookback_price;
        Ok(momentum)
    }
    
    fn calculate_return(&self, market_history: &VecDeque<MarketData>) -> Result<f64, HedgeError> {
        if market_history.len() < 2 {
            return Ok(0.0);
        }
        
        let current_data = market_history.back().unwrap();
        let previous_data = &market_history[market_history.len() - 2];
        
        let return_value = (current_data.close - previous_data.close) / previous_data.close;
        Ok(return_value)
    }
    
    fn get_name(&self) -> &str {
        &self.name
    }
}

/// Mean reversion factor calculator
#[derive(Debug, Clone)]
pub struct MeanReversionFactor {
    name: String,
}

impl MeanReversionFactor {
    pub fn new() -> Self {
        Self {
            name: "MeanReversion".to_string(),
        }
    }
}

impl FactorCalculator for MeanReversionFactor {
    fn calculate_loading(&self, market_history: &VecDeque<MarketData>) -> Result<f64, HedgeError> {
        if market_history.len() < 20 {
            return Ok(0.0);
        }
        
        let prices: Vec<f64> = market_history.iter().map(|d| d.close).collect();
        let recent_prices = &prices[prices.len() - 20..];
        
        let mean_price = recent_prices.iter().sum::<f64>() / recent_prices.len() as f64;
        let current_price = *prices.last().unwrap();
        
        let mean_reversion = (mean_price - current_price) / current_price;
        Ok(mean_reversion)
    }
    
    fn calculate_return(&self, market_history: &VecDeque<MarketData>) -> Result<f64, HedgeError> {
        if market_history.len() < 2 {
            return Ok(0.0);
        }
        
        let current_data = market_history.back().unwrap();
        let previous_data = &market_history[market_history.len() - 2];
        
        let return_value = -(current_data.close - previous_data.close) / previous_data.close;
        Ok(return_value)
    }
    
    fn get_name(&self) -> &str {
        &self.name
    }
}

/// Liquidity factor calculator
#[derive(Debug, Clone)]
pub struct LiquidityFactor {
    name: String,
}

impl LiquidityFactor {
    pub fn new() -> Self {
        Self {
            name: "Liquidity".to_string(),
        }
    }
}

impl FactorCalculator for LiquidityFactor {
    fn calculate_loading(&self, market_history: &VecDeque<MarketData>) -> Result<f64, HedgeError> {
        if market_history.len() < 20 {
            return Ok(0.0);
        }
        
        let volumes: Vec<f64> = market_history.iter().map(|d| d.volume).collect();
        let recent_volumes = &volumes[volumes.len() - 20..];
        
        let avg_volume = recent_volumes.iter().sum::<f64>() / recent_volumes.len() as f64;
        let current_volume = *volumes.last().unwrap();
        
        let liquidity = current_volume / avg_volume;
        Ok(liquidity)
    }
    
    fn calculate_return(&self, market_history: &VecDeque<MarketData>) -> Result<f64, HedgeError> {
        if market_history.len() < 2 {
            return Ok(0.0);
        }
        
        let current_data = market_history.back().unwrap();
        let previous_data = &market_history[market_history.len() - 2];
        
        let volume_change = (current_data.volume - previous_data.volume) / previous_data.volume;
        Ok(volume_change)
    }
    
    fn get_name(&self) -> &str {
        &self.name
    }
}

/// Quality factor calculator
#[derive(Debug, Clone)]
pub struct QualityFactor {
    name: String,
}

impl QualityFactor {
    pub fn new() -> Self {
        Self {
            name: "Quality".to_string(),
        }
    }
}

impl FactorCalculator for QualityFactor {
    fn calculate_loading(&self, market_history: &VecDeque<MarketData>) -> Result<f64, HedgeError> {
        if market_history.len() < 20 {
            return Ok(0.0);
        }
        
        let prices: Vec<f64> = market_history.iter().map(|d| d.close).collect();
        let returns = math::returns(&prices)?;
        
        // Quality measured as consistency of returns
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        
        let quality = if variance > 0.0 {
            mean_return / variance.sqrt()
        } else {
            0.0
        };
        
        Ok(quality)
    }
    
    fn calculate_return(&self, market_history: &VecDeque<MarketData>) -> Result<f64, HedgeError> {
        if market_history.len() < 2 {
            return Ok(0.0);
        }
        
        let current_data = market_history.back().unwrap();
        let previous_data = &market_history[market_history.len() - 2];
        
        // Quality return based on price stability
        let price_stability = 1.0 - (current_data.high - current_data.low) / current_data.close;
        let previous_stability = 1.0 - (previous_data.high - previous_data.low) / previous_data.close;
        
        Ok(price_stability - previous_stability)
    }
    
    fn get_name(&self) -> &str {
        &self.name
    }
}

/// Growth factor calculator
#[derive(Debug, Clone)]
pub struct GrowthFactor {
    name: String,
}

impl GrowthFactor {
    pub fn new() -> Self {
        Self {
            name: "Growth".to_string(),
        }
    }
}

impl FactorCalculator for GrowthFactor {
    fn calculate_loading(&self, market_history: &VecDeque<MarketData>) -> Result<f64, HedgeError> {
        if market_history.len() < 50 {
            return Ok(0.0);
        }
        
        let prices: Vec<f64> = market_history.iter().map(|d| d.close).collect();
        let short_term = &prices[prices.len() - 10..];
        let long_term = &prices[prices.len() - 50..prices.len() - 40];
        
        let short_term_avg = short_term.iter().sum::<f64>() / short_term.len() as f64;
        let long_term_avg = long_term.iter().sum::<f64>() / long_term.len() as f64;
        
        let growth = (short_term_avg - long_term_avg) / long_term_avg;
        Ok(growth)
    }
    
    fn calculate_return(&self, market_history: &VecDeque<MarketData>) -> Result<f64, HedgeError> {
        if market_history.len() < 2 {
            return Ok(0.0);
        }
        
        let current_data = market_history.back().unwrap();
        let previous_data = &market_history[market_history.len() - 2];
        
        let growth_return = (current_data.close - previous_data.close) / previous_data.close;
        Ok(growth_return)
    }
    
    fn get_name(&self) -> &str {
        &self.name
    }
}

/// Size factor calculator
#[derive(Debug, Clone)]
pub struct SizeFactor {
    name: String,
}

impl SizeFactor {
    pub fn new() -> Self {
        Self {
            name: "Size".to_string(),
        }
    }
}

impl FactorCalculator for SizeFactor {
    fn calculate_loading(&self, market_history: &VecDeque<MarketData>) -> Result<f64, HedgeError> {
        if market_history.is_empty() {
            return Ok(0.0);
        }
        
        let current_data = market_history.back().unwrap();
        let market_cap_proxy = current_data.close * current_data.volume;
        
        // Size factor (log of market cap proxy)
        let size = market_cap_proxy.ln();
        Ok(size)
    }
    
    fn calculate_return(&self, market_history: &VecDeque<MarketData>) -> Result<f64, HedgeError> {
        if market_history.len() < 2 {
            return Ok(0.0);
        }
        
        let current_data = market_history.back().unwrap();
        let previous_data = &market_history[market_history.len() - 2];
        
        let current_size = (current_data.close * current_data.volume).ln();
        let previous_size = (previous_data.close * previous_data.volume).ln();
        
        Ok(current_size - previous_size)
    }
    
    fn get_name(&self) -> &str {
        &self.name
    }
}

/// Profitability factor calculator
#[derive(Debug, Clone)]
pub struct ProfitabilityFactor {
    name: String,
}

impl ProfitabilityFactor {
    pub fn new() -> Self {
        Self {
            name: "Profitability".to_string(),
        }
    }
}

impl FactorCalculator for ProfitabilityFactor {
    fn calculate_loading(&self, market_history: &VecDeque<MarketData>) -> Result<f64, HedgeError> {
        if market_history.len() < 20 {
            return Ok(0.0);
        }
        
        let prices: Vec<f64> = market_history.iter().map(|d| d.close).collect();
        let returns = math::returns(&prices)?;
        
        // Profitability measured as average return
        let profitability = returns.iter().sum::<f64>() / returns.len() as f64;
        Ok(profitability)
    }
    
    fn calculate_return(&self, market_history: &VecDeque<MarketData>) -> Result<f64, HedgeError> {
        if market_history.len() < 2 {
            return Ok(0.0);
        }
        
        let current_data = market_history.back().unwrap();
        let previous_data = &market_history[market_history.len() - 2];
        
        let return_value = (current_data.close - previous_data.close) / previous_data.close;
        Ok(return_value)
    }
    
    fn get_name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_standard_factor_model_creation() {
        let config = HedgeConfig::default();
        let factor_model = StandardFactorModel::new(config).unwrap();
        
        assert_eq!(factor_model.factor_names.len(), 8);
        assert_eq!(factor_model.factor_loadings.len(), 8);
    }

    #[test]
    fn test_volatility_factor() {
        let factor = VolatilityFactor::new();
        assert_eq!(factor.get_name(), "Volatility");
        
        let mut market_history = VecDeque::new();
        market_history.push_back(MarketData::new(
            "BTCUSD".to_string(),
            Utc::now(),
            [100.0, 105.0, 95.0, 102.0, 1000.0]
        ));
        market_history.push_back(MarketData::new(
            "BTCUSD".to_string(),
            Utc::now(),
            [102.0, 108.0, 98.0, 105.0, 1100.0]
        ));
        
        let loading = factor.calculate_loading(&market_history).unwrap();
        assert!(loading > 0.0);
        
        let return_value = factor.calculate_return(&market_history).unwrap();
        assert!(return_value.abs() < 1.0);
    }

    #[test]
    fn test_momentum_factor() {
        let factor = MomentumFactor::new();
        assert_eq!(factor.get_name(), "Momentum");
        
        let mut market_history = VecDeque::new();
        for i in 0..25 {
            market_history.push_back(MarketData::new(
                "BTCUSD".to_string(),
                Utc::now(),
                [100.0 + i as f64, 105.0 + i as f64, 95.0 + i as f64, 102.0 + i as f64, 1000.0]
            ));
        }
        
        let loading = factor.calculate_loading(&market_history).unwrap();
        assert!(loading > 0.0);
    }

    #[test]
    fn test_factor_model_update() {
        let config = HedgeConfig::default();
        let mut factor_model = StandardFactorModel::new(config).unwrap();
        
        let market_data = MarketData::new(
            "BTCUSD".to_string(),
            Utc::now(),
            [100.0, 105.0, 95.0, 102.0, 1000.0]
        );
        
        factor_model.update(&market_data).unwrap();
        
        assert_eq!(factor_model.time_step, 1);
        assert_eq!(factor_model.market_history.len(), 1);
    }

    #[test]
    fn test_factor_exposures() {
        let config = HedgeConfig::default();
        let mut factor_model = StandardFactorModel::new(config).unwrap();
        
        for i in 0..10 {
            let market_data = MarketData::new(
                "BTCUSD".to_string(),
                Utc::now(),
                [100.0 + i as f64, 105.0 + i as f64, 95.0 + i as f64, 102.0 + i as f64, 1000.0]
            );
            factor_model.update(&market_data).unwrap();
        }
        
        let exposures = factor_model.get_exposures().unwrap();
        assert_eq!(exposures.len(), 8);
    }
}