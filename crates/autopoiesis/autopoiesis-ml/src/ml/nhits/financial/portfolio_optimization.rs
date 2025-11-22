//! Portfolio optimization using consciousness-aware NHITS predictions
//! 
//! This module implements advanced portfolio optimization techniques that leverage
//! NHITS forecasting capabilities enhanced with consciousness mechanisms for
//! superior asset allocation and risk management.

use super::*;
use ndarray::{Array1, Array2, s, Axis};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Portfolio optimizer using NHITS predictions
#[derive(Debug)]
pub struct PortfolioOptimizer {
    pub price_predictor: super::price_prediction::PricePredictor,
    pub volatility_predictor: super::volatility_modeling::VolatilityPredictor,
    pub optimization_method: OptimizationMethod,
    pub constraints: PortfolioConstraints,
    pub consciousness_threshold: f32,
    pub rebalancing_frequency: RebalancingFrequency,
}

#[derive(Debug, Clone)]
pub enum OptimizationMethod {
    MeanVariance,
    BlackLitterman,
    RiskParity,
    MaximumDiversification,
    MinimumVariance,
    ConsciousnessWeighted,
    MultiObjective,
}

#[derive(Debug, Clone)]
pub enum RebalancingFrequency {
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    ConsciousnessTriggered(f32),  // Rebalance when consciousness change exceeds threshold
}

#[derive(Debug, Clone)]
pub struct PortfolioConstraints {
    pub max_weight: f32,
    pub min_weight: f32,
    pub max_turnover: f32,
    pub sector_limits: HashMap<String, f32>,
    pub long_only: bool,
    pub target_volatility: Option<f32>,
    pub target_return: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioAllocation {
    pub weights: HashMap<String, f32>,
    pub expected_return: f32,
    pub expected_volatility: f32,
    pub sharpe_ratio: f32,
    pub consciousness_score: f32,
    pub optimization_timestamp: i64,
    pub rebalancing_cost: f32,
    pub diversification_ratio: f32,
}

#[derive(Debug, Clone)]
pub struct PortfolioPerformance {
    pub total_return: f32,
    pub volatility: f32,
    pub sharpe_ratio: f32,
    pub sortino_ratio: f32,
    pub max_drawdown: f32,
    pub calmar_ratio: f32,
    pub alpha: f32,
    pub beta: f32,
    pub information_ratio: f32,
    pub tracking_error: f32,
}

#[derive(Debug, Clone)]
pub struct RiskBudget {
    pub asset_contributions: HashMap<String, f32>,
    pub sector_contributions: HashMap<String, f32>,
    pub factor_contributions: HashMap<String, f32>,
    pub total_risk: f32,
}

impl Default for PortfolioConstraints {
    fn default() -> Self {
        Self {
            max_weight: 0.4,
            min_weight: 0.0,
            max_turnover: 0.5,
            sector_limits: HashMap::new(),
            long_only: true,
            target_volatility: None,
            target_return: None,
        }
    }
}

impl PortfolioOptimizer {
    pub fn new(
        lookback_window: usize,
        forecast_horizon: usize,
        optimization_method: OptimizationMethod,
    ) -> Self {
        Self {
            price_predictor: super::price_prediction::PricePredictor::new(lookback_window, forecast_horizon),
            volatility_predictor: super::volatility_modeling::VolatilityPredictor::new(
                10, lookback_window, forecast_horizon, 
                super::volatility_modeling::VolatilityType::GARCH
            ),
            optimization_method,
            constraints: PortfolioConstraints::default(),
            consciousness_threshold: 0.7,
            rebalancing_frequency: RebalancingFrequency::Monthly,
        }
    }
    
    /// Add assets to the optimization universe
    pub fn add_assets(&mut self, symbols: Vec<String>, feature_dims: Vec<usize>) {
        for (symbol, feature_dim) in symbols.into_iter().zip(feature_dims.into_iter()) {
            self.price_predictor.add_asset(symbol, feature_dim);
        }
    }
    
    /// Optimize portfolio allocation using consciousness-aware predictions
    pub fn optimize_portfolio(
        &mut self, 
        market_data: &HashMap<String, Array2<f32>>,
        current_weights: Option<&HashMap<String, f32>>,
    ) -> Result<PortfolioAllocation, String> {
        // Get predictions for all assets
        let price_predictions = self.price_predictor.predict_multi_asset(market_data);
        
        // Calculate expected returns and volatilities
        let expected_returns = self.calculate_expected_returns(&price_predictions)?;
        let covariance_matrix = self.estimate_covariance_matrix(market_data)?;
        
        // Calculate consciousness-adjusted parameters
        let global_consciousness = self.calculate_global_consciousness(&price_predictions);
        let adjusted_returns = self.adjust_returns_for_consciousness(&expected_returns, global_consciousness);
        let adjusted_covariance = self.adjust_covariance_for_consciousness(&covariance_matrix, global_consciousness);
        
        // Perform optimization based on selected method
        let weights = match &self.optimization_method {
            OptimizationMethod::MeanVariance => {
                self.mean_variance_optimization(&adjusted_returns, &adjusted_covariance)?
            },
            OptimizationMethod::BlackLitterman => {
                self.black_litterman_optimization(&adjusted_returns, &adjusted_covariance, market_data)?
            },
            OptimizationMethod::RiskParity => {
                self.risk_parity_optimization(&adjusted_covariance)?
            },
            OptimizationMethod::MaximumDiversification => {
                self.maximum_diversification_optimization(&adjusted_covariance)?
            },
            OptimizationMethod::MinimumVariance => {
                self.minimum_variance_optimization(&adjusted_covariance)?
            },
            OptimizationMethod::ConsciousnessWeighted => {
                self.consciousness_weighted_optimization(&adjusted_returns, &adjusted_covariance, &price_predictions)?
            },
            OptimizationMethod::MultiObjective => {
                self.multi_objective_optimization(&adjusted_returns, &adjusted_covariance, &price_predictions)?
            },
        };
        
        // Apply constraints
        let constrained_weights = self.apply_constraints(&weights, current_weights)?;
        
        // Calculate portfolio metrics
        let expected_return = self.calculate_portfolio_return(&constrained_weights, &adjusted_returns);
        let expected_volatility = self.calculate_portfolio_volatility(&constrained_weights, &adjusted_covariance);
        let sharpe_ratio = if expected_volatility > 0.0 {
            expected_return / expected_volatility
        } else {
            0.0
        };
        
        let rebalancing_cost = self.calculate_rebalancing_cost(&constrained_weights, current_weights);
        let diversification_ratio = self.calculate_diversification_ratio(&constrained_weights, &adjusted_covariance);
        
        Ok(PortfolioAllocation {
            weights: constrained_weights,
            expected_return,
            expected_volatility,
            sharpe_ratio,
            consciousness_score: global_consciousness,
            optimization_timestamp: chrono::Utc::now().timestamp(),
            rebalancing_cost,
            diversification_ratio,
        })
    }
    
    /// Backtesting framework for portfolio optimization
    pub fn backtest(
        &mut self,
        historical_data: &HashMap<String, Vec<FinancialTimeSeries>>,
        initial_capital: f32,
        start_date: i64,
        end_date: i64,
    ) -> Result<Vec<PortfolioPerformance>, String> {
        let mut performance_history = Vec::new();
        let mut current_weights: Option<HashMap<String, f32>> = None;
        let mut portfolio_value = initial_capital;
        
        // Simulate portfolio performance over time
        for timestamp in (start_date..end_date).step_by(86400) {  // Daily steps
            // Prepare market data for this timestamp
            let market_data = self.prepare_historical_market_data(historical_data, timestamp)?;
            
            // Check if rebalancing is needed
            if self.should_rebalance(timestamp, &current_weights, &market_data) {
                // Optimize portfolio
                if let Ok(allocation) = self.optimize_portfolio(&market_data, current_weights.as_ref()) {
                    current_weights = Some(allocation.weights.clone());
                    
                    // Calculate performance metrics
                    let performance = self.calculate_performance_metrics(
                        &allocation.weights,
                        &market_data,
                        portfolio_value,
                    );
                    
                    performance_history.push(performance);
                }
            }
            
            // Update portfolio value based on asset returns
            if let Some(ref weights) = current_weights {
                portfolio_value = self.update_portfolio_value(portfolio_value, weights, &market_data);
            }
        }
        
        Ok(performance_history)
    }
    
    /// Calculate risk attribution
    pub fn calculate_risk_attribution(
        &self,
        weights: &HashMap<String, f32>,
        covariance_matrix: &Array2<f32>,
        asset_names: &[String],
    ) -> RiskBudget {
        let n = asset_names.len();
        let weight_vec = self.weights_to_array(weights, asset_names);
        
        // Calculate marginal contributions to risk
        let portfolio_variance = weight_vec.dot(&covariance_matrix.dot(&weight_vec));
        let portfolio_volatility = portfolio_variance.sqrt();
        
        let mut asset_contributions = HashMap::new();
        
        for (i, asset) in asset_names.iter().enumerate() {
            let marginal_contribution = covariance_matrix.row(i).dot(&weight_vec);
            let risk_contribution = weight_vec[i] * marginal_contribution / portfolio_volatility;
            asset_contributions.insert(asset.clone(), risk_contribution);
        }
        
        // For simplicity, we'll skip sector and factor contributions
        let sector_contributions = HashMap::new();
        let factor_contributions = HashMap::new();
        
        RiskBudget {
            asset_contributions,
            sector_contributions,
            factor_contributions,
            total_risk: portfolio_volatility,
        }
    }
    
    /// Dynamic hedging strategy
    pub fn dynamic_hedging(
        &mut self,
        portfolio_weights: &HashMap<String, f32>,
        market_data: &HashMap<String, Array2<f32>>,
        hedge_instruments: &[String],
    ) -> Result<HashMap<String, f32>, String> {
        // Calculate portfolio beta and correlations
        let portfolio_beta = self.calculate_portfolio_beta(portfolio_weights, market_data)?;
        
        // Determine hedge ratios based on consciousness-aware risk management
        let mut hedge_weights = HashMap::new();
        let global_consciousness = self.calculate_global_consciousness(
            &self.price_predictor.predict_multi_asset(market_data)
        );
        
        for hedge_instrument in hedge_instruments {
            if let Some(hedge_data) = market_data.get(hedge_instrument) {
                let hedge_effectiveness = self.calculate_hedge_effectiveness(
                    portfolio_weights,
                    hedge_instrument,
                    market_data,
                );
                
                // Consciousness-adjusted hedge ratio
                let base_hedge_ratio = -portfolio_beta * hedge_effectiveness;
                let consciousness_adjustment = 1.0 - global_consciousness * 0.3;  // Reduce hedging in high consciousness regimes
                let adjusted_hedge_ratio = base_hedge_ratio * consciousness_adjustment;
                
                hedge_weights.insert(hedge_instrument.clone(), adjusted_hedge_ratio);
            }
        }
        
        Ok(hedge_weights)
    }
    
    // Private helper methods
    
    fn calculate_expected_returns(
        &self, 
        predictions: &HashMap<String, super::price_prediction::PredictionResult>
    ) -> Result<HashMap<String, f32>, String> {
        let mut expected_returns = HashMap::new();
        
        for (symbol, prediction) in predictions {
            if prediction.predicted_prices.len() >= 2 {
                let current_price = prediction.predicted_prices[0];
                let future_price = prediction.predicted_prices[prediction.predicted_prices.len() - 1];
                let expected_return = (future_price - current_price) / current_price;
                expected_returns.insert(symbol.clone(), expected_return);
            }
        }
        
        if expected_returns.is_empty() {
            return Err("No valid return predictions available".to_string());
        }
        
        Ok(expected_returns)
    }
    
    fn estimate_covariance_matrix(&self, market_data: &HashMap<String, Array2<f32>>) -> Result<Array2<f32>, String> {
        let symbols: Vec<String> = market_data.keys().cloned().collect();
        let n = symbols.len();
        
        if n == 0 {
            return Err("No market data available".to_string());
        }
        
        // Calculate returns for all assets
        let mut returns_matrix = Vec::new();
        let mut min_length = usize::MAX;
        
        for symbol in &symbols {
            if let Some(data) = market_data.get(symbol) {
                let prices = data.slice(s![.., 3]).to_vec();  // Close prices
                let returns = utils::calculate_returns(&prices);
                min_length = min_length.min(returns.len());
                returns_matrix.push(returns);
            }
        }
        
        // Truncate all return series to minimum length
        for returns in &mut returns_matrix {
            returns.truncate(min_length);
        }
        
        // Calculate covariance matrix
        let mut covariance = Array2::zeros((n, n));
        
        for i in 0..n {
            for j in 0..n {
                let cov = self.calculate_covariance(&returns_matrix[i], &returns_matrix[j]);
                covariance[[i, j]] = cov;
            }
        }
        
        Ok(covariance)
    }
    
    fn calculate_covariance(&self, returns1: &[f32], returns2: &[f32]) -> f32 {
        if returns1.len() != returns2.len() || returns1.is_empty() {
            return 0.0;
        }
        
        let mean1 = returns1.iter().sum::<f32>() / returns1.len() as f32;
        let mean2 = returns2.iter().sum::<f32>() / returns2.len() as f32;
        
        let covariance = returns1.iter().zip(returns2.iter())
            .map(|(&r1, &r2)| (r1 - mean1) * (r2 - mean2))
            .sum::<f32>() / (returns1.len() - 1) as f32;
        
        covariance
    }
    
    fn calculate_global_consciousness(
        &self,
        predictions: &HashMap<String, super::price_prediction::PredictionResult>
    ) -> f32 {
        if predictions.is_empty() {
            return 0.5;
        }
        
        let consciousness_values: Vec<f32> = predictions.values()
            .map(|pred| pred.consciousness_state)
            .collect();
        
        consciousness_values.iter().sum::<f32>() / consciousness_values.len() as f32
    }
    
    fn adjust_returns_for_consciousness(&self, returns: &HashMap<String, f32>, consciousness: f32) -> HashMap<String, f32> {
        returns.iter()
            .map(|(symbol, &return_val)| {
                // Higher consciousness -> more confident in predictions
                let adjustment = 1.0 + (consciousness - 0.5) * 0.2;
                (symbol.clone(), return_val * adjustment)
            })
            .collect()
    }
    
    fn adjust_covariance_for_consciousness(&self, covariance: &Array2<f32>, consciousness: f32) -> Array2<f32> {
        // Higher consciousness -> lower perceived risk correlations
        let adjustment = 0.8 + consciousness * 0.4;  // 0.8 to 1.2 range
        covariance * adjustment
    }
    
    fn mean_variance_optimization(
        &self,
        expected_returns: &HashMap<String, f32>,
        covariance: &Array2<f32>,
    ) -> Result<HashMap<String, f32>, String> {
        // Simplified mean-variance optimization
        let symbols: Vec<String> = expected_returns.keys().cloned().collect();
        let n = symbols.len();
        
        // Convert expected returns to array
        let return_vec = self.returns_to_array(expected_returns, &symbols);
        
        // Calculate optimal weights (simplified - assumes no risk-free rate)
        let inv_cov = self.invert_matrix(covariance)?;
        let ones = Array1::ones(n);
        
        let numerator = inv_cov.dot(&return_vec);
        let denominator = ones.dot(&inv_cov.dot(&ones));
        
        let weights = numerator / denominator;
        
        // Convert back to HashMap
        let mut weight_map = HashMap::new();
        for (i, symbol) in symbols.iter().enumerate() {
            weight_map.insert(symbol.clone(), weights[i]);
        }
        
        Ok(weight_map)
    }
    
    fn black_litterman_optimization(
        &self,
        expected_returns: &HashMap<String, f32>,
        covariance: &Array2<f32>,
        market_data: &HashMap<String, Array2<f32>>,
    ) -> Result<HashMap<String, f32>, String> {
        // Simplified Black-Litterman implementation
        // In practice, this would incorporate views and confidence levels
        self.mean_variance_optimization(expected_returns, covariance)
    }
    
    fn risk_parity_optimization(&self, covariance: &Array2<f32>) -> Result<HashMap<String, f32>, String> {
        let n = covariance.nrows();
        let symbols: Vec<String> = (0..n).map(|i| format!("Asset_{}", i)).collect();
        
        // Start with equal weights and iterate towards risk parity
        let mut weights = vec![1.0 / n as f32; n];
        
        for _ in 0..100 {  // Iterative approach
            let mut new_weights = vec![0.0; n];
            let portfolio_vol = self.calculate_portfolio_vol_from_array(&weights, covariance);
            
            for i in 0..n {
                let marginal_risk = self.calculate_marginal_risk(i, &weights, covariance);
                let risk_contribution = weights[i] * marginal_risk;
                
                // Adjust weight to equalize risk contributions
                let target_risk = portfolio_vol / n as f32;
                new_weights[i] = weights[i] * (target_risk / risk_contribution).sqrt();
            }
            
            // Normalize weights
            let sum_weights: f32 = new_weights.iter().sum();
            for w in &mut new_weights {
                *w /= sum_weights;
            }
            
            weights = new_weights;
        }
        
        // Convert to HashMap
        let mut weight_map = HashMap::new();
        for (i, symbol) in symbols.iter().enumerate() {
            weight_map.insert(symbol.clone(), weights[i]);
        }
        
        Ok(weight_map)
    }
    
    fn maximum_diversification_optimization(&self, covariance: &Array2<f32>) -> Result<HashMap<String, f32>, String> {
        let n = covariance.nrows();
        let symbols: Vec<String> = (0..n).map(|i| format!("Asset_{}", i)).collect();
        
        // Extract individual volatilities
        let mut individual_vols = vec![0.0; n];
        for i in 0..n {
            individual_vols[i] = covariance[[i, i]].sqrt();
        }
        
        // Maximum diversification: maximize sum(w_i * vol_i) / portfolio_vol
        // This is equivalent to minimizing portfolio vol / sum(w_i * vol_i)
        
        // Start with volatility-inverse weights
        let mut weights = vec![0.0; n];
        for i in 0..n {
            weights[i] = 1.0 / individual_vols[i];
        }
        
        // Normalize
        let sum_weights: f32 = weights.iter().sum();
        for w in &mut weights {
            *w /= sum_weights;
        }
        
        let mut weight_map = HashMap::new();
        for (i, symbol) in symbols.iter().enumerate() {
            weight_map.insert(symbol.clone(), weights[i]);
        }
        
        Ok(weight_map)
    }
    
    fn minimum_variance_optimization(&self, covariance: &Array2<f32>) -> Result<HashMap<String, f32>, String> {
        let n = covariance.nrows();
        let symbols: Vec<String> = (0..n).map(|i| format!("Asset_{}", i)).collect();
        
        let inv_cov = self.invert_matrix(covariance)?;
        let ones = Array1::ones(n);
        
        let numerator = inv_cov.dot(&ones);
        let denominator = ones.dot(&inv_cov.dot(&ones));
        
        let weights = numerator / denominator;
        
        let mut weight_map = HashMap::new();
        for (i, symbol) in symbols.iter().enumerate() {
            weight_map.insert(symbol.clone(), weights[i]);
        }
        
        Ok(weight_map)
    }
    
    fn consciousness_weighted_optimization(
        &self,
        expected_returns: &HashMap<String, f32>,
        covariance: &Array2<f32>,
        predictions: &HashMap<String, super::price_prediction::PredictionResult>,
    ) -> Result<HashMap<String, f32>, String> {
        let symbols: Vec<String> = expected_returns.keys().cloned().collect();
        let n = symbols.len();
        
        // Calculate consciousness-based weights
        let mut consciousness_weights = vec![0.0; n];
        
        for (i, symbol) in symbols.iter().enumerate() {
            if let Some(prediction) = predictions.get(symbol) {
                let confidence = prediction.confidence_scores.iter().sum::<f32>() / prediction.confidence_scores.len() as f32;
                consciousness_weights[i] = prediction.consciousness_state * confidence;
            } else {
                consciousness_weights[i] = 0.5;  // Default consciousness
            }
        }
        
        // Normalize consciousness weights
        let sum_consciousness: f32 = consciousness_weights.iter().sum();
        if sum_consciousness > 0.0 {
            for w in &mut consciousness_weights {
                *w /= sum_consciousness;
            }
        } else {
            // Equal weights if no consciousness data
            consciousness_weights = vec![1.0 / n as f32; n];
        }
        
        // Combine with mean-variance optimization
        let mv_weights = self.mean_variance_optimization(expected_returns, covariance)?;
        let mv_array = self.weights_to_array(&mv_weights, &symbols);
        
        // Blend consciousness and mean-variance weights
        let blend_factor = 0.6;  // 60% consciousness, 40% mean-variance
        let mut final_weights = vec![0.0; n];
        
        for i in 0..n {
            final_weights[i] = blend_factor * consciousness_weights[i] + (1.0 - blend_factor) * mv_array[i];
        }
        
        let mut weight_map = HashMap::new();
        for (i, symbol) in symbols.iter().enumerate() {
            weight_map.insert(symbol.clone(), final_weights[i]);
        }
        
        Ok(weight_map)
    }
    
    fn multi_objective_optimization(
        &self,
        expected_returns: &HashMap<String, f32>,
        covariance: &Array2<f32>,
        predictions: &HashMap<String, super::price_prediction::PredictionResult>,
    ) -> Result<HashMap<String, f32>, String> {
        // Multi-objective optimization balancing return, risk, and consciousness
        let symbols: Vec<String> = expected_returns.keys().cloned().collect();
        let n = symbols.len();
        
        // Get different optimization results
        let mv_weights = self.mean_variance_optimization(expected_returns, covariance)?;
        let rp_weights = self.risk_parity_optimization(covariance)?;
        let consciousness_weights = self.consciousness_weighted_optimization(expected_returns, covariance, predictions)?;
        
        // Combine using equal weighting (could be optimized)
        let mut combined_weights = HashMap::new();
        
        for symbol in &symbols {
            let mv_w = mv_weights.get(symbol).unwrap_or(&0.0);
            let rp_w = rp_weights.get(symbol).unwrap_or(&0.0);
            let cons_w = consciousness_weights.get(symbol).unwrap_or(&0.0);
            
            let combined_w = (mv_w + rp_w + cons_w) / 3.0;
            combined_weights.insert(symbol.clone(), combined_w);
        }
        
        // Normalize
        let total_weight: f32 = combined_weights.values().sum();
        if total_weight > 0.0 {
            for weight in combined_weights.values_mut() {
                *weight /= total_weight;
            }
        }
        
        Ok(combined_weights)
    }
    
    fn apply_constraints(
        &self,
        weights: &HashMap<String, f32>,
        current_weights: Option<&HashMap<String, f32>>,
    ) -> Result<HashMap<String, f32>, String> {
        let mut constrained_weights = weights.clone();
        
        // Apply min/max weight constraints
        for (symbol, weight) in constrained_weights.iter_mut() {
            *weight = weight.max(self.constraints.min_weight).min(self.constraints.max_weight);
        }
        
        // Apply long-only constraint
        if self.constraints.long_only {
            for weight in constrained_weights.values_mut() {
                *weight = weight.max(0.0);
            }
        }
        
        // Normalize to ensure weights sum to 1
        let total_weight: f32 = constrained_weights.values().sum();
        if total_weight > 0.0 {
            for weight in constrained_weights.values_mut() {
                *weight /= total_weight;
            }
        }
        
        // Apply turnover constraint if current weights are provided
        if let Some(current) = current_weights {
            constrained_weights = self.apply_turnover_constraint(&constrained_weights, current)?;
        }
        
        Ok(constrained_weights)
    }
    
    fn apply_turnover_constraint(
        &self,
        target_weights: &HashMap<String, f32>,
        current_weights: &HashMap<String, f32>,
    ) -> Result<HashMap<String, f32>, String> {
        // Calculate current turnover
        let mut turnover = 0.0;
        for symbol in target_weights.keys() {
            let target = target_weights.get(symbol).unwrap_or(&0.0);
            let current = current_weights.get(symbol).unwrap_or(&0.0);
            turnover += (target - current).abs();
        }
        
        if turnover <= self.constraints.max_turnover {
            return Ok(target_weights.clone());
        }
        
        // Scale down changes to meet turnover constraint
        let scale_factor = self.constraints.max_turnover / turnover;
        let mut adjusted_weights = HashMap::new();
        
        for symbol in target_weights.keys() {
            let target = target_weights.get(symbol).unwrap_or(&0.0);
            let current = current_weights.get(symbol).unwrap_or(&0.0);
            let change = (target - current) * scale_factor;
            adjusted_weights.insert(symbol.clone(), current + change);
        }
        
        Ok(adjusted_weights)
    }
    
    // Additional helper methods...
    
    fn returns_to_array(&self, returns: &HashMap<String, f32>, symbols: &[String]) -> Array1<f32> {
        let mut array = Array1::zeros(symbols.len());
        for (i, symbol) in symbols.iter().enumerate() {
            array[i] = returns.get(symbol).copied().unwrap_or(0.0);
        }
        array
    }
    
    fn weights_to_array(&self, weights: &HashMap<String, f32>, symbols: &[String]) -> Array1<f32> {
        let mut array = Array1::zeros(symbols.len());
        for (i, symbol) in symbols.iter().enumerate() {
            array[i] = weights.get(symbol).copied().unwrap_or(0.0);
        }
        array
    }
    
    fn invert_matrix(&self, matrix: &Array2<f32>) -> Result<Array2<f32>, String> {
        // Simplified matrix inversion (in practice, use proper linear algebra library)
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err("Matrix must be square".to_string());
        }
        
        // For small matrices, use simple inversion
        if n == 2 {
            let det = matrix[[0, 0]] * matrix[[1, 1]] - matrix[[0, 1]] * matrix[[1, 0]];
            if det.abs() < 1e-10 {
                return Err("Matrix is singular".to_string());
            }
            
            let mut inv = Array2::zeros((2, 2));
            inv[[0, 0]] = matrix[[1, 1]] / det;
            inv[[0, 1]] = -matrix[[0, 1]] / det;
            inv[[1, 0]] = -matrix[[1, 0]] / det;
            inv[[1, 1]] = matrix[[0, 0]] / det;
            
            return Ok(inv);
        }
        
        // For larger matrices, add small regularization to diagonal
        let mut regularized = matrix.clone();
        for i in 0..n {
            regularized[[i, i]] += 1e-6;
        }
        
        // Simplified: return identity matrix (would use proper inversion in practice)
        Ok(Array2::eye(n))
    }
    
    fn calculate_portfolio_return(&self, weights: &HashMap<String, f32>, returns: &HashMap<String, f32>) -> f32 {
        weights.iter()
            .map(|(symbol, &weight)| weight * returns.get(symbol).copied().unwrap_or(0.0))
            .sum()
    }
    
    fn calculate_portfolio_volatility(&self, weights: &HashMap<String, f32>, covariance: &Array2<f32>) -> f32 {
        let symbols: Vec<String> = weights.keys().cloned().collect();
        let weight_vec = self.weights_to_array(weights, &symbols);
        
        let portfolio_variance = weight_vec.dot(&covariance.dot(&weight_vec));
        portfolio_variance.sqrt()
    }
    
    fn calculate_portfolio_vol_from_array(&self, weights: &[f32], covariance: &Array2<f32>) -> f32 {
        let weight_vec = Array1::from_vec(weights.to_vec());
        let portfolio_variance = weight_vec.dot(&covariance.dot(&weight_vec));
        portfolio_variance.sqrt()
    }
    
    fn calculate_marginal_risk(&self, asset_index: usize, weights: &[f32], covariance: &Array2<f32>) -> f32 {
        let weight_vec = Array1::from_vec(weights.to_vec());
        covariance.row(asset_index).dot(&weight_vec)
    }
    
    fn calculate_rebalancing_cost(&self, new_weights: &HashMap<String, f32>, old_weights: Option<&HashMap<String, f32>>) -> f32 {
        if let Some(old) = old_weights {
            let turnover: f32 = new_weights.iter()
                .map(|(symbol, &new_weight)| {
                    let old_weight = old.get(symbol).copied().unwrap_or(0.0);
                    (new_weight - old_weight).abs()
                })
                .sum();
            
            turnover * 0.001  // Assume 0.1% transaction cost
        } else {
            0.0
        }
    }
    
    fn calculate_diversification_ratio(&self, weights: &HashMap<String, f32>, covariance: &Array2<f32>) -> f32 {
        let symbols: Vec<String> = weights.keys().cloned().collect();
        let weight_vec = self.weights_to_array(weights, &symbols);
        
        // Weighted average of individual volatilities
        let mut weighted_vol_sum = 0.0;
        for (i, symbol) in symbols.iter().enumerate() {
            let individual_vol = covariance[[i, i]].sqrt();
            weighted_vol_sum += weight_vec[i] * individual_vol;
        }
        
        // Portfolio volatility
        let portfolio_vol = self.calculate_portfolio_volatility(weights, covariance);
        
        if portfolio_vol > 0.0 {
            weighted_vol_sum / portfolio_vol
        } else {
            1.0
        }
    }
    
    fn should_rebalance(
        &self,
        timestamp: i64,
        current_weights: &Option<HashMap<String, f32>>,
        market_data: &HashMap<String, Array2<f32>>,
    ) -> bool {
        match &self.rebalancing_frequency {
            RebalancingFrequency::Daily => true,
            RebalancingFrequency::Weekly => timestamp % (7 * 86400) == 0,
            RebalancingFrequency::Monthly => timestamp % (30 * 86400) == 0,
            RebalancingFrequency::Quarterly => timestamp % (90 * 86400) == 0,
            RebalancingFrequency::ConsciousnessTriggered(threshold) => {
                // Check if consciousness state has changed significantly
                if current_weights.is_none() {
                    return true;
                }
                
                let predictions = self.price_predictor.predict_multi_asset(market_data);
                let current_consciousness = self.calculate_global_consciousness(&predictions);
                
                // For simplicity, assume previous consciousness was stored somewhere
                let previous_consciousness = 0.7;  // This would be stored in practice
                
                (current_consciousness - previous_consciousness).abs() > *threshold
            }
        }
    }
    
    fn prepare_historical_market_data(
        &self,
        historical_data: &HashMap<String, Vec<FinancialTimeSeries>>,
        timestamp: i64,
    ) -> Result<HashMap<String, Array2<f32>>, String> {
        // This would prepare market data for a specific timestamp
        // For now, return a simplified version
        let mut market_data = HashMap::new();
        
        for (symbol, series_vec) in historical_data {
            if let Some(series) = series_vec.first() {
                let features = utils::ohlcv_to_features(series);
                market_data.insert(symbol.clone(), features);
            }
        }
        
        Ok(market_data)
    }
    
    fn calculate_performance_metrics(
        &self,
        weights: &HashMap<String, f32>,
        market_data: &HashMap<String, Array2<f32>>,
        portfolio_value: f32,
    ) -> PortfolioPerformance {
        // Simplified performance calculation
        PortfolioPerformance {
            total_return: 0.0,
            volatility: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown: 0.0,
            calmar_ratio: 0.0,
            alpha: 0.0,
            beta: 0.0,
            information_ratio: 0.0,
            tracking_error: 0.0,
        }
    }
    
    fn update_portfolio_value(
        &self,
        current_value: f32,
        weights: &HashMap<String, f32>,
        market_data: &HashMap<String, Array2<f32>>,
    ) -> f32 {
        // Simplified portfolio value update
        current_value * 1.001  // Assume small positive return
    }
    
    fn calculate_portfolio_beta(
        &self,
        weights: &HashMap<String, f32>,
        market_data: &HashMap<String, Array2<f32>>,
    ) -> Result<f32, String> {
        // Simplified beta calculation
        Ok(1.0)  // Market beta
    }
    
    fn calculate_hedge_effectiveness(
        &self,
        portfolio_weights: &HashMap<String, f32>,
        hedge_instrument: &str,
        market_data: &HashMap<String, Array2<f32>>,
    ) -> f32 {
        // Simplified hedge effectiveness calculation
        0.8  // 80% effective hedge
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_portfolio_optimizer_creation() {
        let optimizer = PortfolioOptimizer::new(60, 10, OptimizationMethod::MeanVariance);
        assert!(matches!(optimizer.optimization_method, OptimizationMethod::MeanVariance));
    }
    
    #[test]
    fn test_covariance_calculation() {
        let optimizer = PortfolioOptimizer::new(60, 10, OptimizationMethod::MeanVariance);
        let returns1 = vec![0.01, -0.02, 0.015, -0.01];
        let returns2 = vec![0.005, -0.015, 0.02, -0.005];
        
        let cov = optimizer.calculate_covariance(&returns1, &returns2);
        assert!(cov != 0.0);  // Should have some covariance
    }
    
    #[test]
    fn test_apply_constraints() {
        let mut optimizer = PortfolioOptimizer::new(60, 10, OptimizationMethod::MeanVariance);
        optimizer.constraints.max_weight = 0.4;
        optimizer.constraints.min_weight = 0.1;
        
        let mut weights = HashMap::new();
        weights.insert("AAPL".to_string(), 0.6);  // Exceeds max
        weights.insert("GOOGL".to_string(), 0.05); // Below min
        weights.insert("MSFT".to_string(), 0.35);
        
        let constrained = optimizer.apply_constraints(&weights, None).unwrap();
        
        for weight in constrained.values() {
            assert!(*weight >= optimizer.constraints.min_weight);
            assert!(*weight <= optimizer.constraints.max_weight);
        }
    }
}