use wasm_bindgen::prelude::*;
use js_sys::Array;
use std::collections::HashMap;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

// Note: cwts_ultra::algorithms types are mocked for WASM compatibility
// The full implementation requires the core library to be WASM-compatible

/// Bayesian parameters for probabilistic risk engine
#[derive(Clone, Debug, Default)]
pub struct BayesianParameters {
    pub prior_alpha: f64,
    pub prior_beta: f64,
    pub learning_rate: f64,
}

/// Probabilistic risk engine (WASM-compatible implementation)
pub struct ProbabilisticRiskEngine {
    params: BayesianParameters,
    returns_data: Vec<f64>,
    estimated_volatility: Option<f64>,
}

impl ProbabilisticRiskEngine {
    pub fn new(params: BayesianParameters) -> Self {
        Self {
            params,
            returns_data: Vec::new(),
            estimated_volatility: None,
        }
    }

    pub fn bayesian_parameter_estimation(&mut self, returns: &[f64]) -> Result<(f64, f64), String> {
        if returns.is_empty() {
            return Err("No returns data provided".to_string());
        }

        self.returns_data = returns.to_vec();

        // Calculate sample statistics
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (returns.len() - 1) as f64;
        let volatility = variance.sqrt();

        // Bayesian posterior update (simplified conjugate normal-inverse-gamma)
        let n = returns.len() as f64;
        let posterior_alpha = self.params.prior_alpha + n / 2.0;
        let posterior_beta = self.params.prior_beta + variance * (n - 1.0) / 2.0;

        // Posterior mean and uncertainty
        let posterior_volatility = (posterior_beta / (posterior_alpha - 1.0)).sqrt();
        let uncertainty = posterior_volatility / (n.sqrt());

        self.estimated_volatility = Some(posterior_volatility);

        Ok((posterior_volatility, uncertainty))
    }

    pub fn monte_carlo_var_with_variance_reduction(
        &mut self,
        portfolio_value: f64,
        confidence_levels: &[f64],
        iterations: usize,
    ) -> Result<HashMap<String, f64>, String> {
        let volatility = self.estimated_volatility.unwrap_or(0.02);

        let mut rng = StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, volatility).map_err(|e| e.to_string())?;

        // Antithetic variates for variance reduction
        let mut simulated_returns: Vec<f64> = Vec::with_capacity(iterations * 2);
        for _ in 0..iterations {
            let z = normal.sample(&mut rng);
            simulated_returns.push(z);
            simulated_returns.push(-z); // Antithetic
        }

        // Calculate portfolio losses
        let mut losses: Vec<f64> = simulated_returns
            .iter()
            .map(|r| -portfolio_value * r)
            .collect();
        losses.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let mut results = HashMap::new();
        for &level in confidence_levels {
            let index = ((1.0 - level) * losses.len() as f64) as usize;
            let var = losses.get(index).copied().unwrap_or(0.0);
            let key = format!("var_{}", (level * 100.0) as u32);
            results.insert(key, var);
        }

        Ok(results)
    }

    pub fn model_heavy_tail_distribution(&mut self) -> Result<HeavyTailDistribution, String> {
        if self.returns_data.is_empty() {
            return Err("No returns data available".to_string());
        }

        let returns = &self.returns_data;
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (returns.len() - 1) as f64;
        let std_dev = variance.sqrt();

        // Calculate kurtosis
        let fourth_moment = returns.iter().map(|r| ((r - mean) / std_dev).powi(4)).sum::<f64>() / returns.len() as f64;
        let kurtosis = fourth_moment - 3.0; // Excess kurtosis

        // Estimate degrees of freedom for Student-t (method of moments)
        let nu = if kurtosis > 0.0 {
            4.0 + 6.0 / kurtosis
        } else {
            30.0 // Approximate normal
        };

        Ok(HeavyTailDistribution {
            degrees_of_freedom: nu,
            location: mean,
            scale: std_dev,
            tail_index: 1.0 / nu,
            kurtosis,
        })
    }

    pub fn propagate_uncertainty_real_time(&mut self, _new_price: f64, previous_uncertainty: f64) -> Result<f64, String> {
        // Kalman-like uncertainty propagation
        let process_noise: f64 = 0.001;
        let measurement_noise: f64 = 0.005;

        let predicted_uncertainty = (previous_uncertainty.powi(2) + process_noise.powi(2)).sqrt();
        let kalman_gain = predicted_uncertainty.powi(2) / (predicted_uncertainty.powi(2) + measurement_noise.powi(2));
        let updated_uncertainty = ((1.0 - kalman_gain) * predicted_uncertainty.powi(2)).sqrt();

        Ok(updated_uncertainty)
    }

    pub fn update_regime_probabilities(&mut self, _conditions: &HashMap<String, f64>) -> Result<(), String> {
        // Placeholder for regime switching model
        Ok(())
    }

    pub fn generate_comprehensive_metrics(&mut self, portfolio_value: f64, conditions: &HashMap<String, f64>) -> Result<ProbabilisticRiskMetrics, String> {
        let volatility = self.estimated_volatility.unwrap_or(0.02);
        let var_95 = portfolio_value * volatility * 1.645; // Normal approximation
        let var_99 = portfolio_value * volatility * 2.326;

        Ok(ProbabilisticRiskMetrics {
            var_95,
            var_99,
            expected_shortfall_95: var_95 * 1.1,
            expected_shortfall_99: var_99 * 1.1,
            volatility,
            regime_probability: 0.7,
        })
    }
}

/// Heavy tail distribution parameters
#[derive(Clone, Debug)]
pub struct HeavyTailDistribution {
    pub degrees_of_freedom: f64,
    pub location: f64,
    pub scale: f64,
    pub tail_index: f64,
    pub kurtosis: f64,
}

/// Probabilistic risk metrics
#[derive(Clone, Debug)]
pub struct ProbabilisticRiskMetrics {
    pub var_95: f64,
    pub var_99: f64,
    pub expected_shortfall_95: f64,
    pub expected_shortfall_99: f64,
    pub volatility: f64,
    pub regime_probability: f64,
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct WasmBayesianParameters {
    inner: BayesianParameters,
}

#[wasm_bindgen]
impl WasmBayesianParameters {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmBayesianParameters {
        WasmBayesianParameters {
            inner: BayesianParameters::default(),
        }
    }

    #[wasm_bindgen(setter)]
    pub fn set_prior_alpha(&mut self, value: f64) {
        self.inner.prior_alpha = value;
    }

    #[wasm_bindgen(getter)]
    pub fn prior_alpha(&self) -> f64 {
        self.inner.prior_alpha
    }

    #[wasm_bindgen(setter)]
    pub fn set_prior_beta(&mut self, value: f64) {
        self.inner.prior_beta = value;
    }

    #[wasm_bindgen(getter)]
    pub fn prior_beta(&self) -> f64 {
        self.inner.prior_beta
    }

    #[wasm_bindgen(setter)]
    pub fn set_learning_rate(&mut self, value: f64) {
        self.inner.learning_rate = value;
    }

    #[wasm_bindgen(getter)]
    pub fn learning_rate(&self) -> f64 {
        self.inner.learning_rate
    }
}

#[wasm_bindgen]
pub struct WasmProbabilisticRiskEngine {
    inner: ProbabilisticRiskEngine,
}

#[wasm_bindgen]
impl WasmProbabilisticRiskEngine {
    #[wasm_bindgen(constructor)]
    pub fn new(params: &WasmBayesianParameters) -> WasmProbabilisticRiskEngine {
        console_log!("Initializing Probabilistic Risk Engine");
        WasmProbabilisticRiskEngine {
            inner: ProbabilisticRiskEngine::new(params.inner.clone()),
        }
    }

    #[wasm_bindgen]
    pub fn bayesian_parameter_estimation(&mut self, returns: &[f64]) -> Result<JsValue, JsValue> {
        match self.inner.bayesian_parameter_estimation(returns) {
            Ok((volatility, uncertainty)) => {
                let result = js_sys::Object::new();
                js_sys::Reflect::set(&result, &"volatility".into(), &volatility.into())?;
                js_sys::Reflect::set(&result, &"uncertainty".into(), &uncertainty.into())?;
                Ok(result.into())
            }
            Err(e) => Err(JsValue::from_str(&e.to_string())),
        }
    }

    #[wasm_bindgen]
    pub fn monte_carlo_var(&mut self, portfolio_value: f64, confidence_levels: &[f64], iterations: usize) -> Result<JsValue, JsValue> {
        match self.inner.monte_carlo_var_with_variance_reduction(portfolio_value, confidence_levels, iterations) {
            Ok(results) => {
                let js_object = js_sys::Object::new();
                for (key, value) in results.iter() {
                    js_sys::Reflect::set(&js_object, &key.as_str().into(), &(*value).into())?;
                }
                Ok(js_object.into())
            }
            Err(e) => Err(JsValue::from_str(&e.to_string())),
        }
    }

    #[wasm_bindgen]
    pub fn model_heavy_tail_distribution(&mut self) -> Result<JsValue, JsValue> {
        match self.inner.model_heavy_tail_distribution() {
            Ok(dist) => {
                let result = js_sys::Object::new();
                js_sys::Reflect::set(&result, &"degrees_of_freedom".into(), &dist.degrees_of_freedom.into())?;
                js_sys::Reflect::set(&result, &"location".into(), &dist.location.into())?;
                js_sys::Reflect::set(&result, &"scale".into(), &dist.scale.into())?;
                js_sys::Reflect::set(&result, &"tail_index".into(), &dist.tail_index.into())?;
                js_sys::Reflect::set(&result, &"kurtosis".into(), &dist.kurtosis.into())?;
                Ok(result.into())
            }
            Err(e) => Err(JsValue::from_str(&e.to_string())),
        }
    }

    #[wasm_bindgen]
    pub fn propagate_uncertainty(&mut self, new_price: f64, previous_uncertainty: f64) -> Result<f64, JsValue> {
        match self.inner.propagate_uncertainty_real_time(new_price, previous_uncertainty) {
            Ok(uncertainty) => Ok(uncertainty),
            Err(e) => Err(JsValue::from_str(&e.to_string())),
        }
    }

    #[wasm_bindgen]
    pub fn update_regime_probabilities(&mut self, market_conditions: JsValue) -> Result<(), JsValue> {
        let conditions: HashMap<String, f64> = serde_wasm_bindgen::from_value(market_conditions)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        match self.inner.update_regime_probabilities(&conditions) {
            Ok(()) => Ok(()),
            Err(e) => Err(JsValue::from_str(&e)),
        }
    }

    #[wasm_bindgen]
    pub fn generate_comprehensive_metrics(&mut self, portfolio_value: f64, market_conditions: JsValue) -> Result<JsValue, JsValue> {
        let conditions: HashMap<String, f64> = serde_wasm_bindgen::from_value(market_conditions)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        match self.inner.generate_comprehensive_metrics(portfolio_value, &conditions) {
            Ok(metrics) => {
                let result = js_sys::Object::new();
                js_sys::Reflect::set(&result, &"var_95".into(), &metrics.var_95.into())?;
                js_sys::Reflect::set(&result, &"var_99".into(), &metrics.var_99.into())?;
                js_sys::Reflect::set(&result, &"expected_shortfall_95".into(), &metrics.expected_shortfall_95.into())?;
                js_sys::Reflect::set(&result, &"expected_shortfall_99".into(), &metrics.expected_shortfall_99.into())?;
                js_sys::Reflect::set(&result, &"volatility".into(), &metrics.volatility.into())?;
                js_sys::Reflect::set(&result, &"regime_probability".into(), &metrics.regime_probability.into())?;
                Ok(result.into())
            }
            Err(e) => Err(JsValue::from_str(&e)),
        }
    }
}

// Utility functions for JavaScript interop

#[wasm_bindgen]
pub fn create_sample_market_data(volatility: f64, trend: f64, noise_level: f64, length: usize) -> Array {
    use rand::prelude::*;
    use rand_distr::Normal;

    let mut rng = StdRng::from_entropy();
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut price = 100.0;
    let prices = Array::new();

    for i in 0..length {
        let noise = normal.sample(&mut rng) * noise_level;
        let trend_component = trend * (i as f64 / length as f64);
        let volatility_component = volatility * normal.sample(&mut rng);
        
        price += trend_component + volatility_component + noise;
        prices.push(&JsValue::from(price));
    }

    prices
}

#[wasm_bindgen]
pub fn calculate_returns_from_prices(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return Vec::new();
    }

    prices
        .windows(2)
        .map(|window| (window[1] - window[0]) / window[0])
        .collect()
}

#[wasm_bindgen]
pub fn calculate_rolling_volatility(returns: &[f64], window_size: usize) -> Array {
    if returns.len() < window_size || window_size < 2 {
        return Array::new();
    }

    let volatilities = Array::new();

    for i in window_size..=returns.len() {
        let window_returns = &returns[i - window_size..i];
        let mean = window_returns.iter().sum::<f64>() / window_size as f64;
        let variance = window_returns
            .iter()
            .map(|&r| (r - mean).powi(2))
            .sum::<f64>() / (window_size - 1) as f64;
        
        volatilities.push(&JsValue::from(variance.sqrt() * (252.0_f64).sqrt()));
    }

    volatilities
}

// Performance benchmark for WASM
#[wasm_bindgen]
pub fn benchmark_monte_carlo(iterations: usize) -> JsValue {
    use std::time::Instant;
    
    let start = Instant::now();
    
    let mut engine = ProbabilisticRiskEngine::new(BayesianParameters::default());
    
    // Generate synthetic data
    use rand::prelude::*;
    use rand_distr::Normal;
    let mut rng = StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, 0.02).unwrap();
    let returns: Vec<f64> = (0..252).map(|_| normal.sample(&mut rng)).collect();
    
    // Add returns to engine
    let _ = engine.bayesian_parameter_estimation(&returns);
    
    // Run Monte Carlo
    let confidence_levels = vec![0.95, 0.99];
    let result = engine.monte_carlo_var_with_variance_reduction(
        100000.0, &confidence_levels, iterations
    );
    
    let duration = start.elapsed();
    
    let benchmark_result = js_sys::Object::new();
    js_sys::Reflect::set(&benchmark_result, &"duration_ms".into(), 
                         &(duration.as_millis() as f64).into()).unwrap();
    js_sys::Reflect::set(&benchmark_result, &"iterations".into(), 
                         &(iterations as f64).into()).unwrap();
    js_sys::Reflect::set(&benchmark_result, &"success".into(), 
                         &result.is_ok().into()).unwrap();
    
    if let Ok(vars) = result {
        let var_95 = *vars.get("var_95").unwrap_or(&0.0);
        let var_99 = *vars.get("var_99").unwrap_or(&0.0);
        js_sys::Reflect::set(&benchmark_result, &"var_95".into(),
                             &var_95.into()).unwrap();
        js_sys::Reflect::set(&benchmark_result, &"var_99".into(),
                             &var_99.into()).unwrap();
    }
    
    console_log!("Monte Carlo benchmark completed: {} iterations in {}ms", 
                 iterations, duration.as_millis());
    
    benchmark_result.into()
}

// Initialize WASM module
#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
    console_log!("CWTS Probabilistic Risk Engine WASM module initialized");
}

// Memory management utilities
#[wasm_bindgen]
pub fn get_memory_usage() -> JsValue {
    let result = js_sys::Object::new();
    
    // WASM memory is managed by the runtime, but we can provide estimates
    js_sys::Reflect::set(&result, &"estimated_heap_kb".into(), 
                         &1024.0.into()).unwrap(); // Placeholder
    
    result.into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_wasm_probabilistic_engine() {
        let params = WasmBayesianParameters::new();
        let mut engine = WasmProbabilisticRiskEngine::new(&params);
        
        let returns = vec![0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.008, -0.012, 0.018, -0.007];
        
        let result = engine.bayesian_parameter_estimation(&returns);
        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_sample_data_generation() {
        let prices = create_sample_market_data(0.2, 0.1, 0.01, 100);
        assert_eq!(prices.length(), 100);
    }

    #[wasm_bindgen_test]
    fn test_returns_calculation() {
        let prices = vec![100.0, 101.0, 102.5, 101.0, 103.0];
        let returns = calculate_returns_from_prices(&prices);
        assert_eq!(returns.len(), 4);
        assert!((returns[0] - 0.01).abs() < 1e-10); // 1% return
    }
}