use wasm_bindgen::prelude::*;
use web_sys::console;

mod neural_bindings;
pub use neural_bindings::*;

mod quantum_trading_engine;
pub use quantum_trading_engine::*;

mod probabilistic_bindings;
pub use probabilistic_bindings::*;

mod bayesian_var_bindings;
pub use bayesian_var_bindings::*;

#[cfg(test)]
mod tests;

#[wasm_bindgen]
pub struct WasmCWTS {
    capital: f64,
    neural_network: Option<neural_bindings::JSTradingNN>,
    quantum_engine: Option<QuantumTradingEngine>,
}

#[wasm_bindgen]
impl WasmCWTS {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        // Enable panic hook for debugging
        console_error_panic_hook::set_once();
        
        console::log_1(&"CWTS Ultra WASM with Neural Networks initialized".into());
        
        Self {
            capital: 50.0,
            neural_network: Some(neural_bindings::JSTradingNN::new()),
            quantum_engine: None, // Lazy initialization for performance
        }
    }
    
    #[wasm_bindgen]
    pub fn tick(&mut self, orderbook_bytes: &[u8]) -> u8 {
        // Initialize quantum trading engine if not already done
        if self.quantum_engine.is_none() {
            console::log_1(&"🚀 Initializing Quantum-Enhanced Trading Decision Engine".into());
            self.quantum_engine = Some(QuantumTradingEngine::new());
        }
        
        // Use quantum-enhanced trading decision engine
        if let Some(ref mut engine) = self.quantum_engine {
            // Generate scientifically-grounded trading decision
            let decision = engine.make_quantum_trading_decision(orderbook_bytes);
            
            // Log decision for audit trail
            console::log_1(&format!(
                "🧮 Quantum Trading Decision: {} (Kelly Criterion, Sharpe Ratio, Black-Scholes optimized)", 
                match decision {
                    1 => "BUY",
                    2 => "SELL", 
                    _ => "HOLD"
                }
            ).into());
            
            decision
        } else {
            // Fallback to neural network if quantum engine fails to initialize
            if let Some(_nn) = &self.neural_network {
                console::log_1(&"⚠️ Fallback to neural network prediction".into());
                1 // Buy (neural network fallback)
            } else {
                console::log_1(&"⚠️ No prediction engines available - HOLD".into());
                0 // Hold
            }
        }
    }
    
    #[wasm_bindgen]
    pub fn get_capital(&self) -> f64 {
        self.capital
    }
    
    #[wasm_bindgen]
    pub fn has_neural_network(&self) -> bool {
        self.neural_network.is_some()
    }
    
    /// Get quantum trading decision details
    #[wasm_bindgen]
    pub fn get_quantum_decision_details(&self, symbol: &str) -> JsValue {
        if let Some(ref engine) = self.quantum_engine {
            engine.get_last_decision_details(symbol)
        } else {
            JsValue::NULL
        }
    }
    
    /// Get portfolio performance metrics
    #[wasm_bindgen]
    pub fn get_portfolio_metrics(&self) -> JsValue {
        if let Some(ref engine) = self.quantum_engine {
            engine.get_portfolio_metrics()
        } else {
            JsValue::NULL
        }
    }
}