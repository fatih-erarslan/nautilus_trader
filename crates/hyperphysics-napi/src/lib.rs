//! # HyperPhysics NAPI - Zero-Overhead Node.js Bindings
//!
//! Zero-overhead Node.js bindings for HyperPhysics using NAPI-RS.
//! Provides direct V8 memory access with TypedArray zero-copy.

#![deny(clippy::all)]

use napi::bindgen_prelude::*;
use napi_derive::napi;

mod consciousness;
mod dilithium;
mod finance;
mod risk;

pub use consciousness::*;
pub use dilithium::*;
pub use finance::*;
pub use risk::*;

/// HyperPhysics Engine - Main entry point for Node.js
#[napi]
pub struct HyperPhysicsEngine {
    use_gpu: bool,
    thread_pool_size: u32,
    security_level: String,
}

#[napi]
impl HyperPhysicsEngine {
    /// Create a new HyperPhysics engine instance
    #[napi(constructor)]
    pub fn new(options: Option<EngineOptions>) -> Result<Self> {
        let opts = options.unwrap_or_default();
        Ok(Self {
            use_gpu: opts.use_gpu.unwrap_or(false),
            thread_pool_size: opts.thread_pool_size.unwrap_or(num_cpus::get() as u32),
            security_level: opts.security_level.unwrap_or_else(|| "standard".to_string()),
        })
    }

    /// Compute Integrated Information (Φ) for a pBit lattice
    #[napi]
    pub fn compute_phi(&self, lattice: Float64Array, width: u32, height: u32) -> Result<f64> {
        consciousness::compute_phi_sync(&lattice, width, height)
    }

    /// Sign a message using Dilithium post-quantum signatures
    #[napi]
    pub fn sign_message(&self, message: Buffer, security_level: Option<String>) -> Result<SignatureResult> {
        let level = security_level.unwrap_or_else(|| self.security_level.clone());
        dilithium::sign_message_sync(&message, &level)
    }

    /// Calculate risk metrics for a return series
    #[napi]
    pub fn calculate_risk(&self, returns: Float64Array, confidence: Option<f64>) -> Result<RiskMetricsResult> {
        let conf = confidence.unwrap_or(0.95);
        risk::calculate_risk_sync(&returns, conf)
    }

    /// Calculate Black-Scholes option price and Greeks
    #[napi]
    pub fn calculate_option_price(
        &self,
        spot: f64,
        strike: f64,
        rate: f64,
        volatility: f64,
        time_to_maturity: f64,
    ) -> Result<OptionPriceResult> {
        finance::calculate_option_price_internal(spot, strike, rate, volatility, time_to_maturity)
    }

    /// Get engine status
    #[napi]
    pub fn status(&self) -> EngineStatus {
        EngineStatus {
            initialized: true,
            use_gpu: self.use_gpu,
            thread_pool_size: self.thread_pool_size,
            security_level: self.security_level.clone(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// Engine configuration options
#[napi(object)]
#[derive(Default)]
pub struct EngineOptions {
    pub use_gpu: Option<bool>,
    pub thread_pool_size: Option<u32>,
    pub security_level: Option<String>,
}

/// Engine status information
#[napi(object)]
pub struct EngineStatus {
    pub initialized: bool,
    pub use_gpu: bool,
    pub thread_pool_size: u32,
    pub security_level: String,
    pub version: String,
}

/// Signature result from sign_message
#[napi(object)]
pub struct SignatureResult {
    pub signature: Buffer,
    pub public_key: Buffer,
    pub security_level: String,
    pub size: u32,
}

/// Risk metrics result
#[napi(object)]
pub struct RiskMetricsResult {
    pub var_95: f64,
    pub var_99: f64,
    pub expected_shortfall: f64,
    pub volatility: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
}

/// Option pricing result
#[napi(object)]
pub struct OptionPriceResult {
    pub call_price: f64,
    pub put_price: f64,
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_compilation() {
        assert!(true);
    }
}
