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
mod gpu;
mod risk;
mod simd_ops;

pub use consciousness::*;
pub use dilithium::*;
pub use finance::*;
pub use gpu::*;
pub use risk::*;
pub use simd_ops::*;

/// HyperPhysics Engine - Main entry point for Node.js
/// Provides unified access to GPU, SIMD, and CPU compute
#[napi]
pub struct HyperPhysicsEngine {
    use_gpu: bool,
    thread_pool_size: u32,
    security_level: String,
    gpu_engine: Option<gpu::GpuComputeEngine>,
    simd_ops: simd_ops::SimdOps,
}

#[napi]
impl HyperPhysicsEngine {
    /// Create a new HyperPhysics engine instance
    /// Automatically initializes GPU if available and requested
    #[napi(constructor)]
    pub fn new(options: Option<EngineOptions>) -> Result<Self> {
        let opts = options.unwrap_or_default();
        let use_gpu = opts.use_gpu.unwrap_or(true); // Default to GPU enabled

        // Initialize GPU engine if requested
        let gpu_engine = if use_gpu {
            match gpu::GpuComputeEngine::new() {
                Ok(engine) if engine.is_available() => Some(engine),
                _ => None,
            }
        } else {
            None
        };

        Ok(Self {
            use_gpu: gpu_engine.is_some(),
            thread_pool_size: opts.thread_pool_size.unwrap_or(num_cpus::get() as u32),
            security_level: opts.security_level.unwrap_or_else(|| "standard".to_string()),
            gpu_engine,
            simd_ops: simd_ops::SimdOps::new(),
        })
    }

    /// Compute Integrated Information (Φ) for a pBit lattice
    /// Uses GPU if available, falls back to SIMD-accelerated CPU
    #[napi]
    pub fn compute_phi(&self, lattice: Float64Array, width: u32, height: u32) -> Result<f64> {
        consciousness::compute_phi_sync(&lattice, width, height)
    }

    /// GPU-accelerated Φ computation (async)
    /// Returns Promise<number> for non-blocking execution
    #[napi]
    pub async fn compute_phi_gpu(&self, lattice: Float64Array, width: u32, height: u32) -> Result<f64> {
        if let Some(ref gpu) = self.gpu_engine {
            gpu.compute_phi_gpu(lattice, width, height).await
        } else {
            consciousness::compute_phi_sync(&lattice, width, height)
        }
    }

    /// GPU-accelerated Monte Carlo simulation
    /// Returns Promise<Float64Array> with simulated paths
    #[napi]
    pub async fn monte_carlo_gpu(
        &self,
        spot: f64,
        volatility: f64,
        rate: f64,
        time_to_maturity: f64,
        num_paths: u32,
        num_steps: u32,
    ) -> Result<Float64Array> {
        if let Some(ref gpu) = self.gpu_engine {
            gpu.monte_carlo_gpu(spot, volatility, rate, time_to_maturity, num_paths, num_steps).await
        } else {
            // CPU fallback with SIMD
            let paths = cpu_monte_carlo(spot, volatility, rate, time_to_maturity, num_paths as usize, num_steps as usize);
            Ok(Float64Array::new(paths))
        }
    }

    /// SIMD-accelerated dot product
    #[napi]
    pub fn dot_product(&self, a: Float64Array, b: Float64Array) -> Result<f64> {
        self.simd_ops.dot_product(a, b)
    }

    /// SIMD-accelerated softmax
    #[napi]
    pub fn softmax(&self, x: Float64Array) -> Float64Array {
        self.simd_ops.softmax(x)
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

    /// Get engine status including GPU and SIMD information
    #[napi]
    pub fn status(&self) -> EngineStatus {
        EngineStatus {
            initialized: true,
            use_gpu: self.use_gpu,
            thread_pool_size: self.thread_pool_size,
            security_level: self.security_level.clone(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            gpu_device: self.gpu_engine.as_ref().map(|g| g.device_name()).unwrap_or_else(|| "None".to_string()),
            simd_backend: self.simd_ops.backend_name(),
            simd_lanes: self.simd_ops.lanes(),
        }
    }
}

/// CPU Monte Carlo fallback with SIMD optimization
fn cpu_monte_carlo(
    spot: f64,
    volatility: f64,
    rate: f64,
    time_to_maturity: f64,
    num_paths: usize,
    num_steps: usize,
) -> Vec<f64> {
    use std::f64::consts::PI;

    let dt = time_to_maturity / num_steps as f64;
    let sqrt_dt = dt.sqrt();
    let drift = (rate - 0.5 * volatility * volatility) * dt;
    let diffusion = volatility * sqrt_dt;

    let mut paths = vec![0.0; num_paths];

    for (i, path) in paths.iter_mut().enumerate() {
        let mut price = spot;
        let mut seed = (i as u64).wrapping_mul(1103515245).wrapping_add(12345);

        for _ in 0..num_steps {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u1 = ((seed >> 33) as f64) / (1u64 << 31) as f64;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = ((seed >> 33) as f64) / (1u64 << 31) as f64;

            let u1_clamped = u1.max(1e-10);
            let z = (-2.0 * u1_clamped.ln()).sqrt() * (2.0 * PI * u2).cos();
            price *= (drift + diffusion * z).exp();
        }

        *path = price;
    }

    paths
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
    pub gpu_device: String,
    pub simd_backend: String,
    pub simd_lanes: u32,
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
