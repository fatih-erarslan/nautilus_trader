//! Native Rust bindings for Wolfram MCP Server
//! 
//! Provides high-performance mathematical computations and Wolfram integration
//! via NAPI-RS for Bun.js/Node.js interop.

use napi::bindgen_prelude::*;
use napi_derive::napi;
use num_complex::Complex64;
use std::f64::consts::PI;
use std::process::Command;

/// Wolfram computation result
#[napi(object)]
pub struct WolframResult {
    pub success: bool,
    pub result: String,
    pub error: Option<String>,
    pub computation_time_ms: f64,
}

/// Hyperbolic geometry point in Poincaré disk
#[napi(object)]
pub struct PoincarePoint {
    pub x: f64,
    pub y: f64,
}

/// Execute WolframScript locally with timeout
#[napi]
pub async fn execute_wolfram_script(code: String, timeout_secs: Option<u32>) -> Result<WolframResult> {
    let timeout = timeout_secs.unwrap_or(30);
    let start = std::time::Instant::now();
    
    let output = tokio::task::spawn_blocking(move || {
        Command::new("wolframscript")
            .args(["-code", &code])
            .output()
    })
    .await
    .map_err(|e| Error::from_reason(format!("Task join error: {}", e)))?
    .map_err(|e| Error::from_reason(format!("WolframScript execution failed: {}", e)))?;
    
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    
    if output.status.success() {
        let result = String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter(|line| {
                !line.contains("Loading from Wolfram") &&
                !line.contains("Prefetching") &&
                !line.contains("Connecting")
            })
            .collect::<Vec<_>>()
            .join("\n")
            .trim()
            .to_string();
        
        Ok(WolframResult {
            success: true,
            result,
            error: None,
            computation_time_ms: elapsed,
        })
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        Ok(WolframResult {
            success: false,
            result: String::new(),
            error: Some(stderr),
            computation_time_ms: elapsed,
        })
    }
}

/// Compute hyperbolic distance in Poincaré disk model (native Rust implementation)
#[napi]
pub fn hyperbolic_distance(p1: PoincarePoint, p2: PoincarePoint) -> f64 {
    let z1 = Complex64::new(p1.x, p1.y);
    let z2 = Complex64::new(p2.x, p2.y);
    
    let norm1_sq = z1.norm_sqr();
    let norm2_sq = z2.norm_sqr();
    
    // Check points are inside unit disk
    if norm1_sq >= 1.0 || norm2_sq >= 1.0 {
        return f64::INFINITY;
    }
    
    let diff = z1 - z2;
    let diff_norm_sq = diff.norm_sqr();
    
    // Hyperbolic distance formula: d(z1, z2) = 2 * arctanh(|z1 - z2| / sqrt((1-|z1|²)(1-|z2|²) + |z1-z2|²))
    let denom = ((1.0 - norm1_sq) * (1.0 - norm2_sq) + diff_norm_sq).sqrt();
    let ratio = diff.norm() / denom;
    
    2.0 * ratio.atanh()
}

/// Compute Möbius addition in Poincaré disk
#[napi]
pub fn mobius_add(a: PoincarePoint, b: PoincarePoint) -> PoincarePoint {
    let za = Complex64::new(a.x, a.y);
    let zb = Complex64::new(b.x, b.y);
    
    // Möbius addition: (a + b) / (1 + conj(a) * b)
    let numerator = za + zb;
    let denominator = Complex64::new(1.0, 0.0) + za.conj() * zb;
    let result = numerator / denominator;
    
    PoincarePoint {
        x: result.re,
        y: result.im,
    }
}

/// Compute geodesic path between two points in Poincaré disk
#[napi]
pub fn compute_geodesic(start: PoincarePoint, end: PoincarePoint, num_points: u32) -> Vec<PoincarePoint> {
    let z1 = Complex64::new(start.x, start.y);
    let z2 = Complex64::new(end.x, end.y);
    
    let n = num_points.max(2) as usize;
    let mut points = Vec::with_capacity(n);
    
    // Möbius transformation to move z1 to origin
    let mobius = |z: Complex64, a: Complex64| -> Complex64 {
        (z - a) / (Complex64::new(1.0, 0.0) - a.conj() * z)
    };
    
    let inv_mobius = |z: Complex64, a: Complex64| -> Complex64 {
        (z + a) / (Complex64::new(1.0, 0.0) + a.conj() * z)
    };
    
    // Transform z2 relative to z1 at origin
    let z2_transformed = mobius(z2, z1);
    let angle = z2_transformed.arg();
    let r_max = z2_transformed.norm();
    
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        let r = t * r_max;
        let z_on_geodesic = Complex64::from_polar(r, angle);
        let z_original = inv_mobius(z_on_geodesic, z1);
        
        points.push(PoincarePoint {
            x: z_original.re,
            y: z_original.im,
        });
    }
    
    points
}

/// Compute STDP weight update (exponential kernel)
#[napi]
pub fn stdp_weight_update(
    delta_t: f64,
    tau_plus: f64,
    tau_minus: f64,
    a_plus: f64,
    a_minus: f64,
) -> f64 {
    if delta_t > 0.0 {
        // Post after pre: LTP
        a_plus * (-delta_t / tau_plus).exp()
    } else if delta_t < 0.0 {
        // Pre after post: LTD
        -a_minus * (delta_t / tau_minus).exp()
    } else {
        0.0
    }
}

/// Compute Shannon entropy of a probability distribution
#[napi]
pub fn shannon_entropy(probabilities: Vec<f64>) -> f64 {
    probabilities
        .iter()
        .filter(|&&p| p > 1e-15)
        .map(|&p| -p * p.ln())
        .sum()
}

/// Compute KL divergence between two distributions
#[napi]
pub fn kl_divergence(p: Vec<f64>, q: Vec<f64>) -> f64 {
    if p.len() != q.len() {
        return f64::NAN;
    }
    
    p.iter()
        .zip(q.iter())
        .filter(|(&pi, &qi)| pi > 1e-15 && qi > 1e-15)
        .map(|(&pi, &qi)| pi * (pi / qi).ln())
        .sum()
}

/// Compute softmax of a vector
#[napi]
pub fn softmax(values: Vec<f64>) -> Vec<f64> {
    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_values: Vec<f64> = values.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = exp_values.iter().sum();
    exp_values.iter().map(|&x| x / sum).collect()
}

/// Compute LMSR cost function
#[napi]
pub fn lmsr_cost(quantities: Vec<f64>, b: f64) -> f64 {
    let max_q = quantities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = quantities.iter().map(|&q| ((q - max_q) / b).exp()).sum();
    b * (max_q / b + sum_exp.ln())
}

/// Compute Ising Hamiltonian
#[napi]
pub fn ising_hamiltonian(spins: Vec<i32>, couplings: Vec<f64>, field: f64) -> f64 {
    let n = spins.len();
    let mut energy = 0.0;
    
    // External field term
    for &s in &spins {
        energy -= field * s as f64;
    }
    
    // Coupling term (assuming couplings is flattened upper-triangular)
    let mut idx = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            if idx < couplings.len() {
                energy -= couplings[idx] * (spins[i] * spins[j]) as f64;
                idx += 1;
            }
        }
    }
    
    energy
}

/// Compute Kahneman-Tversky value function
#[napi]
pub fn prospect_value(x: f64, alpha: f64, beta: f64, lambda: f64, reference: f64) -> f64 {
    let deviation = x - reference;
    if deviation >= 0.0 {
        deviation.powf(alpha)
    } else {
        -lambda * (-deviation).powf(beta)
    }
}

/// Compute Landauer bound for bit erasure
#[napi]
pub fn landauer_bound(temperature_kelvin: f64) -> f64 {
    const K_B: f64 = 1.380649e-23; // Boltzmann constant in J/K
    K_B * temperature_kelvin * 2.0_f64.ln()
}

/// Validate that WolframScript is available
#[napi]
pub fn check_wolfram_available() -> bool {
    Command::new("wolframscript")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Get native module version info
#[napi]
pub fn get_native_info() -> String {
    format!(
        "wolfram-native v{} (Rust {} with NAPI-RS)",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::ARCH
    )
}
