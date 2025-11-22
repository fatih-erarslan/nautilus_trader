//! Consciousness module - Zero-copy Φ computation

use napi::bindgen_prelude::*;
use napi_derive::napi;

/// Compute Φ synchronously
pub fn compute_phi_sync(lattice: &Float64Array, width: u32, height: u32) -> Result<f64> {
    let data = lattice.as_ref();
    let expected_size = (width * height) as usize;

    if data.len() != expected_size {
        return Err(Error::new(
            Status::InvalidArg,
            format!("Lattice size mismatch: expected {}, got {}", expected_size, data.len()),
        ));
    }

    Ok(compute_phi_iit(data, width as usize, height as usize))
}

/// IIT 3.0 Φ computation (simplified)
fn compute_phi_iit(lattice: &[f64], width: usize, height: usize) -> f64 {
    let h_full = compute_entropy(lattice);
    if h_full < 1e-10 {
        return 0.0;
    }

    let mut min_phi = f64::MAX;

    // Horizontal partition
    if height > 1 {
        let mid = height / 2;
        let top: Vec<f64> = lattice.iter().take(width * mid).copied().collect();
        let bottom: Vec<f64> = lattice.iter().skip(width * mid).copied().collect();
        let phi_h = h_full - f64::max(compute_entropy(&top), compute_entropy(&bottom));
        min_phi = min_phi.min(phi_h.max(0.0));
    }

    // Vertical partition
    if width > 1 {
        let mid = width / 2;
        let mut left = Vec::with_capacity(mid * height);
        let mut right = Vec::with_capacity((width - mid) * height);
        for row in 0..height {
            for col in 0..width {
                if col < mid {
                    left.push(lattice[row * width + col]);
                } else {
                    right.push(lattice[row * width + col]);
                }
            }
        }
        let phi_v = h_full - f64::max(compute_entropy(&left), compute_entropy(&right));
        min_phi = min_phi.min(phi_v.max(0.0));
    }

    if min_phi == f64::MAX { 0.0 } else { min_phi }
}

fn compute_entropy(probs: &[f64]) -> f64 {
    probs
        .iter()
        .filter(|&&p| p > 0.0 && p < 1.0)
        .map(|&p| -p * p.ln() - (1.0 - p) * (1.0 - p).ln())
        .sum::<f64>()
        / probs.len().max(1) as f64
}

/// Φ Computer class for Node.js
#[napi]
pub struct PhiComputer {
    width: u32,
    height: u32,
}

#[napi]
impl PhiComputer {
    #[napi(constructor)]
    pub fn new(width: u32, height: u32) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::new(Status::InvalidArg, "Width and height must be > 0"));
        }
        Ok(Self { width, height })
    }

    #[napi]
    pub fn compute(&self, lattice: Float64Array) -> Result<f64> {
        compute_phi_sync(&lattice, self.width, self.height)
    }

    #[napi(getter)]
    pub fn size(&self) -> u32 {
        self.width * self.height
    }
}
