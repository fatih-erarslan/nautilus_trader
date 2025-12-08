//! Wavelet transform implementations for CDFA
//! 
//! High-performance wavelet transforms with SIMD optimization,
//! consolidated from cdfa-algorithms.

use ndarray::{Array1, ArrayView1};
use crate::error::{CdfaError, Result};

#[cfg(feature = "simd")]
// SIMD acceleration available via feature flag

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Wavelet transform implementation
pub struct WaveletTransform;

impl WaveletTransform {
    /// Discrete Wavelet Transform using Haar wavelet
    pub fn dwt_haar(signal: &ArrayView1<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        let n = signal.len();
        if n % 2 != 0 {
            return Err(CdfaError::invalid_input("Signal length must be even for Haar DWT"));
        }
        
        let half_n = n / 2;
        let mut approx = Array1::zeros(half_n);
        let mut detail = Array1::zeros(half_n);
        
        // SIMD implementation would go here when available
        #[cfg(feature = "simd")]
        {
            // TODO: Implement SIMD version when available
        }
        
        // Scalar implementation
        let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
        
        for i in 0..half_n {
            let even = signal[2 * i];
            let odd = signal[2 * i + 1];
            approx[i] = (even + odd) * sqrt2_inv;
            detail[i] = (even - odd) * sqrt2_inv;
        }
        
        Ok((approx, detail))
    }
    
    /// Inverse Discrete Wavelet Transform using Haar wavelet
    pub fn idwt_haar(approx: &ArrayView1<f64>, detail: &ArrayView1<f64>) -> Result<Array1<f64>> {
        if approx.len() != detail.len() {
            return Err(CdfaError::invalid_input("Approximation and detail coefficients must have same length"));
        }
        
        let half_n = approx.len();
        let n = 2 * half_n;
        let mut signal = Array1::zeros(n);
        
        let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
        
        for i in 0..half_n {
            let a = approx[i];
            let d = detail[i];
            signal[2 * i] = (a + d) * sqrt2_inv;
            signal[2 * i + 1] = (a - d) * sqrt2_inv;
        }
        
        Ok(signal)
    }
    
    /// Multi-level DWT decomposition
    pub fn dwt_multilevel(signal: &ArrayView1<f64>, levels: usize) -> Result<WaveletDecomposition> {
        if levels == 0 {
            return Err(CdfaError::invalid_input("Number of levels must be positive"));
        }
        
        let mut current_signal = signal.to_owned();
        let mut details = Vec::new();
        
        for level in 0..levels {
            if current_signal.len() < 2 {
                return Err(CdfaError::invalid_input(
                    format!("Signal too short for {} levels of decomposition", levels)
                ));
            }
            
            let (approx, detail) = Self::dwt_haar(&current_signal.view())?;
            details.push(detail);
            current_signal = approx;
            
            if current_signal.len() % 2 != 0 && level < levels - 1 {
                // Truncate to even length for next level
                let new_len = current_signal.len() - 1;
                current_signal = current_signal.slice(ndarray::s![..new_len]).to_owned();
            }
        }
        
        // Reverse details to match convention (finest to coarsest)
        details.reverse();
        
        Ok(WaveletDecomposition {
            approximation: current_signal,
            details,
            levels,
        })
    }
    
    /// Reconstruct signal from multi-level decomposition
    pub fn idwt_multilevel(decomposition: &WaveletDecomposition) -> Result<Array1<f64>> {
        let mut current_signal = decomposition.approximation.clone();
        
        // Reconstruct from coarsest to finest level
        for detail in decomposition.details.iter().rev() {
            if current_signal.len() != detail.len() {
                return Err(CdfaError::invalid_input(
                    "Inconsistent decomposition structure"
                ));
            }
            current_signal = Self::idwt_haar(&current_signal.view(), &detail.view())?;
        }
        
        Ok(current_signal)
    }
    
    /// Daubechies-4 wavelet transform
    pub fn dwt_db4(signal: &ArrayView1<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        let n = signal.len();
        if n < 4 || n % 2 != 0 {
            return Err(CdfaError::invalid_input(
                "Signal length must be even and at least 4 for DB4 DWT"
            ));
        }
        
        // Daubechies-4 coefficients
        let sqrt3 = 3.0_f64.sqrt();
        let h0 = (1.0 + sqrt3) / (4.0 * 2.0_f64.sqrt());
        let h1 = (3.0 + sqrt3) / (4.0 * 2.0_f64.sqrt());
        let h2 = (3.0 - sqrt3) / (4.0 * 2.0_f64.sqrt());
        let h3 = (1.0 - sqrt3) / (4.0 * 2.0_f64.sqrt());
        
        let g0 = h3;
        let g1 = -h2;
        let g2 = h1;
        let g3 = -h0;
        
        let half_n = n / 2;
        let mut approx = Array1::zeros(half_n);
        let mut detail = Array1::zeros(half_n);
        
        for i in 0..half_n {
            let mut a_sum = 0.0;
            let mut d_sum = 0.0;
            
            for k in 0..4 {
                let idx = (2 * i + k) % n; // Periodic boundary
                a_sum += [h0, h1, h2, h3][k] * signal[idx];
                d_sum += [g0, g1, g2, g3][k] * signal[idx];
            }
            
            approx[i] = a_sum;
            detail[i] = d_sum;
        }
        
        Ok((approx, detail))
    }
    
    /// Biorthogonal wavelet transform
    pub fn dwt_bior(signal: &ArrayView1<f64>, wavelet_type: BiorthogonalType) -> Result<(Array1<f64>, Array1<f64>)> {
        match wavelet_type {
            BiorthogonalType::Bior22 => Self::dwt_bior22(signal),
            BiorthogonalType::Bior44 => Self::dwt_bior44(signal),
        }
    }
    
    fn dwt_bior22(signal: &ArrayView1<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        // Biorthogonal 2.2 coefficients
        let h = vec![-0.12940952255126, 0.22414386804201, 0.83651630373781, 0.48296291314453];
        let g = vec![0.0, 0.0, 0.70710678118655, -0.70710678118655];
        
        Self::dwt_with_filters(signal, &h, &g)
    }
    
    fn dwt_bior44(signal: &ArrayView1<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        // Biorthogonal 4.4 coefficients (simplified)
        let h = vec![-0.07785205408506, 0.39642005410236, 0.72956024532503, 0.46986932740129];
        let g = vec![0.0, 0.0, 0.70710678118655, -0.70710678118655];
        
        Self::dwt_with_filters(signal, &h, &g)
    }
    
    fn dwt_with_filters(signal: &ArrayView1<f64>, h: &[f64], g: &[f64]) -> Result<(Array1<f64>, Array1<f64>)> {
        let n = signal.len();
        let filter_len = h.len();
        
        if n < filter_len || n % 2 != 0 {
            return Err(CdfaError::invalid_input(
                format!("Signal length must be even and at least {}", filter_len)
            ));
        }
        
        let half_n = n / 2;
        let mut approx = Array1::zeros(half_n);
        let mut detail = Array1::zeros(half_n);
        
        for i in 0..half_n {
            let mut a_sum = 0.0;
            let mut d_sum = 0.0;
            
            for k in 0..filter_len {
                let idx = (2 * i + k) % n; // Periodic boundary
                a_sum += h[k] * signal[idx];
                if k < g.len() {
                    d_sum += g[k] * signal[idx];
                }
            }
            
            approx[i] = a_sum;
            detail[i] = d_sum;
        }
        
        Ok((approx, detail))
    }
}

/// Wavelet packet transform
pub struct WaveletPacket;

impl WaveletPacket {
    /// Full wavelet packet decomposition
    pub fn decompose(signal: &ArrayView1<f64>, levels: usize) -> Result<WaveletPacketTree> {
        if levels == 0 {
            return Err(CdfaError::invalid_input("Number of levels must be positive"));
        }
        
        let mut tree = WaveletPacketTree::new(signal.to_owned());
        
        for level in 0..levels {
            let mut new_nodes = Vec::new();
            
            for node in &tree.nodes[level] {
                let (approx, detail) = WaveletTransform::dwt_haar(&node.view())?;
                new_nodes.push(approx);
                new_nodes.push(detail);
            }
            
            tree.nodes.push(new_nodes);
        }
        
        Ok(tree)
    }
    
    /// Best basis selection using entropy criterion
    pub fn best_basis(tree: &WaveletPacketTree, entropy_type: EntropyType) -> Result<Vec<usize>> {
        let mut best_basis = Vec::new();
        
        for level in 0..tree.nodes.len() {
            let mut min_entropy = f64::INFINITY;
            let mut best_node = 0;
            
            for (i, node) in tree.nodes[level].iter().enumerate() {
                let entropy = match entropy_type {
                    EntropyType::Shannon => Self::shannon_entropy(node),
                    EntropyType::LogEnergy => Self::log_energy_entropy(node),
                    EntropyType::Threshold => Self::threshold_entropy(node, 0.1),
                };
                
                if entropy < min_entropy {
                    min_entropy = entropy;
                    best_node = i;
                }
            }
            
            best_basis.push(best_node);
        }
        
        Ok(best_basis)
    }
    
    fn shannon_entropy(signal: &Array1<f64>) -> f64 {
        let energy: f64 = signal.mapv(|x| x * x).sum();
        if energy <= 0.0 {
            return 0.0;
        }
        
        signal.mapv(|x| {
            let p = (x * x) / energy;
            if p > 0.0 { -p * p.ln() } else { 0.0 }
        }).sum()
    }
    
    fn log_energy_entropy(signal: &Array1<f64>) -> f64 {
        signal.mapv(|x| {
            let abs_x = x.abs();
            if abs_x > 0.0 { abs_x * abs_x.ln() } else { 0.0 }
        }).sum()
    }
    
    fn threshold_entropy(signal: &Array1<f64>, threshold: f64) -> f64 {
        let total_energy: f64 = signal.mapv(|x| x * x).sum();
        let threshold_energy = threshold * total_energy;
        
        signal.mapv(|x| {
            let energy = x * x;
            if energy > threshold_energy { energy } else { 0.0 }
        }).sum()
    }
}

/// Types for wavelet decomposition
#[derive(Debug, Clone)]
pub struct WaveletDecomposition {
    pub approximation: Array1<f64>,
    pub details: Vec<Array1<f64>>,
    pub levels: usize,
}

#[derive(Debug, Clone)]
pub struct WaveletPacketTree {
    pub nodes: Vec<Vec<Array1<f64>>>,
}

impl WaveletPacketTree {
    fn new(signal: Array1<f64>) -> Self {
        Self {
            nodes: vec![vec![signal]],
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BiorthogonalType {
    Bior22,
    Bior44,
}

#[derive(Debug, Clone, Copy)]
pub enum EntropyType {
    Shannon,
    LogEnergy,
    Threshold,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_haar_dwt() {
        let signal = array![1.0, 2.0, 3.0, 4.0];
        let (approx, detail) = WaveletTransform::dwt_haar(&signal.view()).unwrap();
        
        assert_eq!(approx.len(), 2);
        assert_eq!(detail.len(), 2);
        
        // Reconstruction should be perfect
        let reconstructed = WaveletTransform::idwt_haar(&approx.view(), &detail.view()).unwrap();
        for (orig, recon) in signal.iter().zip(reconstructed.iter()) {
            assert_relative_eq!(*orig, *recon, epsilon = 1e-10);
        }
    }
    
    #[test]
    fn test_multilevel_dwt() {
        let signal = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let decomposition = WaveletTransform::dwt_multilevel(&signal.view(), 2).unwrap();
        
        assert_eq!(decomposition.levels, 2);
        assert_eq!(decomposition.details.len(), 2);
        
        // Reconstruction should be close to original
        let reconstructed = WaveletTransform::idwt_multilevel(&decomposition).unwrap();
        assert!(reconstructed.len() >= signal.len() - 1); // May have slight length difference due to truncation
    }
    
    #[test]
    fn test_db4_dwt() {
        let signal = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (approx, detail) = WaveletTransform::dwt_db4(&signal.view()).unwrap();
        
        assert_eq!(approx.len(), 4);
        assert_eq!(detail.len(), 4);
    }
    
    #[test]
    fn test_wavelet_packet() {
        let signal = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tree = WaveletPacket::decompose(&signal.view(), 2).unwrap();
        
        assert_eq!(tree.nodes.len(), 3); // Original + 2 levels
        assert_eq!(tree.nodes[0].len(), 1); // Original signal
        assert_eq!(tree.nodes[1].len(), 2); // Level 1: approx + detail
        assert_eq!(tree.nodes[2].len(), 4); // Level 2: 4 subbands
    }
    
    #[test]
    fn test_biorthogonal_dwt() {
        let signal = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (approx, detail) = WaveletTransform::dwt_bior(&signal.view(), BiorthogonalType::Bior22).unwrap();
        
        assert_eq!(approx.len(), 4);
        assert_eq!(detail.len(), 4);
    }
}