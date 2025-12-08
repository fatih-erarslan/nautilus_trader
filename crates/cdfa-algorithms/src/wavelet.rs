use ndarray::{Array1, Array2, ArrayView1, s};

/// Discrete Wavelet Transform implementation
/// 
/// Basic implementation of DWT using Haar wavelets
pub struct WaveletTransform;

impl WaveletTransform {
    /// Perform 1D Discrete Wavelet Transform using Haar wavelet
    pub fn dwt_haar(signal: &ArrayView1<f64>) -> Result<(Array1<f64>, Array1<f64>), &'static str> {
        let n = signal.len();
        
        if n < 2 {
            return Err("Signal must have at least 2 samples");
        }
        
        if n % 2 != 0 {
            return Err("Signal length must be even for Haar wavelet");
        }
        
        let half_n = n / 2;
        let mut approximation = Array1::zeros(half_n);
        let mut detail = Array1::zeros(half_n);
        
        let sqrt2 = 2.0_f64.sqrt();
        
        for i in 0..half_n {
            approximation[i] = (signal[2*i] + signal[2*i + 1]) / sqrt2;
            detail[i] = (signal[2*i] - signal[2*i + 1]) / sqrt2;
        }
        
        Ok((approximation, detail))
    }
    
    /// Perform inverse DWT using Haar wavelet
    pub fn idwt_haar(
        approximation: &ArrayView1<f64>, 
        detail: &ArrayView1<f64>
    ) -> Result<Array1<f64>, &'static str> {
        if approximation.len() != detail.len() {
            return Err("Approximation and detail must have same length");
        }
        
        let n = approximation.len();
        let mut reconstructed = Array1::zeros(n * 2);
        
        let sqrt2 = 2.0_f64.sqrt();
        
        for i in 0..n {
            reconstructed[2*i] = (approximation[i] + detail[i]) / sqrt2;
            reconstructed[2*i + 1] = (approximation[i] - detail[i]) / sqrt2;
        }
        
        Ok(reconstructed)
    }
    
    /// Multi-level DWT decomposition
    pub fn wavedec(signal: &ArrayView1<f64>, level: usize) -> Result<Vec<Array1<f64>>, &'static str> {
        if level == 0 {
            return Err("Level must be at least 1");
        }
        
        let mut coefficients = Vec::new();
        let mut current_signal = signal.to_owned();
        
        for _ in 0..level {
            if current_signal.len() < 2 {
                break;
            }
            
            let (approx, detail) = Self::dwt_haar(&current_signal.view())?;
            coefficients.push(detail);
            current_signal = approx;
        }
        
        // Add final approximation
        coefficients.push(current_signal);
        coefficients.reverse(); // Put approximation first
        
        Ok(coefficients)
    }
    
    /// Multi-level inverse DWT reconstruction
    pub fn waverec(coefficients: &[Array1<f64>]) -> Result<Array1<f64>, &'static str> {
        if coefficients.is_empty() {
            return Err("Coefficients cannot be empty");
        }
        
        let mut reconstructed = coefficients[0].clone();
        
        for i in 1..coefficients.len() {
            reconstructed = Self::idwt_haar(&reconstructed.view(), &coefficients[i].view())?;
        }
        
        Ok(reconstructed)
    }
    
    /// Daubechies D4 wavelet transform (more sophisticated than Haar)
    pub fn dwt_db4(signal: &ArrayView1<f64>) -> Result<(Array1<f64>, Array1<f64>), &'static str> {
        let n = signal.len();
        
        if n < 4 {
            return Err("Signal must have at least 4 samples for DB4");
        }
        
        // Daubechies D4 coefficients
        let h0 = (1.0 + 3.0_f64.sqrt()) / (4.0 * 2.0_f64.sqrt());
        let h1 = (3.0 + 3.0_f64.sqrt()) / (4.0 * 2.0_f64.sqrt());
        let h2 = (3.0 - 3.0_f64.sqrt()) / (4.0 * 2.0_f64.sqrt());
        let h3 = (1.0 - 3.0_f64.sqrt()) / (4.0 * 2.0_f64.sqrt());
        
        let g0 = h3;
        let g1 = -h2;
        let g2 = h1;
        let g3 = -h0;
        
        let half_n = n / 2;
        let mut approximation = Array1::zeros(half_n);
        let mut detail = Array1::zeros(half_n);
        
        // Extend signal for boundary handling (periodic extension)
        let mut extended = Array1::zeros(n + 3);
        extended.slice_mut(s![..n]).assign(signal);
        extended[n] = signal[0];
        extended[n + 1] = signal[1];
        extended[n + 2] = signal[2];
        
        for i in 0..half_n {
            let idx = 2 * i;
            approximation[i] = h0 * extended[idx] + h1 * extended[idx + 1] 
                            + h2 * extended[idx + 2] + h3 * extended[(idx + 3) % (n + 3)];
            detail[i] = g0 * extended[idx] + g1 * extended[idx + 1] 
                     + g2 * extended[idx + 2] + g3 * extended[(idx + 3) % (n + 3)];
        }
        
        Ok((approximation, detail))
    }
    
    /// Continuous Wavelet Transform using Morlet wavelet
    pub fn cwt_morlet(
        signal: &ArrayView1<f64>, 
        scales: &ArrayView1<f64>, 
        omega0: f64
    ) -> Result<Array2<f64>, &'static str> {
        let n = signal.len();
        let n_scales = scales.len();
        
        if n == 0 || n_scales == 0 {
            return Err("Signal and scales cannot be empty");
        }
        
        let mut cwt_matrix = Array2::zeros((n_scales, n));
        
        for (i, &scale) in scales.iter().enumerate() {
            for j in 0..n {
                let mut coeff = 0.0;
                
                for k in 0..n {
                    let t = (k as f64 - j as f64) / scale;
                    let wavelet = Self::morlet_wavelet(t, omega0) / scale.sqrt();
                    coeff += signal[k] * wavelet;
                }
                
                cwt_matrix[[i, j]] = coeff;
            }
        }
        
        Ok(cwt_matrix)
    }
    
    /// Morlet wavelet function
    fn morlet_wavelet(t: f64, omega0: f64) -> f64 {
        let pi_pow = std::f64::consts::PI.powf(-0.25);
        let exp_term = (-t * t / 2.0).exp();
        let cos_term = (omega0 * t).cos();
        pi_pow * exp_term * cos_term
    }
    
    /// Wavelet denoising using soft thresholding
    pub fn denoise(
        signal: &ArrayView1<f64>, 
        level: usize, 
        threshold: Option<f64>
    ) -> Result<Array1<f64>, &'static str> {
        // Decompose signal
        let coeffs = Self::wavedec(signal, level)?;
        
        // Calculate threshold if not provided (universal threshold)
        let thresh = threshold.unwrap_or_else(|| {
            let sigma = signal.std(0.0);
            sigma * (2.0 * (signal.len() as f64).ln()).sqrt()
        });
        
        // Apply soft thresholding to detail coefficients
        let mut thresholded_coeffs = coeffs.clone();
        for i in 1..thresholded_coeffs.len() {
            for j in 0..thresholded_coeffs[i].len() {
                let val = thresholded_coeffs[i][j];
                thresholded_coeffs[i][j] = if val.abs() <= thresh {
                    0.0
                } else if val > 0.0 {
                    val - thresh
                } else {
                    val + thresh
                };
            }
        }
        
        // Reconstruct
        Self::waverec(&thresholded_coeffs)
    }
    
    /// Extract wavelet energy features
    pub fn wavelet_energy(coefficients: &[Array1<f64>]) -> Vec<f64> {
        coefficients.iter()
            .map(|coeff| coeff.iter().map(|&x| x * x).sum::<f64>())
            .collect()
    }
}

/// Wavelet packet decomposition for more detailed analysis
pub struct WaveletPacket {
    pub level: usize,
    pub nodes: Vec<Array1<f64>>,
}

impl WaveletPacket {
    /// Create wavelet packet decomposition
    pub fn new(signal: &ArrayView1<f64>, level: usize) -> Result<Self, &'static str> {
        if level == 0 {
            return Err("Level must be at least 1");
        }
        
        let mut nodes = vec![signal.to_owned()];
        
        for current_level in 0..level {
            let n_nodes = 2_usize.pow(current_level as u32);
            let mut new_nodes = Vec::new();
            
            for i in 0..n_nodes {
                if i < nodes.len() {
                    let (approx, detail) = WaveletTransform::dwt_haar(&nodes[i].view())?;
                    new_nodes.push(approx);
                    new_nodes.push(detail);
                }
            }
            
            nodes = new_nodes;
        }
        
        Ok(Self { level, nodes })
    }
    
    /// Get best basis using entropy criterion
    pub fn best_basis(&self, entropy_type: &str) -> Result<Vec<usize>, &'static str> {
        let entropy_fn = match entropy_type {
            "shannon" => |x: &Array1<f64>| {
                x.iter()
                    .filter(|&&v| v.abs() > 1e-10)
                    .map(|&v| -v * v * (v * v).ln())
                    .sum::<f64>()
            },
            "log" => |x: &Array1<f64>| {
                x.iter()
                    .filter(|&&v| v.abs() > 1e-10)
                    .map(|&v| (v * v).ln())
                    .sum::<f64>()
            },
            _ => return Err("Unknown entropy type"),
        };
        
        // Simple best basis selection (can be improved)
        let entropies: Vec<f64> = self.nodes.iter()
            .map(|node| entropy_fn(node))
            .collect();
        
        // Select nodes with lowest entropy
        let mut indices: Vec<usize> = (0..self.nodes.len()).collect();
        indices.sort_by(|&a, &b| entropies[a].partial_cmp(&entropies[b]).unwrap());
        
        Ok(indices[..indices.len()/2].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_dwt_haar() {
        let signal = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (approx, detail) = WaveletTransform::dwt_haar(&signal.view()).unwrap();
        
        assert_eq!(approx.len(), 4);
        assert_eq!(detail.len(), 4);
        
        // Test perfect reconstruction
        let reconstructed = WaveletTransform::idwt_haar(&approx.view(), &detail.view()).unwrap();
        for i in 0..signal.len() {
            assert!((reconstructed[i] - signal[i]).abs() < 1e-10);
        }
    }
    
    #[test]
    fn test_multilevel_decomposition() {
        let signal = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let coeffs = WaveletTransform::wavedec(&signal.view(), 3).unwrap();
        
        assert_eq!(coeffs.len(), 4); // 1 approx + 3 details
        
        // Test reconstruction
        let reconstructed = WaveletTransform::waverec(&coeffs).unwrap();
        for i in 0..signal.len() {
            assert!((reconstructed[i] - signal[i]).abs() < 1e-10);
        }
    }
    
    #[test]
    fn test_wavelet_denoising() {
        let mut signal = Array1::linspace(0.0, 10.0, 64);
        // Add noise
        for i in 0..signal.len() {
            signal[i] += 0.1 * ((i as f64).sin());
        }
        
        let denoised = WaveletTransform::denoise(&signal.view(), 3, Some(0.1)).unwrap();
        assert_eq!(denoised.len(), signal.len());
    }
    
    #[test]
    fn test_cwt_morlet() {
        let signal = array![
            0.0, 0.707, 1.0, 0.707, 0.0, -0.707, -1.0, -0.707,
            0.0, 0.707, 1.0, 0.707, 0.0, -0.707, -1.0, -0.707
        ];
        let scales = array![1.0, 2.0, 4.0, 8.0];
        
        let cwt = WaveletTransform::cwt_morlet(&signal.view(), &scales.view(), 5.0).unwrap();
        assert_eq!(cwt.dim(), (4, 16));
    }
    
    #[test]
    fn test_wavelet_packet() {
        let signal = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let wp = WaveletPacket::new(&signal.view(), 2).unwrap();
        
        assert_eq!(wp.nodes.len(), 4); // 2^2 nodes at level 2
    }
}