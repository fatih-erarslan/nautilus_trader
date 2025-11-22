// Multi-scale time series decomposition for NHITS

use std::collections::VecDeque;

pub struct MultiScaleDecomposer {
    method: DecompositionMethod,
    scales: Vec<TimeScale>,
    filters: Vec<DecompositionFilter>,
}

#[derive(Clone, Copy)]
pub enum DecompositionMethod {
    Wavelet,
    EMD,       // Empirical Mode Decomposition
    STL,       // Seasonal-Trend decomposition using Loess
    SSA,       // Singular Spectrum Analysis
    VMD,       // Variational Mode Decomposition
    CEEMDAN,   // Complete Ensemble EMD with Adaptive Noise
}

pub struct TimeScale {
    pub scale_factor: usize,
    pub window_size: usize,
    pub stride: usize,
}

pub struct DecomposedSeries {
    pub trend: Vec<f32>,
    pub seasonal: Vec<f32>,
    pub residual: Vec<f32>,
    pub scales: Vec<ScaleComponent>,
}

pub struct ScaleComponent {
    pub scale: usize,
    pub component: Vec<f32>,
    pub frequency: f32,
    pub amplitude: f32,
}

struct DecompositionFilter {
    filter_type: FilterType,
    cutoff_frequency: f32,
    order: usize,
}

#[derive(Clone, Copy)]
enum FilterType {
    LowPass,
    HighPass,
    BandPass,
    BandStop,
}

impl MultiScaleDecomposer {
    pub fn new(method: DecompositionMethod) -> Self {
        let scales = vec![
            TimeScale { scale_factor: 1, window_size: 24, stride: 1 },   // Hourly
            TimeScale { scale_factor: 24, window_size: 7, stride: 1 },   // Daily
            TimeScale { scale_factor: 168, window_size: 4, stride: 1 },  // Weekly
        ];
        
        let filters = vec![
            DecompositionFilter::low_pass(0.1, 4),
            DecompositionFilter::band_pass(0.1, 0.4, 4),
            DecompositionFilter::high_pass(0.4, 4),
        ];
        
        Self {
            method,
            scales,
            filters,
        }
    }
    
    pub fn decompose(&self, x: &[f32], window: usize) -> DecomposedSeries {
        match self.method {
            DecompositionMethod::Wavelet => self.wavelet_decomposition(x, window),
            DecompositionMethod::EMD => self.emd_decomposition(x),
            DecompositionMethod::STL => self.stl_decomposition(x, window),
            DecompositionMethod::SSA => self.ssa_decomposition(x, window),
            DecompositionMethod::VMD => self.vmd_decomposition(x),
            DecompositionMethod::CEEMDAN => self.ceemdan_decomposition(x),
        }
    }
    
    fn wavelet_decomposition(&self, x: &[f32], levels: usize) -> DecomposedSeries {
        let mut coefficients = Vec::new();
        let mut current = x.to_vec();
        
        for level in 0..levels {
            let (approx, detail) = self.dwt_step(&current);
            coefficients.push(ScaleComponent {
                scale: 2_usize.pow(level as u32),
                component: detail,
                frequency: 0.5 / (2_usize.pow(level as u32) as f32),
                amplitude: self.calculate_amplitude(&approx),
            });
            current = approx;
        }
        
        // Reconstruct components
        let trend = current;
        let seasonal = self.reconstruct_seasonal(&coefficients);
        let residual = self.calculate_residual(x, &trend, &seasonal);
        
        DecomposedSeries {
            trend,
            seasonal,
            residual,
            scales: coefficients,
        }
    }
    
    fn dwt_step(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = x.len() / 2;
        let mut approx = Vec::with_capacity(n);
        let mut detail = Vec::with_capacity(n);
        
        // Haar wavelet transform (simplified)
        for i in 0..n {
            let a = x[2 * i];
            let b = x[2 * i + 1];
            approx.push((a + b) / 2.0_f32.sqrt());
            detail.push((a - b) / 2.0_f32.sqrt());
        }
        
        (approx, detail)
    }
    
    fn emd_decomposition(&self, x: &[f32]) -> DecomposedSeries {
        let mut imfs = Vec::new();
        let mut residue = x.to_vec();
        
        for _ in 0..5 {  // Max 5 IMFs
            let imf = self.extract_imf(&residue);
            if self.is_monotonic(&imf) {
                break;
            }
            
            for (i, &val) in imf.iter().enumerate() {
                residue[i] -= val;
            }
            
            imfs.push(imf);
        }
        
        // Map IMFs to components
        let trend = residue;
        let seasonal = if imfs.len() > 0 { imfs[0].clone() } else { vec![0.0; x.len()] };
        let residual = if imfs.len() > 1 { imfs[1].clone() } else { vec![0.0; x.len()] };
        
        let scales = imfs.into_iter().enumerate().map(|(i, component)| {
            ScaleComponent {
                scale: i + 1,
                component,
                frequency: 1.0 / (2.0_f32.powi(i as i32 + 1)),
                amplitude: 0.0,
            }
        }).collect();
        
        DecomposedSeries {
            trend,
            seasonal,
            residual,
            scales,
        }
    }
    
    fn extract_imf(&self, x: &[f32]) -> Vec<f32> {
        let mut h = x.to_vec();
        let max_iter = 100;
        
        for _ in 0..max_iter {
            let (upper_env, lower_env) = self.find_envelopes(&h);
            let mean_env: Vec<f32> = upper_env.iter()
                .zip(lower_env.iter())
                .map(|(u, l)| (u + l) / 2.0)
                .collect();
            
            let mut h_new = Vec::with_capacity(h.len());
            for (i, &val) in h.iter().enumerate() {
                h_new.push(val - mean_env[i]);
            }
            
            if self.is_imf(&h_new) {
                return h_new;
            }
            
            h = h_new;
        }
        
        h
    }
    
    fn find_envelopes(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut maxima = Vec::new();
        let mut minima = Vec::new();
        
        // Find local maxima and minima
        for i in 1..x.len() - 1 {
            if x[i] > x[i - 1] && x[i] > x[i + 1] {
                maxima.push((i, x[i]));
            }
            if x[i] < x[i - 1] && x[i] < x[i + 1] {
                minima.push((i, x[i]));
            }
        }
        
        // Interpolate envelopes (simplified linear interpolation)
        let upper = self.interpolate_envelope(&maxima, x.len());
        let lower = self.interpolate_envelope(&minima, x.len());
        
        (upper, lower)
    }
    
    fn interpolate_envelope(&self, points: &[(usize, f32)], length: usize) -> Vec<f32> {
        let mut envelope = vec![0.0; length];
        
        if points.is_empty() {
            return envelope;
        }
        
        // Simple linear interpolation
        for i in 0..length {
            let (left, right) = self.find_surrounding_points(i, points);
            
            if let (Some(l), Some(r)) = (left, right) {
                let t = (i - l.0) as f32 / (r.0 - l.0) as f32;
                envelope[i] = l.1 * (1.0 - t) + r.1 * t;
            } else if let Some(l) = left {
                envelope[i] = l.1;
            } else if let Some(r) = right {
                envelope[i] = r.1;
            }
        }
        
        envelope
    }
    
    fn find_surrounding_points(&self, index: usize, points: &[(usize, f32)]) -> (Option<(usize, f32)>, Option<(usize, f32)>) {
        let mut left = None;
        let mut right = None;
        
        for &point in points {
            if point.0 <= index {
                left = Some(point);
            }
            if point.0 >= index && right.is_none() {
                right = Some(point);
            }
        }
        
        (left, right)
    }
    
    fn is_imf(&self, x: &[f32]) -> bool {
        // Check if it's an Intrinsic Mode Function
        let mut n_extrema = 0;
        let mut n_zero_crossings = 0;
        
        for i in 1..x.len() - 1 {
            if (x[i] > x[i - 1] && x[i] > x[i + 1]) || 
               (x[i] < x[i - 1] && x[i] < x[i + 1]) {
                n_extrema += 1;
            }
            
            if x[i - 1] * x[i] < 0.0 {
                n_zero_crossings += 1;
            }
        }
        
        (n_extrema - n_zero_crossings).abs() <= 1
    }
    
    fn is_monotonic(&self, x: &[f32]) -> bool {
        let increasing = x.windows(2).all(|w| w[1] >= w[0]);
        let decreasing = x.windows(2).all(|w| w[1] <= w[0]);
        increasing || decreasing
    }
    
    fn stl_decomposition(&self, x: &[f32], period: usize) -> DecomposedSeries {
        // Simplified STL decomposition
        let trend = self.extract_trend_loess(x, period * 2 + 1);
        let detrended: Vec<f32> = x.iter().zip(trend.iter())
            .map(|(a, b)| a - b)
            .collect();
        
        let seasonal = self.extract_seasonal_pattern(&detrended, period);
        let residual = self.calculate_residual(x, &trend, &seasonal);
        
        DecomposedSeries {
            trend,
            seasonal,
            residual,
            scales: Vec::new(),
        }
    }
    
    fn extract_trend_loess(&self, x: &[f32], window: usize) -> Vec<f32> {
        // Moving average for simplicity (real LOESS would be more complex)
        let mut trend = Vec::with_capacity(x.len());
        let half_window = window / 2;
        
        for i in 0..x.len() {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(x.len());
            let sum: f32 = x[start..end].iter().sum();
            trend.push(sum / (end - start) as f32);
        }
        
        trend
    }
    
    fn extract_seasonal_pattern(&self, x: &[f32], period: usize) -> Vec<f32> {
        let mut seasonal = vec![0.0; x.len()];
        let mut counts = vec![0; period];
        let mut sums = vec![0.0; period];
        
        // Calculate average for each position in period
        for (i, &val) in x.iter().enumerate() {
            let pos = i % period;
            sums[pos] += val;
            counts[pos] += 1;
        }
        
        let pattern: Vec<f32> = sums.iter().zip(counts.iter())
            .map(|(s, c)| if *c > 0 { s / *c as f32 } else { 0.0 })
            .collect();
        
        // Apply pattern
        for i in 0..x.len() {
            seasonal[i] = pattern[i % period];
        }
        
        seasonal
    }
    
    fn ssa_decomposition(&self, x: &[f32], window: usize) -> DecomposedSeries {
        // Singular Spectrum Analysis
        let trajectory_matrix = self.build_trajectory_matrix(x, window);
        let (u, s, v) = self.svd_decomposition(&trajectory_matrix);
        
        // Reconstruct components from principal components
        let trend = self.reconstruct_from_svd(&u, &s, &v, 0, 2);
        let seasonal = self.reconstruct_from_svd(&u, &s, &v, 2, 5);
        let residual = self.calculate_residual(x, &trend, &seasonal);
        
        DecomposedSeries {
            trend,
            seasonal,
            residual,
            scales: Vec::new(),
        }
    }
    
    fn build_trajectory_matrix(&self, x: &[f32], window: usize) -> Vec<Vec<f32>> {
        let k = x.len() - window + 1;
        let mut matrix = vec![vec![0.0; k]; window];
        
        for i in 0..window {
            for j in 0..k {
                matrix[i][j] = x[i + j];
            }
        }
        
        matrix
    }
    
    fn svd_decomposition(&self, matrix: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<f32>, Vec<Vec<f32>>) {
        // Simplified SVD (real implementation would use proper linear algebra library)
        let m = matrix.len();
        let n = matrix[0].len();
        let min_dim = m.min(n);
        
        let u = vec![vec![0.0; m]; min_dim];
        let s = vec![1.0; min_dim];  // Simplified singular values
        let v = vec![vec![0.0; n]; min_dim];
        
        (u, s, v)
    }
    
    fn reconstruct_from_svd(&self, u: &[Vec<f32>], s: &[f32], v: &[Vec<f32>], 
                           start_comp: usize, end_comp: usize) -> Vec<f32> {
        // Simplified reconstruction
        let length = u[0].len() + v[0].len() - 1;
        vec![0.0; length]
    }
    
    fn vmd_decomposition(&self, x: &[f32]) -> DecomposedSeries {
        // Variational Mode Decomposition (simplified)
        // This would require iterative optimization in real implementation
        let k = 3;  // Number of modes
        let mut modes = vec![vec![0.0; x.len()]; k];
        
        // Simplified: divide signal into frequency bands
        for i in 0..k {
            modes[i] = self.apply_filter(x, &self.filters[i.min(self.filters.len() - 1)]);
        }
        
        DecomposedSeries {
            trend: modes[0].clone(),
            seasonal: modes[1].clone(),
            residual: if k > 2 { modes[2].clone() } else { vec![0.0; x.len()] },
            scales: Vec::new(),
        }
    }
    
    fn ceemdan_decomposition(&self, x: &[f32]) -> DecomposedSeries {
        // Complete Ensemble EMD with Adaptive Noise
        let n_ensembles = 100;
        let noise_std = 0.1;
        let mut ensemble_imfs = Vec::new();
        
        for _ in 0..n_ensembles {
            let noisy = self.add_noise(x, noise_std);
            let decomp = self.emd_decomposition(&noisy);
            ensemble_imfs.push(decomp);
        }
        
        // Average the ensembles
        self.average_decompositions(&ensemble_imfs)
    }
    
    fn add_noise(&self, x: &[f32], std: f32) -> Vec<f32> {
        x.iter().map(|&val| {
            val + (rand::random::<f32>() - 0.5) * 2.0 * std
        }).collect()
    }
    
    fn average_decompositions(&self, decomps: &[DecomposedSeries]) -> DecomposedSeries {
        if decomps.is_empty() {
            return DecomposedSeries {
                trend: Vec::new(),
                seasonal: Vec::new(),
                residual: Vec::new(),
                scales: Vec::new(),
            };
        }
        
        let len = decomps[0].trend.len();
        let mut avg_trend = vec![0.0; len];
        let mut avg_seasonal = vec![0.0; len];
        let mut avg_residual = vec![0.0; len];
        
        for decomp in decomps {
            for i in 0..len {
                avg_trend[i] += decomp.trend[i];
                avg_seasonal[i] += decomp.seasonal[i];
                avg_residual[i] += decomp.residual[i];
            }
        }
        
        let n = decomps.len() as f32;
        for i in 0..len {
            avg_trend[i] /= n;
            avg_seasonal[i] /= n;
            avg_residual[i] /= n;
        }
        
        DecomposedSeries {
            trend: avg_trend,
            seasonal: avg_seasonal,
            residual: avg_residual,
            scales: Vec::new(),
        }
    }
    
    fn apply_filter(&self, x: &[f32], filter: &DecompositionFilter) -> Vec<f32> {
        // Simplified filtering
        match filter.filter_type {
            FilterType::LowPass => self.low_pass_filter(x, filter.cutoff_frequency),
            FilterType::HighPass => self.high_pass_filter(x, filter.cutoff_frequency),
            FilterType::BandPass => self.band_pass_filter(x, filter.cutoff_frequency, 0.3),
            FilterType::BandStop => self.band_stop_filter(x, filter.cutoff_frequency, 0.3),
        }
    }
    
    fn low_pass_filter(&self, x: &[f32], cutoff: f32) -> Vec<f32> {
        // Simple moving average as low-pass filter
        let window = (1.0 / cutoff) as usize;
        let mut filtered = Vec::with_capacity(x.len());
        
        for i in 0..x.len() {
            let start = i.saturating_sub(window / 2);
            let end = (i + window / 2 + 1).min(x.len());
            let sum: f32 = x[start..end].iter().sum();
            filtered.push(sum / (end - start) as f32);
        }
        
        filtered
    }
    
    fn high_pass_filter(&self, x: &[f32], cutoff: f32) -> Vec<f32> {
        let low_pass = self.low_pass_filter(x, cutoff);
        x.iter().zip(low_pass.iter())
            .map(|(orig, low)| orig - low)
            .collect()
    }
    
    fn band_pass_filter(&self, x: &[f32], low_cutoff: f32, high_cutoff: f32) -> Vec<f32> {
        let high_passed = self.high_pass_filter(x, low_cutoff);
        self.low_pass_filter(&high_passed, high_cutoff)
    }
    
    fn band_stop_filter(&self, x: &[f32], low_cutoff: f32, high_cutoff: f32) -> Vec<f32> {
        let band_passed = self.band_pass_filter(x, low_cutoff, high_cutoff);
        x.iter().zip(band_passed.iter())
            .map(|(orig, band)| orig - band)
            .collect()
    }
    
    fn reconstruct_seasonal(&self, coefficients: &[ScaleComponent]) -> Vec<f32> {
        if coefficients.is_empty() {
            return Vec::new();
        }
        
        let len = coefficients[0].component.len();
        let mut seasonal = vec![0.0; len];
        
        // Sum mid-frequency components
        for comp in coefficients.iter().skip(1).take(2) {
            for (i, &val) in comp.component.iter().enumerate() {
                if i < seasonal.len() {
                    seasonal[i] += val;
                }
            }
        }
        
        seasonal
    }
    
    fn calculate_residual(&self, original: &[f32], trend: &[f32], seasonal: &[f32]) -> Vec<f32> {
        original.iter().enumerate().map(|(i, &val)| {
            val - trend.get(i).unwrap_or(&0.0) - seasonal.get(i).unwrap_or(&0.0)
        }).collect()
    }
    
    fn calculate_amplitude(&self, x: &[f32]) -> f32 {
        let max = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min = x.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        (max - min) / 2.0
    }
}

impl DecompositionFilter {
    fn low_pass(cutoff: f32, order: usize) -> Self {
        Self {
            filter_type: FilterType::LowPass,
            cutoff_frequency: cutoff,
            order,
        }
    }
    
    fn high_pass(cutoff: f32, order: usize) -> Self {
        Self {
            filter_type: FilterType::HighPass,
            cutoff_frequency: cutoff,
            order,
        }
    }
    
    fn band_pass(low: f32, high: f32, order: usize) -> Self {
        Self {
            filter_type: FilterType::BandPass,
            cutoff_frequency: (low + high) / 2.0,
            order,
        }
    }
}

use rand;