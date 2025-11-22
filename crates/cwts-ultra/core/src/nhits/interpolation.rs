// Hierarchical interpolation for NHITS forecasting

use std::f32::consts::PI;

pub struct HierarchicalInterpolator {
    method: InterpolationMethod,
    learnable_weights: Vec<f32>,
    basis_functions: Vec<BasisFunction>,
}

#[derive(Clone, Copy)]
pub enum InterpolationMethod {
    Linear,
    Cubic,
    Spline,
    Neural,
    Hermite,
    Akima,
}

pub struct BasisFunction {
    function_type: BasisType,
    parameters: Vec<f32>,
}

#[derive(Clone, Copy)]
enum BasisType {
    Polynomial,
    Trigonometric,
    Exponential,
    Gaussian,
    RBF,
}

impl HierarchicalInterpolator {
    pub fn new(method: InterpolationMethod) -> Self {
        Self {
            method,
            learnable_weights: vec![1.0; 10],
            basis_functions: Self::initialize_basis_functions(),
        }
    }
    
    fn initialize_basis_functions() -> Vec<BasisFunction> {
        vec![
            BasisFunction {
                function_type: BasisType::Polynomial,
                parameters: vec![1.0, 0.5, 0.25],
            },
            BasisFunction {
                function_type: BasisType::Trigonometric,
                parameters: vec![1.0, 2.0, 3.0],
            },
            BasisFunction {
                function_type: BasisType::Gaussian,
                parameters: vec![0.0, 1.0],
            },
        ]
    }
    
    pub fn interpolate(&self, stack_outputs: &[Vec<f32>]) -> Vec<f32> {
        match self.method {
            InterpolationMethod::Linear => self.linear_interpolation(stack_outputs),
            InterpolationMethod::Cubic => self.cubic_interpolation(stack_outputs),
            InterpolationMethod::Spline => self.spline_interpolation(stack_outputs),
            InterpolationMethod::Neural => self.neural_interpolation(stack_outputs),
            InterpolationMethod::Hermite => self.hermite_interpolation(stack_outputs),
            InterpolationMethod::Akima => self.akima_interpolation(stack_outputs),
        }
    }
    
    fn linear_interpolation(&self, stack_outputs: &[Vec<f32>]) -> Vec<f32> {
        if stack_outputs.is_empty() {
            return Vec::new();
        }
        
        let target_len = stack_outputs[0].len();
        let mut interpolated = vec![0.0; target_len];
        
        for (stack_idx, output) in stack_outputs.iter().enumerate() {
            let scale_factor = 2_usize.pow(stack_idx as u32);
            
            for i in 0..target_len {
                let source_idx = i / scale_factor;
                let t = (i % scale_factor) as f32 / scale_factor as f32;
                
                if source_idx < output.len() - 1 {
                    let y0 = output[source_idx];
                    let y1 = output[source_idx + 1];
                    interpolated[i] += self.learnable_weights[stack_idx] * (y0 * (1.0 - t) + y1 * t);
                } else if source_idx < output.len() {
                    interpolated[i] += self.learnable_weights[stack_idx] * output[source_idx];
                }
            }
        }
        
        // Normalize by weight sum
        let weight_sum: f32 = self.learnable_weights[..stack_outputs.len()].iter().sum();
        for val in interpolated.iter_mut() {
            *val /= weight_sum;
        }
        
        interpolated
    }
    
    fn cubic_interpolation(&self, stack_outputs: &[Vec<f32>]) -> Vec<f32> {
        if stack_outputs.is_empty() {
            return Vec::new();
        }
        
        let target_len = stack_outputs[0].len();
        let mut interpolated = vec![0.0; target_len];
        
        for (stack_idx, output) in stack_outputs.iter().enumerate() {
            let scale_factor = 2_usize.pow(stack_idx as u32);
            
            for i in 0..target_len {
                let source_idx = i / scale_factor;
                let t = (i % scale_factor) as f32 / scale_factor as f32;
                
                if source_idx > 0 && source_idx < output.len() - 2 {
                    let y0 = output[source_idx - 1];
                    let y1 = output[source_idx];
                    let y2 = output[source_idx + 1];
                    let y3 = output[source_idx + 2];
                    
                    let a0 = y3 - y2 - y0 + y1;
                    let a1 = y0 - y1 - a0;
                    let a2 = y2 - y0;
                    let a3 = y1;
                    
                    let value = a0 * t.powi(3) + a1 * t.powi(2) + a2 * t + a3;
                    interpolated[i] += self.learnable_weights[stack_idx] * value;
                } else if source_idx < output.len() {
                    interpolated[i] += self.learnable_weights[stack_idx] * output[source_idx];
                }
            }
        }
        
        // Normalize
        let weight_sum: f32 = self.learnable_weights[..stack_outputs.len()].iter().sum();
        for val in interpolated.iter_mut() {
            *val /= weight_sum;
        }
        
        interpolated
    }
    
    fn spline_interpolation(&self, stack_outputs: &[Vec<f32>]) -> Vec<f32> {
        if stack_outputs.is_empty() {
            return Vec::new();
        }
        
        let target_len = stack_outputs[0].len();
        let mut interpolated = vec![0.0; target_len];
        
        for (stack_idx, output) in stack_outputs.iter().enumerate() {
            let spline = self.compute_spline_coefficients(output);
            let scale_factor = 2_usize.pow(stack_idx as u32);
            
            for i in 0..target_len {
                let x = i as f32 / scale_factor as f32;
                let value = self.evaluate_spline(&spline, x);
                interpolated[i] += self.learnable_weights[stack_idx] * value;
            }
        }
        
        // Normalize
        let weight_sum: f32 = self.learnable_weights[..stack_outputs.len()].iter().sum();
        for val in interpolated.iter_mut() {
            *val /= weight_sum;
        }
        
        interpolated
    }
    
    fn compute_spline_coefficients(&self, y: &[f32]) -> SplineCoefficients {
        let n = y.len();
        if n < 2 {
            return SplineCoefficients::default();
        }
        
        // Natural cubic spline coefficients
        let mut h = vec![1.0; n - 1];  // Assume uniform spacing
        let mut alpha = vec![0.0; n - 1];
        
        for i in 1..n - 1 {
            alpha[i] = 3.0 * (y[i + 1] - y[i]) / h[i] - 3.0 * (y[i] - y[i - 1]) / h[i - 1];
        }
        
        let mut l = vec![1.0; n];
        let mut mu = vec![0.0; n];
        let mut z = vec![0.0; n];
        
        for i in 1..n - 1 {
            l[i] = 2.0 * (h[i - 1] + h[i]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }
        
        let mut c = vec![0.0; n];
        let mut b = vec![0.0; n - 1];
        let mut d = vec![0.0; n - 1];
        
        for j in (0..n - 1).rev() {
            c[j] = z[j] - mu[j] * c[j + 1];
            b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0;
            d[j] = (c[j + 1] - c[j]) / (3.0 * h[j]);
        }
        
        SplineCoefficients {
            a: y.to_vec(),
            b,
            c: c[..n - 1].to_vec(),
            d,
        }
    }
    
    fn evaluate_spline(&self, spline: &SplineCoefficients, x: f32) -> f32 {
        let i = (x as usize).min(spline.a.len() - 2);
        let dx = x - i as f32;
        
        spline.a[i] + spline.b[i] * dx + spline.c[i] * dx.powi(2) + spline.d[i] * dx.powi(3)
    }
    
    fn neural_interpolation(&self, stack_outputs: &[Vec<f32>]) -> Vec<f32> {
        if stack_outputs.is_empty() {
            return Vec::new();
        }
        
        let target_len = stack_outputs[0].len();
        let mut interpolated = vec![0.0; target_len];
        
        // Use basis functions for neural interpolation
        for (stack_idx, output) in stack_outputs.iter().enumerate() {
            for i in 0..target_len {
                let mut value = 0.0;
                
                for (basis_idx, basis) in self.basis_functions.iter().enumerate() {
                    let basis_value = self.evaluate_basis(basis, i as f32 / target_len as f32);
                    let output_idx = (i * output.len()) / target_len;
                    
                    if output_idx < output.len() {
                        value += basis_value * output[output_idx] * 
                                self.learnable_weights[basis_idx % self.learnable_weights.len()];
                    }
                }
                
                interpolated[i] += value;
            }
        }
        
        // Apply non-linearity
        for val in interpolated.iter_mut() {
            *val = val.tanh();
        }
        
        interpolated
    }
    
    fn hermite_interpolation(&self, stack_outputs: &[Vec<f32>]) -> Vec<f32> {
        if stack_outputs.is_empty() {
            return Vec::new();
        }
        
        let target_len = stack_outputs[0].len();
        let mut interpolated = vec![0.0; target_len];
        
        for (stack_idx, output) in stack_outputs.iter().enumerate() {
            let scale_factor = 2_usize.pow(stack_idx as u32);
            
            for i in 0..target_len {
                let source_idx = i / scale_factor;
                let t = (i % scale_factor) as f32 / scale_factor as f32;
                
                if source_idx < output.len() - 1 {
                    let y0 = output[source_idx];
                    let y1 = output[source_idx + 1];
                    
                    // Estimate derivatives
                    let m0 = if source_idx > 0 {
                        (y1 - output[source_idx - 1]) / 2.0
                    } else {
                        y1 - y0
                    };
                    
                    let m1 = if source_idx < output.len() - 2 {
                        (output[source_idx + 2] - y0) / 2.0
                    } else {
                        y1 - y0
                    };
                    
                    // Hermite basis functions
                    let h00 = (1.0 + 2.0 * t) * (1.0 - t).powi(2);
                    let h10 = t * (1.0 - t).powi(2);
                    let h01 = t.powi(2) * (3.0 - 2.0 * t);
                    let h11 = t.powi(2) * (t - 1.0);
                    
                    let value = h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1;
                    interpolated[i] += self.learnable_weights[stack_idx] * value;
                } else if source_idx < output.len() {
                    interpolated[i] += self.learnable_weights[stack_idx] * output[source_idx];
                }
            }
        }
        
        // Normalize
        let weight_sum: f32 = self.learnable_weights[..stack_outputs.len()].iter().sum();
        for val in interpolated.iter_mut() {
            *val /= weight_sum;
        }
        
        interpolated
    }
    
    fn akima_interpolation(&self, stack_outputs: &[Vec<f32>]) -> Vec<f32> {
        if stack_outputs.is_empty() {
            return Vec::new();
        }
        
        let target_len = stack_outputs[0].len();
        let mut interpolated = vec![0.0; target_len];
        
        for (stack_idx, output) in stack_outputs.iter().enumerate() {
            if output.len() < 5 {
                // Fall back to cubic for small datasets
                return self.cubic_interpolation(stack_outputs);
            }
            
            let slopes = self.compute_akima_slopes(output);
            let scale_factor = 2_usize.pow(stack_idx as u32);
            
            for i in 0..target_len {
                let source_idx = i / scale_factor;
                let t = (i % scale_factor) as f32 / scale_factor as f32;
                
                if source_idx < output.len() - 1 && source_idx < slopes.len() - 1 {
                    let y0 = output[source_idx];
                    let y1 = output[source_idx + 1];
                    let m0 = slopes[source_idx];
                    let m1 = slopes[source_idx + 1];
                    
                    // Hermite interpolation with Akima slopes
                    let h00 = (1.0 + 2.0 * t) * (1.0 - t).powi(2);
                    let h10 = t * (1.0 - t).powi(2);
                    let h01 = t.powi(2) * (3.0 - 2.0 * t);
                    let h11 = t.powi(2) * (t - 1.0);
                    
                    let value = h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1;
                    interpolated[i] += self.learnable_weights[stack_idx] * value;
                }
            }
        }
        
        // Normalize
        let weight_sum: f32 = self.learnable_weights[..stack_outputs.len()].iter().sum();
        for val in interpolated.iter_mut() {
            *val /= weight_sum;
        }
        
        interpolated
    }
    
    fn compute_akima_slopes(&self, y: &[f32]) -> Vec<f32> {
        let n = y.len();
        let mut slopes = vec![0.0; n];
        
        // Compute segment slopes
        let mut m = vec![0.0; n + 3];
        for i in 0..n - 1 {
            m[i + 2] = y[i + 1] - y[i];
        }
        
        // Extend boundaries
        m[1] = 2.0 * m[2] - m[3];
        m[0] = 2.0 * m[1] - m[2];
        m[n + 1] = 2.0 * m[n] - m[n - 1];
        m[n + 2] = 2.0 * m[n + 1] - m[n];
        
        // Compute Akima slopes
        for i in 0..n {
            let w1 = (m[i + 3] - m[i + 2]).abs();
            let w2 = (m[i + 1] - m[i]).abs();
            
            if w1 + w2 > 0.0 {
                slopes[i] = (w1 * m[i + 1] + w2 * m[i + 2]) / (w1 + w2);
            } else {
                slopes[i] = (m[i + 1] + m[i + 2]) / 2.0;
            }
        }
        
        slopes
    }
    
    fn evaluate_basis(&self, basis: &BasisFunction, t: f32) -> f32 {
        match basis.function_type {
            BasisType::Polynomial => {
                basis.parameters.iter().enumerate()
                    .map(|(i, &coef)| coef * t.powi(i as i32))
                    .sum()
            }
            BasisType::Trigonometric => {
                basis.parameters.iter().enumerate()
                    .map(|(i, &coef)| coef * (2.0 * PI * (i + 1) as f32 * t).sin())
                    .sum()
            }
            BasisType::Exponential => {
                basis.parameters.iter()
                    .map(|&coef| coef * t.exp())
                    .sum()
            }
            BasisType::Gaussian => {
                let mean = basis.parameters[0];
                let std = basis.parameters[1];
                (-0.5 * ((t - mean) / std).powi(2)).exp()
            }
            BasisType::RBF => {
                let center = basis.parameters[0];
                let width = basis.parameters[1];
                (-(t - center).powi(2) / (2.0 * width.powi(2))).exp()
            }
        }
    }
}

#[derive(Default)]
struct SplineCoefficients {
    a: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
    d: Vec<f32>,
}