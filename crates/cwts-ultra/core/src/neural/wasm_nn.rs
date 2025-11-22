//! WebAssembly Neural Networks with SIMD128 Acceleration
//! 
//! High-performance neural network implementations optimized for WebAssembly runtime.
//! Features SIMD128 acceleration when available with automatic scalar fallback.
//! Designed for browser and Node.js compatibility with minimal memory footprint.
//! 
//! Performance targets:
//! - Forward pass: <200μs for 32-neuron layers
//! - SIMD operations: 3-4x speedup over scalar
//! - Memory usage: <1MB for typical networks
//! - Quantization: int8/int16 support for mobile deployment

use std::arch::wasm32::*;
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use web_sys::console;

use crate::simd::wasm32::{WasmFeatures, WasmMemory};

/// Neural network layer with WASM optimizations
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WasmNeuralLayer {
    weights: Vec<f32>,
    biases: Vec<f32>,
    input_size: usize,
    output_size: usize,
    features: WasmFeatures,
    quantized: bool,
    quantization_scale: f32,
    quantization_offset: i8,
}

/// Activation functions optimized for WASM SIMD128
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum WasmActivation {
    ReLU = 0,
    Sigmoid = 1,
    Tanh = 2,
    LeakyReLU = 3,
    Swish = 4,
    GELU = 5,
}

/// Neural network quantization modes
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum QuantizationMode {
    None = 0,
    Int8 = 1,
    Int16 = 2,
    Dynamic = 3,
}

#[wasm_bindgen]
impl WasmNeuralLayer {
    /// Create a new neural layer optimized for WebAssembly
    #[wasm_bindgen(constructor)]
    pub fn new(input_size: usize, output_size: usize) -> Self {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();

        let weights = Self::initialize_weights(input_size, output_size);
        let biases = vec![0.0; output_size];

        Self {
            weights,
            biases,
            input_size,
            output_size,
            features: WasmFeatures::detect(),
            quantized: false,
            quantization_scale: 1.0,
            quantization_offset: 0,
        }
    }

    /// Initialize weights using Xavier/Glorot initialization
    fn initialize_weights(input_size: usize, output_size: usize) -> Vec<f32> {
        let limit = (6.0_f32 / (input_size + output_size) as f32).sqrt();
        (0..(input_size * output_size))
            .map(|i| {
                // Use deterministic pseudo-random for consistent initialization
                let x = (i as f32 * 0.618033988749895) % 1.0; // Golden ratio for good distribution
                (x - 0.5) * 2.0 * limit
            })
            .collect()
    }

    /// Enable quantization for reduced memory usage and faster inference
    #[wasm_bindgen]
    pub fn quantize(&mut self, mode: u8, dynamic_range: bool) {
        let quantization_mode = match mode {
            1 => QuantizationMode::Int8,
            2 => QuantizationMode::Int16,
            3 => QuantizationMode::Dynamic,
            _ => QuantizationMode::None,
        };

        match quantization_mode {
            QuantizationMode::Int8 => self.quantize_int8(dynamic_range),
            QuantizationMode::Int16 => self.quantize_int16(dynamic_range),
            QuantizationMode::Dynamic => self.quantize_dynamic(),
            QuantizationMode::None => {
                self.quantized = false;
                self.quantization_scale = 1.0;
                self.quantization_offset = 0;
            }
        }
    }

    /// Quantize to 8-bit integers
    fn quantize_int8(&mut self, dynamic_range: bool) {
        if dynamic_range {
            let min_val = self.weights.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = self.weights.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            
            self.quantization_scale = (max_val - min_val) / 255.0;
            self.quantization_offset = (min_val / self.quantization_scale) as i8;
        } else {
            self.quantization_scale = 1.0 / 127.0;
            self.quantization_offset = 0;
        }

        self.quantized = true;
        
        #[cfg(target_arch = "wasm32")]
        console::log_1(&format!("Quantized to INT8: scale={:.6}, offset={}", 
                               self.quantization_scale, self.quantization_offset).into());
    }

    /// Quantize to 16-bit integers
    fn quantize_int16(&mut self, dynamic_range: bool) {
        if dynamic_range {
            let min_val = self.weights.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = self.weights.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            
            self.quantization_scale = (max_val - min_val) / 65535.0;
            self.quantization_offset = (min_val / self.quantization_scale) as i8;
        } else {
            self.quantization_scale = 1.0 / 32767.0;
            self.quantization_offset = 0;
        }

        self.quantized = true;
    }

    /// Dynamic quantization based on activation statistics
    fn quantize_dynamic(&mut self) {
        // For dynamic quantization, we use statistics from previous forward passes
        // This is a simplified version - in practice, you'd collect statistics over time
        self.quantization_scale = 1.0 / 127.0;
        self.quantization_offset = 0;
        self.quantized = true;
    }

    /// Forward pass through the layer with SIMD acceleration
    #[wasm_bindgen]
    pub fn forward(&self, input: &[f32], activation: u8) -> Vec<f32> {
        let mut output = vec![0.0; self.output_size];
        
        if self.features.has_simd128 && input.len() >= 4 {
            unsafe {
                self.forward_simd128(input, &mut output);
            }
        } else {
            self.forward_scalar(input, &mut output);
        }

        // Apply activation function
        let activation_fn = match activation {
            1 => WasmActivation::Sigmoid,
            2 => WasmActivation::Tanh,
            3 => WasmActivation::LeakyReLU,
            4 => WasmActivation::Swish,
            5 => WasmActivation::GELU,
            _ => WasmActivation::ReLU,
        };

        self.apply_activation(&mut output, activation_fn);
        output
    }

    /// SIMD128-accelerated forward pass
    #[cfg(target_feature = "simd128")]
    unsafe fn forward_simd128(&self, input: &[f32], output: &mut [f32]) {
        let input_chunks = self.input_size / 4;
        let remainder = self.input_size % 4;

        for out_idx in 0..self.output_size {
            let mut acc = f32x4_splat(0.0);
            
            // Process 4 inputs at a time
            for chunk in 0..input_chunks {
                let input_idx = chunk * 4;
                let weight_idx = out_idx * self.input_size + input_idx;
                
                // Load 4 inputs and 4 weights
                let inputs = v128_load(&input[input_idx] as *const f32 as *const v128);
                let weights = v128_load(&self.weights[weight_idx] as *const f32 as *const v128);
                
                let input_vec = f32x4_convert_i32x4(inputs);
                let weight_vec = f32x4_convert_i32x4(weights);
                
                // Multiply and accumulate
                acc = f32x4_add(acc, f32x4_mul(input_vec, weight_vec));
            }
            
            // Horizontal sum
            let mut sum = f32x4_extract_lane::<0>(acc) +
                         f32x4_extract_lane::<1>(acc) +
                         f32x4_extract_lane::<2>(acc) +
                         f32x4_extract_lane::<3>(acc);
            
            // Handle remaining elements
            for i in 0..remainder {
                let input_idx = input_chunks * 4 + i;
                let weight_idx = out_idx * self.input_size + input_idx;
                sum += input[input_idx] * self.weights[weight_idx];
            }
            
            // Add bias and handle quantization
            output[out_idx] = if self.quantized {
                sum * self.quantization_scale + self.quantization_offset as f32 + self.biases[out_idx]
            } else {
                sum + self.biases[out_idx]
            };
        }
    }

    /// Fallback to SIMD implementation when SIMD128 is not available
    #[cfg(not(target_feature = "simd128"))]
    unsafe fn forward_simd128(&self, input: &[f32], output: &mut [f32]) {
        self.forward_scalar(input, output);
    }

    /// Scalar implementation for compatibility
    fn forward_scalar(&self, input: &[f32], output: &mut [f32]) {
        for out_idx in 0..self.output_size {
            let mut sum = 0.0;
            
            for in_idx in 0..self.input_size {
                let weight_idx = out_idx * self.input_size + in_idx;
                sum += input[in_idx] * self.weights[weight_idx];
            }
            
            output[out_idx] = if self.quantized {
                sum * self.quantization_scale + self.quantization_offset as f32 + self.biases[out_idx]
            } else {
                sum + self.biases[out_idx]
            };
        }
    }

    /// Apply activation function with SIMD acceleration when possible
    fn apply_activation(&self, values: &mut [f32], activation: WasmActivation) {
        if self.features.has_simd128 && values.len() >= 4 {
            unsafe {
                self.apply_activation_simd128(values, activation);
            }
        } else {
            self.apply_activation_scalar(values, activation);
        }
    }

    /// SIMD128 activation functions
    #[cfg(target_feature = "simd128")]
    unsafe fn apply_activation_simd128(&self, values: &mut [f32], activation: WasmActivation) {
        let chunks = values.len() / 4;
        let remainder = values.len() % 4;

        for chunk in 0..chunks {
            let idx = chunk * 4;
            let vals = v128_load(&values[idx] as *const f32 as *const v128);
            let vec = f32x4_convert_i32x4(vals);
            
            let result = match activation {
                WasmActivation::ReLU => {
                    let zero = f32x4_splat(0.0);
                    f32x4_pmax(vec, zero)
                }
                WasmActivation::Sigmoid => {
                    // Approximate sigmoid: 1 / (1 + exp(-x))
                    let one = f32x4_splat(1.0);
                    let neg_vec = f32x4_neg(vec);
                    // Approximate exp(-x) for WASM efficiency
                    let exp_approx = self.simd_exp_approx(neg_vec);
                    let denom = f32x4_add(one, exp_approx);
                    f32x4_div(one, denom)
                }
                WasmActivation::Tanh => {
                    // Approximate tanh using sigmoid: tanh(x) = 2*sigmoid(2x) - 1
                    let two = f32x4_splat(2.0);
                    let one = f32x4_splat(1.0);
                    let two_x = f32x4_mul(vec, two);
                    let neg_two_x = f32x4_neg(two_x);
                    let exp_approx = self.simd_exp_approx(neg_two_x);
                    let sigmoid = f32x4_div(one, f32x4_add(one, exp_approx));
                    f32x4_sub(f32x4_mul(two, sigmoid), one)
                }
                WasmActivation::LeakyReLU => {
                    let zero = f32x4_splat(0.0);
                    let alpha = f32x4_splat(0.01);
                    let mask = f32x4_gt(vec, zero);
                    f32x4_select(mask, vec, f32x4_mul(vec, alpha))
                }
                WasmActivation::Swish => {
                    // Swish: x * sigmoid(x)
                    let one = f32x4_splat(1.0);
                    let neg_vec = f32x4_neg(vec);
                    let exp_approx = self.simd_exp_approx(neg_vec);
                    let sigmoid = f32x4_div(one, f32x4_add(one, exp_approx));
                    f32x4_mul(vec, sigmoid)
                }
                WasmActivation::GELU => {
                    // Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
                    let half = f32x4_splat(0.5);
                    let one = f32x4_splat(1.0);
                    let sqrt_2_pi = f32x4_splat(0.7978845608); // sqrt(2/π)
                    let coeff = f32x4_splat(0.044715);
                    
                    let x_cubed = f32x4_mul(f32x4_mul(vec, vec), vec);
                    let inner = f32x4_add(vec, f32x4_mul(coeff, x_cubed));
                    let scaled = f32x4_mul(sqrt_2_pi, inner);
                    
                    // Approximate tanh
                    let two = f32x4_splat(2.0);
                    let two_scaled = f32x4_mul(two, scaled);
                    let neg_two_scaled = f32x4_neg(two_scaled);
                    let exp_approx = self.simd_exp_approx(neg_two_scaled);
                    let tanh_approx = f32x4_div(
                        f32x4_sub(one, exp_approx),
                        f32x4_add(one, exp_approx)
                    );
                    
                    f32x4_mul(f32x4_mul(vec, half), f32x4_add(one, tanh_approx))
                }
            };

            v128_store(&mut values[idx] as *mut f32 as *mut v128, result);
        }

        // Handle remaining elements
        for i in 0..remainder {
            let idx = chunks * 4 + i;
            values[idx] = self.scalar_activation(values[idx], activation);
        }
    }

    /// Fast exponential approximation for SIMD
    #[cfg(target_feature = "simd128")]
    unsafe fn simd_exp_approx(&self, x: v128) -> v128 {
        // Pade approximation for exp(x) that's efficient in WASM
        // exp(x) ≈ (1 + x/2) / (1 - x/2) for small x
        let half = f32x4_splat(0.5);
        let one = f32x4_splat(1.0);
        
        let x_half = f32x4_mul(f32x4_convert_i32x4(x), half);
        let numer = f32x4_add(one, x_half);
        let denom = f32x4_sub(one, x_half);
        
        f32x4_div(numer, denom)
    }

    /// Fallback to scalar activation when SIMD128 is not available
    #[cfg(not(target_feature = "simd128"))]
    unsafe fn apply_activation_simd128(&self, values: &mut [f32], activation: WasmActivation) {
        self.apply_activation_scalar(values, activation);
    }

    /// Scalar activation functions
    fn apply_activation_scalar(&self, values: &mut [f32], activation: WasmActivation) {
        for value in values.iter_mut() {
            *value = self.scalar_activation(*value, activation);
        }
    }

    /// Single value activation function
    fn scalar_activation(&self, x: f32, activation: WasmActivation) -> f32 {
        match activation {
            WasmActivation::ReLU => x.max(0.0),
            WasmActivation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            WasmActivation::Tanh => x.tanh(),
            WasmActivation::LeakyReLU => if x > 0.0 { x } else { 0.01 * x },
            WasmActivation::Swish => x / (1.0 + (-x).exp()),
            WasmActivation::GELU => {
                // Approximate GELU
                let sqrt_2_pi = 0.7978845608_f32; // sqrt(2/π)
                let coeff = 0.044715_f32;
                let tanh_input = sqrt_2_pi * (x + coeff * x * x * x);
                x * 0.5 * (1.0 + tanh_input.tanh())
            }
        }
    }

    /// Get layer information for debugging
    #[wasm_bindgen]
    pub fn get_info(&self) -> String {
        format!(
            "WasmNeuralLayer: {}→{}, SIMD: {}, Quantized: {} (scale: {:.6})",
            self.input_size,
            self.output_size,
            self.features.has_simd128,
            self.quantized,
            self.quantization_scale
        )
    }

    /// Get memory usage in bytes
    #[wasm_bindgen]
    pub fn memory_usage(&self) -> usize {
        let weights_size = self.weights.len() * std::mem::size_of::<f32>();
        let biases_size = self.biases.len() * std::mem::size_of::<f32>();
        let struct_size = std::mem::size_of::<Self>();
        
        weights_size + biases_size + struct_size
    }

    /// Batch forward pass for multiple inputs
    #[wasm_bindgen]
    pub fn batch_forward(&self, inputs: &[f32], batch_size: usize, activation: u8) -> Vec<f32> {
        let mut outputs = Vec::with_capacity(batch_size * self.output_size);
        
        for batch_idx in 0..batch_size {
            let input_start = batch_idx * self.input_size;
            let input_end = input_start + self.input_size;
            
            if input_end <= inputs.len() {
                let input_slice = &inputs[input_start..input_end];
                let output = self.forward(input_slice, activation);
                outputs.extend(output);
            }
        }
        
        outputs
    }
}

/// Complete neural network with multiple layers
#[wasm_bindgen]
pub struct WasmNeuralNetwork {
    layers: Vec<WasmNeuralLayer>,
    features: WasmFeatures,
    total_params: usize,
}

#[wasm_bindgen]
impl WasmNeuralNetwork {
    /// Create a new neural network
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();

        Self {
            layers: Vec::new(),
            features: WasmFeatures::detect(),
            total_params: 0,
        }
    }

    /// Add a layer to the network
    #[wasm_bindgen]
    pub fn add_layer(&mut self, input_size: usize, output_size: usize) {
        let layer = WasmNeuralLayer::new(input_size, output_size);
        self.total_params += layer.weights.len() + layer.biases.len();
        self.layers.push(layer);
    }

    /// Forward pass through the entire network
    #[wasm_bindgen]
    pub fn predict(&self, input: &[f32], activations: &[u8]) -> Vec<f32> {
        if self.layers.is_empty() {
            return input.to_vec();
        }

        let mut current_input = input.to_vec();
        
        for (i, layer) in self.layers.iter().enumerate() {
            let activation = activations.get(i).copied().unwrap_or(0);
            current_input = layer.forward(&current_input, activation);
        }
        
        current_input
    }

    /// Quantize all layers in the network
    #[wasm_bindgen]
    pub fn quantize_network(&mut self, mode: u8, dynamic_range: bool) {
        for layer in &mut self.layers {
            layer.quantize(mode, dynamic_range);
        }
        
        #[cfg(target_arch = "wasm32")]
        console::log_1(&format!("Network quantized: {} layers", self.layers.len()).into());
    }

    /// Get network statistics
    #[wasm_bindgen]
    pub fn get_stats(&self) -> String {
        let memory_usage: usize = self.layers.iter().map(|l| l.memory_usage()).sum();
        let memory_mb = memory_usage as f64 / (1024.0 * 1024.0);
        
        format!(
            "WasmNeuralNetwork: {} layers, {} params, {:.2}MB, SIMD: {}",
            self.layers.len(),
            self.total_params,
            memory_mb,
            self.features.has_simd128
        )
    }

    /// Benchmark forward pass performance
    #[wasm_bindgen]
    pub fn benchmark(&self, input: &[f32], activations: &[u8], iterations: usize) -> f64 {
        if self.layers.is_empty() || input.is_empty() {
            return 0.0;
        }

        // Simple timing - in production, use performance.now() in browser
        let start_time = Self::get_timestamp();
        
        for _ in 0..iterations {
            let _ = self.predict(input, activations);
        }
        
        let end_time = Self::get_timestamp();
        let total_time = end_time - start_time;
        
        // Return average time per iteration in milliseconds
        total_time / iterations as f64
    }

    /// Get current timestamp (simplified for WASM compatibility)
    fn get_timestamp() -> f64 {
        // In a real WASM environment, you'd use performance.now() or Date.now()
        // This is a placeholder for compilation
        #[cfg(target_arch = "wasm32")]
        {
            // Use web_sys to get performance.now() in browser
            web_sys::window()
                .and_then(|w| w.performance())
                .map(|p| p.now())
                .unwrap_or(0.0)
        }
        
        #[cfg(not(target_arch = "wasm32"))]
        {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64() * 1000.0
        }
    }

    /// Create a pre-configured network for common use cases
    #[wasm_bindgen]
    pub fn create_classifier(input_size: usize, hidden_sizes: &[usize], num_classes: usize) -> WasmNeuralNetwork {
        let mut network = Self::new();
        
        if hidden_sizes.is_empty() {
            network.add_layer(input_size, num_classes);
            return network;
        }
        
        // Input to first hidden layer
        network.add_layer(input_size, hidden_sizes[0]);
        
        // Hidden layers
        for i in 1..hidden_sizes.len() {
            network.add_layer(hidden_sizes[i - 1], hidden_sizes[i]);
        }
        
        // Last hidden to output
        network.add_layer(hidden_sizes[hidden_sizes.len() - 1], num_classes);
        
        network
    }
}

/// Streaming neural network for real-time inference
#[wasm_bindgen]
pub struct WasmStreamingNN {
    network: WasmNeuralNetwork,
    buffer: Vec<f32>,
    buffer_size: usize,
    activations: Vec<u8>,
    features: WasmFeatures,
}

#[wasm_bindgen]
impl WasmStreamingNN {
    /// Create a streaming neural network
    #[wasm_bindgen(constructor)]
    pub fn new(buffer_size: usize) -> Self {
        Self {
            network: WasmNeuralNetwork::new(),
            buffer: Vec::with_capacity(buffer_size),
            buffer_size,
            activations: Vec::new(),
            features: WasmFeatures::detect(),
        }
    }

    /// Add a layer to the streaming network
    #[wasm_bindgen]
    pub fn add_layer(&mut self, input_size: usize, output_size: usize, activation: u8) {
        self.network.add_layer(input_size, output_size);
        self.activations.push(activation);
    }

    /// Process streaming data
    #[wasm_bindgen]
    pub fn process_stream(&mut self, data: &[f32]) -> Vec<f32> {
        let mut results = Vec::new();
        
        for &value in data {
            self.buffer.push(value);
            
            if self.buffer.len() >= self.buffer_size {
                // Process full buffer
                let prediction = self.network.predict(&self.buffer, &self.activations);
                results.extend(prediction);
                
                // Slide buffer (keep last buffer_size/2 elements for overlap)
                let keep_size = self.buffer_size / 2;
                self.buffer.drain(0..self.buffer.len() - keep_size);
            }
        }
        
        results
    }

    /// Get streaming buffer status
    #[wasm_bindgen]
    pub fn buffer_status(&self) -> String {
        format!("Buffer: {}/{} ({}%)", 
               self.buffer.len(), 
               self.buffer_size,
               (self.buffer.len() * 100) / self.buffer_size)
    }
}

/// Neural network utilities and helpers
pub struct WasmNeuralUtils;

impl WasmNeuralUtils {
    /// Convert float32 array to quantized int8 with optimal scaling
    pub fn quantize_f32_to_i8(data: &[f32], dynamic_range: bool) -> (Vec<i8>, f32, i8) {
        if data.is_empty() {
            return (Vec::new(), 1.0, 0);
        }

        let (scale, offset) = if dynamic_range {
            let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let scale = (max_val - min_val) / 255.0;
            let offset = (min_val / scale) as i8;
            (scale, offset)
        } else {
            (1.0 / 127.0, 0i8)
        };

        let quantized: Vec<i8> = data.iter()
            .map(|&x| ((x / scale) as i32 - offset as i32).clamp(-128, 127) as i8)
            .collect();

        (quantized, scale, offset)
    }

    /// Dequantize int8 array back to float32
    pub fn dequantize_i8_to_f32(data: &[i8], scale: f32, offset: i8) -> Vec<f32> {
        data.iter()
            .map(|&x| (x as f32 + offset as f32) * scale)
            .collect()
    }

    /// Calculate model complexity score (for deployment decisions)
    pub fn calculate_complexity(layers: &[WasmNeuralLayer]) -> f64 {
        let total_params: usize = layers.iter()
            .map(|l| l.weights.len() + l.biases.len())
            .sum();
        
        let total_operations: usize = layers.iter()
            .map(|l| l.input_size * l.output_size * 2) // multiply-add operations
            .sum();

        // Complexity score balances parameters and operations
        (total_params as f64 * 0.6 + total_operations as f64 * 0.4) / 1000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_creation() {
        let layer = WasmNeuralLayer::new(10, 5);
        assert_eq!(layer.input_size, 10);
        assert_eq!(layer.output_size, 5);
        assert_eq!(layer.weights.len(), 50);
        assert_eq!(layer.biases.len(), 5);
    }

    #[test]
    fn test_forward_pass() {
        let layer = WasmNeuralLayer::new(4, 3);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = layer.forward(&input, 0); // ReLU activation
        
        assert_eq!(output.len(), 3);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_quantization() {
        let mut layer = WasmNeuralLayer::new(4, 2);
        layer.quantize(1, false); // INT8, fixed range
        
        assert!(layer.quantized);
        assert!((layer.quantization_scale - 1.0 / 127.0).abs() < 1e-6);
    }

    #[test]
    fn test_network_creation() {
        let mut network = WasmNeuralNetwork::new();
        network.add_layer(10, 5);
        network.add_layer(5, 2);
        
        let input = vec![1.0; 10];
        let activations = vec![0, 1]; // ReLU, Sigmoid
        let output = network.predict(&input, &activations);
        
        assert_eq!(output.len(), 2);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_streaming_network() {
        let mut streaming = WasmStreamingNN::new(8);
        streaming.add_layer(8, 4, 0);
        streaming.add_layer(4, 2, 1);
        
        let data = vec![1.0; 16];
        let results = streaming.process_stream(&data);
        
        assert!(!results.is_empty());
        assert!(results.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_activation_functions() {
        let layer = WasmNeuralLayer::new(1, 1);
        let input = vec![1.0];
        
        // Test different activations
        let relu = layer.forward(&input, 0);
        let sigmoid = layer.forward(&input, 1);
        let tanh = layer.forward(&input, 2);
        let leaky_relu = layer.forward(&input, 3);
        
        assert!(relu[0] >= 0.0);
        assert!(sigmoid[0] >= 0.0 && sigmoid[0] <= 1.0);
        assert!(tanh[0] >= -1.0 && tanh[0] <= 1.0);
        assert!(leaky_relu.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_batch_processing() {
        let layer = WasmNeuralLayer::new(3, 2);
        let inputs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 batches of 3 inputs
        let outputs = layer.batch_forward(&inputs, 2, 0);
        
        assert_eq!(outputs.len(), 4); // 2 batches * 2 outputs
        assert!(outputs.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_quantization_utils() {
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let (quantized, scale, offset) = WasmNeuralUtils::quantize_f32_to_i8(&data, true);
        let dequantized = WasmNeuralUtils::dequantize_i8_to_f32(&quantized, scale, offset);
        
        assert_eq!(quantized.len(), data.len());
        assert_eq!(dequantized.len(), data.len());
        
        // Check that quantization/dequantization preserves approximate values
        for (original, recovered) in data.iter().zip(dequantized.iter()) {
            assert!((original - recovered).abs() < 0.1);
        }
    }

    #[test]
    fn test_classifier_creation() {
        let hidden_sizes = vec![64, 32, 16];
        let network = WasmNeuralNetwork::create_classifier(128, &hidden_sizes, 10);
        
        let input = vec![1.0; 128];
        let activations = vec![0, 0, 0, 1]; // ReLU for hidden, Sigmoid for output
        let output = network.predict(&input, &activations);
        
        assert_eq!(output.len(), 10);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_memory_usage() {
        let layer = WasmNeuralLayer::new(100, 50);
        let memory = layer.memory_usage();
        
        let expected_weights = 100 * 50 * 4; // f32 = 4 bytes
        let expected_biases = 50 * 4;
        let expected_minimum = expected_weights + expected_biases;
        
        assert!(memory >= expected_minimum);
    }
}