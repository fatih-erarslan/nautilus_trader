// ReLU Activation Function Compute Shader for ruv_FANN GPU Acceleration
// Optimized for high-frequency trading neural network operations

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= arrayLength(&input_data)) {
        return;
    }
    
    // ReLU activation: max(0, x)
    output_data[index] = max(0.0, input_data[index]);
}

// Leaky ReLU variant with configurable negative slope
@compute @workgroup_size(256)
fn leaky_relu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= arrayLength(&input_data)) {
        return;
    }
    
    let negative_slope = 0.01; // Configurable parameter
    let x = input_data[index];
    
    if (x >= 0.0) {
        output_data[index] = x;
    } else {
        output_data[index] = negative_slope * x;
    }
}

// Parametric ReLU with learnable parameters
@compute @workgroup_size(256)
fn parametric_relu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= arrayLength(&input_data)) {
        return;
    }
    
    // In a real implementation, this would come from a parameter buffer
    let alpha = 0.25; // Learnable parameter
    let x = input_data[index];
    
    if (x >= 0.0) {
        output_data[index] = x;
    } else {
        output_data[index] = alpha * x;
    }
}

// ELU (Exponential Linear Unit) activation
@compute @workgroup_size(256)
fn elu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= arrayLength(&input_data)) {
        return;
    }
    
    let alpha = 1.0;
    let x = input_data[index];
    
    if (x >= 0.0) {
        output_data[index] = x;
    } else {
        output_data[index] = alpha * (exp(x) - 1.0);
    }
}

// Swish activation function (x * sigmoid(x))
@compute @workgroup_size(256)
fn swish(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= arrayLength(&input_data)) {
        return;
    }
    
    let x = input_data[index];
    let sigmoid_x = 1.0 / (1.0 + exp(-x));
    output_data[index] = x * sigmoid_x;
}