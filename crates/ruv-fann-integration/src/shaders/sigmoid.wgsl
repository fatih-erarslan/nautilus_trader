// Sigmoid Activation Function Compute Shader for ruv_FANN GPU Acceleration
// Optimized for high-frequency trading neural network operations

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= arrayLength(&input_data)) {
        return;
    }
    
    let x = input_data[index];
    
    // Numerically stable sigmoid computation
    // For large positive values, sigmoid(x) approaches 1
    // For large negative values, sigmoid(x) approaches 0
    
    if (x > 50.0) {
        output_data[index] = 1.0;
    } else if (x < -50.0) {
        output_data[index] = 0.0;
    } else if (x >= 0.0) {
        // For positive x: sigmoid(x) = 1 / (1 + exp(-x))
        let exp_neg_x = exp(-x);
        output_data[index] = 1.0 / (1.0 + exp_neg_x);
    } else {
        // For negative x: sigmoid(x) = exp(x) / (1 + exp(x))
        // This avoids overflow in exp(-x) when x is very negative
        let exp_x = exp(x);
        output_data[index] = exp_x / (1.0 + exp_x);
    }
}

// Fast sigmoid approximation using rational function
@compute @workgroup_size(256)
fn sigmoid_fast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= arrayLength(&input_data)) {
        return;
    }
    
    let x = input_data[index];
    
    // Fast approximation: sigmoid(x) ≈ 0.5 + 0.25 * x / (1 + |x|)
    // This is much faster but less accurate
    if (abs(x) > 10.0) {
        output_data[index] = select(0.0, 1.0, x > 0.0);
    } else {
        output_data[index] = 0.5 + 0.25 * x / (1.0 + abs(x));
    }
}

// High precision sigmoid with additional numerical stability
@compute @workgroup_size(256)
fn sigmoid_precise(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= arrayLength(&input_data)) {
        return;
    }
    
    let x = input_data[index];
    
    // Handle extreme values
    if (x > 700.0) {
        output_data[index] = 1.0;
    } else if (x < -700.0) {
        output_data[index] = 0.0;
    } else {
        // Use the symmetric property for better numerical stability
        if (x >= 0.0) {
            let exp_neg_x = exp(-x);
            output_data[index] = 1.0 / (1.0 + exp_neg_x);
        } else {
            let exp_x = exp(x);
            output_data[index] = exp_x / (1.0 + exp_x);
        }
    }
}

// Sigmoid derivative for backpropagation
@compute @workgroup_size(256)
fn sigmoid_derivative(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= arrayLength(&input_data)) {
        return;
    }
    
    let sigmoid_x = input_data[index]; // Assuming input is already sigmoid(x)
    // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    output_data[index] = sigmoid_x * (1.0 - sigmoid_x);
}

// Log-sigmoid for numerical stability in loss computations
@compute @workgroup_size(256)
fn log_sigmoid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= arrayLength(&input_data)) {
        return;
    }
    
    let x = input_data[index];
    
    // log(sigmoid(x)) = log(1 / (1 + exp(-x))) = -log(1 + exp(-x))
    // For numerical stability, use different formulations based on sign
    
    if (x > 50.0) {
        output_data[index] = 0.0; // log(1) = 0
    } else if (x < -50.0) {
        output_data[index] = x; // log(exp(x)) = x when x is very negative
    } else if (x >= 0.0) {
        output_data[index] = -log(1.0 + exp(-x));
    } else {
        output_data[index] = x - log(1.0 + exp(x));
    }
}

// Softplus function: softplus(x) = log(1 + exp(x))
@compute @workgroup_size(256)
fn softplus(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= arrayLength(&input_data)) {
        return;
    }
    
    let x = input_data[index];
    
    // Numerically stable softplus
    if (x > 50.0) {
        output_data[index] = x; // For large x, softplus(x) ≈ x
    } else if (x < -50.0) {
        output_data[index] = 0.0; // For very negative x, softplus(x) ≈ 0
    } else {
        output_data[index] = log(1.0 + exp(x));
    }
}

// Mish activation function: mish(x) = x * tanh(softplus(x))
@compute @workgroup_size(256)
fn mish(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= arrayLength(&input_data)) {
        return;
    }
    
    let x = input_data[index];
    
    // Compute softplus(x)
    let softplus_x = select(log(1.0 + exp(x)), x, x > 50.0);
    
    // Compute tanh(softplus(x))
    let exp_2_softplus = exp(2.0 * softplus_x);
    let tanh_softplus = (exp_2_softplus - 1.0) / (exp_2_softplus + 1.0);
    
    output_data[index] = x * tanh_softplus;
}