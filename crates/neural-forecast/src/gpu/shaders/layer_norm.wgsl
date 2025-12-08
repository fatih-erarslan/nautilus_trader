// Layer normalization shader
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let size = arrayLength(&input);
    
    if (index >= size) {
        return;
    }
    
    // Compute mean
    var mean = 0.0;
    for (var i = 0u; i < size; i++) {
        mean += input[i];
    }
    mean /= f32(size);
    
    // Compute variance
    var variance = 0.0;
    for (var i = 0u; i < size; i++) {
        let diff = input[i] - mean;
        variance += diff * diff;
    }
    variance /= f32(size);
    
    // Normalize
    let eps = 1e-6;
    let std_dev = sqrt(variance + eps);
    let normalized = (input[index] - mean) / std_dev;
    
    // Apply scale and shift
    output[index] = gamma[index] * normalized + beta[index];
}