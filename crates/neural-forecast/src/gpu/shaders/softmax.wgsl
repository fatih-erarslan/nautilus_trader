// Softmax activation shader
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read_write> temp: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let size = arrayLength(&input);
    
    if (index >= size) {
        return;
    }
    
    // Find maximum for numerical stability
    var max_val = input[0];
    for (var i = 1u; i < size; i++) {
        max_val = max(max_val, input[i]);
    }
    
    // Compute exp(x - max)
    temp[index] = exp(input[index] - max_val);
    
    // Compute sum
    var sum = 0.0;
    for (var i = 0u; i < size; i++) {
        sum += temp[i];
    }
    
    // Normalize
    output[index] = temp[index] / sum;
}