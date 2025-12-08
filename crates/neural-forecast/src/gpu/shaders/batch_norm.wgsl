// Batch normalization shader
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> running_mean: array<f32>;
@group(0) @binding(2) var<storage, read> running_var: array<f32>;
@group(0) @binding(3) var<storage, read> gamma: array<f32>;
@group(0) @binding(4) var<storage, read> beta: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= arrayLength(&output)) {
        return;
    }
    
    let eps = 1e-6;
    let normalized = (input[index] - running_mean[index]) / sqrt(running_var[index] + eps);
    output[index] = gamma[index] * normalized + beta[index];
}