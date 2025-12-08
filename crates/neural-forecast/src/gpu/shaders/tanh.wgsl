// Tanh activation shader
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    let x = input[index];
    let exp_pos = exp(x);
    let exp_neg = exp(-x);
    output[index] = (exp_pos - exp_neg) / (exp_pos + exp_neg);
}