// Attention mechanism shader
@group(0) @binding(0) var<storage, read> queries: array<f32>;
@group(0) @binding(1) var<storage, read> keys: array<f32>;
@group(0) @binding(2) var<storage, read> values: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<storage, read_write> attention_weights: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let seq_len = arrayLength(&queries) / 64u; // Assuming 64-dim embeddings
    
    if (index >= seq_len) {
        return;
    }
    
    // Compute attention scores
    for (var i = 0u; i < seq_len; i++) {
        var score = 0.0;
        for (var d = 0u; d < 64u; d++) {
            score += queries[index * 64u + d] * keys[i * 64u + d];
        }
        attention_weights[index * seq_len + i] = score / 8.0; // Scale by sqrt(d_k)
    }
    
    // Apply softmax to attention weights
    var max_score = attention_weights[index * seq_len];
    for (var i = 1u; i < seq_len; i++) {
        max_score = max(max_score, attention_weights[index * seq_len + i]);
    }
    
    var sum_exp = 0.0;
    for (var i = 0u; i < seq_len; i++) {
        let exp_val = exp(attention_weights[index * seq_len + i] - max_score);
        attention_weights[index * seq_len + i] = exp_val;
        sum_exp += exp_val;
    }
    
    // Normalize and compute output
    for (var i = 0u; i < seq_len; i++) {
        attention_weights[index * seq_len + i] /= sum_exp;
    }
    
    // Compute weighted sum of values
    for (var d = 0u; d < 64u; d++) {
        var weighted_sum = 0.0;
        for (var i = 0u; i < seq_len; i++) {
            weighted_sum += attention_weights[index * seq_len + i] * values[i * 64u + d];
        }
        output[index * 64u + d] = weighted_sum;
    }
}