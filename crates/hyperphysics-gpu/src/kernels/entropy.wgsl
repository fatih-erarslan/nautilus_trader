// Entropy Calculation Kernel (Shannon Entropy)
//
// Computes S = -Σ_i [p_i ln(p_i) + (1-p_i) ln(1-p_i)]
// Uses parallel reduction for efficient global sum.

struct PBit {
    state: u32,
    bias: f32,
    coupling_offset: u32,
    coupling_count: u32,
}

@group(0) @binding(0) var<storage, read> pbits: array<PBit>;
@group(0) @binding(1) var<storage, read_write> entropy_partial: array<f32>;
@group(0) @binding(2) var<uniform> params: EntropyParams;

struct EntropyParams {
    n_pbits: u32,
    temperature: f32,
    n_workgroups: u32,
}

var<workgroup> shared_entropy: array<f32, 256>;

// Safe logarithm avoiding log(0)
fn safe_log(x: f32) -> f32 {
    let epsilon = 1e-10;
    return log(max(x, epsilon));
}

// Shannon entropy contribution for single pBit
fn pbit_entropy(probability: f32) -> f32 {
    let p = clamp(probability, 0.0, 1.0);
    let q = 1.0 - p;

    var entropy = 0.0;
    if (p > 1e-10) {
        entropy -= p * safe_log(p);
    }
    if (q > 1e-10) {
        entropy -= q * safe_log(q);
    }

    return entropy;
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let global_idx = global_id.x;
    let local_idx = local_id.x;

    var entropy = 0.0;

    if (global_idx < params.n_pbits) {
        // Compute probability from current state
        // For now, use empirical average (could compute from Boltzmann distribution)
        let state = f32(pbits[global_idx].state);

        // Simple approach: treat state as probability estimate
        // More sophisticated: compute p from Boltzmann weights
        let probability = state; // TODO: Improve with proper Boltzmann calculation

        entropy = pbit_entropy(probability);
    }

    // Store in shared memory for reduction
    shared_entropy[local_idx] = entropy;
    workgroupBarrier();

    // Parallel reduction
    var stride = 128u;
    while (stride > 0u) {
        if (local_idx < stride && local_idx + stride < 256u) {
            shared_entropy[local_idx] += shared_entropy[local_idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Write workgroup result
    if (local_idx == 0u) {
        entropy_partial[workgroup_id.x] = shared_entropy[0];
    }
}

// Final reduction pass (same as energy kernel)
@compute @workgroup_size(256)
fn reduce_final(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let idx = global_id.x;
    let local_idx = local_id.x;

    var value = 0.0;
    if (idx < params.n_workgroups) {
        value = entropy_partial[idx];
    }

    shared_entropy[local_idx] = value;
    workgroupBarrier();

    var stride = 128u;
    while (stride > 0u) {
        if (local_idx < stride) {
            shared_entropy[local_idx] += shared_entropy[local_idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (local_idx == 0u) {
        entropy_partial[0] = shared_entropy[0];
    }
}
