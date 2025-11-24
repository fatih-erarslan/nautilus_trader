// Probabilistic Bit (pBit) Sampling Kernel
// Implements Boltzmann distribution for quantum-inspired optimization

struct PBitParams {
    temperature: f32,
    coupling_strength: f32,
    bias: f32,
    time_step: f32,
}

@group(0) @binding(0) var<uniform> params: PBitParams;
@group(0) @binding(1) var<storage, read> spins: array<f32>;        // Current spin states [-1, 1]
@group(0) @binding(2) var<storage, read> couplings: array<f32>;    // J_ij coupling matrix (flattened)
@group(0) @binding(3) var<storage, read> seeds: array<u32>;        // Random seeds
@group(0) @binding(4) var<storage, read_write> output: array<f32>; // New spin states

// PCG hash for random numbers
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_uniform(seed: u32) -> f32 {
    return f32(pcg_hash(seed)) / 4294967296.0;
}

// Sigmoid activation with temperature
fn sigmoid(x: f32, temp: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x / max(temp, 1e-6)));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n_spins = arrayLength(&spins);

    if idx >= n_spins {
        return;
    }

    // Calculate local field from couplings
    var local_field = params.bias;
    for (var j = 0u; j < n_spins; j++) {
        if j != idx {
            let coupling_idx = idx * n_spins + j;
            local_field += params.coupling_strength * couplings[coupling_idx] * spins[j];
        }
    }

    // Boltzmann probability of spin-up
    let prob_up = sigmoid(local_field, params.temperature);

    // Stochastic update with proper seeding
    let rand_val = rand_uniform(seeds[idx] ^ (idx * 1103515245u));

    // Output new spin state
    if rand_val < prob_up {
        output[idx] = 1.0;
    } else {
        output[idx] = -1.0;
    }
}
