// pBit Metropolis-Hastings Compute Shader
// 
// Implements checkerboard parallel updates for Ising model dynamics.
// Each workgroup processes a tile of pBits independently.

// Constants
const WORKGROUP_SIZE: u32 = 256u;
const BITS_PER_WORD: u32 = 32u;

// Coupling entry in CSR format
struct CouplingEntry {
    neighbor_idx: u32,
    coupling_strength: f32,
}

// Simulation parameters
struct Params {
    num_pbits: u32,
    phase: u32,        // 0 = red (even), 1 = black (odd)
    beta: f32,         // 1/T
    seed: u32,
}

// Buffers
@group(0) @binding(0) var<storage, read_write> states: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> biases: array<f32>;
@group(0) @binding(2) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(3) var<storage, read> entries: array<CouplingEntry>;
@group(0) @binding(4) var<uniform> params: Params;

// xorshift32 RNG
fn xorshift32(state: ptr<function, u32>) -> u32 {
    var x = *state;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    *state = x;
    return x;
}

// Random float in [0, 1)
fn rand_f32(state: ptr<function, u32>) -> f32 {
    return f32(xorshift32(state)) / 4294967296.0;
}

// Get spin from packed state: 0 → -1.0, 1 → +1.0
fn get_spin(idx: u32) -> f32 {
    let word_idx = idx / BITS_PER_WORD;
    let bit_idx = idx % BITS_PER_WORD;
    let word = atomicLoad(&states[word_idx]);
    let bit = (word >> bit_idx) & 1u;
    return select(-1.0, 1.0, bit == 1u);
}

// Flip a bit atomically
fn flip_bit(idx: u32) {
    let word_idx = idx / BITS_PER_WORD;
    let bit_idx = idx % BITS_PER_WORD;
    let mask = 1u << bit_idx;
    atomicXor(&states[word_idx], mask);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn metropolis_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Calculate pBit index for checkerboard phase
    let idx = gid.x * 2u + params.phase;
    
    if (idx >= params.num_pbits) {
        return;
    }
    
    // Initialize per-thread RNG
    var rng_state = params.seed + gid.x * 2654435761u;
    // Warm up RNG
    for (var i = 0u; i < 4u; i++) {
        rng_state = xorshift32(&rng_state);
    }
    
    // Get current spin
    let spin_i = get_spin(idx);
    
    // Calculate effective field h_i = bias + Σ_j J_ij * s_j
    var h = biases[idx];
    let start = row_ptr[idx];
    let end = row_ptr[idx + 1u];
    
    for (var e = start; e < end; e++) {
        let entry = entries[e];
        let spin_j = get_spin(entry.neighbor_idx);
        h += entry.coupling_strength * spin_j;
    }
    
    // Energy change for flip: ΔE = 2 * s_i * h_i
    let delta_e = 2.0 * spin_i * h;
    
    // Metropolis criterion
    var accept = delta_e <= 0.0;
    if (!accept) {
        let prob = exp(-params.beta * delta_e);
        accept = rand_f32(&rng_state) < prob;
    }
    
    if (accept) {
        flip_bit(idx);
    }
}

// Magnetization reduction kernel
@group(0) @binding(0) var<storage, read> states_readonly: array<u32>;
@group(0) @binding(1) var<storage, read_write> partial_sums: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> reduce_params: vec2<u32>; // [num_words, 0]

var<workgroup> local_sum: array<u32, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn count_ones_kernel(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let num_words = reduce_params.x;
    
    // Count ones in this word
    var count = 0u;
    if (gid.x < num_words) {
        count = countOneBits(states_readonly[gid.x]);
    }
    
    // Store in shared memory
    local_sum[lid.x] = count;
    workgroupBarrier();
    
    // Reduction within workgroup
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            local_sum[lid.x] += local_sum[lid.x + stride];
        }
        workgroupBarrier();
    }
    
    // First thread writes result
    if (lid.x == 0u) {
        atomicAdd(&partial_sums[wid.x], local_sum[0]);
    }
}

// Energy calculation kernel
struct EnergyParams {
    num_pbits: u32,
    num_edges: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> energy_states: array<u32>;
@group(0) @binding(1) var<storage, read> energy_biases: array<f32>;
@group(0) @binding(2) var<storage, read> energy_row_ptr: array<u32>;
@group(0) @binding(3) var<storage, read> energy_entries: array<CouplingEntry>;
@group(0) @binding(4) var<storage, read_write> energy_output: array<atomic<i32>>;
@group(0) @binding(5) var<uniform> energy_params: EnergyParams;

fn get_spin_readonly(idx: u32) -> f32 {
    let word_idx = idx / BITS_PER_WORD;
    let bit_idx = idx % BITS_PER_WORD;
    let word = energy_states[word_idx];
    let bit = (word >> bit_idx) & 1u;
    return select(-1.0, 1.0, bit == 1u);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn energy_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    
    if (idx >= energy_params.num_pbits) {
        return;
    }
    
    let spin_i = get_spin_readonly(idx);
    var local_energy = 0.0;
    
    // Bias term: -h_i * s_i
    local_energy -= energy_biases[idx] * spin_i;
    
    // Coupling term: -Σ_j J_ij * s_i * s_j (only count i < j to avoid double counting)
    let start = energy_row_ptr[idx];
    let end = energy_row_ptr[idx + 1u];
    
    for (var e = start; e < end; e++) {
        let entry = energy_entries[e];
        if (entry.neighbor_idx > idx) {  // Only count edge once
            let spin_j = get_spin_readonly(entry.neighbor_idx);
            local_energy -= entry.coupling_strength * spin_i * spin_j;
        }
    }
    
    // Atomic add (using fixed-point: multiply by 1000 and cast to i32)
    let energy_fixed = i32(local_energy * 1000.0);
    atomicAdd(&energy_output[0], energy_fixed);
}
