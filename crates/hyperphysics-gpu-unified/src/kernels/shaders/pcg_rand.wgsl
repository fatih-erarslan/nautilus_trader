// PCG-inspired Random Number Generator
// Statistically superior to xorshift, proper [0,1) range

struct RandomState {
    seeds: array<u32, 4>,
}

@group(0) @binding(0) var<storage, read_write> state: RandomState;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

// PCG hash function - better statistical properties than xorshift
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Generate uniform random in [0, 1)
// CRITICAL: Use 2^32 (4294967296) not 2^32-1 for proper range
fn rand_uniform(seed: u32) -> f32 {
    return f32(pcg_hash(seed)) / 4294967296.0;
}

// Box-Muller transform for normal distribution
// CRITICAL: Clamp u1 to prevent log(0) = -Inf causing NaN
fn rand_normal(seed1: u32, seed2: u32) -> f32 {
    let u1 = max(rand_uniform(seed1), 1e-10);
    let u2 = rand_uniform(seed2);
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let array_len = arrayLength(&output);

    if idx >= array_len {
        return;
    }

    // Use index-based seeding for reproducibility
    let seed = state.seeds[idx % 4u] ^ (idx * 1103515245u + 12345u);
    output[idx] = rand_uniform(seed);
}
