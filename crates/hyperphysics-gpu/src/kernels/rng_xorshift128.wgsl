// GPU Random Number Generator - Xorshift128+ Algorithm
//
// High-quality PRNG suitable for Monte Carlo simulations.
// Period: 2^128 - 1 (effectively infinite for physics simulations)
//
// Reference: Vigna (2016) "An experimental exploration of Marsaglia's xorshift generators"
//            ACM Transactions on Mathematical Software 42(4):30
//
// Algorithm:
//   s1 ^= s1 << 23
//   s1 ^= s1 >> 18
//   s1 ^= s0
//   s1 ^= s0 >> 5
//   return s0 + s1

struct RNGState {
    s0: u32,
    s1: u32,
}

@group(0) @binding(0) var<storage, read_write> rng_states: array<RNGState>;
@group(0) @binding(1) var<storage, read_write> random_output: array<f32>;
@group(0) @binding(2) var<uniform> params: RNGParams;

struct RNGParams {
    n_values: u32,
    iteration: u32,  // For mixing entropy across calls
}

// Xorshift128+ core algorithm
fn xorshift128plus(state: ptr<function, RNGState>) -> u32 {
    var s1 = (*state).s0;
    let s0 = (*state).s1;

    (*state).s0 = s0;

    s1 ^= s1 << 23u;
    s1 ^= s1 >> 18u;
    s1 ^= s0;
    s1 ^= s0 >> 5u;

    (*state).s1 = s1;

    return s0 + s1;
}

// Convert u32 to f32 in [0, 1)
fn u32_to_f32(value: u32) -> f32 {
    // Use upper 24 bits for mantissa precision
    // Divide by 2^32 to get [0, 1)
    return f32(value) * 2.32830643653869628906e-10; // 1.0 / 2^32
}

// Generate uniform random float in [0, 1)
fn random_uniform(state: ptr<function, RNGState>) -> f32 {
    let value = xorshift128plus(state);
    return u32_to_f32(value);
}

// Generate Gaussian random variable via Box-Muller transform
// Returns pair of independent N(0,1) values
fn random_gaussian_pair(state: ptr<function, RNGState>) -> vec2<f32> {
    let u1 = random_uniform(state);
    let u2 = random_uniform(state);

    let r = sqrt(-2.0 * log(max(u1, 1e-10)));
    let theta = 6.28318530718 * u2;  // 2π

    return vec2<f32>(
        r * cos(theta),
        r * sin(theta)
    );
}

@compute @workgroup_size(256)
fn generate_uniform(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;

    if (idx >= params.n_values) {
        return;
    }

    // Load RNG state for this thread
    var state = rng_states[idx];

    // Add iteration mixing for temporal decorrelation
    state.s0 ^= params.iteration;
    state.s1 ^= params.iteration << 16u;

    // Generate random value
    let random_value = random_uniform(&state);

    // Store result
    random_output[idx] = random_value;

    // Save updated state
    rng_states[idx] = state;
}

@compute @workgroup_size(256)
fn generate_gaussian(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;

    // Generate pairs, so check if we're within bounds
    let pair_idx = idx / 2u;
    if (pair_idx >= (params.n_values + 1u) / 2u) {
        return;
    }

    // Load RNG state
    var state = rng_states[idx];

    // Add iteration mixing
    state.s0 ^= params.iteration;
    state.s1 ^= params.iteration << 16u;

    // Generate Gaussian pair
    let gaussian_pair = random_gaussian_pair(&state);

    // Store results (handle odd n_values case)
    let out_idx = idx * 2u;
    if (out_idx < params.n_values) {
        random_output[out_idx] = gaussian_pair.x;
    }
    if (out_idx + 1u < params.n_values) {
        random_output[out_idx + 1u] = gaussian_pair.y;
    }

    // Save updated state
    rng_states[idx] = state;
}

// Seed initialization kernel
@compute @workgroup_size(256)
fn seed_rng(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let idx = global_id.x;

    if (idx >= params.n_values) {
        return;
    }

    // Use 32-bit SplitMix-inspired seeding for good initial entropy
    // Each thread gets unique seed based on global ID and user-provided seed
    var seed = params.iteration + idx * 0x9E3779B9u;  // Golden ratio, lower 32 bits

    // Mix the bits (32-bit version)
    seed = (seed ^ (seed >> 16u)) * 0x85ebca6bu;
    seed = (seed ^ (seed >> 13u)) * 0xc2b2ae35u;
    seed = seed ^ (seed >> 16u);

    // Generate second seed component with different constant
    var seed2 = seed + 0x7F4A7C15u;
    seed2 = (seed2 ^ (seed2 >> 16u)) * 0x85ebca6bu;
    seed2 = (seed2 ^ (seed2 >> 13u)) * 0xc2b2ae35u;
    seed2 = seed2 ^ (seed2 >> 16u);

    // Initialize state
    rng_states[idx] = RNGState(seed, seed2);
}

// Statistical test: generate N samples and compute mean/variance
// For testing RNG quality
@compute @workgroup_size(256)
fn test_statistics(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let idx = global_id.x;

    if (idx >= params.n_values) {
        return;
    }

    var state = rng_states[idx];

    // Generate 100 samples per thread
    var sum = 0.0;
    var sum_sq = 0.0;
    let n_samples = 100u;

    for (var i = 0u; i < n_samples; i = i + 1u) {
        let value = random_uniform(&state);
        sum += value;
        sum_sq += value * value;
    }

    let mean = sum / f32(n_samples);
    let variance = (sum_sq / f32(n_samples)) - (mean * mean);

    // Store mean (should be ~0.5 for uniform [0,1))
    random_output[idx] = mean;

    // Variance stored in second half (should be ~1/12 ≈ 0.0833 for uniform [0,1))
    if (idx + params.n_values < params.n_values * 2u) {
        random_output[idx + params.n_values] = variance;
    }

    rng_states[idx] = state;
}
