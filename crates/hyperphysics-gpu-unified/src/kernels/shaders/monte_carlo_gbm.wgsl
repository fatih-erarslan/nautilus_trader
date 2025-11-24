// Monte Carlo Geometric Brownian Motion Kernel
// For financial derivatives pricing and risk simulation

struct MCParams {
    spot_price: f32,       // S0: Initial asset price
    risk_free_rate: f32,   // r: Risk-free interest rate
    volatility: f32,       // σ: Volatility (annualized)
    time_to_maturity: f32, // T: Time to maturity (years)
    num_steps: u32,        // Number of time steps
    strike_price: f32,     // K: Strike price (for options)
    _padding: vec2<f32>,   // Alignment padding
}

@group(0) @binding(0) var<uniform> params: MCParams;
@group(0) @binding(1) var<storage, read> seeds: array<u32>;
@group(0) @binding(2) var<storage, read_write> paths: array<f32>;    // Final prices
@group(0) @binding(3) var<storage, read_write> payoffs: array<f32>; // Option payoffs

// PCG hash for random numbers
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_uniform(seed: u32) -> f32 {
    return f32(pcg_hash(seed)) / 4294967296.0;
}

// Box-Muller transform for normal distribution
fn rand_normal(seed1: u32, seed2: u32) -> f32 {
    let u1 = max(rand_uniform(seed1), 1e-10);
    let u2 = rand_uniform(seed2);
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let path_idx = gid.x;
    let num_paths = arrayLength(&paths);

    if path_idx >= num_paths {
        return;
    }

    // Time step size
    let dt = params.time_to_maturity / f32(params.num_steps);
    let sqrt_dt = sqrt(dt);

    // Drift and diffusion coefficients
    // Using risk-neutral measure: drift = (r - σ²/2)
    let drift = (params.risk_free_rate - 0.5 * params.volatility * params.volatility) * dt;
    let diffusion = params.volatility * sqrt_dt;

    // Initialize price
    var price = params.spot_price;

    // Get base seed for this path
    let base_seed = seeds[path_idx % arrayLength(&seeds)];

    // Simulate path using GBM: dS = S(r*dt + σ*dW)
    for (var step = 0u; step < params.num_steps; step++) {
        // Generate two seeds for Box-Muller
        let seed1 = pcg_hash(base_seed ^ (step * 2u) ^ (path_idx * 65537u));
        let seed2 = pcg_hash(base_seed ^ (step * 2u + 1u) ^ (path_idx * 65537u));

        let z = rand_normal(seed1, seed2);

        // Log-normal evolution (exact discretization)
        price = price * exp(drift + diffusion * z);
    }

    // Store final price
    paths[path_idx] = price;

    // Calculate call option payoff: max(S_T - K, 0)
    payoffs[path_idx] = max(price - params.strike_price, 0.0);
}
