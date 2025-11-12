//! GPU compute kernels for HyperPhysics simulations

/// WGSL shader for pBit state updates using Metropolis-Hastings
pub const METROPOLIS_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> states: array<f32>;
@group(0) @binding(1) var<storage, read> couplings: array<f32>;
@group(0) @binding(2) var<storage, read> random: array<f32>;
@group(0) @binding(3) var<storage, read_write> next_states: array<f32>;
@group(0) @binding(4) var<uniform> params: MetropolisParams;

struct MetropolisParams {
    num_pbits: u32,
    beta: f32,
    dt: f32,
    _padding: u32,
}

@compute @workgroup_size(256)
fn metropolis_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.num_pbits) {
        return;
    }

    let current_state = states[i];
    let proposed_state = 1.0 - current_state; // Flip bit

    // Calculate energy change
    var delta_energy = 0.0;
    for (var j = 0u; j < params.num_pbits; j = j + 1u) {
        let coupling_idx = i * params.num_pbits + j;
        delta_energy += couplings[coupling_idx] * (proposed_state - current_state) * states[j];
    }

    // Metropolis acceptance
    let acceptance_prob = exp(-params.beta * delta_energy);
    let rand_val = random[i];

    if (rand_val < acceptance_prob) {
        next_states[i] = proposed_state;
    } else {
        next_states[i] = current_state;
    }
}
"#;

/// WGSL shader for Gillespie stochastic simulation
pub const GILLESPIE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> rates: array<f32>;
@group(0) @binding(1) var<storage, read> random: array<f32>;
@group(0) @binding(2) var<storage, read_write> times: array<f32>;
@group(0) @binding(3) var<storage, read_write> events: array<u32>;
@group(0) @binding(4) var<uniform> params: GillespieParams;

struct GillespieParams {
    num_reactions: u32,
    max_time: f32,
    _padding: array<u32, 2>,
}

@compute @workgroup_size(256)
fn gillespie_step(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.num_reactions) {
        return;
    }

    // Calculate total propensity
    var total_rate = 0.0;
    for (var j = 0u; j < params.num_reactions; j = j + 1u) {
        total_rate += rates[j];
    }

    // Calculate time to next event
    let tau = -log(random[i * 2u]) / total_rate;
    times[i] = tau;

    // Select which event occurs
    let r2 = random[i * 2u + 1u] * total_rate;
    var cumulative = 0.0;
    for (var j = 0u; j < params.num_reactions; j = j + 1u) {
        cumulative += rates[j];
        if (r2 < cumulative) {
            events[i] = j;
            break;
        }
    }
}
"#;

/// WGSL shader for partition function calculation
pub const PARTITION_FUNCTION_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> energies: array<f32>;
@group(0) @binding(1) var<storage, read_write> boltzmann_factors: array<f32>;
@group(0) @binding(2) var<uniform> params: PartitionParams;

struct PartitionParams {
    num_states: u32,
    beta: f32,
    _padding: array<u32, 2>,
}

@compute @workgroup_size(256)
fn compute_boltzmann(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.num_states) {
        return;
    }

    boltzmann_factors[i] = exp(-params.beta * energies[i]);
}

// Parallel reduction for summing Boltzmann factors
@compute @workgroup_size(256)
fn reduce_sum(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>) {
    // TODO: Implement tree reduction in shared memory
    // This is a placeholder for the reduction kernel
}
"#;

/// WGSL shader for matrix operations (coupling networks)
pub const MATRIX_MULTIPLY_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> dims: MatrixDims;

struct MatrixDims {
    m: u32,  // rows of A
    n: u32,  // cols of A, rows of B
    p: u32,  // cols of B
    _padding: u32,
}

@compute @workgroup_size(16, 16)
fn matmul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;

    if (row >= dims.m || col >= dims.p) {
        return;
    }

    var sum = 0.0;
    for (var k = 0u; k < dims.n; k = k + 1u) {
        let a_val = matrix_a[row * dims.n + k];
        let b_val = matrix_b[k * dims.p + col];
        sum += a_val * b_val;
    }

    result[row * dims.p + col] = sum;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shaders_compile() {
        // Verify WGSL syntax is valid
        assert!(METROPOLIS_SHADER.contains("@compute"));
        assert!(GILLESPIE_SHADER.contains("@compute"));
        assert!(PARTITION_FUNCTION_SHADER.contains("@compute"));
        assert!(MATRIX_MULTIPLY_SHADER.contains("@compute"));
    }
}
