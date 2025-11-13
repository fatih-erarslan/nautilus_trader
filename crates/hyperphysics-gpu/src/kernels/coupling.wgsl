// Coupling network computation shader
// Implements J_ij = J_0 * exp(-d_H(i,j) / λ) for exponential decay coupling

@group(0) @binding(0) var<storage, read> positions: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read> coupling_params: CouplingParams;
@group(0) @binding(2) var<storage, read_write> coupling_matrix: array<f32>; // flattened N×N matrix

struct CouplingParams {
    j0: f32,           // Base coupling strength
    lambda: f32,       // Decay length scale
    cutoff_distance: f32, // Maximum coupling distance
    num_nodes: u32,    // Number of nodes
}

// Workgroup size for 2D coupling matrix computation
@compute @workgroup_size(8, 8, 1)
fn compute_coupling_matrix(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;
    let n = coupling_params.num_nodes;
    
    // Bounds check
    if (i >= n || j >= n) {
        return;
    }
    
    // Matrix index in flattened array
    let matrix_index = i * n + j;
    
    // Self-coupling is zero
    if (i == j) {
        coupling_matrix[matrix_index] = 0.0;
        return;
    }
    
    let p = positions[i];
    let q = positions[j];
    
    // Calculate hyperbolic distance
    let distance = hyperbolic_distance(p, q);
    
    // Apply cutoff
    if (distance > coupling_params.cutoff_distance) {
        coupling_matrix[matrix_index] = 0.0;
        return;
    }
    
    // Exponential decay coupling: J_ij = J_0 * exp(-d_H(i,j) / λ)
    let coupling_strength = coupling_params.j0 * exp(-distance / coupling_params.lambda);
    coupling_matrix[matrix_index] = coupling_strength;
}

// Helper function for hyperbolic distance calculation
fn hyperbolic_distance(p: vec3<f32>, q: vec3<f32>) -> f32 {
    let p_norm_sq = dot(p, p);
    let q_norm_sq = dot(q, q);
    
    let eps = 1e-10f;
    let p_clamped_norm_sq = min(p_norm_sq, 1.0 - eps);
    let q_clamped_norm_sq = min(q_norm_sq, 1.0 - eps);
    
    let diff = p - q;
    let diff_norm_sq = dot(diff, diff);
    
    let denominator = (1.0 - p_clamped_norm_sq) * (1.0 - q_clamped_norm_sq);
    let argument = 1.0 + 2.0 * diff_norm_sq / denominator;
    let safe_argument = max(argument, 1.0 + eps);
    
    return log(safe_argument + sqrt(safe_argument * safe_argument - 1.0));
}

// Sparse coupling computation (only for neighbors within cutoff)
@group(1) @binding(0) var<storage, read> neighbor_pairs: array<vec2<u32>>;
@group(1) @binding(1) var<storage, read> neighbor_distances: array<f32>;
@group(1) @binding(2) var<storage, read_write> sparse_couplings: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn compute_sparse_couplings(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= arrayLength(&neighbor_pairs)) {
        return;
    }
    
    let distance = neighbor_distances[index];
    
    // Apply cutoff
    if (distance > coupling_params.cutoff_distance) {
        sparse_couplings[index] = 0.0;
        return;
    }
    
    // Exponential decay coupling
    let coupling_strength = coupling_params.j0 * exp(-distance / coupling_params.lambda);
    sparse_couplings[index] = coupling_strength;
}

// Effective field calculation: h_eff,i = bias_i + Σ_j J_ij * s_j
@group(2) @binding(0) var<storage, read> pbit_states: array<f32>; // 0.0 or 1.0
@group(2) @binding(1) var<storage, read> biases: array<f32>;
@group(2) @binding(2) var<storage, read> coupling_matrix_full: array<f32>; // N×N matrix
@group(2) @binding(3) var<storage, read_write> effective_fields: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn compute_effective_fields(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let n = coupling_params.num_nodes;
    
    if (i >= n) {
        return;
    }
    
    var h_eff = biases[i];
    
    // Sum coupling contributions: Σ_j J_ij * s_j
    for (var j = 0u; j < n; j = j + 1u) {
        if (i != j) {
            let coupling = coupling_matrix_full[i * n + j];
            let state = pbit_states[j];
            h_eff = h_eff + coupling * state;
        }
    }
    
    effective_fields[i] = h_eff;
}

// Transition rate calculation: r_i = ν * |tanh(h_eff,i / (2T))|
@group(3) @binding(0) var<storage, read> effective_fields_input: array<f32>;
@group(3) @binding(1) var<storage, read> rate_params: RateParams;
@group(3) @binding(2) var<storage, read_write> transition_rates: array<f32>;

struct RateParams {
    nu: f32,        // Attempt frequency
    temperature: f32, // Temperature T
}

@compute @workgroup_size(64, 1, 1)
fn compute_transition_rates(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    
    if (i >= arrayLength(&effective_fields_input)) {
        return;
    }
    
    let h_eff = effective_fields_input[i];
    let argument = h_eff / (2.0 * rate_params.temperature);
    
    // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    let exp_2x = exp(2.0 * argument);
    let tanh_val = (exp_2x - 1.0) / (exp_2x + 1.0);
    
    // Transition rate: r_i = ν * |tanh(h_eff,i / (2T))|
    let rate = rate_params.nu * abs(tanh_val);
    transition_rates[i] = rate;
}

// Cumulative rate calculation for Gillespie algorithm
@group(4) @binding(0) var<storage, read> rates: array<f32>;
@group(4) @binding(1) var<storage, read_write> cumulative_rates: array<f32>;
@group(4) @binding(2) var<storage, read_write> total_rate: array<f32>; // single element

@compute @workgroup_size(1, 1, 1)
fn compute_cumulative_rates(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = arrayLength(&rates);
    
    if (n == 0u) {
        total_rate[0] = 0.0;
        return;
    }
    
    var cumulative = 0.0f;
    
    for (var i = 0u; i < n; i = i + 1u) {
        cumulative = cumulative + rates[i];
        cumulative_rates[i] = cumulative;
    }
    
    total_rate[0] = cumulative;
}
