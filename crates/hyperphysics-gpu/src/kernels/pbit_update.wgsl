// pBit State Update Kernel
//
// Implements Gillespie-style stochastic state evolution for probabilistic bits.
// Each workgroup processes a chunk of pBits in parallel.

struct PBit {
    state: u32,           // Current state (0 or 1)
    bias: f32,            // External bias field
    coupling_offset: u32, // Offset into coupling array
    coupling_count: u32,  // Number of coupled neighbors
}

struct Coupling {
    target_idx: u32,      // Index of coupled pBit
    strength: f32,        // Coupling strength J_ij
}

@group(0) @binding(0) var<storage, read> pbits_in: array<PBit>;
@group(0) @binding(1) var<storage, read_write> pbits_out: array<PBit>;
@group(0) @binding(2) var<storage, read> couplings: array<Coupling>;
@group(0) @binding(3) var<storage, read> random_values: array<f32>;
@group(0) @binding(4) var<uniform> params: SimParams;

struct SimParams {
    temperature: f32,
    dt: f32,
    n_pbits: u32,
    seed: u32,
}

// Sigmoid activation function
fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

// Calculate effective field on pBit i
fn effective_field(pbit_idx: u32) -> f32 {
    let pbit = pbits_in[pbit_idx];
    var field = pbit.bias;

    // Sum coupling contributions from neighbors
    let coupling_start = pbit.coupling_offset;
    let coupling_end = coupling_start + pbit.coupling_count;

    for (var i = coupling_start; i < coupling_end; i = i + 1u) {
        let coupling = couplings[i];
        let neighbor_state = f32(pbits_in[coupling.target_idx].state);

        // Convert {0,1} → {-1,+1} for Ising model
        let spin = 2.0 * neighbor_state - 1.0;
        field += coupling.strength * spin;
    }

    return field;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.n_pbits) {
        return;
    }

    // Calculate effective field
    let h_eff = effective_field(idx);

    // Compute flip probability via sigmoid
    let beta = 1.0 / params.temperature;
    let p_flip = sigmoid(beta * h_eff);

    // Stochastic state update
    let random = random_values[idx];
    let new_state = select(0u, 1u, random < p_flip);

    // Write updated state
    var pbit_out = pbits_in[idx];
    pbit_out.state = new_state;
    pbits_out[idx] = pbit_out;
}
