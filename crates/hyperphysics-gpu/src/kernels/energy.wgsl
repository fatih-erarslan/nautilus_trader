// Energy Calculation Kernel (Ising Hamiltonian)
//
// Computes E = -Σ_i Σ_j J_ij s_i s_j using parallel reduction.
// Uses two-pass algorithm: local sums in workgroups, then global reduction.

struct PBit {
    state: u32,
    bias: f32,
    coupling_offset: u32,
    coupling_count: u32,
}

struct Coupling {
    target_idx: u32,
    strength: f32,
}

@group(0) @binding(0) var<storage, read> pbits: array<PBit>;
@group(0) @binding(1) var<storage, read> couplings: array<Coupling>;
@group(0) @binding(2) var<storage, read_write> energy_partial: array<f32>;
@group(0) @binding(3) var<uniform> params: EnergyParams;

struct EnergyParams {
    n_pbits: u32,
    n_workgroups: u32,
}

var<workgroup> shared_energy: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let global_idx = global_id.x;
    let local_idx = local_id.x;

    var energy = 0.0;

    // Each thread computes energy contribution for its pBit
    if (global_idx < params.n_pbits) {
        let pbit = pbits[global_idx];
        let state_i = f32(pbit.state) * 2.0 - 1.0; // {0,1} → {-1,+1}

        // Sum over all couplings
        let coupling_start = pbit.coupling_offset;
        let coupling_end = coupling_start + pbit.coupling_count;

        for (var i = coupling_start; i < coupling_end; i = i + 1u) {
            let coupling = couplings[i];
            let state_j = f32(pbits[coupling.target_idx].state) * 2.0 - 1.0;

            // E = -J_ij * s_i * s_j (only count each pair once via i < j check)
            if (global_idx < coupling.target_idx) {
                energy -= coupling.strength * state_i * state_j;
            }
        }
    }

    // Store local energy in shared memory
    shared_energy[local_idx] = energy;
    workgroupBarrier();

    // Parallel reduction within workgroup
    var stride = 128u;
    while (stride > 0u) {
        if (local_idx < stride && local_idx + stride < 256u) {
            shared_energy[local_idx] += shared_energy[local_idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // First thread writes workgroup sum to global memory
    if (local_idx == 0u) {
        energy_partial[workgroup_id.x] = shared_energy[0];
    }
}

// Second pass: reduce partial sums
@compute @workgroup_size(256)
fn reduce_final(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let idx = global_id.x;
    let local_idx = local_id.x;

    // Load partial sum
    var value = 0.0;
    if (idx < params.n_workgroups) {
        value = energy_partial[idx];
    }

    shared_energy[local_idx] = value;
    workgroupBarrier();

    // Final reduction
    var stride = 128u;
    while (stride > 0u) {
        if (local_idx < stride) {
            shared_energy[local_idx] += shared_energy[local_idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Write final result
    if (local_idx == 0u) {
        energy_partial[0] = shared_energy[0];
    }
}
