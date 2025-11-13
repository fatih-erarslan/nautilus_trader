// Integrated Information (Φ) approximation shader
// Implements fast approximation algorithms for consciousness metrics

@group(0) @binding(0) var<storage, read> pbit_states: array<f32>; // Current states
@group(0) @binding(1) var<storage, read> coupling_matrix: array<f32>; // N×N coupling matrix
@group(0) @binding(2) var<storage, read> phi_params: PhiParams;
@group(0) @binding(3) var<storage, read_write> phi_result: array<f32>; // Single value

struct PhiParams {
    num_nodes: u32,
    temperature: f32,
    approximation_method: u32, // 0=upper_bound, 1=lower_bound, 2=hierarchical
    partition_size: u32,       // For hierarchical method
}

// Upper bound Φ approximation using maximum bipartition
@compute @workgroup_size(1, 1, 1)
fn compute_phi_upper_bound(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = phi_params.num_nodes;
    
    if (n < 2u) {
        phi_result[0] = 0.0;
        return;
    }
    
    var max_effective_info = 0.0f;
    
    // Try all possible bipartitions (exponential, but approximated)
    // For efficiency, sample representative partitions
    let max_partitions = min(256u, 1u << (n - 1u)); // Limit to 256 samples
    
    for (var partition_id = 1u; partition_id < max_partitions; partition_id = partition_id + 1u) {
        let effective_info = compute_effective_information(partition_id, n);
        max_effective_info = max(max_effective_info, effective_info);
    }
    
    phi_result[0] = max_effective_info;
}

// Lower bound Φ approximation using minimum cut
@compute @workgroup_size(1, 1, 1)
fn compute_phi_lower_bound(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = phi_params.num_nodes;
    
    if (n < 2u) {
        phi_result[0] = 0.0;
        return;
    }
    
    // Find minimum cut approximation
    var min_cut_weight = 1e10f; // Large initial value
    
    // Sample balanced partitions (50-50 split approximations)
    for (var i = 1u; i < n; i = i + 1u) {
        let cut_weight = compute_cut_weight(i, n);
        min_cut_weight = min(min_cut_weight, cut_weight);
    }
    
    // Convert cut weight to effective information approximation
    let phi_lower = min_cut_weight * 0.1; // Scaling factor
    phi_result[0] = phi_lower;
}

// Hierarchical Φ computation for large systems
@compute @workgroup_size(1, 1, 1)
fn compute_phi_hierarchical(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = phi_params.num_nodes;
    let block_size = phi_params.partition_size;
    
    if (n < 2u || block_size == 0u) {
        phi_result[0] = 0.0;
        return;
    }
    
    let num_blocks = (n + block_size - 1u) / block_size; // Ceiling division
    
    var total_phi = 0.0f;
    
    // Compute Φ for each hierarchical block
    for (var block = 0u; block < num_blocks; block = block + 1u) {
        let start_idx = block * block_size;
        let end_idx = min(start_idx + block_size, n);
        let block_n = end_idx - start_idx;
        
        if (block_n >= 2u) {
            let block_phi = compute_block_phi(start_idx, block_n);
            total_phi = total_phi + block_phi;
        }
    }
    
    // Add inter-block integration
    let inter_block_phi = compute_inter_block_phi(num_blocks, block_size);
    total_phi = total_phi + inter_block_phi;
    
    phi_result[0] = total_phi;
}

// Helper: Compute effective information for a partition
fn compute_effective_information(partition_id: u32, n: u32) -> f32 {
    var subset_a_entropy = 0.0f;
    var subset_b_entropy = 0.0f;
    var joint_entropy = 0.0f;
    
    // Simplified entropy calculation based on coupling strengths
    var a_size = 0u;
    var b_size = 0u;
    
    for (var i = 0u; i < n; i = i + 1u) {
        let in_subset_a = (partition_id & (1u << i)) != 0u;
        
        if (in_subset_a) {
            a_size = a_size + 1u;
            subset_a_entropy = subset_a_entropy + compute_node_entropy(i);
        } else {
            b_size = b_size + 1u;
            subset_b_entropy = subset_b_entropy + compute_node_entropy(i);
        }
    }
    
    // Avoid degenerate partitions
    if (a_size == 0u || b_size == 0u) {
        return 0.0;
    }
    
    // Joint entropy approximation
    joint_entropy = subset_a_entropy + subset_b_entropy;
    
    // Add cross-partition coupling effects
    let cross_coupling = compute_cross_partition_coupling(partition_id, n);
    joint_entropy = joint_entropy - cross_coupling;
    
    // Effective information: EI = H(A) + H(B) - H(A,B)
    let effective_info = subset_a_entropy + subset_b_entropy - joint_entropy;
    return max(effective_info, 0.0);
}

// Helper: Compute entropy contribution of a single node
fn compute_node_entropy(node_idx: u32) -> f32 {
    let state = pbit_states[node_idx];
    let p = sigmoid_probability(node_idx);
    
    // Shannon entropy: H = -p*log(p) - (1-p)*log(1-p)
    let eps = 1e-10f;
    let p_safe = clamp(p, eps, 1.0 - eps);
    let entropy = -p_safe * log(p_safe) - (1.0 - p_safe) * log(1.0 - p_safe);
    
    return entropy;
}

// Helper: Compute sigmoid probability for a node
fn sigmoid_probability(node_idx: u32) -> f32 {
    let n = phi_params.num_nodes;
    var h_eff = 0.0f;
    
    // Compute effective field: h_eff = Σ_j J_ij * s_j
    for (var j = 0u; j < n; j = j + 1u) {
        if (node_idx != j) {
            let coupling = coupling_matrix[node_idx * n + j];
            let state = pbit_states[j];
            h_eff = h_eff + coupling * state;
        }
    }
    
    // Sigmoid: P(s=1) = 1/(1 + exp(-h_eff/T))
    let argument = h_eff / phi_params.temperature;
    return 1.0 / (1.0 + exp(-argument));
}

// Helper: Compute cut weight between partitions
fn compute_cut_weight(split_point: u32, n: u32) -> f32 {
    var cut_weight = 0.0f;
    
    for (var i = 0u; i < split_point; i = i + 1u) {
        for (var j = split_point; j < n; j = j + 1u) {
            let coupling_ij = coupling_matrix[i * n + j];
            let coupling_ji = coupling_matrix[j * n + i];
            cut_weight = cut_weight + abs(coupling_ij) + abs(coupling_ji);
        }
    }
    
    return cut_weight;
}

// Helper: Compute cross-partition coupling strength
fn compute_cross_partition_coupling(partition_id: u32, n: u32) -> f32 {
    var cross_coupling = 0.0f;
    
    for (var i = 0u; i < n; i = i + 1u) {
        for (var j = 0u; j < n; j = j + 1u) {
            if (i != j) {
                let i_in_a = (partition_id & (1u << i)) != 0u;
                let j_in_a = (partition_id & (1u << j)) != 0u;
                
                // Cross-partition coupling
                if (i_in_a != j_in_a) {
                    let coupling = coupling_matrix[i * n + j];
                    cross_coupling = cross_coupling + abs(coupling);
                }
            }
        }
    }
    
    return cross_coupling * 0.01; // Scaling factor
}

// Helper: Compute Φ for a hierarchical block
fn compute_block_phi(start_idx: u32, block_size: u32) -> f32 {
    if (block_size < 2u) {
        return 0.0;
    }
    
    // Simplified block Φ calculation
    var block_coupling = 0.0f;
    
    for (var i = 0u; i < block_size; i = i + 1u) {
        for (var j = 0u; j < block_size; j = j + 1u) {
            if (i != j) {
                let global_i = start_idx + i;
                let global_j = start_idx + j;
                let n = phi_params.num_nodes;
                let coupling = coupling_matrix[global_i * n + global_j];
                block_coupling = block_coupling + abs(coupling);
            }
        }
    }
    
    // Convert coupling strength to Φ approximation
    return block_coupling * 0.05; // Scaling factor
}

// Helper: Compute inter-block integration
fn compute_inter_block_phi(num_blocks: u32, block_size: u32) -> f32 {
    if (num_blocks < 2u) {
        return 0.0;
    }
    
    var inter_coupling = 0.0f;
    let n = phi_params.num_nodes;
    
    // Sum coupling between different blocks
    for (var block_a = 0u; block_a < num_blocks; block_a = block_a + 1u) {
        for (var block_b = block_a + 1u; block_b < num_blocks; block_b = block_b + 1u) {
            let start_a = block_a * block_size;
            let start_b = block_b * block_size;
            let end_a = min(start_a + block_size, n);
            let end_b = min(start_b + block_size, n);
            
            for (var i = start_a; i < end_a; i = i + 1u) {
                for (var j = start_b; j < end_b; j = j + 1u) {
                    let coupling = coupling_matrix[i * n + j];
                    inter_coupling = inter_coupling + abs(coupling);
                }
            }
        }
    }
    
    return inter_coupling * 0.02; // Scaling factor
}
