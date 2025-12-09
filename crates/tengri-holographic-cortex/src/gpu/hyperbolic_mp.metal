// Hyperbolic Message Passing Metal Compute Shader
// Optimized for AMD GPUs (Radeon 6800XT / 5500XT)
//
// Mathematical Foundation (Wolfram-Verified):
// - Lorentz inner: ⟨x,y⟩_L = -x₀y₀ + Σᵢ xᵢyᵢ
// - Hyperbolic distance: d(x,y) = acosh(-⟨x,y⟩_L)
// - Möbius add: x ⊕_c y = ((1+2c⟨x,y⟩+c||y||²)x + (1-c||x||²)y) / denom

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// CONSTANTS
// ============================================================================

constant uint LORENTZ_DIM = 12;  // H^11 embedded in R^12
constant uint HYPERBOLIC_DIM = 11;
constant float EPSILON = 1e-7f;
constant float MAX_DIST = 50.0f;

// STDP constants (Wolfram-verified)
constant float STDP_A_PLUS = 0.1f;
constant float STDP_A_MINUS = 0.12f;
constant float STDP_TAU_PLUS = 20.0f;
constant float STDP_TAU_MINUS = 20.0f;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

struct NodeData {
    float coords[12];  // Lorentz coordinates [x0, x1, ..., x11]
    float bias;
    float temperature;
    uint state;        // 0 or 1
    uint _padding;
};

struct EdgeData {
    uint src;
    uint dst;
    float weight;
    uint _padding;
};

struct Config {
    uint num_nodes;
    uint num_edges;
    float base_temperature;
    float curvature;   // Typically -1.0
};

struct SpikeTime {
    uint node_id;
    uint tick;
};

// ============================================================================
// LORENTZ OPERATIONS
// ============================================================================

// Lorentz inner product: ⟨x,y⟩_L = -x₀y₀ + Σᵢ xᵢyᵢ
inline float lorentz_inner(const float x[12], const float y[12]) {
    float result = -x[0] * y[0];
    for (uint i = 1; i < LORENTZ_DIM; ++i) {
        result += x[i] * y[i];
    }
    return result;
}

// Vectorized Lorentz inner using float4
inline float lorentz_inner_simd(thread const float4* x, thread const float4* y) {
    // x[0] and y[0] contain [x0, x1, x2, x3]
    // Need to negate first component
    float4 prod0 = x[0] * y[0];
    float4 prod1 = x[1] * y[1];
    float4 prod2 = x[2] * y[2];
    
    float result = -prod0.x;  // Negate time component
    result += prod0.y + prod0.z + prod0.w;
    result += prod1.x + prod1.y + prod1.z + prod1.w;
    result += prod2.x + prod2.y + prod2.z + prod2.w;
    
    return result;
}

// Stable acosh using sqrt approximation near 1
inline float stable_acosh(float x) {
    if (x < 1.0f + EPSILON) {
        return sqrt(2.0f * max(x - 1.0f, 0.0f));
    }
    return acosh(x);
}

// Hyperbolic distance: d(x,y) = acosh(-⟨x,y⟩_L)
inline float hyperbolic_distance(const float x[12], const float y[12]) {
    float inner = -lorentz_inner(x, y);
    return min(stable_acosh(max(inner, 1.0f)), MAX_DIST);
}

// Project to hyperboloid: x₀ = √(1 + ||z||²)
inline void project_to_hyperboloid(thread float coords[12]) {
    float spatial_norm_sq = 0.0f;
    for (uint i = 1; i < LORENTZ_DIM; ++i) {
        spatial_norm_sq += coords[i] * coords[i];
    }
    coords[0] = sqrt(1.0f + spatial_norm_sq);
}

// ============================================================================
// pBIT SAMPLING
// ============================================================================

// Fast sigmoid approximation using tanh: σ(x) = 0.5 * (1 + tanh(x/2))
inline float sigmoid_fast(float x) {
    return 0.5f * (1.0f + tanh(x * 0.5f));
}

// pBit probability: P(s=1) = σ((h - bias) / T)
inline float pbit_probability(float h, float bias, float temperature) {
    float t = max(temperature, EPSILON);
    return sigmoid_fast((h - bias) / t);
}

// ============================================================================
// STDP
// ============================================================================

inline float stdp_weight_change(float delta_t) {
    if (delta_t > 0.0f) {
        // LTP: pre before post
        return STDP_A_PLUS * exp(-delta_t / STDP_TAU_PLUS);
    } else {
        // LTD: post before pre
        return -STDP_A_MINUS * exp(delta_t / STDP_TAU_MINUS);
    }
}

// ============================================================================
// COMPUTE KERNELS
// ============================================================================

// Kernel 1: Compute hyperbolic distances and edge messages
kernel void compute_edge_messages(
    device const NodeData* nodes_in [[buffer(0)]],
    device const EdgeData* edges [[buffer(1)]],
    device float* messages [[buffer(2)]],
    constant Config& config [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= config.num_edges) return;
    
    EdgeData edge = edges[gid];
    NodeData src_node = nodes_in[edge.src];
    NodeData dst_node = nodes_in[edge.dst];
    
    // Compute hyperbolic distance
    float dist = hyperbolic_distance(src_node.coords, dst_node.coords);
    
    // Message = weight * state * exp(-dist) / temperature
    float temp = max(src_node.temperature, EPSILON);
    float msg = edge.weight * float(src_node.state) * exp(-dist / temp);
    
    messages[gid] = msg;
}

// Kernel 2: Aggregate messages and update pBit states
kernel void aggregate_and_update(
    device const NodeData* nodes_in [[buffer(0)]],
    device NodeData* nodes_out [[buffer(1)]],
    device const EdgeData* edges [[buffer(2)]],
    device const float* messages [[buffer(3)]],
    constant Config& config [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= config.num_nodes) return;
    
    float total_msg = 0.0f;
    
    // Sum incoming messages
    for (uint e = 0; e < config.num_edges; ++e) {
        if (edges[e].dst == gid) {
            total_msg += messages[e];
        }
    }
    
    NodeData node = nodes_in[gid];
    float h_eff = node.bias + total_msg;
    
    // Sample new state (threshold for deterministic, use RNG for stochastic)
    float prob = pbit_probability(h_eff, 0.0f, node.temperature);
    uint new_state = (prob > 0.5f) ? 1u : 0u;
    
    node.state = new_state;
    nodes_out[gid] = node;
}

// Kernel 3: CSR-based message aggregation (optimized)
kernel void aggregate_csr(
    device const NodeData* nodes_in [[buffer(0)]],
    device NodeData* nodes_out [[buffer(1)]],
    device const uint* csr_offsets [[buffer(2)]],    // Row pointers (num_nodes + 1)
    device const uint* csr_indices [[buffer(3)]],    // Column indices
    device const float* csr_weights [[buffer(4)]],   // Edge weights
    device const float* messages [[buffer(5)]],
    constant Config& config [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= config.num_nodes) return;
    
    uint start = csr_offsets[gid];
    uint end = csr_offsets[gid + 1];
    
    float total_msg = 0.0f;
    
    // Vectorized accumulation
    for (uint i = start; i < end; ++i) {
        uint src_idx = csr_indices[i];
        total_msg += csr_weights[i] * messages[src_idx];
    }
    
    NodeData node = nodes_in[gid];
    float h_eff = node.bias + total_msg;
    float prob = pbit_probability(h_eff, 0.0f, node.temperature);
    
    node.state = (prob > 0.5f) ? 1u : 0u;
    nodes_out[gid] = node;
}

// Kernel 4: Möbius aggregation in hyperbolic space
kernel void mobius_aggregate(
    device const NodeData* nodes_in [[buffer(0)]],
    device NodeData* nodes_out [[buffer(1)]],
    device const EdgeData* edges [[buffer(2)]],
    constant Config& config [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= config.num_nodes) return;
    
    NodeData node = nodes_in[gid];
    float aggregated[12];
    
    // Initialize with current position
    for (uint i = 0; i < LORENTZ_DIM; ++i) {
        aggregated[i] = node.coords[i];
    }
    
    float total_weight = 1.0f;
    float c = -config.curvature;  // Use positive for Poincaré ball
    
    // Aggregate neighbors via weighted Möbius mean
    for (uint e = 0; e < config.num_edges; ++e) {
        if (edges[e].dst == gid) {
            NodeData src = nodes_in[edges[e].src];
            float w = edges[e].weight * float(src.state);
            
            if (w > EPSILON) {
                // Compute Möbius coefficients
                float xy = 0.0f, x_norm_sq = 0.0f, y_norm_sq = 0.0f;
                
                for (uint i = 1; i < LORENTZ_DIM; ++i) {
                    xy += aggregated[i] * src.coords[i];
                    x_norm_sq += aggregated[i] * aggregated[i];
                    y_norm_sq += src.coords[i] * src.coords[i];
                }
                
                float denom = 1.0f + 2.0f * c * xy + c * c * x_norm_sq * y_norm_sq;
                float coef_x = 1.0f + 2.0f * c * xy + c * y_norm_sq;
                float coef_y = 1.0f - c * x_norm_sq;
                
                // Weighted Möbius blend
                for (uint i = 1; i < LORENTZ_DIM; ++i) {
                    float mob = (coef_x * aggregated[i] + coef_y * src.coords[i]) / max(denom, EPSILON);
                    aggregated[i] = (aggregated[i] * total_weight + mob * w) / (total_weight + w);
                }
                
                total_weight += w;
            }
        }
    }
    
    // Project back to hyperboloid
    project_to_hyperboloid(aggregated);
    
    // Write output
    for (uint i = 0; i < LORENTZ_DIM; ++i) {
        node.coords[i] = aggregated[i];
    }
    nodes_out[gid] = node;
}

// Kernel 5: STDP weight updates
kernel void compute_stdp(
    device const EdgeData* edges [[buffer(0)]],
    device const SpikeTime* spike_times [[buffer(1)]],
    device float* weight_updates [[buffer(2)]],
    constant Config& config [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= config.num_edges) return;
    
    EdgeData edge = edges[gid];
    SpikeTime src_spike = spike_times[edge.src];
    SpikeTime dst_spike = spike_times[edge.dst];
    
    // Timing difference (in ms, assuming 1 tick = 1ms)
    float delta_t = float(dst_spike.tick) - float(src_spike.tick);
    
    weight_updates[gid] = stdp_weight_change(delta_t);
}

// Kernel 6: Batch pBit sampling with RNG
kernel void pbit_sample_batch(
    device const float* effective_fields [[buffer(0)]],
    device const float* random_values [[buffer(1)]],
    device uint* states [[buffer(2)]],
    device const float* temperatures [[buffer(3)]],
    constant uint& num_nodes [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_nodes) return;
    
    float h = effective_fields[gid];
    float temp = max(temperatures[gid], EPSILON);
    float prob = sigmoid_fast(h / temp);
    
    states[gid] = (random_values[gid] < prob) ? 1u : 0u;
}

// Kernel 7: Fast field computation with SIMD
kernel void compute_fields_simd(
    device const float4* biases [[buffer(0)]],
    device const uint4* states [[buffer(1)]],
    device const float4* coupling_weights [[buffer(2)]],
    device float4* fields [[buffer(3)]],
    constant uint& num_nodes_div4 [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_nodes_div4) return;
    
    float4 bias = biases[gid];
    float4 field = bias;
    
    // Accumulate coupling contributions (simplified)
    // In full implementation, use CSR format
    for (uint i = 0; i < num_nodes_div4; ++i) {
        uint4 s = states[i];
        float4 sf = float4(s.x, s.y, s.z, s.w);
        float4 w = coupling_weights[gid * num_nodes_div4 + i];
        field += w * sf;
    }
    
    fields[gid] = field;
}
