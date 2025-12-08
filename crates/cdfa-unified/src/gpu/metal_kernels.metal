//! Metal compute shaders for CDFA GPU operations
//!
//! This file contains Metal Shading Language (MSL) compute kernels
//! optimized for Apple Silicon and Metal-compatible GPUs.

#include <metal_stdlib>
using namespace metal;

/// Matrix multiplication kernel
kernel void matrix_multiply(
    const device float* a [[buffer(0)]],
    const device float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    const device uint& m [[buffer(3)]],
    const device uint& n [[buffer(4)]],
    const device uint& k [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= m || col >= n) return;
    
    float sum = 0.0;
    for (uint i = 0; i < k; i++) {
        sum += a[row * k + i] * b[i * n + col];
    }
    c[row * n + col] = sum;
}

/// Optimized matrix multiplication with shared memory (threadgroup memory)
kernel void matrix_multiply_tiled(
    const device float* a [[buffer(0)]],
    const device float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    const device uint& m [[buffer(3)]],
    const device uint& n [[buffer(4)]],
    const device uint& k [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    const uint TILE_SIZE = 16;
    threadgroup float a_tile[TILE_SIZE][TILE_SIZE];
    threadgroup float b_tile[TILE_SIZE][TILE_SIZE];
    
    uint row = tgid.y * TILE_SIZE + lid.y;
    uint col = tgid.x * TILE_SIZE + lid.x;
    
    float sum = 0.0;
    
    for (uint tile = 0; tile < (k + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tiles into threadgroup memory
        uint a_col = tile * TILE_SIZE + lid.x;
        uint b_row = tile * TILE_SIZE + lid.y;
        
        a_tile[lid.y][lid.x] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0;
        b_tile[lid.y][lid.x] = (b_row < k && col < n) ? b[b_row * n + col] : 0.0;
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial sum
        for (uint i = 0; i < TILE_SIZE; i++) {
            sum += a_tile[lid.y][i] * b_tile[i][lid.x];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}

/// Element-wise operations kernel
kernel void element_wise_op(
    const device float* a [[buffer(0)]],
    const device float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    const device uint& size [[buffer(3)]],
    const device uint& op_type [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    switch (op_type) {
        case 0: // Addition
            c[gid] = a[gid] + b[gid];
            break;
        case 1: // Multiplication
            c[gid] = a[gid] * b[gid];
            break;
        case 2: // Subtraction
            c[gid] = a[gid] - b[gid];
            break;
        case 3: // Division
            c[gid] = a[gid] / b[gid];
            break;
        case 4: // Power
            c[gid] = pow(a[gid], b[gid]);
            break;
        case 5: // Max
            c[gid] = max(a[gid], b[gid]);
            break;
        case 6: // Min
            c[gid] = min(a[gid], b[gid]);
            break;
        default:
            c[gid] = a[gid];
    }
}

/// Vectorized element-wise operations (float4)
kernel void element_wise_op_vec4(
    const device float4* a [[buffer(0)]],
    const device float4* b [[buffer(1)]],
    device float4* c [[buffer(2)]],
    const device uint& size [[buffer(3)]],
    const device uint& op_type [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    float4 a_vec = a[gid];
    float4 b_vec = b[gid];
    
    switch (op_type) {
        case 0: // Addition
            c[gid] = a_vec + b_vec;
            break;
        case 1: // Multiplication
            c[gid] = a_vec * b_vec;
            break;
        case 2: // Subtraction
            c[gid] = a_vec - b_vec;
            break;
        case 3: // Division
            c[gid] = a_vec / b_vec;
            break;
        case 4: // Power
            c[gid] = pow(a_vec, b_vec);
            break;
        case 5: // Max
            c[gid] = max(a_vec, b_vec);
            break;
        case 6: // Min
            c[gid] = min(a_vec, b_vec);
            break;
        default:
            c[gid] = a_vec;
    }
}

/// Reduction sum kernel with threadgroup optimization
kernel void reduce_sum(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    const device uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]]
) {
    threadgroup float shared_data[256];
    
    // Load data into threadgroup memory
    uint idx = gid;
    shared_data[lid] = (idx < size) ? input[idx] : 0.0;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction in threadgroup memory
    for (uint s = threadgroup_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared_data[lid] += shared_data[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result for this threadgroup
    if (lid == 0) {
        output[gid / threadgroup_size] = shared_data[0];
    }
}

/// Parallel reduction sum with simd-group (warp) optimization
kernel void reduce_sum_simd(
    const device float* input [[buffer(0)]],
    device atomic<float>* output [[buffer(1)]],
    const device uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float simd_sums[8]; // Assuming max 8 simd groups per threadgroup
    
    float value = (gid < size) ? input[gid] : 0.0;
    
    // Simd-group reduction
    value = simd_sum(value);
    
    // First thread in each simd-group writes to threadgroup memory
    if (simd_lane_id == 0) {
        simd_sums[simd_group_id] = value;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Final reduction across simd-groups
    if (simd_group_id == 0 && simd_lane_id < 8) {
        float final_sum = simd_lane_id < 8 ? simd_sums[simd_lane_id] : 0.0;
        final_sum = simd_sum(final_sum);
        
        if (simd_lane_id == 0) {
            atomic_fetch_add_explicit(output, final_sum, memory_order_relaxed);
        }
    }
}

/// Pearson correlation diversity calculation
kernel void pearson_diversity(
    const device float* correlation_matrix [[buffer(0)]],
    device float* diversity_scores [[buffer(1)]],
    const device uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    float sum = 0.0;
    float sum_sq = 0.0;
    uint count = 0;
    
    for (uint j = 0; j < n; j++) {
        if (j != gid) {
            float corr = correlation_matrix[gid * n + j];
            sum += corr;
            sum_sq += corr * corr;
            count++;
        }
    }
    
    if (count > 0) {
        float mean = sum / count;
        float variance = (sum_sq / count) - (mean * mean);
        diversity_scores[gid] = sqrt(variance);
    } else {
        diversity_scores[gid] = 0.0;
    }
}

/// Kendall tau diversity calculation (simplified)
kernel void kendall_diversity(
    const device float* ranks_matrix [[buffer(0)]],
    device float* diversity_scores [[buffer(1)]],
    const device uint& n [[buffer(2)]],
    const device uint& m [[buffer(3)]], // number of items being ranked
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    float total_tau = 0.0;
    uint comparisons = 0;
    
    for (uint j = 0; j < n; j++) {
        if (j != gid) {
            float concordant = 0.0;
            float discordant = 0.0;
            
            // Count concordant and discordant pairs
            for (uint i1 = 0; i1 < m; i1++) {
                for (uint i2 = i1 + 1; i2 < m; i2++) {
                    float rank1_i = ranks_matrix[gid * m + i1];
                    float rank1_j = ranks_matrix[gid * m + i2];
                    float rank2_i = ranks_matrix[j * m + i1];
                    float rank2_j = ranks_matrix[j * m + i2];
                    
                    float sign1 = sign(rank1_i - rank1_j);
                    float sign2 = sign(rank2_i - rank2_j);
                    
                    if (sign1 * sign2 > 0) {
                        concordant += 1.0;
                    } else if (sign1 * sign2 < 0) {
                        discordant += 1.0;
                    }
                }
            }
            
            float tau = (concordant - discordant) / (concordant + discordant);
            total_tau += abs(tau);
            comparisons++;
        }
    }
    
    diversity_scores[gid] = comparisons > 0 ? (1.0 - total_tau / comparisons) : 0.0;
}

/// Fast Fourier Transform butterfly operation
kernel void fft_butterfly(
    device float2* data [[buffer(0)]],
    const device uint& n [[buffer(1)]],
    const device uint& stage [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint stride = 1 << stage;
    uint half_stride = stride >> 1;
    
    if (gid >= n / 2) return;
    
    uint group = gid / half_stride;
    uint pos = gid % half_stride;
    uint i = group * stride + pos;
    uint j = i + half_stride;
    
    float angle = -2.0 * M_PI_F * pos / stride;
    float2 w = float2(cos(angle), sin(angle));
    
    float2 u = data[i];
    float2 v = float2(
        data[j].x * w.x - data[j].y * w.y,
        data[j].x * w.y + data[j].y * w.x
    );
    
    data[i] = u + v;
    data[j] = u - v;
}

/// Convolution kernel
kernel void convolution_1d(
    const device float* input [[buffer(0)]],
    const device float* kernel_data [[buffer(1)]],
    device float* output [[buffer(2)]],
    const device uint& input_size [[buffer(3)]],
    const device uint& kernel_size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= input_size) return;
    
    float sum = 0.0;
    uint half_kernel = kernel_size / 2;
    
    for (uint k = 0; k < kernel_size; k++) {
        int input_idx = (int)gid + (int)k - (int)half_kernel;
        if (input_idx >= 0 && input_idx < input_size) {
            sum += input[input_idx] * kernel_data[k];
        }
    }
    
    output[gid] = sum;
}

/// Variance calculation kernel
kernel void calculate_variance(
    const device float* data [[buffer(0)]],
    const device float& mean [[buffer(1)]],
    device float* output [[buffer(2)]],
    const device uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    float diff = data[gid] - mean;
    output[gid] = diff * diff;
}

/// Normalize vector kernel
kernel void normalize_vector(
    device float* data [[buffer(0)]],
    const device uint& size [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    // First pass: compute sum of squares (would need reduction)
    // This is simplified - real implementation would use two-pass approach
    threadgroup float sum_sq_shared[256];
    uint lid = gid % 256;
    
    sum_sq_shared[lid] = (gid < size) ? data[gid] * data[gid] : 0.0;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Simple reduction (not optimal)
    if (lid == 0) {
        float total = 0.0;
        for (uint i = 0; i < 256 && (gid - lid + i) < size; i++) {
            total += sum_sq_shared[i];
        }
        float norm = sqrt(total);
        
        // Normalize
        for (uint i = 0; i < 256 && (gid - lid + i) < size; i++) {
            if (norm > 0.0) {
                data[gid - lid + i] /= norm;
            }
        }
    }
}