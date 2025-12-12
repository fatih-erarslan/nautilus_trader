// High-Performance CUDA Kernels for QBMIA Quantum Operations
// Optimized for Hopper/Ampere architectures with PTX 8.0+

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuComplex.h>
#include <cooperative_groups.h>
#include <cuda/std/complex>

namespace cg = cooperative_groups;

// Constants for quantum operations
constexpr float PI = 3.14159265358979323846f;
constexpr int WARP_SIZE = 32;
constexpr int MAX_QUBITS = 16;
constexpr int MAX_STATE_DIM = 1 << MAX_QUBITS;

// Complex number operations using CUDA's optimized complex library
using Complex32 = cuda::std::complex<float>;
using Complex64 = cuda::std::complex<double>;

// Shared memory configuration for different kernels
extern __shared__ char shared_mem[];

// Template for precision-agnostic operations
template<typename T, typename ComplexT>
struct QuantumOps {
    static constexpr T SQRT2_INV = T(0.7071067811865475);
    static constexpr T SQRT2 = T(1.4142135623730951);
};

// =============================================================================
// QUANTUM GATE KERNELS
// =============================================================================

// Hadamard gate kernel - optimized for single qubit operations
template<typename T, typename ComplexT>
__global__ void hadamard_gate_kernel(
    ComplexT* state_vector,
    const int qubit_idx,
    const int num_qubits,
    const int batch_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int state_dim = 1 << num_qubits;
    const int total_elements = state_dim * batch_size;
    
    if (tid >= total_elements) return;
    
    // Determine batch and state index
    const int batch_idx = tid / state_dim;
    const int state_idx = tid % state_dim;
    
    // Check if this state index is affected by the gate
    const int qubit_mask = 1 << qubit_idx;
    if ((state_idx & qubit_mask) != 0) return; // Only process |0⟩ states
    
    // Calculate paired state index (|1⟩ state)
    const int paired_idx = state_idx | qubit_mask;
    
    // Load states from global memory (coalesced access)
    const int base_offset = batch_idx * state_dim;
    ComplexT state0 = state_vector[base_offset + state_idx];
    ComplexT state1 = state_vector[base_offset + paired_idx];
    
    // Apply Hadamard transformation
    const T sqrt2_inv = QuantumOps<T, ComplexT>::SQRT2_INV;
    ComplexT new_state0 = (state0 + state1) * sqrt2_inv;
    ComplexT new_state1 = (state0 - state1) * sqrt2_inv;
    
    // Write back to global memory
    state_vector[base_offset + state_idx] = new_state0;
    state_vector[base_offset + paired_idx] = new_state1;
}

// CNOT gate kernel - optimized for two-qubit operations
template<typename T, typename ComplexT>
__global__ void cnot_gate_kernel(
    ComplexT* state_vector,
    const int control_qubit,
    const int target_qubit,
    const int num_qubits,
    const int batch_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int state_dim = 1 << num_qubits;
    const int total_elements = state_dim * batch_size;
    
    if (tid >= total_elements) return;
    
    const int batch_idx = tid / state_dim;
    const int state_idx = tid % state_dim;
    
    // Check if control qubit is |1⟩
    const int control_mask = 1 << control_qubit;
    if ((state_idx & control_mask) == 0) return; // Only process control=1 states
    
    // Check if target qubit is |0⟩
    const int target_mask = 1 << target_qubit;
    if ((state_idx & target_mask) != 0) return; // Only process target=0 states
    
    // Calculate flipped state index
    const int flipped_idx = state_idx | target_mask;
    
    // Swap amplitudes
    const int base_offset = batch_idx * state_dim;
    ComplexT temp = state_vector[base_offset + state_idx];
    state_vector[base_offset + state_idx] = state_vector[base_offset + flipped_idx];
    state_vector[base_offset + flipped_idx] = temp;
}

// Rotation gate kernels (RX, RY, RZ) with shared memory optimization
template<typename T, typename ComplexT>
__global__ void rx_gate_kernel(
    ComplexT* state_vector,
    const T angle,
    const int qubit_idx,
    const int num_qubits,
    const int batch_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int state_dim = 1 << num_qubits;
    
    // Precompute rotation matrix elements
    const T half_angle = angle * T(0.5);
    const T cos_val = cos(half_angle);
    const T sin_val = sin(half_angle);
    const ComplexT rotation_00(cos_val, T(0));
    const ComplexT rotation_01(T(0), -sin_val);
    const ComplexT rotation_10(T(0), -sin_val);
    const ComplexT rotation_11(cos_val, T(0));
    
    // Process states
    for (int idx = tid; idx < state_dim * batch_size; idx += gridDim.x * blockDim.x) {
        const int batch_idx = idx / state_dim;
        const int state_idx = idx % state_dim;
        
        const int qubit_mask = 1 << qubit_idx;
        if ((state_idx & qubit_mask) != 0) continue;
        
        const int paired_idx = state_idx | qubit_mask;
        const int base_offset = batch_idx * state_dim;
        
        // Load states
        ComplexT state0 = state_vector[base_offset + state_idx];
        ComplexT state1 = state_vector[base_offset + paired_idx];
        
        // Apply rotation
        ComplexT new_state0 = rotation_00 * state0 + rotation_01 * state1;
        ComplexT new_state1 = rotation_10 * state0 + rotation_11 * state1;
        
        // Store results
        state_vector[base_offset + state_idx] = new_state0;
        state_vector[base_offset + paired_idx] = new_state1;
    }
}

template<typename T, typename ComplexT>
__global__ void ry_gate_kernel(
    ComplexT* state_vector,
    const T angle,
    const int qubit_idx,
    const int num_qubits,
    const int batch_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int state_dim = 1 << num_qubits;
    
    // Precompute rotation matrix elements
    const T half_angle = angle * T(0.5);
    const T cos_val = cos(half_angle);
    const T sin_val = sin(half_angle);
    const ComplexT rotation_00(cos_val, T(0));
    const ComplexT rotation_01(-sin_val, T(0));
    const ComplexT rotation_10(sin_val, T(0));
    const ComplexT rotation_11(cos_val, T(0));
    
    // Process states with grid-stride loop
    for (int idx = tid; idx < state_dim * batch_size; idx += gridDim.x * blockDim.x) {
        const int batch_idx = idx / state_dim;
        const int state_idx = idx % state_dim;
        
        const int qubit_mask = 1 << qubit_idx;
        if ((state_idx & qubit_mask) != 0) continue;
        
        const int paired_idx = state_idx | qubit_mask;
        const int base_offset = batch_idx * state_dim;
        
        // Load states
        ComplexT state0 = state_vector[base_offset + state_idx];
        ComplexT state1 = state_vector[base_offset + paired_idx];
        
        // Apply rotation
        ComplexT new_state0 = rotation_00 * state0 + rotation_01 * state1;
        ComplexT new_state1 = rotation_10 * state0 + rotation_11 * state1;
        
        // Store results
        state_vector[base_offset + state_idx] = new_state0;
        state_vector[base_offset + paired_idx] = new_state1;
    }
}

template<typename T, typename ComplexT>
__global__ void rz_gate_kernel(
    ComplexT* state_vector,
    const T angle,
    const int qubit_idx,
    const int num_qubits,
    const int batch_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int state_dim = 1 << num_qubits;
    
    // Precompute phase factors
    const T half_angle = angle * T(0.5);
    const ComplexT phase_0(cos(-half_angle), sin(-half_angle));
    const ComplexT phase_1(cos(half_angle), sin(half_angle));
    
    // Process states
    for (int idx = tid; idx < state_dim * batch_size; idx += gridDim.x * blockDim.x) {
        const int batch_idx = idx / state_dim;
        const int state_idx = idx % state_dim;
        const int base_offset = batch_idx * state_dim;
        
        const int qubit_mask = 1 << qubit_idx;
        if ((state_idx & qubit_mask) == 0) {
            state_vector[base_offset + state_idx] *= phase_0;
        } else {
            state_vector[base_offset + state_idx] *= phase_1;
        }
    }
}

// =============================================================================
// STATE VECTOR MANIPULATION KERNELS
// =============================================================================

// Normalize state vector using CUB-style reduction
template<typename T, typename ComplexT>
__global__ void normalize_state_kernel(
    ComplexT* state_vector,
    const int state_dim,
    const int batch_size
) {
    extern __shared__ T shared_norm[];
    
    const int tid = threadIdx.x;
    const int batch_idx = blockIdx.x;
    const int base_offset = batch_idx * state_dim;
    
    // Phase 1: Compute norm squared using block reduction
    T local_norm = T(0);
    for (int i = tid; i < state_dim; i += blockDim.x) {
        ComplexT val = state_vector[base_offset + i];
        local_norm += norm(val);
    }
    
    shared_norm[tid] = local_norm;
    __syncthreads();
    
    // Warp-level reduction
    if (tid < 32) {
        cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cg::this_thread_block());
        for (int offset = tile.size() / 2; offset > 0; offset /= 2) {
            shared_norm[tid] += tile.shfl_down(shared_norm[tid], offset);
        }
    }
    __syncthreads();
    
    // Phase 2: Normalize
    const T norm_factor = rsqrt(shared_norm[0]);
    for (int i = tid; i < state_dim; i += blockDim.x) {
        state_vector[base_offset + i] *= norm_factor;
    }
}

// Complex number element-wise operations
template<typename T, typename ComplexT>
__global__ void complex_multiply_kernel(
    ComplexT* result,
    const ComplexT* a,
    const ComplexT* b,
    const int size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    
    for (int i = tid; i < size; i += stride) {
        result[i] = a[i] * b[i];
    }
}

// =============================================================================
// EXPECTATION VALUE CALCULATION
// =============================================================================

template<typename T, typename ComplexT>
__global__ void expectation_value_kernel(
    T* expectation,
    const ComplexT* state_vector,
    const T* observable_matrix,
    const int num_qubits,
    const int batch_size
) {
    extern __shared__ T shared_exp[];
    
    const int tid = threadIdx.x;
    const int batch_idx = blockIdx.x;
    const int state_dim = 1 << num_qubits;
    const int base_offset = batch_idx * state_dim;
    
    T local_exp = T(0);
    
    // Compute ⟨ψ|O|ψ⟩ using tiled matrix multiplication
    for (int i = tid; i < state_dim; i += blockDim.x) {
        ComplexT psi_i = state_vector[base_offset + i];
        ComplexT o_psi_i(T(0), T(0));
        
        // Matrix-vector multiplication for row i
        for (int j = 0; j < state_dim; ++j) {
            T o_ij = observable_matrix[i * state_dim + j];
            ComplexT psi_j = state_vector[base_offset + j];
            o_psi_i += o_ij * psi_j;
        }
        
        // Accumulate ⟨ψ|O|ψ⟩
        local_exp += real(conj(psi_i) * o_psi_i);
    }
    
    // Reduce within block
    shared_exp[tid] = local_exp;
    __syncthreads();
    
    // Warp reduction
    if (tid < 32) {
        cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cg::this_thread_block());
        for (int offset = tile.size() / 2; offset > 0; offset /= 2) {
            shared_exp[tid] += tile.shfl_down(shared_exp[tid], offset);
        }
    }
    
    if (tid == 0) {
        expectation[batch_idx] = shared_exp[0];
    }
}

// =============================================================================
// QUANTUM CIRCUIT GRADIENT COMPUTATION
// =============================================================================

template<typename T, typename ComplexT>
__global__ void parameter_shift_gradient_kernel(
    T* gradients,
    ComplexT* state_vector,
    ComplexT* work_state,
    const T* parameters,
    const int* gate_sequence,
    const int num_gates,
    const int num_params,
    const int num_qubits,
    const T shift_angle = PI / T(2)
) {
    const int param_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (param_idx >= num_params) return;
    
    const int state_dim = 1 << num_qubits;
    
    // Copy initial state to work state
    for (int i = tid; i < state_dim; i += blockDim.x) {
        work_state[i] = state_vector[i];
    }
    __syncthreads();
    
    // Apply circuit with shifted parameter (forward)
    T forward_exp = T(0);
    // ... (circuit application code)
    
    // Apply circuit with shifted parameter (backward)
    T backward_exp = T(0);
    // ... (circuit application code)
    
    // Compute gradient using parameter shift rule
    if (tid == 0) {
        gradients[param_idx] = (forward_exp - backward_exp) / (T(2) * sin(shift_angle));
    }
}

// =============================================================================
// NASH EQUILIBRIUM MATRIX OPERATIONS
// =============================================================================

template<typename T>
__global__ void nash_equilibrium_kernel(
    T* strategies,
    const T* payoff_matrix,
    const int num_strategies,
    const int max_iterations,
    const T convergence_threshold
) {
    extern __shared__ T shared_payoff[];
    
    const int tid = threadIdx.x;
    const int strategy_idx = blockIdx.x;
    
    if (strategy_idx >= num_strategies) return;
    
    // Initialize uniform strategies
    T current_strategy = T(1) / num_strategies;
    
    // Fictitious play iteration
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Compute best response
        T max_payoff = T(-1e10);
        int best_response = 0;
        
        for (int i = tid; i < num_strategies; i += blockDim.x) {
            T expected_payoff = T(0);
            for (int j = 0; j < num_strategies; ++j) {
                expected_payoff += payoff_matrix[i * num_strategies + j] * strategies[j];
            }
            
            // Atomic max reduction
            if (expected_payoff > max_payoff) {
                max_payoff = expected_payoff;
                best_response = i;
            }
        }
        
        // Update strategy using exponential weights
        __syncthreads();
        
        // Check convergence
        T strategy_diff = abs(strategies[strategy_idx] - current_strategy);
        if (strategy_diff < convergence_threshold) {
            break;
        }
    }
}

// =============================================================================
// SPECIALIZED KERNELS FOR TRADING
// =============================================================================

// Quantum feature map for market data
template<typename T, typename ComplexT>
__global__ void quantum_feature_map_kernel(
    ComplexT* quantum_state,
    const T* market_features,
    const int num_features,
    const int num_qubits,
    const int batch_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = tid / (1 << num_qubits);
    const int state_idx = tid % (1 << num_qubits);
    
    if (batch_idx >= batch_size) return;
    
    // Encode classical features into quantum amplitudes
    T amplitude = T(1);
    T phase = T(0);
    
    for (int i = 0; i < num_features && i < num_qubits; ++i) {
        if ((state_idx >> i) & 1) {
            phase += market_features[batch_idx * num_features + i];
        }
    }
    
    // Normalize and set quantum state
    const int state_dim = 1 << num_qubits;
    amplitude /= sqrt(T(state_dim));
    
    quantum_state[tid] = ComplexT(amplitude * cos(phase), amplitude * sin(phase));
}

// Quantum portfolio optimization kernel
template<typename T, typename ComplexT>
__global__ void quantum_portfolio_optimization_kernel(
    T* optimal_weights,
    const ComplexT* quantum_state,
    const T* returns_matrix,
    const T* covariance_matrix,
    const int num_assets,
    const int num_qubits,
    const T risk_aversion
) {
    extern __shared__ T shared_opt[];
    
    const int tid = threadIdx.x;
    const int asset_idx = blockIdx.x;
    
    if (asset_idx >= num_assets) return;
    
    // Quantum-enhanced mean-variance optimization
    T expected_return = T(0);
    T portfolio_risk = T(0);
    
    // Use quantum state to guide optimization
    const int state_dim = 1 << num_qubits;
    for (int i = tid; i < state_dim; i += blockDim.x) {
        T prob = norm(quantum_state[i]);
        
        // Map quantum state to portfolio configuration
        T weight = T(0);
        for (int j = 0; j < num_assets && j < num_qubits; ++j) {
            if ((i >> j) & 1) {
                weight += T(1) / num_assets;
            }
        }
        
        expected_return += prob * returns_matrix[asset_idx] * weight;
        portfolio_risk += prob * covariance_matrix[asset_idx * num_assets + asset_idx] * weight * weight;
    }
    
    // Store optimization result
    if (tid == 0) {
        optimal_weights[asset_idx] = expected_return - risk_aversion * portfolio_risk;
    }
}

// =============================================================================
// KERNEL LAUNCH HELPERS (C++ Interface)
// =============================================================================

extern "C" {
    // Single precision versions
    void launch_hadamard_gate_f32(
        void* state_vector,
        int qubit_idx,
        int num_qubits,
        int batch_size,
        cudaStream_t stream
    );
    
    void launch_cnot_gate_f32(
        void* state_vector,
        int control_qubit,
        int target_qubit,
        int num_qubits,
        int batch_size,
        cudaStream_t stream
    );
    
    void launch_rx_gate_f32(
        void* state_vector,
        float angle,
        int qubit_idx,
        int num_qubits,
        int batch_size,
        cudaStream_t stream
    );
    
    void launch_ry_gate_f32(
        void* state_vector,
        float angle,
        int qubit_idx,
        int num_qubits,
        int batch_size,
        cudaStream_t stream
    );
    
    void launch_rz_gate_f32(
        void* state_vector,
        float angle,
        int qubit_idx,
        int num_qubits,
        int batch_size,
        cudaStream_t stream
    );
    
    void launch_normalize_state_f32(
        void* state_vector,
        int state_dim,
        int batch_size,
        cudaStream_t stream
    );
    
    void launch_expectation_value_f32(
        float* expectation,
        const void* state_vector,
        const float* observable_matrix,
        int num_qubits,
        int batch_size,
        cudaStream_t stream
    );
    
    void launch_quantum_feature_map_f32(
        void* quantum_state,
        const float* market_features,
        int num_features,
        int num_qubits,
        int batch_size,
        cudaStream_t stream
    );
    
    void launch_quantum_portfolio_optimization_f32(
        float* optimal_weights,
        const void* quantum_state,
        const float* returns_matrix,
        const float* covariance_matrix,
        int num_assets,
        int num_qubits,
        float risk_aversion,
        cudaStream_t stream
    );
    
    // Double precision versions
    void launch_hadamard_gate_f64(
        void* state_vector,
        int qubit_idx,
        int num_qubits,
        int batch_size,
        cudaStream_t stream
    );
    
    // ... (additional f64 versions)
}

// Implementation of launch helpers
void launch_hadamard_gate_f32(
    void* state_vector,
    int qubit_idx,
    int num_qubits,
    int batch_size,
    cudaStream_t stream
) {
    const int state_dim = 1 << num_qubits;
    const int total_elements = state_dim * batch_size;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;
    
    hadamard_gate_kernel<float, Complex32><<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<Complex32*>(state_vector),
        qubit_idx,
        num_qubits,
        batch_size
    );
}

void launch_cnot_gate_f32(
    void* state_vector,
    int control_qubit,
    int target_qubit,
    int num_qubits,
    int batch_size,
    cudaStream_t stream
) {
    const int state_dim = 1 << num_qubits;
    const int total_elements = state_dim * batch_size;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;
    
    cnot_gate_kernel<float, Complex32><<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<Complex32*>(state_vector),
        control_qubit,
        target_qubit,
        num_qubits,
        batch_size
    );
}

void launch_normalize_state_f32(
    void* state_vector,
    int state_dim,
    int batch_size,
    cudaStream_t stream
) {
    const int block_size = 256;
    const int shared_size = block_size * sizeof(float);
    
    normalize_state_kernel<float, Complex32><<<batch_size, block_size, shared_size, stream>>>(
        reinterpret_cast<Complex32*>(state_vector),
        state_dim,
        batch_size
    );
}

void launch_expectation_value_f32(
    float* expectation,
    const void* state_vector,
    const float* observable_matrix,
    int num_qubits,
    int batch_size,
    cudaStream_t stream
) {
    const int block_size = 256;
    const int shared_size = block_size * sizeof(float);
    
    expectation_value_kernel<float, Complex32><<<batch_size, block_size, shared_size, stream>>>(
        expectation,
        reinterpret_cast<const Complex32*>(state_vector),
        observable_matrix,
        num_qubits,
        batch_size
    );
}

void launch_quantum_feature_map_f32(
    void* quantum_state,
    const float* market_features,
    int num_features,
    int num_qubits,
    int batch_size,
    cudaStream_t stream
) {
    const int state_dim = 1 << num_qubits;
    const int total_elements = state_dim * batch_size;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;
    
    quantum_feature_map_kernel<float, Complex32><<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<Complex32*>(quantum_state),
        market_features,
        num_features,
        num_qubits,
        batch_size
    );
}