// Parallel Sum Reduction Kernel
// Optimized for RDNA2 wavefront size (64 threads)

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let global_idx = gid.x;
    let local_idx = lid.x;
    let input_len = arrayLength(&input);

    // Load data into shared memory
    if global_idx < input_len {
        shared_data[local_idx] = input[global_idx];
    } else {
        shared_data[local_idx] = 0.0;
    }

    workgroupBarrier();

    // Parallel reduction in shared memory
    // Unrolled for RDNA2 wavefront optimization
    if local_idx < 128u {
        shared_data[local_idx] += shared_data[local_idx + 128u];
    }
    workgroupBarrier();

    if local_idx < 64u {
        shared_data[local_idx] += shared_data[local_idx + 64u];
    }
    workgroupBarrier();

    // Within single wavefront - no barrier needed
    if local_idx < 32u {
        shared_data[local_idx] += shared_data[local_idx + 32u];
        shared_data[local_idx] += shared_data[local_idx + 16u];
        shared_data[local_idx] += shared_data[local_idx + 8u];
        shared_data[local_idx] += shared_data[local_idx + 4u];
        shared_data[local_idx] += shared_data[local_idx + 2u];
        shared_data[local_idx] += shared_data[local_idx + 1u];
    }

    // Write workgroup result
    if local_idx == 0u {
        output[wid.x] = shared_data[0];
    }
}
