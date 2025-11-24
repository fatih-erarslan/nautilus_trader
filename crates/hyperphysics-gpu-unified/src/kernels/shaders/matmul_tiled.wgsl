// Tiled Matrix Multiplication Kernel
// Optimized for RDNA2 with 16x16 tiles and shared memory

struct MatrixDims {
    M: u32, // Rows of A, rows of C
    N: u32, // Cols of B, cols of C
    K: u32, // Cols of A, rows of B
    _padding: u32,
}

const TILE_SIZE: u32 = 16u;

@group(0) @binding(0) var<uniform> dims: MatrixDims;
@group(0) @binding(1) var<storage, read> mat_a: array<f32>;
@group(0) @binding(2) var<storage, read> mat_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> mat_c: array<f32>;

var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let row = gid.y;
    let col = gid.x;
    let local_row = lid.y;
    let local_col = lid.x;

    var sum = 0.0;

    // Number of tiles needed to cover K dimension
    let num_tiles = (dims.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t++) {
        // Collaborative loading of tiles into shared memory
        let a_row = row;
        let a_col = t * TILE_SIZE + local_col;
        let b_row = t * TILE_SIZE + local_row;
        let b_col = col;

        // Load A tile (with bounds check)
        if a_row < dims.M && a_col < dims.K {
            tile_a[local_row][local_col] = mat_a[a_row * dims.K + a_col];
        } else {
            tile_a[local_row][local_col] = 0.0;
        }

        // Load B tile (with bounds check)
        if b_row < dims.K && b_col < dims.N {
            tile_b[local_row][local_col] = mat_b[b_row * dims.N + b_col];
        } else {
            tile_b[local_row][local_col] = 0.0;
        }

        workgroupBarrier();

        // Compute partial dot product for this tile
        for (var k = 0u; k < TILE_SIZE; k++) {
            sum += tile_a[local_row][k] * tile_b[k][local_col];
        }

        workgroupBarrier();
    }

    // Write result (with bounds check)
    if row < dims.M && col < dims.N {
        mat_c[row * dims.N + col] = sum;
    }
}
