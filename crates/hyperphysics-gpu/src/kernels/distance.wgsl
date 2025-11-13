// Hyperbolic distance calculation shader
// Implements d_H(p,q) = acosh(1 + 2||p-q||² / ((1-||p||²)(1-||q||²)))

@group(0) @binding(0) var<storage, read> points_a: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read> points_b: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read_write> distances: array<f32>;

// Workgroup size optimized for GPU warps/wavefronts
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    // Bounds check
    if (index >= arrayLength(&points_a)) {
        return;
    }
    
    let p = points_a[index];
    let q = points_b[index];
    
    // Ensure points are within Poincaré disk (||p|| < 1, ||q|| < 1)
    let p_norm_sq = dot(p, p);
    let q_norm_sq = dot(q, q);
    
    // Clamp to prevent numerical issues at boundary
    let eps = 1e-10f;
    let p_clamped_norm_sq = min(p_norm_sq, 1.0 - eps);
    let q_clamped_norm_sq = min(q_norm_sq, 1.0 - eps);
    
    // Calculate ||p - q||²
    let diff = p - q;
    let diff_norm_sq = dot(diff, diff);
    
    // Hyperbolic distance formula
    // d_H(p,q) = acosh(1 + 2||p-q||² / ((1-||p||²)(1-||q||²)))
    let denominator = (1.0 - p_clamped_norm_sq) * (1.0 - q_clamped_norm_sq);
    let argument = 1.0 + 2.0 * diff_norm_sq / denominator;
    
    // Ensure argument >= 1 for acosh
    let safe_argument = max(argument, 1.0 + eps);
    
    // acosh(x) = ln(x + sqrt(x² - 1))
    let distance = log(safe_argument + sqrt(safe_argument * safe_argument - 1.0));
    
    distances[index] = distance;
}

// Batch distance calculation for lattice neighbors
@group(1) @binding(0) var<storage, read> lattice_points: array<vec3<f32>>;
@group(1) @binding(1) var<storage, read> neighbor_indices: array<vec2<u32>>; // pairs (i, j)
@group(1) @binding(2) var<storage, read_write> neighbor_distances: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn compute_neighbor_distances(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= arrayLength(&neighbor_indices)) {
        return;
    }
    
    let pair = neighbor_indices[index];
    let i = pair.x;
    let j = pair.y;
    
    // Bounds check
    if (i >= arrayLength(&lattice_points) || j >= arrayLength(&lattice_points)) {
        return;
    }
    
    let p = lattice_points[i];
    let q = lattice_points[j];
    
    // Same distance calculation as above
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
    
    let distance = log(safe_argument + sqrt(safe_argument * safe_argument - 1.0));
    
    neighbor_distances[index] = distance;
}

// Geodesic distance along path (approximate)
@group(2) @binding(0) var<storage, read> path_points: array<vec3<f32>>;
@group(2) @binding(1) var<storage, read_write> path_length: array<f32>; // single element

@compute @workgroup_size(1, 1, 1)
fn compute_geodesic_length(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let num_points = arrayLength(&path_points);
    
    if (num_points < 2u) {
        path_length[0] = 0.0;
        return;
    }
    
    var total_length = 0.0f;
    
    // Sum distances between consecutive points
    for (var i = 0u; i < num_points - 1u; i = i + 1u) {
        let p = path_points[i];
        let q = path_points[i + 1u];
        
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
        
        let segment_length = log(safe_argument + sqrt(safe_argument * safe_argument - 1.0));
        total_length = total_length + segment_length;
    }
    
    path_length[0] = total_length;
}
