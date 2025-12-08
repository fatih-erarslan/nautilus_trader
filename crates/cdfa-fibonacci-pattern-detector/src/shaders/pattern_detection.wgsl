// WGSL compute shader for high-performance pattern detection
// Implements XABCD harmonic pattern detection on GPU

struct PatternData {
    swing_high: u32,
    swing_low: u32,
    price: f32,
    timestamp: u32,
}

struct PatternCandidate {
    x_idx: u32,
    a_idx: u32,
    b_idx: u32,
    c_idx: u32,
    d_idx: u32,
    quality: f32,
    pattern_type: u32,
    padding: u32,
}

struct PatternConfig {
    min_pattern_size: f32,
    max_pattern_size: f32,
    ratio_tolerance: f32,
    max_candidates: u32,
    data_length: u32,
    swing_high_count: u32,
    swing_low_count: u32,
    padding: u32,
}

@group(0) @binding(0) var<storage, read> pattern_data: array<PatternData>;
@group(0) @binding(1) var<storage, read> swing_indices: array<u32>;
@group(0) @binding(2) var<uniform> config: PatternConfig;
@group(0) @binding(3) var<storage, read_write> candidates: array<PatternCandidate>;
@group(0) @binding(4) var<storage, read_write> candidate_count: atomic<u32>;

// Fibonacci ratio constants
const FIBONACCI_618: f32 = 0.618;
const FIBONACCI_786: f32 = 0.786;
const FIBONACCI_382: f32 = 0.382;
const FIBONACCI_500: f32 = 0.5;
const FIBONACCI_886: f32 = 0.886;
const FIBONACCI_1272: f32 = 1.272;
const FIBONACCI_1618: f32 = 1.618;
const FIBONACCI_2618: f32 = 2.618;
const FIBONACCI_3618: f32 = 3.618;
const FIBONACCI_113: f32 = 1.13;

// Pattern type constants
const PATTERN_GARTLEY: u32 = 0u;
const PATTERN_BUTTERFLY: u32 = 1u;
const PATTERN_BAT: u32 = 2u;
const PATTERN_CRAB: u32 = 3u;
const PATTERN_SHARK: u32 = 4u;
const PATTERN_THREE_DRIVE: u32 = 5u;

// Calculate Fibonacci ratios between pattern points
fn calculate_ratios(x: f32, a: f32, b: f32, c: f32, d: f32) -> array<f32, 9> {
    let xa = abs(a - x);
    let ab = abs(b - a);
    let bc = abs(c - b);
    let cd = abs(d - c);
    let ad = abs(d - a);
    let xb = abs(b - x);
    let xc = abs(c - x);
    let xd = abs(d - x);
    
    // Avoid division by zero
    if (xa < 1e-6 || ab < 1e-6 || bc < 1e-6 || cd < 1e-6) {
        return array<f32, 9>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    
    var ratios: array<f32, 9>;
    ratios[0] = ab / xa;  // AB/XA
    ratios[1] = bc / ab;  // BC/AB
    ratios[2] = cd / bc;  // CD/BC
    ratios[3] = bc / xa;  // BC/XA
    ratios[4] = cd / ab;  // CD/AB
    ratios[5] = ad / xa;  // AD/XA
    ratios[6] = xb / xa;  // XB/XA
    ratios[7] = xc / xa;  // XC/XA
    ratios[8] = xd / xa;  // XD/XA
    
    return ratios;
}

// Calculate pattern quality for Gartley pattern
fn calculate_gartley_quality(ratios: array<f32, 9>) -> f32 {
    let ab_xa_target = FIBONACCI_618;
    let bc_ab_min = FIBONACCI_382;
    let bc_ab_max = FIBONACCI_886;
    let cd_bc_target = FIBONACCI_1272;
    
    let ab_xa_score = 1.0 - abs(ratios[0] - ab_xa_target) / config.ratio_tolerance;
    let bc_ab_score = select(0.0, 1.0, ratios[1] >= bc_ab_min && ratios[1] <= bc_ab_max);
    let cd_bc_score = 1.0 - abs(ratios[2] - cd_bc_target) / config.ratio_tolerance;
    
    return max(0.0, (ab_xa_score + bc_ab_score + cd_bc_score) / 3.0);
}

// Calculate pattern quality for Butterfly pattern
fn calculate_butterfly_quality(ratios: array<f32, 9>) -> f32 {
    let ab_xa_target = FIBONACCI_786;
    let bc_ab_min = FIBONACCI_382;
    let bc_ab_max = FIBONACCI_886;
    let cd_bc_min = FIBONACCI_1618;
    let cd_bc_max = FIBONACCI_2618;
    
    let ab_xa_score = 1.0 - abs(ratios[0] - ab_xa_target) / config.ratio_tolerance;
    let bc_ab_score = select(0.0, 1.0, ratios[1] >= bc_ab_min && ratios[1] <= bc_ab_max);
    let cd_bc_score = select(0.0, 1.0, ratios[2] >= cd_bc_min && ratios[2] <= cd_bc_max);
    
    return max(0.0, (ab_xa_score + bc_ab_score + cd_bc_score) / 3.0);
}

// Calculate pattern quality for Bat pattern
fn calculate_bat_quality(ratios: array<f32, 9>) -> f32 {
    let ab_xa_min = FIBONACCI_382;
    let ab_xa_max = FIBONACCI_500;
    let bc_ab_min = FIBONACCI_382;
    let bc_ab_max = FIBONACCI_886;
    let cd_bc_min = FIBONACCI_1618;
    let cd_bc_max = FIBONACCI_2618;
    
    let ab_xa_score = select(0.0, 1.0, ratios[0] >= ab_xa_min && ratios[0] <= ab_xa_max);
    let bc_ab_score = select(0.0, 1.0, ratios[1] >= bc_ab_min && ratios[1] <= bc_ab_max);
    let cd_bc_score = select(0.0, 1.0, ratios[2] >= cd_bc_min && ratios[2] <= cd_bc_max);
    
    return max(0.0, (ab_xa_score + bc_ab_score + cd_bc_score) / 3.0);
}

// Calculate pattern quality for Crab pattern
fn calculate_crab_quality(ratios: array<f32, 9>) -> f32 {
    let ab_xa_min = FIBONACCI_382;
    let ab_xa_max = FIBONACCI_618;
    let bc_ab_min = FIBONACCI_382;
    let bc_ab_max = FIBONACCI_886;
    let cd_bc_min = FIBONACCI_2618;
    let cd_bc_max = FIBONACCI_3618;
    
    let ab_xa_score = select(0.0, 1.0, ratios[0] >= ab_xa_min && ratios[0] <= ab_xa_max);
    let bc_ab_score = select(0.0, 1.0, ratios[1] >= bc_ab_min && ratios[1] <= bc_ab_max);
    let cd_bc_score = select(0.0, 1.0, ratios[2] >= cd_bc_min && ratios[2] <= cd_bc_max);
    
    return max(0.0, (ab_xa_score + bc_ab_score + cd_bc_score) / 3.0);
}

// Calculate pattern quality for Shark pattern
fn calculate_shark_quality(ratios: array<f32, 9>) -> f32 {
    let ab_xa_target = FIBONACCI_500;
    let bc_ab_min = FIBONACCI_113;
    let bc_ab_max = FIBONACCI_1618;
    let cd_bc_target = FIBONACCI_1618;
    
    let ab_xa_score = 1.0 - abs(ratios[0] - ab_xa_target) / config.ratio_tolerance;
    let bc_ab_score = select(0.0, 1.0, ratios[1] >= bc_ab_min && ratios[1] <= bc_ab_max);
    let cd_bc_score = 1.0 - abs(ratios[2] - cd_bc_target) / config.ratio_tolerance;
    
    return max(0.0, (ab_xa_score + bc_ab_score + cd_bc_score) / 3.0);
}

// Calculate pattern quality for Three Drive pattern
fn calculate_three_drive_quality(ratios: array<f32, 9>) -> f32 {
    let ab_xa_target = FIBONACCI_618;
    let bc_ab_target = FIBONACCI_1272;
    let cd_bc_target = FIBONACCI_786;
    
    let ab_xa_score = 1.0 - abs(ratios[0] - ab_xa_target) / config.ratio_tolerance;
    let bc_ab_score = 1.0 - abs(ratios[1] - bc_ab_target) / config.ratio_tolerance;
    let cd_bc_score = 1.0 - abs(ratios[2] - cd_bc_target) / config.ratio_tolerance;
    
    return max(0.0, (ab_xa_score + bc_ab_score + cd_bc_score) / 3.0);
}

// Validate pattern sequence
fn validate_pattern_sequence(x: f32, a: f32, b: f32, c: f32, d: f32) -> bool {
    // Check for valid XABCD sequence
    // Bullish: X < A > B < C > D
    // Bearish: X > A < B > C < D
    
    let bullish = (x < a) && (a > b) && (b < c) && (c > d);
    let bearish = (x > a) && (a < b) && (b > c) && (c < d);
    
    return bullish || bearish;
}

// Check if indices form a valid pattern size
fn validate_pattern_size(x: f32, a: f32, b: f32, c: f32, d: f32) -> bool {
    let prices = array<f32, 5>(x, a, b, c, d);
    let mut min_price = prices[0];
    let mut max_price = prices[0];
    
    for (var i: u32 = 1u; i < 5u; i++) {
        min_price = min(min_price, prices[i]);
        max_price = max(max_price, prices[i]);
    }
    
    let range = max_price - min_price;
    let avg_price = (min_price + max_price) / 2.0;
    let size_ratio = range / avg_price;
    
    return size_ratio >= config.min_pattern_size && size_ratio <= config.max_pattern_size;
}

// Main compute shader entry point
@compute @workgroup_size(256)
fn find_pattern_candidates(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;
    
    // Get swing high count (first half of swing_indices)
    let swing_high_count = config.swing_high_count;
    
    if (thread_id >= swing_high_count) {
        return;
    }
    
    // Get X index (swing high)
    let x_idx = swing_indices[thread_id];
    if (x_idx >= config.data_length) {
        return;
    }
    
    let x_price = pattern_data[x_idx].price;
    
    // Find A points (swing lows after X)
    for (var a_i: u32 = 0u; a_i < config.swing_low_count; a_i++) {
        let a_idx = swing_indices[swing_high_count + a_i] & 0x7FFFFFFFu; // Remove high bit
        if (a_idx <= x_idx || a_idx >= config.data_length) {
            continue;
        }
        
        let a_price = pattern_data[a_idx].price;
        
        // Find B points (swing highs after A)
        for (var b_i: u32 = thread_id + 1u; b_i < swing_high_count; b_i++) {
            let b_idx = swing_indices[b_i];
            if (b_idx <= a_idx || b_idx >= config.data_length) {
                continue;
            }
            
            let b_price = pattern_data[b_idx].price;
            
            // Find C points (swing lows after B)
            for (var c_i: u32 = 0u; c_i < config.swing_low_count; c_i++) {
                let c_idx = swing_indices[swing_high_count + c_i] & 0x7FFFFFFFu;
                if (c_idx <= b_idx || c_idx >= config.data_length) {
                    continue;
                }
                
                let c_price = pattern_data[c_idx].price;
                
                // Find D points (swing highs after C)
                for (var d_i: u32 = b_i + 1u; d_i < swing_high_count; d_i++) {
                    let d_idx = swing_indices[d_i];
                    if (d_idx <= c_idx || d_idx >= config.data_length) {
                        continue;
                    }
                    
                    let d_price = pattern_data[d_idx].price;
                    
                    // Validate pattern sequence
                    if (!validate_pattern_sequence(x_price, a_price, b_price, c_price, d_price)) {
                        continue;
                    }
                    
                    // Validate pattern size
                    if (!validate_pattern_size(x_price, a_price, b_price, c_price, d_price)) {
                        continue;
                    }
                    
                    // Calculate ratios
                    let ratios = calculate_ratios(x_price, a_price, b_price, c_price, d_price);
                    
                    // Test each pattern type
                    var best_quality = 0.0;
                    var best_pattern = 0u;
                    
                    // Gartley
                    let gartley_quality = calculate_gartley_quality(ratios);
                    if (gartley_quality > best_quality && gartley_quality >= 0.8) {
                        best_quality = gartley_quality;
                        best_pattern = PATTERN_GARTLEY;
                    }
                    
                    // Butterfly
                    let butterfly_quality = calculate_butterfly_quality(ratios);
                    if (butterfly_quality > best_quality && butterfly_quality >= 0.7) {
                        best_quality = butterfly_quality;
                        best_pattern = PATTERN_BUTTERFLY;
                    }
                    
                    // Bat
                    let bat_quality = calculate_bat_quality(ratios);
                    if (bat_quality > best_quality && bat_quality >= 0.75) {
                        best_quality = bat_quality;
                        best_pattern = PATTERN_BAT;
                    }
                    
                    // Crab
                    let crab_quality = calculate_crab_quality(ratios);
                    if (crab_quality > best_quality && crab_quality >= 0.7) {
                        best_quality = crab_quality;
                        best_pattern = PATTERN_CRAB;
                    }
                    
                    // Shark
                    let shark_quality = calculate_shark_quality(ratios);
                    if (shark_quality > best_quality && shark_quality >= 0.75) {
                        best_quality = shark_quality;
                        best_pattern = PATTERN_SHARK;
                    }
                    
                    // Three Drive
                    let three_drive_quality = calculate_three_drive_quality(ratios);
                    if (three_drive_quality > best_quality && three_drive_quality >= 0.8) {
                        best_quality = three_drive_quality;
                        best_pattern = PATTERN_THREE_DRIVE;
                    }
                    
                    // Store candidate if quality is good enough
                    if (best_quality > 0.0) {
                        let candidate_idx = atomicAdd(&candidate_count, 1u);
                        if (candidate_idx < config.max_candidates) {
                            candidates[candidate_idx] = PatternCandidate(
                                x_idx,
                                a_idx,
                                b_idx,
                                c_idx,
                                d_idx,
                                best_quality,
                                best_pattern,
                                0u
                            );
                        }
                    }
                }
            }
        }
    }
}