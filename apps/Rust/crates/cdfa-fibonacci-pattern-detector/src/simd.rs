//! SIMD optimizations for pattern detection

#[cfg(feature = "simd")]
use wide::f32x8;

/// SIMD-accelerated ratio calculation
#[cfg(feature = "simd")]
pub fn simd_calculate_ratios(prices: &[f32]) -> Vec<f32> {
    let mut ratios = Vec::with_capacity(prices.len() - 1);
    let chunks = prices.len() / 8;
    
    for i in 0..chunks {
        let start_idx = i * 8;
        if start_idx + 8 < prices.len() {
            let current = f32x8::new([
                prices[start_idx], prices[start_idx + 1], prices[start_idx + 2], prices[start_idx + 3],
                prices[start_idx + 4], prices[start_idx + 5], prices[start_idx + 6], prices[start_idx + 7],
            ]);
            
            let next = f32x8::new([
                prices[start_idx + 1], prices[start_idx + 2], prices[start_idx + 3], prices[start_idx + 4],
                prices[start_idx + 5], prices[start_idx + 6], prices[start_idx + 7], 
                if start_idx + 8 < prices.len() { prices[start_idx + 8] } else { prices[start_idx + 7] },
            ]);
            
            let ratio = next / current;
            let ratio_array = ratio.to_array();
            
            for &r in &ratio_array[..7] {
                if r.is_finite() && r > 0.0 {
                    ratios.push(r);
                }
            }
        }
    }
    
    // Handle remaining elements
    for i in (chunks * 8)..(prices.len() - 1) {
        if prices[i] != 0.0 {
            let ratio = prices[i + 1] / prices[i];
            if ratio.is_finite() && ratio > 0.0 {
                ratios.push(ratio);
            }
        }
    }
    
    ratios
}

/// Fallback scalar ratio calculation
#[cfg(not(feature = "simd"))]
pub fn simd_calculate_ratios(prices: &[f32]) -> Vec<f32> {
    scalar_calculate_ratios(prices)
}

/// Scalar implementation for ratio calculation
pub fn scalar_calculate_ratios(prices: &[f32]) -> Vec<f32> {
    let mut ratios = Vec::with_capacity(prices.len() - 1);
    
    for i in 0..(prices.len() - 1) {
        if prices[i] != 0.0 {
            let ratio = prices[i + 1] / prices[i];
            if ratio.is_finite() && ratio > 0.0 {
                ratios.push(ratio);
            }
        }
    }
    
    ratios
}

/// SIMD-accelerated pattern matching
#[cfg(feature = "simd")]
pub fn simd_pattern_match(
    pattern_ratios: &[f32; 4],
    candidate_ratios: &[f32; 4],
    tolerance: f32,
) -> bool {
    let pattern = f32x8::new([
        pattern_ratios[0], pattern_ratios[1], pattern_ratios[2], pattern_ratios[3],
        0.0, 0.0, 0.0, 0.0,
    ]);
    
    let candidate = f32x8::new([
        candidate_ratios[0], candidate_ratios[1], candidate_ratios[2], candidate_ratios[3],
        0.0, 0.0, 0.0, 0.0,
    ]);
    
    let tolerance_vec = f32x8::splat(tolerance);
    let diff = (pattern - candidate).abs();
    let within_tolerance = diff.cmp_le(tolerance_vec);
    
    // Check first 4 elements
    let mask_array = within_tolerance.to_array();
    mask_array[0] != 0 && mask_array[1] != 0 && mask_array[2] != 0 && mask_array[3] != 0
}

/// Fallback scalar pattern matching
#[cfg(not(feature = "simd"))]
pub fn simd_pattern_match(
    pattern_ratios: &[f32; 4],
    candidate_ratios: &[f32; 4],
    tolerance: f32,
) -> bool {
    scalar_pattern_match(pattern_ratios, candidate_ratios, tolerance)
}

/// Scalar implementation for pattern matching
pub fn scalar_pattern_match(
    pattern_ratios: &[f32; 4],
    candidate_ratios: &[f32; 4],
    tolerance: f32,
) -> bool {
    for i in 0..4 {
        if (pattern_ratios[i] - candidate_ratios[i]).abs() > tolerance {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ratio_calculation() {
        let prices = vec![1.0, 1.1, 1.2, 1.0, 0.9];
        let ratios = simd_calculate_ratios(&prices);
        
        assert_eq!(ratios.len(), 4);
        assert!((ratios[0] - 1.1).abs() < f32::EPSILON);
        assert!((ratios[1] - (1.2 / 1.1)).abs() < 0.001);
    }
    
    #[test]
    fn test_pattern_matching() {
        let pattern = [0.618, 0.786, 1.27, 0.786];
        let candidate = [0.62, 0.78, 1.25, 0.79];
        
        assert!(simd_pattern_match(&pattern, &candidate, 0.05));
        assert!(!simd_pattern_match(&pattern, &candidate, 0.01));
    }
}