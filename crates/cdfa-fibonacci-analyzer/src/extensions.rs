//! Fibonacci extension calculations

use wide::f64x4;

/// Calculator for Fibonacci extensions beyond 100%
#[derive(Debug)]
pub struct ExtensionCalculator {
    extension_levels: Vec<f64>,
}

impl ExtensionCalculator {
    pub fn new() -> Self {
        Self {
            extension_levels: vec![
                1.236, 1.382, 1.618, 2.0, 2.236, 2.618, 3.0, 3.618, 4.236
            ],
        }
    }
    
    /// Calculate extension levels from a swing
    pub fn calculate_extensions(&self, swing_low: f64, swing_high: f64, base: f64) -> Vec<f64> {
        let range = swing_high - swing_low;
        let mut extensions = Vec::with_capacity(self.extension_levels.len());
        
        // Process extensions in groups of 4 for SIMD
        let chunks = self.extension_levels.chunks_exact(4);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            let levels = f64x4::new([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let range_vec = f64x4::splat(range);
            let base_vec = f64x4::splat(base);
            
            let ext_values = base_vec + levels * range_vec;
            extensions.extend_from_slice(&ext_values.to_array());
        }
        
        // Process remaining extensions
        for &level in remainder {
            extensions.push(base + range * level);
        }
        
        extensions
    }
    
    /// Calculate projection levels from ABC pattern
    pub fn calculate_projections(&self, a: f64, b: f64, c: f64) -> Vec<(f64, &'static str)> {
        let ab_range = (b - a).abs();
        let bc_range = (c - b).abs();
        
        let mut projections = Vec::new();
        
        // Standard projections
        projections.push((c + ab_range * 0.618, "61.8% AB"));
        projections.push((c + ab_range * 1.0, "100% AB"));
        projections.push((c + ab_range * 1.618, "161.8% AB"));
        
        // BC projections
        projections.push((c + bc_range * 1.618, "161.8% BC"));
        projections.push((c + bc_range * 2.618, "261.8% BC"));
        
        projections
    }
    
    /// Find nearest extension level
    pub fn find_nearest_extension(&self, price: f64, swing_low: f64, swing_high: f64, base: f64) -> Option<(f64, f64)> {
        let extensions = self.calculate_extensions(swing_low, swing_high, base);
        
        extensions.into_iter()
            .map(|ext| (ext, (price - ext).abs()))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    }
}

impl Default for ExtensionCalculator {
    fn default() -> Self {
        Self::new()
    }
}