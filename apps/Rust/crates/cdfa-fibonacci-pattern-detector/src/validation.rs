//! Pattern validation utilities

use crate::types::{ValidationResult, HarmonicRatios};

/// Validate pattern ratios
pub fn validate_ratios(
    ratios: &HarmonicRatios,
    ab_xa: f64,
    bc_ab: f64, 
    cd_bc: f64,
    ad_xa: f64,
    tolerance: f64,
) -> ValidationResult {
    let mut ratio_scores = Vec::new();
    let mut failed_ratios = Vec::new();
    let mut total_deviation = 0.0;
    
    // Check AB/XA ratio
    let ab_xa_score = validate_ratio_range(ab_xa, ratios.ab_xa_min, ratios.ab_xa_max, tolerance);
    ratio_scores.push(ab_xa_score);
    if ab_xa_score == 0.0 {
        failed_ratios.push("AB/XA".to_string());
    }
    total_deviation += calculate_ratio_deviation(ab_xa, ratios.ab_xa_min, ratios.ab_xa_max);
    
    // Check BC/AB ratio
    let bc_ab_score = validate_ratio_range(bc_ab, ratios.bc_ab_min, ratios.bc_ab_max, tolerance);
    ratio_scores.push(bc_ab_score);
    if bc_ab_score == 0.0 {
        failed_ratios.push("BC/AB".to_string());
    }
    total_deviation += calculate_ratio_deviation(bc_ab, ratios.bc_ab_min, ratios.bc_ab_max);
    
    // Check CD/BC ratio  
    let cd_bc_score = validate_ratio_range(cd_bc, ratios.cd_bc_min, ratios.cd_bc_max, tolerance);
    ratio_scores.push(cd_bc_score);
    if cd_bc_score == 0.0 {
        failed_ratios.push("CD/BC".to_string());
    }
    total_deviation += calculate_ratio_deviation(cd_bc, ratios.cd_bc_min, ratios.cd_bc_max);
    
    // Check AD/XA ratio
    let ad_xa_score = validate_ratio_range(ad_xa, ratios.ad_xa_min, ratios.ad_xa_max, tolerance);
    ratio_scores.push(ad_xa_score);
    if ad_xa_score == 0.0 {
        failed_ratios.push("AD/XA".to_string());
    }
    total_deviation += calculate_ratio_deviation(ad_xa, ratios.ad_xa_min, ratios.ad_xa_max);
    
    let overall_score = ratio_scores.iter().sum::<f64>() / ratio_scores.len() as f64;
    let is_valid = failed_ratios.is_empty() && overall_score >= 0.7;
    
    ValidationResult {
        is_valid,
        score: overall_score,
        ratio_scores,
        total_deviation,
        failed_ratios,
    }
}

/// Validate if a ratio falls within the expected range
fn validate_ratio_range(ratio: f64, min_val: f64, max_val: f64, tolerance: f64) -> f64 {
    let adjusted_min = min_val - tolerance;
    let adjusted_max = max_val + tolerance;
    
    if ratio >= adjusted_min && ratio <= adjusted_max {
        // Calculate score based on how close to ideal range
        let ideal_center = (min_val + max_val) / 2.0;
        let max_distance = (max_val - min_val) / 2.0 + tolerance;
        let distance = (ratio - ideal_center).abs();
        
        if max_distance > 0.0 {
            (max_distance - distance) / max_distance
        } else {
            1.0
        }
    } else {
        0.0
    }
}

/// Calculate deviation from ideal ratio range
fn calculate_ratio_deviation(ratio: f64, min_val: f64, max_val: f64) -> f64 {
    if ratio < min_val {
        min_val - ratio
    } else if ratio > max_val {
        ratio - max_val
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ratio_validation() {
        let ratios = HarmonicRatios {
            ab_xa_min: 0.618,
            ab_xa_max: 0.618,
            bc_ab_min: 0.382,
            bc_ab_max: 0.886,
            cd_bc_min: 1.13,
            cd_bc_max: 1.618,
            ad_xa_min: 0.786,
            ad_xa_max: 0.786,
        };
        
        let result = validate_ratios(&ratios, 0.618, 0.5, 1.3, 0.786, 0.05);
        assert!(result.is_valid);
        assert!(result.score > 0.7);
    }
    
    #[test]
    fn test_invalid_ratio() {
        let ratios = HarmonicRatios {
            ab_xa_min: 0.618,
            ab_xa_max: 0.618,
            bc_ab_min: 0.382,
            bc_ab_max: 0.886,
            cd_bc_min: 1.13,
            cd_bc_max: 1.618,
            ad_xa_min: 0.786,
            ad_xa_max: 0.786,
        };
        
        let result = validate_ratios(&ratios, 0.2, 0.5, 1.3, 0.786, 0.05);
        assert!(!result.is_valid);
        assert!(!result.failed_ratios.is_empty());
    }
}