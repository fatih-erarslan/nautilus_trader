//! Harmonic pattern configurations

use crate::types::{PatternConfig, PatternType, HarmonicRatios};

/// Gartley pattern configuration
pub fn gartley_config() -> PatternConfig {
    PatternConfig {
        pattern_type: PatternType::Gartley,
        ratios: HarmonicRatios {
            ab_xa_min: 0.618,
            ab_xa_max: 0.618,
            bc_ab_min: 0.382,
            bc_ab_max: 0.886,
            cd_bc_min: 1.13,
            cd_bc_max: 1.618,
            ad_xa_min: 0.786,
            ad_xa_max: 0.786,
        },
        tolerance: 0.05,
        min_pattern_size: 20,
    }
}

/// Butterfly pattern configuration  
pub fn butterfly_config() -> PatternConfig {
    PatternConfig {
        pattern_type: PatternType::Butterfly,
        ratios: HarmonicRatios {
            ab_xa_min: 0.786,
            ab_xa_max: 0.786,
            bc_ab_min: 0.382,
            bc_ab_max: 0.886,
            cd_bc_min: 1.618,
            cd_bc_max: 2.618,
            ad_xa_min: 1.27,
            ad_xa_max: 1.618,
        },
        tolerance: 0.05,
        min_pattern_size: 20,
    }
}

/// Bat pattern configuration
pub fn bat_config() -> PatternConfig {
    PatternConfig {
        pattern_type: PatternType::Bat,
        ratios: HarmonicRatios {
            ab_xa_min: 0.382,
            ab_xa_max: 0.5,
            bc_ab_min: 0.382,
            bc_ab_max: 0.886,
            cd_bc_min: 1.618,
            cd_bc_max: 2.618,
            ad_xa_min: 0.886,
            ad_xa_max: 0.886,
        },
        tolerance: 0.05,
        min_pattern_size: 20,
    }
}

/// Crab pattern configuration
pub fn crab_config() -> PatternConfig {
    PatternConfig {
        pattern_type: PatternType::Crab,
        ratios: HarmonicRatios {
            ab_xa_min: 0.382,
            ab_xa_max: 0.618,
            bc_ab_min: 0.382,
            bc_ab_max: 0.886,
            cd_bc_min: 2.24,
            cd_bc_max: 3.618,
            ad_xa_min: 1.618,
            ad_xa_max: 1.618,
        },
        tolerance: 0.05,
        min_pattern_size: 20,
    }
}

/// Shark pattern configuration
pub fn shark_config() -> PatternConfig {
    PatternConfig {
        pattern_type: PatternType::Shark,
        ratios: HarmonicRatios {
            ab_xa_min: 0.382,
            ab_xa_max: 0.618,
            bc_ab_min: 1.13,
            bc_ab_max: 1.618,
            cd_bc_min: 1.618,
            cd_bc_max: 2.24,
            ad_xa_min: 0.886,
            ad_xa_max: 1.13,
        },
        tolerance: 0.05,
        min_pattern_size: 20,
    }
}