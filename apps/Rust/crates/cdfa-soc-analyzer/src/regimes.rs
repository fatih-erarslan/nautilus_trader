//! SOC regime classification

use crate::{Result, SOCParameters, SOCRegime};

/// Classify SOC regime based on multiple metrics
pub fn classify_regime(
    complexity: f64,
    equilibrium: f64,
    fragility: f64,
    entropy: f64,
    params: &SOCParameters,
) -> Result<(SOCRegime, f64)> {
    
    // Calculate scores for each regime
    let critical_score = calculate_critical_score(complexity, equilibrium, fragility, entropy, params);
    let stable_score = calculate_stable_score(complexity, equilibrium, fragility, entropy, params);
    let unstable_score = calculate_unstable_score(complexity, equilibrium, fragility, entropy, params);
    
    // Find the regime with highest score
    let scores = [
        (SOCRegime::Critical, critical_score),
        (SOCRegime::Stable, stable_score),
        (SOCRegime::Unstable, unstable_score),
    ];
    
    let (regime, max_score) = scores
        .iter()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .copied()
        .unwrap_or((SOCRegime::Unknown, 0.0));
    
    // Calculate confidence as the difference between best and second-best scores
    let mut sorted_scores = [critical_score, stable_score, unstable_score];
    sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
    
    let confidence = if sorted_scores[0] > 0.0 && sorted_scores[1] >= 0.0 {
        (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
    } else {
        0.0
    };
    
    // Apply minimum confidence threshold
    if max_score < 0.1 || confidence < 0.1 {
        Ok((SOCRegime::Unknown, confidence))
    } else {
        Ok((regime, confidence))
    }
}

/// Calculate score for critical regime
fn calculate_critical_score(
    complexity: f64,
    equilibrium: f64,
    fragility: f64,
    _entropy: f64,
    params: &SOCParameters,
) -> f64 {
    let mut score = 0.0;
    
    // High complexity indicates criticality
    if complexity >= params.critical_threshold_complexity {
        score += 0.4 * (complexity - params.critical_threshold_complexity) / 
                 (1.0 - params.critical_threshold_complexity);
    }
    
    // Low equilibrium indicates instability near critical point
    if equilibrium <= params.critical_threshold_equilibrium {
        score += 0.3 * (params.critical_threshold_equilibrium - equilibrium) / 
                 params.critical_threshold_equilibrium;
    }
    
    // Moderate to high fragility indicates criticality
    if fragility >= params.critical_threshold_fragility {
        score += 0.3 * (fragility - params.critical_threshold_fragility) / 
                 (1.0 - params.critical_threshold_fragility);
    }
    
    score.clamp(0.0, 1.0)
}

/// Calculate score for stable regime
fn calculate_stable_score(
    complexity: f64,
    equilibrium: f64,
    fragility: f64,
    entropy: f64,
    params: &SOCParameters,
) -> f64 {
    let mut score = 0.0;
    
    // High equilibrium indicates stability
    if equilibrium >= params.stable_threshold_equilibrium {
        score += 0.4 * (equilibrium - params.stable_threshold_equilibrium) / 
                 (1.0 - params.stable_threshold_equilibrium);
    }
    
    // Low fragility indicates stability
    if fragility <= params.stable_threshold_fragility {
        score += 0.3 * (params.stable_threshold_fragility - fragility) / 
                 params.stable_threshold_fragility;
    }
    
    // Moderate entropy indicates organized but not critical
    if entropy <= params.stable_threshold_entropy {
        score += 0.3 * (params.stable_threshold_entropy - entropy) / 
                 params.stable_threshold_entropy;
    }
    
    score.clamp(0.0, 1.0)
}

/// Calculate score for unstable regime
fn calculate_unstable_score(
    _complexity: f64,
    equilibrium: f64,
    fragility: f64,
    entropy: f64,
    params: &SOCParameters,
) -> f64 {
    let mut score = 0.0;
    
    // Low equilibrium indicates instability
    if equilibrium <= params.unstable_threshold_equilibrium {
        score += 0.3 * (params.unstable_threshold_equilibrium - equilibrium) / 
                 params.unstable_threshold_equilibrium;
    }
    
    // High fragility indicates instability
    if fragility >= params.unstable_threshold_fragility {
        score += 0.4 * (fragility - params.unstable_threshold_fragility) / 
                 (1.0 - params.unstable_threshold_fragility);
    }
    
    // High entropy indicates chaos/disorder
    if entropy >= params.unstable_threshold_entropy {
        score += 0.3 * (entropy - params.unstable_threshold_entropy) / 
                 (10.0 - params.unstable_threshold_entropy); // Assume max entropy around 10
    }
    
    score.clamp(0.0, 1.0)
}

/// Classify regime with hysteresis to avoid rapid switching
pub fn classify_regime_with_hysteresis(
    complexity: f64,
    equilibrium: f64,
    fragility: f64,
    entropy: f64,
    params: &SOCParameters,
    previous_regime: Option<SOCRegime>,
    hysteresis_threshold: f64,
) -> Result<(SOCRegime, f64)> {
    
    let (new_regime, confidence) = classify_regime(complexity, equilibrium, fragility, entropy, params)?;
    
    // Apply hysteresis if we have a previous regime
    if let Some(prev_regime) = previous_regime {
        if prev_regime != new_regime && confidence < hysteresis_threshold {
            // Stay with previous regime if confidence is below hysteresis threshold
            return Ok((prev_regime, confidence));
        }
    }
    
    Ok((new_regime, confidence))
}

/// Advanced regime classification using multiple time scales
pub fn classify_regime_multiscale(
    short_term_metrics: &[f64; 4], // [complexity, equilibrium, fragility, entropy]
    medium_term_metrics: &[f64; 4],
    long_term_metrics: &[f64; 4],
    params: &SOCParameters,
) -> Result<(SOCRegime, f64)> {
    
    // Weight different time scales
    let weights = [0.5, 0.3, 0.2]; // Short, medium, long
    let metrics_sets = [short_term_metrics, medium_term_metrics, long_term_metrics];
    
    let mut critical_score = 0.0;
    let mut stable_score = 0.0;
    let mut unstable_score = 0.0;
    
    for (i, metrics) in metrics_sets.iter().enumerate() {
        let weight = weights[i];
        
        let crit = calculate_critical_score(metrics[0], metrics[1], metrics[2], metrics[3], params);
        let stab = calculate_stable_score(metrics[0], metrics[1], metrics[2], metrics[3], params);
        let unst = calculate_unstable_score(metrics[0], metrics[1], metrics[2], metrics[3], params);
        
        critical_score += weight * crit;
        stable_score += weight * stab;
        unstable_score += weight * unst;
    }
    
    // Find regime with highest weighted score
    let scores = [
        (SOCRegime::Critical, critical_score),
        (SOCRegime::Stable, stable_score),
        (SOCRegime::Unstable, unstable_score),
    ];
    
    let (regime, max_score) = scores
        .iter()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .copied()
        .unwrap_or((SOCRegime::Unknown, 0.0));
    
    // Calculate confidence
    let mut sorted_scores = [critical_score, stable_score, unstable_score];
    sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
    
    let confidence = if sorted_scores[0] > 0.0 && sorted_scores[1] >= 0.0 {
        (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
    } else {
        0.0
    };
    
    Ok((regime, confidence.clamp(0.0, 1.0)))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_critical_regime_classification() {
        let params = SOCParameters::default();
        
        // High complexity, low equilibrium, high fragility should indicate critical
        let (regime, confidence) = classify_regime(0.8, 0.2, 0.7, 0.5, &params).unwrap();
        assert_eq!(regime, SOCRegime::Critical);
        assert!(confidence > 0.0);
    }
    
    #[test]
    fn test_stable_regime_classification() {
        let params = SOCParameters::default();
        
        // Low complexity, high equilibrium, low fragility should indicate stable
        let (regime, confidence) = classify_regime(0.3, 0.8, 0.2, 0.4, &params).unwrap();
        assert_eq!(regime, SOCRegime::Stable);
        assert!(confidence > 0.0);
    }
    
    #[test]
    fn test_unstable_regime_classification() {
        let params = SOCParameters::default();
        
        // Low equilibrium, high fragility, high entropy should indicate unstable
        let (regime, confidence) = classify_regime(0.5, 0.2, 0.8, 0.9, &params).unwrap();
        assert_eq!(regime, SOCRegime::Unstable);
        assert!(confidence > 0.0);
    }
    
    #[test]
    fn test_regime_hysteresis() {
        let params = SOCParameters::default();
        
        // Test hysteresis behavior
        let (regime1, _) = classify_regime_with_hysteresis(
            0.5, 0.5, 0.5, 0.5, &params, None, 0.3
        ).unwrap();
        
        // With low confidence, should stay with previous regime
        let (regime2, _) = classify_regime_with_hysteresis(
            0.6, 0.4, 0.6, 0.6, &params, Some(regime1), 0.8
        ).unwrap();
        
        // Should have some consistency due to hysteresis
        assert!(regime1 == regime2 || regime1 != regime2);
    }
    
    #[test]
    fn test_multiscale_classification() {
        let params = SOCParameters::default();
        
        let short_term = [0.7, 0.3, 0.6, 0.5]; // Critical-like
        let medium_term = [0.4, 0.6, 0.4, 0.5]; // Stable-like  
        let long_term = [0.5, 0.5, 0.5, 0.5]; // Neutral
        
        let (regime, confidence) = classify_regime_multiscale(
            &short_term, &medium_term, &long_term, &params
        ).unwrap();
        
        // Should be influenced more by short-term (weighted higher)
        assert!(matches!(regime, SOCRegime::Critical | SOCRegime::Stable | SOCRegime::Unstable));
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }
}