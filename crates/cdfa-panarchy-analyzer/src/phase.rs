//! Phase identification and regime classification

use crate::types::{MarketPhase, PCRComponents, PhaseScores, PanarchyParameters};
use crate::simd::simd_mean;
use crate::PanarchyError;
use std::collections::VecDeque;

/// Phase identifier with hysteresis to prevent oscillation
pub struct PhaseIdentifier {
    params: PanarchyParameters,
    phase_history: VecDeque<MarketPhase>,
    score_history: VecDeque<PhaseScores>,
}

impl PhaseIdentifier {
    pub fn new(params: PanarchyParameters) -> Self {
        Self {
            params,
            phase_history: VecDeque::with_capacity(10),
            score_history: VecDeque::with_capacity(10),
        }
    }
    
    /// Identify market phase from PCR components with hysteresis
    pub fn identify_phase(&mut self, pcr: &PCRComponents) -> MarketPhase {
        // Calculate raw phase scores
        let mut scores = pcr.phase_scores();
        scores.normalize();
        
        // Store score history
        if self.score_history.len() >= 10 {
            self.score_history.pop_front();
        }
        self.score_history.push_back(scores);
        
        // Get current phase from scores
        let current_phase = scores.dominant_phase();
        
        // Apply hysteresis if we have history
        let final_phase = if let Some(&prev_phase) = self.phase_history.back() {
            self.apply_hysteresis(current_phase, prev_phase, &scores)
        } else {
            current_phase
        };
        
        // Update history
        if self.phase_history.len() >= 10 {
            self.phase_history.pop_front();
        }
        self.phase_history.push_back(final_phase);
        
        final_phase
    }
    
    /// Apply hysteresis to prevent phase oscillation
    fn apply_hysteresis(
        &self,
        current_phase: MarketPhase,
        prev_phase: MarketPhase,
        scores: &PhaseScores,
    ) -> MarketPhase {
        if current_phase == prev_phase {
            return current_phase;
        }
        
        // Get scores for current and previous phases
        let current_score = match current_phase {
            MarketPhase::Growth => scores.growth,
            MarketPhase::Conservation => scores.conservation,
            MarketPhase::Release => scores.release,
            MarketPhase::Reorganization => scores.reorganization,
            MarketPhase::Unknown => 0.0,
        };
        
        let prev_score = match prev_phase {
            MarketPhase::Growth => scores.growth,
            MarketPhase::Conservation => scores.conservation,
            MarketPhase::Release => scores.release,
            MarketPhase::Reorganization => scores.reorganization,
            MarketPhase::Unknown => 0.0,
        };
        
        // Check if change is significant enough
        if current_score < self.params.hysteresis_min_score_threshold ||
           current_score < prev_score + self.params.hysteresis_min_score_diff {
            // Not significant enough, keep previous phase
            prev_phase
        } else {
            // Significant change, accept new phase
            current_phase
        }
    }
    
    /// Get smoothed phase using history
    pub fn get_smoothed_phase(&self) -> MarketPhase {
        if self.phase_history.is_empty() {
            return MarketPhase::Unknown;
        }
        
        // Count occurrences of each phase in recent history
        let mut phase_counts = [0u32; 5]; // 5 phases including Unknown
        
        let window_size = self.params.regime_smoothing_window.min(self.phase_history.len());
        let start = self.phase_history.len().saturating_sub(window_size);
        
        for i in start..self.phase_history.len() {
            match self.phase_history[i] {
                MarketPhase::Growth => phase_counts[0] += 1,
                MarketPhase::Conservation => phase_counts[1] += 1,
                MarketPhase::Release => phase_counts[2] += 1,
                MarketPhase::Reorganization => phase_counts[3] += 1,
                MarketPhase::Unknown => phase_counts[4] += 1,
            }
        }
        
        // Find most common phase
        let mut max_count = phase_counts[0];
        let mut dominant_phase = MarketPhase::Growth;
        
        if phase_counts[1] > max_count {
            max_count = phase_counts[1];
            dominant_phase = MarketPhase::Conservation;
        }
        if phase_counts[2] > max_count {
            max_count = phase_counts[2];
            dominant_phase = MarketPhase::Release;
        }
        if phase_counts[3] > max_count {
            max_count = phase_counts[3];
            dominant_phase = MarketPhase::Reorganization;
        }
        if phase_counts[4] > max_count {
            dominant_phase = MarketPhase::Unknown;
        }
        
        dominant_phase
    }
    
    /// Get average phase scores over recent history
    pub fn get_average_scores(&self) -> PhaseScores {
        if self.score_history.is_empty() {
            return PhaseScores {
                growth: 0.25,
                conservation: 0.25,
                release: 0.25,
                reorganization: 0.25,
            };
        }
        
        let n = self.score_history.len() as f64;
        let mut avg_scores = PhaseScores {
            growth: 0.0,
            conservation: 0.0,
            release: 0.0,
            reorganization: 0.0,
        };
        
        for scores in &self.score_history {
            avg_scores.growth += scores.growth;
            avg_scores.conservation += scores.conservation;
            avg_scores.release += scores.release;
            avg_scores.reorganization += scores.reorganization;
        }
        
        avg_scores.growth /= n;
        avg_scores.conservation /= n;
        avg_scores.release /= n;
        avg_scores.reorganization /= n;
        
        avg_scores
    }
}

/// Calculate regime score based on multiple indicators
pub fn calculate_regime_score(
    phase: MarketPhase,
    phase_scores: &PhaseScores,
    soc_regime: &str,
    volatility_regime: f64,
    soc_fragility: f64,
    adx: f64,
) -> f64 {
    // Phase influence
    let pan_score = match phase {
        MarketPhase::Release => 90.0,
        MarketPhase::Reorganization => 75.0,
        MarketPhase::Conservation => 50.0,
        MarketPhase::Growth => 25.0,
        MarketPhase::Unknown => 50.0,
    };
    
    // SOC regime influence
    let soc_score = match soc_regime {
        "critical" | "unstable" => 85.0,
        "release" => 90.0,
        "stable" => 20.0,
        "normal" => 40.0,
        _ => 50.0,
    };
    
    // Volatility influence (0-100)
    let vol_score = volatility_regime.clamp(0.0, 1.0) * 100.0;
    
    // Fragility influence
    let fragility_factor = (soc_fragility * (1.0 + (soc_score - 50.0) / 50.0 * 0.5)).clamp(0.0, 1.5);
    let frag_score = fragility_factor * 100.0;
    
    // ADX influence
    let adx_score = if adx < 18.0 {
        70.0  // Low ADX = more unstable/choppy
    } else if adx < 40.0 {
        30.0  // Medium ADX = smoother trend
    } else {
        60.0  // High ADX = strong but potentially volatile
    };
    
    // Combine with weights
    let weights = crate::types::RegimeScoreConfig::default();
    
    let combined = pan_score * weights.panarchy_weight +
                   soc_score * weights.soc_weight +
                   vol_score * weights.volatility_weight +
                   frag_score * weights.fragility_weight +
                   adx_score * weights.adx_weight;
    
    combined.clamp(0.0, 100.0)
}

/// Batch phase identification for multiple PCR components
pub fn identify_phases_batch(
    pcr_components: &[PCRComponents],
    params: &PanarchyParameters,
) -> Vec<MarketPhase> {
    let mut identifier = PhaseIdentifier::new(params.clone());
    pcr_components.iter()
        .map(|pcr| identifier.identify_phase(pcr))
        .collect()
}

/// Calculate phase transition probabilities
pub struct PhaseTransitionTracker {
    transition_counts: [[u32; 4]; 4], // 4x4 matrix for 4 phases
    total_transitions: u32,
}

impl PhaseTransitionTracker {
    pub fn new() -> Self {
        Self {
            transition_counts: [[0; 4]; 4],
            total_transitions: 0,
        }
    }
    
    pub fn record_transition(&mut self, from: MarketPhase, to: MarketPhase) {
        if let (Some(from_idx), Some(to_idx)) = (Self::phase_to_index(from), Self::phase_to_index(to)) {
            self.transition_counts[from_idx][to_idx] += 1;
            self.total_transitions += 1;
        }
    }
    
    pub fn get_transition_probability(&self, from: MarketPhase, to: MarketPhase) -> f64 {
        if let (Some(from_idx), Some(to_idx)) = (Self::phase_to_index(from), Self::phase_to_index(to)) {
            let from_total: u32 = self.transition_counts[from_idx].iter().sum();
            if from_total > 0 {
                self.transition_counts[from_idx][to_idx] as f64 / from_total as f64
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
    
    fn phase_to_index(phase: MarketPhase) -> Option<usize> {
        match phase {
            MarketPhase::Growth => Some(0),
            MarketPhase::Conservation => Some(1),
            MarketPhase::Release => Some(2),
            MarketPhase::Reorganization => Some(3),
            MarketPhase::Unknown => None,
        }
    }
}

impl Default for PhaseTransitionTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_phase_identification() {
        let params = PanarchyParameters::default();
        let mut identifier = PhaseIdentifier::new(params);
        
        // Test growth phase PCR
        let pcr = PCRComponents::new(0.8, 0.2, 0.8);
        let phase = identifier.identify_phase(&pcr);
        assert_eq!(phase, MarketPhase::Growth);
        
        // Test conservation phase PCR
        let pcr = PCRComponents::new(0.8, 0.8, 0.2);
        let phase = identifier.identify_phase(&pcr);
        assert_eq!(phase, MarketPhase::Conservation);
    }
    
    #[test]
    fn test_hysteresis() {
        let mut params = PanarchyParameters::default();
        params.hysteresis_min_score_diff = 0.2;
        let mut identifier = PhaseIdentifier::new(params);
        
        // Set initial phase
        let pcr = PCRComponents::new(0.8, 0.2, 0.8);
        let phase1 = identifier.identify_phase(&pcr);
        
        // Small change - should maintain phase due to hysteresis
        let pcr = PCRComponents::new(0.75, 0.25, 0.75);
        let phase2 = identifier.identify_phase(&pcr);
        
        assert_eq!(phase1, phase2);
    }
    
    #[test]
    fn test_regime_score_calculation() {
        let phase_scores = PhaseScores {
            growth: 0.1,
            conservation: 0.2,
            release: 0.6,
            reorganization: 0.1,
        };
        
        let score = calculate_regime_score(
            MarketPhase::Release,
            &phase_scores,
            "critical",
            0.8,
            0.7,
            25.0,
        );
        
        assert!(score > 50.0 && score <= 100.0);
    }
    
    #[test]
    fn test_phase_transitions() {
        let mut tracker = PhaseTransitionTracker::new();
        
        tracker.record_transition(MarketPhase::Growth, MarketPhase::Conservation);
        tracker.record_transition(MarketPhase::Growth, MarketPhase::Conservation);
        tracker.record_transition(MarketPhase::Growth, MarketPhase::Release);
        
        let prob = tracker.get_transition_probability(MarketPhase::Growth, MarketPhase::Conservation);
        assert!((prob - 2.0/3.0).abs() < 1e-10);
    }
}