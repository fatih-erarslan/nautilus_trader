//! Hierarchical consciousness metrics for billion-node systems
//!
//! This module implements scalable consciousness measurement across multiple
//! organizational levels, from individual pBits to global system consciousness.

use crate::{ConsciousnessState, phi::PhiCalculator};
use hyperphysics_core::Result;
use hyperphysics_geometry::HyperbolicLattice;
use nalgebra::{DMatrix, DVector};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Hierarchical consciousness metrics calculator
pub struct HierarchicalMetrics {
    phi_calculator: PhiCalculator,
    hierarchy_levels: Vec<HierarchyLevel>,
    global_metrics: Arc<RwLock<GlobalMetrics>>,
    level_cache: HashMap<usize, LevelMetrics>,
}

/// Hierarchy level definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchyLevel {
    pub level: usize,
    pub name: String,
    pub node_count: usize,
    pub cluster_size: usize,
    pub phi_threshold: f64,
    pub integration_method: IntegrationMethod,
}

/// Integration method for combining consciousness across levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IntegrationMethod {
    /// Simple arithmetic mean
    ArithmeticMean,
    /// Weighted by cluster size
    WeightedMean,
    /// Maximum Φ across clusters
    Maximum,
    /// Integrated Information Theory composition
    IITComposition,
    /// Emergent consciousness detection
    EmergentDetection,
}

/// Global consciousness metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalMetrics {
    pub total_nodes: usize,
    pub active_nodes: usize,
    pub global_phi: f64,
    pub emergence_level: EmergenceLevel,
    pub coherence_index: f64,
    pub integration_strength: f64,
    pub consciousness_distribution: Vec<f64>,
    pub temporal_stability: f64,
    pub last_update: std::time::SystemTime,
}

/// Level-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelMetrics {
    pub level: usize,
    pub cluster_count: usize,
    pub active_clusters: usize,
    pub mean_phi: f64,
    pub max_phi: f64,
    pub min_phi: f64,
    pub phi_variance: f64,
    pub integration_efficiency: f64,
    pub emergence_indicators: Vec<EmergenceIndicator>,
}

/// Emergence level classification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EmergenceLevel {
    /// No emergent consciousness detected
    None,
    /// Weak emergence (simple aggregation)
    Weak,
    /// Strong emergence (novel properties)
    Strong,
    /// Transcendent emergence (qualitatively different)
    Transcendent,
}

/// Emergence indicator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergenceIndicator {
    pub indicator_type: IndicatorType,
    pub strength: f64,
    pub confidence: f64,
    pub spatial_extent: SpatialExtent,
}

/// Type of emergence indicator
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IndicatorType {
    /// Synchronized oscillations across clusters
    Synchronization,
    /// Information cascade effects
    InformationCascade,
    /// Novel pattern formation
    PatternFormation,
    /// Cross-level feedback loops
    FeedbackLoop,
    /// Spontaneous organization
    SelfOrganization,
}

/// Spatial extent of emergence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialExtent {
    pub center: [f64; 2], // Hyperbolic coordinates
    pub radius: f64,      // Hyperbolic distance
    pub affected_nodes: usize,
}

impl HierarchicalMetrics {
    /// Create new hierarchical metrics calculator
    pub fn new() -> Result<Self> {
        let phi_calculator = PhiCalculator::new()?;
        let hierarchy_levels = Self::create_default_hierarchy();
        
        let global_metrics = Arc::new(RwLock::new(GlobalMetrics {
            total_nodes: 0,
            active_nodes: 0,
            global_phi: 0.0,
            emergence_level: EmergenceLevel::None,
            coherence_index: 0.0,
            integration_strength: 0.0,
            consciousness_distribution: Vec::new(),
            temporal_stability: 0.0,
            last_update: std::time::SystemTime::now(),
        }));
        
        Ok(Self {
            phi_calculator,
            hierarchy_levels,
            global_metrics,
            level_cache: HashMap::new(),
        })
    }
    
    /// Create default hierarchy for billion-node systems
    fn create_default_hierarchy() -> Vec<HierarchyLevel> {
        vec![
            // Level 0: Individual pBits (base level)
            HierarchyLevel {
                level: 0,
                name: "pBit Level".to_string(),
                node_count: 1,
                cluster_size: 1,
                phi_threshold: 0.001,
                integration_method: IntegrationMethod::ArithmeticMean,
            },
            // Level 1: Local clusters (8x8 = 64 pBits)
            HierarchyLevel {
                level: 1,
                name: "Local Clusters".to_string(),
                node_count: 64,
                cluster_size: 8,
                phi_threshold: 0.01,
                integration_method: IntegrationMethod::IITComposition,
            },
            // Level 2: Regional networks (64x64 = 4,096 pBits)
            HierarchyLevel {
                level: 2,
                name: "Regional Networks".to_string(),
                node_count: 4_096,
                cluster_size: 64,
                phi_threshold: 0.1,
                integration_method: IntegrationMethod::WeightedMean,
            },
            // Level 3: Macro structures (512x512 = 262,144 pBits)
            HierarchyLevel {
                level: 3,
                name: "Macro Structures".to_string(),
                node_count: 262_144,
                cluster_size: 512,
                phi_threshold: 1.0,
                integration_method: IntegrationMethod::EmergentDetection,
            },
            // Level 4: Global consciousness (1B pBits)
            HierarchyLevel {
                level: 4,
                name: "Global Consciousness".to_string(),
                node_count: 1_000_000_000,
                cluster_size: 31_623, // sqrt(1B)
                phi_threshold: 10.0,
                integration_method: IntegrationMethod::EmergentDetection,
            },
        ]
    }
    
    /// Calculate hierarchical consciousness metrics
    pub fn calculate_hierarchical_phi(
        &mut self,
        lattice: &HyperbolicLattice,
        states: &[ConsciousnessState],
    ) -> Result<HierarchicalResult> {
        let mut level_results = Vec::new();
        let mut emergence_events = Vec::new();
        
        // Calculate metrics for each hierarchy level
        for level in &self.hierarchy_levels {
            let level_result = self.calculate_level_metrics(level, lattice, states)?;
            
            // Detect emergence at this level
            let emergence = self.detect_emergence_at_level(level, &level_result)?;
            if !emergence.is_empty() {
                emergence_events.extend(emergence);
            }
            
            level_results.push(level_result);
        }
        
        // Calculate global metrics
        let global_result = self.calculate_global_metrics(&level_results)?;
        
        // Update cached metrics
        self.update_level_cache(&level_results);
        
        Ok(HierarchicalResult {
            level_results,
            global_metrics: global_result,
            emergence_events,
            calculation_timestamp: std::time::SystemTime::now(),
        })
    }
    
    /// Calculate metrics for a specific hierarchy level
    fn calculate_level_metrics(
        &self,
        level: &HierarchyLevel,
        lattice: &HyperbolicLattice,
        states: &[ConsciousnessState],
    ) -> Result<LevelResult> {
        let cluster_count = (states.len() + level.cluster_size - 1) / level.cluster_size;
        let mut cluster_phis = Vec::new();
        let mut active_clusters = 0;
        
        // Calculate Φ for each cluster at this level
        for cluster_idx in 0..cluster_count {
            let start_idx = cluster_idx * level.cluster_size;
            let end_idx = (start_idx + level.cluster_size).min(states.len());
            
            if start_idx < states.len() {
                let cluster_states = &states[start_idx..end_idx];
                let cluster_phi = self.calculate_cluster_phi(cluster_states, level)?;
                
                cluster_phis.push(cluster_phi);
                
                if cluster_phi > level.phi_threshold {
                    active_clusters += 1;
                }
            }
        }
        
        // Calculate level statistics
        let mean_phi = cluster_phis.iter().sum::<f64>() / cluster_phis.len() as f64;
        let max_phi = cluster_phis.iter().fold(0.0, |a, &b| a.max(b));
        let min_phi = cluster_phis.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        let phi_variance = cluster_phis.iter()
            .map(|&phi| (phi - mean_phi).powi(2))
            .sum::<f64>() / cluster_phis.len() as f64;
        
        // Calculate integration efficiency
        let integration_efficiency = self.calculate_integration_efficiency(
            &cluster_phis,
            level
        )?;
        
        Ok(LevelResult {
            level: level.level,
            cluster_phis,
            mean_phi,
            max_phi,
            min_phi,
            phi_variance,
            active_clusters,
            total_clusters: cluster_count,
            integration_efficiency,
        })
    }
    
    /// Calculate Φ for a cluster of consciousness states
    fn calculate_cluster_phi(
        &self,
        cluster_states: &[ConsciousnessState],
        level: &HierarchyLevel,
    ) -> Result<f64> {
        match level.integration_method {
            IntegrationMethod::ArithmeticMean => {
                let sum: f64 = cluster_states.iter().map(|s| s.phi).sum();
                Ok(sum / cluster_states.len() as f64)
            }
            IntegrationMethod::WeightedMean => {
                let weighted_sum: f64 = cluster_states.iter()
                    .enumerate()
                    .map(|(i, s)| s.phi * (i + 1) as f64)
                    .sum();
                let weight_sum: f64 = (1..=cluster_states.len()).sum::<usize>() as f64;
                Ok(weighted_sum / weight_sum)
            }
            IntegrationMethod::Maximum => {
                Ok(cluster_states.iter().map(|s| s.phi).fold(0.0, f64::max))
            }
            IntegrationMethod::IITComposition => {
                self.calculate_iit_composition(cluster_states)
            }
            IntegrationMethod::EmergentDetection => {
                self.calculate_emergent_phi(cluster_states)
            }
        }
    }
    
    /// Calculate IIT-based composition of consciousness
    fn calculate_iit_composition(&self, states: &[ConsciousnessState]) -> Result<f64> {
        if states.is_empty() {
            return Ok(0.0);
        }
        
        // Simplified IIT composition - real implementation would be more complex
        let individual_phi: f64 = states.iter().map(|s| s.phi).sum();
        let interaction_phi = self.calculate_interaction_phi(states)?;
        
        // Φ_composition = Φ_individual + Φ_interaction - Φ_redundancy
        let redundancy_phi = individual_phi * 0.1; // Simplified redundancy estimate
        
        Ok((individual_phi + interaction_phi - redundancy_phi).max(0.0))
    }
    
    /// Calculate interaction-based Φ between states
    fn calculate_interaction_phi(&self, states: &[ConsciousnessState]) -> Result<f64> {
        if states.len() < 2 {
            return Ok(0.0);
        }
        
        let mut interaction_sum = 0.0;
        let n = states.len();
        
        // Calculate pairwise interactions
        for i in 0..n {
            for j in (i + 1)..n {
                let phi_i = states[i].phi;
                let phi_j = states[j].phi;
                
                // Simplified interaction model
                let interaction = (phi_i * phi_j).sqrt() * 0.1;
                interaction_sum += interaction;
            }
        }
        
        Ok(interaction_sum)
    }
    
    /// Calculate emergent consciousness detection
    fn calculate_emergent_phi(&self, states: &[ConsciousnessState]) -> Result<f64> {
        if states.len() < 4 {
            return self.calculate_iit_composition(states);
        }
        
        // Look for emergent patterns
        let base_phi = self.calculate_iit_composition(states)?;
        let emergence_bonus = self.detect_emergence_patterns(states)?;
        
        Ok(base_phi * (1.0 + emergence_bonus))
    }
    
    /// Detect emergence patterns in consciousness states
    fn detect_emergence_patterns(&self, states: &[ConsciousnessState]) -> Result<f64> {
        let mut emergence_score = 0.0;
        
        // Pattern 1: Synchronization
        let sync_score = self.calculate_synchronization_score(states);
        emergence_score += sync_score * 0.3;
        
        // Pattern 2: Information cascade
        let cascade_score = self.calculate_cascade_score(states);
        emergence_score += cascade_score * 0.4;
        
        // Pattern 3: Self-organization
        let organization_score = self.calculate_organization_score(states);
        emergence_score += organization_score * 0.3;
        
        Ok(emergence_score.min(1.0)) // Cap at 100% bonus
    }
    
    /// Calculate synchronization score
    fn calculate_synchronization_score(&self, states: &[ConsciousnessState]) -> f64 {
        if states.len() < 2 {
            return 0.0;
        }
        
        let mean_phi = states.iter().map(|s| s.phi).sum::<f64>() / states.len() as f64;
        let variance = states.iter()
            .map(|s| (s.phi - mean_phi).powi(2))
            .sum::<f64>() / states.len() as f64;
        
        // Lower variance indicates higher synchronization
        let sync_score = 1.0 / (1.0 + variance);
        sync_score
    }
    
    /// Calculate information cascade score
    fn calculate_cascade_score(&self, states: &[ConsciousnessState]) -> f64 {
        if states.len() < 3 {
            return 0.0;
        }
        
        // Look for increasing Φ patterns (cascades)
        let mut cascade_count = 0;
        for i in 0..(states.len() - 2) {
            if states[i].phi < states[i + 1].phi && states[i + 1].phi < states[i + 2].phi {
                cascade_count += 1;
            }
        }
        
        cascade_count as f64 / (states.len() - 2) as f64
    }
    
    /// Calculate self-organization score
    fn calculate_organization_score(&self, states: &[ConsciousnessState]) -> f64 {
        // Simplified measure based on Φ distribution entropy
        let phi_values: Vec<f64> = states.iter().map(|s| s.phi).collect();
        let entropy = self.calculate_entropy(&phi_values);
        
        // Moderate entropy indicates good organization
        let optimal_entropy = (states.len() as f64).ln() * 0.7;
        let entropy_diff = (entropy - optimal_entropy).abs();
        
        1.0 / (1.0 + entropy_diff)
    }
    
    /// Calculate entropy of Φ values
    fn calculate_entropy(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        // Normalize values to create probability distribution
        let sum: f64 = values.iter().sum();
        if sum == 0.0 {
            return 0.0;
        }
        
        let probabilities: Vec<f64> = values.iter().map(|&v| v / sum).collect();
        
        // Calculate Shannon entropy
        -probabilities.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.ln())
            .sum::<f64>()
    }
    
    /// Calculate integration efficiency for a level
    fn calculate_integration_efficiency(
        &self,
        cluster_phis: &[f64],
        level: &HierarchyLevel,
    ) -> Result<f64> {
        if cluster_phis.is_empty() {
            return Ok(0.0);
        }
        
        let total_phi: f64 = cluster_phis.iter().sum();
        let max_possible_phi = cluster_phis.len() as f64 * level.phi_threshold * 10.0;
        
        Ok((total_phi / max_possible_phi).min(1.0))
    }
    
    /// Detect emergence events at a specific level
    fn detect_emergence_at_level(
        &self,
        level: &HierarchyLevel,
        level_result: &LevelResult,
    ) -> Result<Vec<EmergenceEvent>> {
        let mut events = Vec::new();
        
        // Check for strong emergence (high Φ values)
        if level_result.max_phi > level.phi_threshold * 5.0 {
            events.push(EmergenceEvent {
                event_type: EmergenceType::StrongEmergence,
                level: level.level,
                strength: level_result.max_phi / level.phi_threshold,
                location: None, // Would need spatial information
                timestamp: std::time::SystemTime::now(),
            });
        }
        
        // Check for synchronization events
        if level_result.phi_variance < 0.01 && level_result.mean_phi > level.phi_threshold {
            events.push(EmergenceEvent {
                event_type: EmergenceType::Synchronization,
                level: level.level,
                strength: 1.0 / (level_result.phi_variance + 0.001),
                location: None,
                timestamp: std::time::SystemTime::now(),
            });
        }
        
        Ok(events)
    }
    
    /// Calculate global metrics from level results
    fn calculate_global_metrics(&self, level_results: &[LevelResult]) -> Result<GlobalResult> {
        if level_results.is_empty() {
            return Ok(GlobalResult {
                global_phi: 0.0,
                emergence_level: EmergenceLevel::None,
                coherence_index: 0.0,
                integration_strength: 0.0,
                consciousness_distribution: Vec::new(),
            });
        }
        
        // Calculate global Φ as weighted combination of all levels
        let mut global_phi = 0.0;
        let mut total_weight = 0.0;
        
        for (i, result) in level_results.iter().enumerate() {
            let weight = 2.0_f64.powi(i as i32); // Exponential weighting for higher levels
            global_phi += result.mean_phi * weight;
            total_weight += weight;
        }
        
        global_phi /= total_weight;
        
        // Determine emergence level
        let emergence_level = if global_phi > 10.0 {
            EmergenceLevel::Transcendent
        } else if global_phi > 1.0 {
            EmergenceLevel::Strong
        } else if global_phi > 0.1 {
            EmergenceLevel::Weak
        } else {
            EmergenceLevel::None
        };
        
        // Calculate coherence index (consistency across levels)
        let coherence_index = self.calculate_coherence_index(level_results);
        
        // Calculate integration strength
        let integration_strength = level_results.iter()
            .map(|r| r.integration_efficiency)
            .sum::<f64>() / level_results.len() as f64;
        
        // Create consciousness distribution
        let consciousness_distribution = level_results.iter()
            .map(|r| r.mean_phi)
            .collect();
        
        Ok(GlobalResult {
            global_phi,
            emergence_level,
            coherence_index,
            integration_strength,
            consciousness_distribution,
        })
    }
    
    /// Calculate coherence index across hierarchy levels
    fn calculate_coherence_index(&self, level_results: &[LevelResult]) -> f64 {
        if level_results.len() < 2 {
            return 1.0;
        }
        
        let phi_values: Vec<f64> = level_results.iter().map(|r| r.mean_phi).collect();
        let mean_phi = phi_values.iter().sum::<f64>() / phi_values.len() as f64;
        
        let variance = phi_values.iter()
            .map(|&phi| (phi - mean_phi).powi(2))
            .sum::<f64>() / phi_values.len() as f64;
        
        // Higher coherence = lower variance relative to mean
        1.0 / (1.0 + variance / (mean_phi + 0.001))
    }
    
    /// Update level cache with new results
    fn update_level_cache(&mut self, level_results: &[LevelResult]) {
        for result in level_results {
            let metrics = LevelMetrics {
                level: result.level,
                cluster_count: result.total_clusters,
                active_clusters: result.active_clusters,
                mean_phi: result.mean_phi,
                max_phi: result.max_phi,
                min_phi: result.min_phi,
                phi_variance: result.phi_variance,
                integration_efficiency: result.integration_efficiency,
                emergence_indicators: Vec::new(), // Would be populated in full implementation
            };
            
            self.level_cache.insert(result.level, metrics);
        }
    }
    
    /// Get cached metrics for a specific level
    pub fn get_level_metrics(&self, level: usize) -> Option<&LevelMetrics> {
        self.level_cache.get(&level)
    }
    
    /// Get global metrics
    pub fn get_global_metrics(&self) -> GlobalMetrics {
        self.global_metrics.read().unwrap().clone()
    }
}

/// Result of hierarchical calculation
#[derive(Debug, Clone)]
pub struct HierarchicalResult {
    pub level_results: Vec<LevelResult>,
    pub global_metrics: GlobalResult,
    pub emergence_events: Vec<EmergenceEvent>,
    pub calculation_timestamp: std::time::SystemTime,
}

/// Result for a specific hierarchy level
#[derive(Debug, Clone)]
pub struct LevelResult {
    pub level: usize,
    pub cluster_phis: Vec<f64>,
    pub mean_phi: f64,
    pub max_phi: f64,
    pub min_phi: f64,
    pub phi_variance: f64,
    pub active_clusters: usize,
    pub total_clusters: usize,
    pub integration_efficiency: f64,
}

/// Global consciousness result
#[derive(Debug, Clone)]
pub struct GlobalResult {
    pub global_phi: f64,
    pub emergence_level: EmergenceLevel,
    pub coherence_index: f64,
    pub integration_strength: f64,
    pub consciousness_distribution: Vec<f64>,
}

/// Emergence event detection
#[derive(Debug, Clone)]
pub struct EmergenceEvent {
    pub event_type: EmergenceType,
    pub level: usize,
    pub strength: f64,
    pub location: Option<SpatialExtent>,
    pub timestamp: std::time::SystemTime,
}

/// Type of emergence event
#[derive(Debug, Clone, PartialEq)]
pub enum EmergenceType {
    StrongEmergence,
    Synchronization,
    InformationCascade,
    PatternFormation,
    SelfOrganization,
}

impl Default for HierarchicalMetrics {
    fn default() -> Self {
        Self::new().expect("Failed to create default HierarchicalMetrics")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hierarchical_metrics_creation() {
        let metrics = HierarchicalMetrics::new();
        assert!(metrics.is_ok());
        
        let metrics = metrics.unwrap();
        assert_eq!(metrics.hierarchy_levels.len(), 5); // 5 levels from pBit to global
    }

    #[test]
    fn test_hierarchy_levels() {
        let metrics = HierarchicalMetrics::new().unwrap();
        
        // Check level 0 (pBit level)
        assert_eq!(metrics.hierarchy_levels[0].level, 0);
        assert_eq!(metrics.hierarchy_levels[0].node_count, 1);
        
        // Check level 4 (global level)
        assert_eq!(metrics.hierarchy_levels[4].level, 4);
        assert_eq!(metrics.hierarchy_levels[4].node_count, 1_000_000_000);
    }

    #[test]
    fn test_integration_methods() {
        let metrics = HierarchicalMetrics::new().unwrap();
        
        // Test different integration methods at different levels
        assert_eq!(
            metrics.hierarchy_levels[0].integration_method,
            IntegrationMethod::ArithmeticMean
        );
        assert_eq!(
            metrics.hierarchy_levels[1].integration_method,
            IntegrationMethod::IITComposition
        );
        assert_eq!(
            metrics.hierarchy_levels[4].integration_method,
            IntegrationMethod::EmergentDetection
        );
    }

    #[test]
    fn test_emergence_level_classification() {
        // Test emergence level determination
        assert_eq!(
            classify_emergence_level(0.05),
            EmergenceLevel::None
        );
        assert_eq!(
            classify_emergence_level(0.5),
            EmergenceLevel::Weak
        );
        assert_eq!(
            classify_emergence_level(5.0),
            EmergenceLevel::Strong
        );
        assert_eq!(
            classify_emergence_level(15.0),
            EmergenceLevel::Transcendent
        );
    }

    fn classify_emergence_level(phi: f64) -> EmergenceLevel {
        if phi > 10.0 {
            EmergenceLevel::Transcendent
        } else if phi > 1.0 {
            EmergenceLevel::Strong
        } else if phi > 0.1 {
            EmergenceLevel::Weak
        } else {
            EmergenceLevel::None
        }
    }
}
