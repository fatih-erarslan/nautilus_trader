//! Antifragile Coalition - Strategic alliances that grow stronger under stress

use crate::error::{QBMIAError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Antifragile coalition system for market stress adaptation
#[derive(Debug, Clone)]
pub struct AntifragileCoalition {
    pub members: HashSet<String>,
    pub stress_history: Vec<StressEvent>,
    pub adaptation_mechanisms: Vec<AdaptationMechanism>,
    pub coalition_strength: f64,
    pub antifragility_score: f64,
}

/// Coalition analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoalitionAnalysis {
    pub coalition_id: String,
    pub member_count: usize,
    pub stress_resistance: f64,
    pub adaptation_capacity: f64,
    pub antifragility_metrics: HashMap<String, f64>,
    pub recommended_actions: Vec<String>,
}

/// Market stress event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressEvent {
    pub timestamp: f64,
    pub stress_type: StressType,
    pub intensity: f64,
    pub duration: f64,
    pub affected_members: HashSet<String>,
    pub coalition_response: CoalitionResponse,
}

/// Types of market stress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StressType {
    VolatilitySpike,
    LiquidityCrisis,
    MarketManipulation,
    SystemicRisk,
    RegulationChange,
    ExternalShock,
    CompetitivePressure,
}

/// Coalition response to stress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoalitionResponse {
    pub response_type: ResponseType,
    pub resource_allocation: HashMap<String, f64>,
    pub coordination_level: f64,
    pub effectiveness: f64,
}

/// Types of coalition responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseType {
    ResourceSharing,
    InformationPooling,
    CoordinatedDefense,
    StrategicRetreat,
    CounterAttack,
    Adaptation,
    Innovation,
}

/// Adaptation mechanisms that improve under stress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationMechanism {
    pub mechanism_type: AdaptationType,
    pub trigger_threshold: f64,
    pub adaptation_rate: f64,
    pub strengthening_factor: f64,
    pub parameters: HashMap<String, f64>,
}

/// Types of adaptation mechanisms
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AdaptationType {
    StressInoculation,
    RedundancyBuilding,
    DiversificationIncrease,
    LearningAcceleration,
    NetworkReinforcement,
    FlexibilityEnhancement,
    ResilienceUpgrade,
}

impl AntifragileCoalition {
    /// Create new antifragile coalition
    pub fn new(members: HashSet<String>) -> Self {
        let initial_strength = (members.len() as f64).ln() + 1.0; // Network effect
        
        Self {
            members,
            stress_history: Vec::new(),
            adaptation_mechanisms: Self::default_adaptation_mechanisms(),
            coalition_strength: initial_strength,
            antifragility_score: 0.5, // Start neutral
        }
    }

    /// Add member to coalition
    pub fn add_member(&mut self, member: String) -> Result<()> {
        if self.members.insert(member.clone()) {
            // Network effect increases strength
            self.coalition_strength += (self.members.len() as f64).ln() * 0.1;
            Ok(())
        } else {
            Err(QBMIAError::strategy(format!("Member {} already in coalition", member)))
        }
    }

    /// Remove member from coalition
    pub fn remove_member(&mut self, member: &str) -> Result<bool> {
        if self.members.remove(member) {
            // Recalculate strength after member loss
            self.coalition_strength = (self.members.len() as f64).ln().max(1.0);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Process stress event and adapt coalition
    pub async fn process_stress_event(&mut self, stress_event: StressEvent) -> Result<CoalitionResponse> {
        // Record stress event
        self.stress_history.push(stress_event.clone());

        // Determine coalition response
        let response = self.determine_optimal_response(&stress_event).await?;

        // Apply antifragile adaptation
        self.apply_antifragile_adaptation(&stress_event, &response).await?;

        // Update coalition metrics
        self.update_coalition_metrics(&stress_event, &response).await?;

        Ok(response)
    }

    /// Determine optimal response to stress
    async fn determine_optimal_response(&self, stress_event: &StressEvent) -> Result<CoalitionResponse> {
        let response_type = match stress_event.stress_type {
            StressType::VolatilitySpike => {
                if stress_event.intensity > 0.7 {
                    ResponseType::CoordinatedDefense
                } else {
                    ResponseType::ResourceSharing
                }
            },
            StressType::LiquidityCrisis => ResponseType::ResourceSharing,
            StressType::MarketManipulation => ResponseType::CoordinatedDefense,
            StressType::SystemicRisk => ResponseType::StrategicRetreat,
            StressType::RegulationChange => ResponseType::Adaptation,
            StressType::ExternalShock => ResponseType::Innovation,
            StressType::CompetitivePressure => ResponseType::CounterAttack,
        };

        // Calculate resource allocation based on member capacity and need
        let resource_allocation = self.calculate_resource_allocation(stress_event).await?;

        // Determine coordination level based on stress intensity and coalition experience
        let coordination_level = self.calculate_coordination_level(stress_event).await?;

        // Estimate response effectiveness
        let effectiveness = self.estimate_response_effectiveness(&response_type, stress_event).await?;

        Ok(CoalitionResponse {
            response_type,
            resource_allocation,
            coordination_level,
            effectiveness,
        })
    }

    /// Calculate optimal resource allocation
    async fn calculate_resource_allocation(&self, stress_event: &StressEvent) -> Result<HashMap<String, f64>> {
        let mut allocation = HashMap::new();
        let total_capacity = self.members.len() as f64;

        for member in &self.members {
            let base_allocation = 1.0 / total_capacity;
            
            // Increase allocation for affected members
            let stress_multiplier = if stress_event.affected_members.contains(member) {
                1.0 + stress_event.intensity
            } else {
                1.0 - stress_event.intensity * 0.2 // Reduce for unaffected to help affected
            };

            allocation.insert(member.clone(), base_allocation * stress_multiplier);
        }

        // Normalize allocations
        let total: f64 = allocation.values().sum();
        if total > 0.0 {
            for value in allocation.values_mut() {
                *value /= total;
            }
        }

        Ok(allocation)
    }

    /// Calculate coordination level needed
    async fn calculate_coordination_level(&self, stress_event: &StressEvent) -> Result<f64> {
        let base_coordination = 0.5;
        let stress_factor = stress_event.intensity;
        let experience_factor = self.get_stress_experience_factor(&stress_event.stress_type);
        let size_factor = (self.members.len() as f64).ln() / 10.0; // Larger coalitions need more coordination

        let coordination = base_coordination + stress_factor * 0.3 + experience_factor * 0.2 + size_factor;
        Ok(coordination.min(1.0).max(0.0))
    }

    /// Get experience factor for stress type
    fn get_stress_experience_factor(&self, stress_type: &StressType) -> f64 {
        let similar_events = self.stress_history.iter()
            .filter(|event| std::mem::discriminant(&event.stress_type) == std::mem::discriminant(stress_type))
            .count();

        // More experience with similar stress types improves response
        (similar_events as f64 * 0.1).min(0.5)
    }

    /// Estimate response effectiveness
    async fn estimate_response_effectiveness(
        &self,
        response_type: &ResponseType,
        stress_event: &StressEvent,
    ) -> Result<f64> {
        let base_effectiveness = match response_type {
            ResponseType::ResourceSharing => 0.7,
            ResponseType::InformationPooling => 0.6,
            ResponseType::CoordinatedDefense => 0.8,
            ResponseType::StrategicRetreat => 0.5,
            ResponseType::CounterAttack => 0.7,
            ResponseType::Adaptation => 0.6,
            ResponseType::Innovation => 0.5,
        };

        // Adjust for coalition strength and stress intensity
        let strength_factor = (self.coalition_strength / 10.0).min(1.0);
        let stress_adjustment = 1.0 - (stress_event.intensity * 0.3);
        let antifragility_bonus = self.antifragility_score * 0.2;

        let effectiveness = base_effectiveness * strength_factor * stress_adjustment + antifragility_bonus;
        Ok(effectiveness.min(1.0).max(0.0))
    }

    /// Apply antifragile adaptation mechanisms
    async fn apply_antifragile_adaptation(
        &mut self,
        stress_event: &StressEvent,
        response: &CoalitionResponse,
    ) -> Result<()> {
        let stress_intensity = stress_event.intensity;

        // Process adaptation mechanisms sequentially to avoid borrowing conflicts
        let mut i = 0;
        while i < self.adaptation_mechanisms.len() {
            if stress_intensity >= self.adaptation_mechanisms[i].trigger_threshold {
                let mechanism_type = self.adaptation_mechanisms[i].mechanism_type;
                // Apply the mechanism logic directly here
                match mechanism_type {
                    AdaptationType::StressInoculation => {
                        self.adaptation_mechanisms[i].strengthening_factor *= 1.0 + stress_event.intensity * 0.1;
                    }
                    AdaptationType::RedundancyBuilding => {
                        self.adaptation_mechanisms[i].strengthening_factor *= 1.1;
                    }
                    AdaptationType::LearningAcceleration => {
                        self.adaptation_mechanisms[i].adaptation_rate *= 1.0 + stress_event.intensity * 0.05;
                    }
                    AdaptationType::FlexibilityEnhancement => {
                        self.adaptation_mechanisms[i].adaptation_rate *= 1.2;
                    }
                    AdaptationType::DiversificationIncrease => {
                        self.adaptation_mechanisms[i].trigger_threshold *= 0.95; // Lower threshold for faster response
                    }
                    AdaptationType::NetworkReinforcement => {
                        self.adaptation_mechanisms[i].strengthening_factor *= 1.05;
                    }
                    AdaptationType::ResilienceUpgrade => {
                        self.adaptation_mechanisms[i].strengthening_factor *= 1.15;
                    }
                }
            }
            i += 1;
        }

        // Update antifragility score based on successful adaptation
        if response.effectiveness > 0.6 {
            let improvement = response.effectiveness * stress_intensity * 0.1;
            self.antifragility_score = (self.antifragility_score + improvement).min(1.0);
        }

        Ok(())
    }

    /// Apply specific adaptation mechanism
    async fn apply_adaptation_mechanism(
        &mut self,
        mechanism: &mut AdaptationMechanism,
        stress_event: &StressEvent,
        response: &CoalitionResponse,
    ) -> Result<()> {
        match mechanism.mechanism_type {
            AdaptationType::StressInoculation => {
                // Build immunity to similar stresses
                mechanism.strengthening_factor *= 1.0 + stress_event.intensity * 0.1;
                self.coalition_strength += stress_event.intensity * 0.05;
            },
            AdaptationType::RedundancyBuilding => {
                // Increase backup systems and resources
                if response.effectiveness > 0.5 {
                    mechanism.strengthening_factor *= 1.1;
                    self.coalition_strength += 0.1;
                }
            },
            AdaptationType::DiversificationIncrease => {
                // Encourage member diversification
                mechanism.adaptation_rate *= 1.0 + stress_event.intensity * 0.05;
            },
            AdaptationType::LearningAcceleration => {
                // Speed up learning from stress events
                mechanism.adaptation_rate *= 1.2;
            },
            AdaptationType::NetworkReinforcement => {
                // Strengthen member connections
                if self.members.len() > 1 {
                    let network_bonus = (self.members.len() as f64).ln() * 0.02;
                    self.coalition_strength += network_bonus;
                }
            },
            AdaptationType::FlexibilityEnhancement => {
                // Improve adaptability to different stress types
                mechanism.trigger_threshold *= 0.95; // Lower threshold for faster response
            },
            AdaptationType::ResilienceUpgrade => {
                // Overall resilience improvement
                self.coalition_strength += stress_event.intensity * mechanism.strengthening_factor * 0.03;
            },
        }

        Ok(())
    }

    /// Update coalition metrics after stress response
    async fn update_coalition_metrics(
        &mut self,
        stress_event: &StressEvent,
        response: &CoalitionResponse,
    ) -> Result<()> {
        // Update coalition strength based on response effectiveness
        let strength_change = (response.effectiveness - 0.5) * stress_event.intensity * 0.1;
        self.coalition_strength += strength_change;
        self.coalition_strength = self.coalition_strength.max(0.1); // Minimum strength

        // Prune old stress history to maintain relevance
        const MAX_HISTORY: usize = 100;
        if self.stress_history.len() > MAX_HISTORY {
            self.stress_history.drain(0..self.stress_history.len() - MAX_HISTORY);
        }

        Ok(())
    }

    /// Analyze coalition performance and antifragility
    pub async fn analyze_coalition(&self) -> Result<CoalitionAnalysis> {
        let stress_resistance = self.calculate_stress_resistance().await?;
        let adaptation_capacity = self.calculate_adaptation_capacity().await?;
        let antifragility_metrics = self.calculate_antifragility_metrics().await?;
        let recommended_actions = self.generate_recommendations().await?;

        Ok(CoalitionAnalysis {
            coalition_id: format!("coalition_{}", self.members.len()),
            member_count: self.members.len(),
            stress_resistance,
            adaptation_capacity,
            antifragility_metrics,
            recommended_actions,
        })
    }

    /// Calculate stress resistance
    async fn calculate_stress_resistance(&self) -> Result<f64> {
        if self.stress_history.is_empty() {
            return Ok(0.5); // Neutral if no history
        }

        let recent_events = self.stress_history.iter().rev().take(10);
        let avg_effectiveness = recent_events
            .map(|event| event.coalition_response.effectiveness)
            .sum::<f64>() / 10.0_f64.min(self.stress_history.len() as f64);

        // Combine with coalition strength
        let resistance = (avg_effectiveness + self.coalition_strength / 10.0) / 2.0;
        Ok(resistance.min(1.0))
    }

    /// Calculate adaptation capacity
    async fn calculate_adaptation_capacity(&self) -> Result<f64> {
        let mechanism_strength: f64 = self.adaptation_mechanisms
            .iter()
            .map(|m| m.strengthening_factor * m.adaptation_rate)
            .sum::<f64>() / self.adaptation_mechanisms.len() as f64;

        let diversity_bonus = (self.members.len() as f64).ln() / 5.0;
        let experience_bonus = (self.stress_history.len() as f64).ln() / 10.0;

        let capacity = (mechanism_strength + diversity_bonus + experience_bonus) / 3.0;
        Ok(capacity.min(1.0))
    }

    /// Calculate detailed antifragility metrics
    async fn calculate_antifragility_metrics(&self) -> Result<HashMap<String, f64>> {
        let mut metrics = HashMap::new();

        metrics.insert("overall_antifragility".to_string(), self.antifragility_score);
        metrics.insert("coalition_strength".to_string(), self.coalition_strength / 10.0);
        metrics.insert("stress_experience".to_string(), (self.stress_history.len() as f64).ln() / 5.0);
        metrics.insert("member_diversity".to_string(), (self.members.len() as f64).ln() / 5.0);
        
        // Calculate improvement trend
        if self.stress_history.len() >= 5 {
            let recent_avg = self.stress_history.iter().rev().take(5)
                .map(|e| e.coalition_response.effectiveness)
                .sum::<f64>() / 5.0;
            let historical_avg = self.stress_history.iter()
                .map(|e| e.coalition_response.effectiveness)
                .sum::<f64>() / self.stress_history.len() as f64;
            
            metrics.insert("improvement_trend".to_string(), recent_avg - historical_avg);
        }

        Ok(metrics)
    }

    /// Generate recommendations for coalition improvement
    async fn generate_recommendations(&self) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();

        if self.members.len() < 3 {
            recommendations.push("Consider recruiting more members to enhance network effects".to_string());
        }

        if self.antifragility_score < 0.3 {
            recommendations.push("Focus on stress inoculation exercises to build antifragility".to_string());
        }

        if self.coalition_strength < 2.0 {
            recommendations.push("Strengthen member coordination and resource sharing".to_string());
        }

        if self.stress_history.len() < 5 {
            recommendations.push("Gain more experience with controlled stress testing".to_string());
        }

        let mechanism_count = self.adaptation_mechanisms.len();
        if mechanism_count < 5 {
            recommendations.push("Implement additional adaptation mechanisms".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("Coalition shows strong antifragile characteristics - maintain current strategies".to_string());
        }

        Ok(recommendations)
    }

    /// Create default adaptation mechanisms
    fn default_adaptation_mechanisms() -> Vec<AdaptationMechanism> {
        vec![
            AdaptationMechanism {
                mechanism_type: AdaptationType::StressInoculation,
                trigger_threshold: 0.3,
                adaptation_rate: 1.0,
                strengthening_factor: 1.0,
                parameters: HashMap::new(),
            },
            AdaptationMechanism {
                mechanism_type: AdaptationType::RedundancyBuilding,
                trigger_threshold: 0.5,
                adaptation_rate: 0.8,
                strengthening_factor: 1.0,
                parameters: HashMap::new(),
            },
            AdaptationMechanism {
                mechanism_type: AdaptationType::NetworkReinforcement,
                trigger_threshold: 0.4,
                adaptation_rate: 1.1,
                strengthening_factor: 1.0,
                parameters: HashMap::new(),
            },
        ]
    }
}

impl Default for AntifragileCoalition {
    fn default() -> Self {
        Self::new(HashSet::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_coalition_creation() {
        let members = vec!["alice".to_string(), "bob".to_string(), "charlie".to_string()].into_iter().collect();
        let coalition = AntifragileCoalition::new(members);
        assert_eq!(coalition.members.len(), 3);
        assert!(coalition.coalition_strength > 1.0);
    }

    #[tokio::test]
    async fn test_stress_event_processing() {
        let members = vec!["alice".to_string(), "bob".to_string()].into_iter().collect();
        let mut coalition = AntifragileCoalition::new(members);

        let stress_event = StressEvent {
            timestamp: 1000.0,
            stress_type: StressType::VolatilitySpike,
            intensity: 0.8,
            duration: 60.0,
            affected_members: vec!["alice".to_string()].into_iter().collect(),
            coalition_response: CoalitionResponse {
                response_type: ResponseType::ResourceSharing,
                resource_allocation: HashMap::new(),
                coordination_level: 0.0,
                effectiveness: 0.0,
            },
        };

        let response = coalition.process_stress_event(stress_event).await.unwrap();
        assert!(response.effectiveness > 0.0);
        assert!(coalition.stress_history.len() == 1);
    }

    #[tokio::test]
    async fn test_antifragile_adaptation() {
        let members = vec!["alice".to_string(), "bob".to_string()].into_iter().collect();
        let mut coalition = AntifragileCoalition::new(members);
        let initial_strength = coalition.coalition_strength;

        // Create high-intensity stress event
        let stress_event = StressEvent {
            timestamp: 1000.0,
            stress_type: StressType::VolatilitySpike,
            intensity: 0.9,
            duration: 120.0,
            affected_members: coalition.members.clone(),
            coalition_response: CoalitionResponse {
                response_type: ResponseType::CoordinatedDefense,
                resource_allocation: HashMap::new(),
                coordination_level: 0.8,
                effectiveness: 0.8,
            },
        };

        coalition.process_stress_event(stress_event).await.unwrap();

        // Coalition should be stronger after successful stress response
        assert!(coalition.coalition_strength >= initial_strength);
        assert!(coalition.antifragility_score > 0.5);
    }

    #[tokio::test]
    async fn test_coalition_analysis() {
        let members = vec!["alice".to_string(), "bob".to_string(), "charlie".to_string()].into_iter().collect();
        let coalition = AntifragileCoalition::new(members);

        let analysis = coalition.analyze_coalition().await.unwrap();
        assert_eq!(analysis.member_count, 3);
        assert!(analysis.stress_resistance >= 0.0 && analysis.stress_resistance <= 1.0);
        assert!(analysis.adaptation_capacity >= 0.0 && analysis.adaptation_capacity <= 1.0);
        assert!(!analysis.recommended_actions.is_empty());
    }
}