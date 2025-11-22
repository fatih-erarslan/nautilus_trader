//! Octopus Camouflage Handler for adaptive pair selection
//! Implements dynamic camouflage to avoid detection by market participants

use crate::mcp::{ParasiticPairlistManager, tools::ToolHandler};
use crate::{Result, Error};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;
use chrono::Utc;

/// Handler for activating octopus camouflage patterns
pub struct CamouflageHandler {
    manager: Arc<ParasiticPairlistManager>,
}

impl CamouflageHandler {
    pub fn new(manager: Arc<ParasiticPairlistManager>) -> Self {
        Self { manager }
    }
    
    /// Assess current threat level in the market
    async fn assess_threat_level(&self, threat_level: &str) -> Result<f64> {
        let numerical_threat = match threat_level.to_lowercase().as_str() {
            "low" => 0.2,
            "medium" => 0.5,
            "high" => 0.8,
            "critical" => 1.0,
            _ => 0.5, // Default to medium
        };
        
        // Integrate with existing Octopus organism if available
        if let Ok(Some(octopus_metrics)) = self.manager.get_organism("octopus").await {
            // Factor in organism performance
            let organism_factor = octopus_metrics.accuracy_rate;
            Ok(numerical_threat * (1.0 + organism_factor * 0.3))
        } else {
            Ok(numerical_threat)
        }
    }
    
    /// Generate appropriate camouflage pattern based on threat and environment
    async fn generate_camouflage_pattern(&self, threat_level: f64, pattern_type: &str) -> Result<Value> {
        let base_effectiveness = match pattern_type.to_lowercase().as_str() {
            "mimetic" => 0.9,      // Mimics successful patterns
            "disruptive" => 0.8,   // Breaks up recognizable shapes
            "transparent" => 0.7,  // Reduces visibility
            "adaptive" => 0.95,    // Adapts to environment
            _ => 0.6,
        };
        
        // Adjust effectiveness based on threat level
        let adjusted_effectiveness = base_effectiveness * (2.0 - threat_level).max(0.1);
        
        let camouflage_pattern = json!({
            "pattern_type": pattern_type,
            "effectiveness": adjusted_effectiveness,
            "adaptation_speed": (1.0 - threat_level) * 0.8 + 0.2, // Higher threat = faster adaptation
            "chromatophore_activity": threat_level * 0.9 + 0.1,
            "texture_mimicry": adjusted_effectiveness * 0.8,
            "behavior_modification": threat_level * 0.7,
            "detection_avoidance": adjusted_effectiveness
        });
        
        Ok(camouflage_pattern)
    }
    
    /// Calculate adaptation strategy based on current market conditions
    async fn calculate_adaptation_strategy(&self, threat_level: f64, camouflage_effectiveness: f64) -> Result<Value> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(45);
        
        // Determine strategy based on threat level and camouflage effectiveness
        let strategy = if threat_level > 0.8 {
            "stealth_mode"
        } else if camouflage_effectiveness > 0.8 {
            "aggressive_mimicry" 
        } else {
            "balanced_adaptation"
        };
        
        let pair_rotation_frequency = match threat_level {
            t if t > 0.8 => 30,  // 30 seconds in high threat
            t if t > 0.5 => 120, // 2 minutes in medium threat
            _ => 300,            // 5 minutes in low threat
        };
        
        Ok(json!({
            "strategy_type": strategy,
            "pair_rotation_frequency_seconds": pair_rotation_frequency,
            "camouflage_intensity": threat_level * 0.8 + 0.2,
            "stealth_factor": (1.0 - threat_level) * camouflage_effectiveness,
            "mimicry_targets": self.select_mimicry_targets().await?,
            "behavior_randomization": threat_level * 0.6,
            "detection_probability": (1.0 - camouflage_effectiveness) * threat_level
        }))
    }
    
    /// Select appropriate mimicry targets for camouflage
    async fn select_mimicry_targets(&self) -> Result<Vec<String>> {
        let tracked_pairs = self.manager.get_tracked_pairs().await?;
        
        // Select diverse pairs for mimicry to avoid patterns
        let mut targets = Vec::new();
        let step = if tracked_pairs.len() > 5 { tracked_pairs.len() / 5 } else { 1 };
        
        for (i, pair) in tracked_pairs.iter().enumerate() {
            if i % step == 0 && targets.len() < 5 {
                targets.push(pair.clone());
            }
        }
        
        if targets.is_empty() && !tracked_pairs.is_empty() {
            targets.push(tracked_pairs[0].clone());
        }
        
        Ok(targets)
    }
}

#[async_trait]
impl ToolHandler for CamouflageHandler {
    async fn handle(&self, input: Value) -> Result<Value> {
        let start_time = std::time::Instant::now();
        
        let threat_level_str = input.get("threat_level")
            .and_then(|v| v.as_str())
            .unwrap_or("medium");
        
        let camouflage_pattern = input.get("camouflage_pattern")
            .and_then(|v| v.as_str())
            .unwrap_or("adaptive");
        
        // Assess threat level
        let threat_level = self.assess_threat_level(threat_level_str).await?;
        
        // Generate camouflage pattern
        let pattern = self.generate_camouflage_pattern(threat_level, camouflage_pattern).await?;
        let pattern_effectiveness = pattern.get("effectiveness").and_then(|v| v.as_f64()).unwrap_or(0.0);
        
        // Calculate adaptation strategy
        let adaptation_strategy = self.calculate_adaptation_strategy(threat_level, pattern_effectiveness).await?;
        
        // Update manager's camouflage level
        self.manager.update_camouflage_level(pattern_effectiveness).await?;
        
        let execution_time_ns = start_time.elapsed().as_nanos() as u64;
        
        // Ensure sub-millisecond performance
        if execution_time_ns >= 1_000_000 {
            return Err(Error::PerformanceViolation {
                actual_ns: execution_time_ns,
                max_ns: 1_000_000,
            });
        }
        
        Ok(json!({
            "camouflage_active": true,
            "threat_assessment": {
                "threat_level_input": threat_level_str,
                "numerical_threat_level": threat_level,
                "threat_classification": if threat_level > 0.8 { "critical" } else if threat_level > 0.5 { "high" } else { "manageable" }
            },
            "camouflage_pattern": pattern,
            "adaptation_strategy": adaptation_strategy,
            "octopus_integration": {
                "organism_available": self.manager.get_organism("octopus").await?.is_some(),
                "chromatophore_response_time_ms": (threat_level * 100.0) as u64,
                "texture_adaptation_active": true
            },
            "execution_time_ns": execution_time_ns,
            "activation_timestamp": Utc::now(),
            "camouflage_effectiveness": pattern_effectiveness
        }))
    }
    
    async fn validate_input(&self, input: &Value) -> Result<()> {
        if !input.is_object() {
            return Err(Error::Configuration("Input must be an object".to_string()));
        }
        
        if input.get("threat_level").is_none() {
            return Err(Error::Configuration("threat_level is required".to_string()));
        }
        
        if let Some(threat_level) = input.get("threat_level").and_then(|v| v.as_str()) {
            match threat_level.to_lowercase().as_str() {
                "low" | "medium" | "high" | "critical" => {},
                _ => return Err(Error::Configuration("threat_level must be one of: low, medium, high, critical".to_string())),
            }
        }
        
        if let Some(pattern) = input.get("camouflage_pattern").and_then(|v| v.as_str()) {
            match pattern.to_lowercase().as_str() {
                "mimetic" | "disruptive" | "transparent" | "adaptive" => {},
                _ => return Err(Error::Configuration("camouflage_pattern must be one of: mimetic, disruptive, transparent, adaptive".to_string())),
            }
        }
        
        Ok(())
    }
    
    fn supports_websocket(&self) -> bool {
        true
    }
    
    async fn subscribe(&self, subscription_data: Value) -> Result<String> {
        let subscription_id = format!("camouflage-{}", Utc::now().timestamp());
        self.manager.add_subscription(subscription_id.clone(), "activate_octopus_camouflage".to_string(), subscription_data).await?;
        Ok(subscription_id)
    }
    
    async fn unsubscribe(&self, subscription_id: &str) -> Result<bool> {
        self.manager.remove_subscription(subscription_id).await
    }
}