//! Robin Hood Protocol - Wealth redistribution analysis and fairness optimization

use crate::error::{QBMIAError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Robin Hood Protocol for market fairness analysis
#[derive(Debug, Clone)]
pub struct RobinHoodProtocol {
    pub wealth_threshold: f64,
    pub redistribution_factor: f64,
    pub fairness_metrics: HashMap<String, f64>,
}

/// Wealth distribution analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WealthDistributionAnalysis {
    pub gini_coefficient: f64,
    pub wealth_concentration: HashMap<String, f64>,
    pub fairness_score: f64,
    pub redistribution_recommendations: Vec<String>,
}

impl RobinHoodProtocol {
    /// Create new Robin Hood protocol
    pub fn new(wealth_threshold: f64, redistribution_factor: f64) -> Self {
        Self {
            wealth_threshold,
            redistribution_factor,
            fairness_metrics: HashMap::new(),
        }
    }

    /// Analyze wealth distribution in market
    pub async fn analyze_wealth_distribution(
        &self,
        participant_wealth: &HashMap<String, f64>,
    ) -> Result<WealthDistributionAnalysis> {
        let gini = self.calculate_gini_coefficient(participant_wealth)?;
        let concentration = self.calculate_wealth_concentration(participant_wealth)?;
        let fairness = self.calculate_fairness_score(gini, &concentration)?;
        let recommendations = self.generate_redistribution_recommendations(
            participant_wealth,
            gini,
            fairness,
        )?;

        Ok(WealthDistributionAnalysis {
            gini_coefficient: gini,
            wealth_concentration: concentration,
            fairness_score: fairness,
            redistribution_recommendations: recommendations,
        })
    }

    /// Calculate Gini coefficient for wealth distribution
    fn calculate_gini_coefficient(&self, wealth: &HashMap<String, f64>) -> Result<f64> {
        let values: Vec<f64> = wealth.values().cloned().collect();
        if values.is_empty() {
            return Ok(0.0);
        }

        let mut sorted_values = values;
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted_values.len() as f64;
        let total_wealth: f64 = sorted_values.iter().sum();

        if total_wealth == 0.0 {
            return Ok(0.0);
        }

        let mut gini_sum = 0.0;
        for (i, &value) in sorted_values.iter().enumerate() {
            gini_sum += (2.0 * (i as f64 + 1.0) - n - 1.0) * value;
        }

        Ok(gini_sum / (n * total_wealth))
    }

    /// Calculate wealth concentration metrics
    fn calculate_wealth_concentration(&self, wealth: &HashMap<String, f64>) -> Result<HashMap<String, f64>> {
        let values: Vec<f64> = wealth.values().cloned().collect();
        if values.is_empty() {
            return Ok(HashMap::new());
        }

        let total_wealth: f64 = values.iter().sum();
        let mut sorted_values = values;
        sorted_values.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let mut metrics = HashMap::new();

        // Top 1% concentration
        let top_1_percent_count = (sorted_values.len() as f64 * 0.01).ceil() as usize;
        if top_1_percent_count > 0 {
            let top_1_percent_wealth: f64 = sorted_values.iter().take(top_1_percent_count).sum();
            metrics.insert("top_1_percent".to_string(), top_1_percent_wealth / total_wealth);
        }

        // Top 5% concentration
        let top_5_percent_count = (sorted_values.len() as f64 * 0.05).ceil() as usize;
        if top_5_percent_count > 0 {
            let top_5_percent_wealth: f64 = sorted_values.iter().take(top_5_percent_count).sum();
            metrics.insert("top_5_percent".to_string(), top_5_percent_wealth / total_wealth);
        }

        // Top 10% concentration
        let top_10_percent_count = (sorted_values.len() as f64 * 0.10).ceil() as usize;
        if top_10_percent_count > 0 {
            let top_10_percent_wealth: f64 = sorted_values.iter().take(top_10_percent_count).sum();
            metrics.insert("top_10_percent".to_string(), top_10_percent_wealth / total_wealth);
        }

        Ok(metrics)
    }

    /// Calculate overall fairness score
    fn calculate_fairness_score(
        &self,
        gini: f64,
        concentration: &HashMap<String, f64>,
    ) -> Result<f64> {
        // Lower Gini coefficient = higher fairness
        let gini_score = 1.0 - gini;

        // Lower concentration = higher fairness
        let concentration_score = if let Some(&top_10) = concentration.get("top_10_percent") {
            1.0 - (top_10 - 0.1).max(0.0) / 0.9 // Normalize assuming fair distribution would be 10%
        } else {
            0.5
        };

        // Weighted average
        let fairness = (gini_score * 0.6 + concentration_score * 0.4).max(0.0).min(1.0);
        
        Ok(fairness)
    }

    /// Generate redistribution recommendations
    fn generate_redistribution_recommendations(
        &self,
        wealth: &HashMap<String, f64>,
        gini: f64,
        fairness: f64,
    ) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();

        if gini > 0.4 {
            recommendations.push("High inequality detected - consider progressive taxation".to_string());
        }

        if fairness < 0.3 {
            recommendations.push("Poor market fairness - implement wealth redistribution mechanisms".to_string());
        }

        let total_wealth: f64 = wealth.values().sum();
        let avg_wealth = total_wealth / wealth.len() as f64;
        let high_wealth_count = wealth.values().filter(|&&w| w > self.wealth_threshold * avg_wealth).count();

        if high_wealth_count as f64 / wealth.len() as f64 > 0.1 {
            recommendations.push(format!(
                "{}% of participants control excessive wealth - apply Robin Hood redistribution",
                (high_wealth_count as f64 / wealth.len() as f64 * 100.0) as u32
            ));
        }

        if recommendations.is_empty() {
            recommendations.push("Wealth distribution appears fair - maintain current policies".to_string());
        }

        Ok(recommendations)
    }

    /// Apply Robin Hood redistribution
    pub async fn apply_redistribution(
        &self,
        wealth: &mut HashMap<String, f64>,
    ) -> Result<HashMap<String, f64>> {
        let total_wealth: f64 = wealth.values().sum();
        let avg_wealth = total_wealth / wealth.len() as f64;
        let redistribution_threshold = self.wealth_threshold * avg_wealth;

        let mut redistribution_pool = 0.0;
        let mut recipients = Vec::new();

        // Collect from wealthy participants
        for (participant, participant_wealth) in wealth.iter_mut() {
            if *participant_wealth > redistribution_threshold {
                let excess = *participant_wealth - redistribution_threshold;
                let tax = excess * self.redistribution_factor;
                *participant_wealth -= tax;
                redistribution_pool += tax;
            } else {
                recipients.push(participant.clone());
            }
        }

        // Distribute to less wealthy participants
        if !recipients.is_empty() && redistribution_pool > 0.0 {
            let per_recipient = redistribution_pool / recipients.len() as f64;
            for recipient in recipients {
                if let Some(wealth_value) = wealth.get_mut(&recipient) {
                    *wealth_value += per_recipient;
                }
            }
        }

        Ok(wealth.clone())
    }
}

impl Default for RobinHoodProtocol {
    fn default() -> Self {
        Self::new(2.0, 0.1) // 2x average wealth threshold, 10% redistribution factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gini_calculation() {
        let protocol = RobinHoodProtocol::default();
        let mut wealth = HashMap::new();
        wealth.insert("alice".to_string(), 100.0);
        wealth.insert("bob".to_string(), 200.0);
        wealth.insert("charlie".to_string(), 300.0);

        let gini = protocol.calculate_gini_coefficient(&wealth).unwrap();
        assert!(gini >= 0.0 && gini <= 1.0);
    }

    #[tokio::test]
    async fn test_wealth_distribution_analysis() {
        let protocol = RobinHoodProtocol::default();
        let mut wealth = HashMap::new();
        wealth.insert("alice".to_string(), 100.0);
        wealth.insert("bob".to_string(), 1000.0);
        wealth.insert("charlie".to_string(), 10000.0);

        let analysis = protocol.analyze_wealth_distribution(&wealth).await.unwrap();
        assert!(analysis.gini_coefficient > 0.0);
        assert!(analysis.fairness_score >= 0.0 && analysis.fairness_score <= 1.0);
        assert!(!analysis.redistribution_recommendations.is_empty());
    }

    #[tokio::test]
    async fn test_redistribution() {
        let protocol = RobinHoodProtocol::default();
        let mut wealth = HashMap::new();
        wealth.insert("alice".to_string(), 100.0);
        wealth.insert("bob".to_string(), 1000.0);

        let initial_total: f64 = wealth.values().sum();
        let redistributed = protocol.apply_redistribution(&mut wealth).await.unwrap();
        let final_total: f64 = redistributed.values().sum();

        // Total wealth should be conserved
        assert!((initial_total - final_total).abs() < 1e-10);
        
        // Wealth should be more evenly distributed
        let bob_final = redistributed.get("bob").unwrap();
        assert!(*bob_final < 1000.0); // Bob should have less
    }
}