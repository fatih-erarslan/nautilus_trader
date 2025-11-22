//! # Mycelial Network Analysis
//! 
//! Fungal network-inspired correlation analysis for inter-pair relationships

use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};
use chrono::{DateTime, Utc};
use nalgebra::{DMatrix, DVector};

/// Mycelial network analyzer for inter-pair correlations
pub struct MycelialNetworkAnalyzer {
    /// Network sensitivity threshold
    pub sensitivity: f64,
    /// Minimum correlation strength
    pub min_correlation: f64,
    /// Active correlation networks
    pub networks: HashMap<String, CorrelationNetwork>,
    /// Historical network data
    pub network_history: Vec<NetworkSnapshot>,
}

/// Correlation network representing connected trading pairs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationNetwork {
    pub network_id: String,
    pub root_pairs: Vec<String>,
    pub connected_pairs: Vec<String>,
    pub correlation_matrix: Vec<Vec<f64>>,
    pub network_strength: f64,
    pub formation_time: DateTime<Utc>,
    pub last_update: DateTime<Utc>,
    pub spores: Vec<NetworkSpore>,
}

/// Network spore for spreading correlations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkSpore {
    pub spore_id: String,
    pub source_pair: String,
    pub target_candidates: Vec<String>,
    pub propagation_strength: f64,
    pub maturity_score: f64,
    pub release_conditions: Vec<ReleaseCondition>,
}

/// Conditions for spore release and network expansion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseCondition {
    pub condition_type: ConditionType,
    pub threshold: f64,
    pub current_value: f64,
    pub met: bool,
}

/// Types of release conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConditionType {
    VolumeThreshold,
    VolatilitySpike,
    PriceCorrelation,
    TimeElapsed,
    MarketCondition,
}

/// Snapshot of network state at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkSnapshot {
    pub timestamp: DateTime<Utc>,
    pub network_count: usize,
    pub total_pairs_connected: usize,
    pub average_correlation: f64,
    pub network_density: f64,
    pub dominant_networks: Vec<String>,
}

/// Mycelial connection between pairs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MycelialConnection {
    pub connection_id: String,
    pub pair_a: String,
    pub pair_b: String,
    pub correlation_strength: f64,
    pub connection_type: ConnectionType,
    pub nutrient_flow: NutrientFlow,
    pub stability_score: f64,
    pub formation_timestamp: DateTime<Utc>,
}

/// Types of mycelial connections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectionType {
    Direct,
    Indirect,
    Symbiotic,
    Competitive,
    Parasitic,
}

/// Nutrient flow between connected pairs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NutrientFlow {
    pub flow_direction: FlowDirection,
    pub flow_rate: f64,
    pub nutrient_type: NutrientType,
    pub efficiency: f64,
}

/// Direction of nutrient flow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FlowDirection {
    AtoB,
    BtoA,
    Bidirectional,
    Stagnant,
}

/// Types of nutrients exchanged
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NutrientType {
    Volume,
    Volatility,
    Momentum,
    Liquidity,
    Information,
}

/// Network growth pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkGrowthPattern {
    pub pattern_id: String,
    pub growth_rate: f64,
    pub expansion_direction: Vec<String>,
    pub resource_requirements: Vec<ResourceRequirement>,
    pub competitive_advantages: Vec<String>,
}

/// Resource requirement for network growth
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirement {
    pub resource_type: String,
    pub amount_required: f64,
    pub availability: f64,
    pub competition_factor: f64,
}

impl MycelialNetworkAnalyzer {
    pub fn new(sensitivity: f64, min_correlation: f64) -> Self {
        Self {
            sensitivity,
            min_correlation,
            networks: HashMap::new(),
            network_history: Vec::new(),
        }
    }
    
    /// Analyze correlations and build mycelial networks
    pub async fn analyze_correlations(
        &mut self, 
        pairs: &[crate::pairlist::TradingPair]
    ) -> Vec<CorrelationNetwork> {
        // Calculate correlation matrix
        let correlation_matrix = self.calculate_correlation_matrix(pairs).await;
        
        // Identify network clusters
        let clusters = self.identify_correlation_clusters(&correlation_matrix, pairs);
        
        // Build networks from clusters
        let mut networks = Vec::new();
        for cluster in clusters {
            if let Some(network) = self.build_network_from_cluster(cluster, pairs).await {
                networks.push(network.clone());
                self.networks.insert(network.network_id.clone(), network);
            }
        }
        
        // Update network history
        self.update_network_snapshot().await;
        
        networks
    }
    
    /// Calculate correlation matrix between pairs
    async fn calculate_correlation_matrix(
        &self, 
        pairs: &[crate::pairlist::TradingPair]
    ) -> DMatrix<f64> {
        let n = pairs.len();
        let mut matrix = DMatrix::zeros(n, n);
        
        // Mock correlation calculation - would use real price/volume data
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    matrix[(i, j)] = 1.0;
                } else {
                    // Mock correlation based on pair similarity
                    let correlation = self.mock_calculate_pair_correlation(&pairs[i], &pairs[j]);
                    matrix[(i, j)] = correlation;
                    matrix[(j, i)] = correlation;
                }
            }
        }
        
        matrix
    }
    
    /// Mock correlation calculation
    fn mock_calculate_pair_correlation(
        &self, 
        _pair_a: &crate::pairlist::TradingPair, 
        _pair_b: &crate::pairlist::TradingPair
    ) -> f64 {
        // Mock implementation - would calculate real correlation
        fastrand::f64() * 2.0 - 1.0
    }
    
    /// Identify correlation clusters using graph analysis
    fn identify_correlation_clusters(
        &self, 
        correlation_matrix: &DMatrix<f64>, 
        pairs: &[crate::pairlist::TradingPair]
    ) -> Vec<Vec<usize>> {
        let mut clusters = Vec::new();
        let mut visited = HashSet::new();
        
        for i in 0..pairs.len() {
            if !visited.contains(&i) {
                let cluster = self.depth_first_search(i, correlation_matrix, &mut visited);
                if cluster.len() > 1 {
                    clusters.push(cluster);
                }
            }
        }
        
        clusters
    }
    
    /// Depth-first search for finding connected components
    fn depth_first_search(
        &self,
        start: usize,
        correlation_matrix: &DMatrix<f64>,
        visited: &mut HashSet<usize>
    ) -> Vec<usize> {
        let mut cluster = Vec::new();
        let mut stack = vec![start];
        
        while let Some(node) = stack.pop() {
            if !visited.contains(&node) {
                visited.insert(node);
                cluster.push(node);
                
                // Find connected nodes
                for j in 0..correlation_matrix.ncols() {
                    if !visited.contains(&j) && correlation_matrix[(node, j)].abs() >= self.min_correlation {
                        stack.push(j);
                    }
                }
            }
        }
        
        cluster
    }
    
    /// Build a correlation network from a cluster
    async fn build_network_from_cluster(
        &self,
        cluster: Vec<usize>,
        pairs: &[crate::pairlist::TradingPair]
    ) -> Option<CorrelationNetwork> {
        if cluster.len() < 2 {
            return None;
        }
        
        let network_id = uuid::Uuid::new_v4().to_string();
        let pair_ids: Vec<String> = cluster.iter()
            .map(|&i| pairs[i].pair_id.clone())
            .collect();
        
        // Calculate correlation matrix for this cluster
        let mut correlation_matrix = Vec::new();
        for i in 0..cluster.len() {
            let mut row = Vec::new();
            for j in 0..cluster.len() {
                let correlation = if i == j {
                    1.0
                } else {
                    self.mock_calculate_pair_correlation(&pairs[cluster[i]], &pairs[cluster[j]])
                };
                row.push(correlation);
            }
            correlation_matrix.push(row);
        }
        
        // Calculate network strength
        let network_strength = self.calculate_network_strength(&correlation_matrix);
        
        // Generate spores for network expansion
        let spores = self.generate_network_spores(&pair_ids);
        
        Some(CorrelationNetwork {
            network_id,
            root_pairs: pair_ids[..2].to_vec(), // First two as roots
            connected_pairs: pair_ids,
            correlation_matrix,
            network_strength,
            formation_time: Utc::now(),
            last_update: Utc::now(),
            spores,
        })
    }
    
    /// Calculate overall network strength
    fn calculate_network_strength(&self, correlation_matrix: &[Vec<f64>]) -> f64 {
        let mut total_correlation = 0.0;
        let mut count = 0;
        
        for i in 0..correlation_matrix.len() {
            for j in (i + 1)..correlation_matrix[i].len() {
                total_correlation += correlation_matrix[i][j].abs();
                count += 1;
            }
        }
        
        if count > 0 {
            total_correlation / count as f64
        } else {
            0.0
        }
    }
    
    /// Generate spores for network expansion
    fn generate_network_spores(&self, pair_ids: &[String]) -> Vec<NetworkSpore> {
        let mut spores = Vec::new();
        
        for pair_id in pair_ids {
            let spore = NetworkSpore {
                spore_id: uuid::Uuid::new_v4().to_string(),
                source_pair: pair_id.clone(),
                target_candidates: vec![], // Would be populated with potential targets
                propagation_strength: fastrand::f64(),
                maturity_score: 0.0,
                release_conditions: vec![
                    ReleaseCondition {
                        condition_type: ConditionType::VolumeThreshold,
                        threshold: 1000000.0,
                        current_value: 0.0,
                        met: false,
                    },
                ],
            };
            spores.push(spore);
        }
        
        spores
    }
    
    /// Update network snapshot for historical tracking
    async fn update_network_snapshot(&mut self) {
        let total_pairs_connected: usize = self.networks.values()
            .map(|n| n.connected_pairs.len())
            .sum();
            
        let average_correlation = if !self.networks.is_empty() {
            self.networks.values()
                .map(|n| n.network_strength)
                .sum::<f64>() / self.networks.len() as f64
        } else {
            0.0
        };
        
        let network_density = if total_pairs_connected > 0 {
            self.networks.len() as f64 / total_pairs_connected as f64
        } else {
            0.0
        };
        
        let dominant_networks: Vec<String> = self.networks.values()
            .filter(|n| n.network_strength > average_correlation)
            .map(|n| n.network_id.clone())
            .collect();
        
        let snapshot = NetworkSnapshot {
            timestamp: Utc::now(),
            network_count: self.networks.len(),
            total_pairs_connected,
            average_correlation,
            network_density,
            dominant_networks,
        };
        
        self.network_history.push(snapshot);
        
        // Keep only last 1000 snapshots
        if self.network_history.len() > 1000 {
            self.network_history.remove(0);
        }
    }
    
    /// Get parasitic opportunities from network analysis
    pub fn get_network_opportunities(&self) -> Vec<NetworkOpportunity> {
        let mut opportunities = Vec::new();
        
        for network in self.networks.values() {
            // Find weak connections that can be parasitized
            for (i, row) in network.correlation_matrix.iter().enumerate() {
                for (j, &correlation) in row.iter().enumerate() {
                    if i != j && correlation.abs() < 0.3 && correlation.abs() > 0.1 {
                        opportunities.push(NetworkOpportunity {
                            opportunity_id: uuid::Uuid::new_v4().to_string(),
                            network_id: network.network_id.clone(),
                            source_pair: network.connected_pairs[i].clone(),
                            target_pair: network.connected_pairs[j].clone(),
                            weakness_score: 1.0 - correlation.abs(),
                            exploitation_strategy: "correlation_break".to_string(),
                            expected_yield: correlation.abs() * 100.0,
                            risk_assessment: 1.0 - network.network_strength,
                        });
                    }
                }
            }
        }
        
        opportunities.sort_by(|a, b| {
            b.expected_yield.partial_cmp(&a.expected_yield).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        opportunities
    }
    
    /// Get network by ID
    pub fn get_network(&self, network_id: &str) -> Option<&CorrelationNetwork> {
        self.networks.get(network_id)
    }
    
    /// Get all active networks
    pub fn get_all_networks(&self) -> Vec<&CorrelationNetwork> {
        self.networks.values().collect()
    }
}

/// Network-based parasitic opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkOpportunity {
    pub opportunity_id: String,
    pub network_id: String,
    pub source_pair: String,
    pub target_pair: String,
    pub weakness_score: f64,
    pub exploitation_strategy: String,
    pub expected_yield: f64,
    pub risk_assessment: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mycelial_analyzer_creation() {
        let analyzer = MycelialNetworkAnalyzer::new(0.8, 0.3);
        assert_eq!(analyzer.sensitivity, 0.8);
        assert_eq!(analyzer.min_correlation, 0.3);
        assert!(analyzer.networks.is_empty());
    }
    
    #[test]
    fn test_correlation_matrix_calculation() {
        let analyzer = MycelialNetworkAnalyzer::new(0.8, 0.3);
        let matrix = DMatrix::from_vec(2, 2, vec![1.0, 0.5, 0.5, 1.0]);
        assert_eq!(matrix[(0, 0)], 1.0);
        assert_eq!(matrix[(0, 1)], 0.5);
        assert_eq!(matrix[(1, 0)], 0.5);
        assert_eq!(matrix[(1, 1)], 1.0);
    }
    
    #[test]
    fn test_network_strength_calculation() {
        let analyzer = MycelialNetworkAnalyzer::new(0.8, 0.3);
        let correlation_matrix = vec![
            vec![1.0, 0.8, 0.6],
            vec![0.8, 1.0, 0.7],
            vec![0.6, 0.7, 1.0],
        ];
        let strength = analyzer.calculate_network_strength(&correlation_matrix);
        assert!(strength > 0.0);
        assert!(strength <= 1.0);
    }
}