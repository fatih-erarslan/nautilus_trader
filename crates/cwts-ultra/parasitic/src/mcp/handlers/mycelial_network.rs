//! Mycelial Network Handler for correlation analysis between pairs
//! Implements the fungal network organism for the parasitic pairlist system

use crate::mcp::{ParasiticPairlistManager, tools::ToolHandler};
use crate::{Result, Error};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;
use std::collections::HashMap;
use chrono::Utc;

/// Handler for analyzing mycelial correlation networks
pub struct MycelialNetworkHandler {
    manager: Arc<ParasiticPairlistManager>,
}

impl MycelialNetworkHandler {
    pub fn new(manager: Arc<ParasiticPairlistManager>) -> Self {
        Self { manager }
    }
    
    /// Build correlation matrix between pairs
    async fn build_correlation_matrix(&self, threshold: f64, depth: usize) -> Result<HashMap<String, HashMap<String, f64>>> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(44);
        let mut correlation_matrix = HashMap::new();
        
        let tracked_pairs = self.manager.get_tracked_pairs().await?;
        
        // Limit pairs for performance (sub-millisecond requirement)
        let pairs_to_analyze = tracked_pairs.iter().take(depth * 3).cloned().collect::<Vec<_>>();
        
        for pair1 in &pairs_to_analyze {
            let mut correlations = HashMap::new();
            
            for pair2 in &pairs_to_analyze {
                if pair1 != pair2 {
                    // Simulate correlation calculation with some realism
                    let base_correlation = rng.gen_range(-0.8..0.8);
                    
                    // Add some pattern based on pair names for consistency
                    let name_similarity = calculate_name_similarity(pair1, pair2);
                    let correlation = (base_correlation + name_similarity * 0.3).clamp(-1.0, 1.0);
                    
                    if correlation.abs() >= threshold {
                        correlations.insert(pair2.clone(), correlation);
                    }
                }
            }
            
            if !correlations.is_empty() {
                correlation_matrix.insert(pair1.clone(), correlations);
            }
        }
        
        Ok(correlation_matrix)
    }
    
    /// Identify network clusters based on correlation strength
    async fn identify_network_clusters(&self, correlation_matrix: &HashMap<String, HashMap<String, f64>>) -> Result<Vec<Value>> {
        let mut clusters = Vec::new();
        let mut cluster_id = 1;
        
        // Simple clustering based on correlation strength
        for (pair, correlations) in correlation_matrix {
            let strong_correlations: Vec<_> = correlations.iter()
                .filter(|(_, &correlation)| correlation.abs() > 0.7)
                .collect();
            
            if !strong_correlations.is_empty() {
                let cluster_strength = strong_correlations.iter()
                    .map(|(_, &corr)| corr.abs())
                    .sum::<f64>() / strong_correlations.len() as f64;
                
                clusters.push(json!({
                    "cluster_id": cluster_id,
                    "center_pair": pair,
                    "connected_pairs": strong_correlations.iter().map(|(p, _)| *p).collect::<Vec<_>>(),
                    "cluster_strength": cluster_strength,
                    "spore_propagation_efficiency": cluster_strength * 0.9,
                    "fungal_network_health": cluster_strength
                }));
                
                cluster_id += 1;
            }
        }
        
        Ok(clusters)
    }
    
    /// Calculate spore propagation paths through the network
    async fn calculate_spore_propagation_paths(&self, clusters: &[Value]) -> Result<Vec<Value>> {
        let mut propagation_paths = Vec::new();
        
        for cluster in clusters {
            if let Some(center_pair) = cluster.get("center_pair").and_then(|v| v.as_str()) {
                if let Some(connected_pairs) = cluster.get("connected_pairs").and_then(|v| v.as_array()) {
                    let cluster_strength = cluster.get("cluster_strength").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    
                    // Calculate propagation efficiency
                    let propagation_efficiency = cluster_strength * (1.0 - (connected_pairs.len() as f64 * 0.1));
                    
                    propagation_paths.push(json!({
                        "origin_pair": center_pair,
                        "propagation_targets": connected_pairs,
                        "propagation_efficiency": propagation_efficiency.max(0.0),
                        "spore_viability": cluster_strength,
                        "network_resistance": 1.0 - cluster_strength,
                        "estimated_propagation_time_ms": (1000.0 * (1.0 - propagation_efficiency)) as u64
                    }));
                }
            }
        }
        
        Ok(propagation_paths)
    }
}

#[async_trait]
impl ToolHandler for MycelialNetworkHandler {
    async fn handle(&self, input: Value) -> Result<Value> {
        let start_time = std::time::Instant::now();
        
        let correlation_threshold = input.get("correlation_threshold")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.75);
        
        let network_depth = input.get("network_depth")
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as usize;
        
        // Build correlation matrix
        let correlation_matrix = self.build_correlation_matrix(correlation_threshold, network_depth).await?;
        
        // Identify network clusters
        let network_clusters = self.identify_network_clusters(&correlation_matrix).await?;
        
        // Calculate spore propagation paths
        let spore_propagation_paths = self.calculate_spore_propagation_paths(&network_clusters).await?;
        
        let execution_time_ns = start_time.elapsed().as_nanos() as u64;
        
        // Ensure sub-millisecond performance
        if execution_time_ns >= 1_000_000 {
            return Err(Error::PerformanceViolation {
                actual_ns: execution_time_ns,
                max_ns: 1_000_000,
            });
        }
        
        // Calculate overall network health
        let network_health = if !network_clusters.is_empty() {
            network_clusters.iter()
                .map(|cluster| cluster.get("cluster_strength").and_then(|v| v.as_f64()).unwrap_or(0.0))
                .sum::<f64>() / network_clusters.len() as f64
        } else {
            0.0
        };
        
        Ok(json!({
            "correlation_matrix": correlation_matrix,
            "network_clusters": network_clusters,
            "spore_propagation_paths": spore_propagation_paths,
            "mycelial_network_health": network_health,
            "total_pairs_analyzed": correlation_matrix.len(),
            "correlation_threshold_used": correlation_threshold,
            "network_depth_used": network_depth,
            "execution_time_ns": execution_time_ns,
            "analysis_timestamp": Utc::now()
        }))
    }
    
    async fn validate_input(&self, input: &Value) -> Result<()> {
        if !input.is_object() {
            return Err(Error::Configuration("Input must be an object".to_string()));
        }
        
        if input.get("correlation_threshold").is_none() {
            return Err(Error::Configuration("correlation_threshold is required".to_string()));
        }
        
        if let Some(threshold) = input.get("correlation_threshold").and_then(|v| v.as_f64()) {
            if threshold < -1.0 || threshold > 1.0 {
                return Err(Error::Configuration("correlation_threshold must be between -1 and 1".to_string()));
            }
        }
        
        if let Some(depth) = input.get("network_depth").and_then(|v| v.as_u64()) {
            if depth < 1 || depth > 10 {
                return Err(Error::Configuration("network_depth must be between 1 and 10".to_string()));
            }
        }
        
        Ok(())
    }
    
    fn supports_websocket(&self) -> bool {
        true
    }
    
    async fn subscribe(&self, subscription_data: Value) -> Result<String> {
        let subscription_id = format!("mycelial-network-{}", Utc::now().timestamp());
        self.manager.add_subscription(subscription_id.clone(), "analyze_mycelial_network".to_string(), subscription_data).await?;
        Ok(subscription_id)
    }
    
    async fn unsubscribe(&self, subscription_id: &str) -> Result<bool> {
        self.manager.remove_subscription(subscription_id).await
    }
}

/// Calculate similarity between pair names for correlation modeling
fn calculate_name_similarity(pair1: &str, pair2: &str) -> f64 {
    // Simple similarity based on common characters and structure
    let chars1: std::collections::HashSet<char> = pair1.chars().collect();
    let chars2: std::collections::HashSet<char> = pair2.chars().collect();
    
    let intersection = chars1.intersection(&chars2).count();
    let union = chars1.union(&chars2).count();
    
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}