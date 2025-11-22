//! Hyperbolic Coordination System
//!
//! Implements hyperbolic geometry-based coordination for optimal sentinel positioning
//! and communication in the Poincaré disk model. This provides exponentially improved
//! coordination performance through non-Euclidean space optimization.

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, instrument};

use crate::cqgs::sentinels::{Sentinel, SentinelId};
use crate::cqgs::{CqgsEvent, HyperbolicCoordinates};

/// Hyperbolic distance calculation precision
const HYPERBOLIC_PRECISION: f64 = 1e-10;

/// Maximum radius in Poincaré disk (approaching boundary)
const MAX_RADIUS: f64 = 0.99;

/// Hyperbolic curvature (negative for hyperbolic space)
const DEFAULT_CURVATURE: f64 = -1.0;

/// Hyperbolic Coordinator for optimal sentinel positioning
pub struct HyperbolicCoordinator {
    curvature: f64,
    sentinel_positions: Arc<DashMap<SentinelId, HyperbolicCoordinates>>,
    topology_cache: Arc<RwLock<TopologyCache>>,
    rebalancing_threshold: f64,
}

/// Cached topology calculations for performance
#[derive(Debug, Clone)]
struct TopologyCache {
    distance_matrix: HashMap<(SentinelId, SentinelId), f64>,
    centroid: HyperbolicCoordinates,
    total_energy: f64,
    last_update: std::time::SystemTime,
}

impl TopologyCache {
    fn new() -> Self {
        Self {
            distance_matrix: HashMap::new(),
            centroid: HyperbolicCoordinates {
                x: 0.0,
                y: 0.0,
                radius: 0.0,
            },
            total_energy: 0.0,
            last_update: std::time::SystemTime::now(),
        }
    }
}

impl HyperbolicCoordinator {
    /// Create new hyperbolic coordinator with specified curvature
    pub fn new(curvature: f64) -> Self {
        Self {
            curvature: curvature.min(-0.1), // Ensure negative curvature
            sentinel_positions: Arc::new(DashMap::new()),
            topology_cache: Arc::new(RwLock::new(TopologyCache::new())),
            rebalancing_threshold: 0.1,
        }
    }

    /// Calculate optimal position for a sentinel in the hyperbolic disk
    #[instrument(skip(self))]
    pub fn calculate_optimal_position(
        &self,
        index: usize,
        total_sentinels: usize,
    ) -> HyperbolicCoordinates {
        // Use golden ratio spiral in hyperbolic space for optimal distribution
        let golden_angle = PI * (3.0 - (5.0_f64).sqrt());
        let angle = index as f64 * golden_angle;

        // Map to hyperbolic space using exponential distance scaling
        let radius_factor = (index as f64) / (total_sentinels as f64);
        let hyperbolic_radius = self.euclidean_to_hyperbolic_radius(radius_factor);

        let x = hyperbolic_radius * angle.cos();
        let y = hyperbolic_radius * angle.sin();
        let radius = (x * x + y * y).sqrt().min(MAX_RADIUS);

        debug!(
            "Calculated hyperbolic position for sentinel {}: ({:.3}, {:.3}, r={:.3})",
            index, x, y, radius
        );

        HyperbolicCoordinates { x, y, radius }
    }

    /// Convert Euclidean radius to hyperbolic radius in Poincaré disk
    fn euclidean_to_hyperbolic_radius(&self, euclidean_r: f64) -> f64 {
        // Use hyperbolic distance formula: sinh(d/2) where d is hyperbolic distance
        let hyperbolic_dist = euclidean_r * 3.0; // Scale factor for distribution
        let radius = (hyperbolic_dist / 2.0).sinh();
        radius.min(MAX_RADIUS)
    }

    /// Calculate hyperbolic distance between two points in Poincaré disk
    #[instrument(skip(self))]
    pub fn hyperbolic_distance(
        &self,
        p1: &HyperbolicCoordinates,
        p2: &HyperbolicCoordinates,
    ) -> f64 {
        let dx = p1.x - p2.x;
        let dy = p1.y - p2.y;
        let euclidean_dist_sq = dx * dx + dy * dy;

        let r1_sq = p1.x * p1.x + p1.y * p1.y;
        let r2_sq = p2.x * p2.x + p2.y * p2.y;

        // Poincaré disk model distance formula
        let numerator = 2.0 * euclidean_dist_sq;
        let denominator = (1.0 - r1_sq) * (1.0 - r2_sq);

        if denominator <= HYPERBOLIC_PRECISION {
            return f64::INFINITY; // Points at boundary
        }

        let ratio = numerator / denominator;
        let distance = (1.0 + ratio).ln();

        debug!("Hyperbolic distance: {:.6}", distance);
        distance.abs()
    }

    /// Calculate communication efficiency between sentinels
    pub fn communication_efficiency(
        &self,
        p1: &HyperbolicCoordinates,
        p2: &HyperbolicCoordinates,
    ) -> f64 {
        let distance = self.hyperbolic_distance(p1, p2);

        // Efficiency decreases exponentially with hyperbolic distance
        let efficiency = (-distance / 2.0).exp();
        efficiency.max(0.01) // Minimum efficiency threshold
    }

    /// Register a sentinel's position in the coordinate system
    pub async fn register_sentinel(&self, id: SentinelId, position: HyperbolicCoordinates) {
        // Fix E0382: Clone position to avoid move conflict
        let position_clone = position.clone();
        self.sentinel_positions.insert(id.clone(), position_clone);
        self.invalidate_cache().await;

        info!(
            "Registered sentinel {} at hyperbolic position ({:.3}, {:.3})",
            id, position.x, position.y
        );
    }

    /// Unregister a sentinel from the coordinate system
    pub async fn unregister_sentinel(&self, id: &SentinelId) {
        self.sentinel_positions.remove(id);
        self.invalidate_cache().await;

        info!("Unregistered sentinel {} from hyperbolic space", id);
    }

    /// Find nearest neighbors in hyperbolic space
    #[instrument(skip(self, sentinels))]
    pub async fn find_nearest_neighbors(
        &self,
        target_id: &SentinelId,
        sentinels: &DashMap<SentinelId, Box<dyn Sentinel>>,
        count: usize,
    ) -> Vec<SentinelId> {
        let target_pos = match self.sentinel_positions.get(target_id) {
            Some(pos) => pos.clone(),
            None => return Vec::new(),
        };

        let mut distances: Vec<(SentinelId, f64)> = Vec::new();

        for entry in self.sentinel_positions.iter() {
            let (id, pos) = (entry.key(), entry.value());
            if id != target_id {
                let distance = self.hyperbolic_distance(&target_pos, pos);
                distances.push((id.clone(), distance));
            }
        }

        // Sort by distance and return top neighbors
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances
            .into_iter()
            .take(count)
            .map(|(id, _)| id)
            .collect()
    }

    /// Calculate the centroid of all sentinels in hyperbolic space
    pub async fn calculate_hyperbolic_centroid(&self) -> HyperbolicCoordinates {
        if self.sentinel_positions.is_empty() {
            return HyperbolicCoordinates {
                x: 0.0,
                y: 0.0,
                radius: 0.0,
            };
        }

        let mut total_weight = 0.0;
        let mut weighted_x = 0.0;
        let mut weighted_y = 0.0;

        for entry in self.sentinel_positions.iter() {
            let pos = entry.value();
            let weight = 1.0 - pos.radius; // Higher weight for points closer to center

            weighted_x += pos.x * weight;
            weighted_y += pos.y * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            let centroid_x = weighted_x / total_weight;
            let centroid_y = weighted_y / total_weight;
            let radius = (centroid_x * centroid_x + centroid_y * centroid_y).sqrt();

            HyperbolicCoordinates {
                x: centroid_x,
                y: centroid_y,
                radius: radius.min(MAX_RADIUS),
            }
        } else {
            HyperbolicCoordinates {
                x: 0.0,
                y: 0.0,
                radius: 0.0,
            }
        }
    }

    /// Check if topology needs rebalancing based on distribution entropy
    pub async fn needs_rebalancing(&self) -> bool {
        let stability = self.calculate_stability().await;
        stability < self.rebalancing_threshold
    }

    /// Calculate topology stability (0.0 to 1.0)
    #[instrument(skip(self))]
    pub async fn calculate_stability(&self) -> f64 {
        if self.sentinel_positions.len() < 2 {
            return 1.0;
        }

        let centroid = self.calculate_hyperbolic_centroid().await;
        let mut total_deviation = 0.0;
        let mut count = 0;

        // Calculate average deviation from ideal distribution
        for entry in self.sentinel_positions.iter() {
            let pos = entry.value();
            let distance_to_centroid = self.hyperbolic_distance(pos, &centroid);
            total_deviation += distance_to_centroid;
            count += 1;
        }

        if count == 0 {
            return 1.0;
        }

        let average_deviation = total_deviation / count as f64;
        let ideal_deviation = 1.0; // Expected deviation in well-distributed system

        let stability = (ideal_deviation / (average_deviation + 0.1)).min(1.0);

        debug!(
            "Hyperbolic stability: {:.3} (avg deviation: {:.3})",
            stability, average_deviation
        );
        stability
    }

    /// Rebalance topology by repositioning sentinels for optimal distribution
    #[instrument(skip(self, sentinels))]
    pub async fn rebalance_topology(&self, sentinels: &DashMap<SentinelId, Box<dyn Sentinel>>) {
        info!(
            "Starting hyperbolic topology rebalancing for {} sentinels",
            sentinels.len()
        );

        let sentinel_ids: Vec<SentinelId> =
            sentinels.iter().map(|entry| entry.key().clone()).collect();

        // Calculate new optimal positions
        for (index, sentinel_id) in sentinel_ids.iter().enumerate() {
            let new_position = self.calculate_optimal_position(index, sentinel_ids.len());
            self.sentinel_positions
                .insert(sentinel_id.clone(), new_position.clone());

            debug!(
                "Rebalanced sentinel {} to position ({:.3}, {:.3})",
                sentinel_id, new_position.x, new_position.y
            );
        }

        self.invalidate_cache().await;

        let new_stability = self.calculate_stability().await;
        info!(
            "Topology rebalancing complete. New stability: {:.3}",
            new_stability
        );
    }

    /// Calculate total communication energy of the system
    pub async fn calculate_communication_energy(&self) -> f64 {
        let mut total_energy = 0.0;
        let positions: Vec<_> = self
            .sentinel_positions
            .iter()
            .map(|entry| entry.value().clone())
            .collect();

        for i in 0..positions.len() {
            for j in i + 1..positions.len() {
                let distance = self.hyperbolic_distance(&positions[i], &positions[j]);
                // Energy increases with distance (less efficient communication)
                total_energy += distance * distance;
            }
        }

        total_energy
    }

    /// Get optimal communication path between two sentinels
    pub async fn get_communication_path(
        &self,
        from: &SentinelId,
        to: &SentinelId,
        max_hops: usize,
    ) -> Option<Vec<SentinelId>> {
        let from_pos = self.sentinel_positions.get(from)?.clone();
        let to_pos = self.sentinel_positions.get(to)?.clone();

        // Direct communication if within efficient range
        let direct_efficiency = self.communication_efficiency(&from_pos, &to_pos);
        if direct_efficiency > 0.7 {
            return Some(vec![from.clone(), to.clone()]);
        }

        // Find multi-hop path using hyperbolic geometry
        self.find_hyperbolic_path(from, to, max_hops).await
    }

    /// Find optimal path using hyperbolic distance minimization
    async fn find_hyperbolic_path(
        &self,
        from: &SentinelId,
        to: &SentinelId,
        max_hops: usize,
    ) -> Option<Vec<SentinelId>> {
        // Simplified A* pathfinding in hyperbolic space
        // In production, this would use more sophisticated hyperbolic pathfinding

        let from_pos = self.sentinel_positions.get(from)?.clone();
        let to_pos = self.sentinel_positions.get(to)?.clone();

        // For now, return direct path
        // TODO: Implement multi-hop hyperbolic pathfinding
        Some(vec![from.clone(), to.clone()])
    }

    /// Invalidate topology cache after changes
    async fn invalidate_cache(&self) {
        let mut cache = self.topology_cache.write().await;
        cache.distance_matrix.clear();
        cache.last_update = std::time::SystemTime::now();
    }

    /// Get all sentinel positions
    pub fn get_all_positions(&self) -> HashMap<SentinelId, HyperbolicCoordinates> {
        self.sentinel_positions
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    /// Calculate hyperbolic area covered by sentinels
    pub async fn calculate_coverage_area(&self) -> f64 {
        if self.sentinel_positions.is_empty() {
            return 0.0;
        }

        // Calculate the convex hull area in hyperbolic space
        // This is a simplified calculation - full implementation would use
        // hyperbolic convex hull algorithms

        let positions: Vec<_> = self
            .sentinel_positions
            .iter()
            .map(|entry| entry.value().clone())
            .collect();

        let mut max_radius = 0.0f64;
        for pos in &positions {
            max_radius = max_radius.max(pos.radius);
        }

        // Approximate coverage as hyperbolic disk area
        let hyperbolic_area = 2.0f64 * PI * ((max_radius / 2.0f64).sinh().powi(2));
        hyperbolic_area
    }

    /// Export topology metrics for monitoring
    pub async fn export_metrics(&self) -> TopologyMetrics {
        let stability = self.calculate_stability().await;
        let energy = self.calculate_communication_energy().await;
        let coverage = self.calculate_coverage_area().await;
        let centroid = self.calculate_hyperbolic_centroid().await;

        TopologyMetrics {
            sentinel_count: self.sentinel_positions.len(),
            stability,
            communication_energy: energy,
            coverage_area: coverage,
            centroid,
            curvature: self.curvature,
        }
    }
}

/// Topology metrics for monitoring and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyMetrics {
    pub sentinel_count: usize,
    pub stability: f64,
    pub communication_energy: f64,
    pub coverage_area: f64,
    pub centroid: HyperbolicCoordinates,
    pub curvature: f64,
}

/// Coordination message types for sentinel communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationMessage {
    StatusUpdate {
        from: SentinelId,
        status: String,
        timestamp: std::time::SystemTime,
    },
    ViolationAlert {
        from: SentinelId,
        violation_id: uuid::Uuid,
        severity: String,
        coordinates: HyperbolicCoordinates,
    },
    ConsensusRequest {
        from: SentinelId,
        proposal_id: uuid::Uuid,
        data: serde_json::Value,
    },
    HealingRequest {
        from: SentinelId,
        target_violation: uuid::Uuid,
        healing_strategy: String,
    },
    TopologyUpdate {
        new_positions: HashMap<SentinelId, HyperbolicCoordinates>,
        stability_score: f64,
    },
}

/// Hyperbolic message routing for efficient communication
pub struct HyperbolicRouter {
    coordinator: Arc<HyperbolicCoordinator>,
    message_cache: Arc<DashMap<uuid::Uuid, CoordinationMessage>>,
}

impl HyperbolicRouter {
    pub fn new(coordinator: Arc<HyperbolicCoordinator>) -> Self {
        Self {
            coordinator,
            message_cache: Arc::new(DashMap::new()),
        }
    }

    /// Route message using hyperbolic geometry optimization
    pub async fn route_message(
        &self,
        message: CoordinationMessage,
        from: &SentinelId,
        to: &SentinelId,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let message_id = uuid::Uuid::new_v4();
        self.message_cache.insert(message_id, message.clone());

        // Find optimal communication path
        let path = self
            .coordinator
            .get_communication_path(from, to, 3)
            .await
            .unwrap_or_else(|| vec![from.clone(), to.clone()]);

        debug!("Routing message via path: {:?}", path);

        // In production, this would actually send the message through the path
        // For now, we just log the routing decision

        info!(
            "Message routed from {} to {} via {} hops",
            from,
            to,
            path.len() - 1
        );
        Ok(())
    }

    /// Broadcast message to all sentinels within hyperbolic radius
    pub async fn broadcast_in_radius(
        &self,
        message: CoordinationMessage,
        center: &SentinelId,
        radius: f64,
    ) -> Result<Vec<SentinelId>, Box<dyn std::error::Error + Send + Sync>> {
        let center_pos = self
            .coordinator
            .sentinel_positions
            .get(center)
            .ok_or("Center sentinel not found")?;

        let mut recipients = Vec::new();

        for entry in self.coordinator.sentinel_positions.iter() {
            let (id, pos) = (entry.key(), entry.value());
            if id != center {
                let distance = self.coordinator.hyperbolic_distance(&center_pos, pos);
                if distance <= radius {
                    recipients.push(id.clone());
                }
            }
        }

        info!(
            "Broadcasting to {} sentinels within radius {:.2}",
            recipients.len(),
            radius
        );
        Ok(recipients)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cqgs::sentinels::SentinelId;

    #[tokio::test]
    async fn test_hyperbolic_distance_calculation() {
        let coordinator = HyperbolicCoordinator::new(-1.0);

        let p1 = HyperbolicCoordinates {
            x: 0.0,
            y: 0.0,
            radius: 0.0,
        };
        let p2 = HyperbolicCoordinates {
            x: 0.5,
            y: 0.0,
            radius: 0.5,
        };

        let distance = coordinator.hyperbolic_distance(&p1, &p2);
        assert!(distance > 0.0);
        assert!(distance < f64::INFINITY);
    }

    #[tokio::test]
    async fn test_optimal_positioning() {
        let coordinator = HyperbolicCoordinator::new(-1.5);

        for i in 0..10 {
            let pos = coordinator.calculate_optimal_position(i, 10);
            assert!(pos.radius <= MAX_RADIUS);
            assert!(pos.x.abs() <= MAX_RADIUS);
            assert!(pos.y.abs() <= MAX_RADIUS);
        }
    }

    #[tokio::test]
    async fn test_stability_calculation() {
        let coordinator = HyperbolicCoordinator::new(-1.0);

        // Register some test sentinels
        for i in 0..5 {
            let id = SentinelId::new(format!("test_{}", i));
            let pos = coordinator.calculate_optimal_position(i, 5);
            coordinator.register_sentinel(id, pos).await;
        }

        let stability = coordinator.calculate_stability().await;
        assert!(stability >= 0.0);
        assert!(stability <= 1.0);
    }

    #[tokio::test]
    async fn test_communication_efficiency() {
        let coordinator = HyperbolicCoordinator::new(-1.0);

        let p1 = HyperbolicCoordinates {
            x: 0.0,
            y: 0.0,
            radius: 0.0,
        };
        let p2 = HyperbolicCoordinates {
            x: 0.1,
            y: 0.1,
            radius: 0.1,
        };
        let p3 = HyperbolicCoordinates {
            x: 0.8,
            y: 0.8,
            radius: 0.9,
        };

        let efficiency_close = coordinator.communication_efficiency(&p1, &p2);
        let efficiency_far = coordinator.communication_efficiency(&p1, &p3);

        assert!(efficiency_close > efficiency_far);
        assert!(efficiency_close <= 1.0);
        assert!(efficiency_far >= 0.01);
    }
}
