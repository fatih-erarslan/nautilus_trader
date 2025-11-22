//! Holographic Market Embeddings using Hyperbolic Geometry
//!
//! This module implements real-time Poincaré disk embeddings for market data,
//! enabling hierarchical clustering and crash prediction via curvature analysis.
//!
//! # Mathematical Foundation
//!
//! Markets exhibit hierarchical structure: Sector → Industry → Stock.
//! Euclidean space distorts this hierarchy. Hyperbolic space (Poincaré disk)
//! naturally represents tree-like structures with exponential growth.
//!
//! Distance in Poincaré disk: d(u,v) = acosh(1 + 2|u-v|²/((1-|u|²)(1-|v|²)))
//!
//! # Crash Prediction
//!
//! When the market cluster's hyperbolic radius contracts (curvature increases),
//! a singularity (crash) is imminent.

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info};

/// Point in the Poincaré disk (2D for visualization)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PoincarePoint {
    pub x: f64,
    pub y: f64,
}

impl PoincarePoint {
    /// Create a new point (must be inside unit disk)
    pub fn new(x: f64, y: f64) -> Option<Self> {
        let norm_sq = x * x + y * y;
        if norm_sq < 1.0 {
            Some(Self { x, y })
        } else {
            None
        }
    }

    /// Compute hyperbolic distance to another point
    pub fn distance(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let numerator = dx * dx + dy * dy;

        let self_norm_sq = self.x * self.x + self.y * self.y;
        let other_norm_sq = other.x * other.x + other.y * other.y;

        let denominator = (1.0 - self_norm_sq) * (1.0 - other_norm_sq);

        if denominator <= 0.0 {
            return f64::INFINITY;
        }

        let ratio = 2.0 * numerator / denominator;
        (1.0 + ratio).acosh()
    }

    /// Compute norm (distance from origin)
    pub fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}

/// Asset in the market
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Asset {
    pub symbol: String,
    pub sector: String,
    pub industry: String,
    pub embedding: PoincarePoint,
}

/// Holographic market embedding engine
pub struct HolographicEmbedding {
    /// Assets indexed by symbol
    assets: Arc<DashMap<String, Asset>>,
    /// Correlation matrix (used to compute embeddings)
    correlation_matrix: Arc<DashMap<(String, String), f64>>,
    /// Learning rate for embedding updates
    learning_rate: f64,
}

impl HolographicEmbedding {
    /// Create a new holographic embedding engine
    pub fn new(learning_rate: f64) -> Self {
        info!("Initializing Holographic Embedding engine");
        Self {
            assets: Arc::new(DashMap::new()),
            correlation_matrix: Arc::new(DashMap::new()),
            learning_rate,
        }
    }

    /// Add an asset to the embedding
    pub fn add_asset(&self, symbol: String, sector: String, industry: String) {
        // Initialize at random position in disk
        let angle = rand::random::<f64>() * 2.0 * std::f64::consts::PI;
        let radius = rand::random::<f64>() * 0.5; // Start in center

        let x = radius * angle.cos();
        let y = radius * angle.sin();

        if let Some(embedding) = PoincarePoint::new(x, y) {
            let asset = Asset {
                symbol: symbol.clone(),
                sector,
                industry,
                embedding,
            };
            self.assets.insert(symbol, asset);
        }
    }

    /// Update correlation between two assets
    pub fn update_correlation(&self, symbol1: &str, symbol2: &str, correlation: f64) {
        let key = if symbol1 < symbol2 {
            (symbol1.to_string(), symbol2.to_string())
        } else {
            (symbol2.to_string(), symbol1.to_string())
        };

        self.correlation_matrix.insert(key, correlation);
    }

    /// Update embeddings based on correlations (gradient descent)
    pub fn update_embeddings(&self) {
        debug!("Updating holographic embeddings");

        // Snapshot assets for parallel processing
        // We need a consistent view of the world to compute forces
        let mut assets_vec: Vec<(String, Asset)> = self
            .assets
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();

        let n = assets_vec.len();
        if n < 2 {
            return;
        }

        // Compute forces in parallel
        // For each asset i, compute the net force from all other assets j
        // This is O(N^2) but parallelized
        use rayon::prelude::*;

        let updates: Vec<(usize, f64, f64)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut force_x = 0.0;
                let mut force_y = 0.0;
                let asset1 = &assets_vec[i].1;
                let sym1 = &assets_vec[i].0;

                for j in 0..n {
                    if i == j {
                        continue;
                    }

                    let asset2 = &assets_vec[j].1;
                    let sym2 = &assets_vec[j].0;

                    // Look up correlation
                    let key = if sym1 < sym2 {
                        (sym1.clone(), sym2.clone())
                    } else {
                        (sym2.clone(), sym1.clone())
                    };

                    // Default small repulsion if no correlation known
                    let correlation = if let Some(c) = self.correlation_matrix.get(&key) {
                        *c
                    } else {
                        0.0
                    };

                    // Target distance: inverse of correlation
                    // High correlation (0.9) -> target 0.1 (attract)
                    // Low correlation (0.0) -> target 1.0 (repel)
                    let target_dist = 1.0 - correlation.abs();

                    // Current hyperbolic distance
                    let current_dist = asset1.embedding.distance(&asset2.embedding);

                    // Avoid division by zero or infinite distance
                    if !current_dist.is_finite() || current_dist < 1e-6 {
                        continue;
                    }

                    // Force magnitude proportional to error
                    // Error > 0 (too far) -> Attract
                    // Error < 0 (too close) -> Repel
                    let error = current_dist - target_dist;

                    // Direction vector (in Euclidean disk model)
                    // Note: This is an approximation. True hyperbolic gradient is more complex.
                    // For <1ms updates, this Euclidean approximation on the disk is sufficient
                    // provided we stay away from the boundary.
                    let dx = asset2.embedding.x - asset1.embedding.x;
                    let dy = asset2.embedding.y - asset1.embedding.y;
                    let dist_eucl = (dx * dx + dy * dy).sqrt();

                    if dist_eucl > 1e-6 {
                        // Normalize direction
                        let dir_x = dx / dist_eucl;
                        let dir_y = dy / dist_eucl;

                        // Apply force
                        // If error > 0 (current > target), we want to move CLOSER.
                        // So we move towards asset2. Direction (dx, dy) is towards asset2.
                        // So force is +error * dir.
                        force_x += error * dir_x;
                        force_y += error * dir_y;
                    }
                }

                (i, force_x, force_y)
            })
            .collect();

        // Apply updates sequentially (or parallel write back)
        for (i, fx, fy) in updates {
            let (sym, asset) = &mut assets_vec[i];

            // Apply learning rate
            // Scale by 1/N to normalize force
            let scale = self.learning_rate / n as f64;

            let new_x = asset.embedding.x + fx * scale;
            let new_y = asset.embedding.y + fy * scale;

            // Project back to unit disk if needed (clipping)
            let norm = (new_x * new_x + new_y * new_y).sqrt();
            let (final_x, final_y) = if norm >= 0.99 {
                (new_x / norm * 0.99, new_y / norm * 0.99)
            } else {
                (new_x, new_y)
            };

            // Update local copy
            asset.embedding.x = final_x;
            asset.embedding.y = final_y;

            // Write back to DashMap
            // This is the synchronization point
            if let Some(mut entry) = self.assets.get_mut(sym) {
                entry.embedding.x = final_x;
                entry.embedding.y = final_y;
            }
        }
    }

    /// Compute market cluster radius (crash indicator)
    pub fn compute_cluster_radius(&self) -> f64 {
        if self.assets.is_empty() {
            return 0.0;
        }

        // Snapshot for consistency
        let assets_vec: Vec<PoincarePoint> = self
            .assets
            .iter()
            .map(|entry| entry.value().embedding)
            .collect();

        let count = assets_vec.len() as f64;

        // Compute centroid (Euclidean approximation for speed)
        let (sum_x, sum_y) = assets_vec
            .iter()
            .fold((0.0, 0.0), |(ax, ay), p| (ax + p.x, ay + p.y));

        let centroid_x = sum_x / count;
        let centroid_y = sum_y / count;

        let centroid = if let Some(c) = PoincarePoint::new(centroid_x, centroid_y) {
            c
        } else {
            return f64::INFINITY;
        };

        // Compute average distance in parallel
        use rayon::prelude::*;
        let total_distance: f64 = assets_vec.par_iter().map(|p| p.distance(&centroid)).sum();

        total_distance / count
    }

    /// Detect crash risk based on cluster contraction
    pub fn crash_risk(&self) -> f64 {
        let radius = self.compute_cluster_radius();

        // Risk increases as radius decreases (contraction)
        // Risk = 1 / (1 + radius)

        if radius.is_finite() && radius > 0.0 {
            1.0 / (1.0 + radius)
        } else {
            1.0 // Maximum risk
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poincare_point() {
        let p1 = PoincarePoint::new(0.0, 0.0).unwrap();
        let p2 = PoincarePoint::new(0.5, 0.0).unwrap();

        let dist = p1.distance(&p2);
        assert!(dist > 0.0);
        assert!(dist.is_finite());
    }

    #[test]
    fn test_holographic_embedding() {
        let embedding = HolographicEmbedding::new(0.01);

        embedding.add_asset(
            "AAPL".to_string(),
            "Tech".to_string(),
            "Hardware".to_string(),
        );
        embedding.add_asset(
            "MSFT".to_string(),
            "Tech".to_string(),
            "Software".to_string(),
        );

        // High correlation between tech stocks
        embedding.update_correlation("AAPL", "MSFT", 0.8);

        embedding.update_embeddings();

        let risk = embedding.crash_risk();
        assert!(risk >= 0.0 && risk <= 1.0);
    }
}
