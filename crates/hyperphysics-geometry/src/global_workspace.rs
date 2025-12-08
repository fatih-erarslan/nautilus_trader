//! # Global Workspace Theory Implementation
//!
//! Baars' Global Workspace Theory (GWT) for conscious broadcast in hyperbolic SNNs.
//!
//! ## Theoretical Foundation
//!
//! GWT proposes that consciousness arises from a "global workspace" that:
//! 1. Integrates information from specialized processors
//! 2. Broadcasts winning coalitions to all processors
//! 3. Creates a unified conscious experience
//!
//! ## Hyperbolic Extension
//!
//! In hyperbolic space, the workspace has natural geometric properties:
//! - Exponential growth of information capacity with radius
//! - Geodesic broadcasting along minimal paths
//! - Curvature-dependent coalition dynamics
//!
//! ## References
//!
//! - Baars (1988) "A Cognitive Theory of Consciousness" Cambridge
//! - Dehaene & Naccache (2001) "Towards a cognitive neuroscience of consciousness"
//! - Mashour et al. (2020) "Conscious Processing and the Global Neuronal Workspace"

use crate::hyperbolic_snn::LorentzVec;
use crate::chunk_processor::TemporalChunk;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Configuration for Global Workspace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalWorkspaceConfig {
    /// Number of specialist modules
    pub num_specialists: usize,
    /// Competition threshold for broadcast ignition
    pub ignition_threshold: f64,
    /// Broadcast decay time constant
    pub broadcast_decay: f64,
    /// Maximum coalition size
    pub max_coalition_size: usize,
    /// Workspace capacity (bits)
    pub workspace_capacity: usize,
    /// Minimum activation for workspace entry
    pub access_threshold: f64,
    /// Competition winner-take-all sharpness
    pub competition_temperature: f64,
}

impl Default for GlobalWorkspaceConfig {
    fn default() -> Self {
        Self {
            num_specialists: 8,
            ignition_threshold: 0.6,
            broadcast_decay: 50.0, // ms
            max_coalition_size: 5,
            workspace_capacity: 7, // Miller's 7±2
            access_threshold: 0.3,
            competition_temperature: 1.0,
        }
    }
}

/// Specialist module that competes for workspace access
#[derive(Debug, Clone)]
pub struct SpecialistModule {
    /// Unique identifier
    pub id: usize,
    /// Module type/function
    pub module_type: SpecialistType,
    /// Current activation level
    pub activation: f64,
    /// Position in hyperbolic space (centroid)
    pub position: LorentzVec,
    /// Current representation content
    pub content: WorkspaceContent,
    /// Connection strengths to other specialists
    pub connections: HashMap<usize, f64>,
    /// History of activations
    pub activation_history: VecDeque<f64>,
    /// Last broadcast time
    pub last_broadcast_time: f64,
}

/// Types of specialist modules
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpecialistType {
    /// Sensory processing
    Sensory,
    /// Motor planning
    Motor,
    /// Memory retrieval
    Memory,
    /// Attention control
    Attention,
    /// Language processing
    Language,
    /// Spatial reasoning
    Spatial,
    /// Temporal reasoning
    Temporal,
    /// Emotional valuation
    Emotional,
    /// Executive control
    Executive,
    /// Default/unspecified
    Generic,
}

/// Content that can be broadcast through workspace
#[derive(Debug, Clone)]
pub struct WorkspaceContent {
    /// Unique content ID
    pub id: u64,
    /// Creation time
    pub time: f64,
    /// Hyperbolic position of content
    pub position: LorentzVec,
    /// Feature vector
    pub features: Vec<f64>,
    /// Salience/importance score
    pub salience: f64,
    /// Source specialist
    pub source: usize,
    /// Semantic tag
    pub tag: String,
}

impl Default for WorkspaceContent {
    fn default() -> Self {
        Self {
            id: 0,
            time: 0.0,
            position: LorentzVec::origin(),
            features: vec![],
            salience: 0.0,
            source: 0,
            tag: String::new(),
        }
    }
}

impl SpecialistModule {
    /// Create new specialist module
    pub fn new(id: usize, module_type: SpecialistType, position: LorentzVec) -> Self {
        Self {
            id,
            module_type,
            activation: 0.0,
            position,
            content: WorkspaceContent::default(),
            connections: HashMap::new(),
            activation_history: VecDeque::with_capacity(100),
            last_broadcast_time: 0.0,
        }
    }

    /// Update activation from local processing
    pub fn update_activation(&mut self, input: f64, time: f64) {
        // Leaky integration
        let decay = (-1.0_f64 / 20.0).exp(); // 20ms time constant
        self.activation = self.activation * decay + input * (1.0 - decay);
        self.activation = self.activation.clamp(0.0, 1.0);

        self.activation_history.push_back(self.activation);
        if self.activation_history.len() > 100 {
            self.activation_history.pop_front();
        }
    }

    /// Receive broadcast from workspace
    pub fn receive_broadcast(&mut self, content: &WorkspaceContent, strength: f64) {
        // Modulate activation based on broadcast relevance
        let relevance = self.compute_relevance(content);
        self.activation = (self.activation + strength * relevance).clamp(0.0, 1.0);
    }

    /// Compute relevance of content to this specialist
    fn compute_relevance(&self, content: &WorkspaceContent) -> f64 {
        // Distance-based relevance (closer = more relevant in hyperbolic space)
        let distance = self.position.hyperbolic_distance(&content.position);
        (-distance / 2.0).exp()
    }

    /// Submit content to workspace
    pub fn propose_content(&mut self, features: Vec<f64>, salience: f64, time: f64) {
        static CONTENT_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

        self.content = WorkspaceContent {
            id: CONTENT_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            time,
            position: self.position,
            features,
            salience,
            source: self.id,
            tag: format!("{:?}", self.module_type),
        };
    }
}

/// Coalition of specialists competing for broadcast
#[derive(Debug, Clone)]
pub struct Coalition {
    /// Member specialist IDs
    pub members: Vec<usize>,
    /// Coalition strength (product of activations)
    pub strength: f64,
    /// Combined content
    pub content: WorkspaceContent,
    /// Formation time
    pub formation_time: f64,
    /// Hyperbolic centroid
    pub centroid: LorentzVec,
}

impl Coalition {
    /// Create coalition from specialists
    pub fn from_specialists(specialists: &[&SpecialistModule], time: f64) -> Self {
        let members: Vec<usize> = specialists.iter().map(|s| s.id).collect();

        // Compute coalition strength as product of activations
        let strength: f64 = specialists.iter()
            .map(|s| s.activation)
            .product();

        // Merge features from all specialists
        let mut features = Vec::new();
        for s in specialists {
            features.extend(s.content.features.iter().cloned());
        }

        // Compute centroid using Fréchet mean approximation
        let centroid = Self::frechet_mean(specialists.iter().map(|s| &s.position));

        // Combined salience
        let salience: f64 = specialists.iter()
            .map(|s| s.content.salience * s.activation)
            .sum::<f64>() / specialists.len() as f64;

        let content = WorkspaceContent {
            id: 0,
            time,
            position: centroid,
            features,
            salience,
            source: members[0],
            tag: "coalition".to_string(),
        };

        Self {
            members,
            strength,
            content,
            formation_time: time,
            centroid,
        }
    }

    /// Compute Fréchet mean in hyperbolic space using Riemannian gradient descent
    ///
    /// The Fréchet mean minimizes F(m) = Σᵢ wᵢ d²(m, xᵢ)
    /// Gradient: ∇F(m) = -2 Σᵢ wᵢ d(m,xᵢ) * Log_m(xᵢ)
    ///
    /// Convergence: For points in geodesic ball of radius r < π/(2√|K|),
    /// gradient descent converges with rate O(1/k) where K=-1 for H³.
    ///
    /// Reference: Karcher (1977) "Riemannian center of mass"
    fn frechet_mean<'a>(positions: impl Iterator<Item = &'a LorentzVec>) -> LorentzVec {
        let positions: Vec<_> = positions.collect();
        if positions.is_empty() {
            return LorentzVec::origin();
        }
        if positions.len() == 1 {
            return *positions[0];
        }

        // Initialize with Euclidean mean projected to hyperboloid
        let n = positions.len() as f64;
        let mut mean = {
            let sum_x: f64 = positions.iter().map(|p| p.x).sum::<f64>() / n;
            let sum_y: f64 = positions.iter().map(|p| p.y).sum::<f64>() / n;
            let sum_z: f64 = positions.iter().map(|p| p.z).sum::<f64>() / n;
            let spatial_sq = sum_x * sum_x + sum_y * sum_y + sum_z * sum_z;
            let t = (1.0 + spatial_sq).sqrt();
            LorentzVec::new(t, sum_x, sum_y, sum_z)
        };

        // Riemannian gradient descent parameters
        let max_iterations = 100;
        let tolerance = 1e-10;
        let initial_step_size = 0.5;

        for iteration in 0..max_iterations {
            // Compute gradient: ∇F(m) = -2 Σᵢ d(m,xᵢ) * Log_m(xᵢ)
            let mut grad_t = 0.0;
            let mut grad_x = 0.0;
            let mut grad_y = 0.0;
            let mut grad_z = 0.0;

            for p in &positions {
                let log_vec = mean.log_map(p);
                let dist = mean.hyperbolic_distance(p);

                // Weight by distance (for weighted mean, modify here)
                let weight = 2.0 * dist;

                grad_t += weight * log_vec.t;
                grad_x += weight * log_vec.x;
                grad_y += weight * log_vec.y;
                grad_z += weight * log_vec.z;
            }

            // Gradient magnitude (in tangent space metric)
            let grad_norm_sq = -grad_t * grad_t + grad_x * grad_x + grad_y * grad_y + grad_z * grad_z;
            let grad_norm = grad_norm_sq.abs().sqrt();

            if grad_norm < tolerance {
                break;
            }

            // Adaptive step size with Armijo line search approximation
            let step = initial_step_size / (1.0 + 0.1 * iteration as f64);

            // Update: m_new = Exp_m(-step * ∇F)
            // The negative gradient direction
            let update_vec = LorentzVec::new(
                -step * grad_t / grad_norm,
                -step * grad_x / grad_norm,
                -step * grad_y / grad_norm,
                -step * grad_z / grad_norm,
            );

            // Exponential map for the update
            mean = mean.exp_map(&update_vec, step * grad_norm);
        }

        mean
    }

    /// Compute weighted Fréchet mean with given weights
    pub fn frechet_mean_weighted(positions: &[LorentzVec], weights: &[f64]) -> LorentzVec {
        if positions.is_empty() || weights.is_empty() {
            return LorentzVec::origin();
        }
        if positions.len() == 1 {
            return positions[0];
        }

        // Normalize weights
        let weight_sum: f64 = weights.iter().sum();
        let normalized_weights: Vec<f64> = weights.iter().map(|w| w / weight_sum).collect();

        // Initialize with weighted Euclidean mean
        let mut mean = {
            let sum_x: f64 = positions.iter().zip(&normalized_weights)
                .map(|(p, w)| p.x * w).sum();
            let sum_y: f64 = positions.iter().zip(&normalized_weights)
                .map(|(p, w)| p.y * w).sum();
            let sum_z: f64 = positions.iter().zip(&normalized_weights)
                .map(|(p, w)| p.z * w).sum();
            let spatial_sq = sum_x * sum_x + sum_y * sum_y + sum_z * sum_z;
            let t = (1.0 + spatial_sq).sqrt();
            LorentzVec::new(t, sum_x, sum_y, sum_z)
        };

        // Riemannian gradient descent
        let max_iterations = 100;
        let tolerance = 1e-10;

        for iteration in 0..max_iterations {
            let mut grad_t = 0.0;
            let mut grad_x = 0.0;
            let mut grad_y = 0.0;
            let mut grad_z = 0.0;

            for (p, &w) in positions.iter().zip(&normalized_weights) {
                let log_vec = mean.log_map(p);
                let dist = mean.hyperbolic_distance(p);
                let weight = 2.0 * w * dist;

                grad_t += weight * log_vec.t;
                grad_x += weight * log_vec.x;
                grad_y += weight * log_vec.y;
                grad_z += weight * log_vec.z;
            }

            let grad_norm_sq = -grad_t * grad_t + grad_x * grad_x + grad_y * grad_y + grad_z * grad_z;
            let grad_norm = grad_norm_sq.abs().sqrt();

            if grad_norm < tolerance {
                break;
            }

            let step = 0.5 / (1.0 + 0.1 * iteration as f64);
            let update_vec = LorentzVec::new(
                -step * grad_t / grad_norm,
                -step * grad_x / grad_norm,
                -step * grad_y / grad_norm,
                -step * grad_z / grad_norm,
            );

            mean = mean.exp_map(&update_vec, step * grad_norm);
        }

        mean
    }
}

/// Broadcast event from workspace
#[derive(Debug, Clone)]
pub struct BroadcastEvent {
    /// Content being broadcast
    pub content: WorkspaceContent,
    /// Source coalition
    pub coalition: Coalition,
    /// Broadcast time
    pub time: f64,
    /// Broadcast strength
    pub strength: f64,
    /// Ignition achieved
    pub ignited: bool,
}

/// Global Workspace implementation
pub struct GlobalWorkspace {
    /// Configuration
    config: GlobalWorkspaceConfig,
    /// Specialist modules
    pub specialists: Vec<SpecialistModule>,
    /// Current workspace content
    pub workspace: Option<WorkspaceContent>,
    /// Active coalitions
    coalitions: Vec<Coalition>,
    /// Broadcast history
    pub broadcast_history: VecDeque<BroadcastEvent>,
    /// Current time
    time: f64,
    /// Is workspace ignited?
    pub is_ignited: bool,
    /// Statistics
    pub stats: WorkspaceStats,
}

/// Workspace statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkspaceStats {
    /// Total broadcasts
    pub total_broadcasts: u64,
    /// Total ignitions
    pub total_ignitions: u64,
    /// Average coalition size
    pub avg_coalition_size: f64,
    /// Average broadcast duration
    pub avg_broadcast_duration: f64,
    /// Competition wins by specialist type
    pub wins_by_type: HashMap<String, u64>,
}

impl GlobalWorkspace {
    /// Create new global workspace
    pub fn new(config: GlobalWorkspaceConfig) -> Self {
        let specialists = Self::create_default_specialists(config.num_specialists);

        Self {
            config,
            specialists,
            workspace: None,
            coalitions: Vec::new(),
            broadcast_history: VecDeque::with_capacity(100),
            time: 0.0,
            is_ignited: false,
            stats: WorkspaceStats::default(),
        }
    }

    /// Create default specialist modules distributed in hyperbolic space
    fn create_default_specialists(num: usize) -> Vec<SpecialistModule> {
        let types = [
            SpecialistType::Sensory,
            SpecialistType::Motor,
            SpecialistType::Memory,
            SpecialistType::Attention,
            SpecialistType::Language,
            SpecialistType::Spatial,
            SpecialistType::Temporal,
            SpecialistType::Emotional,
        ];

        (0..num).map(|i| {
            // Distribute on a circle in hyperbolic space
            let angle = 2.0 * std::f64::consts::PI * i as f64 / num as f64;
            let radius = 0.5;
            let x = radius * angle.cos();
            let y = radius * angle.sin();
            let t = (1.0 + x * x + y * y).sqrt();

            let module_type = types[i % types.len()];
            SpecialistModule::new(i, module_type, LorentzVec::new(t, x, y, 0.0))
        }).collect()
    }

    /// Process input from chunk processor
    pub fn process_chunk(&mut self, chunk: &TemporalChunk, specialist_id: usize) {
        if specialist_id >= self.specialists.len() {
            return;
        }

        let specialist = &mut self.specialists[specialist_id];

        // Update activation based on chunk activity
        specialist.update_activation(chunk.representation.activity, chunk.end_time);

        // Propose content if above threshold
        if specialist.activation > self.config.access_threshold {
            let features = chunk.representation.temporal_signature.clone();
            let salience = chunk.representation.confidence * chunk.representation.activity;
            specialist.propose_content(features, salience, chunk.end_time);
        }
    }

    /// Main competition and broadcast step
    pub fn step(&mut self, dt: f64) -> Option<BroadcastEvent> {
        self.time += dt;

        // Decay current workspace content
        if self.is_ignited {
            self.decay_workspace(dt);
        }

        // Form coalitions from active specialists
        self.form_coalitions();

        // Run competition
        let winner = self.compete();

        // Broadcast if winner exceeds threshold
        if let Some(coalition) = winner {
            if coalition.strength > self.config.ignition_threshold {
                return Some(self.broadcast(coalition));
            }
        }

        None
    }

    /// Decay workspace content over time
    fn decay_workspace(&mut self, dt: f64) {
        if let Some(ref mut content) = self.workspace {
            content.salience *= (-dt / self.config.broadcast_decay).exp();

            if content.salience < 0.1 {
                self.is_ignited = false;
                self.workspace = None;
            }
        }
    }

    /// Form coalitions from co-active specialists
    fn form_coalitions(&mut self) {
        self.coalitions.clear();

        // Get active specialists
        let active: Vec<usize> = self.specialists.iter()
            .filter(|s| s.activation > self.config.access_threshold)
            .map(|s| s.id)
            .collect();

        if active.is_empty() {
            return;
        }

        // Form coalitions based on connectivity and proximity
        // Simple: each active specialist forms a coalition with connected active neighbors
        for &id in &active {
            let specialist = &self.specialists[id];
            let mut coalition_members = vec![id];

            // Add connected active specialists
            for (&neighbor_id, &strength) in &specialist.connections {
                if active.contains(&neighbor_id) && strength > 0.3 {
                    if coalition_members.len() < self.config.max_coalition_size {
                        coalition_members.push(neighbor_id);
                    }
                }
            }

            // Also add spatially close specialists (hyperbolic proximity)
            for &other_id in &active {
                if other_id != id && !coalition_members.contains(&other_id) {
                    let other = &self.specialists[other_id];
                    let distance = specialist.position.hyperbolic_distance(&other.position);
                    if distance < 1.0 && coalition_members.len() < self.config.max_coalition_size {
                        coalition_members.push(other_id);
                    }
                }
            }

            // Create coalition
            let members: Vec<&SpecialistModule> = coalition_members.iter()
                .map(|&id| &self.specialists[id])
                .collect();

            if !members.is_empty() {
                self.coalitions.push(Coalition::from_specialists(&members, self.time));
            }
        }
    }

    /// Competition via softmax selection
    fn compete(&self) -> Option<Coalition> {
        if self.coalitions.is_empty() {
            return None;
        }

        // Softmax competition
        let max_strength = self.coalitions.iter()
            .map(|c| c.strength)
            .fold(f64::NEG_INFINITY, f64::max);

        let exp_strengths: Vec<f64> = self.coalitions.iter()
            .map(|c| ((c.strength - max_strength) / self.config.competition_temperature).exp())
            .collect();

        let sum_exp: f64 = exp_strengths.iter().sum();

        if sum_exp < 1e-10 {
            return self.coalitions.first().cloned();
        }

        // Winner-take-all selection (deterministic for reproducibility)
        let winner_idx = exp_strengths.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)?;

        Some(self.coalitions[winner_idx].clone())
    }

    /// Broadcast winning coalition to all specialists
    fn broadcast(&mut self, coalition: Coalition) -> BroadcastEvent {
        self.is_ignited = true;
        self.workspace = Some(coalition.content.clone());

        // Broadcast to all specialists
        let content = coalition.content.clone();
        let strength = coalition.strength;

        for specialist in &mut self.specialists {
            if !coalition.members.contains(&specialist.id) {
                specialist.receive_broadcast(&content, strength);
            }
        }

        // Update statistics
        self.stats.total_broadcasts += 1;
        self.stats.total_ignitions += 1;
        self.stats.avg_coalition_size = 0.99 * self.stats.avg_coalition_size
            + 0.01 * coalition.members.len() as f64;

        // Track wins by type
        for &member_id in &coalition.members {
            let type_str = format!("{:?}", self.specialists[member_id].module_type);
            *self.stats.wins_by_type.entry(type_str).or_insert(0) += 1;
        }

        let event = BroadcastEvent {
            content: coalition.content.clone(),
            coalition,
            time: self.time,
            strength,
            ignited: true,
        };

        self.broadcast_history.push_back(event.clone());
        if self.broadcast_history.len() > 100 {
            self.broadcast_history.pop_front();
        }

        event
    }

    /// Get current workspace content
    pub fn current_content(&self) -> Option<&WorkspaceContent> {
        self.workspace.as_ref()
    }

    /// Get specialist by type
    pub fn get_specialist(&self, module_type: SpecialistType) -> Option<&SpecialistModule> {
        self.specialists.iter().find(|s| s.module_type == module_type)
    }

    /// Get mutable specialist by type
    pub fn get_specialist_mut(&mut self, module_type: SpecialistType) -> Option<&mut SpecialistModule> {
        self.specialists.iter_mut().find(|s| s.module_type == module_type)
    }

    /// Update specialist connections based on co-activation
    pub fn update_connections(&mut self, learning_rate: f64) {
        let activations: Vec<f64> = self.specialists.iter().map(|s| s.activation).collect();

        for i in 0..self.specialists.len() {
            for j in (i + 1)..self.specialists.len() {
                // Hebbian-style: strengthen when both active
                let co_activation = activations[i] * activations[j];
                let current = *self.specialists[i].connections.get(&j).unwrap_or(&0.0);
                let new_strength = (current + learning_rate * co_activation).clamp(0.0, 1.0);

                self.specialists[i].connections.insert(j, new_strength);
                self.specialists[j].connections.insert(i, new_strength);
            }
        }
    }

    /// Check if workspace is consciously accessed (ignited)
    pub fn is_conscious(&self) -> bool {
        self.is_ignited
    }

    /// Get broadcast rate
    pub fn broadcast_rate(&self) -> f64 {
        if self.time < 1.0 {
            return 0.0;
        }
        self.stats.total_broadcasts as f64 / self.time * 1000.0 // per second
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_specialist_activation() {
        let mut specialist = SpecialistModule::new(
            0,
            SpecialistType::Sensory,
            LorentzVec::origin(),
        );

        specialist.update_activation(0.8, 0.0);
        assert!(specialist.activation > 0.0);
        assert!(specialist.activation < 1.0);
    }

    #[test]
    fn test_coalition_formation() {
        let specialists = vec![
            SpecialistModule::new(0, SpecialistType::Sensory, LorentzVec::origin()),
            SpecialistModule::new(1, SpecialistType::Motor, LorentzVec::from_spatial(0.1, 0.0, 0.0)),
        ];

        let refs: Vec<&SpecialistModule> = specialists.iter().collect();
        let coalition = Coalition::from_specialists(&refs, 0.0);

        assert_eq!(coalition.members.len(), 2);
    }

    #[test]
    fn test_workspace_broadcast() {
        let config = GlobalWorkspaceConfig::default();
        let mut workspace = GlobalWorkspace::new(config);

        // Activate some specialists
        workspace.specialists[0].activation = 0.8;
        workspace.specialists[1].activation = 0.7;
        workspace.specialists[0].propose_content(vec![0.5], 0.9, 0.0);
        workspace.specialists[1].propose_content(vec![0.6], 0.8, 0.0);

        // Add connections
        workspace.specialists[0].connections.insert(1, 0.5);
        workspace.specialists[1].connections.insert(0, 0.5);

        let event = workspace.step(1.0);

        // Should have formed coalitions
        assert!(!workspace.coalitions.is_empty() || event.is_some());
    }

    #[test]
    fn test_workspace_decay() {
        let config = GlobalWorkspaceConfig::default();
        let mut workspace = GlobalWorkspace::new(config);

        // Force ignition
        workspace.is_ignited = true;
        workspace.workspace = Some(WorkspaceContent {
            salience: 1.0,
            ..Default::default()
        });

        // Step should decay
        workspace.step(100.0);

        // Salience should have decayed
        if let Some(ref content) = workspace.workspace {
            assert!(content.salience < 1.0);
        }
    }

    #[test]
    fn test_frechet_mean() {
        let positions = vec![
            LorentzVec::origin(),
            LorentzVec::from_spatial(0.2, 0.0, 0.0),
            LorentzVec::from_spatial(-0.2, 0.0, 0.0),
        ];

        let mean = Coalition::frechet_mean(positions.iter());

        // Mean should be close to origin
        assert!(mean.x.abs() < 0.01);
    }
}
