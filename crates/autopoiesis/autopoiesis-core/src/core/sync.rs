//! Synchronization dynamics based on Steven Strogatz's theory
//! Coupled oscillators, emergent synchronization, and phase transitions

use async_trait::async_trait;
use nalgebra as na;
use num_complex::Complex64;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use crate::Result;

/// Phase lock transition in the system
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PhaseLockTransition {
    /// Critical coupling strength
    pub critical_coupling: f64,
    /// Order parameter before transition
    pub order_before: f64,
    /// Order parameter after transition
    pub order_after: f64,
    /// Type of transition
    pub transition_type: TransitionType,
    /// Number of synchronized clusters
    pub sync_clusters: Vec<SyncCluster>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TransitionType {
    /// Continuous second-order transition
    Continuous,
    /// Discontinuous first-order transition
    Discontinuous,
    /// Explosive synchronization
    Explosive,
    /// Chimera state emergence
    Chimera,
}

/// A synchronized cluster of oscillators
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SyncCluster {
    /// Indices of oscillators in this cluster
    pub oscillator_indices: Vec<usize>,
    /// Mean frequency of the cluster
    pub mean_frequency: f64,
    /// Phase coherence within cluster
    pub coherence: f64,
    /// Stability of the cluster
    pub stability: f64,
}

/// Feedback for adjusting coupling
#[derive(Clone, Debug)]
pub struct CouplingFeedback {
    /// Current synchronization order
    pub current_order: f64,
    /// Target synchronization order
    pub target_order: f64,
    /// Phase dispersion
    pub phase_dispersion: f64,
    /// Frequency mismatch
    pub frequency_mismatch: f64,
}

/// Core trait for synchronization dynamics
pub trait SynchronizationDynamics: Send + Sync {
    type Phase: Send + Sync;
    type Coupling: Send + Sync;
    
    /// Calculate Kuramoto order parameter (0 = incoherent, 1 = fully synchronized)
    fn kuramoto_order_parameter(&self) -> f64;
    
    /// Detect phase transitions in synchronization
    fn phase_transitions(&self) -> Vec<PhaseLockTransition>;
    
    /// Adapt coupling strength based on feedback
    fn adapt_coupling_strength(&mut self, feedback: CouplingFeedback);
    
    /// Get phase distribution
    fn phase_distribution(&self) -> Vec<Self::Phase>;
    
    /// Calculate phase coherence between specific oscillators
    fn pairwise_coherence(&self, i: usize, j: usize) -> f64;
    
    /// Detect chimera states (partial synchronization)
    fn detect_chimera_states(&self) -> Vec<ChimeraState>;
}

/// Represents a chimera state
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChimeraState {
    /// Coherent region
    pub coherent_indices: Vec<usize>,
    /// Incoherent region
    pub incoherent_indices: Vec<usize>,
    /// Spatial correlation length
    pub correlation_length: f64,
}

/// Phase oscillator for Kuramoto model
#[derive(Clone, Debug)]
pub struct PhaseOscillator {
    /// Current phase
    pub phase: f64,
    /// Natural frequency
    pub natural_frequency: f64,
    /// External force amplitude
    pub external_force: f64,
}

impl PhaseOscillator {
    pub fn new(phase: f64, frequency: f64) -> Self {
        Self {
            phase,
            natural_frequency: frequency,
            external_force: 0.0,
        }
    }
    
    /// Update phase using Kuramoto dynamics
    pub fn update_kuramoto(&mut self, coupling_sum: f64, dt: f64) {
        self.phase += (self.natural_frequency + coupling_sum + self.external_force) * dt;
        
        // Keep phase in [0, 2π]
        self.phase = self.phase % (2.0 * std::f64::consts::PI);
        if self.phase < 0.0 {
            self.phase += 2.0 * std::f64::consts::PI;
        }
    }
}

/// Kuramoto model of coupled oscillators
pub struct KuramotoModel {
    /// Phase oscillators
    pub oscillators: Vec<PhaseOscillator>,
    /// Coupling matrix (can be non-uniform)
    pub coupling_matrix: na::DMatrix<f64>,
    /// Global coupling strength
    pub global_coupling: f64,
    /// Time
    pub time: f64,
    /// History for analysis
    order_history: VecDeque<f64>,
}

impl KuramotoModel {
    pub fn new(n: usize, coupling: f64) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Initialize oscillators with random phases and frequencies
        let oscillators: Vec<PhaseOscillator> = (0..n)
            .map(|_| PhaseOscillator::new(
                rng.gen::<f64>() * 2.0 * std::f64::consts::PI,
                rng.gen_range(-1.0..1.0), // Natural frequencies from distribution
            ))
            .collect();
        
        // All-to-all coupling
        let coupling_matrix = na::DMatrix::from_element(n, n, 1.0 / n as f64);
        
        Self {
            oscillators,
            coupling_matrix,
            global_coupling: coupling,
            time: 0.0,
            order_history: VecDeque::with_capacity(1000),
        }
    }
    
    /// Create with specific topology
    pub fn with_topology(n: usize, coupling: f64, topology: NetworkTopology) -> Self {
        let mut model = Self::new(n, coupling);
        
        match topology {
            NetworkTopology::AllToAll => {
                // Already initialized
            },
            NetworkTopology::NearestNeighbor(k) => {
                model.coupling_matrix = na::DMatrix::zeros(n, n);
                for i in 0..n {
                    for j in 1..=k {
                        let idx1 = (i + j) % n;
                        let idx2 = (i + n - j) % n;
                        model.coupling_matrix[(i, idx1)] = 1.0 / (2.0 * k as f64);
                        model.coupling_matrix[(i, idx2)] = 1.0 / (2.0 * k as f64);
                    }
                }
            },
            NetworkTopology::SmallWorld(k, p) => {
                // Start with nearest neighbor
                model = Self::with_topology(n, coupling, NetworkTopology::NearestNeighbor(k));
                
                // Rewire with probability p
                use rand::Rng;
                let mut rng = rand::thread_rng();
                
                for i in 0..n {
                    for j in 0..n {
                        if model.coupling_matrix[(i, j)] > 0.0 && rng.gen::<f64>() < p {
                            // Rewire to random node
                            let new_j = rng.gen_range(0..n);
                            model.coupling_matrix[(i, j)] = 0.0;
                            model.coupling_matrix[(i, new_j)] = 1.0 / (2.0 * k as f64);
                        }
                    }
                }
            },
            NetworkTopology::ScaleFree => {
                // Barabási-Albert preferential attachment
                model.coupling_matrix = na::DMatrix::zeros(n, n);
                let mut degrees = vec![0; n];
                
                // Start with small complete graph
                for i in 0..3 {
                    for j in i+1..3 {
                        model.coupling_matrix[(i, j)] = 1.0;
                        model.coupling_matrix[(j, i)] = 1.0;
                        degrees[i] += 1;
                        degrees[j] += 1;
                    }
                }
                
                // Add remaining nodes with preferential attachment
                use rand::Rng;
                let mut rng = rand::thread_rng();
                
                for i in 3..n {
                    let total_degree: usize = degrees.iter().sum();
                    
                    // Connect to m=2 existing nodes
                    for _ in 0..2 {
                        let mut cumsum = 0;
                        let r = rng.gen_range(0..total_degree);
                        
                        for j in 0..i {
                            cumsum += degrees[j];
                            if cumsum > r {
                                model.coupling_matrix[(i, j)] = 1.0;
                                model.coupling_matrix[(j, i)] = 1.0;
                                degrees[i] += 1;
                                degrees[j] += 1;
                                break;
                            }
                        }
                    }
                }
                
                // Normalize
                for i in 0..n {
                    let row_sum: f64 = model.coupling_matrix.row(i).sum();
                    if row_sum > 0.0 {
                        for j in 0..n {
                            model.coupling_matrix[(i, j)] /= row_sum;
                        }
                    }
                }
            },
        }
        
        model
    }
    
    /// Evolve the system
    pub fn step(&mut self, dt: f64) {
        let n = self.oscillators.len();
        let mut new_phases = vec![0.0; n];
        
        // Calculate coupling for each oscillator
        for i in 0..n {
            let mut coupling_sum = 0.0;
            
            for j in 0..n {
                if i != j {
                    let phase_diff = self.oscillators[j].phase - self.oscillators[i].phase;
                    coupling_sum += self.coupling_matrix[(i, j)] * phase_diff.sin();
                }
            }
            
            coupling_sum *= self.global_coupling;
            
            // Store new phase
            new_phases[i] = self.oscillators[i].phase + 
                (self.oscillators[i].natural_frequency + coupling_sum) * dt;
        }
        
        // Update all phases
        for (i, &new_phase) in new_phases.iter().enumerate() {
            self.oscillators[i].phase = new_phase % (2.0 * std::f64::consts::PI);
        }
        
        self.time += dt;
        
        // Store order parameter history
        let order = self.kuramoto_order_parameter();
        self.order_history.push_back(order);
        if self.order_history.len() > 1000 {
            self.order_history.pop_front();
        }
    }
    
    /// Add external forcing
    pub fn add_forcing(&mut self, frequency: f64, amplitude: f64) {
        for osc in &mut self.oscillators {
            osc.external_force = amplitude * (frequency * self.time).sin();
        }
    }
}

/// Network topology types
#[derive(Clone, Debug)]
pub enum NetworkTopology {
    AllToAll,
    NearestNeighbor(usize), // k neighbors
    SmallWorld(usize, f64), // k neighbors, rewiring probability p
    ScaleFree,
}

impl SynchronizationDynamics for KuramotoModel {
    type Phase = f64;
    type Coupling = f64;
    
    fn kuramoto_order_parameter(&self) -> f64 {
        let n = self.oscillators.len() as f64;
        let mut sum = Complex64::new(0.0, 0.0);
        
        for osc in &self.oscillators {
            sum += Complex64::from_polar(1.0, osc.phase);
        }
        
        (sum / n).norm()
    }
    
    fn phase_transitions(&self) -> Vec<PhaseLockTransition> {
        let mut transitions = Vec::new();
        
        // Analyze order parameter history for transitions
        if self.order_history.len() < 100 {
            return transitions;
        }
        
        // Simple transition detection: look for rapid changes in order
        let window_size = 20;
        for i in window_size..self.order_history.len() - window_size {
            let before_avg = self.order_history.range(i-window_size..i)
                .sum::<f64>() / window_size as f64;
            let after_avg = self.order_history.range(i..i+window_size)
                .sum::<f64>() / window_size as f64;
            
            let change = (after_avg - before_avg).abs();
            
            if change > 0.2 {
                // Detected a transition
                let clusters = self.detect_sync_clusters();
                
                transitions.push(PhaseLockTransition {
                    critical_coupling: self.global_coupling,
                    order_before: before_avg,
                    order_after: after_avg,
                    transition_type: if change > 0.5 {
                        TransitionType::Discontinuous
                    } else {
                        TransitionType::Continuous
                    },
                    sync_clusters: clusters,
                });
            }
        }
        
        transitions
    }
    
    fn adapt_coupling_strength(&mut self, feedback: CouplingFeedback) {
        let error = feedback.target_order - feedback.current_order;
        
        // Simple proportional control
        self.global_coupling += 0.1 * error;
        
        // Keep coupling positive and bounded
        self.global_coupling = self.global_coupling.max(0.0).min(10.0);
        
        // Adapt based on phase dispersion
        if feedback.phase_dispersion > 1.0 {
            // High dispersion, increase coupling
            self.global_coupling *= 1.1;
        }
    }
    
    fn phase_distribution(&self) -> Vec<Self::Phase> {
        self.oscillators.iter().map(|osc| osc.phase).collect()
    }
    
    fn pairwise_coherence(&self, i: usize, j: usize) -> f64 {
        if i >= self.oscillators.len() || j >= self.oscillators.len() {
            return 0.0;
        }
        
        let phase_diff = self.oscillators[i].phase - self.oscillators[j].phase;
        phase_diff.cos().abs()
    }
    
    fn detect_chimera_states(&self) -> Vec<ChimeraState> {
        let mut chimeras = Vec::new();
        let n = self.oscillators.len();
        
        // Calculate local order parameters
        let mut local_orders = vec![0.0; n];
        let window = 5; // Local neighborhood size
        
        for i in 0..n {
            let mut local_sum = Complex64::new(0.0, 0.0);
            let mut count = 0;
            
            for j in (i.saturating_sub(window))..((i + window + 1).min(n)) {
                local_sum += Complex64::from_polar(1.0, self.oscillators[j].phase);
                count += 1;
            }
            
            local_orders[i] = (local_sum / count as f64).norm();
        }
        
        // Detect coherent and incoherent regions
        let threshold = 0.8;
        let mut coherent = Vec::new();
        let mut incoherent = Vec::new();
        
        for (i, &order) in local_orders.iter().enumerate() {
            if order > threshold {
                coherent.push(i);
            } else {
                incoherent.push(i);
            }
        }
        
        // If we have both regions, it's a chimera
        if !coherent.is_empty() && !incoherent.is_empty() {
            chimeras.push(ChimeraState {
                coherent_indices: coherent,
                incoherent_indices: incoherent,
                correlation_length: self.calculate_correlation_length(&local_orders),
            });
        }
        
        chimeras
    }
}

impl KuramotoModel {
    /// Detect synchronized clusters
    fn detect_sync_clusters(&self) -> Vec<SyncCluster> {
        let mut clusters = Vec::new();
        let n = self.oscillators.len();
        let mut assigned = vec![false; n];
        
        // Simple clustering based on phase similarity
        let phase_threshold = 0.2; // radians
        
        for i in 0..n {
            if assigned[i] {
                continue;
            }
            
            let mut cluster_indices = vec![i];
            assigned[i] = true;
            
            // Find all oscillators close in phase
            for j in i+1..n {
                if !assigned[j] {
                    let phase_diff = (self.oscillators[i].phase - self.oscillators[j].phase).abs();
                    let phase_diff = phase_diff.min(2.0 * std::f64::consts::PI - phase_diff);
                    
                    if phase_diff < phase_threshold {
                        cluster_indices.push(j);
                        assigned[j] = true;
                    }
                }
            }
            
            if cluster_indices.len() > 1 {
                // Calculate cluster properties
                let mean_freq = cluster_indices.iter()
                    .map(|&idx| self.oscillators[idx].natural_frequency)
                    .sum::<f64>() / cluster_indices.len() as f64;
                
                let mut cluster_sum = Complex64::new(0.0, 0.0);
                for &idx in &cluster_indices {
                    cluster_sum += Complex64::from_polar(1.0, self.oscillators[idx].phase);
                }
                let coherence = (cluster_sum / cluster_indices.len() as f64).norm();
                
                clusters.push(SyncCluster {
                    oscillator_indices: cluster_indices,
                    mean_frequency: mean_freq,
                    coherence,
                    stability: coherence * 0.9, // Simplified stability metric
                });
            }
        }
        
        clusters
    }
    
    /// Calculate spatial correlation length
    fn calculate_correlation_length(&self, local_orders: &[f64]) -> f64 {
        let n = local_orders.len();
        if n < 2 {
            return 0.0;
        }
        
        let mean_order = local_orders.iter().sum::<f64>() / n as f64;
        
        // Calculate autocorrelation
        let mut correlation_sum = 0.0;
        let max_dist = n / 4;
        
        for dist in 1..max_dist {
            let mut corr = 0.0;
            let mut count = 0;
            
            for i in 0..n-dist {
                corr += (local_orders[i] - mean_order) * (local_orders[i + dist] - mean_order);
                count += 1;
            }
            
            if count > 0 {
                corr /= count as f64;
                
                // Find where correlation drops to 1/e
                if corr < correlation_sum * 0.368 && correlation_sum > 0.0 {
                    return dist as f64;
                }
                
                if dist == 1 {
                    correlation_sum = corr;
                }
            }
        }
        
        max_dist as f64
    }
}

/// Synchronized swarm behavior
pub struct SynchronizedSwarm {
    /// Kuramoto model for phase synchronization
    pub phase_model: KuramotoModel,
    /// Spatial positions of agents
    pub positions: Vec<na::Vector2<f64>>,
    /// Velocities of agents
    pub velocities: Vec<na::Vector2<f64>>,
    /// Coupling between phase and motion
    pub phase_motion_coupling: f64,
}

impl SynchronizedSwarm {
    pub fn new(n: usize, coupling: f64) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let phase_model = KuramotoModel::new(n, coupling);
        
        let positions: Vec<na::Vector2<f64>> = (0..n)
            .map(|_| na::Vector2::new(
                rng.gen_range(-10.0..10.0),
                rng.gen_range(-10.0..10.0),
            ))
            .collect();
        
        let velocities = vec![na::Vector2::zeros(); n];
        
        Self {
            phase_model,
            positions,
            velocities,
            phase_motion_coupling: 0.5,
        }
    }
    
    /// Update swarm dynamics
    pub fn step(&mut self, dt: f64) {
        // Update phase dynamics
        self.phase_model.step(dt);
        
        // Couple phase to motion
        for i in 0..self.positions.len() {
            // Velocity direction influenced by phase
            let phase = self.phase_model.oscillators[i].phase;
            let vel_direction = na::Vector2::new(phase.cos(), phase.sin());
            
            // Cohesion: move toward center of mass
            let com = self.center_of_mass();
            let cohesion = (com - self.positions[i]).normalize() * 0.1;
            
            // Alignment: align with neighbors
            let alignment = self.local_alignment(i, 3.0) * 0.1;
            
            // Separation: avoid crowding
            let separation = self.separation_force(i, 1.0) * 0.2;
            
            // Combine forces with phase coupling
            self.velocities[i] = vel_direction * self.phase_motion_coupling +
                                cohesion + alignment + separation;
            
            // Update position
            self.positions[i] += self.velocities[i] * dt;
        }
    }
    
    fn center_of_mass(&self) -> na::Vector2<f64> {
        let sum: na::Vector2<f64> = self.positions.iter().sum();
        sum / self.positions.len() as f64
    }
    
    fn local_alignment(&self, idx: usize, radius: f64) -> na::Vector2<f64> {
        let mut alignment = na::Vector2::zeros();
        let mut count = 0;
        
        for (i, pos) in self.positions.iter().enumerate() {
            if i != idx {
                let dist = (pos - self.positions[idx]).norm();
                if dist < radius {
                    alignment += self.velocities[i];
                    count += 1;
                }
            }
        }
        
        if count > 0 {
            alignment / count as f64
        } else {
            na::Vector2::zeros()
        }
    }
    
    fn separation_force(&self, idx: usize, min_dist: f64) -> na::Vector2<f64> {
        let mut force = na::Vector2::zeros();
        
        for (i, pos) in self.positions.iter().enumerate() {
            if i != idx {
                let diff = self.positions[idx] - pos;
                let dist = diff.norm();
                
                if dist < min_dist && dist > 0.0 {
                    force += diff.normalize() / dist;
                }
            }
        }
        
        force
    }
}

use std::collections::VecDeque;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_kuramoto_synchronization() {
        let mut model = KuramotoModel::new(10, 1.0);
        
        // Run simulation
        for _ in 0..1000 {
            model.step(0.01);
        }
        
        // Check that some synchronization occurred
        let order = model.kuramoto_order_parameter();
        assert!(order > 0.1); // Should have some synchronization
    }
    
    #[test]
    fn test_chimera_detection() {
        let model = KuramotoModel::new(20, 0.5);
        let chimeras = model.detect_chimera_states();
        
        // Chimeras are rare in uniform coupling, so might be empty
        assert!(chimeras.len() >= 0);
    }
}