//! Neuronal Field Implementation
//! Based on Grinberg's neuronal field theory of consciousness
//! Implements coherent neuronal oscillations and syntergic synthesis

use crate::prelude::*;
use nalgebra::{Matrix3, Vector3, Complex};
use std::f64::consts::PI;
use std::collections::HashMap;

/// Phase coherence threshold for consciousness emergence
const COHERENCE_THRESHOLD: f64 = 0.85;

/// Critical frequency for gamma-band oscillations (40Hz)
const GAMMA_FREQUENCY: f64 = 40.0;

/// Neuronal field representing collective neural dynamics
#[derive(Debug, Clone)]
pub struct NeuronalField {
    /// Field amplitude at each spatial point
    amplitude_field: Vec<Vec<Vec<f64>>>,
    
    /// Phase distribution across the field
    phase_field: Vec<Vec<Vec<f64>>>,
    
    /// Coherence matrix between regions
    coherence_matrix: Matrix3<f64>,
    
    /// Syntergic coupling strength
    syntergic_coupling: f64,
    
    /// Field dimensions
    dimensions: (usize, usize, usize),
    
    /// Time evolution parameter
    time: f64,
}

impl NeuronalField {
    pub fn new(dimensions: (usize, usize, usize)) -> Self {
        let (x, y, z) = dimensions;
        
        Self {
            amplitude_field: vec![vec![vec![0.0; z]; y]; x],
            phase_field: vec![vec![vec![0.0; z]; y]; x],
            coherence_matrix: Matrix3::identity(),
            syntergic_coupling: 1.0,
            dimensions,
            time: 0.0,
        }
    }
    
    /// Initialize field with coherent oscillations
    pub fn initialize_coherent_state(&mut self, frequency: f64) {
        let (x, y, z) = self.dimensions;
        
        for i in 0..x {
            for j in 0..y {
                for k in 0..z {
                    // Gaussian amplitude distribution
                    let r = ((i as f64 - x as f64/2.0).powi(2) +
                            (j as f64 - y as f64/2.0).powi(2) +
                            (k as f64 - z as f64/2.0).powi(2)).sqrt();
                    
                    self.amplitude_field[i][j][k] = (-r.powi(2) / 100.0).exp();
                    self.phase_field[i][j][k] = 2.0 * PI * frequency * self.time;
                }
            }
        }
    }
    
    /// Compute syntergic synthesis between field regions
    pub fn compute_syntergic_synthesis(&self) -> f64 {
        let mut total_synthesis = 0.0;
        let (x, y, z) = self.dimensions;
        
        // Calculate pairwise coherence contributions
        for i in 0..x-1 {
            for j in 0..y-1 {
                for k in 0..z-1 {
                    let phase_diff = (self.phase_field[i][j][k] - 
                                     self.phase_field[i+1][j+1][k+1]).abs();
                    
                    let coherence = (phase_diff % (2.0 * PI)).cos();
                    let amplitude_product = self.amplitude_field[i][j][k] * 
                                          self.amplitude_field[i+1][j+1][k+1];
                    
                    total_synthesis += coherence * amplitude_product * self.syntergic_coupling;
                }
            }
        }
        
        total_synthesis / ((x * y * z) as f64)
    }
    
    /// Evolve field dynamics using Grinberg's field equations
    pub fn evolve(&mut self, dt: f64) {
        let (x, y, z) = self.dimensions;
        let mut new_amplitude = self.amplitude_field.clone();
        let mut new_phase = self.phase_field.clone();
        
        for i in 1..x-1 {
            for j in 1..y-1 {
                for k in 1..z-1 {
                    // Laplacian for diffusion
                    let laplacian_amp = 
                        self.amplitude_field[i+1][j][k] + self.amplitude_field[i-1][j][k] +
                        self.amplitude_field[i][j+1][k] + self.amplitude_field[i][j-1][k] +
                        self.amplitude_field[i][j][k+1] + self.amplitude_field[i][j][k-1] -
                        6.0 * self.amplitude_field[i][j][k];
                    
                    // Phase gradient for wave propagation
                    let phase_gradient = Vector3::new(
                        self.phase_field[i+1][j][k] - self.phase_field[i-1][j][k],
                        self.phase_field[i][j+1][k] - self.phase_field[i][j-1][k],
                        self.phase_field[i][j][k+1] - self.phase_field[i][j][k-1]
                    );
                    
                    // Syntergic coupling term
                    let syntergic_term = self.compute_local_syntergy(i, j, k);
                    
                    // Update amplitude with diffusion and syntergic feedback
                    new_amplitude[i][j][k] += dt * (0.1 * laplacian_amp + syntergic_term);
                    
                    // Update phase with frequency and nonlinear coupling
                    new_phase[i][j][k] += dt * (2.0 * PI * GAMMA_FREQUENCY + 
                                                0.01 * phase_gradient.norm());
                }
            }
        }
        
        self.amplitude_field = new_amplitude;
        self.phase_field = new_phase;
        self.time += dt;
        
        // Update coherence matrix
        self.update_coherence_matrix();
    }
    
    /// Compute local syntergic contribution
    fn compute_local_syntergy(&self, i: usize, j: usize, k: usize) -> f64 {
        let mut syntergy = 0.0;
        
        // Check neighbors for phase coherence
        for di in -1..=1 {
            for dj in -1..=1 {
                for dk in -1..=1 {
                    if di == 0 && dj == 0 && dk == 0 { continue; }
                    
                    let ni = (i as i32 + di) as usize;
                    let nj = (j as i32 + dj) as usize;
                    let nk = (k as i32 + dk) as usize;
                    
                    if ni < self.dimensions.0 && nj < self.dimensions.1 && nk < self.dimensions.2 {
                        let phase_diff = (self.phase_field[i][j][k] - 
                                         self.phase_field[ni][nj][nk]).abs();
                        syntergy += phase_diff.cos() * self.amplitude_field[ni][nj][nk];
                    }
                }
            }
        }
        
        syntergy * self.syntergic_coupling / 26.0 // Normalize by number of neighbors
    }
    
    /// Update global coherence matrix
    fn update_coherence_matrix(&mut self) {
        let regions = self.partition_into_regions();
        
        for i in 0..3 {
            for j in 0..3 {
                self.coherence_matrix[(i, j)] = 
                    self.compute_region_coherence(&regions[i], &regions[j]);
            }
        }
    }
    
    /// Partition field into functional regions
    fn partition_into_regions(&self) -> Vec<Vec<(usize, usize, usize)>> {
        let (x, y, z) = self.dimensions;
        let mut regions = vec![Vec::new(); 3];
        
        for i in 0..x {
            for j in 0..y {
                for k in 0..z {
                    let region_idx = (3 * i / x) % 3;
                    regions[region_idx].push((i, j, k));
                }
            }
        }
        
        regions
    }
    
    /// Compute coherence between two regions
    fn compute_region_coherence(&self, region1: &[(usize, usize, usize)], 
                               region2: &[(usize, usize, usize)]) -> f64 {
        let mut coherence_sum = 0.0;
        let mut count = 0;
        
        for &(i1, j1, k1) in region1 {
            for &(i2, j2, k2) in region2 {
                let phase_diff = (self.phase_field[i1][j1][k1] - 
                                 self.phase_field[i2][j2][k2]).abs();
                coherence_sum += phase_diff.cos();
                count += 1;
            }
        }
        
        if count > 0 {
            coherence_sum / count as f64
        } else {
            0.0
        }
    }
    
    /// Check if field has achieved consciousness threshold
    pub fn is_conscious(&self) -> bool {
        let avg_coherence = self.coherence_matrix.trace() / 3.0;
        avg_coherence >= COHERENCE_THRESHOLD
    }
    
    /// Get consciousness quality metric
    pub fn consciousness_quality(&self) -> f64 {
        let coherence = self.coherence_matrix.trace() / 3.0;
        let syntergy = self.compute_syntergic_synthesis();
        let complexity = self.compute_complexity();
        
        // Weighted combination of factors
        0.4 * coherence + 0.4 * syntergy + 0.2 * complexity
    }
    
    /// Compute field complexity using entropy-like measure
    fn compute_complexity(&self) -> f64 {
        let mut entropy = 0.0;
        let (x, y, z) = self.dimensions;
        let total_points = (x * y * z) as f64;
        
        // Compute amplitude distribution entropy
        let mut amplitude_bins = HashMap::new();
        
        for i in 0..x {
            for j in 0..y {
                for k in 0..z {
                    let bin = (self.amplitude_field[i][j][k] * 10.0) as i32;
                    *amplitude_bins.entry(bin).or_insert(0) += 1;
                }
            }
        }
        
        for count in amplitude_bins.values() {
            let p = *count as f64 / total_points;
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }
        
        entropy / (total_points.ln()) // Normalize
    }
    
    /// Apply external stimulus to field
    pub fn apply_stimulus(&mut self, position: (usize, usize, usize), 
                         strength: f64, frequency: f64) {
        let (x, y, z) = position;
        if x < self.dimensions.0 && y < self.dimensions.1 && z < self.dimensions.2 {
            self.amplitude_field[x][y][z] += strength;
            self.phase_field[x][y][z] = 2.0 * PI * frequency * self.time;
        }
    }
    
    /// Get field state for visualization
    pub fn get_field_state(&self) -> FieldState {
        FieldState {
            amplitude: self.amplitude_field.clone(),
            phase: self.phase_field.clone(),
            coherence: self.coherence_matrix.clone(),
            consciousness_level: self.consciousness_quality(),
            is_conscious: self.is_conscious(),
        }
    }
}

/// Field state snapshot for analysis
#[derive(Debug, Clone)]
pub struct FieldState {
    pub amplitude: Vec<Vec<Vec<f64>>>,
    pub phase: Vec<Vec<Vec<f64>>>,
    pub coherence: Matrix3<f64>,
    pub consciousness_level: f64,
    pub is_conscious: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_neuronal_field_creation() {
        let field = NeuronalField::new((10, 10, 10));
        assert_eq!(field.dimensions, (10, 10, 10));
        assert!(!field.is_conscious());
    }
    
    #[test]
    fn test_coherent_state_initialization() {
        let mut field = NeuronalField::new((5, 5, 5));
        field.initialize_coherent_state(GAMMA_FREQUENCY);
        
        // Check central amplitude is highest
        assert!(field.amplitude_field[2][2][2] > field.amplitude_field[0][0][0]);
    }
    
    #[test]
    fn test_syntergic_synthesis() {
        let mut field = NeuronalField::new((5, 5, 5));
        field.initialize_coherent_state(GAMMA_FREQUENCY);
        
        let synthesis = field.compute_syntergic_synthesis();
        assert!(synthesis > 0.0);
    }
    
    #[test]
    fn test_field_evolution() {
        let mut field = NeuronalField::new((5, 5, 5));
        field.initialize_coherent_state(GAMMA_FREQUENCY);
        
        let initial_quality = field.consciousness_quality();
        field.evolve(0.01);
        let evolved_quality = field.consciousness_quality();
        
        // Quality should change after evolution
        assert!((evolved_quality - initial_quality).abs() > 0.0);
    }
}