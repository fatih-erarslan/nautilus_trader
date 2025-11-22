//! Dissipative structures based on Ilya Prigogine's theory
//! Order through fluctuations in far-from-equilibrium systems

use async_trait::async_trait;
use nalgebra as na;
use std::collections::VecDeque;
use serde::{Deserialize, Serialize};

use crate::Result;

/// Represents a bifurcation point in the system
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BifurcationPoint {
    /// Control parameter value at bifurcation
    pub parameter_value: f64,
    /// Type of bifurcation
    pub bifurcation_type: BifurcationType,
    /// Critical fluctuation amplitude
    pub critical_amplitude: f64,
    /// Available branches after bifurcation
    pub branches: Vec<BranchInfo>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BifurcationType {
    PitchforkSupercritical,
    PitchforkSubcritical,
    Hopf,
    SaddleNode,
    Transcritical,
    PeriodDoubling,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BranchInfo {
    pub stability: Stability,
    pub attractor_type: AttractorType,
    pub basin_volume: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Stability {
    Stable,
    Unstable,
    Metastable,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AttractorType {
    FixedPoint,
    LimitCycle,
    StrangeAttractor,
    Torus,
}

/// Feedback mechanism for coupling adjustments
#[derive(Clone, Debug)]
pub struct CouplingFeedback {
    pub current_entropy_production: f64,
    pub target_entropy_production: f64,
    pub fluctuation_intensity: f64,
    pub distance_from_equilibrium: f64,
}

/// Core trait for dissipative structures
pub trait DissipativeStructure: Send + Sync {
    type Energy: Send + Sync;
    type Entropy: Send + Sync;
    
    /// Calculate current entropy production rate
    fn entropy_production(&self) -> Self::Entropy;
    
    /// Identify bifurcation points in parameter space
    fn bifurcation_points(&self) -> Vec<BifurcationPoint>;
    
    /// Maintain system far from equilibrium through energy pumping
    fn maintain_far_from_equilibrium(&mut self, energy_flow: Self::Energy);
    
    /// Calculate Lyapunov exponents to characterize dynamics
    fn lyapunov_exponents(&self) -> Vec<f64> {
        // Default implementation for systems that don't override
        vec![0.0]
    }
    
    /// Detect current distance from equilibrium
    fn distance_from_equilibrium(&self) -> f64;
    
    /// Apply fluctuations to explore phase space
    fn apply_fluctuations(&mut self, intensity: f64);
}

/// Brusselator - Classic dissipative structure model
pub struct Brusselator {
    /// Concentration of species X
    pub x: f64,
    /// Concentration of species Y  
    pub y: f64,
    /// Control parameter A
    pub a: f64,
    /// Control parameter B
    pub b: f64,
    /// Spatial dimension (if applicable)
    pub spatial_points: Option<Vec<na::Vector2<f64>>>,
    /// History for Lyapunov calculation
    trajectory_history: VecDeque<(f64, f64)>,
    /// Time step
    dt: f64,
}

impl Brusselator {
    pub fn new(a: f64, b: f64) -> Self {
        Self {
            x: 1.0,
            y: 1.0,
            a,
            b,
            spatial_points: None,
            trajectory_history: VecDeque::with_capacity(1000),
            dt: 0.01,
        }
    }
    
    /// Time evolution using the Brusselator equations
    pub fn evolve(&mut self, dt: f64) {
        let dx = self.a - (self.b + 1.0) * self.x + self.x * self.x * self.y;
        let dy = self.b * self.x - self.x * self.x * self.y;
        
        self.x += dx * dt;
        self.y += dy * dt;
        
        // Store history
        self.trajectory_history.push_back((self.x, self.y));
        if self.trajectory_history.len() > 1000 {
            self.trajectory_history.pop_front();
        }
    }
    
    /// Calculate the Jacobian matrix at current state
    pub fn jacobian(&self) -> na::Matrix2<f64> {
        na::Matrix2::new(
            -(self.b + 1.0) + 2.0 * self.x * self.y, self.x * self.x,
            self.b - 2.0 * self.x * self.y, -self.x * self.x
        )
    }
}

impl DissipativeStructure for Brusselator {
    type Energy = f64;
    type Entropy = f64;
    
    fn entropy_production(&self) -> Self::Entropy {
        // Entropy production rate for Brusselator
        // Based on reaction fluxes and thermodynamic forces
        let flux_forward = self.a + self.x * self.x * self.y;
        let flux_backward = (self.b + 1.0) * self.x;
        
        // Simplified entropy production (would need chemical potentials for exact)
        (flux_forward - flux_backward).abs() * 0.1
    }
    
    fn bifurcation_points(&self) -> Vec<BifurcationPoint> {
        // Hopf bifurcation occurs at b = 1 + a²
        let critical_b = 1.0 + self.a * self.a;
        
        vec![BifurcationPoint {
            parameter_value: critical_b,
            bifurcation_type: BifurcationType::Hopf,
            critical_amplitude: 0.01,
            branches: vec![
                BranchInfo {
                    stability: if self.b < critical_b { Stability::Stable } else { Stability::Unstable },
                    attractor_type: AttractorType::FixedPoint,
                    basin_volume: 0.5,
                },
                BranchInfo {
                    stability: if self.b >= critical_b { Stability::Stable } else { Stability::Unstable },
                    attractor_type: AttractorType::LimitCycle,
                    basin_volume: 0.5,
                },
            ],
        }]
    }
    
    fn maintain_far_from_equilibrium(&mut self, energy_flow: Self::Energy) {
        // Adjust control parameters to maintain non-equilibrium
        self.a = (self.a + energy_flow * 0.01).max(0.0);
        
        // Ensure we stay in interesting parameter regime
        if self.b < 1.0 + self.a * self.a - 0.5 {
            self.b += 0.1;
        }
    }
    
    fn distance_from_equilibrium(&self) -> f64 {
        // Distance from fixed point (a, b/a)
        let x_eq = self.a;
        let y_eq = self.b / self.a;
        
        ((self.x - x_eq).powi(2) + (self.y - y_eq).powi(2)).sqrt()
    }
    
    fn apply_fluctuations(&mut self, intensity: f64) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        self.x += rng.gen_range(-intensity..intensity);
        self.y += rng.gen_range(-intensity..intensity);
        
        // Keep concentrations positive
        self.x = self.x.max(0.001);
        self.y = self.y.max(0.001);
    }
    
    fn lyapunov_exponents(&self) -> Vec<f64> {
        // Calculate largest Lyapunov exponent from trajectory
        if self.trajectory_history.len() < 100 {
            return vec![0.0];
        }
        
        let mut sum = 0.0;
        let mut count = 0;
        
        // Use pairs of nearby points to estimate divergence
        for i in 0..self.trajectory_history.len() - 10 {
            if let (Some(p1), Some(p2)) = (
                self.trajectory_history.get(i),
                self.trajectory_history.get(i + 1)
            ) {
                let initial_sep = ((p2.0 - p1.0).powi(2) + (p2.1 - p1.1).powi(2)).sqrt();
                
                if let (Some(p1_later), Some(p2_later)) = (
                    self.trajectory_history.get(i + 10),
                    self.trajectory_history.get(i + 11)
                ) {
                    let final_sep = ((p2_later.0 - p1_later.0).powi(2) + 
                                   (p2_later.1 - p1_later.1).powi(2)).sqrt();
                    
                    if initial_sep > 1e-10 {
                        sum += (final_sep / initial_sep).ln();
                        count += 1;
                    }
                }
            }
        }
        
        if count > 0 {
            vec![sum / (count as f64 * 10.0 * self.dt)]
        } else {
            vec![0.0]
        }
    }
}

/// Bénard convection - thermal dissipative structure
pub struct BenardConvection {
    /// Temperature field
    pub temperature_field: na::DMatrix<f64>,
    /// Velocity field (2D)
    pub velocity_field: Vec<na::Vector2<f64>>,
    /// Rayleigh number (control parameter)
    pub rayleigh_number: f64,
    /// Prandtl number
    pub prandtl_number: f64,
    /// Grid dimensions
    pub nx: usize,
    pub ny: usize,
}

impl BenardConvection {
    pub fn new(nx: usize, ny: usize, rayleigh: f64) -> Self {
        let temperature_field = na::DMatrix::from_fn(ny, nx, |i, _| {
            // Linear temperature profile initially
            1.0 - (i as f64) / (ny as f64)
        });
        
        let velocity_field = vec![na::Vector2::zeros(); nx * ny];
        
        Self {
            temperature_field,
            velocity_field,
            rayleigh_number: rayleigh,
            prandtl_number: 1.0,
            nx,
            ny,
        }
    }
    
    /// Calculate Nusselt number (heat transport efficiency)
    pub fn nusselt_number(&self) -> f64 {
        // Heat flux at boundaries
        let mut heat_flux = 0.0;
        for j in 0..self.nx {
            let dt_dy = (self.temperature_field[(1, j)] - self.temperature_field[(0, j)]) * self.ny as f64;
            heat_flux += dt_dy;
        }
        heat_flux / self.nx as f64
    }
}

impl DissipativeStructure for BenardConvection {
    type Energy = f64;
    type Entropy = f64;
    
    fn entropy_production(&self) -> Self::Entropy {
        // Entropy production from heat flow and viscous dissipation
        let mut entropy_prod = 0.0;
        
        // Temperature gradient contribution
        for i in 1..self.ny-1 {
            for j in 1..self.nx-1 {
                let grad_t_x = (self.temperature_field[(i, j+1)] - self.temperature_field[(i, j-1)]) / 2.0;
                let grad_t_y = (self.temperature_field[(i+1, j)] - self.temperature_field[(i-1, j)]) / 2.0;
                
                entropy_prod += (grad_t_x.powi(2) + grad_t_y.powi(2)) / self.temperature_field[(i, j)].powi(2);
            }
        }
        
        entropy_prod / (self.nx * self.ny) as f64
    }
    
    fn bifurcation_points(&self) -> Vec<BifurcationPoint> {
        // Critical Rayleigh number for onset of convection
        let critical_ra = 1707.76; // For rigid boundaries
        
        vec![BifurcationPoint {
            parameter_value: critical_ra,
            bifurcation_type: BifurcationType::PitchforkSupercritical,
            critical_amplitude: 0.01,
            branches: vec![
                BranchInfo {
                    stability: if self.rayleigh_number < critical_ra { 
                        Stability::Stable 
                    } else { 
                        Stability::Unstable 
                    },
                    attractor_type: AttractorType::FixedPoint,
                    basin_volume: 0.5,
                },
                BranchInfo {
                    stability: if self.rayleigh_number >= critical_ra { 
                        Stability::Stable 
                    } else { 
                        Stability::Unstable 
                    },
                    attractor_type: AttractorType::LimitCycle,
                    basin_volume: 0.5,
                },
            ],
        }]
    }
    
    fn maintain_far_from_equilibrium(&mut self, energy_flow: Self::Energy) {
        // Maintain temperature difference
        for j in 0..self.nx {
            self.temperature_field[(0, j)] = 1.0 + energy_flow * 0.1;
            self.temperature_field[(self.ny-1, j)] = 0.0;
        }
    }
    
    fn distance_from_equilibrium(&self) -> f64 {
        // Average temperature gradient as measure
        let mut total_gradient = 0.0;
        for i in 1..self.ny {
            for j in 0..self.nx {
                let dt = (self.temperature_field[(i, j)] - self.temperature_field[(i-1, j)]).abs();
                total_gradient += dt;
            }
        }
        total_gradient / ((self.nx * self.ny) as f64)
    }
    
    fn apply_fluctuations(&mut self, intensity: f64) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Add random perturbations to temperature field
        for i in 1..self.ny-1 {
            for j in 1..self.nx-1 {
                self.temperature_field[(i, j)] += rng.gen_range(-intensity..intensity);
            }
        }
    }
}

/// Reaction-diffusion system for pattern formation
pub struct ReactionDiffusionSystem {
    /// Concentration fields for multiple species
    pub concentrations: Vec<na::DMatrix<f64>>,
    /// Diffusion coefficients
    pub diffusion_coeffs: Vec<f64>,
    /// Reaction parameters
    pub reaction_params: ReactionParameters,
    /// Grid spacing
    pub dx: f64,
}

#[derive(Clone, Debug)]
pub struct ReactionParameters {
    pub feed_rate: f64,
    pub kill_rate: f64,
    pub reaction_strength: f64,
}

impl DissipativeStructure for ReactionDiffusionSystem {
    type Energy = Vec<f64>;
    type Entropy = f64;
    
    fn entropy_production(&self) -> Self::Entropy {
        // Entropy from diffusion and reactions
        let mut entropy = 0.0;
        
        for conc_field in &self.concentrations {
            for i in 1..conc_field.nrows()-1 {
                for j in 1..conc_field.ncols()-1 {
                    // Diffusion entropy
                    let laplacian = (conc_field[(i+1, j)] + conc_field[(i-1, j)] +
                                   conc_field[(i, j+1)] + conc_field[(i, j-1)] -
                                   4.0 * conc_field[(i, j)]) / (self.dx * self.dx);
                    
                    entropy += laplacian.powi(2);
                }
            }
        }
        
        entropy / self.concentrations[0].len() as f64
    }
    
    fn bifurcation_points(&self) -> Vec<BifurcationPoint> {
        // Turing instability analysis
        Vec::new() // Simplified for now
    }
    
    fn maintain_far_from_equilibrium(&mut self, energy_flow: Self::Energy) {
        // Adjust feed rates based on energy input
        for (i, &energy) in energy_flow.iter().enumerate() {
            if i == 0 {
                self.reaction_params.feed_rate = (self.reaction_params.feed_rate + energy * 0.001).max(0.0);
            }
        }
    }
    
    fn distance_from_equilibrium(&self) -> f64 {
        // Variance in concentration as measure
        let mean = self.concentrations[0].mean();
        let variance = self.concentrations[0].iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / self.concentrations[0].len() as f64;
        
        variance.sqrt()
    }
    
    fn apply_fluctuations(&mut self, intensity: f64) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for conc_field in &mut self.concentrations {
            for val in conc_field.iter_mut() {
                *val += rng.gen_range(-intensity..intensity);
                *val = val.max(0.0); // Keep concentrations positive
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_brusselator_bifurcation() {
        let mut brusselator = Brusselator::new(1.0, 3.0);
        let bifurcations = brusselator.bifurcation_points();
        assert_eq!(bifurcations.len(), 1);
        assert_eq!(bifurcations[0].parameter_value, 2.0); // b = 1 + a²
    }
    
    #[test]
    fn test_entropy_production() {
        let brusselator = Brusselator::new(1.0, 3.0);
        let entropy = brusselator.entropy_production();
        assert!(entropy >= 0.0);
    }
}