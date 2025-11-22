//! # Quantum Tunneling Module
//! 
//! Minimal quantum tunneling implementation.

/// Quantum tunneling probability calculation
pub fn tunneling_probability(barrier_height: f64, particle_energy: f64) -> f64 {
    if particle_energy >= barrier_height {
        1.0
    } else {
        let ratio = particle_energy / barrier_height;
        ratio.exp()
    }
}