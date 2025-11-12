//! Geodesic calculations using Runge-Kutta 4th order integration
//!
//! Research: Lee (2018) "Introduction to Riemannian Manifolds"

use nalgebra as na;
use crate::{poincare::PoincarePoint, Result};

/// Geodesic in hyperbolic 3-space
#[derive(Debug, Clone)]
pub struct Geodesic {
    start: PoincarePoint,
    initial_velocity: na::Vector3<f64>,
}

impl Geodesic {
    /// Create geodesic from starting point and initial velocity
    pub fn new(start: PoincarePoint, initial_velocity: na::Vector3<f64>) -> Self {
        Self {
            start,
            initial_velocity,
        }
    }

    /// Compute point along geodesic at parameter t
    ///
    /// Uses Runge-Kutta 4th order integration of the geodesic equation:
    /// d²xⁱ/dt² + Γⁱⱼₖ (dxʲ/dt)(dxᵏ/dt) = 0
    pub fn point_at(&self, t: f64) -> Result<PoincarePoint> {
        const DT: f64 = 0.01; // Integration step size
        let num_steps = (t / DT).ceil() as usize;
        let dt = t / num_steps as f64;

        let mut pos = self.start.coords();
        let mut vel = self.initial_velocity;

        for _ in 0..num_steps {
            // RK4 integration
            let (k1_pos, k1_vel) = self.geodesic_equation(pos, vel);
            let (k2_pos, k2_vel) = self.geodesic_equation(
                pos + 0.5 * dt * k1_pos,
                vel + 0.5 * dt * k1_vel,
            );
            let (k3_pos, k3_vel) = self.geodesic_equation(
                pos + 0.5 * dt * k2_pos,
                vel + 0.5 * dt * k2_vel,
            );
            let (k4_pos, k4_vel) = self.geodesic_equation(
                pos + dt * k3_pos,
                vel + dt * k3_vel,
            );

            pos += (dt / 6.0) * (k1_pos + 2.0 * k2_pos + 2.0 * k3_pos + k4_pos);
            vel += (dt / 6.0) * (k1_vel + 2.0 * k2_vel + 2.0 * k3_vel + k4_vel);

            // Project back to Poincaré disk if needed
            let norm = pos.norm();
            if norm >= 0.99 {
                pos *= 0.98 / norm;
            }
        }

        PoincarePoint::new(pos)
    }

    /// Geodesic equation in Poincaré disk coordinates
    ///
    /// Returns (dx/dt, d²x/dt²)
    fn geodesic_equation(
        &self,
        pos: na::Vector3<f64>,
        vel: na::Vector3<f64>,
    ) -> (na::Vector3<f64>, na::Vector3<f64>) {
        let norm_sq = pos.norm_squared();
        let conformal_factor = 2.0 / (1.0 - norm_sq);

        // Christoffel symbols for Poincaré disk
        let gamma = (2.0 / (1.0 - norm_sq)) * pos;
        let vel_dot_pos = vel.dot(&pos);

        let acceleration = -gamma * vel_dot_pos * conformal_factor;

        (vel, acceleration)
    }

    /// Calculate length of geodesic from t=0 to t=T
    pub fn length(&self, t_final: f64) -> Result<f64> {
        const DT: f64 = 0.01;
        let num_steps = (t_final / DT).ceil() as usize;
        let dt = t_final / num_steps as f64;

        let mut length = 0.0;
        let mut prev_point = self.start;

        for i in 1..=num_steps {
            let t = i as f64 * dt;
            let curr_point = self.point_at(t)?;
            length += prev_point.hyperbolic_distance(&curr_point);
            prev_point = curr_point;
        }

        Ok(length)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geodesic_at_origin() {
        let start = PoincarePoint::origin();
        let velocity = na::Vector3::new(0.1, 0.0, 0.0);
        let geodesic = Geodesic::new(start, velocity);

        let point = geodesic.point_at(1.0).unwrap();
        assert!(point.norm() > 0.0);
        assert!(point.norm() < 1.0);
    }

    #[test]
    fn test_geodesic_returns_to_disk() {
        let start = PoincarePoint::new(na::Vector3::new(0.5, 0.0, 0.0)).unwrap();
        let velocity = na::Vector3::new(0.2, 0.3, 0.1);
        let geodesic = Geodesic::new(start, velocity);

        for t in [0.1, 0.5, 1.0, 2.0] {
            let point = geodesic.point_at(t).unwrap();
            assert!(point.norm() < 1.0, "Geodesic left disk at t={}", t);
        }
    }
}
