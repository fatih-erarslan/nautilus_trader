//! Z3 SMT solver integration for runtime verification

use z3::ast::{Ast, Real};
use z3::*;

use crate::VerificationResult;

/// Z3-based theorem verifier
pub struct Z3Verifier<'ctx> {
    ctx: &'ctx Context,
    solver: Solver<'ctx>,
}

impl<'ctx> Z3Verifier<'ctx> {
    /// Create new verifier with given context
    pub fn new(ctx: &'ctx Context) -> Self {
        let cfg = Config::new();
        let solver = Solver::new(ctx);
        Self { ctx, solver }
    }

    /// Verify probability is in [0, 1]
    ///
    /// Uses Z3 to prove: ∀p. 0 ≤ p ≤ 1
    pub fn verify_probability_bounds(&self, p: f64) -> bool {
        self.solver.push();

        let p_var = Real::from_real(self.ctx, (p * 1000.0) as i32, 1000);
        let zero = Real::from_real(self.ctx, 0, 1);
        let one = Real::from_real(self.ctx, 1, 1);

        // Assert constraints: 0 ≤ p ≤ 1
        self.solver.assert(&p_var.ge(&zero));
        self.solver.assert(&p_var.le(&one));

        let result = self.solver.check() == SatResult::Sat;
        self.solver.pop(1);

        result
    }

    /// Verify second law of thermodynamics: ΔS ≥ 0
    pub fn verify_second_law(&self, delta_s: f64, tolerance: f64) -> VerificationResult {
        self.solver.push();

        let ds = Real::from_real(self.ctx, (delta_s * 1e10) as i32, 1e10 as i32);
        let tol = Real::from_real(self.ctx, -(tolerance * 1e10) as i32, 1e10 as i32);

        // Assert ΔS ≥ -tolerance
        self.solver.assert(&ds.ge(&tol));

        let result = match self.solver.check() {
            SatResult::Sat => VerificationResult::Verified,
            SatResult::Unsat => {
                VerificationResult::Violated(format!("ΔS = {} < -{}", delta_s, tolerance))
            }
            SatResult::Unknown => VerificationResult::Unknown,
        };

        self.solver.pop(1);
        result
    }

    /// Verify Landauer's bound: E ≥ k_B T ln(2) * N_erasures
    pub fn verify_landauer_bound(
        &self,
        energy_j: f64,
        erasures: usize,
        temperature_k: f64,
    ) -> VerificationResult {
        const K_B: f64 = 1.380649e-23; // Boltzmann constant
        let min_energy = erasures as f64 * K_B * temperature_k * 2.0_f64.ln();

        self.solver.push();

        // Convert to scaled integers to avoid floating point in Z3
        let scale = 1e23;
        let e_scaled = Real::from_real(self.ctx, (energy_j * scale) as i32, scale as i32);
        let min_scaled =
            Real::from_real(self.ctx, (min_energy * scale) as i32, scale as i32);

        // Assert E ≥ E_min
        self.solver.assert(&e_scaled.ge(&min_scaled));

        let result = match self.solver.check() {
            SatResult::Sat => VerificationResult::Verified,
            SatResult::Unsat => VerificationResult::Violated(format!(
                "Landauer violation: E={} < E_min={} for {} erasures at {}K",
                energy_j, min_energy, erasures, temperature_k
            )),
            SatResult::Unknown => VerificationResult::Unknown,
        };

        self.solver.pop(1);
        result
    }

    /// Verify hyperbolic triangle inequality
    ///
    /// For points p, q, r in Poincaré disk: d(p,r) ≤ d(p,q) + d(q,r)
    pub fn verify_triangle_inequality(
        &self,
        d_pr: f64,
        d_pq: f64,
        d_qr: f64,
    ) -> VerificationResult {
        self.solver.push();

        let scale = 1000;
        let dist_pr = Real::from_real(self.ctx, (d_pr * scale as f64) as i32, scale);
        let dist_pq = Real::from_real(self.ctx, (d_pq * scale as f64) as i32, scale);
        let dist_qr = Real::from_real(self.ctx, (d_qr * scale as f64) as i32, scale);

        // Assert d(p,r) ≤ d(p,q) + d(q,r)
        let sum = Real::add(self.ctx, &[&dist_pq, &dist_qr]);
        self.solver.assert(&dist_pr.le(&sum));

        let result = match self.solver.check() {
            SatResult::Sat => VerificationResult::Verified,
            SatResult::Unsat => VerificationResult::Violated(format!(
                "Triangle inequality violated: {} > {} + {}",
                d_pr, d_pq, d_qr
            )),
            SatResult::Unknown => VerificationResult::Unknown,
        };

        self.solver.pop(1);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probability_verification() {
        let cfg = Config::new();
        let ctx = Context::new(&cfg);
        let verifier = Z3Verifier::new(&ctx);

        assert!(verifier.verify_probability_bounds(0.0));
        assert!(verifier.verify_probability_bounds(0.5));
        assert!(verifier.verify_probability_bounds(1.0));

        assert!(!verifier.verify_probability_bounds(-0.1));
        assert!(!verifier.verify_probability_bounds(1.1));
    }

    #[test]
    fn test_second_law_verification() {
        let cfg = Config::new();
        let ctx = Context::new(&cfg);
        let verifier = Z3Verifier::new(&ctx);

        assert!(verifier.verify_second_law(0.1, 1e-10).is_verified());
        assert!(verifier.verify_second_law(0.0, 1e-10).is_verified());

        assert!(!verifier.verify_second_law(-1.0, 1e-10).is_verified());
    }

    #[test]
    fn test_triangle_inequality() {
        let cfg = Config::new();
        let ctx = Context::new(&cfg);
        let verifier = Z3Verifier::new(&ctx);

        // Valid triangle
        assert!(verifier
            .verify_triangle_inequality(3.0, 1.0, 2.0)
            .is_verified());
        assert!(verifier
            .verify_triangle_inequality(2.0, 1.0, 1.0)
            .is_verified());

        // Degenerate case (equality)
        assert!(verifier
            .verify_triangle_inequality(2.0, 1.0, 1.0)
            .is_verified());
    }
}
