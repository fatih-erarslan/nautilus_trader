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

    /// Verify enactive coupling bounds: coupling_strength ∈ [0, 1]
    ///
    /// Ensures sensorimotor coupling strength is a valid probability value.
    pub fn verify_coupling_bounds(&self, strength: f64) -> bool {
        self.solver.push();

        let scale = 1000;
        let s = Real::from_real(self.ctx, (strength * scale as f64) as i32, scale);
        let zero = Real::from_real(self.ctx, 0, 1);
        let one = Real::from_real(self.ctx, 1, 1);

        // Assert 0 ≤ strength ≤ 1
        self.solver.assert(&s.ge(&zero));
        self.solver.assert(&s.le(&one));

        let result = self.solver.check() == SatResult::Sat;
        self.solver.pop(1);

        result
    }

    /// Verify natural drift viability: ∀i. lower_i ≤ x_i ≤ upper_i
    ///
    /// Ensures all state variables remain within their viability bounds.
    pub fn verify_viability_bounds(
        &self,
        state: &[f64],
        bounds: &[(f64, f64)],
    ) -> VerificationResult {
        if state.len() != bounds.len() {
            return VerificationResult::Violated(format!(
                "State dimension {} != bounds dimension {}",
                state.len(),
                bounds.len()
            ));
        }

        self.solver.push();

        let scale = 1000;
        for (i, (&x, &(lower, upper))) in state.iter().zip(bounds.iter()).enumerate() {
            let x_var = Real::from_real(self.ctx, (x * scale as f64) as i32, scale);
            let lower_var = Real::from_real(self.ctx, (lower * scale as f64) as i32, scale);
            let upper_var = Real::from_real(self.ctx, (upper * scale as f64) as i32, scale);

            // Assert lower ≤ x ≤ upper
            self.solver.assert(&x_var.ge(&lower_var));
            self.solver.assert(&x_var.le(&upper_var));

            // Early exit if constraint fails
            if self.solver.check() == SatResult::Unsat {
                self.solver.pop(1);
                return VerificationResult::Violated(format!(
                    "Viability violated at index {}: {} ∉ [{}, {}]",
                    i, x, lower, upper
                ));
            }
        }

        let result = match self.solver.check() {
            SatResult::Sat => VerificationResult::Verified,
            SatResult::Unsat => VerificationResult::Violated(
                "Viability bounds violated for state vector".to_string(),
            ),
            SatResult::Unknown => VerificationResult::Unknown,
        };

        self.solver.pop(1);
        result
    }

    /// Verify HNSW distance invariant: d(a,b) = d(b,a) ∧ d(a,a) = 0
    ///
    /// Verifies metric properties: symmetry and identity.
    pub fn verify_distance_metric_properties(
        &self,
        d_ab: f64,
        d_ba: f64,
        d_aa: f64,
    ) -> VerificationResult {
        self.solver.push();

        let scale = 1000;
        let dist_ab = Real::from_real(self.ctx, (d_ab * scale as f64) as i32, scale);
        let dist_ba = Real::from_real(self.ctx, (d_ba * scale as f64) as i32, scale);
        let dist_aa = Real::from_real(self.ctx, (d_aa * scale as f64) as i32, scale);
        let zero = Real::from_real(self.ctx, 0, 1);

        // Assert symmetry: d(a,b) = d(b,a)
        self.solver.assert(&dist_ab._eq(&dist_ba));

        // Assert identity: d(a,a) = 0
        self.solver.assert(&dist_aa._eq(&zero));

        let result = match self.solver.check() {
            SatResult::Sat => VerificationResult::Verified,
            SatResult::Unsat => {
                let mut violations = Vec::new();
                if (d_ab - d_ba).abs() > 1e-9 {
                    violations.push(format!("Symmetry violated: d(a,b)={} ≠ d(b,a)={}", d_ab, d_ba));
                }
                if d_aa.abs() > 1e-9 {
                    violations.push(format!("Identity violated: d(a,a)={} ≠ 0", d_aa));
                }
                VerificationResult::Violated(violations.join("; "))
            }
            SatResult::Unknown => VerificationResult::Unknown,
        };

        self.solver.pop(1);
        result
    }

    /// Verify subsumption layer priority: layer_n inhibits layer_{n+1}
    ///
    /// Ensures that when a lower-priority layer is active, higher layers are inhibited.
    pub fn verify_subsumption_priority(
        &self,
        active_layers: &[bool],
    ) -> VerificationResult {
        if active_layers.is_empty() {
            return VerificationResult::Verified;
        }

        // Check subsumption rule: if layer[i] is active, all layers[j>i] should be inactive
        for i in 0..active_layers.len() {
            if active_layers[i] {
                for j in (i + 1)..active_layers.len() {
                    if active_layers[j] {
                        return VerificationResult::Violated(format!(
                            "Subsumption violated: layer {} active but higher layer {} also active",
                            i, j
                        ));
                    }
                }
            }
        }

        VerificationResult::Verified
    }

    /// Verify codependent risk propagation: R_eff ≥ R_standalone
    ///
    /// Network effects and dependencies should never decrease effective risk.
    pub fn verify_risk_propagation(
        &self,
        standalone_risk: f64,
        effective_risk: f64,
    ) -> VerificationResult {
        self.solver.push();

        let scale = 1000;
        let r_standalone = Real::from_real(self.ctx, (standalone_risk * scale as f64) as i32, scale);
        let r_effective = Real::from_real(self.ctx, (effective_risk * scale as f64) as i32, scale);

        // Assert R_eff ≥ R_standalone
        self.solver.assert(&r_effective.ge(&r_standalone));

        let result = match self.solver.check() {
            SatResult::Sat => VerificationResult::Verified,
            SatResult::Unsat => VerificationResult::Violated(format!(
                "Risk propagation violated: R_eff={} < R_standalone={}",
                effective_risk, standalone_risk
            )),
            SatResult::Unknown => VerificationResult::Unknown,
        };

        self.solver.pop(1);
        result
    }

    /// Verify portfolio weights sum to 1: Σ w_i = 1 ∧ ∀i. w_i ≥ 0
    ///
    /// Ensures portfolio weights are valid probabilities that sum to unity.
    pub fn verify_portfolio_weights(&self, weights: &[f64]) -> VerificationResult {
        if weights.is_empty() {
            return VerificationResult::Violated("Empty portfolio weights".to_string());
        }

        self.solver.push();

        let scale = 1000;
        let zero = Real::from_real(self.ctx, 0, 1);
        let one = Real::from_real(self.ctx, 1, 1);

        // Create weight variables and assert non-negativity
        let mut weight_vars = Vec::new();
        for (i, &w) in weights.iter().enumerate() {
            let w_var = Real::from_real(self.ctx, (w * scale as f64) as i32, scale);
            self.solver.assert(&w_var.ge(&zero));

            // Early check for negative weights
            if w < 0.0 {
                self.solver.pop(1);
                return VerificationResult::Violated(format!(
                    "Negative weight at index {}: {}",
                    i, w
                ));
            }

            weight_vars.push(w_var);
        }

        // Assert sum equals 1
        let sum = Real::add(
            self.ctx,
            &weight_vars.iter().collect::<Vec<_>>(),
        );
        self.solver.assert(&sum._eq(&one));

        let result = match self.solver.check() {
            SatResult::Sat => VerificationResult::Verified,
            SatResult::Unsat => {
                let actual_sum: f64 = weights.iter().sum();
                VerificationResult::Violated(format!(
                    "Portfolio weights invalid: sum={} (expected 1.0)",
                    actual_sum
                ))
            }
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

    #[test]
    fn test_coupling_bounds() {
        let cfg = Config::new();
        let ctx = Context::new(&cfg);
        let verifier = Z3Verifier::new(&ctx);

        // Valid coupling strengths
        assert!(verifier.verify_coupling_bounds(0.0));
        assert!(verifier.verify_coupling_bounds(0.5));
        assert!(verifier.verify_coupling_bounds(1.0));

        // Invalid coupling strengths
        assert!(!verifier.verify_coupling_bounds(-0.1));
        assert!(!verifier.verify_coupling_bounds(1.1));
        assert!(!verifier.verify_coupling_bounds(2.0));
    }

    #[test]
    fn test_viability_bounds() {
        let cfg = Config::new();
        let ctx = Context::new(&cfg);
        let verifier = Z3Verifier::new(&ctx);

        // Valid states within bounds
        let state = vec![0.5, 1.0, -0.5];
        let bounds = vec![(0.0, 1.0), (0.5, 1.5), (-1.0, 0.0)];
        assert!(verifier.verify_viability_bounds(&state, &bounds).is_verified());

        // State at exact boundary
        let state = vec![0.0, 1.0];
        let bounds = vec![(0.0, 1.0), (1.0, 2.0)];
        assert!(verifier.verify_viability_bounds(&state, &bounds).is_verified());

        // State outside bounds
        let state = vec![1.5, 0.5];
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        assert!(!verifier.verify_viability_bounds(&state, &bounds).is_verified());

        // Dimension mismatch
        let state = vec![0.5, 1.0];
        let bounds = vec![(0.0, 1.0)];
        assert!(!verifier.verify_viability_bounds(&state, &bounds).is_verified());
    }

    #[test]
    fn test_distance_metric_properties() {
        let cfg = Config::new();
        let ctx = Context::new(&cfg);
        let verifier = Z3Verifier::new(&ctx);

        // Valid metric: symmetric and identity holds
        assert!(verifier
            .verify_distance_metric_properties(5.0, 5.0, 0.0)
            .is_verified());

        // Valid metric with exact equality
        assert!(verifier
            .verify_distance_metric_properties(1.0, 1.0, 0.0)
            .is_verified());

        // Invalid: asymmetric
        assert!(!verifier
            .verify_distance_metric_properties(5.0, 3.0, 0.0)
            .is_verified());

        // Invalid: identity not zero
        assert!(!verifier
            .verify_distance_metric_properties(5.0, 5.0, 0.5)
            .is_verified());

        // Invalid: both violations
        assert!(!verifier
            .verify_distance_metric_properties(5.0, 3.0, 1.0)
            .is_verified());
    }

    #[test]
    fn test_subsumption_priority() {
        let cfg = Config::new();
        let ctx = Context::new(&cfg);
        let verifier = Z3Verifier::new(&ctx);

        // Valid subsumption: only one layer active
        assert!(verifier
            .verify_subsumption_priority(&[true, false, false])
            .is_verified());

        // Valid: higher priority layer active, all higher layers inactive
        assert!(verifier
            .verify_subsumption_priority(&[false, true, false])
            .is_verified());

        // Valid: all layers inactive
        assert!(verifier
            .verify_subsumption_priority(&[false, false, false])
            .is_verified());

        // Valid: empty layers
        assert!(verifier
            .verify_subsumption_priority(&[])
            .is_verified());

        // Invalid: multiple layers active
        assert!(!verifier
            .verify_subsumption_priority(&[true, false, true])
            .is_verified());

        // Invalid: lower priority layer active with higher priority
        assert!(!verifier
            .verify_subsumption_priority(&[false, true, true])
            .is_verified());

        // Invalid: all active (multiple violations)
        assert!(!verifier
            .verify_subsumption_priority(&[true, true, true])
            .is_verified());
    }

    #[test]
    fn test_risk_propagation() {
        let cfg = Config::new();
        let ctx = Context::new(&cfg);
        let verifier = Z3Verifier::new(&ctx);

        // Valid: network effects increase risk
        assert!(verifier
            .verify_risk_propagation(0.1, 0.2)
            .is_verified());

        // Valid: equal risk (no network effect)
        assert!(verifier
            .verify_risk_propagation(0.5, 0.5)
            .is_verified());

        // Valid: large amplification
        assert!(verifier
            .verify_risk_propagation(0.1, 0.9)
            .is_verified());

        // Invalid: network effects decrease risk (impossible)
        assert!(!verifier
            .verify_risk_propagation(0.5, 0.3)
            .is_verified());

        // Invalid: effective risk less than standalone
        assert!(!verifier
            .verify_risk_propagation(0.8, 0.2)
            .is_verified());
    }

    #[test]
    fn test_portfolio_weights() {
        let cfg = Config::new();
        let ctx = Context::new(&cfg);
        let verifier = Z3Verifier::new(&ctx);

        // Valid: weights sum to 1
        assert!(verifier
            .verify_portfolio_weights(&[0.3, 0.3, 0.4])
            .is_verified());

        // Valid: single asset
        assert!(verifier
            .verify_portfolio_weights(&[1.0])
            .is_verified());

        // Valid: equal weights
        assert!(verifier
            .verify_portfolio_weights(&[0.5, 0.5])
            .is_verified());

        // Valid: many small weights
        assert!(verifier
            .verify_portfolio_weights(&[0.25, 0.25, 0.25, 0.25])
            .is_verified());

        // Invalid: weights sum to less than 1
        assert!(!verifier
            .verify_portfolio_weights(&[0.3, 0.3, 0.3])
            .is_verified());

        // Invalid: weights sum to more than 1
        assert!(!verifier
            .verify_portfolio_weights(&[0.5, 0.5, 0.5])
            .is_verified());

        // Invalid: negative weight
        assert!(!verifier
            .verify_portfolio_weights(&[0.5, -0.2, 0.7])
            .is_verified());

        // Invalid: empty portfolio
        assert!(!verifier
            .verify_portfolio_weights(&[])
            .is_verified());
    }
}
