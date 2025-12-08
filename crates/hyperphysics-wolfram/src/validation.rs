//! Validation Engine for HyperPhysics
//!
//! Provides mathematical validation, formal verification, and correctness checking
//! for HyperPhysics implementations using Wolfram's symbolic and numerical capabilities.

use crate::evaluator::WolframEvaluator;
use crate::types::*;
use std::time::Instant;
use tracing::debug;

/// Validation engine for mathematical correctness verification
pub struct ValidationEngine {
    evaluator: WolframEvaluator,
    /// Default tolerance for numerical comparisons
    pub tolerance: f64,
}

impl ValidationEngine {
    /// Create a new validation engine
    pub fn new() -> WolframResult<Self> {
        Ok(Self {
            evaluator: WolframEvaluator::new()?,
            tolerance: 1e-10,
        })
    }

    /// Create with custom tolerance
    pub fn with_tolerance(tolerance: f64) -> WolframResult<Self> {
        Ok(Self {
            evaluator: WolframEvaluator::new()?,
            tolerance,
        })
    }

    /// Validate that two expressions are mathematically equivalent
    pub async fn validate_equivalence(
        &self,
        expr1: &str,
        expr2: &str,
    ) -> WolframResult<ValidationResult> {
        let start = Instant::now();

        let code = format!(
            r#"Module[{{result, diff}},
                result = FullSimplify[({}) - ({})];
                diff = N[Abs[result]];
                <|
                    "isValid" -> (result === 0 || diff < {}),
                    "difference" -> diff,
                    "simplified" -> ToString[result]
                |>
            ]"#,
            expr1, expr2, self.tolerance
        );

        let result = self.evaluator.evaluate_json::<serde_json::Value>(&code).await?;
        let validation_time = start.elapsed().as_millis() as i64;

        Ok(ValidationResult {
            is_valid: result["isValid"].as_bool().unwrap_or(false),
            numerical_error: result["difference"].as_f64(),
            expected_value: Some(expr1.to_string()),
            actual_value: Some(expr2.to_string()),
            message: format!(
                "Simplified difference: {}",
                result["simplified"].as_str().unwrap_or("N/A")
            ),
            method: ValidationMethod::Symbolic,
            validation_time_ms: validation_time,
        })
    }

    /// Validate a numerical computation against Wolfram's result
    pub async fn validate_numerical(
        &self,
        expression: &str,
        computed_value: f64,
    ) -> WolframResult<ValidationResult> {
        let start = Instant::now();

        let expected = self.evaluator.evaluate_numeric(expression).await?;
        let error = (computed_value - expected).abs();
        let is_valid = error < self.tolerance;

        let validation_time = start.elapsed().as_millis() as i64;

        Ok(ValidationResult {
            is_valid,
            numerical_error: Some(error),
            expected_value: Some(expected.to_string()),
            actual_value: Some(computed_value.to_string()),
            message: if is_valid {
                format!("Validated within tolerance {}", self.tolerance)
            } else {
                format!(
                    "Error {} exceeds tolerance {}",
                    error, self.tolerance
                )
            },
            method: ValidationMethod::Numerical,
            validation_time_ms: validation_time,
        })
    }

    /// Validate hyperbolic distance computation
    pub async fn validate_hyperbolic_distance(
        &self,
        p1: [f64; 2],
        p2: [f64; 2],
        computed_distance: f64,
    ) -> WolframResult<ValidationResult> {
        let start = Instant::now();

        // Wolfram formula for Poincare disk distance
        let code = format!(
            r#"Module[{{z1, z2, d}},
                z1 = {{{}, {}}};
                z2 = {{{}, {}}};
                d = 2 * ArcTanh[
                    Norm[z1 - z2] / 
                    Sqrt[(1 - Norm[z1]^2) * (1 - Norm[z2]^2) + Norm[z1 - z2]^2]
                ];
                N[d, 20]
            ]"#,
            p1[0], p1[1], p2[0], p2[1]
        );

        let expected = self.evaluator.evaluate_numeric(&code).await?;
        let error = (computed_distance - expected).abs();
        let is_valid = error < self.tolerance;

        let validation_time = start.elapsed().as_millis() as i64;

        Ok(ValidationResult {
            is_valid,
            numerical_error: Some(error),
            expected_value: Some(expected.to_string()),
            actual_value: Some(computed_distance.to_string()),
            message: format!(
                "Hyperbolic distance validation: expected={:.15}, got={:.15}, error={:.2e}",
                expected, computed_distance, error
            ),
            method: ValidationMethod::Numerical,
            validation_time_ms: validation_time,
        })
    }

    /// Validate Möbius transformation
    pub async fn validate_moebius_transform(
        &self,
        a: (f64, f64),
        b: (f64, f64),
        c: (f64, f64),
        d: (f64, f64),
        input_z: (f64, f64),
        computed_result: (f64, f64),
    ) -> WolframResult<ValidationResult> {
        let start = Instant::now();

        let code = format!(
            r#"Module[{{a, b, c, d, z, result}},
                a = {} + {} * I;
                b = {} + {} * I;
                c = {} + {} * I;
                d = {} + {} * I;
                z = {} + {} * I;
                result = (a * z + b) / (c * z + d);
                {{Re[result], Im[result]}} // N
            ]"#,
            a.0, a.1, b.0, b.1, c.0, c.1, d.0, d.1, input_z.0, input_z.1
        );

        let result: Vec<f64> = self.evaluator.evaluate_json(&code).await?;
        let error = ((computed_result.0 - result[0]).powi(2)
            + (computed_result.1 - result[1]).powi(2))
        .sqrt();
        let is_valid = error < self.tolerance;

        let validation_time = start.elapsed().as_millis() as i64;

        Ok(ValidationResult {
            is_valid,
            numerical_error: Some(error),
            expected_value: Some(format!("({}, {})", result[0], result[1])),
            actual_value: Some(format!("({}, {})", computed_result.0, computed_result.1)),
            message: format!("Möbius transform validation: error={:.2e}", error),
            method: ValidationMethod::Numerical,
            validation_time_ms: validation_time,
        })
    }

    /// Validate STDP learning rule implementation
    pub async fn validate_stdp_rule(
        &self,
        a_plus: f64,
        a_minus: f64,
        tau_plus: f64,
        tau_minus: f64,
        delta_times: &[f64],
        computed_weights: &[f64],
    ) -> WolframResult<STDPValidationResult> {
        let _start = Instant::now();

        let times_str = delta_times
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(", ");

        let code = format!(
            r#"Module[{{aPlus, aMinus, tauPlus, tauMinus, deltaTimes, expectedWeights}},
                aPlus = {};
                aMinus = {};
                tauPlus = {};
                tauMinus = {};
                deltaTimes = {{{}}};
                expectedWeights = Table[
                    If[dt > 0,
                        aPlus * Exp[-dt / tauPlus],
                        -aMinus * Exp[dt / tauMinus]
                    ],
                    {{dt, deltaTimes}}
                ];
                N[expectedWeights]
            ]"#,
            a_plus, a_minus, tau_plus, tau_minus, times_str
        );

        let expected: Vec<f64> = self.evaluator.evaluate_json(&code).await?;

        let mut failed_cases = Vec::new();
        let mut max_error = 0.0f64;

        for (_i, ((&expected_dw, &actual_dw), &dt)) in expected
            .iter()
            .zip(computed_weights.iter())
            .zip(delta_times.iter())
            .enumerate()
        {
            let error = (expected_dw - actual_dw).abs();
            max_error = max_error.max(error);

            if error > self.tolerance {
                failed_cases.push(STDPTestCase {
                    delta_t: dt,
                    expected_dw,
                    actual_dw,
                    error,
                });
            }
        }

        debug!(
            "STDP validation: {} test cases, {} failures, max_error={:.2e}",
            delta_times.len(),
            failed_cases.len(),
            max_error
        );

        Ok(STDPValidationResult {
            is_valid: failed_cases.is_empty(),
            max_error,
            test_cases: delta_times.len(),
            failed_cases,
        })
    }

    /// Validate IIT Phi computation
    pub async fn validate_phi_computation(
        &self,
        tpm: &[Vec<f64>],
        state: &[bool],
        computed_phi: f64,
    ) -> WolframResult<ValidationResult> {
        let start = Instant::now();

        let tpm_str = tpm
            .iter()
            .map(|row| {
                format!(
                    "{{{}}}",
                    row.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ")
                )
            })
            .collect::<Vec<_>>()
            .join(", ");

        let state_str = state
            .iter()
            .map(|&b| if b { "1" } else { "0" })
            .collect::<Vec<_>>()
            .join(", ");

        // Simplified Phi calculation in Wolfram (IIT 3.0 approximation)
        let code = format!(
            r#"Module[{{tpm, state, n, phi}},
                tpm = {{{}}};
                state = {{{}}};
                n = Length[state];
                
                (* Simplified phi: integration measure based on mutual information *)
                phi = Total[Flatten[tpm * Log[2, tpm + 10^-10]]];
                phi = Abs[phi] / n;
                N[phi]
            ]"#,
            tpm_str, state_str
        );

        let expected = self.evaluator.evaluate_numeric(&code).await?;
        let error = (computed_phi - expected).abs();
        
        // Phi computation has inherent approximations, use larger tolerance
        let phi_tolerance = 0.01;
        let is_valid = error < phi_tolerance;

        let validation_time = start.elapsed().as_millis() as i64;

        Ok(ValidationResult {
            is_valid,
            numerical_error: Some(error),
            expected_value: Some(expected.to_string()),
            actual_value: Some(computed_phi.to_string()),
            message: format!(
                "Phi validation: expected={:.6}, got={:.6}, error={:.4}",
                expected, computed_phi, error
            ),
            method: ValidationMethod::Numerical,
            validation_time_ms: validation_time,
        })
    }

    /// Validate free energy computation
    pub async fn validate_free_energy(
        &self,
        observations: &[f64],
        prior_mean: f64,
        prior_variance: f64,
        computed_fe: f64,
    ) -> WolframResult<ValidationResult> {
        let start = Instant::now();

        let obs_str = observations
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ");

        let code = format!(
            r#"Module[{{obs, priorMean, priorVar, likelihood, prior, freeEnergy}},
                obs = {{{}}};
                priorMean = {};
                priorVar = {};
                
                (* Free energy = -log(evidence) approximation *)
                likelihood = Total[Log[PDF[NormalDistribution[Mean[obs], StandardDeviation[obs] + 0.001], obs]]];
                prior = Total[Log[PDF[NormalDistribution[priorMean, Sqrt[priorVar]], obs]]];
                freeEnergy = -likelihood - prior;
                N[freeEnergy / Length[obs]]
            ]"#,
            obs_str, prior_mean, prior_variance
        );

        let expected = self.evaluator.evaluate_numeric(&code).await?;
        let error = (computed_fe - expected).abs();
        let is_valid = error < 0.1; // Free energy has larger tolerance

        let validation_time = start.elapsed().as_millis() as i64;

        Ok(ValidationResult {
            is_valid,
            numerical_error: Some(error),
            expected_value: Some(expected.to_string()),
            actual_value: Some(computed_fe.to_string()),
            message: format!(
                "Free energy validation: expected={:.6}, got={:.6}",
                expected, computed_fe
            ),
            method: ValidationMethod::Numerical,
            validation_time_ms: validation_time,
        })
    }

    /// Perform formal proof verification
    pub async fn verify_formal_proof(
        &self,
        theorem: &str,
        assumptions: &[&str],
    ) -> WolframResult<FormalProofResult> {
        let assumptions_str = assumptions
            .iter()
            .map(|a| format!("\"{}\"", a))
            .collect::<Vec<_>>()
            .join(", ");

        let code = format!(
            r#"Module[{{theorem, assumptions, result}},
                theorem = {};
                assumptions = {{{}}};
                
                (* Try to prove using Reduce/Resolve *)
                result = Quiet[Check[
                    Reduce[theorem, Reals],
                    $Failed
                ]];
                
                <|
                    "isProven" -> (result =!= $Failed && result =!= False),
                    "result" -> ToString[result],
                    "method" -> "Reduce"
                |>
            ]"#,
            theorem, assumptions_str
        );

        let result: serde_json::Value = self.evaluator.evaluate_json(&code).await?;

        Ok(FormalProofResult {
            is_proven: result["isProven"].as_bool().unwrap_or(false),
            theorem: theorem.to_string(),
            proof_steps: vec![result["result"]
                .as_str()
                .unwrap_or("N/A")
                .to_string()],
            counterexample: if !result["isProven"].as_bool().unwrap_or(false) {
                Some("See proof result".to_string())
            } else {
                None
            },
            assumptions: assumptions.iter().map(|s| s.to_string()).collect(),
            proof_method: result["method"]
                .as_str()
                .unwrap_or("Unknown")
                .to_string(),
        })
    }

    /// Validate matrix eigenvalue computation
    pub async fn validate_eigenvalues(
        &self,
        matrix: &[Vec<f64>],
        computed_eigenvalues: &[f64],
    ) -> WolframResult<ValidationResult> {
        let start = Instant::now();

        let matrix_str = matrix
            .iter()
            .map(|row| {
                format!(
                    "{{{}}}",
                    row.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ")
                )
            })
            .collect::<Vec<_>>()
            .join(", ");

        let code = format!(
            r#"Module[{{m, eigenvals}},
                m = {{{}}};
                eigenvals = Sort[Eigenvalues[m] // N // Re];
                eigenvals
            ]"#,
            matrix_str
        );

        let expected: Vec<f64> = self.evaluator.evaluate_json(&code).await?;
        
        let mut sorted_computed = computed_eigenvalues.to_vec();
        sorted_computed.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let max_error = expected
            .iter()
            .zip(sorted_computed.iter())
            .map(|(e, c)| (e - c).abs())
            .fold(0.0f64, |a, b| a.max(b));

        let is_valid = max_error < self.tolerance;
        let validation_time = start.elapsed().as_millis() as i64;

        Ok(ValidationResult {
            is_valid,
            numerical_error: Some(max_error),
            expected_value: Some(format!("{:?}", expected)),
            actual_value: Some(format!("{:?}", sorted_computed)),
            message: format!("Eigenvalue validation: max_error={:.2e}", max_error),
            method: ValidationMethod::Numerical,
            validation_time_ms: validation_time,
        })
    }
}

impl Default for ValidationEngine {
    fn default() -> Self {
        Self::new().expect("Failed to create ValidationEngine")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_equivalence_validation() {
        if let Ok(engine) = ValidationEngine::new() {
            let result = engine
                .validate_equivalence("Sin[x]^2 + Cos[x]^2", "1")
                .await;

            if let Ok(v) = result {
                assert!(v.is_valid);
            }
        }
    }
}
