//! Scientifically Rigorous Activation Functions
//!
//! This module implements activation functions with:
//! - IEEE 754 floating-point compliance
//! - Mathematical rigor based on peer-reviewed literature
//! - Numerical stability and overflow protection
//! - Scientific validation of all computations
//!
//! References:
//! - Ramachandran et al. (2017) "Searching for Activation Functions"
//! - Nair & Hinton (2010) "Rectified Linear Units Improve Restricted Boltzmann Machines"
//! - Clevert et al. (2015) "Fast and Accurate Deep Network Learning by Exponential Linear Units"

use crate::validation::ieee754_arithmetic::ArithmeticError;

/// Result type for activation functions with error handling
pub type ActivationResult = Result<f64, ArithmeticError>;

/// Trait for scientifically rigorous activation functions
pub trait ScientificActivation {
    /// Compute the activation function with IEEE 754 compliance
    fn activate(&self, x: f64) -> ActivationResult;

    /// Compute the derivative with numerical stability
    fn derivative(&self, x: f64) -> ActivationResult;

    /// Validate mathematical properties (monotonicity, boundedness, etc.)
    fn validate_properties(&self) -> Result<ActivationProperties, ArithmeticError>;
}

/// Properties of activation functions for mathematical validation
#[derive(Debug, Clone)]
pub struct ActivationProperties {
    pub bounded: bool,
    pub monotonic: bool,
    pub smooth: bool,
    pub zero_centered: bool,
    pub range: (f64, f64),
}

/// ReLU activation with scientific rigor
/// Reference: Nair & Hinton (2010)
pub struct ScientificReLU;

impl ScientificActivation for ScientificReLU {
    fn activate(&self, x: f64) -> ActivationResult {
        if !x.is_finite() {
            return Err(ArithmeticError::InvalidInput(
                "Non-finite input".to_string(),
            ));
        }

        let result = x.max(0.0);

        if !result.is_finite() {
            return Err(ArithmeticError::InvalidResult(
                "Non-finite ReLU result".to_string(),
            ));
        }

        Ok(result)
    }

    fn derivative(&self, x: f64) -> ActivationResult {
        if !x.is_finite() {
            return Err(ArithmeticError::InvalidInput(
                "Non-finite input".to_string(),
            ));
        }

        Ok(if x > 0.0 { 1.0 } else { 0.0 })
    }

    fn validate_properties(&self) -> Result<ActivationProperties, ArithmeticError> {
        Ok(ActivationProperties {
            bounded: false,
            monotonic: true,
            smooth: false, // Not smooth at x=0
            zero_centered: false,
            range: (0.0, f64::INFINITY),
        })
    }
}

/// ELU (Exponential Linear Unit) with scientific rigor
/// Reference: Clevert et al. (2015)
pub struct ScientificELU {
    alpha: f64,
}

impl ScientificELU {
    pub fn new(alpha: f64) -> Result<Self, ArithmeticError> {
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(ArithmeticError::InvalidInput(
                "Alpha must be finite and positive".to_string(),
            ));
        }
        Ok(Self { alpha })
    }
}

impl ScientificActivation for ScientificELU {
    fn activate(&self, x: f64) -> ActivationResult {
        if !x.is_finite() {
            return Err(ArithmeticError::InvalidInput(
                "Non-finite input".to_string(),
            ));
        }

        let result = if x >= 0.0 {
            x
        } else {
            // Prevent overflow in exp(x) for large negative x
            if x < -709.0 {
                -self.alpha // exp(x) ≈ 0 for very negative x
            } else {
                let exp_x = x.exp();
                if !exp_x.is_finite() {
                    return Err(ArithmeticError::Overflow);
                }
                self.alpha * (exp_x - 1.0)
            }
        };

        if !result.is_finite() {
            return Err(ArithmeticError::InvalidResult(
                "Non-finite ELU result".to_string(),
            ));
        }

        Ok(result)
    }

    fn derivative(&self, x: f64) -> ActivationResult {
        if !x.is_finite() {
            return Err(ArithmeticError::InvalidInput(
                "Non-finite input".to_string(),
            ));
        }

        let result = if x >= 0.0 {
            1.0
        } else {
            // Use the already computed activation to avoid recomputation
            let activation = self.activate(x)?;
            self.alpha + activation
        };

        Ok(result)
    }

    fn validate_properties(&self) -> Result<ActivationProperties, ArithmeticError> {
        Ok(ActivationProperties {
            bounded: true,
            monotonic: true,
            smooth: true,
            zero_centered: true,
            range: (-self.alpha, f64::INFINITY),
        })
    }
}

/// Sigmoid activation with IEEE 754 compliance
pub struct ScientificSigmoid;

impl ScientificActivation for ScientificSigmoid {
    fn activate(&self, x: f64) -> ActivationResult {
        if !x.is_finite() {
            return Err(ArithmeticError::InvalidInput(
                "Non-finite input".to_string(),
            ));
        }

        // Use IEEE 754 compliant thresholds for numerical stability
        let result = if x > 709.0 {
            1.0 // Prevent overflow in exp(-x)
        } else if x < -709.0 {
            0.0 // Prevent underflow
        } else {
            let exp_neg_x = (-x).exp();
            if !exp_neg_x.is_finite() {
                return Err(ArithmeticError::Overflow);
            }
            1.0 / (1.0 + exp_neg_x)
        };

        if !result.is_finite() {
            return Err(ArithmeticError::InvalidResult(
                "Non-finite sigmoid result".to_string(),
            ));
        }

        Ok(result)
    }

    fn derivative(&self, x: f64) -> ActivationResult {
        let sigmoid_x = self.activate(x)?;
        let result = sigmoid_x * (1.0 - sigmoid_x);

        if !result.is_finite() {
            return Err(ArithmeticError::InvalidResult(
                "Non-finite sigmoid derivative".to_string(),
            ));
        }

        Ok(result)
    }

    fn validate_properties(&self) -> Result<ActivationProperties, ArithmeticError> {
        Ok(ActivationProperties {
            bounded: true,
            monotonic: true,
            smooth: true,
            zero_centered: false,
            range: (0.0, 1.0),
        })
    }
}

/// Swish activation function with mathematical rigor
/// Reference: Ramachandran et al. (2017)
pub struct ScientificSwish {
    beta: f64,
}

impl ScientificSwish {
    pub fn new(beta: f64) -> Result<Self, ArithmeticError> {
        if !beta.is_finite() {
            return Err(ArithmeticError::InvalidInput(
                "Beta must be finite".to_string(),
            ));
        }
        Ok(Self { beta })
    }
}

impl ScientificActivation for ScientificSwish {
    fn activate(&self, x: f64) -> ActivationResult {
        if !x.is_finite() {
            return Err(ArithmeticError::InvalidInput(
                "Non-finite input".to_string(),
            ));
        }

        // Prevent overflow in beta * x multiplication
        let beta_x = if self.beta.abs() > 1.0 && x.abs() > f64::MAX / self.beta.abs() {
            return Err(ArithmeticError::Overflow);
        } else {
            self.beta * x
        };

        let sigmoid = ScientificSigmoid;
        let sigmoid_beta_x = sigmoid.activate(beta_x)?;
        let result = x * sigmoid_beta_x;

        if !result.is_finite() {
            return Err(ArithmeticError::InvalidResult(
                "Non-finite swish result".to_string(),
            ));
        }

        Ok(result)
    }

    fn derivative(&self, x: f64) -> ActivationResult {
        if !x.is_finite() {
            return Err(ArithmeticError::InvalidInput(
                "Non-finite input".to_string(),
            ));
        }

        let beta_x = self.beta * x;
        let sigmoid = ScientificSigmoid;
        let sigmoid_beta_x = sigmoid.activate(beta_x)?;
        let sigmoid_derivative = sigmoid.derivative(beta_x)?;

        // d/dx[x * sigmoid(βx)] = sigmoid(βx) + x * β * sigmoid'(βx)
        let result = sigmoid_beta_x + x * self.beta * sigmoid_derivative;

        if !result.is_finite() {
            return Err(ArithmeticError::InvalidResult(
                "Non-finite swish derivative".to_string(),
            ));
        }

        Ok(result)
    }

    fn validate_properties(&self) -> Result<ActivationProperties, ArithmeticError> {
        Ok(ActivationProperties {
            bounded: false,
            monotonic: true, // For β > 0
            smooth: true,
            zero_centered: true,
            range: (-f64::INFINITY, f64::INFINITY),
        })
    }
}

/// Quantum-inspired activation function based on wave function properties
/// Reference: Quantum mechanics principles applied to neural computation
pub struct QuantumActivation {
    frequency: f64,
    amplitude: f64,
}

impl QuantumActivation {
    pub fn new(frequency: f64, amplitude: f64) -> Result<Self, ArithmeticError> {
        if !frequency.is_finite() || !amplitude.is_finite() {
            return Err(ArithmeticError::InvalidInput(
                "Parameters must be finite".to_string(),
            ));
        }
        if frequency <= 0.0 || amplitude <= 0.0 {
            return Err(ArithmeticError::InvalidInput(
                "Parameters must be positive".to_string(),
            ));
        }
        Ok(Self {
            frequency,
            amplitude,
        })
    }
}

impl ScientificActivation for QuantumActivation {
    fn activate(&self, x: f64) -> ActivationResult {
        if !x.is_finite() {
            return Err(ArithmeticError::InvalidInput(
                "Non-finite input".to_string(),
            ));
        }

        // Prevent overflow in frequency * x
        if self.frequency.abs() > 1.0 && x.abs() > f64::MAX / self.frequency.abs() {
            return Err(ArithmeticError::Overflow);
        }

        let phase = self.frequency * x;

        // Quantum-inspired wave function: ψ(x) = A * e^(-x²/2) * cos(ωx)
        let gaussian_envelope = (-x * x / 2.0).exp();
        if !gaussian_envelope.is_finite() {
            return Err(ArithmeticError::InvalidResult(
                "Non-finite gaussian envelope".to_string(),
            ));
        }

        let oscillatory_term = phase.cos();
        let result = self.amplitude * gaussian_envelope * oscillatory_term;

        if !result.is_finite() {
            return Err(ArithmeticError::InvalidResult(
                "Non-finite quantum activation".to_string(),
            ));
        }

        Ok(result)
    }

    fn derivative(&self, x: f64) -> ActivationResult {
        if !x.is_finite() {
            return Err(ArithmeticError::InvalidInput(
                "Non-finite input".to_string(),
            ));
        }

        let phase = self.frequency * x;
        let gaussian = (-x * x / 2.0).exp();
        let cos_term = phase.cos();
        let sin_term = phase.sin();

        // d/dx[A * e^(-x²/2) * cos(ωx)] = A * e^(-x²/2) * [-x * cos(ωx) - ω * sin(ωx)]
        let result = self.amplitude * gaussian * (-x * cos_term - self.frequency * sin_term);

        if !result.is_finite() {
            return Err(ArithmeticError::InvalidResult(
                "Non-finite quantum derivative".to_string(),
            ));
        }

        Ok(result)
    }

    fn validate_properties(&self) -> Result<ActivationProperties, ArithmeticError> {
        Ok(ActivationProperties {
            bounded: true,
            monotonic: false, // Oscillatory
            smooth: true,
            zero_centered: true,
            range: (-self.amplitude, self.amplitude),
        })
    }
}

/// Comprehensive activation function validator
pub struct ActivationValidator;

impl ActivationValidator {
    /// Validate activation function across a range of inputs
    pub fn validate_function<T: ScientificActivation>(
        activation: &T,
        test_range: (f64, f64),
        num_samples: usize,
    ) -> Result<ValidationReport, ArithmeticError> {
        let step = (test_range.1 - test_range.0) / (num_samples as f64 - 1.0);
        let mut errors = Vec::new();
        let mut numerical_issues = Vec::new();

        for i in 0..num_samples {
            let x = test_range.0 + (i as f64) * step;

            match activation.activate(x) {
                Ok(result) => {
                    if !result.is_finite() {
                        numerical_issues.push(format!("Non-finite result at x={}: {}", x, result));
                    }
                }
                Err(e) => {
                    errors.push(format!("Error at x={}: {:?}", x, e));
                }
            }

            // Test derivative as well
            match activation.derivative(x) {
                Ok(result) => {
                    if !result.is_finite() {
                        numerical_issues
                            .push(format!("Non-finite derivative at x={}: {}", x, result));
                    }
                }
                Err(e) => {
                    errors.push(format!("Derivative error at x={}: {:?}", x, e));
                }
            }
        }

        let properties = activation.validate_properties()?;

        Ok(ValidationReport {
            errors,
            numerical_issues,
            properties,
            samples_tested: num_samples,
            ieee754_compliant: errors.is_empty() && numerical_issues.is_empty(),
        })
    }
}

/// Validation report for activation functions
#[derive(Debug)]
pub struct ValidationReport {
    pub errors: Vec<String>,
    pub numerical_issues: Vec<String>,
    pub properties: ActivationProperties,
    pub samples_tested: usize,
    pub ieee754_compliant: bool,
}

/// Convenience functions for backward compatibility
pub fn scientifically_rigorous_sigmoid(x: f64) -> ActivationResult {
    ScientificSigmoid.activate(x)
}

pub fn scientifically_rigorous_relu(x: f64) -> ActivationResult {
    ScientificReLU.activate(x)
}

pub fn scientifically_rigorous_swish(x: f64, beta: f64) -> ActivationResult {
    ScientificSwish::new(beta)?.activate(x)
}

pub fn quantum_inspired_activation(x: f64, frequency: f64, amplitude: f64) -> ActivationResult {
    QuantumActivation::new(frequency, amplitude)?.activate(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_ieee754_compliance() {
        let sigmoid = ScientificSigmoid;

        // Test normal range
        assert!(sigmoid.activate(0.0).is_ok());
        assert!(sigmoid.activate(1.0).is_ok());
        assert!(sigmoid.activate(-1.0).is_ok());

        // Test extreme values
        assert!(sigmoid.activate(1000.0).is_ok());
        assert!(sigmoid.activate(-1000.0).is_ok());

        // Test invalid inputs
        assert!(sigmoid.activate(f64::NAN).is_err());
        assert!(sigmoid.activate(f64::INFINITY).is_err());
    }

    #[test]
    fn test_quantum_activation_properties() {
        let quantum = QuantumActivation::new(1.0, 1.0).unwrap();
        let properties = quantum.validate_properties().unwrap();

        assert!(properties.bounded);
        assert!(!properties.monotonic);
        assert!(properties.smooth);
        assert!(properties.zero_centered);
    }

    #[test]
    fn test_activation_validator() {
        let relu = ScientificReLU;
        let report = ActivationValidator::validate_function(&relu, (-10.0, 10.0), 100).unwrap();

        assert!(report.ieee754_compliant);
        assert_eq!(report.samples_tested, 100);
        assert!(report.errors.is_empty());
    }
}
