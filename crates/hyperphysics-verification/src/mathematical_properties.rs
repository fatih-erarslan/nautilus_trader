//! Mathematical properties and theorems for formal verification
//!
//! This module defines the mathematical properties that must be verified
//! for the HyperPhysics system to be considered mathematically sound.

use serde::{Serialize, Deserialize};

/// Mathematical property definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathematicalProperty {
    pub name: String,
    pub category: PropertyCategory,
    pub description: String,
    pub formal_statement: String,
    pub verification_methods: Vec<VerificationMethod>,
    pub criticality: CriticalityLevel,
    pub dependencies: Vec<String>,
}

/// Categories of mathematical properties
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PropertyCategory {
    HyperbolicGeometry,
    ProbabilityTheory,
    Thermodynamics,
    ConsciousnessMetrics,
    StochasticProcesses,
    InformationTheory,
}

/// Verification methods available
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VerificationMethod {
    Z3SMT,
    Lean4Proof,
    PropertyBasedTesting,
    RuntimeInvariant,
    MathematicalProof,
}

/// Criticality levels for properties
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CriticalityLevel {
    Critical,    // System cannot function without this
    Important,   // Significant impact on correctness
    Desirable,   // Nice to have but not essential
}

/// Get all mathematical properties that must be verified
pub fn get_all_properties() -> Vec<MathematicalProperty> {
    vec![
        // Hyperbolic Geometry Properties
        MathematicalProperty {
            name: "hyperbolic_triangle_inequality".to_string(),
            category: PropertyCategory::HyperbolicGeometry,
            description: "Triangle inequality holds in hyperbolic space".to_string(),
            formal_statement: "∀p,q,r ∈ H³: d_H(p,q) ≤ d_H(p,r) + d_H(r,q)".to_string(),
            verification_methods: vec![
                VerificationMethod::Z3SMT,
                VerificationMethod::PropertyBasedTesting,
                VerificationMethod::RuntimeInvariant,
            ],
            criticality: CriticalityLevel::Critical,
            dependencies: vec!["hyperbolic_distance_definition".to_string()],
        },
        
        MathematicalProperty {
            name: "hyperbolic_distance_positivity".to_string(),
            category: PropertyCategory::HyperbolicGeometry,
            description: "Hyperbolic distance is always non-negative".to_string(),
            formal_statement: "∀p,q ∈ H³: d_H(p,q) ≥ 0 ∧ (d_H(p,q) = 0 ⟺ p = q)".to_string(),
            verification_methods: vec![
                VerificationMethod::Z3SMT,
                VerificationMethod::PropertyBasedTesting,
                VerificationMethod::RuntimeInvariant,
            ],
            criticality: CriticalityLevel::Critical,
            dependencies: vec![],
        },
        
        MathematicalProperty {
            name: "hyperbolic_distance_symmetry".to_string(),
            category: PropertyCategory::HyperbolicGeometry,
            description: "Hyperbolic distance is symmetric".to_string(),
            formal_statement: "∀p,q ∈ H³: d_H(p,q) = d_H(q,p)".to_string(),
            verification_methods: vec![
                VerificationMethod::Z3SMT,
                VerificationMethod::PropertyBasedTesting,
                VerificationMethod::RuntimeInvariant,
            ],
            criticality: CriticalityLevel::Critical,
            dependencies: vec![],
        },
        
        MathematicalProperty {
            name: "poincare_disk_bounds".to_string(),
            category: PropertyCategory::HyperbolicGeometry,
            description: "All points in Poincare disk have norm < 1".to_string(),
            formal_statement: "∀p ∈ D³: ||p|| < 1".to_string(),
            verification_methods: vec![
                VerificationMethod::Z3SMT,
                VerificationMethod::RuntimeInvariant,
            ],
            criticality: CriticalityLevel::Critical,
            dependencies: vec![],
        },
        
        // Probability Theory Properties
        MathematicalProperty {
            name: "probability_bounds".to_string(),
            category: PropertyCategory::ProbabilityTheory,
            description: "All probabilities are bounded between 0 and 1".to_string(),
            formal_statement: "∀P: 0 ≤ P ≤ 1".to_string(),
            verification_methods: vec![
                VerificationMethod::Z3SMT,
                VerificationMethod::PropertyBasedTesting,
                VerificationMethod::RuntimeInvariant,
            ],
            criticality: CriticalityLevel::Critical,
            dependencies: vec![],
        },
        
        MathematicalProperty {
            name: "sigmoid_monotonicity".to_string(),
            category: PropertyCategory::ProbabilityTheory,
            description: "Sigmoid function is monotonically increasing".to_string(),
            formal_statement: "∀x₁,x₂,T>0: x₁ < x₂ ⟹ σ(x₁/T) < σ(x₂/T)".to_string(),
            verification_methods: vec![
                VerificationMethod::Z3SMT,
                VerificationMethod::PropertyBasedTesting,
            ],
            criticality: CriticalityLevel::Important,
            dependencies: vec!["probability_bounds".to_string()],
        },
        
        MathematicalProperty {
            name: "boltzmann_normalization".to_string(),
            category: PropertyCategory::ProbabilityTheory,
            description: "Boltzmann distribution is properly normalized".to_string(),
            formal_statement: "∑_s P(s) = 1 where P(s) = exp(-E(s)/kT)/Z".to_string(),
            verification_methods: vec![
                VerificationMethod::Z3SMT,
                VerificationMethod::PropertyBasedTesting,
            ],
            criticality: CriticalityLevel::Critical,
            dependencies: vec!["probability_bounds".to_string()],
        },
        
        // Thermodynamic Properties
        MathematicalProperty {
            name: "energy_conservation".to_string(),
            category: PropertyCategory::Thermodynamics,
            description: "Energy is conserved in isolated systems".to_string(),
            formal_statement: "∀isolated system: dE/dt = 0".to_string(),
            verification_methods: vec![
                VerificationMethod::Z3SMT,
                VerificationMethod::PropertyBasedTesting,
                VerificationMethod::RuntimeInvariant,
            ],
            criticality: CriticalityLevel::Critical,
            dependencies: vec![],
        },
        
        MathematicalProperty {
            name: "entropy_monotonicity".to_string(),
            category: PropertyCategory::Thermodynamics,
            description: "Entropy never decreases in isolated systems".to_string(),
            formal_statement: "∀isolated system: dS/dt ≥ 0".to_string(),
            verification_methods: vec![
                VerificationMethod::Z3SMT,
                VerificationMethod::PropertyBasedTesting,
                VerificationMethod::RuntimeInvariant,
            ],
            criticality: CriticalityLevel::Critical,
            dependencies: vec![],
        },
        
        MathematicalProperty {
            name: "landauer_bound".to_string(),
            category: PropertyCategory::Thermodynamics,
            description: "Information erasure requires minimum energy".to_string(),
            formal_statement: "E_erasure ≥ k_B T ln(2) per bit".to_string(),
            verification_methods: vec![
                VerificationMethod::Z3SMT,
                VerificationMethod::PropertyBasedTesting,
                VerificationMethod::RuntimeInvariant,
            ],
            criticality: CriticalityLevel::Critical,
            dependencies: vec![],
        },
        
        // Consciousness Metrics Properties
        MathematicalProperty {
            name: "phi_nonnegativity".to_string(),
            category: PropertyCategory::ConsciousnessMetrics,
            description: "Integrated information Φ is always non-negative".to_string(),
            formal_statement: "∀system S: Φ(S) ≥ 0".to_string(),
            verification_methods: vec![
                VerificationMethod::Z3SMT,
                VerificationMethod::PropertyBasedTesting,
                VerificationMethod::RuntimeInvariant,
            ],
            criticality: CriticalityLevel::Critical,
            dependencies: vec![],
        },
        
        MathematicalProperty {
            name: "iit_axiom_intrinsic_existence".to_string(),
            category: PropertyCategory::ConsciousnessMetrics,
            description: "IIT Axiom: Intrinsic existence".to_string(),
            formal_statement: "Φ > 0 ⟹ ∃ conscious experience".to_string(),
            verification_methods: vec![
                VerificationMethod::Lean4Proof,
                VerificationMethod::MathematicalProof,
            ],
            criticality: CriticalityLevel::Important,
            dependencies: vec!["phi_nonnegativity".to_string()],
        },
        
        MathematicalProperty {
            name: "iit_axiom_composition".to_string(),
            category: PropertyCategory::ConsciousnessMetrics,
            description: "IIT Axiom: Composition".to_string(),
            formal_statement: "System has parts with their own Φ values".to_string(),
            verification_methods: vec![
                VerificationMethod::Lean4Proof,
                VerificationMethod::PropertyBasedTesting,
            ],
            criticality: CriticalityLevel::Important,
            dependencies: vec!["phi_nonnegativity".to_string()],
        },
        
        MathematicalProperty {
            name: "iit_axiom_information".to_string(),
            category: PropertyCategory::ConsciousnessMetrics,
            description: "IIT Axiom: Information".to_string(),
            formal_statement: "System specifies particular state".to_string(),
            verification_methods: vec![
                VerificationMethod::Lean4Proof,
                VerificationMethod::PropertyBasedTesting,
            ],
            criticality: CriticalityLevel::Important,
            dependencies: vec!["phi_nonnegativity".to_string()],
        },
        
        MathematicalProperty {
            name: "iit_axiom_integration".to_string(),
            category: PropertyCategory::ConsciousnessMetrics,
            description: "IIT Axiom: Integration".to_string(),
            formal_statement: "System is irreducible (not separable)".to_string(),
            verification_methods: vec![
                VerificationMethod::Lean4Proof,
                VerificationMethod::PropertyBasedTesting,
            ],
            criticality: CriticalityLevel::Important,
            dependencies: vec!["phi_nonnegativity".to_string()],
        },
        
        MathematicalProperty {
            name: "iit_axiom_exclusion".to_string(),
            category: PropertyCategory::ConsciousnessMetrics,
            description: "IIT Axiom: Exclusion".to_string(),
            formal_statement: "Only maximal Φ matters (unique)".to_string(),
            verification_methods: vec![
                VerificationMethod::Lean4Proof,
                VerificationMethod::PropertyBasedTesting,
            ],
            criticality: CriticalityLevel::Important,
            dependencies: vec!["phi_nonnegativity".to_string()],
        },
        
        // Stochastic Process Properties
        MathematicalProperty {
            name: "gillespie_detailed_balance".to_string(),
            category: PropertyCategory::StochasticProcesses,
            description: "Gillespie algorithm satisfies detailed balance".to_string(),
            formal_statement: "P(i→j)P_eq(i) = P(j→i)P_eq(j)".to_string(),
            verification_methods: vec![
                VerificationMethod::PropertyBasedTesting,
                VerificationMethod::MathematicalProof,
            ],
            criticality: CriticalityLevel::Critical,
            dependencies: vec!["boltzmann_normalization".to_string()],
        },
        
        MathematicalProperty {
            name: "metropolis_acceptance_bounds".to_string(),
            category: PropertyCategory::StochasticProcesses,
            description: "Metropolis acceptance probability is bounded [0,1]".to_string(),
            formal_statement: "∀ΔE,T: 0 ≤ min(1, exp(-ΔE/kT)) ≤ 1".to_string(),
            verification_methods: vec![
                VerificationMethod::Z3SMT,
                VerificationMethod::PropertyBasedTesting,
                VerificationMethod::RuntimeInvariant,
            ],
            criticality: CriticalityLevel::Critical,
            dependencies: vec!["probability_bounds".to_string()],
        },
        
        // Information Theory Properties
        MathematicalProperty {
            name: "mutual_information_nonnegativity".to_string(),
            category: PropertyCategory::InformationTheory,
            description: "Mutual information is always non-negative".to_string(),
            formal_statement: "∀X,Y: I(X;Y) ≥ 0".to_string(),
            verification_methods: vec![
                VerificationMethod::Z3SMT,
                VerificationMethod::PropertyBasedTesting,
            ],
            criticality: CriticalityLevel::Important,
            dependencies: vec![],
        },
        
        MathematicalProperty {
            name: "entropy_subadditivity".to_string(),
            category: PropertyCategory::InformationTheory,
            description: "Joint entropy is subadditive".to_string(),
            formal_statement: "∀X,Y: H(X,Y) ≤ H(X) + H(Y)".to_string(),
            verification_methods: vec![
                VerificationMethod::Z3SMT,
                VerificationMethod::PropertyBasedTesting,
            ],
            criticality: CriticalityLevel::Important,
            dependencies: vec![],
        },
    ]
}

/// Get properties by category
pub fn get_properties_by_category(category: PropertyCategory) -> Vec<MathematicalProperty> {
    get_all_properties()
        .into_iter()
        .filter(|prop| prop.category == category)
        .collect()
}

/// Get critical properties only
pub fn get_critical_properties() -> Vec<MathematicalProperty> {
    get_all_properties()
        .into_iter()
        .filter(|prop| prop.criticality == CriticalityLevel::Critical)
        .collect()
}

/// Get properties that can be verified with a specific method
pub fn get_properties_by_method(method: VerificationMethod) -> Vec<MathematicalProperty> {
    get_all_properties()
        .into_iter()
        .filter(|prop| prop.verification_methods.contains(&method))
        .collect()
}

/// Verify property dependencies are satisfied
pub fn check_dependencies(properties: &[MathematicalProperty]) -> Vec<String> {
    let mut missing_deps = Vec::new();
    let property_names: std::collections::HashSet<String> = 
        properties.iter().map(|p| p.name.clone()).collect();
    
    for prop in properties {
        for dep in &prop.dependencies {
            if !property_names.contains(dep) {
                missing_deps.push(format!("Property '{}' depends on missing '{}'", prop.name, dep));
            }
        }
    }
    
    missing_deps
}

/// Generate verification plan
pub fn generate_verification_plan() -> VerificationPlan {
    let all_properties = get_all_properties();
    let critical_properties = get_critical_properties();
    
    VerificationPlan {
        total_properties: all_properties.len(),
        critical_properties: critical_properties.len(),
        z3_properties: get_properties_by_method(VerificationMethod::Z3SMT).len(),
        lean4_properties: get_properties_by_method(VerificationMethod::Lean4Proof).len(),
        property_test_properties: get_properties_by_method(VerificationMethod::PropertyBasedTesting).len(),
        runtime_properties: get_properties_by_method(VerificationMethod::RuntimeInvariant).len(),
        dependency_violations: check_dependencies(&all_properties),
        properties: all_properties,
    }
}

/// Verification plan summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationPlan {
    pub total_properties: usize,
    pub critical_properties: usize,
    pub z3_properties: usize,
    pub lean4_properties: usize,
    pub property_test_properties: usize,
    pub runtime_properties: usize,
    pub dependency_violations: Vec<String>,
    pub properties: Vec<MathematicalProperty>,
}

impl VerificationPlan {
    /// Check if verification plan is complete
    pub fn is_complete(&self) -> bool {
        self.dependency_violations.is_empty() && 
        self.critical_properties > 0 &&
        self.z3_properties > 0 &&
        self.property_test_properties > 0
    }
    
    /// Get verification coverage statistics
    pub fn get_coverage_stats(&self) -> VerificationCoverage {
        let total = self.total_properties as f64;
        
        VerificationCoverage {
            z3_coverage: (self.z3_properties as f64 / total) * 100.0,
            lean4_coverage: (self.lean4_properties as f64 / total) * 100.0,
            property_test_coverage: (self.property_test_properties as f64 / total) * 100.0,
            runtime_coverage: (self.runtime_properties as f64 / total) * 100.0,
            critical_coverage: (self.critical_properties as f64 / total) * 100.0,
        }
    }
}

/// Verification coverage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationCoverage {
    pub z3_coverage: f64,
    pub lean4_coverage: f64,
    pub property_test_coverage: f64,
    pub runtime_coverage: f64,
    pub critical_coverage: f64,
}
