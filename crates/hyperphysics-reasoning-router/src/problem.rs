//! Problem representation and signature extraction for routing.

use crate::{LatencyTier, ProblemDomain};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Classification of problem structure
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StructureType {
    /// Dense matrix operations
    Dense,
    /// Sparse matrix/graph operations
    Sparse,
    /// Sequential/time series data
    Sequential,
    /// Graph/network structure
    Graph,
    /// Hierarchical/tree structure
    Hierarchical,
    /// Unstructured/flat
    Unstructured,
}

/// Problem type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProblemType {
    /// Find optimal solution
    Optimization,
    /// Simulate physical system
    Simulation,
    /// Predict future values
    Prediction,
    /// Classify input
    Classification,
    /// Verify property holds
    Verification,
    /// Control system
    Control,
    /// General computation
    General,
}

/// Problem signature for routing decisions
///
/// This is the feature vector used by the router to select backends.
/// It captures essential characteristics without requiring full problem analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemSignature {
    /// Problem type
    pub problem_type: ProblemType,
    /// Primary domain
    pub domain: ProblemDomain,
    /// Dimensionality of the problem (number of variables)
    pub dimensionality: u32,
    /// Sparsity ratio [0, 1] where 0=dense, 1=maximally sparse
    pub sparsity: f32,
    /// Required latency tier
    pub latency_budget: LatencyTier,
    /// Structure type
    pub structure: StructureType,
    /// Is the problem stochastic?
    pub is_stochastic: bool,
    /// Does it require gradients?
    pub needs_gradients: bool,
    /// Number of constraints
    pub constraint_count: u32,
    /// Is multi-objective?
    pub is_multi_objective: bool,
    /// Estimated computational complexity (normalized 0-1)
    pub complexity_estimate: f32,
    /// Optional domain-specific hints
    pub hints: HashMap<String, f64>,
}

impl ProblemSignature {
    /// Create a new problem signature with defaults
    pub fn new(problem_type: ProblemType, domain: ProblemDomain) -> Self {
        Self {
            problem_type,
            domain,
            dimensionality: 1,
            sparsity: 0.0,
            latency_budget: LatencyTier::Medium,
            structure: StructureType::Unstructured,
            is_stochastic: false,
            needs_gradients: false,
            constraint_count: 0,
            is_multi_objective: false,
            complexity_estimate: 0.5,
            hints: HashMap::new(),
        }
    }

    /// Builder pattern: set dimensionality
    pub fn with_dimensionality(mut self, dim: u32) -> Self {
        self.dimensionality = dim;
        self
    }

    /// Builder pattern: set sparsity
    pub fn with_sparsity(mut self, sparsity: f32) -> Self {
        self.sparsity = sparsity.clamp(0.0, 1.0);
        self
    }

    /// Builder pattern: set latency budget
    pub fn with_latency_budget(mut self, tier: LatencyTier) -> Self {
        self.latency_budget = tier;
        self
    }

    /// Builder pattern: set structure
    pub fn with_structure(mut self, structure: StructureType) -> Self {
        self.structure = structure;
        self
    }

    /// Builder pattern: mark as stochastic
    pub fn stochastic(mut self) -> Self {
        self.is_stochastic = true;
        self
    }

    /// Builder pattern: requires gradients
    pub fn with_gradients(mut self) -> Self {
        self.needs_gradients = true;
        self
    }

    /// Builder pattern: set constraint count
    pub fn with_constraints(mut self, count: u32) -> Self {
        self.constraint_count = count;
        self
    }

    /// Builder pattern: mark as multi-objective
    pub fn multi_objective(mut self) -> Self {
        self.is_multi_objective = true;
        self
    }

    /// Builder pattern: add hint
    pub fn with_hint(mut self, key: impl Into<String>, value: f64) -> Self {
        self.hints.insert(key.into(), value);
        self
    }

    /// Convert signature to LSH-compatible feature vector
    ///
    /// Returns a normalized feature vector suitable for LSH hashing.
    /// Dimensionality: 16 features
    pub fn to_feature_vector(&self) -> [f32; 16] {
        let mut features = [0.0f32; 16];

        // Problem type (one-hot encoded, 7 slots)
        features[self.problem_type as usize] = 1.0;

        // Domain (normalized)
        features[7] = (self.domain as u8) as f32 / 6.0;

        // Dimensionality (log-scaled, normalized)
        features[8] = (self.dimensionality as f32 + 1.0).log10() / 6.0; // Up to 10^6

        // Sparsity
        features[9] = self.sparsity;

        // Latency tier (normalized)
        features[10] = (self.latency_budget as u8) as f32 / 4.0;

        // Structure type (normalized)
        features[11] = (self.structure as u8) as f32 / 5.0;

        // Boolean flags
        features[12] = if self.is_stochastic { 1.0 } else { 0.0 };
        features[13] = if self.needs_gradients { 1.0 } else { 0.0 };
        features[14] = if self.is_multi_objective { 1.0 } else { 0.0 };

        // Complexity estimate
        features[15] = self.complexity_estimate;

        features
    }

    /// Compute similarity to another signature (cosine similarity)
    pub fn similarity(&self, other: &ProblemSignature) -> f32 {
        let v1 = self.to_feature_vector();
        let v2 = other.to_feature_vector();

        let dot: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm1 > 0.0 && norm2 > 0.0 {
            dot / (norm1 * norm2)
        } else {
            0.0
        }
    }
}

/// A complete problem to be solved
#[derive(Debug, Clone)]
pub struct Problem {
    /// Unique problem ID
    pub id: String,
    /// Problem signature for routing
    pub signature: ProblemSignature,
    /// Problem data (backend-specific interpretation)
    pub data: ProblemData,
    /// Optional objective function for optimization
    pub objective: Option<ObjectiveSpec>,
    /// Optional constraints
    pub constraints: Vec<ConstraintSpec>,
    /// Metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Problem {
    /// Create a new problem
    pub fn new(signature: ProblemSignature, data: ProblemData) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            signature,
            data,
            objective: None,
            constraints: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add objective specification
    pub fn with_objective(mut self, objective: ObjectiveSpec) -> Self {
        self.objective = Some(objective);
        self
    }

    /// Add constraint
    pub fn with_constraint(mut self, constraint: ConstraintSpec) -> Self {
        self.constraints.push(constraint);
        self.signature.constraint_count = self.constraints.len() as u32;
        self
    }
}

/// Problem data variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProblemData {
    /// Vector data (e.g., optimization variables)
    Vector(Vec<f64>),
    /// Matrix data
    Matrix {
        /// Flattened row-major data
        data: Vec<f64>,
        /// Number of rows
        rows: usize,
        /// Number of columns
        cols: usize,
    },
    /// Sparse matrix (COO format)
    SparseMatrix {
        /// Row indices
        row_indices: Vec<usize>,
        /// Column indices
        col_indices: Vec<usize>,
        /// Values
        values: Vec<f64>,
        /// Matrix shape
        shape: (usize, usize),
    },
    /// Time series data
    TimeSeries {
        /// Timestamps (Unix epoch seconds)
        timestamps: Vec<f64>,
        /// Values
        values: Vec<f64>,
    },
    /// Graph data
    Graph {
        /// Number of nodes
        num_nodes: usize,
        /// Edges as (from, to, weight)
        edges: Vec<(usize, usize, f64)>,
    },
    /// Physics simulation initial state
    PhysicsState {
        /// Body positions (flattened 3D)
        positions: Vec<f64>,
        /// Body velocities (flattened 3D)
        velocities: Vec<f64>,
        /// Body masses
        masses: Vec<f64>,
    },
    /// Structured JSON data
    Json(serde_json::Value),
}

/// Objective function specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveSpec {
    /// Objective type
    pub objective_type: ObjectiveType,
    /// Bounds for variables: (min, max) pairs
    pub bounds: Vec<(f64, f64)>,
    /// Is this a maximization problem?
    pub maximize: bool,
}

/// Types of objective functions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObjectiveType {
    /// Black-box function
    BlackBox,
    /// Quadratic function
    Quadratic,
    /// Linear function
    Linear,
    /// Neural network evaluation
    Neural,
    /// Physics simulation
    Simulation,
}

/// Constraint specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintSpec {
    /// Linear constraint: coeffs · x <= bound
    Linear {
        /// Coefficients
        coefficients: Vec<f64>,
        /// Upper bound
        bound: f64,
    },
    /// Box constraint: bounds per variable
    Box {
        /// Variable indices
        variables: Vec<usize>,
        /// Bounds (min, max)
        bounds: (f64, f64),
    },
    /// Equality constraint: f(x) = target
    Equality {
        /// Target value
        target: f64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signature_creation() {
        let sig = ProblemSignature::new(ProblemType::Optimization, ProblemDomain::Financial)
            .with_dimensionality(100)
            .with_sparsity(0.8)
            .with_latency_budget(LatencyTier::Fast)
            .with_constraints(5)
            .stochastic();

        assert_eq!(sig.problem_type, ProblemType::Optimization);
        assert_eq!(sig.domain, ProblemDomain::Financial);
        assert_eq!(sig.dimensionality, 100);
        assert!((sig.sparsity - 0.8).abs() < 0.001);
        assert!(sig.is_stochastic);
        assert_eq!(sig.constraint_count, 5);
    }

    #[test]
    fn test_feature_vector() {
        let sig = ProblemSignature::new(ProblemType::Optimization, ProblemDomain::Physics);
        let features = sig.to_feature_vector();

        assert_eq!(features.len(), 16);
        assert!(features[0] > 0.9); // Optimization flag
    }

    #[test]
    fn test_signature_similarity() {
        let sig1 = ProblemSignature::new(ProblemType::Optimization, ProblemDomain::Financial)
            .with_dimensionality(100);
        let sig2 = ProblemSignature::new(ProblemType::Optimization, ProblemDomain::Financial)
            .with_dimensionality(150);
        let sig3 = ProblemSignature::new(ProblemType::Simulation, ProblemDomain::Physics)
            .with_dimensionality(100);

        let sim_same = sig1.similarity(&sig2);
        let sim_diff = sig1.similarity(&sig3);

        // Similar problems should have higher similarity
        assert!(sim_same > sim_diff);
        assert!(sim_same > 0.9);
    }

    #[test]
    fn test_problem_data_variants() {
        let vec_data = ProblemData::Vector(vec![1.0, 2.0, 3.0]);
        let matrix_data = ProblemData::Matrix {
            data: vec![1.0, 2.0, 3.0, 4.0],
            rows: 2,
            cols: 2,
        };

        // Ensure serialization works
        let _ = serde_json::to_string(&vec_data).unwrap();
        let _ = serde_json::to_string(&matrix_data).unwrap();
    }
}
