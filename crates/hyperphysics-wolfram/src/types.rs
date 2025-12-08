//! Types for HyperPhysics Wolfram Bridge

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during Wolfram operations
#[derive(Error, Debug)]
pub enum WolframError {
    /// No Wolfram installation found on the system
    #[error("No Wolfram installation found. Install WolframScript.app or Mathematica.")]
    NoInstallation,

    /// WolframScript executable not found
    #[error("WolframScript not found at expected path")]
    WolframScriptNotFound,

    /// Execution of WolframScript failed
    #[error("Failed to execute WolframScript: {0}")]
    ExecutionFailed(String),

    /// Evaluation timed out
    #[error("Evaluation timed out after {0} seconds")]
    Timeout(u64),

    /// Failed to parse Wolfram output
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Wolfram kernel reported an error
    #[error("Kernel error: {0}")]
    KernelError(String),

    /// Validation failed
    #[error("Validation failed: {0}")]
    ValidationFailed(String),

    /// Research query failed
    #[error("Research query failed: {0}")]
    ResearchFailed(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

/// Result type for Wolfram operations
pub type WolframResult<T> = Result<T, WolframError>;

/// Information about a Wolfram installation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WolframInstallation {
    /// Path to the installation directory
    pub installation_directory: String,

    /// Path to WolframScript executable
    pub wolfram_script_path: String,

    /// Path to WolframKernel executable
    pub kernel_path: String,

    /// Product name (e.g., "Wolfram Desktop", "Mathematica", "WolframScript.app")
    pub product_name: String,

    /// Version string
    pub version: String,

    /// Whether the installation appears valid
    pub is_valid: bool,
}

/// Options for Wolfram evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationOptions {
    /// Timeout in seconds (default: 30)
    #[serde(default = "default_timeout")]
    pub timeout_seconds: i64,

    /// Return format: "json", "text", "inputform", "fullform"
    #[serde(default = "default_format")]
    pub format: String,

    /// Whether to capture messages
    #[serde(default)]
    pub capture_messages: bool,

    /// Cloud evaluation (requires authentication)
    #[serde(default)]
    pub use_cloud: bool,

    /// Use Code Assistant (Opus 4.5) for enhanced results
    #[serde(default)]
    pub use_code_assistant: bool,
}

impl Default for EvaluationOptions {
    fn default() -> Self {
        Self {
            timeout_seconds: default_timeout(),
            format: default_format(),
            capture_messages: false,
            use_cloud: false,
            use_code_assistant: false,
        }
    }
}

fn default_timeout() -> i64 {
    30
}

fn default_format() -> String {
    "json".to_string()
}

/// Result of an evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    /// The result value (as JSON string or text)
    pub result: String,

    /// Whether the evaluation succeeded
    pub success: bool,

    /// Error message if failed
    pub error: Option<String>,

    /// Messages generated during evaluation
    pub messages: Vec<String>,

    /// Execution time in milliseconds
    pub execution_time_ms: i64,

    /// Format of the result
    pub format: String,
}

/// Result of mathematical validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether the validation passed
    pub is_valid: bool,

    /// Numerical error if applicable
    pub numerical_error: Option<f64>,

    /// Expected value from Wolfram
    pub expected_value: Option<String>,

    /// Actual value being validated
    pub actual_value: Option<String>,

    /// Detailed validation message
    pub message: String,

    /// Validation method used
    pub method: ValidationMethod,

    /// Time taken for validation in milliseconds
    pub validation_time_ms: i64,
}

/// Method used for validation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ValidationMethod {
    /// Symbolic comparison
    Symbolic,
    /// Numerical comparison within tolerance
    Numerical,
    /// Formal proof verification
    FormalProof,
    /// Statistical validation
    Statistical,
    /// Property-based testing
    PropertyBased,
}

/// Result of a formal proof verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormalProofResult {
    /// Whether the proof is valid
    pub is_proven: bool,

    /// The theorem or property being proven
    pub theorem: String,

    /// Proof steps (if available)
    pub proof_steps: Vec<String>,

    /// Counterexample if proof failed
    pub counterexample: Option<String>,

    /// Assumptions used in the proof
    pub assumptions: Vec<String>,

    /// Proof method used
    pub proof_method: String,
}

/// Research result from Wolfram
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchResult {
    /// Query that was executed
    pub query: String,

    /// Summary of findings
    pub summary: String,

    /// Relevant equations/formulas
    pub equations: Vec<String>,

    /// Related topics
    pub related_topics: Vec<String>,

    /// Source references
    pub references: Vec<String>,

    /// Wolfram Alpha integration results
    pub wolfram_alpha_data: Option<String>,
}

/// Hyperbolic geometry computation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperbolicGeometryResult {
    /// Coordinates in Poincare disk model
    pub poincare_coords: Vec<Vec<f64>>,

    /// Geodesic distances between points
    pub geodesic_distances: Vec<Vec<f64>>,

    /// Curvature value
    pub curvature: f64,

    /// Tiling parameter p (number of sides per polygon)
    pub tiling_p: i32,
    /// Tiling parameter q (number of polygons meeting at vertex)
    pub tiling_q: i32,
}

/// Conformal prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformalPredictionResult {
    /// Point prediction
    pub prediction: f64,

    /// Lower bound of prediction interval
    pub lower_bound: f64,

    /// Upper bound of prediction interval
    pub upper_bound: f64,

    /// Coverage level (e.g., 0.9 for 90%)
    pub coverage: f64,

    /// Conformity score
    pub conformity_score: f64,

    /// P-value for the prediction
    pub p_value: f64,
}

/// Graph analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphAnalysisResult {
    /// Node centrality scores
    pub centrality: Vec<f64>,

    /// Community assignments
    pub communities: Vec<i32>,

    /// Clustering coefficients
    pub clustering: Vec<f64>,

    /// Graph diameter
    pub diameter: i32,

    /// Average path length
    pub avg_path_length: f64,
}

/// IIT Phi computation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiComputationResult {
    /// Phi value (integrated information)
    pub phi: f64,

    /// Main complex partition
    pub main_complex: Vec<usize>,

    /// Cause-effect structure
    pub cause_effect_structure: Option<String>,

    /// Computation time in milliseconds
    pub computation_time_ms: i64,
}

/// Free energy computation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeEnergyResult {
    /// Free energy value
    pub free_energy: f64,

    /// Complexity term
    pub complexity: f64,

    /// Accuracy term
    pub accuracy: f64,

    /// KL divergence component
    pub kl_divergence: f64,
}

/// STDP validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct STDPValidationResult {
    /// Whether the STDP rule is correctly implemented
    pub is_valid: bool,

    /// Maximum weight change error
    pub max_error: f64,

    /// Test cases validated
    pub test_cases: usize,

    /// Failed test cases
    pub failed_cases: Vec<STDPTestCase>,
}

/// A single STDP test case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct STDPTestCase {
    /// Time difference (post - pre)
    pub delta_t: f64,

    /// Expected weight change
    pub expected_dw: f64,

    /// Actual weight change
    pub actual_dw: f64,

    /// Error
    pub error: f64,
}

/// Optimization suggestion from Wolfram
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    /// Type of optimization
    pub optimization_type: OptimizationType,

    /// Description of the optimization
    pub description: String,

    /// Estimated performance improvement
    pub estimated_improvement: Option<f64>,

    /// Code suggestion (if applicable)
    pub code_suggestion: Option<String>,

    /// Priority level
    pub priority: OptimizationPriority,
}

/// Type of optimization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum OptimizationType {
    /// Algorithmic optimization
    Algorithmic,
    /// Numerical stability improvement
    NumericalStability,
    /// Memory optimization
    Memory,
    /// Parallelization opportunity
    Parallelization,
    /// SIMD vectorization
    Vectorization,
    /// Mathematical simplification
    MathSimplification,
}

/// Priority of optimization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum OptimizationPriority {
    /// Low priority
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Critical priority
    Critical,
}
