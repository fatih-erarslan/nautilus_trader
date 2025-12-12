//! # MCP Tool Definitions
//!
//! Model Context Protocol tool registry for exposing 49 CQGS sentinels
//! as MCP tools with Dilithium authentication.
//!
//! ## Tool Categories
//!
//! - **Sentinel Execution**: Execute individual or all sentinels
//! - **Quality Analysis**: Calculate quality scores and check gates
//! - **Authentication**: Dilithium key generation, signing, verification
//! - **Hyperbolic Geometry**: H^11 computations
//! - **Symbolic Math**: Expression evaluation and calculus

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;

use crate::{CqgsMcpError, sentinels::*};

#[cfg(feature = "dilithium")]
use crate::dilithium_auth::{DilithiumAuth, AuthToken};

// ============================================================================
// MCP Tool Definition
// ============================================================================

/// MCP tool definition following dilithium-mcp pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool name
    pub name: String,

    /// Tool description
    pub description: String,

    /// Input schema (JSON Schema)
    pub input_schema: JsonValue,

    /// Required capabilities
    pub required_capabilities: Vec<String>,
}

/// MCP tool response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResponse {
    /// Response content
    pub content: Vec<ToolContent>,

    /// Is error
    #[serde(rename = "isError", skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

/// MCP tool content block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolContent {
    /// Content type
    #[serde(rename = "type")]
    pub content_type: String,

    /// Text content
    pub text: String,
}

impl ToolResponse {
    /// Create success response
    pub fn success(text: String) -> Self {
        Self {
            content: vec![ToolContent {
                content_type: "text".to_string(),
                text,
            }],
            is_error: None,
        }
    }

    /// Create error response
    pub fn error(error: String) -> Self {
        Self {
            content: vec![ToolContent {
                content_type: "text".to_string(),
                text: error,
            }],
            is_error: Some(true),
        }
    }

    /// Create JSON response
    pub fn json(value: JsonValue) -> Self {
        Self {
            content: vec![ToolContent {
                content_type: "text".to_string(),
                text: serde_json::to_string_pretty(&value).unwrap_or_default(),
            }],
            is_error: None,
        }
    }
}

// ============================================================================
// MCP Tool Registry
// ============================================================================

/// Registry of all CQGS MCP tools
pub struct McpToolRegistry {
    /// Registered tools
    tools: HashMap<String, ToolDefinition>,

    /// Sentinel executor
    executor: SentinelExecutor,

    /// Authentication system
    #[cfg(feature = "dilithium")]
    auth: DilithiumAuth,
}

impl McpToolRegistry {
    /// Create new tool registry
    pub fn new() -> Result<Self> {
        let mut registry = Self {
            tools: HashMap::new(),
            executor: SentinelExecutor::new(),
            #[cfg(feature = "dilithium")]
            auth: DilithiumAuth::new()?,
        };

        registry.register_tools()?;

        Ok(registry)
    }

    /// Register all tools
    fn register_tools(&mut self) -> Result<()> {
        // Sentinel execution tools
        self.register_sentinel_tools()?;

        // Authentication tools
        #[cfg(feature = "dilithium")]
        self.register_auth_tools()?;

        // Hyperbolic geometry tools
        #[cfg(feature = "hyperbolic")]
        self.register_hyperbolic_tools()?;

        // Symbolic math tools
        #[cfg(feature = "symbolic")]
        self.register_symbolic_tools()?;

        // Fibonacci threshold tools
        self.register_fibonacci_tools()?;

        Ok(())
    }

    /// Register sentinel execution tools
    fn register_sentinel_tools(&mut self) -> Result<()> {
        // Execute all sentinels
        self.tools.insert(
            "sentinel_execute_all".to_string(),
            ToolDefinition {
                name: "sentinel_execute_all".to_string(),
                description: "Execute all 49 CQGS sentinels on a codebase".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "codebase_path": {
                            "type": "string",
                            "description": "Path to codebase to analyze"
                        },
                        "parallel": {
                            "type": "boolean",
                            "description": "Enable parallel execution (default: true)"
                        }
                    },
                    "required": ["codebase_path"]
                }),
                required_capabilities: vec!["sentinel_execute".to_string()],
            },
        );

        // Calculate quality score
        self.tools.insert(
            "sentinel_quality_score".to_string(),
            ToolDefinition {
                name: "sentinel_quality_score".to_string(),
                description: "Calculate overall quality score from sentinel results".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "array",
                            "description": "Array of sentinel results"
                        }
                    },
                    "required": ["results"]
                }),
                required_capabilities: vec!["quality_analysis".to_string()],
            },
        );

        // Check quality gate
        self.tools.insert(
            "sentinel_quality_gate".to_string(),
            ToolDefinition {
                name: "sentinel_quality_gate".to_string(),
                description: "Check if results pass quality gate (GATE_1 through GATE_5)".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "array",
                            "description": "Array of sentinel results"
                        },
                        "gate": {
                            "type": "string",
                            "enum": ["NoForbiddenPatterns", "IntegrationReady", "TestingReady", "ProductionReady", "DeploymentApproved"],
                            "description": "Quality gate to check"
                        }
                    },
                    "required": ["results", "gate"]
                }),
                required_capabilities: vec!["quality_analysis".to_string()],
            },
        );

        Ok(())
    }

    /// Register authentication tools
    #[cfg(feature = "dilithium")]
    fn register_auth_tools(&mut self) -> Result<()> {
        // Generate Dilithium key pair
        self.tools.insert(
            "dilithium_keygen".to_string(),
            ToolDefinition {
                name: "dilithium_keygen".to_string(),
                description: "Generate Dilithium ML-DSA-65 key pair for authentication".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
                required_capabilities: vec![],
            },
        );

        // Register client
        self.tools.insert(
            "dilithium_register_client".to_string(),
            ToolDefinition {
                name: "dilithium_register_client".to_string(),
                description: "Register client with Dilithium authentication".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Client name"
                        },
                        "public_key": {
                            "type": "string",
                            "description": "Hex-encoded Dilithium public key"
                        },
                        "capabilities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Requested capabilities"
                        }
                    },
                    "required": ["name", "public_key"]
                }),
                required_capabilities: vec![],
            },
        );

        Ok(())
    }

    /// Register hyperbolic geometry tools
    #[cfg(feature = "hyperbolic")]
    fn register_hyperbolic_tools(&mut self) -> Result<()> {
        use serde_json::json;

        // Hyperbolic distance
        self.tools.insert(
            "hyperbolic_distance".to_string(),
            ToolDefinition {
                name: "hyperbolic_distance".to_string(),
                description: "Compute hyperbolic distance in H^11 (Lorentz model)".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "point1": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 12,
                            "maxItems": 12,
                            "description": "First point (12D Lorentz coordinates)"
                        },
                        "point2": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 12,
                            "maxItems": 12,
                            "description": "Second point (12D Lorentz coordinates)"
                        }
                    },
                    "required": ["point1", "point2"]
                }),
                required_capabilities: vec!["hyperbolic_compute".to_string()],
            },
        );

        Ok(())
    }

    /// Register symbolic math tools
    #[cfg(feature = "symbolic")]
    fn register_symbolic_tools(&mut self) -> Result<()> {
        use serde_json::json;

        // Shannon entropy
        self.tools.insert(
            "shannon_entropy".to_string(),
            ToolDefinition {
                name: "shannon_entropy".to_string(),
                description: "Compute Shannon entropy H(X) = -Σ p(x) log₂ p(x)".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "probabilities": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Probability distribution (must sum to 1.0)"
                        }
                    },
                    "required": ["probabilities"]
                }),
                required_capabilities: vec!["symbolic_compute".to_string()],
            },
        );

        Ok(())
    }

    /// Register Fibonacci threshold tools
    fn register_fibonacci_tools(&mut self) -> Result<()> {
        use serde_json::json;

        // Get all Fibonacci thresholds
        self.tools.insert(
            "fibonacci_get_thresholds".to_string(),
            ToolDefinition {
                name: "fibonacci_get_thresholds".to_string(),
                description: "Get all Fibonacci-scaled thresholds (F_5 through F_15, φ, φ⁻¹)".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": ["all", "technical_debt", "complexity", "file_size", "entropy"],
                            "description": "Threshold category to retrieve (default: all)"
                        }
                    }
                }),
                required_capabilities: vec!["fibonacci_constants".to_string()],
            },
        );

        // Calculate technical debt
        self.tools.insert(
            "fibonacci_calculate_debt".to_string(),
            ToolDefinition {
                name: "fibonacci_calculate_debt".to_string(),
                description: "Calculate technical debt minutes using Fibonacci-scaled thresholds".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "todo_count": {
                            "type": "integer",
                            "description": "Number of TODO markers"
                        },
                        "fixme_count": {
                            "type": "integer",
                            "description": "Number of FIXME markers"
                        },
                        "hack_count": {
                            "type": "integer",
                            "description": "Number of HACK markers"
                        },
                        "debug_artifacts": {
                            "type": "integer",
                            "description": "Number of debug artifacts (console.log, etc.)"
                        },
                        "complexity": {
                            "type": "number",
                            "description": "Cyclomatic complexity"
                        },
                        "lines_of_code": {
                            "type": "integer",
                            "description": "Total lines of code"
                        }
                    },
                    "required": ["todo_count", "fixme_count", "hack_count", "lines_of_code"]
                }),
                required_capabilities: vec!["fibonacci_calculate".to_string()],
            },
        );

        // Evaluate complexity
        self.tools.insert(
            "fibonacci_complexity_check".to_string(),
            ToolDefinition {
                name: "fibonacci_complexity_check".to_string(),
                description: "Evaluate code complexity against Fibonacci thresholds (F_6=8, F_7=13, F_8=21, F_9=34)".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "complexity": {
                            "type": "number",
                            "description": "Cyclomatic complexity value"
                        }
                    },
                    "required": ["complexity"]
                }),
                required_capabilities: vec!["fibonacci_evaluate".to_string()],
            },
        );

        // File size assessment
        self.tools.insert(
            "fibonacci_file_size_check".to_string(),
            ToolDefinition {
                name: "fibonacci_file_size_check".to_string(),
                description: "Assess file size against Fibonacci thresholds (F_12=144, F_13=233, F_14=377, F_15=610)".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "lines": {
                            "type": "integer",
                            "description": "Number of lines in file"
                        }
                    },
                    "required": ["lines"]
                }),
                required_capabilities: vec!["fibonacci_evaluate".to_string()],
            },
        );

        // Entropy assessment
        self.tools.insert(
            "fibonacci_entropy_check".to_string(),
            ToolDefinition {
                name: "fibonacci_entropy_check".to_string(),
                description: "Evaluate Shannon entropy against golden ratio thresholds (φ⁻¹=0.618 for synthetic, φ=1.618 for high entropy)".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "entropy": {
                            "type": "number",
                            "description": "Shannon entropy value in bits"
                        },
                        "probabilities": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Probability distribution (optional, will calculate entropy if provided)"
                        }
                    }
                }),
                required_capabilities: vec!["fibonacci_evaluate".to_string(), "symbolic_compute".to_string()],
            },
        );

        // Golden ratio constants
        self.tools.insert(
            "fibonacci_golden_ratio".to_string(),
            ToolDefinition {
                name: "fibonacci_golden_ratio".to_string(),
                description: "Get golden ratio constants (φ=1.618033988749895, φ⁻¹=0.618033988749895)".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "power": {
                            "type": "integer",
                            "description": "Calculate φ^n (optional)"
                        }
                    }
                }),
                required_capabilities: vec!["fibonacci_constants".to_string()],
            },
        );

        // Calculate Fibonacci number
        self.tools.insert(
            "fibonacci_calculate_number".to_string(),
            ToolDefinition {
                name: "fibonacci_calculate_number".to_string(),
                description: "Calculate Fibonacci number F_n using Binet's formula: F_n = round(φ^n / √5)".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "n": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Index n for F_n"
                        }
                    },
                    "required": ["n"]
                }),
                required_capabilities: vec!["fibonacci_calculate".to_string()],
            },
        );

        Ok(())
    }

    /// Execute tool by name
    pub fn execute_tool(&self, name: &str, args: JsonValue) -> Result<ToolResponse> {
        match name {
            "sentinel_execute_all" => self.execute_all_sentinels(args),
            "sentinel_quality_score" => self.calculate_quality_score(args),
            "sentinel_quality_gate" => self.check_quality_gate(args),

            #[cfg(feature = "dilithium")]
            "dilithium_keygen" => self.generate_keypair(args),

            #[cfg(feature = "dilithium")]
            "dilithium_register_client" => self.register_client(args),

            #[cfg(feature = "hyperbolic")]
            "hyperbolic_distance" => self.compute_hyperbolic_distance(args),

            #[cfg(feature = "symbolic")]
            "shannon_entropy" => self.compute_shannon_entropy(args),

            "fibonacci_get_thresholds" => self.get_fibonacci_thresholds(args),
            "fibonacci_calculate_debt" => self.calculate_technical_debt(args),
            "fibonacci_complexity_check" => self.check_complexity(args),
            "fibonacci_file_size_check" => self.check_file_size(args),
            "fibonacci_entropy_check" => self.check_entropy(args),
            "fibonacci_golden_ratio" => self.get_golden_ratio(args),
            "fibonacci_calculate_number" => self.calculate_fibonacci_number(args),

            _ => Ok(ToolResponse::error(format!("Unknown tool: {}", name))),
        }
    }

    /// Execute all sentinels
    fn execute_all_sentinels(&self, args: JsonValue) -> Result<ToolResponse> {
        let codebase_path = args["codebase_path"]
            .as_str()
            .ok_or_else(|| CqgsMcpError::Protocol("Missing codebase_path".to_string()))?;

        let results = self.executor.execute_all(codebase_path)?;

        Ok(ToolResponse::json(serde_json::to_value(&results)?))
    }

    /// Calculate quality score
    fn calculate_quality_score(&self, args: JsonValue) -> Result<ToolResponse> {
        let results: Vec<SentinelResult> = serde_json::from_value(args["results"].clone())?;

        let score = self.executor.calculate_overall_score(&results);

        Ok(ToolResponse::json(serde_json::json!({
            "quality_score": score,
            "sentinel_count": results.len(),
        })))
    }

    /// Check quality gate
    fn check_quality_gate(&self, args: JsonValue) -> Result<ToolResponse> {
        let results: Vec<SentinelResult> = serde_json::from_value(args["results"].clone())?;
        let gate_str = args["gate"]
            .as_str()
            .ok_or_else(|| CqgsMcpError::Protocol("Missing gate parameter".to_string()))?;

        let gate = match gate_str {
            "NoForbiddenPatterns" => QualityGate::NoForbiddenPatterns,
            "IntegrationReady" => QualityGate::IntegrationReady,
            "TestingReady" => QualityGate::TestingReady,
            "ProductionReady" => QualityGate::ProductionReady,
            "DeploymentApproved" => QualityGate::DeploymentApproved,
            _ => return Ok(ToolResponse::error(format!("Unknown gate: {}", gate_str))),
        };

        let passed = gate.check(&results);

        Ok(ToolResponse::json(serde_json::json!({
            "gate": gate_str,
            "passed": passed,
            "min_score": gate.min_score(),
        })))
    }

    /// Generate Dilithium key pair
    #[cfg(feature = "dilithium")]
    fn generate_keypair(&self, _args: JsonValue) -> Result<ToolResponse> {
        use crate::dilithium_auth::DilithiumKeyPair;

        let keypair = DilithiumKeyPair::generate()?;

        Ok(ToolResponse::json(serde_json::json!({
            "public_key": hex::encode(keypair.public_key_bytes()),
            "secret_key": hex::encode(keypair.secret_key_bytes()),
            "public_key_id": keypair.public_key_id,
        })))
    }

    /// Register client
    #[cfg(feature = "dilithium")]
    fn register_client(&self, args: JsonValue) -> Result<ToolResponse> {
        let name = args["name"]
            .as_str()
            .ok_or_else(|| CqgsMcpError::Protocol("Missing name".to_string()))?
            .to_string();

        let public_key = args["public_key"]
            .as_str()
            .ok_or_else(|| CqgsMcpError::Protocol("Missing public_key".to_string()))?
            .to_string();

        let capabilities = if let Some(caps) = args["capabilities"].as_array() {
            caps.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        } else {
            vec![]
        };

        let credentials = self.auth.register_client(name, public_key, capabilities, None)?;

        Ok(ToolResponse::json(serde_json::to_value(&credentials)?))
    }

    /// Compute hyperbolic distance
    #[cfg(feature = "hyperbolic")]
    fn compute_hyperbolic_distance(&self, args: JsonValue) -> Result<ToolResponse> {
        use crate::hyperbolic::compute_distance;

        let point1: Vec<f64> = serde_json::from_value(args["point1"].clone())?;
        let point2: Vec<f64> = serde_json::from_value(args["point2"].clone())?;

        match compute_distance(&point1, &point2) {
            Ok(distance) => Ok(ToolResponse::json(serde_json::json!({
                "distance": distance,
            }))),
            Err(e) => Ok(ToolResponse::error(format!("Distance computation failed: {}", e))),
        }
    }

    /// Compute Shannon entropy
    #[cfg(feature = "symbolic")]
    fn compute_shannon_entropy(&self, args: JsonValue) -> Result<ToolResponse> {
        use crate::symbolic::shannon_entropy;

        let probabilities: Vec<f64> = serde_json::from_value(args["probabilities"].clone())?;

        match shannon_entropy(&probabilities) {
            Ok(entropy) => Ok(ToolResponse::json(serde_json::json!({
                "entropy": entropy,
            }))),
            Err(e) => Ok(ToolResponse::error(format!("Shannon entropy error: {}", e))),
        }
    }

    /// Get Fibonacci thresholds
    fn get_fibonacci_thresholds(&self, args: JsonValue) -> Result<ToolResponse> {
        use crate::fibonacci::*;

        let category = args["category"].as_str().unwrap_or("all");

        let response = match category {
            "technical_debt" => {
                let debt = TechnicalDebtMinutes::default();
                serde_json::json!({
                    "category": "technical_debt",
                    "thresholds": {
                        "TODO": debt.todo,
                        "FIXME": debt.fixme,
                        "HACK": debt.hack,
                        "DEBUG_ARTIFACT": debt.debug_artifact,
                        "COMPLEXITY_UNIT": debt.complexity_unit,
                        "LONG_FILE_UNIT": debt.long_file_unit,
                    },
                    "units": "minutes"
                })
            }
            "complexity" => {
                let complexity = ComplexityThresholds::default();
                serde_json::json!({
                    "category": "complexity",
                    "thresholds": {
                        "LOW": complexity.low,
                        "MODERATE": complexity.moderate,
                        "HIGH": complexity.high,
                        "VERY_HIGH": complexity.very_high,
                    }
                })
            }
            "file_size" => {
                let file_size = FileSizeThresholds::default();
                serde_json::json!({
                    "category": "file_size",
                    "thresholds": {
                        "SMALL": file_size.small,
                        "MEDIUM": file_size.medium,
                        "LARGE": file_size.large,
                        "VERY_LARGE": file_size.very_large,
                    },
                    "units": "lines"
                })
            }
            "entropy" => {
                let entropy = EntropyThresholds::default();
                serde_json::json!({
                    "category": "entropy",
                    "thresholds": {
                        "SYNTHETIC": entropy.synthetic,
                        "LOW": entropy.low,
                        "MEDIUM": entropy.medium,
                        "HIGH": entropy.high,
                        "VERY_HIGH": entropy.very_high,
                    },
                    "units": "bits"
                })
            }
            _ => {
                let thresholds = FibonacciThresholds::default();
                serde_json::to_value(&thresholds)?
            }
        };

        Ok(ToolResponse::json(response))
    }

    /// Calculate technical debt
    fn calculate_technical_debt(&self, args: JsonValue) -> Result<ToolResponse> {
        use crate::fibonacci::*;

        let debt = TechnicalDebtMinutes::default();
        let complexity_thresholds = ComplexityThresholds::default();
        let file_size_thresholds = FileSizeThresholds::default();

        let todo_count = args["todo_count"].as_u64().unwrap_or(0) as u32;
        let fixme_count = args["fixme_count"].as_u64().unwrap_or(0) as u32;
        let hack_count = args["hack_count"].as_u64().unwrap_or(0) as u32;
        let debug_artifacts = args["debug_artifacts"].as_u64().unwrap_or(0) as u32;
        let complexity = args["complexity"].as_f64().unwrap_or(0.0) as u32;
        let lines_of_code = args["lines_of_code"].as_u64().unwrap_or(0) as u32;

        // Calculate total debt
        let mut total_debt = 0.0;
        total_debt += (todo_count * debt.todo) as f64;
        total_debt += (fixme_count * debt.fixme) as f64;
        total_debt += (hack_count * debt.hack) as f64;
        total_debt += (debug_artifacts * debt.debug_artifact) as f64;

        // Add complexity penalty
        if complexity > complexity_thresholds.moderate {
            total_debt += ((complexity - complexity_thresholds.moderate) * debt.complexity_unit) as f64;
        }

        // Add file size penalty
        if lines_of_code > file_size_thresholds.large {
            total_debt += ((lines_of_code - file_size_thresholds.large) as f64) * debt.long_file_unit;
        }

        Ok(ToolResponse::json(serde_json::json!({
            "total_debt_minutes": total_debt,
            "total_debt_hours": total_debt / 60.0,
            "breakdown": {
                "TODO": todo_count * debt.todo,
                "FIXME": fixme_count * debt.fixme,
                "HACK": hack_count * debt.hack,
                "DEBUG": debug_artifacts * debt.debug_artifact,
                "complexity_penalty": if complexity > complexity_thresholds.moderate {
                    (complexity - complexity_thresholds.moderate) * debt.complexity_unit
                } else { 0 },
                "file_size_penalty": if lines_of_code > file_size_thresholds.large {
                    ((lines_of_code - file_size_thresholds.large) as f64 * debt.long_file_unit) as u32
                } else { 0 }
            }
        })))
    }

    /// Check complexity
    fn check_complexity(&self, args: JsonValue) -> Result<ToolResponse> {
        use crate::fibonacci::*;

        let complexity = args["complexity"].as_f64().unwrap_or(0.0) as u32;
        let thresholds = ComplexityThresholds::default();

        let level = if complexity <= thresholds.low {
            "LOW"
        } else if complexity <= thresholds.moderate {
            "MODERATE"
        } else if complexity <= thresholds.high {
            "HIGH"
        } else {
            "VERY_HIGH"
        };

        Ok(ToolResponse::json(serde_json::json!({
            "complexity": complexity,
            "level": level,
            "thresholds": {
                "LOW": thresholds.low,
                "MODERATE": thresholds.moderate,
                "HIGH": thresholds.high,
                "VERY_HIGH": thresholds.very_high,
            },
            "passes": complexity <= thresholds.moderate
        })))
    }

    /// Check file size
    fn check_file_size(&self, args: JsonValue) -> Result<ToolResponse> {
        use crate::fibonacci::*;

        let lines = args["lines"].as_u64().unwrap_or(0) as u32;
        let thresholds = FileSizeThresholds::default();

        let size_category = if lines <= thresholds.small {
            "SMALL"
        } else if lines <= thresholds.medium {
            "MEDIUM"
        } else if lines <= thresholds.large {
            "LARGE"
        } else {
            "VERY_LARGE"
        };

        Ok(ToolResponse::json(serde_json::json!({
            "lines": lines,
            "category": size_category,
            "thresholds": {
                "SMALL": thresholds.small,
                "MEDIUM": thresholds.medium,
                "LARGE": thresholds.large,
                "VERY_LARGE": thresholds.very_large,
            },
            "acceptable": lines <= thresholds.large
        })))
    }

    /// Check entropy
    fn check_entropy(&self, args: JsonValue) -> Result<ToolResponse> {
        use crate::fibonacci::*;

        let entropy = if let Some(probs) = args["probabilities"].as_array() {
            #[cfg(feature = "symbolic")]
            {
                use crate::symbolic::shannon_entropy;
                let probabilities: Vec<f64> = probs.iter()
                    .filter_map(|v| v.as_f64())
                    .collect();
                shannon_entropy(&probabilities).unwrap_or(0.0)
            }
            #[cfg(not(feature = "symbolic"))]
            {
                args["entropy"].as_f64().unwrap_or(0.0)
            }
        } else {
            args["entropy"].as_f64().unwrap_or(0.0)
        };

        let thresholds = EntropyThresholds::default();

        let level = if entropy < thresholds.synthetic {
            "SYNTHETIC"
        } else if entropy < thresholds.low {
            "LOW"
        } else if entropy < thresholds.medium {
            "MEDIUM"
        } else if entropy < thresholds.high {
            "HIGH"
        } else {
            "VERY_HIGH"
        };

        let is_synthetic = entropy < thresholds.synthetic;

        Ok(ToolResponse::json(serde_json::json!({
            "entropy": entropy,
            "level": level,
            "is_synthetic": is_synthetic,
            "thresholds": {
                "SYNTHETIC": thresholds.synthetic,
                "LOW": thresholds.low,
                "MEDIUM": thresholds.medium,
                "HIGH": thresholds.high,
                "VERY_HIGH": thresholds.very_high,
            }
        })))
    }

    /// Get golden ratio
    fn get_golden_ratio(&self, args: JsonValue) -> Result<ToolResponse> {
        use crate::fibonacci::*;

        let response = if let Some(power) = args["power"].as_i64() {
            let value = golden_power(power as i32);
            serde_json::json!({
                "phi": PHI,
                "phi_inv": PHI_INV,
                "phi_power_n": value,
                "power": power,
            })
        } else {
            serde_json::json!({
                "phi": PHI,
                "phi_inv": PHI_INV,
                "properties": {
                    "phi_squared": PHI * PHI,
                    "phi_minus_1": PHI - 1.0,
                    "phi_inv_equals_phi_minus_1": (PHI - 1.0 - PHI_INV).abs() < 1e-10,
                }
            })
        };

        Ok(ToolResponse::json(response))
    }

    /// Calculate Fibonacci number
    fn calculate_fibonacci_number(&self, args: JsonValue) -> Result<ToolResponse> {
        use crate::fibonacci::*;

        let n = args["n"].as_u64().unwrap_or(0) as usize;

        let fib_n = fibonacci(n);

        Ok(ToolResponse::json(serde_json::json!({
            "n": n,
            "fibonacci_n": fib_n,
            "formula": "F_n = round(φ^n / √5)",
            "phi": PHI,
        })))
    }

    /// List all tools
    pub fn list_tools(&self) -> Vec<ToolDefinition> {
        self.tools.values().cloned().collect()
    }

    /// Get tool by name
    pub fn get_tool(&self, name: &str) -> Option<&ToolDefinition> {
        self.tools.get(name)
    }
}

impl Default for McpToolRegistry {
    fn default() -> Self {
        Self::new().expect("Failed to create McpToolRegistry")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_registry_creation() {
        let registry = McpToolRegistry::new().unwrap();
        assert!(!registry.tools.is_empty());
    }

    #[test]
    fn test_tool_response_success() {
        let response = ToolResponse::success("Test".to_string());
        assert!(response.is_error.is_none());
        assert_eq!(response.content[0].text, "Test");
    }

    #[test]
    fn test_tool_response_error() {
        let response = ToolResponse::error("Error".to_string());
        assert_eq!(response.is_error, Some(true));
    }
}
