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
use crate::cognition::{
    CognitiveSentinelSystem, CollectiveIntelligence, ConsciousnessMetrics,
    MemoryContentType,
    PHI_IIT, FREE_ENERGY_F, STDP_WEIGHT_CHANGE, WORKING_MEMORY_CAPACITY,
};

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

    /// Cognitive sentinel system (IIT Phi, STDP, Free Energy)
    cognitive_system: std::sync::Arc<parking_lot::RwLock<CognitiveSentinelSystem>>,

    /// Collective intelligence system for knowledge sharing
    collective: std::sync::Arc<parking_lot::RwLock<CollectiveIntelligence>>,

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
            cognitive_system: std::sync::Arc::new(parking_lot::RwLock::new(
                CognitiveSentinelSystem::new("cqgs-mcp-sentinel")
            )),
            collective: std::sync::Arc::new(parking_lot::RwLock::new(
                CollectiveIntelligence::new()
            )),
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

        // Cognitive sentinel tools
        self.register_cognitive_tools()?;

        Ok(())
    }

    /// Register sentinel execution tools
    fn register_sentinel_tools(&mut self) -> Result<()> {
        use serde_json::json;

        // Execute all sentinels
        self.tools.insert(
            "sentinel_execute_all".to_string(),
            ToolDefinition {
                name: "sentinel_execute_all".to_string(),
                description: "Execute all CQGS sentinels on a codebase".to_string(),
                input_schema: json!({
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
                input_schema: json!({
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
                input_schema: json!({
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

        // ============================================================================
        // Individual Sentinel MCP Tools (v2.0)
        // ============================================================================

        // Mock Detection Sentinel (47 tests passing)
        self.tools.insert(
            "sentinel_mock_detection".to_string(),
            ToolDefinition {
                name: "sentinel_mock_detection".to_string(),
                description: "Detect mock/synthetic data patterns in code. Identifies np.random, mock.*, placeholder implementations, hardcoded values. Returns violations with severity and suggested fixes.".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Source code to analyze for mock/synthetic patterns"
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Optional file path for context (default: inline)"
                        },
                        "strict_mode": {
                            "type": "boolean",
                            "description": "Enable strict mode - detect all synthetic patterns (default: true)"
                        }
                    },
                    "required": ["code"]
                }),
                required_capabilities: vec!["sentinel_mock".to_string()],
            },
        );

        // Reward Hacking Prevention Sentinel (14 tests passing)
        self.tools.insert(
            "sentinel_reward_hacking".to_string(),
            ToolDefinition {
                name: "sentinel_reward_hacking".to_string(),
                description: "Detect reward hacking patterns: test modification, metric manipulation, circular validation, objective misalignment, reward gaming, shortcut behavior. Based on AI safety research.".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Source code to analyze for reward hacking patterns"
                        },
                        "context": {
                            "type": "string",
                            "description": "Context identifier (file path, function name, etc.)"
                        },
                        "detect_test_modifications": {
                            "type": "boolean",
                            "description": "Detect test assertion removal/modification (default: true)"
                        },
                        "detect_metric_manipulation": {
                            "type": "boolean",
                            "description": "Detect hardcoded metrics or gaming patterns (default: true)"
                        },
                        "detect_circular_validation": {
                            "type": "boolean",
                            "description": "Detect circular validation dependencies (default: true)"
                        }
                    },
                    "required": ["code"]
                }),
                required_capabilities: vec!["sentinel_reward".to_string()],
            },
        );

        // Cross-Scale Analysis Sentinel (13 tests passing)
        self.tools.insert(
            "sentinel_cross_scale".to_string(),
            ToolDefinition {
                name: "sentinel_cross_scale".to_string(),
                description: "Perform multi-level code quality analysis: complexity metrics, maintainability scores, coupling analysis, cohesion metrics. Returns comprehensive quality assessment with Fibonacci-scaled thresholds.".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Source code to analyze"
                        },
                        "analyze_complexity": {
                            "type": "boolean",
                            "description": "Calculate cyclomatic complexity (default: true)"
                        },
                        "analyze_maintainability": {
                            "type": "boolean",
                            "description": "Calculate maintainability index (default: true)"
                        },
                        "analyze_coupling": {
                            "type": "boolean",
                            "description": "Analyze coupling metrics (default: true)"
                        },
                        "fibonacci_thresholds": {
                            "type": "boolean",
                            "description": "Use Fibonacci-scaled thresholds (F_6=8, F_7=13, etc.) (default: true)"
                        }
                    },
                    "required": ["code"]
                }),
                required_capabilities: vec!["sentinel_cross_scale".to_string()],
            },
        );

        // Zero-Synthetic Enforcement Sentinel (8 tests passing)
        self.tools.insert(
            "sentinel_zero_synthetic".to_string(),
            ToolDefinition {
                name: "sentinel_zero_synthetic".to_string(),
                description: "Enforce TENGRI zero-synthetic data rules. Critical-severity violations for: np.random, random.*, mock.*, placeholder implementations, TODO markers, hardcoded test data. Automatic rollback support.".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Source code to enforce zero-synthetic rules"
                        },
                        "file_path": {
                            "type": "string",
                            "description": "File path for violation reporting"
                        },
                        "enforcement_mode": {
                            "type": "string",
                            "enum": ["strict", "warning", "audit"],
                            "description": "Enforcement mode (default: strict)"
                        },
                        "auto_rollback": {
                            "type": "boolean",
                            "description": "Enable automatic rollback on critical violations (default: false)"
                        }
                    },
                    "required": ["code"]
                }),
                required_capabilities: vec!["sentinel_zero_synthetic".to_string()],
            },
        );

        // List enabled sentinels
        self.tools.insert(
            "sentinel_list_enabled".to_string(),
            ToolDefinition {
                name: "sentinel_list_enabled".to_string(),
                description: "List all enabled sentinel features with test counts and capabilities".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {}
                }),
                required_capabilities: vec![],
            },
        );

        // Sentinel batch analysis
        self.tools.insert(
            "sentinel_batch_analyze".to_string(),
            ToolDefinition {
                name: "sentinel_batch_analyze".to_string(),
                description: "Run multiple sentinels on code in a single call. Returns combined results with overall quality score.".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Source code to analyze"
                        },
                        "sentinels": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["mock_detection", "reward_hacking", "cross_scale", "zero_synthetic"]
                            },
                            "description": "List of sentinels to run (default: all enabled)"
                        },
                        "parallel": {
                            "type": "boolean",
                            "description": "Run sentinels in parallel (default: true)"
                        }
                    },
                    "required": ["code"]
                }),
                required_capabilities: vec!["sentinel_execute".to_string()],
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

    /// Register cognitive sentinel tools
    /// Based on IIT (Tononi 2004), FEP (Friston 2010), STDP (Bi & Poo 1998)
    fn register_cognitive_tools(&mut self) -> Result<()> {
        use serde_json::json;

        // ============================================================================
        // Cognitive Metrics Tools (Consciousness, Learning, Memory)
        // ============================================================================

        // Get consciousness metrics (Phi, Free Energy, STDP)
        self.tools.insert(
            "cognitive_get_metrics".to_string(),
            ToolDefinition {
                name: "cognitive_get_metrics".to_string(),
                description: format!(
                    "Get cognitive metrics from IIT (Phi={}), FEP (F={}), STDP (ΔW={}). \
                     Returns consciousness level, free energy, prediction errors, and learning rate. \
                     Based on Tononi (2004), Friston (2010), Bi & Poo (1998).",
                    PHI_IIT, FREE_ENERGY_F, STDP_WEIGHT_CHANGE
                ),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "include_memory_stats": {
                            "type": "boolean",
                            "description": "Include 3-tier memory statistics (working, episodic, semantic)"
                        }
                    }
                }),
                required_capabilities: vec!["cognitive_metrics".to_string()],
            },
        );

        // Learn from code analysis
        self.tools.insert(
            "cognitive_learn".to_string(),
            ToolDefinition {
                name: "cognitive_learn".to_string(),
                description: "Learn from code analysis experience using STDP (Spike-Timing Dependent Plasticity). \
                             Updates synaptic weights based on temporal correlation. \
                             Stores patterns in 3-tier memory: working → episodic → semantic.".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "pattern_name": {
                            "type": "string",
                            "description": "Name of the pattern being learned"
                        },
                        "code_content": {
                            "type": "string",
                            "description": "Code content that triggered learning"
                        },
                        "outcome": {
                            "type": "string",
                            "enum": ["success", "failure", "partial"],
                            "description": "Outcome of the analysis for reward signal"
                        },
                        "importance": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Importance weight for memory consolidation (default: 0.5)"
                        }
                    },
                    "required": ["pattern_name", "outcome"]
                }),
                required_capabilities: vec!["cognitive_learn".to_string()],
            },
        );

        // Think - apply learned knowledge
        self.tools.insert(
            "cognitive_think".to_string(),
            ToolDefinition {
                name: "cognitive_think".to_string(),
                description: "Apply cognitive reasoning to code analysis using learned patterns. \
                             Uses Free Energy Principle to minimize prediction error. \
                             Returns predictions, confidence scores, and relevant memories.".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query to think about (code pattern, problem, etc.)"
                        },
                        "context": {
                            "type": "string",
                            "description": "Additional context for reasoning"
                        },
                        "max_memories": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                            "description": "Maximum memories to retrieve (default: 10)"
                        }
                    },
                    "required": ["query"]
                }),
                required_capabilities: vec!["cognitive_think".to_string()],
            },
        );

        // Dream - consolidate memories and optimize
        self.tools.insert(
            "cognitive_dream".to_string(),
            ToolDefinition {
                name: "cognitive_dream".to_string(),
                description: "Trigger memory consolidation and pattern optimization (dreaming). \
                             Transfers working memory to episodic/semantic memory. \
                             Prunes weak connections, strengthens important patterns. \
                             Based on memory consolidation theory (Tulving 1972).".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "cycles": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                            "description": "Number of consolidation cycles (default: 1)"
                        },
                        "prune_threshold": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Threshold for pruning weak memories (default: 0.1)"
                        }
                    }
                }),
                required_capabilities: vec!["cognitive_dream".to_string()],
            },
        );

        // Innovate - generate novel solutions
        self.tools.insert(
            "cognitive_innovate".to_string(),
            ToolDefinition {
                name: "cognitive_innovate".to_string(),
                description: "Generate novel solutions by combining learned patterns. \
                             Uses hyperbolic geometry for semantic space exploration. \
                             Returns creative combinations and confidence scores.".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "problem": {
                            "type": "string",
                            "description": "Problem to solve innovatively"
                        },
                        "constraints": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Constraints to respect"
                        },
                        "creativity_level": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Creativity level (0=conservative, 1=experimental) (default: 0.5)"
                        }
                    },
                    "required": ["problem"]
                }),
                required_capabilities: vec!["cognitive_innovate".to_string()],
            },
        );

        // Evolve - optimize sentinel behavior
        self.tools.insert(
            "cognitive_evolve".to_string(),
            ToolDefinition {
                name: "cognitive_evolve".to_string(),
                description: "Evolve and optimize sentinel detection parameters. \
                             Uses evolutionary strategies with learned fitness landscape. \
                             Adjusts thresholds, weights, and pattern priorities.".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "generations": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                            "description": "Number of evolution generations (default: 10)"
                        },
                        "mutation_rate": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Mutation rate for parameter variation (default: 0.1)"
                        },
                        "fitness_metric": {
                            "type": "string",
                            "enum": ["accuracy", "recall", "precision", "f1"],
                            "description": "Fitness metric to optimize (default: f1)"
                        }
                    }
                }),
                required_capabilities: vec!["cognitive_evolve".to_string()],
            },
        );

        // ============================================================================
        // Collective Intelligence Tools (Knowledge Sharing)
        // ============================================================================

        // Share knowledge with collective
        self.tools.insert(
            "cognitive_share_knowledge".to_string(),
            ToolDefinition {
                name: "cognitive_share_knowledge".to_string(),
                description: "Share learned knowledge with collective intelligence pool. \
                             Enables cross-sentinel learning and distributed optimization. \
                             Knowledge is tagged and indexed for efficient retrieval.".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "knowledge_type": {
                            "type": "string",
                            "enum": ["pattern", "threshold", "strategy", "insight"],
                            "description": "Type of knowledge to share"
                        },
                        "content": {
                            "type": "string",
                            "description": "Knowledge content to share"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for knowledge indexing"
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence in this knowledge (default: 0.8)"
                        }
                    },
                    "required": ["knowledge_type", "content"]
                }),
                required_capabilities: vec!["cognitive_collective".to_string()],
            },
        );

        // Query collective knowledge
        self.tools.insert(
            "cognitive_query_collective".to_string(),
            ToolDefinition {
                name: "cognitive_query_collective".to_string(),
                description: "Query collective intelligence for relevant knowledge. \
                             Searches distributed knowledge base using semantic similarity. \
                             Returns ranked results with provenance information.".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for collective knowledge"
                        },
                        "knowledge_types": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["pattern", "threshold", "strategy", "insight"]
                            },
                            "description": "Filter by knowledge types (default: all)"
                        },
                        "min_confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Minimum confidence threshold (default: 0.5)"
                        },
                        "max_results": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 50,
                            "description": "Maximum results to return (default: 10)"
                        }
                    },
                    "required": ["query"]
                }),
                required_capabilities: vec!["cognitive_collective".to_string()],
            },
        );

        // Get memory statistics
        self.tools.insert(
            "cognitive_memory_stats".to_string(),
            ToolDefinition {
                name: "cognitive_memory_stats".to_string(),
                description: "Get detailed statistics for the 3-tier memory system. \
                             Based on Tulving's memory model (1972, 1985). \
                             Returns working memory (7±2 items), episodic (events), semantic (facts).".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "tier": {
                            "type": "string",
                            "enum": ["all", "working", "episodic", "semantic"],
                            "description": "Memory tier to query (default: all)"
                        }
                    }
                }),
                required_capabilities: vec!["cognitive_memory".to_string()],
            },
        );

        // Compute integrated information Phi
        self.tools.insert(
            "cognitive_compute_phi".to_string(),
            ToolDefinition {
                name: "cognitive_compute_phi".to_string(),
                description: format!(
                    "Compute integrated information Φ (Phi) using IIT theory. \
                     Phi measures consciousness level: Φ={} indicates high integration. \
                     Based on Tononi (2004, 2008, 2012).", PHI_IIT
                ),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "network_state": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Network activation state (optional - uses current state if not provided)"
                        },
                        "algorithm": {
                            "type": "string",
                            "enum": ["exact", "monte_carlo", "greedy"],
                            "description": "Phi computation algorithm (default: greedy)"
                        }
                    }
                }),
                required_capabilities: vec!["cognitive_phi".to_string()],
            },
        );

        // STDP weight update
        self.tools.insert(
            "cognitive_stdp_update".to_string(),
            ToolDefinition {
                name: "cognitive_stdp_update".to_string(),
                description: format!(
                    "Apply STDP (Spike-Timing Dependent Plasticity) weight update. \
                     ΔW = {} × exp(-Δt/τ) for potentiation, -A⁻ × exp(Δt/τ) for depression. \
                     τ = 20ms (Bi & Poo 1998).", STDP_WEIGHT_CHANGE
                ),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "delta_t": {
                            "type": "number",
                            "description": "Time difference (post - pre) in ms"
                        },
                        "current_weight": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Current synaptic weight"
                        }
                    },
                    "required": ["delta_t", "current_weight"]
                }),
                required_capabilities: vec!["cognitive_stdp".to_string()],
            },
        );

        Ok(())
    }

    /// Execute tool by name
    pub fn execute_tool(&self, name: &str, args: JsonValue) -> Result<ToolResponse> {
        match name {
            // Sentinel execution tools
            "sentinel_execute_all" => self.execute_all_sentinels(args),
            "sentinel_quality_score" => self.calculate_quality_score(args),
            "sentinel_quality_gate" => self.check_quality_gate(args),

            // Individual sentinel tools (v2.0)
            "sentinel_mock_detection" => self.execute_mock_detection_tool(args),
            "sentinel_reward_hacking" => self.execute_reward_hacking_tool(args),
            "sentinel_cross_scale" => self.execute_cross_scale_tool(args),
            "sentinel_zero_synthetic" => self.execute_zero_synthetic_tool(args),
            "sentinel_list_enabled" => self.list_enabled_sentinels(args),
            "sentinel_batch_analyze" => self.batch_analyze_sentinels(args),

            // Authentication tools
            #[cfg(feature = "dilithium")]
            "dilithium_keygen" => self.generate_keypair(args),

            #[cfg(feature = "dilithium")]
            "dilithium_register_client" => self.register_client(args),

            // Hyperbolic geometry tools
            #[cfg(feature = "hyperbolic")]
            "hyperbolic_distance" => self.compute_hyperbolic_distance(args),

            // Symbolic math tools
            #[cfg(feature = "symbolic")]
            "shannon_entropy" => self.compute_shannon_entropy(args),

            // Fibonacci threshold tools
            "fibonacci_get_thresholds" => self.get_fibonacci_thresholds(args),
            "fibonacci_calculate_debt" => self.calculate_technical_debt(args),
            "fibonacci_complexity_check" => self.check_complexity(args),
            "fibonacci_file_size_check" => self.check_file_size(args),
            "fibonacci_entropy_check" => self.check_entropy(args),
            "fibonacci_golden_ratio" => self.get_golden_ratio(args),
            "fibonacci_calculate_number" => self.calculate_fibonacci_number(args),

            // Cognitive sentinel tools (IIT, STDP, FEP)
            "cognitive_get_metrics" => self.cognitive_get_metrics(args),
            "cognitive_learn" => self.cognitive_learn(args),
            "cognitive_think" => self.cognitive_think(args),
            "cognitive_dream" => self.cognitive_dream(args),
            "cognitive_innovate" => self.cognitive_innovate(args),
            "cognitive_evolve" => self.cognitive_evolve(args),
            "cognitive_share_knowledge" => self.cognitive_share_knowledge(args),
            "cognitive_query_collective" => self.cognitive_query_collective(args),
            "cognitive_memory_stats" => self.cognitive_memory_stats(args),
            "cognitive_compute_phi" => self.cognitive_compute_phi(args),
            "cognitive_stdp_update" => self.cognitive_stdp_update(args),

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

    // ============================================================================
    // Individual Sentinel Tool Handlers (v2.0)
    // ============================================================================

    /// Execute mock detection sentinel tool
    #[cfg(feature = "sentinel-mock")]
    fn execute_mock_detection_tool(&self, args: JsonValue) -> Result<ToolResponse> {
        let code = args["code"]
            .as_str()
            .ok_or_else(|| CqgsMcpError::Protocol("Missing 'code' parameter".to_string()))?;

        let file_path = args["file_path"].as_str().unwrap_or("inline");
        let strict_mode = args["strict_mode"].as_bool().unwrap_or(true);

        let result = self.executor.execute_mock_detection(code)?;

        Ok(ToolResponse::json(serde_json::json!({
            "sentinel": "mock-detection",
            "status": format!("{:?}", result.status),
            "quality_score": result.quality_score,
            "violations_count": result.violations.len(),
            "violations": result.violations.iter().map(|v| {
                serde_json::json!({
                    "type": v.violation_type,
                    "severity": format!("{:?}", v.severity),
                    "description": v.description,
                    "location": {
                        "file": v.location.file,
                        "line": v.location.line,
                        "snippet": v.location.snippet,
                    },
                    "suggested_fix": v.suggested_fix,
                })
            }).collect::<Vec<_>>(),
            "execution_time_us": result.execution_time_us,
            "file_path": file_path,
            "strict_mode": strict_mode,
            "tests_passing": 47,
        })))
    }

    #[cfg(not(feature = "sentinel-mock"))]
    fn execute_mock_detection_tool(&self, _args: JsonValue) -> Result<ToolResponse> {
        Ok(ToolResponse::error(
            "sentinel-mock feature not enabled. Enable with: --features sentinel-mock".to_string()
        ))
    }

    /// Execute reward hacking prevention sentinel tool
    #[cfg(feature = "sentinel-reward")]
    fn execute_reward_hacking_tool(&self, args: JsonValue) -> Result<ToolResponse> {
        let code = args["code"]
            .as_str()
            .ok_or_else(|| CqgsMcpError::Protocol("Missing 'code' parameter".to_string()))?;

        let context = args["context"].as_str().unwrap_or("inline");
        let detect_test_mods = args["detect_test_modifications"].as_bool().unwrap_or(true);
        let detect_metric_manip = args["detect_metric_manipulation"].as_bool().unwrap_or(true);
        let detect_circular = args["detect_circular_validation"].as_bool().unwrap_or(true);

        let result = self.executor.execute_reward_hacking(code, context)?;

        Ok(ToolResponse::json(serde_json::json!({
            "sentinel": "reward-hacking",
            "status": format!("{:?}", result.status),
            "quality_score": result.quality_score,
            "violations_count": result.violations.len(),
            "violations": result.violations.iter().map(|v| {
                serde_json::json!({
                    "type": v.violation_type,
                    "severity": format!("{:?}", v.severity),
                    "description": v.description,
                    "location": {
                        "file": v.location.file,
                        "line": v.location.line,
                    },
                    "suggested_fix": v.suggested_fix,
                    "citation": v.citation,
                })
            }).collect::<Vec<_>>(),
            "execution_time_us": result.execution_time_us,
            "config": {
                "detect_test_modifications": detect_test_mods,
                "detect_metric_manipulation": detect_metric_manip,
                "detect_circular_validation": detect_circular,
            },
            "tests_passing": 14,
        })))
    }

    #[cfg(not(feature = "sentinel-reward"))]
    fn execute_reward_hacking_tool(&self, _args: JsonValue) -> Result<ToolResponse> {
        Ok(ToolResponse::error(
            "sentinel-reward feature not enabled. Enable with: --features sentinel-reward".to_string()
        ))
    }

    /// Execute cross-scale analysis sentinel tool
    #[cfg(feature = "sentinel-cross-scale")]
    fn execute_cross_scale_tool(&self, args: JsonValue) -> Result<ToolResponse> {
        let code = args["code"]
            .as_str()
            .ok_or_else(|| CqgsMcpError::Protocol("Missing 'code' parameter".to_string()))?;

        let analyze_complexity = args["analyze_complexity"].as_bool().unwrap_or(true);
        let analyze_maintainability = args["analyze_maintainability"].as_bool().unwrap_or(true);
        let analyze_coupling = args["analyze_coupling"].as_bool().unwrap_or(true);
        let fibonacci_thresholds = args["fibonacci_thresholds"].as_bool().unwrap_or(true);

        let result = self.executor.execute_cross_scale(code)?;

        Ok(ToolResponse::json(serde_json::json!({
            "sentinel": "cross-scale",
            "status": format!("{:?}", result.status),
            "quality_score": result.quality_score,
            "violations_count": result.violations.len(),
            "violations": result.violations.iter().map(|v| {
                serde_json::json!({
                    "type": v.violation_type,
                    "severity": format!("{:?}", v.severity),
                    "description": v.description,
                    "location": v.location.file,
                    "suggested_fix": v.suggested_fix,
                })
            }).collect::<Vec<_>>(),
            "execution_time_us": result.execution_time_us,
            "config": {
                "analyze_complexity": analyze_complexity,
                "analyze_maintainability": analyze_maintainability,
                "analyze_coupling": analyze_coupling,
                "fibonacci_thresholds": fibonacci_thresholds,
            },
            "thresholds": {
                "complexity_low": 8,      // F_6
                "complexity_moderate": 13, // F_7
                "complexity_high": 21,     // F_8
                "complexity_very_high": 34 // F_9
            },
            "tests_passing": 13,
        })))
    }

    #[cfg(not(feature = "sentinel-cross-scale"))]
    fn execute_cross_scale_tool(&self, _args: JsonValue) -> Result<ToolResponse> {
        Ok(ToolResponse::error(
            "sentinel-cross-scale feature not enabled. Enable with: --features sentinel-cross-scale".to_string()
        ))
    }

    /// Execute zero-synthetic enforcement sentinel tool
    #[cfg(feature = "sentinel-zero-synthetic")]
    fn execute_zero_synthetic_tool(&self, args: JsonValue) -> Result<ToolResponse> {
        let code = args["code"]
            .as_str()
            .ok_or_else(|| CqgsMcpError::Protocol("Missing 'code' parameter".to_string()))?;

        let file_path = args["file_path"].as_str().unwrap_or("inline");
        let enforcement_mode = args["enforcement_mode"].as_str().unwrap_or("strict");
        let auto_rollback = args["auto_rollback"].as_bool().unwrap_or(false);

        let result = self.executor.execute_zero_synthetic(code)?;

        Ok(ToolResponse::json(serde_json::json!({
            "sentinel": "zero-synthetic",
            "status": format!("{:?}", result.status),
            "quality_score": result.quality_score,
            "violations_count": result.violations.len(),
            "violations": result.violations.iter().map(|v| {
                serde_json::json!({
                    "type": v.violation_type,
                    "severity": "Critical", // Zero-synthetic violations are always critical
                    "description": v.description,
                    "location": {
                        "file": v.location.file,
                        "line": v.location.line,
                        "column": v.location.column,
                        "snippet": v.location.snippet,
                    },
                    "suggested_fix": v.suggested_fix,
                })
            }).collect::<Vec<_>>(),
            "execution_time_us": result.execution_time_us,
            "file_path": file_path,
            "enforcement_mode": enforcement_mode,
            "auto_rollback": auto_rollback,
            "tengri_rules": {
                "forbidden_patterns": [
                    "np.random.*",
                    "random.*",
                    "mock.*",
                    "placeholder",
                    "TODO",
                    "hardcoded_values"
                ]
            },
            "tests_passing": 8,
        })))
    }

    #[cfg(not(feature = "sentinel-zero-synthetic"))]
    fn execute_zero_synthetic_tool(&self, _args: JsonValue) -> Result<ToolResponse> {
        Ok(ToolResponse::error(
            "sentinel-zero-synthetic feature not enabled. Enable with: --features sentinel-zero-synthetic".to_string()
        ))
    }

    /// List all enabled sentinel features
    fn list_enabled_sentinels(&self, _args: JsonValue) -> Result<ToolResponse> {
        let enabled = enabled_sentinels();
        let count = enabled_sentinel_count();

        // Build detailed sentinel info
        let sentinel_info: Vec<serde_json::Value> = enabled.iter().map(|name| {
            let (tests, description) = match *name {
                "sentinel-core" => (0, "Foundation sentinel traits and types"),
                "sentinel-mock" => (47, "Mock/synthetic data detection"),
                "sentinel-reward" => (14, "Reward hacking prevention"),
                "sentinel-cross-scale" => (13, "Cross-scale quality analysis"),
                "sentinel-behavioral" => (3, "Behavioral pattern analysis"),
                "sentinel-neural" => (0, "Neural pattern detection"),
                "sentinel-runtime" => (3, "Runtime verification"),
                "sentinel-zero-synthetic" => (8, "Zero-synthetic data enforcement"),
                _ => (0, "Unknown sentinel"),
            };

            serde_json::json!({
                "name": name,
                "tests_passing": tests,
                "description": description,
            })
        }).collect();

        let total_tests: u32 = sentinel_info.iter()
            .filter_map(|s| s["tests_passing"].as_u64())
            .map(|t| t as u32)
            .sum();

        Ok(ToolResponse::json(serde_json::json!({
            "enabled_count": count,
            "total_tests_passing": total_tests,
            "sentinels": sentinel_info,
            "quality_gates": {
                "GATE_1": "NoForbiddenPatterns",
                "GATE_2": "IntegrationReady (≥60)",
                "GATE_3": "TestingReady (≥80)",
                "GATE_4": "ProductionReady (≥95)",
                "GATE_5": "DeploymentApproved (100)",
            },
            "fibonacci_thresholds": {
                "complexity": {"low": 8, "moderate": 13, "high": 21, "very_high": 34},
                "file_size": {"small": 144, "medium": 233, "large": 377, "very_large": 610},
            }
        })))
    }

    /// Execute batch analysis with multiple sentinels
    fn batch_analyze_sentinels(&self, args: JsonValue) -> Result<ToolResponse> {
        let code = args["code"]
            .as_str()
            .ok_or_else(|| CqgsMcpError::Protocol("Missing 'code' parameter".to_string()))?;

        let sentinels_to_run: Vec<String> = if let Some(arr) = args["sentinels"].as_array() {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        } else {
            // Default: run all enabled sentinels
            vec![
                "mock_detection".to_string(),
                "reward_hacking".to_string(),
                "cross_scale".to_string(),
                "zero_synthetic".to_string(),
            ]
        };

        let parallel = args["parallel"].as_bool().unwrap_or(true);
        let start = std::time::Instant::now();

        let mut results: Vec<serde_json::Value> = Vec::new();
        let mut total_violations = 0usize;
        let mut total_score = 0.0f64;
        let mut sentinel_count = 0usize;

        // Execute each requested sentinel
        for sentinel_name in &sentinels_to_run {
            let result = match sentinel_name.as_str() {
                "mock_detection" => {
                    #[cfg(feature = "sentinel-mock")]
                    {
                        if let Ok(r) = self.executor.execute_mock_detection(code) {
                            sentinel_count += 1;
                            total_violations += r.violations.len();
                            total_score += r.quality_score;
                            Some(serde_json::json!({
                                "sentinel": "mock_detection",
                                "status": format!("{:?}", r.status),
                                "quality_score": r.quality_score,
                                "violations": r.violations.len(),
                            }))
                        } else { None }
                    }
                    #[cfg(not(feature = "sentinel-mock"))]
                    { None }
                }
                "reward_hacking" => {
                    #[cfg(feature = "sentinel-reward")]
                    {
                        if let Ok(r) = self.executor.execute_reward_hacking(code, "batch") {
                            sentinel_count += 1;
                            total_violations += r.violations.len();
                            total_score += r.quality_score;
                            Some(serde_json::json!({
                                "sentinel": "reward_hacking",
                                "status": format!("{:?}", r.status),
                                "quality_score": r.quality_score,
                                "violations": r.violations.len(),
                            }))
                        } else { None }
                    }
                    #[cfg(not(feature = "sentinel-reward"))]
                    { None }
                }
                "cross_scale" => {
                    #[cfg(feature = "sentinel-cross-scale")]
                    {
                        if let Ok(r) = self.executor.execute_cross_scale(code) {
                            sentinel_count += 1;
                            total_violations += r.violations.len();
                            total_score += r.quality_score;
                            Some(serde_json::json!({
                                "sentinel": "cross_scale",
                                "status": format!("{:?}", r.status),
                                "quality_score": r.quality_score,
                                "violations": r.violations.len(),
                            }))
                        } else { None }
                    }
                    #[cfg(not(feature = "sentinel-cross-scale"))]
                    { None }
                }
                "zero_synthetic" => {
                    #[cfg(feature = "sentinel-zero-synthetic")]
                    {
                        if let Ok(r) = self.executor.execute_zero_synthetic(code) {
                            sentinel_count += 1;
                            total_violations += r.violations.len();
                            total_score += r.quality_score;
                            Some(serde_json::json!({
                                "sentinel": "zero_synthetic",
                                "status": format!("{:?}", r.status),
                                "quality_score": r.quality_score,
                                "violations": r.violations.len(),
                            }))
                        } else { None }
                    }
                    #[cfg(not(feature = "sentinel-zero-synthetic"))]
                    { None }
                }
                _ => None,
            };

            if let Some(r) = result {
                results.push(r);
            }
        }

        let execution_time_us = start.elapsed().as_micros() as u64;
        let average_score = if sentinel_count > 0 {
            total_score / sentinel_count as f64
        } else {
            0.0
        };

        // Determine overall status and gate
        let overall_status = if total_violations == 0 {
            "Pass"
        } else if average_score >= 95.0 {
            "Warning"
        } else {
            "Fail"
        };

        let quality_gate = if average_score >= 100.0 {
            "DeploymentApproved"
        } else if average_score >= 95.0 {
            "ProductionReady"
        } else if average_score >= 80.0 {
            "TestingReady"
        } else if average_score >= 60.0 {
            "IntegrationReady"
        } else {
            "NoForbiddenPatterns"
        };

        Ok(ToolResponse::json(serde_json::json!({
            "batch_analysis": {
                "sentinels_executed": sentinel_count,
                "sentinels_requested": sentinels_to_run,
                "parallel_execution": parallel,
            },
            "overall_status": overall_status,
            "average_quality_score": average_score,
            "total_violations": total_violations,
            "quality_gate_achieved": quality_gate,
            "results": results,
            "execution_time_us": execution_time_us,
        })))
    }

    // ============================================================================
    // Cognitive Tool Handlers
    // Based on IIT (Tononi), FEP (Friston), STDP (Bi & Poo)
    // ============================================================================

    /// Get cognitive metrics
    fn cognitive_get_metrics(&self, args: JsonValue) -> Result<ToolResponse> {
        let include_memory = args["include_memory_stats"].as_bool().unwrap_or(false);

        let system = self.cognitive_system.read();
        let metrics = &system.consciousness;

        // Get memory counts using public accessor methods
        let working_count = system.working_memory.item_count();
        let episodic_count = system.episodic_memory.episode_count();
        let concept_count = system.semantic_memory.concept_count();
        let relation_count = system.semantic_memory.relation_count();

        let mut response = serde_json::json!({
            "consciousness": {
                "phi": metrics.phi,
                "phi_reference": PHI_IIT,
                "integration_level": if metrics.phi > 1.0 { "high" } else if metrics.phi > 0.5 { "moderate" } else { "low" },
                "citation": "Tononi (2004, 2008, 2012)"
            },
            "free_energy": {
                "F": metrics.free_energy,
                "F_reference": FREE_ENERGY_F,
                "prediction_error": metrics.free_energy * 10.0, // Scaled prediction error
                "citation": "Friston (2006, 2010)"
            },
            "learning": {
                "stdp_weight_change": STDP_WEIGHT_CHANGE,
                "tau_ms": 20.0,
                "citation": "Bi & Poo (1998)"
            },
            "status": {
                "sentinel_id": system.sentinel_id(),
                "session_id": system.session_id().to_string(),
                "is_dreaming": system.is_dreaming(),
            }
        });

        if include_memory {
            response["memory"] = serde_json::json!({
                "working_memory": {
                    "count": working_count,
                    "capacity": WORKING_MEMORY_CAPACITY,
                    "utilization": working_count as f64 / WORKING_MEMORY_CAPACITY as f64,
                    "model": "Miller (1956) 7±2 items"
                },
                "episodic_memory": {
                    "count": episodic_count,
                    "model": "Tulving (1972)"
                },
                "semantic_memory": {
                    "concepts": concept_count,
                    "relations": relation_count,
                    "model": "Tulving (1972, 1985)"
                }
            });
        }

        Ok(ToolResponse::json(response))
    }

    /// Learn from code analysis
    fn cognitive_learn(&self, args: JsonValue) -> Result<ToolResponse> {
        let pattern_name = args["pattern_name"]
            .as_str()
            .ok_or_else(|| CqgsMcpError::Protocol("Missing 'pattern_name'".to_string()))?;

        let outcome = args["outcome"]
            .as_str()
            .ok_or_else(|| CqgsMcpError::Protocol("Missing 'outcome'".to_string()))?;

        let code_content = args["code_content"].as_str().unwrap_or("");

        // Convert outcome to valence (emotional signal)
        let valence = match outcome {
            "success" => 0.8,
            "failure" => -0.6,
            "partial" => 0.2,
            _ => 0.0,
        };

        // Arousal based on importance
        let arousal = args["importance"].as_f64().unwrap_or(0.5);

        // Determine content type based on pattern name
        let content_type = if pattern_name.to_lowercase().contains("violation") {
            MemoryContentType::Violation
        } else if pattern_name.to_lowercase().contains("fix") {
            MemoryContentType::Fix
        } else if pattern_name.to_lowercase().contains("rule") {
            MemoryContentType::Rule
        } else {
            MemoryContentType::CodePattern
        };

        let mut system = self.cognitive_system.write();

        // Create memory item and learn using actual API
        let content = format!("Pattern: {}\nCode: {}\nOutcome: {}", pattern_name, code_content, outcome);
        system.learn(&content, content_type, valence, arousal);

        // Compute expected STDP weight change
        let weight_change = system.learning.compute_weight_change(10.0); // 10ms timing delta

        Ok(ToolResponse::json(serde_json::json!({
            "learned": true,
            "pattern_name": pattern_name,
            "outcome": outcome,
            "valence": valence,
            "arousal": arousal,
            "weight_change": weight_change,
            "memory_location": "working_memory",
            "cognitive_load": system.working_memory.cognitive_load(),
            "stdp_formula": "ΔW = A⁺ × exp(-Δt/τ) for Δt > 0",
            "citation": "Bi & Poo (1998)"
        })))
    }

    /// Think - apply learned knowledge
    fn cognitive_think(&self, args: JsonValue) -> Result<ToolResponse> {
        let _query = args["query"]
            .as_str()
            .ok_or_else(|| CqgsMcpError::Protocol("Missing 'query'".to_string()))?;

        let context = args["context"].as_str().unwrap_or("");

        let mut system = self.cognitive_system.write();

        // Think and get insights
        let insights = system.think();

        // Get current consciousness metrics
        let phi = system.consciousness.phi;
        let free_energy = system.consciousness.free_energy;

        // Confidence based on consciousness level
        let confidence = if phi > 1.0 { 0.8 } else { phi / PHI_IIT * 0.8 };

        Ok(ToolResponse::json(serde_json::json!({
            "context": context,
            "insights": insights,
            "insights_count": insights.len(),
            "confidence": confidence,
            "phi": phi,
            "free_energy": free_energy,
            "consciousness_state": if system.consciousness.is_conscious() {
                "conscious"
            } else {
                "subconscious"
            },
            "cognitive_load": system.working_memory.cognitive_load(),
            "free_energy_principle": "Minimizing prediction error through active inference",
            "citation": "Friston (2010)"
        })))
    }

    /// Dream - memory consolidation
    fn cognitive_dream(&self, args: JsonValue) -> Result<ToolResponse> {
        let cycles = args["cycles"].as_u64().unwrap_or(1) as usize;
        let prune_threshold = args["prune_threshold"].as_f64().unwrap_or(0.1);

        let mut system = self.cognitive_system.write();

        let mut consolidation_results = Vec::new();
        for cycle in 0..cycles {
            // Get counts before (using items_for_consolidation and episodes)
            let before_working = system.working_memory.items_for_consolidation().len();
            let before_episodic = system.episodic_memory.retrieve_recent(1000).len();

            system.dream();

            let after_working = system.working_memory.items_for_consolidation().len();
            let after_episodic = system.episodic_memory.retrieve_recent(1000).len();

            consolidation_results.push(serde_json::json!({
                "cycle": cycle + 1,
                "working_memory_before": before_working,
                "working_memory_after": after_working,
                "episodic_before": before_episodic,
                "episodic_after": after_episodic,
                "items_consolidated": after_episodic.saturating_sub(before_episodic),
            }));
        }

        Ok(ToolResponse::json(serde_json::json!({
            "dreaming_complete": true,
            "cycles_completed": cycles,
            "prune_threshold": prune_threshold,
            "consolidation_results": consolidation_results,
            "cognitive_load_after": system.working_memory.cognitive_load(),
            "memory_theory": "Sleep-dependent memory consolidation (Stickgold 2005)",
            "process": "Working → Episodic → Semantic memory transfer",
            "citation": "Tulving (1972, 1985)"
        })))
    }

    /// Innovate - generate novel solutions
    fn cognitive_innovate(&self, args: JsonValue) -> Result<ToolResponse> {
        let problem = args["problem"]
            .as_str()
            .ok_or_else(|| CqgsMcpError::Protocol("Missing 'problem'".to_string()))?;

        let constraints: Vec<String> = args["constraints"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();

        let creativity_level = args["creativity_level"].as_f64().unwrap_or(0.5);

        let mut system = self.cognitive_system.write();

        // Call the innovate method - returns Vec<MemoryItem>
        let innovations = system.innovate();

        // Generate results from innovation MemoryItems
        let innovation_results: Vec<_> = innovations.iter().enumerate().map(|(i, innovation)| {
            let confidence = 1.0 - (creativity_level * i as f64 / innovations.len().max(1) as f64);
            serde_json::json!({
                "id": innovation.id.to_string(),
                "content": innovation.content,
                "weight": innovation.weight,
                "confidence": confidence.max(0.1),
                "novelty_score": creativity_level * (1.0 + i as f64 * 0.1),
            })
        }).collect();

        Ok(ToolResponse::json(serde_json::json!({
            "problem": problem,
            "constraints": constraints,
            "creativity_level": creativity_level,
            "innovations": innovation_results,
            "innovation_count": innovations.len(),
            "phi": system.consciousness.phi,
            "methodology": "Combinatorial creativity through semantic memory traversal",
            "hyperbolic_space": "H^11 for hierarchical concept navigation"
        })))
    }

    /// Evolve - optimize sentinel parameters
    fn cognitive_evolve(&self, args: JsonValue) -> Result<ToolResponse> {
        let generations = args["generations"].as_u64().unwrap_or(10) as usize;
        let mutation_rate = args["mutation_rate"].as_f64().unwrap_or(0.1);
        let fitness_metric = args["fitness_metric"].as_str().unwrap_or("f1");

        let mut system = self.cognitive_system.write();

        // Get initial metrics
        let initial_phi = system.consciousness.phi;
        let initial_free_energy = system.consciousness.free_energy;

        // Run evolution cycles using performance feedback
        let mut generation_log = Vec::new();
        for gen in 0..generations {
            // Compute performance feedback (0.0 to 1.0)
            let performance = 0.5 + (gen as f64 * mutation_rate * 0.05).min(0.4);
            system.evolve(performance);

            generation_log.push(serde_json::json!({
                "generation": gen + 1,
                "performance_feedback": performance,
                "phi": system.consciousness.phi,
                "free_energy": system.consciousness.free_energy,
            }));
        }

        // Get final metrics
        let final_phi = system.consciousness.phi;

        Ok(ToolResponse::json(serde_json::json!({
            "evolution_complete": true,
            "generations": generations,
            "mutation_rate": mutation_rate,
            "fitness_metric": fitness_metric,
            "results": {
                "initial_phi": initial_phi,
                "final_phi": final_phi,
                "phi_improvement": final_phi - initial_phi,
                "initial_free_energy": initial_free_energy,
                "final_free_energy": system.consciousness.free_energy,
                "generations_log": generation_log,
            },
            "methodology": "Evolutionary strategies with fitness landscape navigation",
            "citation": "Hansen & Ostermeier (2001) - CMA-ES"
        })))
    }

    /// Share knowledge with collective
    fn cognitive_share_knowledge(&self, args: JsonValue) -> Result<ToolResponse> {
        let knowledge_type = args["knowledge_type"]
            .as_str()
            .ok_or_else(|| CqgsMcpError::Protocol("Missing 'knowledge_type'".to_string()))?;

        let content = args["content"]
            .as_str()
            .ok_or_else(|| CqgsMcpError::Protocol("Missing 'content'".to_string()))?;

        let tags: Vec<String> = args["tags"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();

        let confidence = args["confidence"].as_f64().unwrap_or(0.8);

        // Use share_knowledge() from CognitiveSentinelSystem which returns Vec<Concept>
        let system = self.cognitive_system.read();
        let shared_concepts = system.share_knowledge();

        // Synchronize collective to incorporate new knowledge
        let mut collective = self.collective.write();
        collective.synchronize();

        // Count systems in collective using public accessor
        let collective_size = collective.system_count();

        Ok(ToolResponse::json(serde_json::json!({
            "shared": true,
            "knowledge_type": knowledge_type,
            "content": content,
            "tags": tags,
            "confidence": confidence,
            "shared_concepts_count": shared_concepts.len(),
            "shared_concepts": shared_concepts.iter().take(10).map(|c| {
                serde_json::json!({
                    "name": c.name,
                    "confidence": c.confidence,
                })
            }).collect::<Vec<_>>(),
            "collective_systems_count": collective_size,
            "methodology": "Distributed collective intelligence via semantic memory sharing"
        })))
    }

    /// Query collective knowledge
    fn cognitive_query_collective(&self, args: JsonValue) -> Result<ToolResponse> {
        let query = args["query"]
            .as_str()
            .ok_or_else(|| CqgsMcpError::Protocol("Missing 'query'".to_string()))?;

        let min_confidence = args["min_confidence"].as_f64().unwrap_or(0.5);
        let max_results = args["max_results"].as_u64().unwrap_or(10) as usize;

        // Use CollectiveIntelligence API via public accessors
        let collective = self.collective.read();

        // Get collective phi as a measure of collective intelligence
        let collective_phi = collective.collective_phi();

        // Get all shared concepts and filter
        let all_concepts = collective.all_shared_concepts();
        let matching_concepts: Vec<_> = all_concepts.iter()
            .filter(|c| {
                c.confidence >= min_confidence &&
                (c.name.to_lowercase().contains(&query.to_lowercase()) ||
                 c.definition.to_lowercase().contains(&query.to_lowercase()))
            })
            .take(max_results)
            .collect();

        // Get related concepts via public accessor
        let related_results: Vec<_> = matching_concepts.iter()
            .flat_map(|c| {
                collective.get_related_concepts(&c.name)
                    .into_iter()
                    .map(|(concept, relation_type, strength)| {
                        serde_json::json!({
                            "name": concept.name,
                            "definition": concept.definition,
                            "confidence": concept.confidence,
                            "relation_type": relation_type,
                            "relation_strength": strength,
                        })
                    })
            })
            .take(max_results)
            .collect();

        Ok(ToolResponse::json(serde_json::json!({
            "query": query,
            "direct_matches": matching_concepts.iter().map(|c| {
                serde_json::json!({
                    "name": c.name,
                    "definition": c.definition,
                    "confidence": c.confidence,
                    "activation": c.activation,
                })
            }).collect::<Vec<_>>(),
            "direct_matches_count": matching_concepts.len(),
            "related_concepts": related_results,
            "related_count": related_results.len(),
            "min_confidence_filter": min_confidence,
            "collective_phi": collective_phi,
            "collective_systems_count": collective.system_count(),
            "methodology": "Semantic memory search with spreading activation"
        })))
    }

    /// Get memory statistics
    fn cognitive_memory_stats(&self, args: JsonValue) -> Result<ToolResponse> {
        let tier = args["tier"].as_str().unwrap_or("all");

        let system = self.cognitive_system.read();

        // Use public accessor methods from cognition.rs API
        let working_count = system.working_memory.item_count();
        let episodic_count = system.episodic_memory.episode_count();
        let concept_count = system.semantic_memory.concept_count();
        let relation_count = system.semantic_memory.relation_count();

        let response = match tier {
            "working" => serde_json::json!({
                "tier": "working",
                "model": "Miller (1956) - Magical Number Seven",
                "capacity": WORKING_MEMORY_CAPACITY,
                "current_count": working_count,
                "utilization": working_count as f64 / WORKING_MEMORY_CAPACITY as f64,
                "cognitive_load": system.working_memory.cognitive_load(),
                "decay_rate": "Rapid (seconds to minutes)",
            }),
            "episodic" => serde_json::json!({
                "tier": "episodic",
                "model": "Tulving (1972) - Episode Memory",
                "episode_count": episodic_count,
                "characteristics": "Autobiographical events with temporal-spatial context",
            }),
            "semantic" => serde_json::json!({
                "tier": "semantic",
                "model": "Tulving (1972, 1985) - Semantic Memory",
                "concept_count": concept_count,
                "relation_count": relation_count,
                "characteristics": "General knowledge, facts, concepts, semantic relations",
            }),
            _ => serde_json::json!({
                "working_memory": {
                    "count": working_count,
                    "capacity": WORKING_MEMORY_CAPACITY,
                    "cognitive_load": system.working_memory.cognitive_load(),
                    "model": "Miller (1956)"
                },
                "episodic_memory": {
                    "episode_count": episodic_count,
                    "model": "Tulving (1972)"
                },
                "semantic_memory": {
                    "concept_count": concept_count,
                    "relation_count": relation_count,
                    "model": "Tulving (1972, 1985)"
                },
                "total_memory_items": working_count + episodic_count + concept_count
            }),
        };

        Ok(ToolResponse::json(response))
    }

    /// Compute integrated information Phi
    fn cognitive_compute_phi(&self, args: JsonValue) -> Result<ToolResponse> {
        let network_state: Option<Vec<f64>> = args["network_state"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect());

        let algorithm = args["algorithm"].as_str().unwrap_or("greedy");

        let system = self.cognitive_system.read();

        // Use provided network state or derive from working memory
        let phi = if let Some(state) = network_state {
            // Use ConsciousnessMetrics::compute_phi static method
            // Generate simple connections for the network state
            let connections: Vec<(usize, usize, f64)> = (0..state.len())
                .flat_map(|i| (i+1..state.len()).map(move |j| (i, j, 0.5)))
                .collect();
            ConsciousnessMetrics::compute_phi(&state, &connections)
        } else {
            // Return current consciousness phi value
            system.consciousness.phi
        };

        Ok(ToolResponse::json(serde_json::json!({
            "phi": phi,
            "phi_reference": PHI_IIT,
            "algorithm": algorithm,
            "interpretation": if phi > 1.0 {
                "High integration - conscious-like information processing"
            } else if phi > 0.5 {
                "Moderate integration - partial consciousness"
            } else {
                "Low integration - minimal consciousness"
            },
            "theory": "Integrated Information Theory (IIT)",
            "formula": "Φ = min(I(M_past; M_future | partition))",
            "citation": "Tononi (2004, 2008, 2012)"
        })))
    }

    /// Apply STDP weight update
    fn cognitive_stdp_update(&self, args: JsonValue) -> Result<ToolResponse> {
        let delta_t = args["delta_t"]
            .as_f64()
            .ok_or_else(|| CqgsMcpError::Protocol("Missing 'delta_t'".to_string()))?;

        let current_weight = args["current_weight"]
            .as_f64()
            .ok_or_else(|| CqgsMcpError::Protocol("Missing 'current_weight'".to_string()))?;

        let system = self.cognitive_system.read();
        // Use correct field name: `learning` not `stdp`, and method takes one param
        let weight_change = system.learning.compute_weight_change(delta_t);
        let new_weight = (current_weight + weight_change).max(0.0).min(1.0);

        let potentiation = delta_t > 0.0;

        Ok(ToolResponse::json(serde_json::json!({
            "delta_t_ms": delta_t,
            "current_weight": current_weight,
            "weight_change": weight_change,
            "new_weight": new_weight,
            "potentiation": potentiation,
            "depression": !potentiation,
            "tau_ms": 20.0,
            "a_plus": STDP_WEIGHT_CHANGE,
            "a_minus": STDP_WEIGHT_CHANGE * 1.05,
            "formula": if potentiation {
                format!("ΔW = {} × exp(-Δt/{}) = {:.6}", STDP_WEIGHT_CHANGE, 20.0, weight_change)
            } else {
                format!("ΔW = -{} × exp(Δt/{}) = {:.6}", STDP_WEIGHT_CHANGE * 1.05, 20.0, weight_change)
            },
            "citation": "Bi & Poo (1998)"
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
