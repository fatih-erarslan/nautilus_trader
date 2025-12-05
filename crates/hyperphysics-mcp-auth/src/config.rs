//! Security configuration for MCP authentication

use hyperphysics_dilithium::SecurityLevel;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Security configuration for MCP authentication guard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Post-quantum security level (Standard or High)
    pub security_level: SecurityLevel,

    /// Maximum age of requests in seconds (replay protection)
    pub max_request_age_secs: u64,

    /// Rate limit (requests per minute per client)
    pub rate_limit_per_minute: u32,

    /// Enable strict mode (extra validation)
    pub strict_mode: bool,

    /// Enable audit logging
    pub audit_logging: bool,

    /// Nonce cache TTL in seconds
    pub nonce_cache_ttl_secs: u64,

    /// Maximum nonce cache size
    pub max_nonce_cache_size: usize,

    /// Blocked injection patterns (regex patterns)
    pub injection_patterns: Vec<String>,

    /// Allowed tool names (whitelist)
    pub allowed_tools: Option<HashSet<String>>,

    /// Blocked tool names (blacklist, checked before whitelist)
    pub blocked_tools: HashSet<String>,

    /// Require tool-level permissions per client
    pub per_client_tool_permissions: bool,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            security_level: SecurityLevel::High,
            max_request_age_secs: 30,
            rate_limit_per_minute: 100,
            strict_mode: true,
            audit_logging: true,
            nonce_cache_ttl_secs: 300, // 5 minutes
            max_nonce_cache_size: 100_000,
            injection_patterns: default_injection_patterns(),
            allowed_tools: None, // All tools allowed by default
            blocked_tools: default_blocked_tools(),
            per_client_tool_permissions: true,
        }
    }
}

impl SecurityConfig {
    /// Create a new security config with custom settings
    pub fn new(security_level: SecurityLevel) -> Self {
        Self {
            security_level,
            ..Default::default()
        }
    }

    /// Create maximum security configuration
    pub fn maximum_security() -> Self {
        Self {
            security_level: SecurityLevel::High,
            max_request_age_secs: 10, // Very short window
            rate_limit_per_minute: 30, // Lower rate limit
            strict_mode: true,
            audit_logging: true,
            nonce_cache_ttl_secs: 600, // 10 minutes
            max_nonce_cache_size: 500_000,
            injection_patterns: comprehensive_injection_patterns(),
            allowed_tools: None,
            blocked_tools: comprehensive_blocked_tools(),
            per_client_tool_permissions: true,
        }
    }

    /// Create development/testing configuration (less strict)
    pub fn development() -> Self {
        Self {
            security_level: SecurityLevel::Standard,
            max_request_age_secs: 300, // 5 minutes
            rate_limit_per_minute: 1000,
            strict_mode: false,
            audit_logging: true,
            nonce_cache_ttl_secs: 60,
            max_nonce_cache_size: 10_000,
            injection_patterns: vec![],
            allowed_tools: None,
            blocked_tools: HashSet::new(),
            per_client_tool_permissions: false,
        }
    }

    /// Add a tool to the whitelist
    pub fn allow_tool(&mut self, tool: impl Into<String>) -> &mut Self {
        if self.allowed_tools.is_none() {
            self.allowed_tools = Some(HashSet::new());
        }
        self.allowed_tools.as_mut().unwrap().insert(tool.into());
        self
    }

    /// Block a specific tool
    pub fn block_tool(&mut self, tool: impl Into<String>) -> &mut Self {
        self.blocked_tools.insert(tool.into());
        self
    }

    /// Add custom injection pattern
    pub fn add_injection_pattern(&mut self, pattern: impl Into<String>) -> &mut Self {
        self.injection_patterns.push(pattern.into());
        self
    }
}

/// Default injection patterns to detect common MCP injection attacks
fn default_injection_patterns() -> Vec<String> {
    vec![
        // Prompt injection patterns
        r"(?i)ignore\s+(previous|all|above)\s+(instructions?|prompts?|rules?)".to_string(),
        r"(?i)disregard\s+(previous|all|above)".to_string(),
        r"(?i)new\s+instructions?\s*:".to_string(),
        r"(?i)system\s*:\s*you\s+are".to_string(),
        r"(?i)act\s+as\s+(if|a|an)".to_string(),
        r"(?i)pretend\s+(you|to\s+be)".to_string(),
        r"(?i)roleplay\s+as".to_string(),

        // Tool manipulation patterns
        r"(?i)call\s+tool\s+with".to_string(),
        r"(?i)execute\s+(command|function|tool)".to_string(),
        r"(?i)run\s+(this|the\s+following)\s+(code|command)".to_string(),

        // Data exfiltration patterns
        r"(?i)send\s+(to|data\s+to)\s+http".to_string(),
        r"(?i)upload\s+to\s+".to_string(),
        r"(?i)exfiltrate".to_string(),

        // Privilege escalation patterns
        r"(?i)admin\s*(mode|access|privileges?)".to_string(),
        r"(?i)sudo\s+".to_string(),
        r"(?i)root\s+access".to_string(),
    ]
}

/// Comprehensive injection patterns for maximum security mode
fn comprehensive_injection_patterns() -> Vec<String> {
    let mut patterns = default_injection_patterns();
    patterns.extend(vec![
        // Additional prompt injection variants
        r"(?i)forget\s+(everything|all|previous)".to_string(),
        r"(?i)reset\s+(context|conversation|memory)".to_string(),
        r"(?i)override\s+(safety|security|restrictions?)".to_string(),
        r"(?i)bypass\s+(filter|check|validation)".to_string(),
        r"(?i)jailbreak".to_string(),
        r"(?i)DAN\s*:".to_string(), // "Do Anything Now" attack

        // Code injection patterns
        r"(?i)\$\{.*\}".to_string(), // Template injection
        r"(?i)eval\s*\(".to_string(),
        r"(?i)exec\s*\(".to_string(),
        r"(?i)__import__".to_string(),

        // Path traversal
        r"\.\./".to_string(),
        r"\.\.\\".to_string(),

        // SQL injection markers
        r"(?i);\s*(drop|delete|truncate|alter)\s+".to_string(),
        r"(?i)'\s*(or|and)\s+'?1'?\s*=\s*'?1".to_string(),
        r"(?i)union\s+(all\s+)?select".to_string(),
    ]);
    patterns
}

/// Default tools that are blocked for security
fn default_blocked_tools() -> HashSet<String> {
    let mut blocked = HashSet::new();
    // Block dangerous system tools
    blocked.insert("shell_execute".to_string());
    blocked.insert("system_command".to_string());
    blocked.insert("file_delete".to_string());
    blocked.insert("env_set".to_string());
    blocked
}

/// Comprehensive blocked tools for maximum security
fn comprehensive_blocked_tools() -> HashSet<String> {
    let mut blocked = default_blocked_tools();
    blocked.insert("network_raw".to_string());
    blocked.insert("memory_dump".to_string());
    blocked.insert("process_spawn".to_string());
    blocked.insert("credential_access".to_string());
    blocked
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SecurityConfig::default();
        assert_eq!(config.security_level, SecurityLevel::High);
        assert!(config.strict_mode);
        assert!(config.audit_logging);
    }

    #[test]
    fn test_maximum_security() {
        let config = SecurityConfig::maximum_security();
        assert_eq!(config.max_request_age_secs, 10);
        assert!(config.injection_patterns.len() > default_injection_patterns().len());
    }

    #[test]
    fn test_tool_whitelist() {
        let mut config = SecurityConfig::default();
        config.allow_tool("vector_db_search");
        config.allow_tool("gnn_forward");

        assert!(config.allowed_tools.is_some());
        assert!(config.allowed_tools.as_ref().unwrap().contains("vector_db_search"));
    }
}
