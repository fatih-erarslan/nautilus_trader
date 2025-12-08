// Compile-Time Anti-Mock Scanner - AST Pattern Detection
// Copyright (c) 2025 TENGRI Trading Swarm

use std::collections::HashMap;
use regex::Regex;

use super::Violation;

/// Compile-time scanner for mock pattern detection
#[derive(Debug)]
pub struct CompileTimeScanner {
    forbidden_patterns: Vec<ForbiddenPattern>,
    function_blacklist: Vec<String>,
    import_blacklist: Vec<String>,
    macro_blacklist: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ForbiddenPattern {
    pub pattern: Regex,
    pub description: String,
    pub violation_type: ViolationType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViolationType {
    MockFunction,
    TestData,
    SyntheticGenerator,
    ForbiddenImport,
    ForbiddenMacro,
}

impl CompileTimeScanner {
    pub fn new() -> Self {
        let forbidden_patterns = vec![
            // Mock function patterns
            ForbiddenPattern {
                pattern: Regex::new(r"\bmock\s*\(").unwrap(),
                description: "Mock function call".to_string(),
                violation_type: ViolationType::MockFunction,
            },
            ForbiddenPattern {
                pattern: Regex::new(r"\bfake_\w+").unwrap(),
                description: "Fake function identifier".to_string(),
                violation_type: ViolationType::MockFunction,
            },
            ForbiddenPattern {
                pattern: Regex::new(r"\bdummy_\w+").unwrap(),
                description: "Dummy function identifier".to_string(),
                violation_type: ViolationType::MockFunction,
            },
            
            // Test data patterns
            ForbiddenPattern {
                pattern: Regex::new(r"\btest_data\b").unwrap(),
                description: "Test data identifier".to_string(),
                violation_type: ViolationType::TestData,
            },
            ForbiddenPattern {
                pattern: Regex::new(r"\bsample_\w+").unwrap(),
                description: "Sample data identifier".to_string(),
                violation_type: ViolationType::TestData,
            },
            ForbiddenPattern {
                pattern: Regex::new(r"\bexample_\w+").unwrap(),
                description: "Example data identifier".to_string(),
                violation_type: ViolationType::TestData,
            },
            
            // Random/synthetic generators
            ForbiddenPattern {
                pattern: Regex::new(r"\brand::random\s*\(").unwrap(),
                description: "Random data generator".to_string(),
                violation_type: ViolationType::SyntheticGenerator,
            },
            ForbiddenPattern {
                pattern: Regex::new(r"\bthread_rng\s*\(\)").unwrap(),
                description: "Thread RNG generator".to_string(),
                violation_type: ViolationType::SyntheticGenerator,
            },
            ForbiddenPattern {
                pattern: Regex::new(r"\bfastrand::").unwrap(),
                description: "FastRand generator".to_string(),
                violation_type: ViolationType::SyntheticGenerator,
            },
            
            // Hardcoded test values
            ForbiddenPattern {
                pattern: Regex::new(r"\b(12345|abcde|test123|mock123)\b").unwrap(),
                description: "Hardcoded test values".to_string(),
                violation_type: ViolationType::TestData,
            },
            
            // Localhost/test endpoints
            ForbiddenPattern {
                pattern: Regex::new(r#""https?://localhost"#).unwrap(),
                description: "Localhost endpoint".to_string(),
                violation_type: ViolationType::TestData,
            },
            ForbiddenPattern {
                pattern: Regex::new(r#""https?://127\.0\.0\.1"#).unwrap(),
                description: "Local IP endpoint".to_string(),
                violation_type: ViolationType::TestData,
            },
            ForbiddenPattern {
                pattern: Regex::new(r#"".*mock.*\.com"#).unwrap(),
                description: "Mock domain endpoint".to_string(),
                violation_type: ViolationType::TestData,
            },
        ];
        
        let function_blacklist = vec![
            "mock".to_string(),
            "fake_data".to_string(),
            "dummy_data".to_string(),
            "test_data".to_string(),
            "sample_data".to_string(),
            "generate_fake".to_string(),
            "create_mock".to_string(),
            "random_data".to_string(),
        ];
        
        let import_blacklist = vec![
            "mockito".to_string(),
            "wiremock".to_string(),
            "testcontainers".to_string(),
            "fake".to_string(),
            "quickcheck".to_string(), // Only in test modules
            "proptest".to_string(),   // Only in test modules
        ];
        
        let macro_blacklist = vec![
            "mock!".to_string(),
            "fake!".to_string(),
            "test_data!".to_string(),
            "sample!".to_string(),
        ];
        
        Self {
            forbidden_patterns,
            function_blacklist,
            import_blacklist,
            macro_blacklist,
        }
    }
    
    /// Scan source code for violations
    pub fn scan_for_violations(&self, source_code: &str) -> Vec<Violation> {
        let mut violations = Vec::new();
        
        // Check for forbidden patterns
        for pattern in &self.forbidden_patterns {
            for mat in pattern.pattern.find_iter(source_code) {
                let violation = match pattern.violation_type {
                    ViolationType::MockFunction => Violation::MockFunction(mat.as_str().to_string()),
                    ViolationType::TestData => Violation::TestData(mat.as_str().to_string()),
                    ViolationType::SyntheticGenerator => Violation::SyntheticGenerator(mat.as_str().to_string()),
                    ViolationType::ForbiddenImport => Violation::ForbiddenPattern(mat.as_str().to_string()),
                    ViolationType::ForbiddenMacro => Violation::ForbiddenPattern(mat.as_str().to_string()),
                };
                violations.push(violation);
            }
        }
        
        // Check for forbidden functions
        for func in &self.function_blacklist {
            if source_code.contains(func) {
                violations.push(Violation::MockFunction(func.clone()));
            }
        }
        
        // Check for forbidden imports (outside of test modules)
        violations.extend(self.scan_imports(source_code));
        
        // Check for forbidden macros
        for macro_name in &self.macro_blacklist {
            if source_code.contains(macro_name) {
                violations.push(Violation::ForbiddenPattern(macro_name.clone()));
            }
        }
        
        violations
    }
    
    /// Scan for forbidden imports outside test modules
    fn scan_imports(&self, source_code: &str) -> Vec<Violation> {
        let mut violations = Vec::new();
        
        // Simple check - in a real implementation, you'd parse the AST
        let lines: Vec<&str> = source_code.lines().collect();
        let mut in_test_module = false;
        
        for line in lines {
            let trimmed = line.trim();
            
            // Check if entering test module
            if trimmed.contains("#[cfg(test)]") || trimmed.contains("mod tests") {
                in_test_module = true;
                continue;
            }
            
            // Check if exiting module
            if in_test_module && trimmed.starts_with('}') && !trimmed.contains('{') {
                in_test_module = false;
                continue;
            }
            
            // Skip test modules for certain imports
            if in_test_module {
                continue;
            }
            
            // Check for forbidden imports
            if trimmed.starts_with("use ") || trimmed.starts_with("extern crate ") {
                for forbidden in &self.import_blacklist {
                    if trimmed.contains(forbidden) {
                        violations.push(Violation::ForbiddenPattern(
                            format!("Forbidden import: {}", forbidden)
                        ));
                    }
                }
            }
        }
        
        violations
    }
    
    /// Check specific patterns for production readiness
    pub fn check_production_readiness(&self, source_code: &str) -> Result<(), Vec<Violation>> {
        let violations = self.scan_for_violations(source_code);
        
        // Filter critical violations
        let critical_violations: Vec<_> = violations.into_iter()
            .filter(|v| self.is_critical_violation(v))
            .collect();
        
        if critical_violations.is_empty() {
            Ok(())
        } else {
            Err(critical_violations)
        }
    }
    
    fn is_critical_violation(&self, violation: &Violation) -> bool {
        match violation {
            Violation::MockFunction(_) => true,
            Violation::SyntheticGenerator(_) => true,
            Violation::TestData(data) => {
                // Some test data patterns are more critical than others
                data.contains("localhost") || 
                data.contains("mock") || 
                data.contains("fake")
            },
            Violation::ForbiddenPattern(pattern) => {
                pattern.contains("mock") || 
                pattern.contains("fake") ||
                pattern.contains("localhost")
            },
        }
    }
    
    /// Generate detailed violation report
    pub fn generate_violation_report(&self, source_code: &str) -> ViolationReport {
        let violations = self.scan_for_violations(source_code);
        let total_violations = violations.len();
        
        let mut violation_counts = HashMap::new();
        for violation in &violations {
            let key = match violation {
                Violation::MockFunction(_) => "MockFunction",
                Violation::TestData(_) => "TestData", 
                Violation::SyntheticGenerator(_) => "SyntheticGenerator",
                Violation::ForbiddenPattern(_) => "ForbiddenPattern",
            };
            *violation_counts.entry(key.to_string()).or_insert(0) += 1;
        }
        
        let critical_violations = violations.iter()
            .filter(|v| self.is_critical_violation(v))
            .count();
        
        ViolationReport {
            total_violations,
            critical_violations,
            violation_counts,
            violations,
            production_ready: critical_violations == 0,
        }
    }
}

/// Violation report structure
#[derive(Debug)]
pub struct ViolationReport {
    pub total_violations: usize,
    pub critical_violations: usize,
    pub violation_counts: HashMap<String, usize>,
    pub violations: Vec<Violation>,
    pub production_ready: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mock_function_detection() {
        let scanner = CompileTimeScanner::new();
        
        let code_with_mock = r#"
            fn test_function() {
                let data = mock(some_params);
                process_data(data);
            }
        "#;
        
        let violations = scanner.scan_for_violations(code_with_mock);
        assert!(!violations.is_empty());
        assert!(violations.iter().any(|v| matches!(v, Violation::MockFunction(_))));
    }
    
    #[test]
    fn test_test_data_detection() {
        let scanner = CompileTimeScanner::new();
        
        let code_with_test_data = r#"
            fn process() {
                let url = "http://localhost:8080/api";
                let data = test_data.clone();
                let sample = sample_data();
            }
        "#;
        
        let violations = scanner.scan_for_violations(code_with_test_data);
        assert!(!violations.is_empty());
    }
    
    #[test]
    fn test_synthetic_generator_detection() {
        let scanner = CompileTimeScanner::new();
        
        let code_with_random = r#"
            use rand::random;
            fn generate_data() {
                let value = rand::random::<f64>();
                let rng = thread_rng();
            }
        "#;
        
        let violations = scanner.scan_for_violations(code_with_random);
        assert!(!violations.is_empty());
        assert!(violations.iter().any(|v| matches!(v, Violation::SyntheticGenerator(_))));
    }
    
    #[test]
    fn test_clean_code() {
        let scanner = CompileTimeScanner::new();
        
        let clean_code = r#"
            use std::collections::HashMap;
            use reqwest::Client;
            
            async fn fetch_real_data(api_key: &str) -> Result<MarketData> {
                let client = Client::new();
                let response = client
                    .get("https://api.binance.com/api/v3/ticker/price")
                    .header("X-MBX-APIKEY", api_key)
                    .send()
                    .await?;
                
                Ok(response.json().await?)
            }
        "#;
        
        let violations = scanner.scan_for_violations(clean_code);
        assert!(violations.is_empty());
    }
    
    #[test]
    fn test_test_module_exemption() {
        let scanner = CompileTimeScanner::new();
        
        let code_with_test_module = r#"
            use reqwest::Client;
            
            #[cfg(test)]
            mod tests {
                use super::*;
                use mockito::Server;  // This should be allowed in test modules
                
                #[test]
                fn test_api_call() {
                    let server = Server::new();
                    // Test code here
                }
            }
        "#;
        
        let violations = scanner.scan_for_violations(code_with_test_module);
        // Should not have violations for imports in test modules
        assert!(violations.iter().all(|v| !matches!(v, Violation::ForbiddenPattern(p) if p.contains("mockito"))));
    }
    
    #[test]
    fn test_production_readiness_check() {
        let scanner = CompileTimeScanner::new();
        
        let production_code = r#"
            async fn fetch_market_data() -> Result<Vec<MarketData>> {
                let client = reqwest::Client::new();
                let response = client
                    .get("https://api.binance.com/api/v3/ticker/24hr")
                    .send()
                    .await?;
                Ok(response.json().await?)
            }
        "#;
        
        assert!(scanner.check_production_readiness(production_code).is_ok());
        
        let non_production_code = r#"
            fn get_data() {
                let data = mock(test_params);
                let endpoint = "http://localhost:8080";
            }
        "#;
        
        assert!(scanner.check_production_readiness(non_production_code).is_err());
    }
    
    #[test]
    fn test_violation_report() {
        let scanner = CompileTimeScanner::new();
        
        let problematic_code = r#"
            fn test() {
                let data = mock(params);
                let fake_data = fake_data();
                let url = "http://localhost:8080";
                let random_val = rand::random::<f64>();
            }
        "#;
        
        let report = scanner.generate_violation_report(problematic_code);
        assert!(report.total_violations > 0);
        assert!(report.critical_violations > 0);
        assert!(!report.production_ready);
    }
}