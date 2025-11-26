//! Package validation with dependency analysis

use crate::{ValidationOptions, ValidationReport};
use napi::Result;
use serde_json::Value;
use std::path::PathBuf;
use std::collections::HashSet;

pub struct Validator {
    package_path: PathBuf,
}

impl Validator {
    pub fn new(package_path: String) -> Result<Self> {
        Ok(Self {
            package_path: PathBuf::from(package_path),
        })
    }

    pub async fn validate(&self, options: ValidationOptions) -> Result<ValidationReport> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut dependency_issues = Vec::new();
        let mut typescript_issues = Vec::new();

        let check_deps = options.check_dependencies.unwrap_or(true);
        let check_ts = options.check_typescript.unwrap_or(true);
        let check_napi = options.check_napi_bindings.unwrap_or(true);
        let strict = options.strict_mode.unwrap_or(false);

        // Validate package.json exists and is valid
        match self.validate_package_json().await {
            Ok(pkg_json) => {
                // Check dependencies
                if check_deps {
                    let dep_result = self.validate_dependencies(&pkg_json).await;
                    dependency_issues.extend(dep_result.issues);
                    if !dep_result.is_valid {
                        errors.push("Dependency validation failed".to_string());
                    }
                }

                // Check TypeScript definitions
                if check_ts {
                    match self.validate_typescript_definitions().await {
                        Ok(ts_issues) => {
                            if !ts_issues.is_empty() {
                                typescript_issues.extend(ts_issues.clone());
                                if strict {
                                    errors.push("TypeScript definition issues found".to_string());
                                } else {
                                    warnings.push(format!("Found {} TypeScript issues", ts_issues.len()));
                                }
                            }
                        }
                        Err(e) => errors.push(format!("TypeScript validation failed: {}", e)),
                    }
                }

                // Check NAPI bindings
                if check_napi {
                    match self.validate_napi_bindings().await {
                        Ok(napi_issues) => {
                            if !napi_issues.is_empty() {
                                warnings.extend(napi_issues);
                            }
                        }
                        Err(e) => warnings.push(format!("NAPI validation warning: {}", e)),
                    }
                }
            }
            Err(e) => {
                errors.push(format!("Invalid package.json: {}", e));
            }
        }

        let package_name = self.package_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        Ok(ValidationReport {
            package_name,
            is_valid: errors.is_empty(),
            errors,
            warnings,
            dependency_issues,
            typescript_issues,
        })
    }

    async fn validate_package_json(&self) -> Result<Value> {
        let pkg_json_path = self.package_path.join("package.json");

        let content = tokio::fs::read_to_string(&pkg_json_path)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to read package.json: {}", e)))?;

        serde_json::from_str(&content)
            .map_err(|e| napi::Error::from_reason(format!("Invalid JSON: {}", e)))
    }

    async fn validate_dependencies(&self, pkg_json: &Value) -> DependencyValidationResult {
        let mut issues = Vec::new();
        let mut is_valid = true;

        // Check for required dependencies
        if let Some(deps) = pkg_json.get("dependencies").and_then(|d| d.as_object()) {
            // Check for circular dependencies
            let circular = self.detect_circular_dependencies(deps).await;
            if !circular.is_empty() {
                issues.push(format!("Circular dependencies detected: {:?}", circular));
                is_valid = false;
            }

            // Check for outdated dependencies
            let outdated = self.check_outdated_dependencies(deps).await;
            if !outdated.is_empty() {
                issues.push(format!("Outdated dependencies: {:?}", outdated));
            }

            // Check for duplicate dependencies
            if let Some(dev_deps) = pkg_json.get("devDependencies").and_then(|d| d.as_object()) {
                let duplicates = self.find_duplicate_dependencies(deps, dev_deps);
                if !duplicates.is_empty() {
                    issues.push(format!("Duplicate dependencies: {:?}", duplicates));
                }
            }
        } else {
            issues.push("No dependencies found".to_string());
        }

        DependencyValidationResult { is_valid, issues }
    }

    async fn validate_typescript_definitions(&self) -> Result<Vec<String>> {
        let mut issues = Vec::new();

        // Check for index.d.ts or types field in package.json
        let dts_path = self.package_path.join("index.d.ts");
        let types_path = self.package_path.join("types/index.d.ts");

        if !dts_path.exists() && !types_path.exists() {
            issues.push("No TypeScript definitions found (index.d.ts missing)".to_string());
        }

        // Check for proper exports in definition files
        if dts_path.exists() {
            let content = tokio::fs::read_to_string(&dts_path).await?;
            if !content.contains("export") {
                issues.push("TypeScript definitions missing exports".to_string());
            }
        }

        Ok(issues)
    }

    async fn validate_napi_bindings(&self) -> Result<Vec<String>> {
        let mut warnings = Vec::new();

        // Check for .node binary
        let binding_path = self.package_path.join("index.node");
        if !binding_path.exists() {
            warnings.push("NAPI binding file (index.node) not found".to_string());
        }

        // Check for napi dependencies
        let pkg_json = self.validate_package_json().await?;
        if let Some(deps) = pkg_json.get("dependencies").and_then(|d| d.as_object()) {
            if !deps.contains_key("@napi-rs/cli") && !deps.contains_key("napi") {
                warnings.push("Missing NAPI dependencies in package.json".to_string());
            }
        }

        Ok(warnings)
    }

    async fn detect_circular_dependencies(&self, _deps: &serde_json::Map<String, Value>) -> Vec<String> {
        // Simplified circular dependency detection
        // In production, this would traverse the full dependency graph
        Vec::new()
    }

    async fn check_outdated_dependencies(&self, _deps: &serde_json::Map<String, Value>) -> Vec<String> {
        // Simplified outdated dependency check
        // In production, this would query npm registry for latest versions
        Vec::new()
    }

    fn find_duplicate_dependencies(
        &self,
        deps: &serde_json::Map<String, Value>,
        dev_deps: &serde_json::Map<String, Value>,
    ) -> Vec<String> {
        let deps_set: HashSet<_> = deps.keys().collect();
        let dev_deps_set: HashSet<_> = dev_deps.keys().collect();

        deps_set
            .intersection(&dev_deps_set)
            .map(|s| s.to_string())
            .collect()
    }
}

struct DependencyValidationResult {
    is_valid: bool,
    issues: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_duplicate_detection() {
        let validator = Validator {
            package_path: PathBuf::from("/tmp"),
        };

        let mut deps = serde_json::Map::new();
        deps.insert("express".to_string(), Value::String("4.0.0".to_string()));

        let mut dev_deps = serde_json::Map::new();
        dev_deps.insert("express".to_string(), Value::String("4.0.0".to_string()));

        let duplicates = validator.find_duplicate_dependencies(&deps, &dev_deps);
        assert_eq!(duplicates.len(), 1);
        assert_eq!(duplicates[0], "express");
    }
}
