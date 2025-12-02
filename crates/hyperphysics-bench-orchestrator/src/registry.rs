//! Benchmark Registry - Auto-discovers targets across workspace
//!
//! Scans Cargo.toml files to find `[[bench]]` and `[[example]]` targets.

use crate::{OrchestratorError, Result, Target, TargetKind};
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Registry of discovered benchmark/example/test targets
#[derive(Debug)]
pub struct Registry {
    workspace_root: PathBuf,
    targets: Vec<Target>,
}

impl Registry {
    /// Create a new registry for the given workspace
    pub fn new(workspace_root: PathBuf) -> Self {
        Self {
            workspace_root,
            targets: Vec::new(),
        }
    }

    /// Discover all targets in the workspace
    pub fn discover(&mut self) -> Result<()> {
        self.targets.clear();

        // Find all Cargo.toml files in the workspace
        for entry in WalkDir::new(&self.workspace_root)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name() == "Cargo.toml")
        {
            let cargo_toml_path = entry.path();

            // Skip workspace root Cargo.toml (already processed) and vendor directories
            if self.should_skip_path(cargo_toml_path) {
                continue;
            }

            if let Err(e) = self.parse_cargo_toml(cargo_toml_path) {
                // Log but don't fail on individual parse errors
                eprintln!(
                    "Warning: Failed to parse {}: {}",
                    cargo_toml_path.display(),
                    e
                );
            }
        }

        // Sort targets by crate name, then by name
        self.targets.sort_by(|a, b| {
            a.crate_name
                .cmp(&b.crate_name)
                .then_with(|| a.name.cmp(&b.name))
        });

        Ok(())
    }

    /// Check if path should be skipped
    fn should_skip_path(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();

        // Skip vendor directories, target, and other non-relevant paths
        path_str.contains("/vendor/")
            || path_str.contains("/target/")
            || path_str.contains("/cwts-ultra/")
            || path_str.contains("/.git/")
            || path_str.contains("/node_modules/")
    }

    /// Parse a Cargo.toml file for targets
    fn parse_cargo_toml(&mut self, path: &Path) -> Result<()> {
        let content = fs::read_to_string(path)?;
        let toml_value: toml::Value = toml::from_str(&content)?;

        let crate_path = path
            .parent()
            .ok_or_else(|| OrchestratorError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "No parent directory",
            )))?
            .to_path_buf();

        // Extract crate name
        let crate_name = toml_value
            .get("package")
            .and_then(|p| p.get("name"))
            .and_then(|n| n.as_str())
            .unwrap_or("unknown")
            .to_string();

        // Skip if not a hyperphysics crate (focus on core workspace)
        if !self.is_hyperphysics_crate(&crate_name, &crate_path) {
            return Ok(());
        }

        // Parse [[bench]] targets
        if let Some(benches) = toml_value.get("bench").and_then(|b| b.as_array()) {
            for bench in benches {
                if let Some(name) = bench.get("name").and_then(|n| n.as_str()) {
                    let mut target = Target::new(
                        name,
                        &crate_name,
                        crate_path.clone(),
                        TargetKind::Benchmark,
                    );

                    // Try to extract description from bench file
                    target.description = self.extract_description(&crate_path, "benches", name);
                    target.tags = self.infer_tags(&crate_name, name);

                    self.targets.push(target);
                }
            }
        }

        // Parse [[example]] targets
        if let Some(examples) = toml_value.get("example").and_then(|e| e.as_array()) {
            for example in examples {
                if let Some(name) = example.get("name").and_then(|n| n.as_str()) {
                    let mut target = Target::new(
                        name,
                        &crate_name,
                        crate_path.clone(),
                        TargetKind::Example,
                    );

                    target.description = self.extract_description(&crate_path, "examples", name);
                    target.tags = self.infer_tags(&crate_name, name);

                    self.targets.push(target);
                }
            }
        }

        // Also scan for examples directory (implicit examples)
        let examples_dir = crate_path.join("examples");
        if examples_dir.exists() && examples_dir.is_dir() {
            if let Ok(entries) = fs::read_dir(&examples_dir) {
                for entry in entries.filter_map(|e| e.ok()) {
                    let file_name = entry.file_name();
                    let name = file_name.to_string_lossy();

                    if name.ends_with(".rs") && !name.starts_with('_') {
                        let example_name = name.trim_end_matches(".rs");

                        // Check if already registered
                        let already_exists = self.targets.iter().any(|t| {
                            t.crate_name == crate_name
                                && t.name == example_name
                                && t.kind == TargetKind::Example
                        });

                        if !already_exists {
                            let mut target = Target::new(
                                example_name,
                                &crate_name,
                                crate_path.clone(),
                                TargetKind::Example,
                            );

                            target.description =
                                self.extract_description(&crate_path, "examples", example_name);
                            target.tags = self.infer_tags(&crate_name, example_name);

                            self.targets.push(target);
                        }
                    }
                }
            }
        }

        // Also scan for benches directory (implicit benchmarks)
        let benches_dir = crate_path.join("benches");
        if benches_dir.exists() && benches_dir.is_dir() {
            if let Ok(entries) = fs::read_dir(&benches_dir) {
                for entry in entries.filter_map(|e| e.ok()) {
                    let file_name = entry.file_name();
                    let name = file_name.to_string_lossy();

                    if name.ends_with(".rs") && !name.starts_with('_') {
                        let bench_name = name.trim_end_matches(".rs");

                        // Check if already registered
                        let already_exists = self.targets.iter().any(|t| {
                            t.crate_name == crate_name
                                && t.name == bench_name
                                && t.kind == TargetKind::Benchmark
                        });

                        if !already_exists {
                            let mut target = Target::new(
                                bench_name,
                                &crate_name,
                                crate_path.clone(),
                                TargetKind::Benchmark,
                            );

                            target.description =
                                self.extract_description(&crate_path, "benches", bench_name);
                            target.tags = self.infer_tags(&crate_name, bench_name);

                            self.targets.push(target);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Check if crate is part of hyperphysics workspace
    fn is_hyperphysics_crate(&self, name: &str, path: &Path) -> bool {
        // Include crates that start with hyperphysics- or are in core workspace
        name.starts_with("hyperphysics-")
            || name == "active-inference-agent"
            || name == "gpu-marl"
            || name == "holographic-embeddings"
            || name == "ising-optimizer"
            || name == "game-theory-engine"
            || name == "lmsr"
            || name == "prospect-theory"
            || name == "ats-core"
            || name == "quantum-circuit"
            || name == "hyper-risk-engine"
            || path.starts_with(self.workspace_root.join("crates"))
    }

    /// Extract description from source file doc comments
    fn extract_description(&self, crate_path: &Path, dir: &str, name: &str) -> Option<String> {
        let file_path = crate_path.join(dir).join(format!("{}.rs", name));

        if let Ok(content) = fs::read_to_string(&file_path) {
            // Extract //! doc comments
            let mut desc_lines = Vec::new();
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("//!") {
                    let doc = trimmed.trim_start_matches("//!").trim();
                    if !doc.is_empty() && !doc.starts_with('#') && !doc.starts_with("```") {
                        desc_lines.push(doc.to_string());
                        if desc_lines.len() >= 2 {
                            break;
                        }
                    }
                } else if !trimmed.is_empty() && !trimmed.starts_with("//") {
                    break;
                }
            }

            if !desc_lines.is_empty() {
                return Some(desc_lines.join(" "));
            }
        }

        None
    }

    /// Infer tags from crate and target names
    fn infer_tags(&self, crate_name: &str, target_name: &str) -> Vec<String> {
        let mut tags = Vec::new();
        let combined = format!("{} {}", crate_name, target_name).to_lowercase();

        // Performance category tags
        if combined.contains("hnsw") || combined.contains("query") || combined.contains("search") {
            tags.push("vector-search".to_string());
        }
        if combined.contains("lsh") || combined.contains("hash") {
            tags.push("locality-sensitive-hashing".to_string());
        }
        if combined.contains("pbit") || combined.contains("ising") || combined.contains("metropolis") {
            tags.push("physics-simulation".to_string());
        }
        if combined.contains("thermo") || combined.contains("entropy") || combined.contains("hamiltonian") {
            tags.push("thermodynamics".to_string());
        }
        if combined.contains("gpu") || combined.contains("cuda") || combined.contains("metal") {
            tags.push("gpu-accelerated".to_string());
        }
        if combined.contains("simd") || combined.contains("vectorized") {
            tags.push("simd-optimized".to_string());
        }
        if combined.contains("cortical") || combined.contains("spike") || combined.contains("neural") {
            tags.push("neuromorphic".to_string());
        }
        if combined.contains("markov") || combined.contains("inference") || combined.contains("belief") {
            tags.push("probabilistic-inference".to_string());
        }
        if combined.contains("distance") || combined.contains("metric") || combined.contains("similarity") {
            tags.push("distance-metrics".to_string());
        }
        if combined.contains("landauer") || combined.contains("reversible") {
            tags.push("thermodynamic-computing".to_string());
        }

        tags
    }

    /// Get all discovered targets
    pub fn targets(&self) -> &[Target] {
        &self.targets
    }

    /// Get targets by kind
    pub fn targets_by_kind(&self, kind: TargetKind) -> Vec<&Target> {
        self.targets.iter().filter(|t| t.kind == kind).collect()
    }

    /// Get targets by crate
    pub fn targets_by_crate(&self, crate_name: &str) -> Vec<&Target> {
        self.targets
            .iter()
            .filter(|t| t.crate_name == crate_name)
            .collect()
    }

    /// Get unique crate names
    pub fn crates(&self) -> Vec<&str> {
        let mut crates: Vec<&str> = self.targets.iter().map(|t| t.crate_name.as_str()).collect();
        crates.sort();
        crates.dedup();
        crates
    }

    /// Get statistics
    pub fn stats(&self) -> RegistryStats {
        let benchmarks = self.targets.iter().filter(|t| t.kind == TargetKind::Benchmark).count();
        let examples = self.targets.iter().filter(|t| t.kind == TargetKind::Example).count();
        let tests = self.targets.iter().filter(|t| t.kind == TargetKind::Test).count();

        RegistryStats {
            total_targets: self.targets.len(),
            benchmarks,
            examples,
            tests,
            crates: self.crates().len(),
        }
    }
}

/// Registry statistics
#[derive(Debug, Clone)]
pub struct RegistryStats {
    pub total_targets: usize,
    pub benchmarks: usize,
    pub examples: usize,
    pub tests: usize,
    pub crates: usize,
}

impl std::fmt::Display for RegistryStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} targets ({} benchmarks, {} examples, {} tests) across {} crates",
            self.total_targets, self.benchmarks, self.examples, self.tests, self.crates
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_tags() {
        let registry = Registry::new(PathBuf::from("/tmp"));

        let tags = registry.infer_tags("hyperphysics-hnsw", "query_latency");
        assert!(tags.contains(&"vector-search".to_string()));

        let tags = registry.infer_tags("hyperphysics-thermo", "entropy");
        assert!(tags.contains(&"thermodynamics".to_string()));
    }
}
