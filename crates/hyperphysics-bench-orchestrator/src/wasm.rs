//! WASM bindings for browser-based benchmark dashboard
//!
//! Provides JavaScript interoperability for running the orchestrator in browsers.

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
use crate::{
    collector::{BenchmarkReport, Collector},
    registry::Registry,
    reporter::Reporter,
    OrchestratorConfig, Target, TargetKind,
};

#[cfg(feature = "wasm")]
use std::path::PathBuf;

/// WASM-compatible orchestrator wrapper
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmOrchestrator {
    targets: Vec<Target>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmOrchestrator {
    /// Create new orchestrator (WASM entry point)
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        // Set panic hook for better error messages in browser
        console_error_panic_hook::set_once();

        Self {
            targets: Vec::new(),
        }
    }

    /// Load targets from JSON data
    #[wasm_bindgen]
    pub fn load_targets_json(&mut self, json: &str) -> Result<usize, JsValue> {
        let targets: Vec<Target> = serde_json::from_str(json)
            .map_err(|e| JsValue::from_str(&format!("Parse error: {}", e)))?;

        let count = targets.len();
        self.targets = targets;
        Ok(count)
    }

    /// Get target count
    #[wasm_bindgen]
    pub fn target_count(&self) -> usize {
        self.targets.len()
    }

    /// Get targets as JSON
    #[wasm_bindgen]
    pub fn targets_json(&self) -> Result<String, JsValue> {
        serde_json::to_string_pretty(&self.targets)
            .map_err(|e| JsValue::from_str(&format!("Serialize error: {}", e)))
    }

    /// Filter targets by kind
    #[wasm_bindgen]
    pub fn filter_by_kind(&self, kind: &str) -> Result<String, JsValue> {
        let kind = match kind {
            "benchmark" | "bench" => TargetKind::Benchmark,
            "example" => TargetKind::Example,
            "test" => TargetKind::Test,
            _ => return Err(JsValue::from_str("Invalid kind: use benchmark, example, or test")),
        };

        let filtered: Vec<&Target> = self.targets.iter().filter(|t| t.kind == kind).collect();

        serde_json::to_string_pretty(&filtered)
            .map_err(|e| JsValue::from_str(&format!("Serialize error: {}", e)))
    }

    /// Filter targets by crate name
    #[wasm_bindgen]
    pub fn filter_by_crate(&self, crate_name: &str) -> Result<String, JsValue> {
        let filtered: Vec<&Target> = self
            .targets
            .iter()
            .filter(|t| t.crate_name == crate_name)
            .collect();

        serde_json::to_string_pretty(&filtered)
            .map_err(|e| JsValue::from_str(&format!("Serialize error: {}", e)))
    }

    /// Filter targets by tag
    #[wasm_bindgen]
    pub fn filter_by_tag(&self, tag: &str) -> Result<String, JsValue> {
        let filtered: Vec<&Target> = self
            .targets
            .iter()
            .filter(|t| t.tags.iter().any(|t| t == tag))
            .collect();

        serde_json::to_string_pretty(&filtered)
            .map_err(|e| JsValue::from_str(&format!("Serialize error: {}", e)))
    }

    /// Get unique crate names
    #[wasm_bindgen]
    pub fn crates(&self) -> Result<String, JsValue> {
        let mut crates: Vec<&str> = self.targets.iter().map(|t| t.crate_name.as_str()).collect();
        crates.sort();
        crates.dedup();

        serde_json::to_string_pretty(&crates)
            .map_err(|e| JsValue::from_str(&format!("Serialize error: {}", e)))
    }

    /// Get unique tags
    #[wasm_bindgen]
    pub fn tags(&self) -> Result<String, JsValue> {
        let mut tags: Vec<&str> = self
            .targets
            .iter()
            .flat_map(|t| t.tags.iter().map(|s| s.as_str()))
            .collect();
        tags.sort();
        tags.dedup();

        serde_json::to_string_pretty(&tags)
            .map_err(|e| JsValue::from_str(&format!("Serialize error: {}", e)))
    }

    /// Generate HTML report from JSON results
    #[wasm_bindgen]
    pub fn generate_html_report(results_json: &str) -> Result<String, JsValue> {
        let report: BenchmarkReport = serde_json::from_str(results_json)
            .map_err(|e| JsValue::from_str(&format!("Parse error: {}", e)))?;

        Ok(Reporter::to_html(&report, None))
    }

    /// Compare two reports and return comparison JSON
    #[wasm_bindgen]
    pub fn compare_reports(baseline_json: &str, current_json: &str) -> Result<String, JsValue> {
        let baseline: BenchmarkReport = serde_json::from_str(baseline_json)
            .map_err(|e| JsValue::from_str(&format!("Baseline parse error: {}", e)))?;

        let current: BenchmarkReport = serde_json::from_str(current_json)
            .map_err(|e| JsValue::from_str(&format!("Current parse error: {}", e)))?;

        // Create collector and add results
        let mut collector = Collector::new();
        for result in current.results {
            collector.add(result);
        }

        let comparison = collector.compare_with_baseline(&baseline);

        serde_json::to_string_pretty(&comparison)
            .map_err(|e| JsValue::from_str(&format!("Serialize error: {}", e)))
    }

    /// Format time value (exported for JS use)
    #[wasm_bindgen]
    pub fn format_time(ns: f64) -> String {
        if ns >= 1_000_000_000.0 {
            format!("{:.2}s", ns / 1_000_000_000.0)
        } else if ns >= 1_000_000.0 {
            format!("{:.2}ms", ns / 1_000_000.0)
        } else if ns >= 1_000.0 {
            format!("{:.2}µs", ns / 1_000.0)
        } else {
            format!("{:.0}ns", ns)
        }
    }

    /// Calculate percentage change
    #[wasm_bindgen]
    pub fn calculate_change(baseline: f64, current: f64) -> f64 {
        if baseline == 0.0 {
            return 0.0;
        }
        (current - baseline) / baseline * 100.0
    }
}

#[cfg(feature = "wasm")]
impl Default for WasmOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

/// Log to browser console
#[cfg(feature = "wasm")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
    #[wasm_bindgen(js_namespace = console)]
    fn warn(s: &str);
    #[wasm_bindgen(js_namespace = console)]
    fn error(s: &str);
}

/// Console logging helper
#[cfg(feature = "wasm")]
pub fn console_log(msg: &str) {
    log(msg);
}

/// Console warning helper
#[cfg(feature = "wasm")]
pub fn console_warn(msg: &str) {
    warn(msg);
}

/// Console error helper
#[cfg(feature = "wasm")]
pub fn console_error(msg: &str) {
    error(msg);
}
