//! Wolfram Language Evaluation
//!
//! Provides async evaluation of Wolfram Language code via WolframScript.

use crate::discovery::get_default_installation;
use crate::types::*;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tokio::process::Command;
use tokio::time::timeout;
use tracing::{debug, info};

/// Main evaluator for Wolfram Language code
pub struct WolframEvaluator {
    installation: WolframInstallation,
}

impl WolframEvaluator {
    /// Create a new evaluator with the default installation
    pub fn new() -> WolframResult<Self> {
        let installation = get_default_installation()?;
        info!(
            "Using Wolfram installation: {} ({})",
            installation.product_name, installation.version
        );
        Ok(Self { installation })
    }

    /// Create an evaluator with a specific installation
    pub fn with_installation(installation: WolframInstallation) -> Self {
        Self { installation }
    }

    /// Get the current installation info
    pub fn installation(&self) -> &WolframInstallation {
        &self.installation
    }

    /// Evaluate Wolfram Language code
    pub async fn evaluate(
        &self,
        code: &str,
        options: Option<EvaluationOptions>,
    ) -> WolframResult<EvaluationResult> {
        let opts = options.unwrap_or_default();
        let start = Instant::now();

        let wolframscript = &self.installation.wolfram_script_path;
        if wolframscript.is_empty() {
            return Err(WolframError::WolframScriptNotFound);
        }

        // Prepare the code with output formatting
        let formatted_code = match opts.format.as_str() {
            "json" => format!("ExportString[{}, \"JSON\", \"Compact\" -> True]", code),
            "inputform" => format!("InputForm[{}]", code),
            "fullform" => format!("FullForm[{}]", code),
            _ => code.to_string(),
        };

        debug!("Executing WolframScript: {}", formatted_code);

        // Execute WolframScript with timeout
        let result = timeout(
            Duration::from_secs(opts.timeout_seconds as u64),
            self.execute_wolframscript(&formatted_code, &opts),
        )
        .await;

        let execution_time = start.elapsed().as_millis() as i64;

        match result {
            Ok(Ok((stdout, stderr))) => {
                let success = !stdout.contains("$Failed")
                    && !stdout.contains("Syntax::")
                    && !stderr.contains("error");

                let messages: Vec<String> = if opts.capture_messages && !stderr.is_empty() {
                    stderr.lines().map(|s| s.to_string()).collect()
                } else {
                    Vec::new()
                };

                Ok(EvaluationResult {
                    result: stdout.trim().to_string(),
                    success,
                    error: if success { None } else { Some(stderr) },
                    messages,
                    execution_time_ms: execution_time,
                    format: opts.format,
                })
            }
            Ok(Err(e)) => Ok(EvaluationResult {
                result: String::new(),
                success: false,
                error: Some(e.to_string()),
                messages: Vec::new(),
                execution_time_ms: execution_time,
                format: opts.format,
            }),
            Err(_) => Err(WolframError::Timeout(opts.timeout_seconds as u64)),
        }
    }

    /// Execute WolframScript and return stdout/stderr
    async fn execute_wolframscript(
        &self,
        code: &str,
        _options: &EvaluationOptions,
    ) -> WolframResult<(String, String)> {
        let mut cmd = Command::new(&self.installation.wolfram_script_path);
        cmd.arg("-code")
            .arg(code)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let child = cmd.spawn()?;
        let output = child.wait_with_output().await?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if !output.status.success() && stdout.is_empty() {
            return Err(WolframError::ExecutionFailed(stderr));
        }

        Ok((stdout, stderr))
    }

    /// Evaluate and parse as JSON
    pub async fn evaluate_json<T: serde::de::DeserializeOwned>(
        &self,
        code: &str,
    ) -> WolframResult<T> {
        let result = self
            .evaluate(
                code,
                Some(EvaluationOptions {
                    format: "json".to_string(),
                    ..Default::default()
                }),
            )
            .await?;

        if !result.success {
            return Err(WolframError::KernelError(
                result.error.unwrap_or_default(),
            ));
        }

        serde_json::from_str(&result.result).map_err(|e| {
            WolframError::ParseError(format!("Failed to parse JSON: {} - Input: {}", e, result.result))
        })
    }

    /// Evaluate and return as f64
    pub async fn evaluate_numeric(&self, code: &str) -> WolframResult<f64> {
        let result = self
            .evaluate(
                &format!("N[{}]", code),
                Some(EvaluationOptions {
                    format: "text".to_string(),
                    ..Default::default()
                }),
            )
            .await?;

        if !result.success {
            return Err(WolframError::KernelError(
                result.error.unwrap_or_default(),
            ));
        }

        result.result.trim().parse::<f64>().map_err(|e| {
            WolframError::ParseError(format!("Failed to parse number: {} - Input: {}", e, result.result))
        })
    }

    /// Get Wolfram version
    pub async fn get_version(&self) -> WolframResult<String> {
        let result = self
            .evaluate("$Version // ToString", None)
            .await?;

        if result.success {
            Ok(result.result)
        } else {
            Err(WolframError::KernelError(
                result.error.unwrap_or_else(|| "Unknown error".to_string()),
            ))
        }
    }

    /// Check if the evaluator is working
    pub async fn health_check(&self) -> WolframResult<bool> {
        let result = self.evaluate("2 + 2", None).await?;
        Ok(result.success && result.result.contains('4'))
    }
}

impl Default for WolframEvaluator {
    fn default() -> Self {
        Self::new().expect("Failed to create default WolframEvaluator")
    }
}

/// Convenience function to evaluate code with default settings
pub async fn evaluate_code(code: &str) -> WolframResult<EvaluationResult> {
    let evaluator = WolframEvaluator::new()?;
    evaluator.evaluate(code, None).await
}

/// Evaluate code with custom options
pub async fn evaluate_with_options(
    code: &str,
    options: EvaluationOptions,
) -> WolframResult<EvaluationResult> {
    let evaluator = WolframEvaluator::new()?;
    evaluator.evaluate(code, Some(options)).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_simple_evaluation() {
        if let Ok(evaluator) = WolframEvaluator::new() {
            let result = evaluator.evaluate("2 + 2", None).await;
            if let Ok(res) = result {
                assert!(res.result.contains('4'));
            }
        }
    }

    #[tokio::test]
    async fn test_numeric_evaluation() {
        if let Ok(evaluator) = WolframEvaluator::new() {
            if let Ok(result) = evaluator.evaluate_numeric("Pi").await {
                assert!((result - std::f64::consts::PI).abs() < 1e-10);
            }
        }
    }
}
