//! Zero-Mock Validation System
//!
//! Implements comprehensive mock detection and rejection system to enforce
//! 100% real implementation policy across all CQGS-monitored systems.

use dashmap::DashMap;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs;
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info, instrument, warn};
use uuid::Uuid;

use crate::cqgs::sentinels::SentinelId;
use crate::cqgs::{QualityViolation, ViolationSeverity};

/// Mock detection patterns for various programming languages and frameworks
static MOCK_PATTERNS: &[(&str, &str)] = &[
    // Rust patterns
    (r"(?i)\bmock\w*::", "Rust mock usage"),
    (r"(?i)MockObject|MockTrait", "Rust mock object/trait"),
    (r"(?i)#\[cfg\(test\)\].*mock", "Rust test mock"),
    (r"(?i)\.expect\(\)\.return", "Rust mockall expect"),
    // JavaScript/TypeScript patterns
    (r"(?i)jest\.mock\(", "Jest mock function"),
    (
        r"(?i)sinon\.mock|sinon\.stub|sinon\.spy",
        "Sinon mock/stub/spy",
    ),
    (r"(?i)MockedClass|MockedFunction", "TypeScript mocked types"),
    (r"(?i)vi\.mock\(", "Vitest mock function"),
    // Python patterns
    (r"(?i)unittest\.mock|mock\.Mock", "Python unittest mock"),
    (r"(?i)pytest-mock|mocker\.", "Pytest mock"),
    (r"(?i)@mock\.patch|@patch", "Python mock patch decorator"),
    (r"(?i)MagicMock|Mock\(\)", "Python mock objects"),
    // Java patterns
    (r"(?i)Mockito\.|@Mock", "Java Mockito framework"),
    (r"(?i)PowerMockito|EasyMock", "Java mock frameworks"),
    (r"(?i)when\(.*\)\.thenReturn", "Java mock when-then"),
    // C# patterns
    (r"(?i)Moq\.|Mock<", "C# Moq framework"),
    (r"(?i)NSubstitute\.|Substitute\.", "C# NSubstitute"),
    // Go patterns
    (r"(?i)gomock|mockgen", "Go mock generation"),
    (r"(?i)testify/mock", "Go testify mock"),
    // Generic patterns
    (
        r"(?i)\bfake\w*class|\bfake\w*service",
        "Fake implementations",
    ),
    (
        r"(?i)\bdummy\w*data|\bdummy\w*service",
        "Dummy implementations",
    ),
    (r"(?i)\bstub\w*implementation", "Stub implementations"),
    (r"(?i)test\.double|TestDouble", "Test double patterns"),
];

/// File extensions to scan for mock implementations
static SCANNABLE_EXTENSIONS: &[&str] = &[
    "rs", "js", "ts", "jsx", "tsx", "py", "java", "scala", "kt", "cs", "go", "cpp", "hpp", "c",
    "h", "rb", "php", "swift",
];

/// Directories to exclude from mock scanning
static EXCLUDED_DIRECTORIES: &[&str] = &[
    "node_modules",
    "target",
    "build",
    "dist",
    ".git",
    "vendor",
    "__pycache__",
    ".pytest_cache",
    "coverage",
    "test-results",
];

/// Maximum file size to scan (in bytes)
const MAX_SCAN_FILE_SIZE: u64 = 1024 * 1024; // 1MB

/// Mock detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockDetectionResult {
    pub file_path: PathBuf,
    pub line_number: usize,
    pub column: usize,
    pub pattern_matched: String,
    pub context: String,
    pub severity: MockSeverity,
    pub description: String,
    pub remediation_suggestion: String,
}

/// Severity levels for mock violations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MockSeverity {
    Info,     // Detected but may be acceptable
    Warning,  // Should be reviewed
    Error,    // Violates policy
    Critical, // Blocks deployment
}

/// Mock validation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationStats {
    pub files_scanned: u64,
    pub lines_scanned: u64,
    pub mocks_detected: u64,
    pub violations_by_severity: HashMap<MockSeverity, u64>,
    pub patterns_matched: HashMap<String, u64>,
    pub scan_duration_ms: u64,
    pub last_scan: std::time::SystemTime,
}

impl Default for ValidationStats {
    fn default() -> Self {
        Self {
            files_scanned: 0,
            lines_scanned: 0,
            mocks_detected: 0,
            violations_by_severity: HashMap::new(),
            patterns_matched: HashMap::new(),
            scan_duration_ms: 0,
            last_scan: std::time::SystemTime::now(),
        }
    }
}

/// Zero-Mock Validator with comprehensive detection capabilities
pub struct ZeroMockValidator {
    compiled_patterns: Vec<(Regex, String)>,
    violation_cache: Arc<DashMap<PathBuf, Vec<MockDetectionResult>>>,
    stats: Arc<Mutex<ValidationStats>>,
    whitelist: Arc<RwLock<HashSet<PathBuf>>>,
    custom_patterns: Arc<RwLock<Vec<(String, String)>>>,
    enforcement_level: Arc<RwLock<EnforcementLevel>>,
}

/// Enforcement levels for mock validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnforcementLevel {
    Permissive,    // Log only
    Standard,      // Warn and report
    Strict,        // Block on violations
    ZeroTolerance, // Block on any mock detection
}

impl ZeroMockValidator {
    /// Create new zero-mock validator
    pub fn new() -> Self {
        let compiled_patterns: Vec<(Regex, String)> = MOCK_PATTERNS
            .iter()
            .filter_map(|(pattern, desc)| match Regex::new(pattern) {
                Ok(regex) => Some((regex, desc.to_string())),
                Err(e) => {
                    warn!("Failed to compile mock pattern '{}': {}", pattern, e);
                    None
                }
            })
            .collect();

        info!(
            "Compiled {} mock detection patterns",
            compiled_patterns.len()
        );

        Self {
            compiled_patterns,
            violation_cache: Arc::new(DashMap::new()),
            stats: Arc::new(Mutex::new(ValidationStats::default())),
            whitelist: Arc::new(RwLock::new(HashSet::new())),
            custom_patterns: Arc::new(RwLock::new(Vec::new())),
            enforcement_level: Arc::new(RwLock::new(EnforcementLevel::Strict)),
        }
    }

    /// Scan directory tree for mock implementations
    #[instrument(skip(self), fields(path = %root_path.display()))]
    pub async fn scan_directory(
        &self,
        root_path: &Path,
    ) -> Result<Vec<MockDetectionResult>, Box<dyn std::error::Error + Send + Sync>> {
        let start_time = std::time::Instant::now();
        let mut all_violations = Vec::new();
        let mut stats = ValidationStats::default();

        info!(
            "Starting zero-mock validation scan of {}",
            root_path.display()
        );

        // Clear cache for fresh scan
        self.violation_cache.clear();

        // Recursively scan directory
        self.scan_directory_recursive(root_path, &mut all_violations, &mut stats)
            .await?;

        // Update statistics
        stats.scan_duration_ms = start_time.elapsed().as_millis() as u64;
        stats.last_scan = std::time::SystemTime::now();
        *self.stats.lock().await = stats.clone();

        info!(
            "Mock scan completed: {} files, {} violations found in {}ms",
            stats.files_scanned, stats.mocks_detected, stats.scan_duration_ms
        );

        Ok(all_violations)
    }

    /// Recursively scan directory tree
    fn scan_directory_recursive<'a>(
        &'a self,
        dir_path: &'a Path,
        violations: &'a mut Vec<MockDetectionResult>,
        stats: &'a mut ValidationStats,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<(), Box<dyn std::error::Error + Send + Sync>>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(async move {
            // Skip excluded directories
            if let Some(dir_name) = dir_path.file_name().and_then(|n| n.to_str()) {
                if EXCLUDED_DIRECTORIES.contains(&dir_name) {
                    debug!("Skipping excluded directory: {}", dir_path.display());
                    return Ok(());
                }
            }

            let mut entries = fs::read_dir(dir_path).await?;

            while let Some(entry) = entries.next_entry().await? {
                let path = entry.path();

                if path.is_dir() {
                    // Recursively scan subdirectories
                    self.scan_directory_recursive(&path, violations, stats)
                        .await?;
                } else if path.is_file() {
                    // Check if file should be scanned
                    if self.should_scan_file(&path) {
                        match self.scan_file(&path).await {
                            Ok(file_violations) => {
                                stats.files_scanned += 1;
                                stats.mocks_detected += file_violations.len() as u64;
                                violations.extend(file_violations);
                            }
                            Err(e) => {
                                warn!("Failed to scan file {}: {}", path.display(), e);
                            }
                        }
                    }
                }
            }

            Ok(())
        })
    }

    /// Check if file should be scanned for mocks
    fn should_scan_file(&self, file_path: &Path) -> bool {
        // Check file extension
        if let Some(extension) = file_path.extension().and_then(|ext| ext.to_str()) {
            if !SCANNABLE_EXTENSIONS.contains(&extension) {
                return false;
            }
        } else {
            return false;
        }

        // Check file size
        if let Ok(metadata) = std::fs::metadata(file_path) {
            if metadata.len() > MAX_SCAN_FILE_SIZE {
                debug!(
                    "Skipping large file: {} ({} bytes)",
                    file_path.display(),
                    metadata.len()
                );
                return false;
            }
        }

        true
    }

    /// Scan individual file for mock implementations
    #[instrument(skip(self), fields(file = %file_path.display()))]
    pub async fn scan_file(
        &self,
        file_path: &Path,
    ) -> Result<Vec<MockDetectionResult>, Box<dyn std::error::Error + Send + Sync>> {
        // Check whitelist
        let whitelist = self.whitelist.read().await;
        if whitelist.contains(file_path) {
            debug!("File {} is whitelisted, skipping scan", file_path.display());
            return Ok(Vec::new());
        }
        drop(whitelist);

        // Check cache
        if let Some(cached) = self.violation_cache.get(file_path) {
            debug!("Using cached results for {}", file_path.display());
            return Ok(cached.clone());
        }

        let content = fs::read_to_string(file_path).await?;
        let mut violations = Vec::new();

        // Scan each line for mock patterns
        for (line_number, line) in content.lines().enumerate() {
            let line_violations = self.scan_line(file_path, line_number + 1, line)?;
            violations.extend(line_violations);
        }

        // Cache results
        self.violation_cache
            .insert(file_path.to_owned(), violations.clone());

        if !violations.is_empty() {
            debug!(
                "Found {} mock violations in {}",
                violations.len(),
                file_path.display()
            );
        }

        Ok(violations)
    }

    /// Scan individual line for mock patterns
    fn scan_line(
        &self,
        file_path: &Path,
        line_number: usize,
        line_content: &str,
    ) -> Result<Vec<MockDetectionResult>, Box<dyn std::error::Error + Send + Sync>> {
        let mut violations = Vec::new();

        // Skip comments and strings (simple heuristic)
        if line_content.trim_start().starts_with("//")
            || line_content.trim_start().starts_with('#')
            || line_content.trim_start().starts_with("*")
        {
            return Ok(violations);
        }

        // Check against compiled patterns
        for (pattern, description) in &self.compiled_patterns {
            if let Some(captures) = pattern.find(line_content) {
                let severity = self.determine_mock_severity(file_path, line_content);
                let remediation = self.generate_remediation_suggestion(line_content, description);

                violations.push(MockDetectionResult {
                    file_path: file_path.to_owned(),
                    line_number,
                    column: captures.start(),
                    pattern_matched: captures.as_str().to_string(),
                    context: line_content.trim().to_string(),
                    severity,
                    description: description.clone(),
                    remediation_suggestion: remediation,
                });

                debug!(
                    "Mock detected in {}:{}: {}",
                    file_path.display(),
                    line_number,
                    captures.as_str()
                );
            }
        }

        Ok(violations)
    }

    /// Determine severity of mock violation based on context
    fn determine_mock_severity(&self, file_path: &Path, line_content: &str) -> MockSeverity {
        let file_path_str = file_path.to_string_lossy().to_lowercase();
        let line_lower = line_content.to_lowercase();

        // Critical violations
        if line_lower.contains("production") || line_lower.contains("prod") {
            return MockSeverity::Critical;
        }

        // Test files may have more lenient rules
        if file_path_str.contains("test") || file_path_str.contains("spec") {
            // Even in tests, some mocks are not acceptable
            if line_lower.contains("database")
                || line_lower.contains("payment")
                || line_lower.contains("security")
            {
                return MockSeverity::Error;
            }
            return MockSeverity::Warning;
        }

        // Main application files
        if line_lower.contains("mock")
            || line_lower.contains("fake")
            || line_lower.contains("dummy")
        {
            return MockSeverity::Error;
        }

        MockSeverity::Warning
    }

    /// Generate remediation suggestion for mock violation
    fn generate_remediation_suggestion(
        &self,
        line_content: &str,
        pattern_description: &str,
    ) -> String {
        let line_lower = line_content.to_lowercase();

        if line_lower.contains("database") || line_lower.contains("db") {
            "Replace with real database connection using test database instance".to_string()
        } else if line_lower.contains("api") || line_lower.contains("http") {
            "Replace with real API calls using test environment endpoints".to_string()
        } else if line_lower.contains("payment") {
            "Replace with payment gateway sandbox/test environment".to_string()
        } else if line_lower.contains("email") {
            "Replace with email service test environment or capture service".to_string()
        } else if line_lower.contains("file") || line_lower.contains("storage") {
            "Replace with temporary file system or in-memory storage for tests".to_string()
        } else {
            format!(
                "Replace {} with real implementation using appropriate test environment",
                pattern_description
            )
        }
    }

    /// Convert mock detections to CQGS quality violations
    pub async fn convert_to_violations(
        &self,
        detections: Vec<MockDetectionResult>,
        sentinel_id: SentinelId,
    ) -> Vec<QualityViolation> {
        let mut violations = Vec::new();

        for detection in detections {
            let severity = match detection.severity {
                MockSeverity::Info => ViolationSeverity::Info,
                MockSeverity::Warning => ViolationSeverity::Warning,
                MockSeverity::Error => ViolationSeverity::Error,
                MockSeverity::Critical => ViolationSeverity::Critical,
            };

            violations.push(QualityViolation {
                id: Uuid::new_v4(),
                sentinel_id: sentinel_id.clone(),
                severity,
                message: format!("Mock implementation detected: {}", detection.description),
                location: format!(
                    "{}:{}",
                    detection.file_path.display(),
                    detection.line_number
                ),
                timestamp: std::time::SystemTime::now(),
                remediation_suggestion: Some(detection.remediation_suggestion),
                auto_fixable: false, // Mock removal typically requires manual intervention
                hyperbolic_coordinates: None,
            });
        }

        violations
    }

    /// Add file to whitelist (exempt from mock scanning)
    pub async fn whitelist_file(&self, file_path: PathBuf) {
        let mut whitelist = self.whitelist.write().await;
        whitelist.insert(file_path.clone());
        info!("Added {} to mock scanning whitelist", file_path.display());
    }

    /// Remove file from whitelist
    pub async fn remove_from_whitelist(&self, file_path: &Path) {
        let mut whitelist = self.whitelist.write().await;
        whitelist.remove(file_path);
        info!(
            "Removed {} from mock scanning whitelist",
            file_path.display()
        );
    }

    /// Add custom mock detection pattern
    pub async fn add_custom_pattern(
        &self,
        pattern: String,
        description: String,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Validate pattern
        Regex::new(&pattern)?;

        let mut patterns = self.custom_patterns.write().await;
        patterns.push((pattern.clone(), description.clone()));

        info!("Added custom mock pattern: {} ({})", pattern, description);
        Ok(())
    }

    /// Set enforcement level
    pub async fn set_enforcement_level(&self, level: EnforcementLevel) {
        let mut current_level = self.enforcement_level.write().await;
        *current_level = level.clone();
        info!("Set mock enforcement level to: {:?}", level);
    }

    /// Get current enforcement level
    pub async fn get_enforcement_level(&self) -> EnforcementLevel {
        self.enforcement_level.read().await.clone()
    }

    /// Check if violation should block deployment based on enforcement level
    pub async fn should_block_deployment(&self, violations: &[MockDetectionResult]) -> bool {
        let level = self.get_enforcement_level().await;

        match level {
            EnforcementLevel::Permissive => false,
            EnforcementLevel::Standard => {
                violations.iter().any(|v| v.severity >= MockSeverity::Error)
            }
            EnforcementLevel::Strict => violations
                .iter()
                .any(|v| v.severity >= MockSeverity::Warning),
            EnforcementLevel::ZeroTolerance => !violations.is_empty(),
        }
    }

    /// Get validation statistics
    pub async fn get_stats(&self) -> ValidationStats {
        self.stats.lock().await.clone()
    }

    /// Clear violation cache
    pub async fn clear_cache(&self) {
        self.violation_cache.clear();
        info!("Cleared mock violation cache");
    }

    /// Export whitelist for backup/restore
    pub async fn export_whitelist(&self) -> HashSet<PathBuf> {
        self.whitelist.read().await.clone()
    }

    /// Import whitelist from backup
    pub async fn import_whitelist(&self, whitelist: HashSet<PathBuf>) {
        let mut current_whitelist = self.whitelist.write().await;
        *current_whitelist = whitelist;
        info!(
            "Imported mock scanning whitelist with {} entries",
            current_whitelist.len()
        );
    }

    /// Generate detailed report of mock violations
    pub async fn generate_report(
        &self,
        violations: &[MockDetectionResult],
    ) -> MockValidationReport {
        let mut report = MockValidationReport {
            total_violations: violations.len(),
            violations_by_severity: HashMap::new(),
            files_with_violations: HashSet::new(),
            most_common_patterns: HashMap::new(),
            remediation_summary: Vec::new(),
            enforcement_recommendation: EnforcementLevel::Standard,
        };

        // Analyze violations
        for violation in violations {
            *report
                .violations_by_severity
                .entry(violation.severity.clone())
                .or_insert(0) += 1;
            report
                .files_with_violations
                .insert(violation.file_path.clone());
            *report
                .most_common_patterns
                .entry(violation.pattern_matched.clone())
                .or_insert(0) += 1;
        }

        // Generate remediation summary
        let critical_count = report
            .violations_by_severity
            .get(&MockSeverity::Critical)
            .unwrap_or(&0);
        let error_count = report
            .violations_by_severity
            .get(&MockSeverity::Error)
            .unwrap_or(&0);

        if *critical_count > 0 {
            report.remediation_summary.push(
                "IMMEDIATE ACTION REQUIRED: Critical mock violations detected in production code"
                    .to_string(),
            );
            report.enforcement_recommendation = EnforcementLevel::ZeroTolerance;
        }

        if *error_count > 0 {
            report.remediation_summary.push(
                "Replace all mock implementations with real integrations using test environments"
                    .to_string(),
            );
        }

        report
            .remediation_summary
            .push("Review test strategy to eliminate dependency on mocked components".to_string());
        report.remediation_summary.push(
            "Implement integration test environments for all external dependencies".to_string(),
        );

        report
    }
}

/// Comprehensive mock validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockValidationReport {
    pub total_violations: usize,
    pub violations_by_severity: HashMap<MockSeverity, usize>,
    pub files_with_violations: HashSet<PathBuf>,
    pub most_common_patterns: HashMap<String, usize>,
    pub remediation_summary: Vec<String>,
    pub enforcement_recommendation: EnforcementLevel,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::TempDir;
    use tokio::fs;

    #[tokio::test]
    async fn test_mock_pattern_detection() {
        let validator = ZeroMockValidator::new();

        let test_line = "let mock_service = MockService::new();";
        let violations = validator
            .scan_line(&PathBuf::from("test.rs"), 1, test_line)
            .unwrap();

        assert!(!violations.is_empty());
        assert!(violations[0].pattern_matched.contains("Mock"));
    }

    #[tokio::test]
    async fn test_file_scanning() {
        let validator = ZeroMockValidator::new();
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.rs");

        fs::write(&test_file, "fn test() {\n    let mock = Mock::new();\n}")
            .await
            .unwrap();

        let violations = validator.scan_file(&test_file).await.unwrap();
        assert!(!violations.is_empty());
    }

    #[tokio::test]
    async fn test_directory_scanning() {
        let validator = ZeroMockValidator::new();
        let temp_dir = TempDir::new().unwrap();

        // Create test file with mock
        let test_file = temp_dir.path().join("src").join("lib.rs");
        fs::create_dir_all(test_file.parent().unwrap())
            .await
            .unwrap();
        fs::write(&test_file, "use mockall::predicate::*;\n")
            .await
            .unwrap();

        let violations = validator.scan_directory(temp_dir.path()).await.unwrap();
        assert!(!violations.is_empty());
    }

    #[tokio::test]
    async fn test_whitelist_functionality() {
        let validator = ZeroMockValidator::new();
        let test_path = PathBuf::from("whitelisted.rs");

        validator.whitelist_file(test_path.clone()).await;

        let whitelist = validator.export_whitelist().await;
        assert!(whitelist.contains(&test_path));
    }

    #[tokio::test]
    async fn test_enforcement_levels() {
        let validator = ZeroMockValidator::new();

        validator
            .set_enforcement_level(EnforcementLevel::ZeroTolerance)
            .await;
        let level = validator.get_enforcement_level().await;

        assert!(matches!(level, EnforcementLevel::ZeroTolerance));
    }
}
