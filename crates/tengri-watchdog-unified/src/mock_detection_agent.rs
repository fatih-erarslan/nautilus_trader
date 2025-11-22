//! Advanced Mock Detection Agent
//! 
//! Sophisticated detection system for all major mock frameworks, synthetic data, and test doubles
//! Integrates with TENGRI quantum fingerprinting for comprehensive mock elimination

use crate::{TENGRIError, TradingOperation, TENGRIOversightResult, ViolationType};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use regex::{Regex, RegexSet};
use std::time::Instant;
use chrono::{DateTime, Utc};
use uuid::Uuid;
use serde::{Deserialize, Serialize};

/// Mock Framework Detection Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockDetectionConfig {
    pub enabled_detectors: HashSet<String>,
    pub strict_mode: bool,
    pub emergency_on_detection: bool,
    pub scan_timeout_ms: u64,
    pub confidence_threshold: f64,
    pub whitelist_patterns: HashSet<String>,
    pub custom_patterns: HashMap<String, String>,
}

impl Default for MockDetectionConfig {
    fn default() -> Self {
        let mut enabled_detectors = HashSet::new();
        enabled_detectors.insert("mockito".to_string());
        enabled_detectors.insert("wiremock".to_string());
        enabled_detectors.insert("sinon".to_string());
        enabled_detectors.insert("jest".to_string());
        enabled_detectors.insert("moq".to_string());
        enabled_detectors.insert("nsubstitute".to_string());
        enabled_detectors.insert("easymock".to_string());
        enabled_detectors.insert("powermock".to_string());
        enabled_detectors.insert("unittest_mock".to_string());
        enabled_detectors.insert("pytest_mock".to_string());
        enabled_detectors.insert("go_mock".to_string());
        enabled_detectors.insert("testify_mock".to_string());
        enabled_detectors.insert("rspec_mock".to_string());
        enabled_detectors.insert("phpunit_mock".to_string());
        enabled_detectors.insert("rust_mock".to_string());

        Self {
            enabled_detectors,
            strict_mode: true,
            emergency_on_detection: true,
            scan_timeout_ms: 100,
            confidence_threshold: 0.95,
            whitelist_patterns: HashSet::new(),
            custom_patterns: HashMap::new(),
        }
    }
}

/// Mock Detection Result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockDetectionResult {
    pub detected: bool,
    pub framework: Option<String>,
    pub confidence: f64,
    pub location: String,
    pub pattern_matched: String,
    pub violation_type: MockViolationType,
    pub scan_duration_ns: u64,
    pub evidence: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MockViolationType {
    MockFrameworkImport,
    MockObjectCreation,
    MockMethodCall,
    FakeDataGeneration,
    TestDoubleUsage,
    SyntheticAPIResponse,
    MockAnnotation,
    StubImplementation,
    DummyObjectUsage,
    MockConfiguration,
}

/// Framework-Specific Detectors
pub struct MockFrameworkDetector {
    pub name: String,
    pub import_patterns: RegexSet,
    pub usage_patterns: RegexSet,
    pub annotation_patterns: RegexSet,
    pub configuration_patterns: RegexSet,
    pub danger_level: DangerLevel,
}

#[derive(Debug, Clone)]
pub enum DangerLevel {
    Critical,    // Immediate shutdown required
    High,        // Block operation
    Medium,      // Warning with corrective action
    Low,         // Monitor only
}

impl MockFrameworkDetector {
    /// Create Mockito detector (Java/Android)
    pub fn mockito() -> Result<Self, TENGRIError> {
        let import_patterns = RegexSet::new(&[
            r"import\s+org\.mockito\.",
            r"import\s+static\s+org\.mockito\.",
            r"@Mock\s+",
            r"@Spy\s+",
            r"@InjectMocks\s+",
            r"@MockBean\s+",
            r"@SpyBean\s+",
        ]).map_err(|e| TENGRIError::DataIntegrityViolation { 
            reason: format!("Failed to compile Mockito patterns: {}", e) 
        })?;

        let usage_patterns = RegexSet::new(&[
            r"Mockito\.(mock|spy|when|verify|times|never|atLeast|atMost)",
            r"MockitoAnnotations\.initMocks",
            r"\.thenReturn\(",
            r"\.thenThrow\(",
            r"\.doReturn\(",
            r"\.doThrow\(",
            r"\.doAnswer\(",
            r"\.doNothing\(",
            r"verify\(.+\)\.",
            r"verifyNoMoreInteractions\(",
            r"verifyZeroInteractions\(",
            r"reset\(.+\)",
        ]).map_err(|e| TENGRIError::DataIntegrityViolation { 
            reason: format!("Failed to compile Mockito usage patterns: {}", e) 
        })?;

        let annotation_patterns = RegexSet::new(&[
            r"@Mock\b",
            r"@Spy\b",
            r"@InjectMocks\b",
            r"@MockBean\b",
            r"@SpyBean\b",
            r"@MockitoSettings\b",
        ]).map_err(|e| TENGRIError::DataIntegrityViolation { 
            reason: format!("Failed to compile Mockito annotation patterns: {}", e) 
        })?;

        let configuration_patterns = RegexSet::new(&[
            r"MockitoJUnitRunner",
            r"MockitoExtension",
            r"MockitoSession",
            r"MockSettings",
        ]).map_err(|e| TENGRIError::DataIntegrityViolation { 
            reason: format!("Failed to compile Mockito config patterns: {}", e) 
        })?;

        Ok(Self {
            name: "Mockito".to_string(),
            import_patterns,
            usage_patterns,
            annotation_patterns,
            configuration_patterns,
            danger_level: DangerLevel::Critical,
        })
    }

    /// Create WireMock detector (HTTP service mocking)
    pub fn wiremock() -> Result<Self, TENGRIError> {
        let import_patterns = RegexSet::new(&[
            r"import\s+com\.github\.tomakehurst\.wiremock\.",
            r"import\s+static\s+com\.github\.tomakehurst\.wiremock\.",
            r"from\s+wiremock\s+import",
            r"const\s+wiremock\s+=\s+require\(",
        ]).map_err(|e| TENGRIError::DataIntegrityViolation { 
            reason: format!("Failed to compile WireMock import patterns: {}", e) 
        })?;

        let usage_patterns = RegexSet::new(&[
            r"WireMockServer",
            r"WireMockRule",
            r"WireMock\.(stubFor|get|post|put|delete|patch)",
            r"\.willReturn\(",
            r"\.aResponse\(",
            r"\.withStatus\(",
            r"\.withBody\(",
            r"\.withHeader\(",
            r"\.urlMatching\(",
            r"\.urlEqualTo\(",
            r"stubFor\(",
            r"\.scenarios\(",
            r"\.mappings\(",
        ]).map_err(|e| TENGRIError::DataIntegrityViolation { 
            reason: format!("Failed to compile WireMock usage patterns: {}", e) 
        })?;

        let annotation_patterns = RegexSet::new(&[
            r"@RegisterExtension\s+.*WireMock",
            r"@WireMockTest",
        ]).map_err(|e| TENGRIError::DataIntegrityViolation { 
            reason: format!("Failed to compile WireMock annotation patterns: {}", e) 
        })?;

        let configuration_patterns = RegexSet::new(&[
            r"WireMockConfiguration",
            r"wireMockConfig\(",
            r"\.port\(\d+\)",
            r"\.httpsPort\(\d+\)",
        ]).map_err(|e| TENGRIError::DataIntegrityViolation { 
            reason: format!("Failed to compile WireMock config patterns: {}", e) 
        })?;

        Ok(Self {
            name: "WireMock".to_string(),
            import_patterns,
            usage_patterns,
            annotation_patterns,
            configuration_patterns,
            danger_level: DangerLevel::Critical,
        })
    }

    /// Create Jest/Sinon detector (JavaScript/TypeScript)
    pub fn jest_sinon() -> Result<Self, TENGRIError> {
        let import_patterns = RegexSet::new(&[
            r"import\s+.*\bsinon\b",
            r"require\(['\"]sinon['\"]\)",
            r"import\s+.*\{.*jest.*\}",
            r"require\(['\"]jest['\"]\)",
            r"from\s+['\"]sinon['\"]",
            r"@jest/globals",
        ]).map_err(|e| TENGRIError::DataIntegrityViolation { 
            reason: format!("Failed to compile Jest/Sinon import patterns: {}", e) 
        })?;

        let usage_patterns = RegexSet::new(&[
            r"jest\.mock\(",
            r"jest\.spyOn\(",
            r"jest\.fn\(",
            r"mockImplementation\(",
            r"mockReturnValue\(",
            r"mockResolvedValue\(",
            r"mockRejectedValue\(",
            r"sinon\.stub\(",
            r"sinon\.spy\(",
            r"sinon\.mock\(",
            r"sinon\.fake\(",
            r"sinon\.createSandbox\(",
            r"\.restore\(\)",
            r"\.reset\(\)",
            r"\.resetHistory\(\)",
            r"\.withArgs\(",
            r"\.returns\(",
            r"\.resolves\(",
            r"\.rejects\(",
            r"\.throws\(",
            r"\.callsArg\(",
            r"\.yields\(",
        ]).map_err(|e| TENGRIError::DataIntegrityViolation { 
            reason: format!("Failed to compile Jest/Sinon usage patterns: {}", e) 
        })?;

        let annotation_patterns = RegexSet::new(&[
            r"describe\.skip\(",
            r"it\.skip\(",
            r"test\.skip\(",
            r"beforeEach\(",
            r"afterEach\(",
        ]).map_err(|e| TENGRIError::DataIntegrityViolation { 
            reason: format!("Failed to compile Jest/Sinon annotation patterns: {}", e) 
        })?;

        let configuration_patterns = RegexSet::new(&[
            r"jest\.config\.",
            r"setupFilesAfterEnv",
            r"testEnvironment",
            r"clearMocks",
            r"resetMocks",
            r"restoreMocks",
        ]).map_err(|e| TENGRIError::DataIntegrityViolation { 
            reason: format!("Failed to compile Jest/Sinon config patterns: {}", e) 
        })?;

        Ok(Self {
            name: "Jest/Sinon".to_string(),
            import_patterns,
            usage_patterns,
            annotation_patterns,
            configuration_patterns,
            danger_level: DangerLevel::Critical,
        })
    }

    /// Create Python unittest.mock detector
    pub fn python_mock() -> Result<Self, TENGRIError> {
        let import_patterns = RegexSet::new(&[
            r"from\s+unittest\.mock\s+import",
            r"import\s+unittest\.mock",
            r"from\s+unittest\s+import\s+mock",
            r"import\s+mock",
            r"from\s+pytest_mock\s+import",
            r"import\s+pytest_mock",
        ]).map_err(|e| TENGRIError::DataIntegrityViolation { 
            reason: format!("Failed to compile Python mock import patterns: {}", e) 
        })?;

        let usage_patterns = RegexSet::new(&[
            r"Mock\(\)",
            r"MagicMock\(\)",
            r"AsyncMock\(\)",
            r"patch\(",
            r"patch\.object\(",
            r"patch\.multiple\(",
            r"patch\.dict\(",
            r"create_autospec\(",
            r"spec_set\s*=",
            r"side_effect\s*=",
            r"return_value\s*=",
            r"\.assert_called\(",
            r"\.assert_called_once\(",
            r"\.assert_called_with\(",
            r"\.assert_called_once_with\(",
            r"\.assert_not_called\(",
            r"\.assert_has_calls\(",
            r"\.configure_mock\(",
            r"\.reset_mock\(",
            r"mocker\.",
        ]).map_err(|e| TENGRIError::DataIntegrityViolation { 
            reason: format!("Failed to compile Python mock usage patterns: {}", e) 
        })?;

        let annotation_patterns = RegexSet::new(&[
            r"@patch\(",
            r"@patch\.object\(",
            r"@mock\.patch\(",
            r"@pytest\.fixture",
        ]).map_err(|e| TENGRIError::DataIntegrityViolation { 
            reason: format!("Failed to compile Python mock annotation patterns: {}", e) 
        })?;

        let configuration_patterns = RegexSet::new(&[
            r"mock_configure\(",
            r"mock\.DEFAULT",
            r"mock\.FILTER_DIR",
            r"mock\.sentinel",
        ]).map_err(|e| TENGRIError::DataIntegrityViolation { 
            reason: format!("Failed to compile Python mock config patterns: {}", e) 
        })?;

        Ok(Self {
            name: "Python Mock".to_string(),
            import_patterns,
            usage_patterns,
            annotation_patterns,
            configuration_patterns,
            danger_level: DangerLevel::Critical,
        })
    }

    /// Scan code for mock framework usage
    pub fn scan_code(&self, code: &str) -> MockDetectionResult {
        let scan_start = Instant::now();
        let mut evidence = Vec::new();
        let mut confidence = 0.0;
        let mut detected = false;
        let mut violation_type = MockViolationType::MockFrameworkImport;
        let mut pattern_matched = String::new();

        // Check import patterns
        if self.import_patterns.is_match(code) {
            detected = true;
            confidence += 0.4;
            violation_type = MockViolationType::MockFrameworkImport;
            evidence.push("Mock framework import detected".to_string());
            pattern_matched = "import_pattern".to_string();
        }

        // Check usage patterns
        if self.usage_patterns.is_match(code) {
            detected = true;
            confidence += 0.3;
            violation_type = MockViolationType::MockMethodCall;
            evidence.push("Mock framework method call detected".to_string());
            pattern_matched = "usage_pattern".to_string();
        }

        // Check annotation patterns
        if self.annotation_patterns.is_match(code) {
            detected = true;
            confidence += 0.2;
            violation_type = MockViolationType::MockAnnotation;
            evidence.push("Mock framework annotation detected".to_string());
            pattern_matched = "annotation_pattern".to_string();
        }

        // Check configuration patterns
        if self.configuration_patterns.is_match(code) {
            detected = true;
            confidence += 0.1;
            violation_type = MockViolationType::MockConfiguration;
            evidence.push("Mock framework configuration detected".to_string());
            pattern_matched = "config_pattern".to_string();
        }

        let scan_duration = scan_start.elapsed();

        MockDetectionResult {
            detected,
            framework: if detected { Some(self.name.clone()) } else { None },
            confidence,
            location: "code_analysis".to_string(),
            pattern_matched,
            violation_type,
            scan_duration_ns: scan_duration.as_nanos() as u64,
            evidence,
        }
    }
}

/// Advanced Mock Detection Agent
pub struct MockDetectionAgent {
    config: MockDetectionConfig,
    detectors: Vec<MockFrameworkDetector>,
    detection_history: Arc<RwLock<Vec<(DateTime<Utc>, MockDetectionResult)>>>,
    violation_counts: Arc<RwLock<HashMap<String, u64>>>,
    emergency_triggers: Arc<RwLock<u64>>,
}

impl MockDetectionAgent {
    /// Initialize mock detection agent
    pub async fn new(config: MockDetectionConfig) -> Result<Self, TENGRIError> {
        let mut detectors = Vec::new();
        
        // Initialize framework detectors
        if config.enabled_detectors.contains("mockito") {
            detectors.push(MockFrameworkDetector::mockito()?);
        }
        if config.enabled_detectors.contains("wiremock") {
            detectors.push(MockFrameworkDetector::wiremock()?);
        }
        if config.enabled_detectors.contains("jest") || config.enabled_detectors.contains("sinon") {
            detectors.push(MockFrameworkDetector::jest_sinon()?);
        }
        if config.enabled_detectors.contains("unittest_mock") || config.enabled_detectors.contains("pytest_mock") {
            detectors.push(MockFrameworkDetector::python_mock()?);
        }

        let detection_history = Arc::new(RwLock::new(Vec::new()));
        let violation_counts = Arc::new(RwLock::new(HashMap::new()));
        let emergency_triggers = Arc::new(RwLock::new(0));

        Ok(Self {
            config,
            detectors,
            detection_history,
            violation_counts,
            emergency_triggers,
        })
    }

    /// Comprehensive mock detection scan
    pub async fn scan_for_mocks(&self, operation: &TradingOperation) -> Result<TENGRIOversightResult, TENGRIError> {
        let scan_start = Instant::now();
        let mut detection_results = Vec::new();

        // Scan with each detector
        for detector in &self.detectors {
            let result = detector.scan_code(&operation.data_source);
            detection_results.push(result);
        }

        // Analyze results
        let final_result = self.analyze_detection_results(detection_results, operation).await?;

        // Record detection history
        self.record_detection_history(&final_result).await;

        // Check for emergency response
        if final_result.detected && self.config.emergency_on_detection {
            self.trigger_emergency_response(&final_result, operation).await?;
        }

        let scan_duration = scan_start.elapsed();
        if scan_duration.as_millis() > self.config.scan_timeout_ms {
            tracing::warn!("Mock detection scan exceeded timeout: {:?}", scan_duration);
        }

        // Convert to oversight result
        self.convert_to_oversight_result(&final_result).await
    }

    /// Scan file system for mock frameworks
    pub async fn scan_filesystem(&self, path: &str) -> Result<Vec<MockDetectionResult>, TENGRIError> {
        let mut results = Vec::new();
        
        // In a real implementation, this would scan the filesystem
        // For now, we return empty results
        
        Ok(results)
    }

    /// Scan dependencies for mock frameworks
    pub async fn scan_dependencies(&self, manifest_path: &str) -> Result<Vec<MockDetectionResult>, TENGRIError> {
        let mut results = Vec::new();
        
        // In a real implementation, this would scan package.json, pom.xml, requirements.txt, etc.
        // For now, we return empty results
        
        Ok(results)
    }

    /// Get detection statistics
    pub async fn get_detection_stats(&self) -> Result<MockDetectionStats, TENGRIError> {
        let history = self.detection_history.read().await;
        let counts = self.violation_counts.read().await;
        let emergencies = self.emergency_triggers.read().await;

        let total_scans = history.len();
        let total_violations = history.iter().filter(|(_, result)| result.detected).count();
        let recent_violations = history.iter().rev().take(100).cloned().collect();

        Ok(MockDetectionStats {
            total_scans,
            total_violations,
            emergency_triggers: *emergencies,
            violation_counts: counts.clone(),
            recent_violations,
            detection_rate: if total_scans > 0 { total_violations as f64 / total_scans as f64 } else { 0.0 },
        })
    }

    async fn analyze_detection_results(
        &self,
        results: Vec<MockDetectionResult>,
        operation: &TradingOperation,
    ) -> Result<MockDetectionResult, TENGRIError> {
        let mut max_confidence = 0.0;
        let mut detected = false;
        let mut framework = None;
        let mut evidence = Vec::new();
        let mut violation_type = MockViolationType::MockFrameworkImport;
        let mut pattern_matched = String::new();

        for result in results {
            if result.detected {
                detected = true;
                if result.confidence > max_confidence {
                    max_confidence = result.confidence;
                    framework = result.framework;
                    violation_type = result.violation_type;
                    pattern_matched = result.pattern_matched;
                }
                evidence.extend(result.evidence);
            }
        }

        Ok(MockDetectionResult {
            detected,
            framework,
            confidence: max_confidence,
            location: format!("operation:{}", operation.id),
            pattern_matched,
            violation_type,
            scan_duration_ns: 0, // Aggregate duration would be calculated here
            evidence,
        })
    }

    async fn record_detection_history(&self, result: &MockDetectionResult) {
        let mut history = self.detection_history.write().await;
        history.push((Utc::now(), result.clone()));

        // Keep only last 10,000 entries
        if history.len() > 10000 {
            history.drain(0..1000);
        }

        if result.detected {
            let mut counts = self.violation_counts.write().await;
            if let Some(framework) = &result.framework {
                *counts.entry(framework.clone()).or_insert(0) += 1;
            }
        }
    }

    async fn trigger_emergency_response(
        &self,
        result: &MockDetectionResult,
        operation: &TradingOperation,
    ) -> Result<(), TENGRIError> {
        let mut emergency_count = self.emergency_triggers.write().await;
        *emergency_count += 1;

        tracing::error!(
            "EMERGENCY: Mock framework detected - Framework: {:?} - Operation: {} - Confidence: {:.2}",
            result.framework,
            operation.id,
            result.confidence
        );

        Ok(())
    }

    async fn convert_to_oversight_result(&self, result: &MockDetectionResult) -> Result<TENGRIOversightResult, TENGRIError> {
        if result.detected && result.confidence >= self.config.confidence_threshold {
            Ok(TENGRIOversightResult::CriticalViolation {
                violation_type: ViolationType::SyntheticData,
                immediate_shutdown: self.config.emergency_on_detection,
                forensic_data: serde_json::to_vec(result).unwrap_or_default(),
            })
        } else if result.detected {
            Ok(TENGRIOversightResult::Rejected {
                reason: format!("Mock framework detected: {:?}", result.framework),
                emergency_action: crate::EmergencyAction::QuarantineAgent {
                    agent_id: "mock_detected".to_string(),
                },
            })
        } else {
            Ok(TENGRIOversightResult::Approved)
        }
    }
}

/// Detection Statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockDetectionStats {
    pub total_scans: usize,
    pub total_violations: usize,
    pub emergency_triggers: u64,
    pub violation_counts: HashMap<String, u64>,
    pub recent_violations: Vec<(DateTime<Utc>, MockDetectionResult)>,
    pub detection_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OperationType, RiskParameters};

    #[tokio::test]
    async fn test_mockito_detection() {
        let detector = MockFrameworkDetector::mockito().unwrap();
        let code = "import org.mockito.Mock; @Mock private Service service;";
        let result = detector.scan_code(code);
        
        assert!(result.detected);
        assert_eq!(result.framework, Some("Mockito".to_string()));
        assert!(result.confidence > 0.0);
    }

    #[tokio::test]
    async fn test_wiremock_detection() {
        let detector = MockFrameworkDetector::wiremock().unwrap();
        let code = "WireMockServer server = new WireMockServer(); stubFor(get(urlEqualTo(\"/test\")))";
        let result = detector.scan_code(code);
        
        assert!(result.detected);
        assert_eq!(result.framework, Some("WireMock".to_string()));
    }

    #[tokio::test]
    async fn test_jest_sinon_detection() {
        let detector = MockFrameworkDetector::jest_sinon().unwrap();
        let code = "jest.mock('./service'); const mockFn = jest.fn();";
        let result = detector.scan_code(code);
        
        assert!(result.detected);
        assert_eq!(result.framework, Some("Jest/Sinon".to_string()));
    }

    #[tokio::test]
    async fn test_python_mock_detection() {
        let detector = MockFrameworkDetector::python_mock().unwrap();
        let code = "from unittest.mock import Mock, patch; @patch('service.method')";
        let result = detector.scan_code(code);
        
        assert!(result.detected);
        assert_eq!(result.framework, Some("Python Mock".to_string()));
    }

    #[tokio::test]
    async fn test_mock_detection_agent() {
        let config = MockDetectionConfig::default();
        let agent = MockDetectionAgent::new(config).await.unwrap();
        
        let operation = TradingOperation {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            operation_type: OperationType::PlaceOrder,
            data_source: "import org.mockito.Mock; @Mock private Service service;".to_string(),
            mathematical_model: "real_model".to_string(),
            risk_parameters: RiskParameters {
                max_position_size: 1000.0,
                stop_loss: Some(0.02),
                take_profit: Some(0.05),
                confidence_threshold: 0.95,
            },
            agent_id: "test_agent".to_string(),
        };

        let result = agent.scan_for_mocks(&operation).await.unwrap();
        assert!(matches!(result, TENGRIOversightResult::CriticalViolation { .. }));
    }

    #[tokio::test]
    async fn test_clean_code_detection() {
        let config = MockDetectionConfig::default();
        let agent = MockDetectionAgent::new(config).await.unwrap();
        
        let operation = TradingOperation {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            operation_type: OperationType::PlaceOrder,
            data_source: "authentic trading algorithm with real data".to_string(),
            mathematical_model: "real_model".to_string(),
            risk_parameters: RiskParameters {
                max_position_size: 1000.0,
                stop_loss: Some(0.02),
                take_profit: Some(0.05),
                confidence_threshold: 0.95,
            },
            agent_id: "test_agent".to_string(),
        };

        let result = agent.scan_for_mocks(&operation).await.unwrap();
        assert!(matches!(result, TENGRIOversightResult::Approved));
    }
}