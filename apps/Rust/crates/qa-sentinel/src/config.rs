//! Configuration for QA Sentinel

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for the QA Sentinel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaSentinelConfig {
    /// Coverage enforcement settings
    pub coverage: CoverageConfig,
    
    /// Zero-mock testing configuration
    pub zero_mock: ZeroMockConfig,
    
    /// Property-based testing configuration
    pub property_testing: PropertyTestingConfig,
    
    /// Performance testing configuration
    pub performance: PerformanceConfig,
    
    /// Chaos engineering configuration
    pub chaos: ChaosConfig,
    
    /// Quality gates configuration
    pub quality_gates: QualityGatesConfig,
    
    /// Monitoring configuration
    pub monitoring: MonitoringConfig,
    
    /// Security testing configuration
    pub security: SecurityConfig,
    
    /// Reporting configuration
    pub reporting: ReportingConfig,
}

/// Coverage enforcement configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageConfig {
    /// Minimum line coverage percentage (default: 100.0)
    pub min_line_coverage: f64,
    
    /// Minimum branch coverage percentage (default: 100.0)
    pub min_branch_coverage: f64,
    
    /// Minimum function coverage percentage (default: 100.0)
    pub min_function_coverage: f64,
    
    /// Coverage output format
    pub output_format: CoverageOutputFormat,
    
    /// Paths to exclude from coverage
    pub exclude_paths: Vec<String>,
    
    /// Generate HTML reports
    pub generate_html: bool,
    
    /// Coverage engine to use
    pub engine: CoverageEngine,
}

/// Zero-mock testing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroMockConfig {
    /// Enable zero-mock testing
    pub enabled: bool,
    
    /// Real integration endpoints
    pub integration_endpoints: IntegrationEndpoints,
    
    /// Test environment configuration
    pub test_environment: TestEnvironment,
    
    /// Timeout for integration tests
    pub integration_timeout: Duration,
    
    /// Parallel test execution
    pub parallel_execution: bool,
    
    /// Maximum concurrent tests
    pub max_concurrent_tests: usize,
}

/// Property-based testing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyTestingConfig {
    /// Enable property-based testing
    pub enabled: bool,
    
    /// Number of test cases to generate
    pub test_cases: usize,
    
    /// Maximum test case complexity
    pub max_complexity: usize,
    
    /// Shrinking strategy
    pub shrinking: ShrinkingStrategy,
    
    /// Random seed for reproducible tests
    pub seed: Option<u64>,
    
    /// Test timeout
    pub timeout: Duration,
}

/// Performance testing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable performance testing
    pub enabled: bool,
    
    /// Performance benchmarks
    pub benchmarks: Vec<BenchmarkConfig>,
    
    /// Performance thresholds
    pub thresholds: PerformanceThresholds,
    
    /// Regression detection
    pub regression_detection: RegressionDetectionConfig,
    
    /// Profiling configuration
    pub profiling: ProfilingConfig,
}

/// Chaos engineering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChaosConfig {
    /// Enable chaos engineering
    pub enabled: bool,
    
    /// Chaos experiments
    pub experiments: Vec<ChaosExperiment>,
    
    /// Failure injection configuration
    pub failure_injection: FailureInjectionConfig,
    
    /// Network chaos configuration
    pub network_chaos: NetworkChaosConfig,
    
    /// Resource chaos configuration
    pub resource_chaos: ResourceChaosConfig,
}

/// Quality gates configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityGatesConfig {
    /// Enable quality gates
    pub enabled: bool,
    
    /// Quality thresholds
    pub thresholds: QualityThresholds,
    
    /// Blocking failures
    pub blocking_failures: Vec<QualityGateType>,
    
    /// Reporting configuration
    pub reporting: bool,
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable monitoring
    pub enabled: bool,
    
    /// Monitoring interval
    pub interval: Duration,
    
    /// Metrics collection
    pub metrics: MetricsConfig,
    
    /// Alerting configuration
    pub alerting: AlertingConfig,
}

/// Security testing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable security testing
    pub enabled: bool,
    
    /// Security audit configuration
    pub audit: AuditConfig,
    
    /// Vulnerability scanning
    pub vulnerability_scanning: VulnerabilityConfig,
    
    /// Penetration testing
    pub penetration_testing: PenetrationTestingConfig,
}

/// Reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingConfig {
    /// Enable reporting
    pub enabled: bool,
    
    /// Output directory
    pub output_directory: String,
    
    /// Report formats
    pub formats: Vec<ReportFormat>,
    
    /// Include detailed metrics
    pub detailed_metrics: bool,
    
    /// Generate trend analysis
    pub trend_analysis: bool,
}

// Supporting types

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoverageOutputFormat {
    Html,
    Xml,
    Json,
    Lcov,
    Text,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoverageEngine {
    Tarpaulin,
    Grcov,
    Kcov,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationEndpoints {
    pub binance_testnet: String,
    pub coinbase_sandbox: String,
    pub kraken_demo: String,
    pub database_test: String,
    pub redis_test: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestEnvironment {
    pub containers: Vec<TestContainer>,
    pub network_isolation: bool,
    pub resource_limits: ResourceLimits,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestContainer {
    pub name: String,
    pub image: String,
    pub ports: Vec<u16>,
    pub environment: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub memory_mb: u64,
    pub cpu_cores: f64,
    pub disk_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShrinkingStrategy {
    Aggressive,
    Conservative,
    Balanced,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub name: String,
    pub function: String,
    pub samples: usize,
    pub iterations: usize,
    pub warm_up_time: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    pub max_latency_ms: u64,
    pub min_throughput_ops: u64,
    pub max_memory_mb: u64,
    pub max_cpu_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionDetectionConfig {
    pub enabled: bool,
    pub threshold_percent: f64,
    pub baseline_samples: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingConfig {
    pub enabled: bool,
    pub flamegraph: bool,
    pub cpu_profiling: bool,
    pub memory_profiling: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChaosExperiment {
    pub name: String,
    pub experiment_type: ChaosExperimentType,
    pub duration: Duration,
    pub intensity: f64,
    pub target: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChaosExperimentType {
    NetworkLatency,
    NetworkPartition,
    ResourceExhaustion,
    ServiceCrash,
    DiskFull,
    MemoryPressure,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureInjectionConfig {
    pub enabled: bool,
    pub failure_rate: f64,
    pub failure_types: Vec<FailureType>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FailureType {
    NetworkTimeout,
    DatabaseConnection,
    ApiFailure,
    MemoryAllocation,
    DiskIo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkChaosConfig {
    pub enabled: bool,
    pub latency_ms: u64,
    pub packet_loss_percent: f64,
    pub bandwidth_limit_kbps: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceChaosConfig {
    pub enabled: bool,
    pub cpu_pressure_percent: f64,
    pub memory_pressure_percent: f64,
    pub disk_pressure_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    pub min_code_coverage: f64,
    pub max_cyclomatic_complexity: u32,
    pub max_code_duplication: f64,
    pub min_test_pass_rate: f64,
    pub max_security_vulnerabilities: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityGateType {
    Coverage,
    Complexity,
    Duplication,
    Security,
    Performance,
    Tests,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    pub prometheus_endpoint: String,
    pub custom_metrics: Vec<CustomMetric>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomMetric {
    pub name: String,
    pub metric_type: MetricType,
    pub labels: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingConfig {
    pub enabled: bool,
    pub webhook_url: Option<String>,
    pub email_recipients: Vec<String>,
    pub slack_webhook: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    pub enabled: bool,
    pub audit_dependencies: bool,
    pub audit_code: bool,
    pub audit_config: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulnerabilityConfig {
    pub enabled: bool,
    pub scan_dependencies: bool,
    pub scan_containers: bool,
    pub scan_infrastructure: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PenetrationTestingConfig {
    pub enabled: bool,
    pub target_endpoints: Vec<String>,
    pub test_suites: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    Html,
    Json,
    Xml,
    Pdf,
    Markdown,
}

impl Default for QaSentinelConfig {
    fn default() -> Self {
        Self {
            coverage: CoverageConfig::default(),
            zero_mock: ZeroMockConfig::default(),
            property_testing: PropertyTestingConfig::default(),
            performance: PerformanceConfig::default(),
            chaos: ChaosConfig::default(),
            quality_gates: QualityGatesConfig::default(),
            monitoring: MonitoringConfig::default(),
            security: SecurityConfig::default(),
            reporting: ReportingConfig::default(),
        }
    }
}

impl Default for CoverageConfig {
    fn default() -> Self {
        Self {
            min_line_coverage: 100.0,
            min_branch_coverage: 100.0,
            min_function_coverage: 100.0,
            output_format: CoverageOutputFormat::Html,
            exclude_paths: vec![
                "tests/".to_string(),
                "benches/".to_string(),
                "target/".to_string(),
            ],
            generate_html: true,
            engine: CoverageEngine::Tarpaulin,
        }
    }
}

impl Default for ZeroMockConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            integration_endpoints: IntegrationEndpoints::default(),
            test_environment: TestEnvironment::default(),
            integration_timeout: Duration::from_secs(30),
            parallel_execution: true,
            max_concurrent_tests: 10,
        }
    }
}

impl Default for PropertyTestingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            test_cases: 1000,
            max_complexity: 1000,
            shrinking: ShrinkingStrategy::Balanced,
            seed: None,
            timeout: Duration::from_secs(60),
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            benchmarks: vec![],
            thresholds: PerformanceThresholds::default(),
            regression_detection: RegressionDetectionConfig::default(),
            profiling: ProfilingConfig::default(),
        }
    }
}

impl Default for ChaosConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            experiments: vec![],
            failure_injection: FailureInjectionConfig::default(),
            network_chaos: NetworkChaosConfig::default(),
            resource_chaos: ResourceChaosConfig::default(),
        }
    }
}

impl Default for QualityGatesConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            thresholds: QualityThresholds::default(),
            blocking_failures: vec![
                QualityGateType::Coverage,
                QualityGateType::Security,
                QualityGateType::Tests,
            ],
            reporting: true,
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(60),
            metrics: MetricsConfig::default(),
            alerting: AlertingConfig::default(),
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            audit: AuditConfig::default(),
            vulnerability_scanning: VulnerabilityConfig::default(),
            penetration_testing: PenetrationTestingConfig::default(),
        }
    }
}

impl Default for ReportingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            output_directory: "./qa-reports".to_string(),
            formats: vec![ReportFormat::Html, ReportFormat::Json],
            detailed_metrics: true,
            trend_analysis: true,
        }
    }
}

impl Default for IntegrationEndpoints {
    fn default() -> Self {
        Self {
            binance_testnet: "https://testnet.binance.vision".to_string(),
            coinbase_sandbox: "https://api-public.sandbox.pro.coinbase.com".to_string(),
            kraken_demo: "https://api.kraken.com".to_string(),
            database_test: "sqlite://test.db".to_string(),
            redis_test: "redis://localhost:6379".to_string(),
        }
    }
}

impl Default for TestEnvironment {
    fn default() -> Self {
        Self {
            containers: vec![],
            network_isolation: true,
            resource_limits: ResourceLimits::default(),
        }
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            memory_mb: 1024,
            cpu_cores: 2.0,
            disk_mb: 10240,
        }
    }
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            max_latency_ms: 100,
            min_throughput_ops: 1000,
            max_memory_mb: 512,
            max_cpu_percent: 80.0,
        }
    }
}

impl Default for RegressionDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold_percent: 10.0,
            baseline_samples: 100,
        }
    }
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            flamegraph: true,
            cpu_profiling: true,
            memory_profiling: true,
        }
    }
}

impl Default for FailureInjectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            failure_rate: 0.1,
            failure_types: vec![
                FailureType::NetworkTimeout,
                FailureType::DatabaseConnection,
                FailureType::ApiFailure,
            ],
        }
    }
}

impl Default for NetworkChaosConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            latency_ms: 100,
            packet_loss_percent: 1.0,
            bandwidth_limit_kbps: 1000,
        }
    }
}

impl Default for ResourceChaosConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cpu_pressure_percent: 80.0,
            memory_pressure_percent: 80.0,
            disk_pressure_percent: 80.0,
        }
    }
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_code_coverage: 100.0,
            max_cyclomatic_complexity: 10,
            max_code_duplication: 3.0,
            min_test_pass_rate: 100.0,
            max_security_vulnerabilities: 0,
        }
    }
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            prometheus_endpoint: "http://localhost:9090".to_string(),
            custom_metrics: vec![],
        }
    }
}

impl Default for AlertingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            webhook_url: None,
            email_recipients: vec![],
            slack_webhook: None,
        }
    }
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            audit_dependencies: true,
            audit_code: true,
            audit_config: true,
        }
    }
}

impl Default for VulnerabilityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            scan_dependencies: true,
            scan_containers: true,
            scan_infrastructure: true,
        }
    }
}

impl Default for PenetrationTestingConfig {
    fn default() -> Self {
        Self {
            enabled: false, // Disabled by default for safety
            target_endpoints: vec![],
            test_suites: vec![],
        }
    }
}

impl QaSentinelConfig {
    /// Create a high-performance configuration for production testing
    pub fn high_performance() -> Self {
        let mut config = Self::default();
        
        // Increase property testing iterations
        config.property_testing.test_cases = 10000;
        
        // Enable all chaos experiments
        config.chaos.experiments = vec![
            ChaosExperiment {
                name: "network_latency".to_string(),
                experiment_type: ChaosExperimentType::NetworkLatency,
                duration: Duration::from_secs(60),
                intensity: 0.5,
                target: "api-integration".to_string(),
            },
            ChaosExperiment {
                name: "memory_pressure".to_string(),
                experiment_type: ChaosExperimentType::MemoryPressure,
                duration: Duration::from_secs(30),
                intensity: 0.8,
                target: "memory-manager".to_string(),
            },
        ];
        
        // Stricter performance thresholds
        config.performance.thresholds.max_latency_ms = 50;
        config.performance.thresholds.min_throughput_ops = 10000;
        
        config
    }
    
    /// Create a configuration for CI/CD environments
    pub fn ci_cd() -> Self {
        let mut config = Self::default();
        
        // Reduce test cases for faster execution
        config.property_testing.test_cases = 100;
        
        // Disable chaos engineering in CI
        config.chaos.enabled = false;
        
        // Relaxed performance thresholds for CI
        config.performance.thresholds.max_latency_ms = 200;
        config.performance.thresholds.min_throughput_ops = 100;
        
        config
    }
}

impl QaSentinelConfig {
    /// Get monitoring interval
    pub fn monitoring_interval(&self) -> Duration {
        self.monitoring.interval
    }
}