pub mod scanner;
pub mod e2b_client;

// Re-export main scanner types for easy access
pub use scanner::{
    ApiScanner, ScannerConfig, ScanResult, ScanStatus,
    OpenAPISpec, EndpointInfo, Vulnerability, VulnerabilitySeverity,
    PerformanceMetrics, ScannerAgentDB, ScannerAgenticFlow,
    HttpMethod, AuthMethod, VulnerabilityType, DiscoveryMethod,
};

// Re-export E2B client types
pub use e2b_client::{
    E2BClient, SandboxConfig, Sandbox, ExecutionRequest,
    ExecutionResult, FileUpload, LogEntry,
};
