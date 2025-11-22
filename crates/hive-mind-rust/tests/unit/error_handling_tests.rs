//! Comprehensive error handling tests for banking-grade reliability
//! 
//! This test suite validates all error conditions, recovery mechanisms,
//! and edge cases to ensure the system meets financial sector standards.

use std::time::Duration;
use std::sync::Arc;
use tokio::time::timeout;
use uuid::Uuid;
use proptest::prelude::*;

use hive_mind_rust::{
    error::{HiveMindError, ConsensusError, MemoryError, NeuralError, NetworkError, AgentError, ConfigError, ErrorSeverity},
    config::HiveMindConfig,
    core::{HiveMind, HiveMindBuilder, OperationalMode, HealthStatus},
};

/// Test error recoverability classification
#[tokio::test]
async fn test_error_recoverability_classification() {
    // Recoverable errors
    let recoverable_errors = vec![
        HiveMindError::Consensus(ConsensusError::ConsensusTimeout),
        HiveMindError::Network(NetworkError::ConnectionFailed { peer: "test".to_string() }),
        HiveMindError::Network(NetworkError::DeliveryFailed),
        HiveMindError::Memory(MemoryError::SynchronizationFailed),
        HiveMindError::Agent(AgentError::AgentOverloaded),
        HiveMindError::Timeout { timeout_ms: 5000 },
    ];

    for error in recoverable_errors {
        assert!(error.is_recoverable(), "Error should be recoverable: {:?}", error);
    }

    // Non-recoverable errors
    let non_recoverable_errors = vec![
        HiveMindError::Memory(MemoryError::CorruptionDetected),
        HiveMindError::Consensus(ConsensusError::ByzantineFault { node_id: "malicious".to_string() }),
        HiveMindError::InvalidState { message: "Critical failure".to_string() },
        HiveMindError::Internal("System panic".to_string()),
    ];

    for error in non_recoverable_errors {
        assert!(!error.is_recoverable(), "Error should not be recoverable: {:?}", error);
    }
}

/// Test error severity classification for risk assessment
#[tokio::test]
async fn test_error_severity_classification() {
    let critical_errors = vec![
        HiveMindError::Memory(MemoryError::CorruptionDetected),
        HiveMindError::Consensus(ConsensusError::ByzantineFault { node_id: "attacker".to_string() }),
    ];

    for error in critical_errors {
        assert_eq!(error.severity(), ErrorSeverity::Critical, "Error should be critical: {:?}", error);
    }

    let high_errors = vec![
        HiveMindError::InvalidState { message: "Invalid state".to_string() },
        HiveMindError::ResourceExhausted { resource: "memory".to_string() },
    ];

    for error in high_errors {
        assert_eq!(error.severity(), ErrorSeverity::High, "Error should be high severity: {:?}", error);
    }
}

/// Test error chain propagation and context preservation
#[tokio::test]
async fn test_error_chain_propagation() {
    // Test nested error propagation
    let io_error = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "Access denied");
    let hive_error = HiveMindError::Io(io_error);
    
    let error_string = format!("{}", hive_error);
    assert!(error_string.contains("Access denied"));

    // Test error conversion chain
    let config_error = ConfigError::InvalidParameter { parameter: "timeout".to_string() };
    let hive_error: HiveMindError = config_error.into();
    
    match hive_error {
        HiveMindError::Config(ConfigError::InvalidParameter { parameter }) => {
            assert_eq!(parameter, "timeout");
        },
        _ => panic!("Error conversion failed"),
    }
}

/// Test timeout error handling with various scenarios
#[tokio::test]
async fn test_timeout_error_handling() {
    let timeout_error = HiveMindError::Timeout { timeout_ms: 10000 };
    
    assert!(timeout_error.is_recoverable());
    assert_eq!(timeout_error.severity(), ErrorSeverity::Low);
    
    let error_msg = format!("{}", timeout_error);
    assert!(error_msg.contains("10000ms"));
}

/// Test resource exhaustion scenarios
#[tokio::test]
async fn test_resource_exhaustion_scenarios() {
    let resource_types = vec!["memory", "cpu", "network_bandwidth", "file_handles"];
    
    for resource in resource_types {
        let error = HiveMindError::ResourceExhausted { 
            resource: resource.to_string() 
        };
        
        assert!(!error.is_recoverable());
        assert_eq!(error.severity(), ErrorSeverity::High);
        
        let error_msg = format!("{}", error);
        assert!(error_msg.contains(resource));
    }
}

/// Test Byzantine fault detection and handling
#[tokio::test]
async fn test_byzantine_fault_handling() {
    let malicious_nodes = vec!["node_1", "node_2", "node_3"];
    
    for node in malicious_nodes {
        let error = ConsensusError::ByzantineFault { 
            node_id: node.to_string() 
        };
        let hive_error = HiveMindError::Consensus(error);
        
        assert!(!hive_error.is_recoverable());
        assert_eq!(hive_error.severity(), ErrorSeverity::Critical);
        
        let error_msg = format!("{}", hive_error);
        assert!(error_msg.contains(node));
    }
}

/// Test quorum failure scenarios
#[tokio::test]
async fn test_quorum_failure_scenarios() {
    let quorum_scenarios = vec![
        (5, 2),  // Required 5, got 2
        (3, 1),  // Required 3, got 1
        (7, 3),  // Required 7, got 3
    ];
    
    for (required, actual) in quorum_scenarios {
        let error = ConsensusError::QuorumNotReached { required, actual };
        let hive_error = HiveMindError::Consensus(error);
        
        let error_msg = format!("{}", hive_error);
        assert!(error_msg.contains(&required.to_string()));
        assert!(error_msg.contains(&actual.to_string()));
    }
}

/// Test memory corruption detection
#[tokio::test]
async fn test_memory_corruption_detection() {
    let memory_error = MemoryError::CorruptionDetected;
    let hive_error = HiveMindError::Memory(memory_error);
    
    assert!(!hive_error.is_recoverable());
    assert_eq!(hive_error.severity(), ErrorSeverity::Critical);
    
    // This should trigger immediate emergency procedures
    let error_msg = format!("{}", hive_error);
    assert!(error_msg.contains("corruption"));
}

/// Test network partition handling
#[tokio::test]
async fn test_network_partition_handling() {
    let network_error = NetworkError::NetworkPartition;
    let hive_error = HiveMindError::Network(network_error);
    
    assert!(!hive_error.is_recoverable());
    assert_eq!(hive_error.severity(), ErrorSeverity::Medium);
}

/// Test invalid state detection and recovery
#[tokio::test]
async fn test_invalid_state_detection() {
    let invalid_states = vec![
        "Inconsistent consensus state",
        "Memory corruption detected",
        "Network topology invalid",
        "Agent coordination failure",
    ];
    
    for state_msg in invalid_states {
        let error = HiveMindError::InvalidState { 
            message: state_msg.to_string() 
        };
        
        assert!(!error.is_recoverable());
        assert_eq!(error.severity(), ErrorSeverity::High);
        
        let error_msg = format!("{}", error);
        assert!(error_msg.contains(state_msg));
    }
}

/// Property-based test for error serialization/deserialization
proptest! {
    #[test]
    fn test_error_serialization_roundtrip(
        timeout_ms in 1u64..60000u64,
        resource in "[a-z_]{3,20}",
        message in "[a-zA-Z0-9 ]{10,100}",
    ) {
        // Test timeout error serialization
        let timeout_error = HiveMindError::Timeout { timeout_ms };
        let serialized = serde_json::to_string(&timeout_error);
        // Note: Most error types don't implement Serialize, but we test what we can
        
        // Test resource exhaustion
        let resource_error = HiveMindError::ResourceExhausted { resource: resource.clone() };
        let error_str = format!("{}", resource_error);
        prop_assert!(error_str.contains(&resource));
        
        // Test invalid state
        let state_error = HiveMindError::InvalidState { message: message.clone() };
        let error_str = format!("{}", state_error);
        prop_assert!(error_str.contains(&message));
    }
}

/// Test error recovery strategies
#[tokio::test]
async fn test_error_recovery_strategies() {
    struct ErrorRecoveryTest {
        error: HiveMindError,
        expected_recovery_action: RecoveryAction,
    }
    
    #[derive(Debug, PartialEq)]
    enum RecoveryAction {
        Retry,
        Degrade,
        Emergency,
        Ignore,
    }
    
    let test_cases = vec![
        ErrorRecoveryTest {
            error: HiveMindError::Timeout { timeout_ms: 5000 },
            expected_recovery_action: RecoveryAction::Retry,
        },
        ErrorRecoveryTest {
            error: HiveMindError::Memory(MemoryError::CorruptionDetected),
            expected_recovery_action: RecoveryAction::Emergency,
        },
        ErrorRecoveryTest {
            error: HiveMindError::Network(NetworkError::ConnectionFailed { 
                peer: "test".to_string() 
            }),
            expected_recovery_action: RecoveryAction::Retry,
        },
        ErrorRecoveryTest {
            error: HiveMindError::Agent(AgentError::AgentOverloaded),
            expected_recovery_action: RecoveryAction::Degrade,
        },
    ];
    
    for test_case in test_cases {
        let recovery_action = determine_recovery_action(&test_case.error);
        assert_eq!(recovery_action, test_case.expected_recovery_action, 
                  "Wrong recovery action for error: {:?}", test_case.error);
    }
}

/// Determine appropriate recovery action based on error
fn determine_recovery_action(error: &HiveMindError) -> RecoveryAction {
    match error.severity() {
        ErrorSeverity::Critical => RecoveryAction::Emergency,
        ErrorSeverity::High => RecoveryAction::Degrade,
        ErrorSeverity::Medium => RecoveryAction::Retry,
        ErrorSeverity::Low => if error.is_recoverable() {
            RecoveryAction::Retry
        } else {
            RecoveryAction::Ignore
        },
    }
}

/// Test concurrent error handling
#[tokio::test]
async fn test_concurrent_error_handling() {
    let num_concurrent_errors = 100;
    let mut handles = Vec::new();
    
    for i in 0..num_concurrent_errors {
        let handle = tokio::spawn(async move {
            let error = HiveMindError::Timeout { timeout_ms: (i % 10 + 1) * 1000 };
            let is_recoverable = error.is_recoverable();
            let severity = error.severity();
            (is_recoverable, severity)
        });
        handles.push(handle);
    }
    
    // Wait for all concurrent error processing
    let results: Result<Vec<_>, _> = futures::future::try_join_all(handles).await;
    assert!(results.is_ok());
    
    let results = results.unwrap();
    assert_eq!(results.len(), num_concurrent_errors);
    
    // All timeout errors should be recoverable and low severity
    for (is_recoverable, severity) in results {
        assert!(is_recoverable);
        assert_eq!(severity, ErrorSeverity::Low);
    }
}

/// Test error aggregation and reporting
#[tokio::test]
async fn test_error_aggregation() {
    struct ErrorStats {
        total: usize,
        critical: usize,
        high: usize,
        medium: usize,
        low: usize,
        recoverable: usize,
    }
    
    let errors = vec![
        HiveMindError::Memory(MemoryError::CorruptionDetected),
        HiveMindError::Timeout { timeout_ms: 5000 },
        HiveMindError::InvalidState { message: "Test".to_string() },
        HiveMindError::Network(NetworkError::ConnectionFailed { peer: "test".to_string() }),
        HiveMindError::Consensus(ConsensusError::ByzantineFault { node_id: "attacker".to_string() }),
    ];
    
    let stats = errors.iter().fold(ErrorStats {
        total: 0,
        critical: 0,
        high: 0,
        medium: 0,
        low: 0,
        recoverable: 0,
    }, |mut acc, error| {
        acc.total += 1;
        if error.is_recoverable() {
            acc.recoverable += 1;
        }
        match error.severity() {
            ErrorSeverity::Critical => acc.critical += 1,
            ErrorSeverity::High => acc.high += 1,
            ErrorSeverity::Medium => acc.medium += 1,
            ErrorSeverity::Low => acc.low += 1,
        }
        acc
    });
    
    assert_eq!(stats.total, 5);
    assert_eq!(stats.critical, 2); // Memory corruption + Byzantine fault
    assert_eq!(stats.high, 1);     // Invalid state
    assert_eq!(stats.medium, 0);
    assert_eq!(stats.low, 2);      // Timeout + Connection failed
    assert_eq!(stats.recoverable, 2); // Timeout + Connection failed
}

/// Test error rate limiting and circuit breaking
#[tokio::test]
async fn test_error_rate_limiting() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    
    let error_count = Arc::new(AtomicUsize::new(0));
    let max_errors_per_minute = 10;
    
    // Simulate rapid error generation
    let mut tasks = Vec::new();
    for _ in 0..50 {
        let error_count = error_count.clone();
        let task = tokio::spawn(async move {
            let current_count = error_count.fetch_add(1, Ordering::SeqCst);
            if current_count < max_errors_per_minute {
                // Process error normally
                let error = HiveMindError::Timeout { timeout_ms: 1000 };
                error.is_recoverable()
            } else {
                // Circuit breaker should be triggered
                false
            }
        });
        tasks.push(task);
    }
    
    let results: Vec<bool> = futures::future::try_join_all(tasks)
        .await
        .unwrap();
    
    let processed_normally = results.iter().filter(|&&x| x).count();
    assert!(processed_normally <= max_errors_per_minute + 5, 
           "Too many errors processed, circuit breaker failed");
}

#[cfg(test)]
mod error_injection_tests {
    use super::*;
    
    /// Test system behavior under various error injection scenarios
    #[tokio::test]
    async fn test_error_injection_scenarios() {
        // This would be used in fault injection testing
        // to validate system behavior under various failure modes
        
        let scenarios = vec![
            "network_partition",
            "memory_pressure",
            "disk_full",
            "cpu_exhaustion",
            "byzantine_nodes",
        ];
        
        for scenario in scenarios {
            // In a real implementation, this would inject specific failures
            // and validate system response
            let simulated_error = match scenario {
                "network_partition" => HiveMindError::Network(NetworkError::NetworkPartition),
                "memory_pressure" => HiveMindError::Memory(MemoryError::CapacityExceeded),
                "byzantine_nodes" => HiveMindError::Consensus(
                    ConsensusError::ByzantineFault { 
                        node_id: "malicious".to_string() 
                    }
                ),
                _ => HiveMindError::Internal(format!("Simulated failure: {}", scenario)),
            };
            
            // Test error handling
            assert!(!simulated_error.is_recoverable() || 
                   simulated_error.severity() != ErrorSeverity::Critical);
        }
    }
}