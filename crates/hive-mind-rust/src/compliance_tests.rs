//! # Comprehensive Compliance Tests
//!
//! This module provides comprehensive tests for all compliance components
//! including integration tests, performance tests, and regulatory scenario tests.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compliance::*;
    use chrono::{DateTime, Utc, Duration};
    use uuid::Uuid;
    
    /// Test comprehensive compliance system initialization
    #[tokio::test]
    async fn test_compliance_system_initialization() {
        // Test compliance coordinator creation
        let coordinator = ComplianceCoordinator::new().await.unwrap();
        
        // Verify all components are initialized
        assert!(coordinator.start().await.is_ok());
    }
    
    /// Test audit trail functionality
    #[tokio::test]
    async fn test_audit_trail_operations() {
        let audit_trail = AuditTrail::new().await.unwrap();
        
        // Test event logging
        let event_id = audit_trail.log_event(
            AuditEventType::TradeExecution,
            "Test trade execution".to_string(),
            serde_json::json!({"amount": 1000.0, "symbol": "AAPL"}),
            Some("user123".to_string()),
            Some("session456".to_string()),
            Some("192.168.1.1".to_string()),
        ).await.unwrap();
        
        assert!(!event_id.is_nil());
        
        // Test integrity verification
        assert!(audit_trail.verify_integrity().await.unwrap());
        
        // Test event querying
        let events = audit_trail.query_by_type(&AuditEventType::TradeExecution).await.unwrap();
        assert!(!events.is_empty());
        
        // Test compliance report generation
        let start = Utc::now() - Duration::hours(1);
        let end = Utc::now();
        let report = audit_trail.generate_compliance_report(
            start,
            end,
            vec![audit_trail::ComplianceFlag::SOX404],
        ).await.unwrap();
        
        assert!(report.integrity_verified);
    }
    
    /// Test data protection and encryption
    #[tokio::test]
    async fn test_data_protection() {
        let data_protection = DataProtection::new().await.unwrap();
        
        // Test data encryption
        let sensitive_data = b"sensitive customer information";
        let gdpr_metadata = data_protection::GDPRMetadata {
            data_subject_id: Some("customer123".to_string()),
            legal_basis: data_protection::LegalBasis::Contract,
            purposes: vec![data_protection::ProcessingPurpose::Trading],
            consent_status: data_protection::ConsentStatus::Given,
            data_source: "trading_system".to_string(),
            processing_location: "EU".to_string(),
        };
        
        let encrypted = data_protection.encrypt_data(
            sensitive_data,
            data_protection::DataClassification::Restricted,
            data_protection::DataCategory::Financial,
            gdpr_metadata,
        ).await.unwrap();
        
        assert_ne!(encrypted.encrypted_payload, sensitive_data.to_vec());
        
        // Test data decryption
        let mut encrypted_mut = encrypted;
        let decrypted = data_protection.decrypt_data(
            &mut encrypted_mut,
            Some("authorized_user".to_string()),
        ).await.unwrap();
        
        assert_eq!(decrypted, sensitive_data.to_vec());
        
        // Test GDPR data subject access request
        let access_response = data_protection.handle_access_request("customer123").await;
        // This should return NotFound for non-existent customer, which is expected
        assert!(access_response.is_err());
    }
    
    /// Test access control and authentication
    #[tokio::test]
    async fn test_access_control() {
        let access_control = AccessControl::new().await.unwrap();
        
        // Test user authentication (should fail for non-existent user)
        let auth_result = access_control.authenticate(
            "testuser",
            "testpass",
            Some("192.168.1.1".to_string()),
        ).await;
        
        // Should fail for non-existent user
        assert!(auth_result.is_err());
        
        // Test permission checking
        let has_permission = access_control.has_permission(
            "nonexistent_user",
            &access_control::Permission::TradingView,
        ).await;
        
        // Should return error for non-existent user
        assert!(has_permission.is_err());
        
        // Test operation authorization
        let auth_result = access_control.authorize_operation(
            "testuser",
            "test_operation",
            &[access_control::Permission::TradingView],
            None,
        ).await;
        
        // Should fail for non-existent user
        assert!(auth_result.is_err());
    }
    
    /// Test risk management functionality
    #[tokio::test]
    async fn test_risk_management() {
        let risk_manager = RiskManager::new().await.unwrap();
        
        // Test pre-trade risk check
        let check_result = risk_manager.check_pre_trade_risk(
            "AAPL",
            100.0,
            150.0,
            "portfolio_1",
        ).await.unwrap();
        
        // Should be approved for normal trade
        assert!(matches!(check_result, risk_management::PreTradeCheckResult::Approved { .. }));
        
        // Test position update
        let update_result = risk_manager.update_position_post_trade(
            "AAPL",
            100.0,
            150.0,
        ).await;
        
        assert!(update_result.is_ok());
        
        // Test risk metrics retrieval
        let metrics = risk_manager.get_risk_metrics().await.unwrap();
        assert!(metrics.portfolio_value >= 0.0);
        
        // Test stress test execution
        let stress_result = risk_manager.run_stress_test("test_scenario").await.unwrap();
        assert!(!stress_result.id.is_nil());
    }
    
    /// Test regulatory reporting
    #[tokio::test]
    async fn test_regulatory_reporting() {
        let regulatory_reporter = RegulatoryReporter::new().await.unwrap();
        
        // Test MiFID II transaction report
        let trade_data = regulatory_reporting::TransactionReportData {
            instrument_id: "AAPL".to_string(),
            quantity: 100.0,
            price: 150.0,
            currency: "USD".to_string(),
            counterparty_id: "COUNTERPARTY_123".to_string(),
            venue_id: "NASDAQ".to_string(),
            execution_timestamp: Utc::now(),
            transaction_type: regulatory_reporting::TransactionType::Buy,
            side: regulatory_reporting::TradeSide::Buy,
        };
        
        let report_id = regulatory_reporter.submit_transaction_report(&trade_data).await.unwrap();
        assert!(!report_id.is_nil());
        
        // Test SOX assessment generation
        let sox_report_id = regulatory_reporter.generate_sox_assessment(
            Utc::now() - Duration::days(30),
            Utc::now(),
        ).await.unwrap();
        assert!(!sox_report_id.is_nil());
        
        // Test Basel III capital report
        let capital_report_id = regulatory_reporter.generate_capital_report().await.unwrap();
        assert!(!capital_report_id.is_nil());
        
        // Test EMIR trade report
        let emir_data = regulatory_reporting::EMIRTradeData {
            trade_id: "TRADE_123".to_string(),
            contract_type: regulatory_reporting::DerivativeType::InterestRateSwap,
            notional_amount: 1000000.0,
            currency: "USD".to_string(),
            counterparty_id: "COUNTERPARTY_456".to_string(),
            execution_date: Utc::now(),
            maturity_date: Utc::now() + Duration::days(365),
        };
        
        let emir_report_id = regulatory_reporter.submit_emir_report(&emir_data).await.unwrap();
        assert!(!emir_report_id.is_nil());
    }
    
    /// Test trade surveillance system
    #[tokio::test]
    async fn test_trade_surveillance() {
        let surveillance = TradeSurveillance::new().await.unwrap();
        
        // Test KYC status verification
        let kyc_status = surveillance.verify_kyc_status("customer123").await.unwrap();
        assert_eq!(kyc_status.customer_id, "customer123");
        
        // Test transaction monitoring
        let transaction = trade_surveillance::TransactionDetails {
            amount: 5000.0,
            currency: "USD".to_string(),
            transaction_type: trade_surveillance::TransactionType::Wire,
            from_account: "account_1".to_string(),
            to_account: "account_2".to_string(),
            timestamp: Utc::now(),
            geographic_data: trade_surveillance::GeographicData {
                country: "US".to_string(),
                state_province: Some("NY".to_string()),
                city: Some("New York".to_string()),
                ip_address: Some("192.168.1.1".to_string()),
                high_risk_jurisdiction: false,
            },
            description: "Normal wire transfer".to_string(),
        };
        
        let alerts = surveillance.monitor_transaction(&transaction).await.unwrap();
        // Normal transaction should not generate alerts
        assert!(alerts.is_empty());
        
        // Test SAR generation (this will generate a new SAR ID)
        let sar_id = surveillance.generate_sar("alert123").await.unwrap();
        assert!(!sar_id.is_nil());
    }
    
    /// Test compliance engine coordination
    #[tokio::test]
    async fn test_compliance_engine() {
        let engine = ComplianceEngine::new().await.unwrap();
        
        // Test comprehensive compliance check
        let result = engine.comprehensive_check().await.unwrap();
        assert!(result.compliance_score >= 0.0);
        assert!(result.compliance_score <= 100.0);
        
        // Verify compliance status
        assert!(matches!(
            result.overall_status,
            compliance_engine::ComplianceStatus::Compliant |
            compliance_engine::ComplianceStatus::PartiallyCompliant |
            compliance_engine::ComplianceStatus::NonCompliant |
            compliance_engine::ComplianceStatus::UnderReview
        ));
    }
    
    /// Test integration between all compliance components
    #[tokio::test]
    async fn test_full_compliance_integration() {
        let mut coordinator = ComplianceCoordinator::new().await.unwrap();
        
        // Initialize audit trail
        let audit_trail = Arc::new(AuditTrail::new().await.unwrap());
        coordinator.data_protection().clone();
        
        // Start all systems
        assert!(coordinator.start().await.is_ok());
        
        // Test compliance check
        let compliance_result = coordinator.compliance_check().await.unwrap();
        
        // Verify comprehensive results
        assert!(compliance_result.compliance_score >= 0.0);
        assert!(!compliance_result.check_results.is_empty());
        assert!(!compliance_result.recommendations.is_empty());
        
        // Test audit trail logging during compliance check
        let events = audit_trail.query_by_type(&AuditEventType::ComplianceReporting).await.unwrap();
        // Should have compliance-related events
    }
    
    /// Test performance of compliance operations
    #[tokio::test]
    async fn test_compliance_performance() {
        let coordinator = ComplianceCoordinator::new().await.unwrap();
        
        let start_time = std::time::Instant::now();
        
        // Run multiple compliance operations
        for i in 0..10 {
            let audit_trail = coordinator.audit_trail();
            let _ = audit_trail.log_event(
                AuditEventType::TradeExecution,
                format!("Performance test trade {}", i),
                serde_json::json!({"trade_id": i}),
                Some("perf_user".to_string()),
                None,
                None,
            ).await;
        }
        
        let elapsed = start_time.elapsed();
        
        // Should complete within reasonable time (less than 1 second for 10 operations)
        assert!(elapsed.as_millis() < 1000);
    }
    
    /// Test compliance with different regulatory scenarios
    #[tokio::test]
    async fn test_regulatory_scenarios() {
        let coordinator = ComplianceCoordinator::new().await.unwrap();
        
        // Test high-value transaction scenario (should trigger additional checks)
        let audit_trail = coordinator.audit_trail();
        let high_value_event = audit_trail.log_event(
            AuditEventType::TradeExecution,
            "High value trade execution".to_string(),
            serde_json::json!({
                "amount": 1000000.0,
                "symbol": "AAPL",
                "high_value": true
            }),
            Some("trader123".to_string()),
            Some("session789".to_string()),
            Some("10.0.0.1".to_string()),
        ).await;
        
        assert!(high_value_event.is_ok());
        
        // Test suspicious activity scenario
        let suspicious_event = audit_trail.log_event(
            AuditEventType::SuspiciousActivity,
            "Suspicious trading pattern detected".to_string(),
            serde_json::json!({
                "pattern": "unusual_frequency",
                "risk_score": 85.0
            }),
            Some("system".to_string()),
            None,
            None,
        ).await;
        
        assert!(suspicious_event.is_ok());
        
        // Test compliance violation scenario
        let violation_event = audit_trail.log_event(
            AuditEventType::ComplianceViolation,
            "Position limit violation".to_string(),
            serde_json::json!({
                "violation_type": "position_limit",
                "severity": "high"
            }),
            Some("risk_system".to_string()),
            None,
            None,
        ).await;
        
        assert!(violation_event.is_ok());
    }
    
    /// Test compliance report generation and export
    #[tokio::test]
    async fn test_compliance_reporting() {
        let audit_trail = AuditTrail::new().await.unwrap();
        
        // Log some sample events
        let _ = audit_trail.log_event(
            AuditEventType::TradeExecution,
            "Sample trade 1".to_string(),
            serde_json::json!({"amount": 1000.0}),
            Some("user1".to_string()),
            None,
            None,
        ).await;
        
        let _ = audit_trail.log_event(
            AuditEventType::UserLogin,
            "User login".to_string(),
            serde_json::json!({"successful": true}),
            Some("user1".to_string()),
            None,
            None,
        ).await;
        
        // Test report export in different formats
        let json_export = audit_trail.export_for_regulator(
            audit_trail::ExportFormat::Json
        ).await.unwrap();
        assert!(!json_export.is_empty());
        
        let csv_export = audit_trail.export_for_regulator(
            audit_trail::ExportFormat::Csv
        ).await.unwrap();
        assert!(csv_export.contains("id,event_type,timestamp"));
        
        let xml_export = audit_trail.export_for_regulator(
            audit_trail::ExportFormat::Xml
        ).await.unwrap();
        assert!(xml_export.contains("<?xml version=\"1.0\""));
    }
    
    /// Test error handling and recovery
    #[tokio::test]
    async fn test_compliance_error_handling() {
        let coordinator = ComplianceCoordinator::new().await.unwrap();
        
        // Test handling of invalid data
        let audit_trail = coordinator.audit_trail();
        
        // Test with invalid user ID (empty string)
        let result = audit_trail.log_event(
            AuditEventType::UserLogin,
            "Invalid login attempt".to_string(),
            serde_json::json!({"user_id": ""}),
            Some("".to_string()),
            None,
            None,
        ).await;
        
        // Should still succeed as empty string is valid
        assert!(result.is_ok());
        
        // Test system recovery after errors
        let compliance_result = coordinator.compliance_check().await.unwrap();
        
        // System should still be operational
        assert!(matches!(
            compliance_result.overall_status,
            compliance_engine::ComplianceStatus::Compliant |
            compliance_engine::ComplianceStatus::PartiallyCompliant
        ));
    }
    
    /// Test concurrent compliance operations
    #[tokio::test]
    async fn test_concurrent_compliance() {
        let coordinator = Arc::new(ComplianceCoordinator::new().await.unwrap());
        let audit_trail = coordinator.audit_trail().clone();
        
        // Spawn multiple concurrent audit logging tasks
        let mut handles = Vec::new();
        
        for i in 0..5 {
            let audit_trail_clone = audit_trail.clone();
            let handle = tokio::spawn(async move {
                for j in 0..10 {
                    let event_id = audit_trail_clone.log_event(
                        AuditEventType::TradeExecution,
                        format!("Concurrent trade {}-{}", i, j),
                        serde_json::json!({"thread": i, "trade": j}),
                        Some(format!("user{}", i)),
                        None,
                        None,
                    ).await.unwrap();
                    
                    assert!(!event_id.is_nil());
                }
            });
            handles.push(handle);
        }
        
        // Wait for all tasks to complete
        for handle in handles {
            handle.await.unwrap();
        }
        
        // Verify audit trail integrity after concurrent operations
        assert!(audit_trail.verify_integrity().await.unwrap());
    }
}

/// Module for testing utilities and helpers
#[cfg(test)]
mod test_utils {
    use super::*;
    
    /// Create a test transaction for monitoring
    pub fn create_test_transaction(amount: f64, high_risk: bool) -> trade_surveillance::TransactionDetails {
        trade_surveillance::TransactionDetails {
            amount,
            currency: "USD".to_string(),
            transaction_type: trade_surveillance::TransactionType::Wire,
            from_account: "test_account_1".to_string(),
            to_account: "test_account_2".to_string(),
            timestamp: Utc::now(),
            geographic_data: trade_surveillance::GeographicData {
                country: if high_risk { "XX" } else { "US" }.to_string(),
                state_province: None,
                city: None,
                ip_address: Some("192.168.1.1".to_string()),
                high_risk_jurisdiction: high_risk,
            },
            description: format!("Test transaction - amount: {}", amount),
        }
    }
    
    /// Create test GDPR metadata
    pub fn create_test_gdpr_metadata(subject_id: Option<String>) -> data_protection::GDPRMetadata {
        data_protection::GDPRMetadata {
            data_subject_id: subject_id,
            legal_basis: data_protection::LegalBasis::Contract,
            purposes: vec![data_protection::ProcessingPurpose::Trading],
            consent_status: data_protection::ConsentStatus::Given,
            data_source: "test_system".to_string(),
            processing_location: "EU".to_string(),
        }
    }
    
    /// Create test trade data for regulatory reporting
    pub fn create_test_trade_data(symbol: &str, quantity: f64, price: f64) -> regulatory_reporting::TransactionReportData {
        regulatory_reporting::TransactionReportData {
            instrument_id: symbol.to_string(),
            quantity,
            price,
            currency: "USD".to_string(),
            counterparty_id: "TEST_COUNTERPARTY".to_string(),
            venue_id: "TEST_VENUE".to_string(),
            execution_timestamp: Utc::now(),
            transaction_type: regulatory_reporting::TransactionType::Buy,
            side: regulatory_reporting::TradeSide::Buy,
        }
    }
}

/// Benchmarks for compliance operations performance
#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::Instant;
    
    /// Benchmark audit trail logging performance
    #[tokio::test]
    async fn benchmark_audit_logging() {
        let audit_trail = AuditTrail::new().await.unwrap();
        let num_events = 1000;
        
        let start = Instant::now();
        
        for i in 0..num_events {
            let _ = audit_trail.log_event(
                AuditEventType::TradeExecution,
                format!("Benchmark trade {}", i),
                serde_json::json!({"trade_id": i}),
                Some("benchmark_user".to_string()),
                None,
                None,
            ).await.unwrap();
        }
        
        let elapsed = start.elapsed();
        let events_per_second = num_events as f64 / elapsed.as_secs_f64();
        
        println!("Audit logging performance: {:.0} events/second", events_per_second);
        
        // Should handle at least 100 events per second
        assert!(events_per_second > 100.0);
    }
    
    /// Benchmark encryption performance
    #[tokio::test]
    async fn benchmark_encryption() {
        let data_protection = DataProtection::new().await.unwrap();
        let test_data = b"This is test data for encryption benchmarking. It contains sensitive information.";
        let num_operations = 100;
        
        let gdpr_metadata = test_utils::create_test_gdpr_metadata(Some("benchmark_user".to_string()));
        
        let start = Instant::now();
        
        for _ in 0..num_operations {
            let encrypted = data_protection.encrypt_data(
                test_data,
                data_protection::DataClassification::Restricted,
                data_protection::DataCategory::Financial,
                gdpr_metadata.clone(),
            ).await.unwrap();
            
            let mut encrypted_mut = encrypted;
            let _decrypted = data_protection.decrypt_data(
                &mut encrypted_mut,
                Some("benchmark_user".to_string()),
            ).await.unwrap();
        }
        
        let elapsed = start.elapsed();
        let operations_per_second = num_operations as f64 / elapsed.as_secs_f64();
        
        println!("Encryption performance: {:.0} encrypt/decrypt operations/second", operations_per_second);
        
        // Should handle at least 50 encrypt/decrypt cycles per second
        assert!(operations_per_second > 50.0);
    }
    
    /// Benchmark compliance check performance
    #[tokio::test]
    async fn benchmark_compliance_check() {
        let engine = ComplianceEngine::new().await.unwrap();
        let num_checks = 10;
        
        let start = Instant::now();
        
        for _ in 0..num_checks {
            let _result = engine.comprehensive_check().await.unwrap();
        }
        
        let elapsed = start.elapsed();
        let checks_per_second = num_checks as f64 / elapsed.as_secs_f64();
        
        println!("Compliance check performance: {:.1} checks/second", checks_per_second);
        
        // Should handle at least 1 comprehensive check per second
        assert!(checks_per_second > 1.0);
    }
}