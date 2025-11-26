//! Unit Tests for Agent Deployment System
//!
//! Tests cover:
//! 1. Agent deployment logic and validation
//! 2. E2B client wrapper operations
//! 3. OpenRouter client integration
//! 4. Database operations for agent tracking
//! 5. Error handling and edge cases

use serde_json::json;

mod fixtures;
use fixtures::*;

// ============================================================================
// Unit Tests - E2B Client Operations
// ============================================================================

#[tokio::test]
async fn test_e2b_client_initialization() {
    let client = MockE2BClient::new("test-api-key");
    assert!(client.is_initialized());
}

#[tokio::test]
async fn test_e2b_client_create_sandbox() {
    let client = MockE2BClient::new("test-api-key");

    let sandbox_config = SandboxConfig {
        template: "nodejs".to_string(),
        timeout: 3600,
        env_vars: vec![
            ("NODE_ENV".to_string(), "production".to_string()),
        ],
        metadata: json!({
            "agent_type": "researcher",
            "capabilities": ["analysis", "research"]
        }),
    };

    let result = client.create_sandbox(sandbox_config).await;
    assert!(result.is_ok(), "Failed to create sandbox: {:?}", result.err());

    let sandbox = result.unwrap();
    assert!(!sandbox.sandbox_id.is_empty());
    assert_eq!(sandbox.status, "running");
    assert_eq!(sandbox.template, "nodejs");
}

#[tokio::test]
async fn test_e2b_sandbox_lifecycle() {
    let client = MockE2BClient::new("test-api-key");

    // Create sandbox
    let sandbox = client.create_sandbox(SandboxConfig::default()).await.unwrap();
    let sandbox_id = sandbox.sandbox_id.clone();

    // Check status
    let status = client.get_sandbox_status(&sandbox_id).await.unwrap();
    assert_eq!(status, "running");

    // Execute command
    let exec_result = client.execute_command(&sandbox_id, "echo 'Hello World'", None).await;
    assert!(exec_result.is_ok());

    let output = exec_result.unwrap();
    assert_eq!(output.stdout.trim(), "Hello World");
    assert_eq!(output.exit_code, 0);

    // Stop sandbox
    let stop_result = client.stop_sandbox(&sandbox_id).await;
    assert!(stop_result.is_ok());

    // Verify stopped
    let final_status = client.get_sandbox_status(&sandbox_id).await.unwrap();
    assert_eq!(final_status, "stopped");
}

#[tokio::test]
async fn test_e2b_sandbox_environment_variables() {
    let client = MockE2BClient::new("test-api-key");

    let config = SandboxConfig {
        template: "nodejs".to_string(),
        env_vars: vec![
            ("API_KEY".to_string(), "secret-key-123".to_string()),
            ("DEBUG".to_string(), "true".to_string()),
        ],
        ..Default::default()
    };

    let sandbox = client.create_sandbox(config).await.unwrap();

    // Verify env vars are set
    let output = client.execute_command(
        &sandbox.sandbox_id,
        "printenv API_KEY",
        None
    ).await.unwrap();

    assert_eq!(output.stdout.trim(), "secret-key-123");
}

#[tokio::test]
async fn test_e2b_sandbox_file_operations() {
    let client = MockE2BClient::new("test-api-key");
    let sandbox = client.create_sandbox(SandboxConfig::default()).await.unwrap();

    // Upload file
    let file_content = "console.log('Test script');";
    let upload_result = client.upload_file(
        &sandbox.sandbox_id,
        "/app/test.js",
        file_content.as_bytes()
    ).await;
    assert!(upload_result.is_ok());

    // Read file back
    let read_result = client.read_file(&sandbox.sandbox_id, "/app/test.js").await;
    assert!(read_result.is_ok());
    assert_eq!(read_result.unwrap(), file_content);

    // List files
    let files = client.list_files(&sandbox.sandbox_id, "/app").await.unwrap();
    assert!(files.contains(&"/app/test.js".to_string()));
}

#[tokio::test]
async fn test_e2b_sandbox_timeout_handling() {
    let client = MockE2BClient::new("test-api-key");

    let config = SandboxConfig {
        timeout: 1, // 1 second timeout
        ..Default::default()
    };

    let sandbox = client.create_sandbox(config).await.unwrap();

    // Execute long-running command
    let result = client.execute_command(
        &sandbox.sandbox_id,
        "sleep 10",
        Some(500) // 500ms timeout
    ).await;

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("timeout"));
}

#[tokio::test]
async fn test_e2b_sandbox_command_failure() {
    let client = MockE2BClient::new("test-api-key");
    let sandbox = client.create_sandbox(SandboxConfig::default()).await.unwrap();

    let result = client.execute_command(
        &sandbox.sandbox_id,
        "nonexistent-command",
        None
    ).await;

    assert!(result.is_ok()); // Command executes, but fails
    let output = result.unwrap();
    assert_ne!(output.exit_code, 0);
    assert!(!output.stderr.is_empty());
}

// ============================================================================
// Unit Tests - OpenRouter Client
// ============================================================================

#[tokio::test]
async fn test_openrouter_client_initialization() {
    let client = MockOpenRouterClient::new("test-api-key");
    assert!(client.is_initialized());
}

#[tokio::test]
async fn test_openrouter_simple_completion() {
    let client = MockOpenRouterClient::new("test-api-key");

    let messages = vec![
        ChatMessage {
            role: "user".to_string(),
            content: "Hello, how are you?".to_string(),
        }
    ];

    let result = client.create_completion(messages, None).await;
    assert!(result.is_ok());

    let response = result.unwrap();
    assert!(!response.content.is_empty());
    assert!(response.tokens_used > 0);
    assert!(response.cost_usd > 0.0);
}

#[tokio::test]
async fn test_openrouter_streaming_completion() {
    let client = MockOpenRouterClient::new("test-api-key");

    let messages = vec![
        ChatMessage {
            role: "user".to_string(),
            content: "Write a short story".to_string(),
        }
    ];

    let mut stream = client.create_streaming_completion(messages, None).await.unwrap();

    let mut chunks = Vec::new();
    while let Some(chunk) = stream.next().await {
        chunks.push(chunk);
    }

    assert!(!chunks.is_empty());

    // Verify final chunk has usage info
    let final_chunk = chunks.last().unwrap();
    assert!(final_chunk.tokens_used.is_some());
}

#[tokio::test]
async fn test_openrouter_model_selection() {
    let client = MockOpenRouterClient::new("test-api-key");

    let config = CompletionConfig {
        model: "anthropic/claude-3-sonnet".to_string(),
        max_tokens: 1000,
        temperature: 0.7,
        ..Default::default()
    };

    let messages = vec![
        ChatMessage {
            role: "user".to_string(),
            content: "Test message".to_string(),
        }
    ];

    let result = client.create_completion(messages, Some(config)).await;
    assert!(result.is_ok());

    let response = result.unwrap();
    assert_eq!(response.model, "anthropic/claude-3-sonnet");
}

#[tokio::test]
async fn test_openrouter_token_limit() {
    let client = MockOpenRouterClient::new("test-api-key");

    let config = CompletionConfig {
        max_tokens: 10,
        ..Default::default()
    };

    let messages = vec![
        ChatMessage {
            role: "user".to_string(),
            content: "Write a very long response about artificial intelligence".to_string(),
        }
    ];

    let result = client.create_completion(messages, Some(config)).await;
    assert!(result.is_ok());

    let response = result.unwrap();
    assert!(response.tokens_used <= 10);
}

#[tokio::test]
async fn test_openrouter_cost_tracking() {
    let client = MockOpenRouterClient::new("test-api-key");

    let messages = vec![
        ChatMessage {
            role: "user".to_string(),
            content: "Test".to_string(),
        }
    ];

    let response = client.create_completion(messages, None).await.unwrap();

    assert!(response.cost_usd > 0.0);
    assert!(response.tokens_used > 0);

    // Cost should be reasonable (not absurdly high for simple test)
    assert!(response.cost_usd < 1.0);
}

// ============================================================================
// Unit Tests - Agent Deployment Logic
// ============================================================================

#[tokio::test]
async fn test_agent_deployment_basic() {
    let deployer = MockAgentDeployer::new();

    let agent_config = AgentConfig {
        agent_type: AgentType::Researcher,
        name: "research-agent-1".to_string(),
        capabilities: vec!["web_search".to_string(), "data_analysis".to_string()],
        llm_model: "anthropic/claude-3-sonnet".to_string(),
        max_tokens: 4000,
        ..Default::default()
    };

    let result = deployer.deploy_agent(agent_config).await;
    assert!(result.is_ok());

    let deployment = result.unwrap();
    assert!(!deployment.agent_id.is_empty());
    assert!(!deployment.sandbox_id.is_empty());
    assert_eq!(deployment.status, DeploymentStatus::Running);
}

#[tokio::test]
async fn test_agent_deployment_with_custom_template() {
    let deployer = MockAgentDeployer::new();

    let config = AgentConfig {
        agent_type: AgentType::Coder,
        name: "coder-agent-1".to_string(),
        sandbox_template: "nodejs-typescript".to_string(),
        capabilities: vec!["code_generation".to_string()],
        ..Default::default()
    };

    let deployment = deployer.deploy_agent(config).await.unwrap();
    assert_eq!(deployment.sandbox_template, "nodejs-typescript");
}

#[tokio::test]
async fn test_agent_deployment_environment_setup() {
    let deployer = MockAgentDeployer::new();

    let config = AgentConfig {
        agent_type: AgentType::Analyst,
        name: "analyst-1".to_string(),
        env_vars: vec![
            ("DATABASE_URL".to_string(), "postgresql://localhost/test".to_string()),
        ],
        ..Default::default()
    };

    let deployment = deployer.deploy_agent(config).await.unwrap();

    // Verify env vars were passed through
    assert!(deployment.metadata.get("env_configured").is_some());
}

#[tokio::test]
async fn test_agent_status_tracking() {
    let deployer = MockAgentDeployer::new();

    let config = AgentConfig {
        agent_type: AgentType::Coordinator,
        name: "coordinator-1".to_string(),
        ..Default::default()
    };

    let deployment = deployer.deploy_agent(config).await.unwrap();
    let agent_id = deployment.agent_id.clone();

    // Get initial status
    let status = deployer.get_agent_status(&agent_id).await.unwrap();
    assert_eq!(status.status, DeploymentStatus::Running);

    // Simulate some work
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Check status again
    let updated_status = deployer.get_agent_status(&agent_id).await.unwrap();
    assert!(updated_status.uptime_seconds > 0);
}

#[tokio::test]
async fn test_agent_termination() {
    let deployer = MockAgentDeployer::new();

    let config = AgentConfig {
        agent_type: AgentType::Tester,
        name: "tester-1".to_string(),
        ..Default::default()
    };

    let deployment = deployer.deploy_agent(config).await.unwrap();
    let agent_id = deployment.agent_id.clone();

    // Terminate agent
    let result = deployer.terminate_agent(&agent_id).await;
    assert!(result.is_ok());

    // Verify status is terminated
    let status = deployer.get_agent_status(&agent_id).await.unwrap();
    assert_eq!(status.status, DeploymentStatus::Terminated);
}

// ============================================================================
// Unit Tests - Database Operations
// ============================================================================

#[tokio::test]
async fn test_db_store_agent_deployment() {
    let db = MockAgentDatabase::new();

    let deployment = AgentDeployment {
        agent_id: "agent-123".to_string(),
        sandbox_id: "sandbox-456".to_string(),
        agent_type: AgentType::Researcher,
        name: "test-agent".to_string(),
        status: DeploymentStatus::Running,
        created_at: chrono::Utc::now(),
        ..Default::default()
    };

    let result = db.store_deployment(&deployment).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_db_retrieve_agent_deployment() {
    let db = MockAgentDatabase::new();

    let deployment = create_sample_deployment();
    db.store_deployment(&deployment).await.unwrap();

    // Retrieve by agent_id
    let retrieved = db.get_deployment(&deployment.agent_id).await.unwrap();
    assert_eq!(retrieved.agent_id, deployment.agent_id);
    assert_eq!(retrieved.name, deployment.name);
}

#[tokio::test]
async fn test_db_list_active_deployments() {
    let db = MockAgentDatabase::new();

    // Create multiple deployments
    for i in 0..5 {
        let mut deployment = create_sample_deployment();
        deployment.agent_id = format!("agent-{}", i);
        deployment.status = if i < 3 {
            DeploymentStatus::Running
        } else {
            DeploymentStatus::Terminated
        };

        db.store_deployment(&deployment).await.unwrap();
    }

    let active = db.list_active_deployments().await.unwrap();
    assert_eq!(active.len(), 3);
}

#[tokio::test]
async fn test_db_update_deployment_status() {
    let db = MockAgentDatabase::new();

    let deployment = create_sample_deployment();
    db.store_deployment(&deployment).await.unwrap();

    // Update status
    let result = db.update_deployment_status(
        &deployment.agent_id,
        DeploymentStatus::Stopped,
        None
    ).await;
    assert!(result.is_ok());

    // Verify update
    let updated = db.get_deployment(&deployment.agent_id).await.unwrap();
    assert_eq!(updated.status, DeploymentStatus::Stopped);
}

#[tokio::test]
async fn test_db_track_agent_metrics() {
    let db = MockAgentDatabase::new();

    let deployment = create_sample_deployment();
    db.store_deployment(&deployment).await.unwrap();

    // Store metrics
    let metrics = AgentMetrics {
        agent_id: deployment.agent_id.clone(),
        cpu_usage: 45.5,
        memory_mb: 512,
        requests_processed: 100,
        tokens_used: 50000,
        cost_usd: 0.25,
        timestamp: chrono::Utc::now(),
    };

    let result = db.store_metrics(&metrics).await;
    assert!(result.is_ok());

    // Retrieve metrics
    let retrieved = db.get_agent_metrics(&deployment.agent_id, 10).await.unwrap();
    assert_eq!(retrieved.len(), 1);
    assert_eq!(retrieved[0].requests_processed, 100);
}

// ============================================================================
// Unit Tests - Error Handling
// ============================================================================

#[tokio::test]
async fn test_invalid_api_key() {
    let client = MockE2BClient::new("");

    let result = client.create_sandbox(SandboxConfig::default()).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("invalid") ||
            result.unwrap_err().to_string().contains("unauthorized"));
}

#[tokio::test]
async fn test_invalid_sandbox_id() {
    let client = MockE2BClient::new("test-api-key");

    let result = client.get_sandbox_status("nonexistent-sandbox").await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not found"));
}

#[tokio::test]
async fn test_deployment_validation() {
    let deployer = MockAgentDeployer::new();

    // Invalid config - empty name
    let invalid_config = AgentConfig {
        agent_type: AgentType::Researcher,
        name: "".to_string(),
        ..Default::default()
    };

    let result = deployer.deploy_agent(invalid_config).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_resource_limit_exceeded() {
    let deployer = MockAgentDeployer::new();

    // Try to deploy agents exceeding limit
    let mut deployments = Vec::new();

    for i in 0..MockAgentDeployer::MAX_CONCURRENT_AGENTS + 1 {
        let config = AgentConfig {
            agent_type: AgentType::Coder,
            name: format!("agent-{}", i),
            ..Default::default()
        };

        let result = deployer.deploy_agent(config).await;

        if i < MockAgentDeployer::MAX_CONCURRENT_AGENTS {
            assert!(result.is_ok());
            deployments.push(result.unwrap());
        } else {
            // Should fail on exceeding limit
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains("limit"));
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn create_sample_deployment() -> AgentDeployment {
    AgentDeployment {
        agent_id: "test-agent-123".to_string(),
        sandbox_id: "sandbox-456".to_string(),
        agent_type: AgentType::Researcher,
        name: "test-researcher".to_string(),
        status: DeploymentStatus::Running,
        sandbox_template: "nodejs".to_string(),
        llm_model: "anthropic/claude-3-sonnet".to_string(),
        capabilities: vec!["research".to_string(), "analysis".to_string()],
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        terminated_at: None,
        metadata: serde_json::json!({}),
        env_vars: Vec::new(),
    }
}
