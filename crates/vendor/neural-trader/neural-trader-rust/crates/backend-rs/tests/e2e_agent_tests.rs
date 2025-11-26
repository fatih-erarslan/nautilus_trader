//! End-to-End Tests for Agent Deployment System
//!
//! Tests complete user workflows:
//! 1. Deploy agent via API endpoints
//! 2. Execute commands and monitor status
//! 3. Stream logs in real-time
//! 4. Terminate cleanly with cleanup

use axum::http::StatusCode;
use serde_json::json;

mod fixtures;
use fixtures::*;

// ============================================================================
// E2E Tests - API Endpoint Workflows
// ============================================================================

#[tokio::test]
async fn test_e2e_deploy_agent_via_api() {
    let app = create_test_app().await;

    // Deploy agent via POST /api/agents/deploy
    let deploy_request = json!({
        "agent_type": "researcher",
        "name": "e2e-test-agent",
        "capabilities": ["research", "analysis"],
        "llm_model": "anthropic/claude-3-sonnet",
        "max_tokens": 4000
    });

    let response = app.post("/api/agents/deploy")
        .json(&deploy_request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);

    let body: serde_json::Value = response.json().await.unwrap();
    assert!(body["agent_id"].is_string());
    assert!(body["sandbox_id"].is_string());
    assert_eq!(body["status"], "running");

    let agent_id = body["agent_id"].as_str().unwrap();

    // Verify agent is listed
    let list_response = app.get("/api/agents")
        .send()
        .await
        .unwrap();

    assert_eq!(list_response.status(), StatusCode::OK);

    let agents: serde_json::Value = list_response.json().await.unwrap();
    assert!(agents["agents"].as_array().unwrap().iter()
        .any(|a| a["agent_id"] == agent_id));
}

#[tokio::test]
async fn test_e2e_execute_task_via_api() {
    let app = create_test_app().await;

    // Deploy agent
    let agent_id = deploy_test_agent(&app, "task-executor").await;

    // Execute task via POST /api/agents/{id}/tasks
    let task_request = json!({
        "task_type": "research",
        "prompt": "Research Rust async programming best practices",
        "max_tokens": 2000,
        "stream_response": false
    });

    let response = app.post(&format!("/api/agents/{}/tasks", agent_id))
        .json(&task_request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let result: serde_json::Value = response.json().await.unwrap();
    assert_eq!(result["success"], true);
    assert!(result["output"].is_string());
    assert!(result["tokens_used"].as_u64().unwrap() > 0);
    assert!(result["cost_usd"].as_f64().unwrap() > 0.0);
}

#[tokio::test]
async fn test_e2e_execute_code_task() {
    let app = create_test_app().await;

    let agent_id = deploy_test_agent(&app, "code-executor").await;

    // Execute code task
    let code_task = json!({
        "task_type": "code_execution",
        "prompt": "console.log('Hello from agent!');",
        "execute_code": true,
        "language": "javascript"
    });

    let response = app.post(&format!("/api/agents/{}/tasks", agent_id))
        .json(&code_task)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let result: serde_json::Value = response.json().await.unwrap();
    assert_eq!(result["success"], true);
    assert!(result["code_output"].is_object());

    let code_output = &result["code_output"];
    assert_eq!(code_output["exit_code"], 0);
    assert!(code_output["stdout"].as_str().unwrap().contains("Hello from agent!"));
}

#[tokio::test]
async fn test_e2e_agent_status_monitoring() {
    let app = create_test_app().await;

    let agent_id = deploy_test_agent(&app, "status-monitor").await;

    // Get agent status via GET /api/agents/{id}/status
    let response = app.get(&format!("/api/agents/{}/status", agent_id))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let status: serde_json::Value = response.json().await.unwrap();
    assert_eq!(status["agent_id"], agent_id);
    assert_eq!(status["status"], "running");
    assert!(status["uptime_seconds"].as_u64().is_some());
    assert!(status["tasks_completed"].as_u64().is_some());
}

#[tokio::test]
async fn test_e2e_stream_agent_logs() {
    let app = create_test_app().await;

    let agent_id = deploy_test_agent(&app, "log-streamer").await;

    // Execute task to generate logs
    let task = json!({
        "task_type": "test",
        "prompt": "Generate some log output",
        "execute_code": true
    });

    app.post(&format!("/api/agents/{}/tasks", agent_id))
        .json(&task)
        .send()
        .await
        .unwrap();

    // Stream logs via GET /api/agents/{id}/logs/stream
    let response = app.get(&format!("/api/agents/{}/logs/stream", agent_id))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Verify SSE stream
    let mut log_events = Vec::new();
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        let text = String::from_utf8_lossy(&chunk);

        if text.starts_with("data: ") {
            log_events.push(text.to_string());
        }

        if log_events.len() >= 5 {
            break;
        }
    }

    assert!(!log_events.is_empty());
}

#[tokio::test]
async fn test_e2e_agent_metrics() {
    let app = create_test_app().await;

    let agent_id = deploy_test_agent(&app, "metrics-test").await;

    // Execute several tasks
    for i in 0..5 {
        let task = json!({
            "task_type": "test",
            "prompt": format!("Test task {}", i)
        });

        app.post(&format!("/api/agents/{}/tasks", agent_id))
            .json(&task)
            .send()
            .await
            .unwrap();
    }

    // Get metrics via GET /api/agents/{id}/metrics
    let response = app.get(&format!("/api/agents/{}/metrics", agent_id))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let metrics: serde_json::Value = response.json().await.unwrap();
    assert!(metrics["requests_processed"].as_u64().unwrap() >= 5);
    assert!(metrics["total_tokens_used"].as_u64().unwrap() > 0);
    assert!(metrics["total_cost_usd"].as_f64().unwrap() > 0.0);
}

#[tokio::test]
async fn test_e2e_terminate_agent() {
    let app = create_test_app().await;

    let agent_id = deploy_test_agent(&app, "terminate-test").await;

    // Terminate via DELETE /api/agents/{id}
    let response = app.delete(&format!("/api/agents/{}", agent_id))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let result: serde_json::Value = response.json().await.unwrap();
    assert_eq!(result["message"], "Agent terminated successfully");

    // Verify agent is no longer listed as active
    let list_response = app.get("/api/agents?status=active")
        .send()
        .await
        .unwrap();

    let agents: serde_json::Value = list_response.json().await.unwrap();
    assert!(!agents["agents"].as_array().unwrap().iter()
        .any(|a| a["agent_id"] == agent_id));
}

// ============================================================================
// E2E Tests - Swarm Workflows
// ============================================================================

#[tokio::test]
async fn test_e2e_deploy_swarm_via_api() {
    let app = create_test_app().await;

    let swarm_request = json!({
        "topology": "mesh",
        "agent_count": 3,
        "agent_types": [
            {"type": "coordinator", "count": 1},
            {"type": "researcher", "count": 2}
        ]
    });

    let response = app.post("/api/swarms/deploy")
        .json(&swarm_request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);

    let swarm: serde_json::Value = response.json().await.unwrap();
    assert!(swarm["swarm_id"].is_string());
    assert_eq!(swarm["agents"].as_array().unwrap().len(), 3);
    assert_eq!(swarm["topology"], "mesh");
}

#[tokio::test]
async fn test_e2e_swarm_task_execution() {
    let app = create_test_app().await;

    let swarm_id = deploy_test_swarm(&app, 3).await;

    let swarm_task = json!({
        "description": "Research and analyze Rust ecosystem",
        "subtasks": [
            {
                "agent_type": "researcher",
                "prompt": "Research Rust crates ecosystem"
            },
            {
                "agent_type": "researcher",
                "prompt": "Analyze adoption trends"
            }
        ],
        "coordination_strategy": "parallel"
    });

    let response = app.post(&format!("/api/swarms/{}/tasks", swarm_id))
        .json(&swarm_task)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let result: serde_json::Value = response.json().await.unwrap();
    assert_eq!(result["success"], true);
    assert_eq!(result["subtask_results"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn test_e2e_swarm_status_monitoring() {
    let app = create_test_app().await;

    let swarm_id = deploy_test_swarm(&app, 4).await;

    let response = app.get(&format!("/api/swarms/{}/status", swarm_id))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let status: serde_json::Value = response.json().await.unwrap();
    assert_eq!(status["swarm_id"], swarm_id);
    assert_eq!(status["status"], "active");
    assert_eq!(status["active_agents"], 4);
}

#[tokio::test]
async fn test_e2e_swarm_scaling() {
    let app = create_test_app().await;

    let swarm_id = deploy_test_swarm(&app, 3).await;

    // Scale up to 5
    let scale_request = json!({"target_count": 5});

    let response = app.post(&format!("/api/swarms/{}/scale", swarm_id))
        .json(&scale_request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Verify new count
    let status_response = app.get(&format!("/api/swarms/{}/status", swarm_id))
        .send()
        .await
        .unwrap();

    let status: serde_json::Value = status_response.json().await.unwrap();
    assert_eq!(status["active_agents"], 5);
}

// ============================================================================
// E2E Tests - File Operations
// ============================================================================

#[tokio::test]
async fn test_e2e_file_upload_and_execution() {
    let app = create_test_app().await;

    let agent_id = deploy_test_agent(&app, "file-ops").await;

    // Upload file via PUT /api/agents/{id}/files
    let file_content = r#"
        function fibonacci(n) {
            if (n <= 1) return n;
            return fibonacci(n-1) + fibonacci(n-2);
        }
        console.log(fibonacci(10));
    "#;

    let upload_request = json!({
        "path": "/app/fib.js",
        "content": file_content
    });

    let response = app.put(&format!("/api/agents/{}/files", agent_id))
        .json(&upload_request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Execute the uploaded file
    let exec_task = json!({
        "task_type": "execute",
        "prompt": "node /app/fib.js",
        "execute_code": true
    });

    let exec_response = app.post(&format!("/api/agents/{}/tasks", agent_id))
        .json(&exec_task)
        .send()
        .await
        .unwrap();

    assert_eq!(exec_response.status(), StatusCode::OK);

    let result: serde_json::Value = exec_response.json().await.unwrap();
    assert_eq!(result["code_output"]["exit_code"], 0);
    assert!(result["code_output"]["stdout"].as_str().unwrap().contains("55"));
}

#[tokio::test]
async fn test_e2e_list_agent_files() {
    let app = create_test_app().await;

    let agent_id = deploy_test_agent(&app, "file-list").await;

    // Upload multiple files
    for i in 0..3 {
        let upload = json!({
            "path": format!("/app/file{}.txt", i),
            "content": format!("Content of file {}", i)
        });

        app.put(&format!("/api/agents/{}/files", agent_id))
            .json(&upload)
            .send()
            .await
            .unwrap();
    }

    // List files via GET /api/agents/{id}/files
    let response = app.get(&format!("/api/agents/{}/files?path=/app", agent_id))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let files: serde_json::Value = response.json().await.unwrap();
    assert_eq!(files["files"].as_array().unwrap().len(), 3);
}

// ============================================================================
// E2E Tests - Error Scenarios
// ============================================================================

#[tokio::test]
async fn test_e2e_deploy_with_invalid_config() {
    let app = create_test_app().await;

    let invalid_request = json!({
        "agent_type": "invalid_type",
        "name": ""
    });

    let response = app.post("/api/agents/deploy")
        .json(&invalid_request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let error: serde_json::Value = response.json().await.unwrap();
    assert!(error["error"].is_string());
}

#[tokio::test]
async fn test_e2e_task_execution_timeout() {
    let app = create_test_app().await;

    let agent_id = deploy_test_agent(&app, "timeout-test").await;

    let timeout_task = json!({
        "task_type": "execute",
        "prompt": "sleep 60", // Long-running command
        "execute_code": true,
        "timeout_ms": 1000 // 1 second timeout
    });

    let response = app.post(&format!("/api/agents/{}/tasks", agent_id))
        .json(&timeout_task)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
}

#[tokio::test]
async fn test_e2e_nonexistent_agent() {
    let app = create_test_app().await;

    let response = app.get("/api/agents/nonexistent-agent-id/status")
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_e2e_concurrent_task_execution() {
    let app = create_test_app().await;

    let agent_id = deploy_test_agent(&app, "concurrent").await;

    // Execute multiple tasks concurrently
    let mut handles = Vec::new();

    for i in 0..10 {
        let app_clone = app.clone();
        let agent_id_clone = agent_id.clone();

        let handle = tokio::spawn(async move {
            let task = json!({
                "task_type": "test",
                "prompt": format!("Concurrent task {}", i)
            });

            app_clone.post(&format!("/api/agents/{}/tasks", agent_id_clone))
                .json(&task)
                .send()
                .await
        });

        handles.push(handle);
    }

    // Wait for all tasks
    for handle in handles {
        let response = handle.await.unwrap().unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    // Verify metrics
    let metrics_response = app.get(&format!("/api/agents/{}/metrics", agent_id))
        .send()
        .await
        .unwrap();

    let metrics: serde_json::Value = metrics_response.json().await.unwrap();
    assert!(metrics["requests_processed"].as_u64().unwrap() >= 10);
}

// ============================================================================
// E2E Tests - Authentication & Authorization
// ============================================================================

#[tokio::test]
async fn test_e2e_unauthorized_access() {
    let app = create_test_app().await;

    // Try to deploy without auth token
    let request = json!({"agent_type": "researcher", "name": "test"});

    let response = app.post("/api/agents/deploy")
        .json(&request)
        .send_without_auth() // No Authorization header
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_e2e_resource_ownership() {
    let app = create_test_app().await;

    // User 1 deploys agent
    let agent_id = deploy_test_agent_for_user(&app, "ownership-test", "user-1").await;

    // User 2 tries to access it
    let response = app.get(&format!("/api/agents/{}/status", agent_id))
        .with_user("user-2")
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
}

// ============================================================================
// Helper Functions
// ============================================================================

async fn deploy_test_agent(app: &TestApp, name: &str) -> String {
    let request = json!({
        "agent_type": "researcher",
        "name": name,
        "capabilities": ["research"]
    });

    let response = app.post("/api/agents/deploy")
        .json(&request)
        .send()
        .await
        .unwrap();

    let body: serde_json::Value = response.json().await.unwrap();
    body["agent_id"].as_str().unwrap().to_string()
}

async fn deploy_test_agent_for_user(app: &TestApp, name: &str, user_id: &str) -> String {
    let request = json!({
        "agent_type": "researcher",
        "name": name
    });

    let response = app.post("/api/agents/deploy")
        .with_user(user_id)
        .json(&request)
        .send()
        .await
        .unwrap();

    let body: serde_json::Value = response.json().await.unwrap();
    body["agent_id"].as_str().unwrap().to_string()
}

async fn deploy_test_swarm(app: &TestApp, count: usize) -> String {
    let request = json!({
        "topology": "mesh",
        "agent_count": count
    });

    let response = app.post("/api/swarms/deploy")
        .json(&request)
        .send()
        .await
        .unwrap();

    let body: serde_json::Value = response.json().await.unwrap();
    body["swarm_id"].as_str().unwrap().to_string()
}
