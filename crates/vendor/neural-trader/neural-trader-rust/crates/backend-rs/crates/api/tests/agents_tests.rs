use beclever_api::agents::*;
use beclever_api::e2b_client::*;
use beclever_api::openrouter_client::*;

#[cfg(test)]
mod e2b_client_tests {
    use super::*;

    #[test]
    fn test_e2b_client_creation() {
        let client = E2BClient::new("test_key".to_string());
        assert_eq!(client.api_key, "test_key");
    }

    #[test]
    fn test_sandbox_config_serialization() {
        use std::collections::HashMap;

        let config = SandboxConfig {
            template: "base".to_string(),
            timeout: 3600,
            env_vars: Some(HashMap::from([
                ("API_KEY".to_string(), "test".to_string()),
            ])),
            metadata: None,
        };

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("base"));
        assert!(json.contains("3600"));
    }

    #[test]
    fn test_execution_request_defaults() {
        let req = ExecutionRequest {
            code: "echo 'Hello World'".to_string(),
            language: None,
            timeout: None,
            env_vars: None,
        };

        assert_eq!(req.code, "echo 'Hello World'");
        assert!(req.language.is_none());
    }
}

#[cfg(test)]
mod openrouter_client_tests {
    use super::*;

    #[test]
    fn test_openrouter_client_creation() {
        let client = OpenRouterClient::new("test_key".to_string());
        assert_eq!(client.api_key, "test_key");
    }

    #[test]
    fn test_cost_calculation_claude() {
        let client = OpenRouterClient::new("test_key".to_string());
        let usage = Usage {
            prompt_tokens: 1000,
            completion_tokens: 500,
            total_tokens: 1500,
        };

        let cost = client.calculate_cost(&usage, "claude-3.5-sonnet");
        assert!(cost > 0.0);
        assert!(cost < 1.0); // Should be under $1 for 1500 tokens
    }

    #[test]
    fn test_cost_calculation_gpt4() {
        let client = OpenRouterClient::new("test_key".to_string());
        let usage = Usage {
            prompt_tokens: 1000,
            completion_tokens: 500,
            total_tokens: 1500,
        };

        let cost_claude = client.calculate_cost(&usage, "claude-3.5-sonnet");
        let cost_gpt4 = client.calculate_cost(&usage, "gpt-4");

        // GPT-4 should be more expensive than Claude
        assert!(cost_gpt4 > cost_claude);
    }

    #[test]
    fn test_system_prompts() {
        let client = OpenRouterClient::new("test_key".to_string());

        let researcher_prompt = client.get_agent_system_prompt("researcher");
        assert!(researcher_prompt.contains("research"));

        let coder_prompt = client.get_agent_system_prompt("coder");
        assert!(coder_prompt.contains("code"));

        let tester_prompt = client.get_agent_system_prompt("tester");
        assert!(tester_prompt.contains("test"));

        let reviewer_prompt = client.get_agent_system_prompt("reviewer");
        assert!(reviewer_prompt.contains("review"));
    }

    #[test]
    fn test_message_serialization() {
        let message = Message {
            role: "user".to_string(),
            content: "Hello".to_string(),
        };

        let json = serde_json::to_string(&message).unwrap();
        assert!(json.contains("user"));
        assert!(json.contains("Hello"));
    }

    #[test]
    fn test_chat_request_serialization() {
        let request = ChatRequest {
            model: "claude-3.5-sonnet".to_string(),
            messages: vec![
                Message {
                    role: "user".to_string(),
                    content: "Test".to_string(),
                },
            ],
            temperature: Some(0.7),
            max_tokens: Some(1000),
            stream: None,
            top_p: Some(0.9),
            frequency_penalty: None,
            presence_penalty: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("claude-3.5-sonnet"));
        assert!(json.contains("0.7"));
    }
}

#[cfg(test)]
mod agent_deployment_tests {
    use super::*;

    #[test]
    fn test_deploy_agent_request_serialization() {
        use std::collections::HashMap;

        let request = DeployAgentRequest {
            name: "Test Agent".to_string(),
            agent_type: "researcher".to_string(),
            task_description: "Analyze data".to_string(),
            template: "base".to_string(),
            model: "claude-3.5-sonnet".to_string(),
            capabilities: vec!["research".to_string(), "analysis".to_string()],
            environment: HashMap::new(),
            config: serde_json::json!({}),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("Test Agent"));
        assert!(json.contains("researcher"));
    }

    #[test]
    fn test_deploy_swarm_request_serialization() {
        let request = DeploySwarmRequest {
            name: "Test Swarm".to_string(),
            topology: "mesh".to_string(),
            strategy: "balanced".to_string(),
            max_agents: 5,
            agents: vec![
                SwarmAgent {
                    name: "Agent 1".to_string(),
                    agent_type: "researcher".to_string(),
                    task_description: "Research".to_string(),
                    capabilities: vec![],
                },
            ],
            task_description: "Complete project".to_string(),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("Test Swarm"));
        assert!(json.contains("mesh"));
    }

    #[test]
    fn test_agent_status_serialization() {
        let status = AgentStatus {
            id: "123".to_string(),
            name: "Test Agent".to_string(),
            agent_type: "coder".to_string(),
            status: "running".to_string(),
            sandbox_id: Some("sandbox-123".to_string()),
            task_description: "Write code".to_string(),
            result: None,
            error_message: None,
            tokens_used: 1000,
            cost_usd: 0.05,
            execution_time_ms: 5000,
            created_at: "2025-01-01T00:00:00Z".to_string(),
            completed_at: None,
        };

        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("Test Agent"));
        assert!(json.contains("running"));
        assert!(json.contains("sandbox-123"));
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_agent_service_creation() {
        use beclever_api::db::Database;

        let db = Database::new(":memory:".to_string()).unwrap();
        let service = AgentService::new(
            db,
            "test_e2b_key".to_string(),
            "test_openrouter_key".to_string(),
        );

        // Service should be created successfully
        assert!(service.active_agents.read().await.is_empty());
    }

    #[test]
    fn test_log_stream_query_defaults() {
        let query = LogStreamQuery {
            lines: 50,
            level: Some("info".to_string()),
        };

        assert_eq!(query.lines, 50);
        assert_eq!(query.level.unwrap(), "info");
    }

    #[test]
    fn test_execute_command_request() {
        use std::collections::HashMap;

        let request = ExecuteCommandRequest {
            command: "ls -la".to_string(),
            env_vars: HashMap::new(),
            timeout: Some(30),
        };

        assert_eq!(request.command, "ls -la");
        assert_eq!(request.timeout.unwrap(), 30);
    }
}

#[cfg(test)]
mod error_handling_tests {
    use super::*;

    #[test]
    fn test_app_error_types() {
        let db_error = AppError::DatabaseError("DB failed".to_string());
        let not_found = AppError::NotFound("Not found".to_string());
        let bad_request = AppError::BadRequest("Bad data".to_string());
        let internal = AppError::InternalError("Internal".to_string());

        // Errors should be created successfully
        assert!(matches!(db_error, AppError::DatabaseError(_)));
        assert!(matches!(not_found, AppError::NotFound(_)));
        assert!(matches!(bad_request, AppError::BadRequest(_)));
        assert!(matches!(internal, AppError::InternalError(_)));
    }
}
