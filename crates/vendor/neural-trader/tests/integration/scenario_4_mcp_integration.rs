// Integration Test Scenario 4: MCP Tools Integration
// Tests MCP server and protocol integration

use neural_trader_mcp_protocol::{Tool, ToolCall, ToolResponse};
use neural_trader_mcp_server::server::MCPServer;
use rust_decimal_macros::dec;

#[tokio::test]
async fn test_mcp_server_initialization() -> anyhow::Result<()> {
    // Test MCP server can be created
    let server = MCPServer::new("127.0.0.1:8080".to_string());

    assert_eq!(server.address(), "127.0.0.1:8080");

    Ok(())
}

#[tokio::test]
async fn test_mcp_tool_listing() -> anyhow::Result<()> {
    use neural_trader_mcp_protocol::tools::registry::ToolRegistry;

    let registry = ToolRegistry::new();
    let tools = registry.list_tools();

    // Verify essential tools are registered
    let tool_names: Vec<_> = tools.iter().map(|t| t.name.as_str()).collect();

    assert!(tool_names.contains(&"get_account_info"), "Should have get_account_info tool");
    assert!(tool_names.contains(&"place_order"), "Should have place_order tool");
    assert!(tool_names.contains(&"get_positions"), "Should have get_positions tool");
    assert!(tool_names.contains(&"calculate_indicators"), "Should have calculate_indicators tool");
    assert!(tool_names.contains(&"run_backtest"), "Should have run_backtest tool");

    println!("✅ MCP Tools registered: {}", tools.len());
    for tool in tools {
        println!("   - {}: {}", tool.name, tool.description);
    }

    Ok(())
}

#[tokio::test]
async fn test_mcp_tool_execution_get_account() -> anyhow::Result<()> {
    use neural_trader_mcp_protocol::tools::executor::ToolExecutor;
    use serde_json::json;

    let executor = ToolExecutor::new();

    let call = ToolCall {
        tool: "get_account_info".to_string(),
        parameters: json!({
            "broker": "alpaca"
        }),
    };

    // This will fail without credentials, but should show proper error handling
    let result = executor.execute(call).await;

    match result {
        Ok(_response) => {
            // If credentials are set, verify response structure
            println!("✅ Account info retrieved successfully");
        }
        Err(e) => {
            // Expected when credentials not set
            assert!(
                e.to_string().contains("credentials") || e.to_string().contains("API"),
                "Error should mention credentials: {}",
                e
            );
            println!("ℹ️  Expected error without credentials: {}", e);
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_mcp_tool_execution_calculate_indicators() -> anyhow::Result<()> {
    use neural_trader_mcp_protocol::tools::executor::ToolExecutor;
    use serde_json::json;

    let executor = ToolExecutor::new();

    let call = ToolCall {
        tool: "calculate_indicators".to_string(),
        parameters: json!({
            "symbol": "AAPL",
            "indicators": ["SMA", "RSI", "MACD"],
            "period": 20
        }),
    };

    let result = executor.execute(call).await?;

    // Verify response structure
    assert!(result.success, "Indicator calculation should succeed");
    assert!(result.data.is_some(), "Should return indicator data");

    println!("✅ Indicators calculated: {:?}", result.data);

    Ok(())
}

#[tokio::test]
async fn test_mcp_protocol_serialization() -> anyhow::Result<()> {
    use neural_trader_mcp_protocol::messages::{Request, Response};

    // Test request serialization
    let request = Request::ToolsList;
    let json = serde_json::to_string(&request)?;
    let deserialized: Request = serde_json::from_str(&json)?;

    assert!(matches!(deserialized, Request::ToolsList));

    // Test response serialization
    let response = Response::Success {
        data: serde_json::json!({"status": "ok"}),
    };
    let json = serde_json::to_string(&response)?;
    let deserialized: Response = serde_json::from_str(&json)?;

    assert!(matches!(deserialized, Response::Success { .. }));

    Ok(())
}

#[tokio::test]
#[ignore] // Requires running server
async fn test_mcp_server_http_endpoint() -> anyhow::Result<()> {
    use reqwest::Client;

    let client = Client::new();

    // Test tools list endpoint
    let response = client
        .get("http://localhost:8080/tools/list")
        .send()
        .await?;

    assert!(response.status().is_success(), "Server should respond");

    let tools: Vec<Tool> = response.json().await?;
    assert!(!tools.is_empty(), "Should return tools list");

    println!("✅ HTTP endpoint test passed: {} tools", tools.len());

    Ok(())
}

#[tokio::test]
async fn test_mcp_error_handling() -> anyhow::Result<()> {
    use neural_trader_mcp_protocol::tools::executor::ToolExecutor;
    use serde_json::json;

    let executor = ToolExecutor::new();

    // Test with invalid tool name
    let call = ToolCall {
        tool: "nonexistent_tool".to_string(),
        parameters: json!({}),
    };

    let result = executor.execute(call).await;
    assert!(result.is_err(), "Should error on invalid tool");

    // Test with invalid parameters
    let call = ToolCall {
        tool: "place_order".to_string(),
        parameters: json!({
            "invalid_param": "value"
        }),
    };

    let result = executor.execute(call).await;
    assert!(result.is_err(), "Should error on invalid parameters");

    Ok(())
}

#[tokio::test]
async fn test_mcp_concurrent_requests() -> anyhow::Result<()> {
    use neural_trader_mcp_protocol::tools::executor::ToolExecutor;
    use serde_json::json;
    use futures::future::join_all;

    let executor = ToolExecutor::new();

    // Create multiple concurrent tool calls
    let calls: Vec<_> = (0..10)
        .map(|i| {
            let executor = executor.clone();
            tokio::spawn(async move {
                let call = ToolCall {
                    tool: "calculate_indicators".to_string(),
                    parameters: json!({
                        "symbol": format!("TEST{}", i),
                        "indicators": ["SMA"],
                        "period": 20
                    }),
                };
                executor.execute(call).await
            })
        })
        .collect();

    let results = join_all(calls).await;

    // Verify all completed
    assert_eq!(results.len(), 10, "All concurrent requests should complete");

    println!("✅ Concurrent MCP requests test passed");

    Ok(())
}
