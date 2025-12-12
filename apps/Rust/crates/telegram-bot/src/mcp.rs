use serde::{Deserialize, Serialize};
use anyhow::Result;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPCommand {
    pub name: String,
    pub category: String,
    pub description: String,
    pub requires_params: bool,
    pub gpu_accelerated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPExecutionResult {
    pub command: String,
    pub success: bool,
    pub data: serde_json::Value,
    pub execution_time: u64,
    pub gpu_used: bool,
}

pub struct MCPService {
    base_url: String,
    client: reqwest::Client,
    commands: HashMap<String, MCPCommand>,
}

impl MCPService {
    pub async fn new() -> Self {
        let mut service = Self {
            base_url: "http://localhost:8001".to_string(),
            client: reqwest::Client::new(),
            commands: HashMap::new(),
        };
        
        service.initialize_commands();
        service
    }
    
    fn initialize_commands(&mut self) {
        let commands = vec![
            // Core Trading Commands
            MCPCommand {
                name: "ping".to_string(),
                category: "core".to_string(),
                description: "Test server connectivity".to_string(),
                requires_params: false,
                gpu_accelerated: false,
            },
            MCPCommand {
                name: "list_strategies".to_string(),
                category: "core".to_string(),
                description: "List all trading strategies".to_string(),
                requires_params: false,
                gpu_accelerated: true,
            },
            MCPCommand {
                name: "quick_analysis".to_string(),
                category: "core".to_string(),
                description: "Quick market analysis for symbol".to_string(),
                requires_params: true,
                gpu_accelerated: true,
            },
            MCPCommand {
                name: "get_portfolio_status".to_string(),
                category: "core".to_string(),
                description: "Get current portfolio status".to_string(),
                requires_params: false,
                gpu_accelerated: false,
            },
            
            // Neural AI Commands
            MCPCommand {
                name: "neural_forecast".to_string(),
                category: "neural".to_string(),
                description: "Generate neural network forecasts".to_string(),
                requires_params: true,
                gpu_accelerated: true,
            },
            MCPCommand {
                name: "neural_train".to_string(),
                category: "neural".to_string(),
                description: "Train neural forecasting model".to_string(),
                requires_params: true,
                gpu_accelerated: true,
            },
            MCPCommand {
                name: "neural_evaluate".to_string(),
                category: "neural".to_string(),
                description: "Evaluate neural model performance".to_string(),
                requires_params: true,
                gpu_accelerated: true,
            },
            
            // Sentiment Analysis
            MCPCommand {
                name: "analyze_news".to_string(),
                category: "sentiment".to_string(),
                description: "AI sentiment analysis of market news".to_string(),
                requires_params: true,
                gpu_accelerated: true,
            },
            MCPCommand {
                name: "get_news_sentiment".to_string(),
                category: "sentiment".to_string(),
                description: "Real-time news sentiment".to_string(),
                requires_params: true,
                gpu_accelerated: false,
            },
            
            // Trading Operations
            MCPCommand {
                name: "run_backtest".to_string(),
                category: "trading".to_string(),
                description: "Run comprehensive backtest".to_string(),
                requires_params: true,
                gpu_accelerated: true,
            },
            MCPCommand {
                name: "optimize_strategy".to_string(),
                category: "trading".to_string(),
                description: "Optimize strategy parameters".to_string(),
                requires_params: true,
                gpu_accelerated: true,
            },
            MCPCommand {
                name: "risk_analysis".to_string(),
                category: "trading".to_string(),
                description: "Portfolio risk analysis".to_string(),
                requires_params: true,
                gpu_accelerated: true,
            },
            
            // System Commands
            MCPCommand {
                name: "get_system_metrics".to_string(),
                category: "system".to_string(),
                description: "Get system performance metrics".to_string(),
                requires_params: false,
                gpu_accelerated: false,
            },
        ];
        
        for command in commands {
            self.commands.insert(command.name.clone(), command);
        }
    }
    
    pub async fn execute_command(&self, command_name: &str) -> Result<MCPExecutionResult> {
        let url = format!("{}/api/mcp/execute", self.base_url);
        
        let payload = serde_json::json!({
            "command": command_name
        });
        
        let start_time = std::time::Instant::now();
        
        let response = self.client.post(&url)
            .json(&payload)
            .send()
            .await?;
        
        let execution_time = start_time.elapsed().as_millis() as u64;
        
        if response.status().is_success() {
            let data: serde_json::Value = response.json().await?;
            
            let success = data["success"].as_bool().unwrap_or(false);
            let result_data = if success {
                data["data"].clone()
            } else {
                serde_json::json!({"error": data["error"]})
            };
            
            let gpu_used = self.commands.get(command_name)
                .map(|cmd| cmd.gpu_accelerated)
                .unwrap_or(false);
            
            Ok(MCPExecutionResult {
                command: command_name.to_string(),
                success,
                data: result_data,
                execution_time,
                gpu_used,
            })
        } else {
            Err(anyhow::anyhow!("MCP command failed: {}", response.status()))
        }
    }
    
    pub async fn execute_command_with_params(&self, command_name: &str, params: serde_json::Value) -> Result<MCPExecutionResult> {
        let url = format!("{}/api/mcp/execute", self.base_url);
        
        let payload = serde_json::json!({
            "command": command_name,
            "parameters": params
        });
        
        let start_time = std::time::Instant::now();
        
        let response = self.client.post(&url)
            .json(&payload)
            .send()
            .await?;
        
        let execution_time = start_time.elapsed().as_millis() as u64;
        
        if response.status().is_success() {
            let data: serde_json::Value = response.json().await?;
            
            let success = data["success"].as_bool().unwrap_or(false);
            let result_data = if success {
                data["data"].clone()
            } else {
                serde_json::json!({"error": data["error"]})
            };
            
            let gpu_used = self.commands.get(command_name)
                .map(|cmd| cmd.gpu_accelerated)
                .unwrap_or(false);
            
            Ok(MCPExecutionResult {
                command: command_name.to_string(),
                success,
                data: result_data,
                execution_time,
                gpu_used,
            })
        } else {
            Err(anyhow::anyhow!("MCP command failed: {}", response.status()))
        }
    }
    
    pub fn get_available_commands(&self) -> Vec<&MCPCommand> {
        self.commands.values().collect()
    }
    
    pub fn get_commands_by_category(&self, category: &str) -> Vec<&MCPCommand> {
        self.commands.values()
            .filter(|cmd| cmd.category == category)
            .collect()
    }
    
    pub fn get_command_help(&self, command_name: &str) -> Option<String> {
        self.commands.get(command_name).map(|cmd| {
            format!(
                "**{}**\n\
                Category: {}\n\
                Description: {}\n\
                Requires Parameters: {}\n\
                GPU Accelerated: {}",
                cmd.name, cmd.category, cmd.description,
                if cmd.requires_params { "Yes" } else { "No" },
                if cmd.gpu_accelerated { "Yes" } else { "No" }
            )
        })
    }
    
    pub async fn get_command_statistics(&self) -> Result<CommandStatistics> {
        Ok(CommandStatistics {
            total_commands: self.commands.len() as u32,
            gpu_accelerated: self.commands.values()
                .filter(|cmd| cmd.gpu_accelerated)
                .count() as u32,
            categories: self.get_category_stats(),
        })
    }
    
    fn get_category_stats(&self) -> HashMap<String, u32> {
        let mut stats = HashMap::new();
        
        for command in self.commands.values() {
            *stats.entry(command.category.clone()).or_insert(0) += 1;
        }
        
        stats
    }
    
    // Telegram-specific command parsing
    pub fn parse_telegram_command(&self, command_text: &str) -> Result<(String, Option<serde_json::Value>)> {
        let parts: Vec<&str> = command_text.split_whitespace().collect();
        
        if parts.is_empty() {
            return Err(anyhow::anyhow!("Empty command"));
        }
        
        let command_name = parts[0];
        
        // Check if command exists
        if !self.commands.contains_key(command_name) {
            return Err(anyhow::anyhow!("Unknown command: {}", command_name));
        }
        
        let command = &self.commands[command_name];
        
        if command.requires_params && parts.len() == 1 {
            return Err(anyhow::anyhow!("Command '{}' requires parameters", command_name));
        }
        
        // Simple parameter parsing for common commands
        let params = if parts.len() > 1 {
            match command_name {
                "neural_forecast" | "quick_analysis" | "analyze_news" => {
                    Some(serde_json::json!({
                        "symbol": parts[1]
                    }))
                },
                "neural_train" => {
                    if parts.len() >= 3 {
                        Some(serde_json::json!({
                            "model": parts[1],
                            "symbol": parts[2]
                        }))
                    } else {
                        Some(serde_json::json!({
                            "model": "lstm",
                            "symbol": parts[1]
                        }))
                    }
                },
                "run_backtest" => {
                    if parts.len() >= 3 {
                        Some(serde_json::json!({
                            "strategy": parts[1],
                            "period": parts[2]
                        }))
                    } else {
                        Some(serde_json::json!({
                            "strategy": parts[1],
                            "period": "30d"
                        }))
                    }
                },
                _ => {
                    // Generic parameter parsing
                    let param_str = parts[1..].join(" ");
                    Some(serde_json::json!({"query": param_str}))
                }
            }
        } else {
            None
        };
        
        Ok((command_name.to_string(), params))
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CommandStatistics {
    pub total_commands: u32,
    pub gpu_accelerated: u32,
    pub categories: HashMap<String, u32>,
}