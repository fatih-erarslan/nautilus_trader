use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use reqwest::Client;
use std::time::Duration;
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub struct OpenRouterClient {
    pub api_key: String,
    client: Client,
    base_url: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StreamChunk {
    pub id: String,
    pub model: String,
    pub choices: Vec<StreamChoice>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub pricing: Pricing,
    pub context_length: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Pricing {
    pub prompt: String,
    pub completion: String,
}

impl OpenRouterClient {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("Failed to build HTTP client"),
            base_url: "https://openrouter.ai/api/v1".to_string(),
        }
    }

    /// Send a chat completion request
    pub async fn chat_completion(&self, request: ChatRequest) -> Result<ChatResponse> {
        let url = format!("{}/chat/completions", self.base_url);

        let response = self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .header("HTTP-Referer", "https://foxrev.io")
            .header("X-Title", "BeClever AI Agent Deployment")
            .json(&request)
            .send()
            .await
            .context("Failed to send chat completion request")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Chat completion failed ({}): {}", status, error_text);
        }

        let chat_response: ChatResponse = response.json().await
            .context("Failed to parse chat response")?;

        Ok(chat_response)
    }

    /// Send a streaming chat completion request (simplified implementation)
    pub async fn chat_completion_stream(
        &self,
        mut request: ChatRequest,
    ) -> Result<mpsc::Receiver<Result<StreamChunk>>> {
        request.stream = Some(true);

        let url = format!("{}/chat/completions", self.base_url);

        let response = self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .header("HTTP-Referer", "https://foxrev.io")
            .header("X-Title", "BeClever AI Agent Deployment")
            .json(&request)
            .send()
            .await
            .context("Failed to send streaming request")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Streaming request failed ({}): {}", status, error_text);
        }

        let (tx, rx) = mpsc::channel(100);

        tokio::spawn(async move {
            // Use bytes() instead of bytes_stream() for simplicity
            match response.bytes().await {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);

                    for line in text.lines() {
                        if line.starts_with("data: ") {
                            let data = line.trim_start_matches("data: ");

                            if data == "[DONE]" {
                                break;
                            }

                            match serde_json::from_str::<StreamChunk>(data) {
                                Ok(chunk) => {
                                    if tx.send(Ok(chunk)).await.is_err() {
                                        break;
                                    }
                                }
                                Err(e) => {
                                    let _ = tx.send(Err(anyhow::anyhow!("Parse error: {}", e))).await;
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(anyhow::anyhow!("Stream error: {}", e))).await;
                }
            }
        });

        Ok(rx)
    }

    /// Get available models
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let url = format!("{}/models", self.base_url);

        let response = self.client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .context("Failed to list models")?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list models: {}", response.status());
        }

        #[derive(Deserialize)]
        struct ModelsResponse {
            data: Vec<ModelInfo>,
        }

        let models_response: ModelsResponse = response.json().await
            .context("Failed to parse models response")?;

        Ok(models_response.data)
    }

    /// Calculate cost based on usage
    pub fn calculate_cost(&self, usage: &Usage, model: &str) -> f64 {
        // Pricing per 1M tokens (approximate, should be fetched from API)
        let pricing = match model {
            m if m.contains("claude-3.5-sonnet") => (3.0, 15.0),
            m if m.contains("claude-3-sonnet") => (3.0, 15.0),
            m if m.contains("claude-3-haiku") => (0.25, 1.25),
            m if m.contains("gpt-4") => (30.0, 60.0),
            m if m.contains("gpt-3.5-turbo") => (0.5, 1.5),
            _ => (1.0, 2.0), // Default pricing
        };

        let prompt_cost = (usage.prompt_tokens as f64 / 1_000_000.0) * pricing.0;
        let completion_cost = (usage.completion_tokens as f64 / 1_000_000.0) * pricing.1;

        prompt_cost + completion_cost
    }

    /// Execute agent task with LLM
    pub async fn execute_agent_task(
        &self,
        agent_type: &str,
        task_description: &str,
        model: &str,
        context: Option<String>,
    ) -> Result<(String, Usage)> {
        let system_prompt = self.get_agent_system_prompt(agent_type);

        let mut messages = vec![
            Message {
                role: "system".to_string(),
                content: system_prompt,
            },
        ];

        if let Some(ctx) = context {
            messages.push(Message {
                role: "user".to_string(),
                content: format!("Context:\n{}\n\nTask: {}", ctx, task_description),
            });
        } else {
            messages.push(Message {
                role: "user".to_string(),
                content: task_description.to_string(),
            });
        }

        let request = ChatRequest {
            model: model.to_string(),
            messages,
            temperature: Some(0.7),
            max_tokens: Some(4096),
            stream: None,
            top_p: Some(0.9),
            frequency_penalty: None,
            presence_penalty: None,
        };

        let response = self.chat_completion(request).await?;

        let content = response.choices
            .first()
            .ok_or_else(|| anyhow::anyhow!("No response from model"))?
            .message
            .content
            .clone();

        Ok((content, response.usage))
    }

    /// Get system prompt based on agent type
    pub fn get_agent_system_prompt(&self, agent_type: &str) -> String {
        match agent_type {
            "researcher" => {
                "You are a research specialist agent. Analyze requirements, identify patterns, \
                and provide comprehensive research reports with citations and recommendations."
            }
            "coder" => {
                "You are a coding specialist agent. Write clean, efficient, well-documented code \
                following best practices. Include error handling and tests."
            }
            "tester" => {
                "You are a testing specialist agent. Create comprehensive test suites with unit, \
                integration, and edge case tests. Ensure high code coverage."
            }
            "reviewer" => {
                "You are a code review specialist agent. Analyze code quality, security, \
                performance, and best practices. Provide constructive feedback."
            }
            "analyst" => {
                "You are a data analysis specialist agent. Analyze data, identify trends, \
                and provide insights with visualizations and recommendations."
            }
            "optimizer" => {
                "You are an optimization specialist agent. Improve performance, reduce costs, \
                and enhance efficiency. Provide benchmarks and metrics."
            }
            "coordinator" => {
                "You are a coordination specialist agent. Orchestrate tasks across teams, \
                manage dependencies, and ensure smooth workflows."
            }
            _ => {
                "You are a helpful AI assistant. Complete the task efficiently and accurately."
            }
        }.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_creation() {
        let client = OpenRouterClient::new("test_key".to_string());
        assert_eq!(client.api_key, "test_key");
    }

    #[test]
    fn test_cost_calculation() {
        let client = OpenRouterClient::new("test_key".to_string());
        let usage = Usage {
            prompt_tokens: 1000,
            completion_tokens: 500,
            total_tokens: 1500,
        };

        let cost = client.calculate_cost(&usage, "claude-3.5-sonnet");
        assert!(cost > 0.0);
    }

    #[test]
    fn test_system_prompts() {
        let client = OpenRouterClient::new("test_key".to_string());

        let researcher_prompt = client.get_agent_system_prompt("researcher");
        assert!(researcher_prompt.contains("research"));

        let coder_prompt = client.get_agent_system_prompt("coder");
        assert!(coder_prompt.contains("code"));
    }
}
