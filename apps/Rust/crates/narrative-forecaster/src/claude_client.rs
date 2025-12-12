use crate::llm_client::{LLMClient, LLMConfig};
use crate::NarrativeError;
use async_trait::async_trait;
use serde_json::json;

pub struct ClaudeClient {
    config: LLMConfig,
    client: reqwest::Client,
}

impl ClaudeClient {
    pub fn new(config: LLMConfig) -> Result<Self, NarrativeError> {
        if config.api_key.is_none() {
            return Err(NarrativeError::ConfigError("Claude API key is required".to_string()));
        }
        
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_seconds))
            .build()
            .map_err(|e| NarrativeError::NetworkError(e))?;
        
        Ok(Self { config, client })
    }
}

#[async_trait]
impl LLMClient for ClaudeClient {
    async fn generate_response(&self, prompt: &str) -> Result<String, NarrativeError> {
        let url = self.config.base_url.as_ref()
            .ok_or_else(|| NarrativeError::ConfigError("Base URL not configured".to_string()))?;
        
        let api_key = self.config.api_key.as_ref()
            .ok_or_else(|| NarrativeError::ConfigError("API key not configured".to_string()))?;
        
        let payload = json!({
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "system": "You are an expert financial analyst specializing in cryptocurrency markets. You provide detailed, analytical insights using advanced reasoning to predict market movements based on technical analysis, sentiment, and market dynamics. Your responses should be thorough, well-reasoned, and actionable.",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        });
        
        let response = self.client
            .post(url)
            .header("Content-Type", "application/json")
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&payload)
            .send()
            .await
            .map_err(|e| NarrativeError::LLMError(format!("Request failed: {}", e)))?;
        
        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(NarrativeError::LLMError(format!(
                "Claude API error: {} - {}",
                response.status(),
                error_text
            )));
        }
        
        let response_data: serde_json::Value = response.json().await
            .map_err(|e| NarrativeError::LLMError(format!("Failed to parse response: {}", e)))?;
        
        // Extract content from Claude API response format
        let content = response_data["content"][0]["text"]
            .as_str()
            .ok_or_else(|| NarrativeError::LLMError("Invalid response format".to_string()))?;
        
        Ok(content.to_string())
    }
    
    fn provider_name(&self) -> String {
        "Claude".to_string()
    }
    
    fn model_name(&self) -> String {
        self.config.model.clone()
    }
}