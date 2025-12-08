use crate::NarrativeError;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LLMProvider {
    Claude,
    OpenAI,
    Ollama,
    LMStudio,
}

#[derive(Debug, Clone)]
pub struct LLMConfig {
    pub provider: LLMProvider,
    pub api_key: Option<String>,
    pub model: String,
    pub base_url: Option<String>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub timeout_seconds: u64,
}

#[async_trait]
pub trait LLMClient {
    async fn generate_response(&self, prompt: &str) -> Result<String, NarrativeError>;
    fn provider_name(&self) -> String;
    fn model_name(&self) -> String;
}

pub struct OpenAIClient {
    config: LLMConfig,
    client: reqwest::Client,
}

impl OpenAIClient {
    pub fn new(config: LLMConfig) -> Result<Self, NarrativeError> {
        if config.api_key.is_none() {
            return Err(NarrativeError::ConfigError("OpenAI API key is required".to_string()));
        }
        
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_seconds))
            .build()
            .map_err(|e| NarrativeError::NetworkError(e))?;
        
        Ok(Self { config, client })
    }
}

#[async_trait]
impl LLMClient for OpenAIClient {
    async fn generate_response(&self, prompt: &str) -> Result<String, NarrativeError> {
        let url = self.config.base_url.as_ref()
            .unwrap_or(&"https://api.openai.com/v1/chat/completions".to_string());
        
        let api_key = self.config.api_key.as_ref()
            .ok_or_else(|| NarrativeError::ConfigError("API key not configured".to_string()))?;
        
        let payload = json!({
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        });
        
        let response = self.client
            .post(url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&payload)
            .send()
            .await
            .map_err(|e| NarrativeError::LLMError(format!("Request failed: {}", e)))?;
        
        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(NarrativeError::LLMError(format!(
                "OpenAI API error: {} - {}",
                response.status(),
                error_text
            )));
        }
        
        let response_data: serde_json::Value = response.json().await
            .map_err(|e| NarrativeError::LLMError(format!("Failed to parse response: {}", e)))?;
        
        let content = response_data["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| NarrativeError::LLMError("Invalid response format".to_string()))?;
        
        Ok(content.to_string())
    }
    
    fn provider_name(&self) -> String {
        "OpenAI".to_string()
    }
    
    fn model_name(&self) -> String {
        self.config.model.clone()
    }
}

pub struct OllamaClient {
    config: LLMConfig,
    client: reqwest::Client,
}

impl OllamaClient {
    pub fn new(config: LLMConfig) -> Result<Self, NarrativeError> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_seconds))
            .build()
            .map_err(|e| NarrativeError::NetworkError(e))?;
        
        Ok(Self { config, client })
    }
}

#[async_trait]
impl LLMClient for OllamaClient {
    async fn generate_response(&self, prompt: &str) -> Result<String, NarrativeError> {
        let url = self.config.base_url.as_ref()
            .unwrap_or(&"http://localhost:11434/api/chat".to_string());
        
        let payload = json!({
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": false
        });
        
        let response = self.client
            .post(url)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|e| NarrativeError::LLMError(format!("Request failed: {}", e)))?;
        
        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(NarrativeError::LLMError(format!(
                "Ollama API error: {} - {}",
                response.status(),
                error_text
            )));
        }
        
        let response_data: serde_json::Value = response.json().await
            .map_err(|e| NarrativeError::LLMError(format!("Failed to parse response: {}", e)))?;
        
        let content = response_data["message"]["content"]
            .as_str()
            .ok_or_else(|| NarrativeError::LLMError("Invalid response format".to_string()))?;
        
        Ok(content.to_string())
    }
    
    fn provider_name(&self) -> String {
        "Ollama".to_string()
    }
    
    fn model_name(&self) -> String {
        self.config.model.clone()
    }
}

pub struct LMStudioClient {
    config: LLMConfig,
    client: reqwest::Client,
}

impl LMStudioClient {
    pub fn new(config: LLMConfig) -> Result<Self, NarrativeError> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_seconds))
            .build()
            .map_err(|e| NarrativeError::NetworkError(e))?;
        
        Ok(Self { config, client })
    }
}

#[async_trait]
impl LLMClient for LMStudioClient {
    async fn generate_response(&self, prompt: &str) -> Result<String, NarrativeError> {
        let url = self.config.base_url.as_ref()
            .unwrap_or(&"http://localhost:1234/v1/chat/completions".to_string());
        
        let payload = json!({
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        });
        
        let response = self.client
            .post(url)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|e| NarrativeError::LLMError(format!("Request failed: {}", e)))?;
        
        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(NarrativeError::LLMError(format!(
                "LMStudio API error: {} - {}",
                response.status(),
                error_text
            )));
        }
        
        let response_data: serde_json::Value = response.json().await
            .map_err(|e| NarrativeError::LLMError(format!("Failed to parse response: {}", e)))?;
        
        let content = response_data["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| NarrativeError::LLMError("Invalid response format".to_string()))?;
        
        Ok(content.to_string())
    }
    
    fn provider_name(&self) -> String {
        "LMStudio".to_string()
    }
    
    fn model_name(&self) -> String {
        self.config.model.clone()
    }
}