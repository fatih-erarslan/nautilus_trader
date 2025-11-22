use anyhow::Result;
use reqwest::{Client, ClientBuilder, Response};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    time::Duration,
};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, error, info, warn};
use url::Url;
use uuid::Uuid;

use crate::ml::nhits::api::models::*;

/// NHITS API Client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientConfig {
    /// Base URL of the NHITS API server
    pub base_url: String,
    /// Authentication token
    pub auth_token: Option<String>,
    /// Request timeout in seconds
    pub timeout: u64,
    /// Maximum number of retry attempts
    pub max_retries: u32,
    /// Retry delay in milliseconds
    pub retry_delay: u64,
    /// Enable request/response logging
    pub enable_logging: bool,
    /// User agent string
    pub user_agent: String,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:8080/api/v1".to_string(),
            auth_token: None,
            timeout: 30,
            max_retries: 3,
            retry_delay: 1000,
            enable_logging: false,
            user_agent: format!("nhits-api-client/{}", env!("CARGO_PKG_VERSION")),
        }
    }
}

/// NHITS API Client
#[derive(Clone)]
pub struct NHITSClient {
    config: ClientConfig,
    client: Client,
}

impl NHITSClient {
    /// Create a new NHITS API client
    pub fn new(config: ClientConfig) -> Result<Self> {
        let mut builder = ClientBuilder::new()
            .timeout(Duration::from_secs(config.timeout))
            .user_agent(&config.user_agent);

        // Add default headers
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("Content-Type", "application/json".parse()?);
        
        if let Some(token) = &config.auth_token {
            headers.insert(
                "Authorization",
                format!("Bearer {}", token).parse()?,
            );
        }
        
        builder = builder.default_headers(headers);
        
        let client = builder.build()?;
        
        Ok(Self { config, client })
    }

    /// Create a client with default configuration
    pub fn with_base_url(base_url: &str) -> Result<Self> {
        let config = ClientConfig {
            base_url: base_url.to_string(),
            ..Default::default()
        };
        Self::new(config)
    }

    /// Create a client with authentication token
    pub fn with_auth(base_url: &str, token: &str) -> Result<Self> {
        let config = ClientConfig {
            base_url: base_url.to_string(),
            auth_token: Some(token.to_string()),
            ..Default::default()
        };
        Self::new(config)
    }

    /// Check API health
    pub async fn health_check(&self) -> Result<HealthResponse> {
        let url = format!("{}/health", self.config.base_url);
        let response = self.get(&url).await?;
        
        let health: ApiResponse<HealthResponse> = response.json().await?;
        
        if health.success {
            Ok(health.data.unwrap())
        } else {
            Err(anyhow::anyhow!("Health check failed: {:?}", health.error))
        }
    }

    /// List all models
    pub async fn list_models(&self, query: Option<ListModelsQuery>) -> Result<PaginatedResponse<ModelMetadata>> {
        let mut url = format!("{}/models", self.config.base_url);
        
        if let Some(q) = query {
            let params = self.build_query_params(&q)?;
            if !params.is_empty() {
                url.push('?');
                url.push_str(&params);
            }
        }
        
        let response = self.get(&url).await?;
        let result: ApiResponse<PaginatedResponse<ModelMetadata>> = response.json().await?;
        
        if result.success {
            Ok(result.data.unwrap())
        } else {
            Err(anyhow::anyhow!("Failed to list models: {:?}", result.error))
        }
    }

    /// Get a specific model
    pub async fn get_model(&self, model_id: &str) -> Result<ModelMetadata> {
        let url = format!("{}/models/{}", self.config.base_url, model_id);
        let response = self.get(&url).await?;
        
        let result: ApiResponse<ModelMetadata> = response.json().await?;
        
        if result.success {
            Ok(result.data.unwrap())
        } else {
            Err(anyhow::anyhow!("Failed to get model: {:?}", result.error))
        }
    }

    /// Create a new model
    pub async fn create_model(&self, request: CreateModelRequest) -> Result<ModelMetadata> {
        let url = format!("{}/models", self.config.base_url);
        let response = self.post(&url, &request).await?;
        
        let result: ApiResponse<ModelMetadata> = response.json().await?;
        
        if result.success {
            Ok(result.data.unwrap())
        } else {
            Err(anyhow::anyhow!("Failed to create model: {:?}", result.error))
        }
    }

    /// Update an existing model
    pub async fn update_model(&self, model_id: &str, request: UpdateModelRequest) -> Result<ModelMetadata> {
        let url = format!("{}/models/{}", self.config.base_url, model_id);
        let response = self.put(&url, &request).await?;
        
        let result: ApiResponse<ModelMetadata> = response.json().await?;
        
        if result.success {
            Ok(result.data.unwrap())
        } else {
            Err(anyhow::anyhow!("Failed to update model: {:?}", result.error))
        }
    }

    /// Delete a model
    pub async fn delete_model(&self, model_id: &str) -> Result<()> {
        let url = format!("{}/models/{}", self.config.base_url, model_id);
        let response = self.delete(&url).await?;
        
        let result: ApiResponse<serde_json::Value> = response.json().await?;
        
        if result.success {
            Ok(())
        } else {
            Err(anyhow::anyhow!("Failed to delete model: {:?}", result.error))
        }
    }

    /// Train a model
    pub async fn train_model(&self, model_id: &str, request: TrainModelRequest) -> Result<serde_json::Value> {
        let url = format!("{}/models/{}/train", self.config.base_url, model_id);
        let response = self.post(&url, &request).await?;
        
        let result: ApiResponse<serde_json::Value> = response.json().await?;
        
        if result.success {
            Ok(result.data.unwrap())
        } else {
            Err(anyhow::anyhow!("Failed to train model: {:?}", result.error))
        }
    }

    /// Create a forecast
    pub async fn create_forecast(&self, model_id: &str, request: ForecastRequest) -> Result<ForecastJob> {
        let url = format!("{}/models/{}/forecast", self.config.base_url, model_id);
        let response = self.post(&url, &request).await?;
        
        let result: ApiResponse<ForecastJob> = response.json().await?;
        
        if result.success {
            Ok(result.data.unwrap())
        } else {
            Err(anyhow::anyhow!("Failed to create forecast: {:?}", result.error))
        }
    }

    /// List forecast jobs
    pub async fn list_forecasts(&self, query: Option<ListJobsQuery>) -> Result<PaginatedResponse<ForecastJob>> {
        let mut url = format!("{}/forecasts", self.config.base_url);
        
        if let Some(q) = query {
            let params = self.build_jobs_query_params(&q)?;
            if !params.is_empty() {
                url.push('?');
                url.push_str(&params);
            }
        }
        
        let response = self.get(&url).await?;
        let result: ApiResponse<PaginatedResponse<ForecastJob>> = response.json().await?;
        
        if result.success {
            Ok(result.data.unwrap())
        } else {
            Err(anyhow::anyhow!("Failed to list forecasts: {:?}", result.error))
        }
    }

    /// Get a specific forecast job
    pub async fn get_forecast(&self, job_id: Uuid) -> Result<ForecastJob> {
        let url = format!("{}/forecasts/{}", self.config.base_url, job_id);
        let response = self.get(&url).await?;
        
        let result: ApiResponse<ForecastJob> = response.json().await?;
        
        if result.success {
            Ok(result.data.unwrap())
        } else {
            Err(anyhow::anyhow!("Failed to get forecast: {:?}", result.error))
        }
    }

    /// Cancel a forecast job
    pub async fn cancel_forecast(&self, job_id: Uuid) -> Result<()> {
        let url = format!("{}/forecasts/{}/cancel", self.config.base_url, job_id);
        let response = self.post(&url, &serde_json::Value::Null).await?;
        
        let result: ApiResponse<serde_json::Value> = response.json().await?;
        
        if result.success {
            Ok(())
        } else {
            Err(anyhow::anyhow!("Failed to cancel forecast: {:?}", result.error))
        }
    }

    /// Wait for forecast completion
    pub async fn wait_for_forecast(&self, job_id: Uuid, poll_interval: Duration) -> Result<ForecastJob> {
        loop {
            let job = self.get_forecast(job_id).await?;
            
            match job.status {
                JobStatus::Completed => return Ok(job),
                JobStatus::Failed => return Err(anyhow::anyhow!("Forecast failed: {:?}", job.error)),
                JobStatus::Cancelled => return Err(anyhow::anyhow!("Forecast was cancelled")),
                _ => {
                    tokio::time::sleep(poll_interval).await;
                }
            }
        }
    }

    /// Create and wait for forecast completion
    pub async fn forecast_and_wait(
        &self,
        model_id: &str,
        request: ForecastRequest,
        poll_interval: Option<Duration>,
    ) -> Result<ForecastJob> {
        let job = self.create_forecast(model_id, request).await?;
        let interval = poll_interval.unwrap_or(Duration::from_secs(1));
        self.wait_for_forecast(job.id, interval).await
    }

    /// Connect to WebSocket for real-time updates
    pub async fn connect_websocket(&self, params: Option<HashMap<String, String>>) -> Result<NHITSWebSocketClient> {
        let mut ws_url = self.config.base_url.replace("http://", "ws://").replace("https://", "wss://");
        ws_url.push_str("/ws");
        
        if let Some(p) = params {
            let query: Vec<String> = p.iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect();
            if !query.is_empty() {
                ws_url.push('?');
                ws_url.push_str(&query.join("&"));
            }
        }
        
        let url = Url::parse(&ws_url)?;
        let (ws_stream, _) = connect_async(url).await?;
        
        Ok(NHITSWebSocketClient::new(ws_stream))
    }

    // HTTP helper methods
    
    async fn get(&self, url: &str) -> Result<Response> {
        self.request_with_retry(|| self.client.get(url)).await
    }

    async fn post<T: Serialize>(&self, url: &str, body: &T) -> Result<Response> {
        self.request_with_retry(|| self.client.post(url).json(body)).await
    }

    async fn put<T: Serialize>(&self, url: &str, body: &T) -> Result<Response> {
        self.request_with_retry(|| self.client.put(url).json(body)).await
    }

    async fn delete(&self, url: &str) -> Result<Response> {
        self.request_with_retry(|| self.client.delete(url)).await
    }

    async fn request_with_retry<F, Fut>(&self, request_fn: F) -> Result<Response>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<Response, reqwest::Error>>,
    {
        for attempt in 0..=self.config.max_retries {
            match request_fn().await {
                Ok(response) => {
                    if self.config.enable_logging {
                        debug!("Request successful: {} {}", response.status(), response.url());
                    }
                    return Ok(response);
                }
                Err(e) => {
                    if attempt == self.config.max_retries {
                        error!("Request failed after {} attempts: {}", self.config.max_retries + 1, e);
                        return Err(e.into());
                    }
                    
                    warn!("Request failed (attempt {}/{}): {}", attempt + 1, self.config.max_retries + 1, e);
                    tokio::time::sleep(Duration::from_millis(self.config.retry_delay)).await;
                }
            }
        }
        
        unreachable!()
    }

    fn build_query_params(&self, query: &ListModelsQuery) -> Result<String> {
        let mut params = Vec::new();
        
        if let Some(page) = query.page {
            params.push(format!("page={}", page));
        }
        if let Some(page_size) = query.page_size {
            params.push(format!("page_size={}", page_size));
        }
        if let Some(status) = &query.status {
            params.push(format!("status={}", status));
        }
        if let Some(search) = &query.search {
            params.push(format!("search={}", urlencoding::encode(search)));
        }
        if let Some(sort_by) = &query.sort_by {
            params.push(format!("sort_by={}", sort_by));
        }
        if let Some(sort_order) = &query.sort_order {
            params.push(format!("sort_order={}", sort_order));
        }
        
        Ok(params.join("&"))
    }

    fn build_jobs_query_params(&self, query: &ListJobsQuery) -> Result<String> {
        let mut params = Vec::new();
        
        if let Some(page) = query.page {
            params.push(format!("page={}", page));
        }
        if let Some(page_size) = query.page_size {
            params.push(format!("page_size={}", page_size));
        }
        if let Some(status) = &query.status {
            params.push(format!("status={}", status));
        }
        if let Some(model_id) = &query.model_id {
            params.push(format!("model_id={}", model_id));
        }
        if let Some(priority) = &query.priority {
            params.push(format!("priority={}", priority));
        }
        
        Ok(params.join("&"))
    }
}

/// WebSocket client for real-time updates
pub struct NHITSWebSocketClient {
    stream: tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
}

impl NHITSWebSocketClient {
    fn new(stream: tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>) -> Self {
        Self { stream }
    }

    /// Listen for WebSocket messages
    pub async fn listen<F>(&mut self, mut handler: F) -> Result<()>
    where
        F: FnMut(crate::ml::nhits::api::websocket::WsMessage) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>,
    {
        use futures_util::{SinkExt, StreamExt};
        
        while let Some(msg) = self.stream.next().await {
            match msg? {
                Message::Text(text) => {
                    match serde_json::from_str::<crate::ml::nhits::api::websocket::WsMessage>(&text) {
                        Ok(ws_msg) => {
                            handler(ws_msg).await;
                        }
                        Err(e) => {
                            warn!("Failed to parse WebSocket message: {}", e);
                        }
                    }
                }
                Message::Close(_) => {
                    info!("WebSocket connection closed");
                    break;
                }
                _ => {}
            }
        }
        
        Ok(())
    }

    /// Send a message to the WebSocket
    pub async fn send(&mut self, message: &crate::ml::nhits::api::websocket::WsMessage) -> Result<()> {
        use futures_util::SinkExt;
        
        let text = serde_json::to_string(message)?;
        self.stream.send(Message::Text(text)).await?;
        Ok(())
    }

    /// Close the WebSocket connection
    pub async fn close(&mut self) -> Result<()> {
        use futures_util::SinkExt;
        self.stream.close(None).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_config_default() {
        let config = ClientConfig::default();
        assert_eq!(config.base_url, "http://localhost:8080/api/v1");
        assert_eq!(config.timeout, 30);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_client_creation() {
        let config = ClientConfig::default();
        let client = NHITSClient::new(config);
        assert!(client.is_ok());
    }

    #[test]
    fn test_client_with_auth() {
        let client = NHITSClient::with_auth("http://localhost:8080/api/v1", "test-token");
        assert!(client.is_ok());
    }

    #[tokio::test]
    async fn test_query_params_building() {
        let config = ClientConfig::default();
        let client = NHITSClient::new(config).unwrap();
        
        let query = ListModelsQuery {
            page: Some(2),
            page_size: Some(50),
            status: Some("trained".to_string()),
            search: Some("test model".to_string()),
            ..Default::default()
        };
        
        let params = client.build_query_params(&query).unwrap();
        assert!(params.contains("page=2"));
        assert!(params.contains("page_size=50"));
        assert!(params.contains("status=trained"));
        assert!(params.contains("search=test%20model"));
    }
}