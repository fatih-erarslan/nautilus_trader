//! E2B API client

use crate::{
    error::{Error, Result},
    sandbox::{Sandbox, SandboxConfig},
    DEFAULT_API_URL,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// E2B API client
#[derive(Clone)]
pub struct E2BClient {
    inner: Arc<E2BClientInner>,
}

struct E2BClientInner {
    http: Client,
    api_key: String,
    base_url: String,
}

/// API response wrapper
#[derive(Debug, Deserialize)]
struct ApiResponse<T> {
    #[serde(flatten)]
    data: Option<T>,
    error: Option<ApiError>,
}

#[derive(Debug, Deserialize)]
struct ApiError {
    message: String,
    code: Option<String>,
}

/// Sandbox creation response
#[derive(Debug, Deserialize)]
struct CreateSandboxResponse {
    #[serde(rename = "sandboxId")]
    sandbox_id: String,
    #[serde(rename = "clientId")]
    client_id: String,
}

impl E2BClient {
    /// Create a new E2B client with the given API key
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        Self::with_base_url(api_key, DEFAULT_API_URL)
    }

    /// Create a new E2B client with custom base URL
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Result<Self> {
        let api_key = api_key.into();
        if api_key.is_empty() {
            return Err(Error::InvalidApiKey);
        }

        let http = Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()?;

        Ok(Self {
            inner: Arc::new(E2BClientInner {
                http,
                api_key,
                base_url: base_url.into(),
            }),
        })
    }

    /// Create a new sandbox
    pub async fn create_sandbox(&self, config: SandboxConfig) -> Result<Sandbox> {
        let url = format!("{}/sandboxes", self.inner.base_url);

        #[derive(Serialize)]
        struct CreateRequest {
            template: String,
            #[serde(skip_serializing_if = "Option::is_none")]
            timeout: Option<u64>,
            #[serde(skip_serializing_if = "Option::is_none")]
            metadata: Option<std::collections::HashMap<String, String>>,
        }

        let request = CreateRequest {
            template: config.template.clone(),
            timeout: config.timeout_ms,
            metadata: config.metadata.clone(),
        };

        let response = self
            .inner
            .http
            .post(&url)
            .header("X-E2B-API-Key", &self.inner.api_key)
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        let create_response: CreateSandboxResponse = response.json().await?;

        Ok(Sandbox::new(
            create_response.sandbox_id,
            create_response.client_id,
            config,
            self.clone(),
        ))
    }

    /// List active sandboxes
    pub async fn list_sandboxes(&self) -> Result<Vec<crate::SandboxMetadata>> {
        let url = format!("{}/sandboxes", self.inner.base_url);

        let response = self
            .inner
            .http
            .get(&url)
            .header("X-E2B-API-Key", &self.inner.api_key)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        let sandboxes: Vec<crate::SandboxMetadata> = response.json().await?;
        Ok(sandboxes)
    }

    /// Get a sandbox by ID
    pub async fn get_sandbox(&self, sandbox_id: &str) -> Result<Sandbox> {
        let url = format!("{}/sandboxes/{}", self.inner.base_url, sandbox_id);

        let response = self
            .inner
            .http
            .get(&url)
            .header("X-E2B-API-Key", &self.inner.api_key)
            .send()
            .await?;

        let status = response.status();
        if status.as_u16() == 404 {
            return Err(Error::SandboxNotFound(sandbox_id.to_string()));
        }
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        #[derive(Deserialize)]
        struct SandboxInfo {
            #[serde(rename = "sandboxId")]
            sandbox_id: String,
            #[serde(rename = "clientId")]
            client_id: String,
            template: String,
        }

        let info: SandboxInfo = response.json().await?;

        Ok(Sandbox::new(
            info.sandbox_id,
            info.client_id,
            SandboxConfig {
                template: info.template,
                ..Default::default()
            },
            self.clone(),
        ))
    }

    /// Internal: Get HTTP client
    pub(crate) fn http(&self) -> &Client {
        &self.inner.http
    }

    /// Internal: Get API key
    pub(crate) fn api_key(&self) -> &str {
        &self.inner.api_key
    }

    /// Internal: Get base URL
    pub(crate) fn base_url(&self) -> &str {
        &self.inner.base_url
    }
}
