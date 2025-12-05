//! Secure transport wrapper for MCP endpoints
//!
//! This module provides a secure wrapper that can be placed around any existing
//! MCP server implementation to add authentication without modifying the underlying code.

use crate::{
    McpAuthGuard, McpAuthError, McpAuthResult,
    McpRequest, SignedMcpRequest, McpResponse,
};
use std::sync::Arc;

/// Secure MCP transport that wraps existing MCP handlers
pub struct SecureMcpTransport<H> {
    /// Authentication guard
    guard: Arc<McpAuthGuard>,

    /// Underlying MCP handler
    handler: H,

    /// Whether to sign responses
    sign_responses: bool,
}

impl<H> SecureMcpTransport<H>
where
    H: McpHandler,
{
    /// Create new secure transport
    pub fn new(guard: Arc<McpAuthGuard>, handler: H) -> Self {
        Self {
            guard,
            handler,
            sign_responses: true,
        }
    }

    /// Create transport without response signing (faster)
    pub fn without_response_signing(guard: Arc<McpAuthGuard>, handler: H) -> Self {
        Self {
            guard,
            handler,
            sign_responses: false,
        }
    }

    /// Process a signed request through the secure transport
    pub async fn process(&self, signed_request: SignedMcpRequest) -> McpResponse {
        let request_id = signed_request.request.id.clone();

        // Verify and authorize
        if let Err(e) = self.guard.verify_and_authorize(&signed_request) {
            return self.create_error_response(&request_id, e);
        }

        // Execute the underlying handler
        let response = self.handler.handle(signed_request.request).await;

        // Optionally sign the response
        if self.sign_responses {
            match self.guard.sign_response(response) {
                Ok(signed) => signed,
                Err(e) => self.create_error_response(&request_id, e),
            }
        } else {
            response
        }
    }

    /// Process raw JSON request (for STDIO transport compatibility)
    pub async fn process_json(&self, json: &str) -> String {
        // Parse the signed request
        let signed_request: Result<SignedMcpRequest, _> = serde_json::from_str(json);

        let response = match signed_request {
            Ok(req) => self.process(req).await,
            Err(e) => McpResponse::error(
                "unknown",
                -32700,
                format!("Parse error: {}", e),
            ),
        };

        serde_json::to_string(&response).unwrap_or_else(|_| {
            r#"{"jsonrpc":"2.0","id":null,"error":{"code":-32603,"message":"Internal error"}}"#.to_string()
        })
    }

    /// Create error response from McpAuthError
    fn create_error_response(&self, request_id: &str, error: McpAuthError) -> McpResponse {
        match error {
            McpAuthError::InvalidSignature => {
                McpResponse::error(request_id, -32001, "Invalid signature")
            }
            McpAuthError::ClientNotFound { client_id } => {
                McpResponse::error(request_id, -32002, format!("Unknown client: {}", client_id))
            }
            McpAuthError::ClientNotAuthorized { client_id } => {
                McpResponse::error(request_id, -32003, format!("Client not authorized: {}", client_id))
            }
            McpAuthError::RequestExpired { .. } => {
                McpResponse::error(request_id, -32004, "Request expired")
            }
            McpAuthError::NonceReused { .. } => {
                McpResponse::error(request_id, -32005, "Replay attack detected")
            }
            McpAuthError::RateLimitExceeded { .. } => {
                McpResponse::rate_limited(request_id)
            }
            McpAuthError::ToolNotPermitted { tool, .. } => {
                McpResponse::error(request_id, -32007, format!("Tool not permitted: {}", tool))
            }
            McpAuthError::InjectionDetected { pattern } => {
                McpResponse::error(request_id, -32008, format!("Security violation: {}", pattern))
            }
            _ => McpResponse::error(request_id, -32600, error.to_string()),
        }
    }

    /// Get reference to the guard
    pub fn guard(&self) -> &Arc<McpAuthGuard> {
        &self.guard
    }

    /// Get reference to the handler
    pub fn handler(&self) -> &H {
        &self.handler
    }
}

/// Trait for MCP request handlers
#[async_trait::async_trait]
pub trait McpHandler: Send + Sync {
    /// Handle an MCP request and return a response
    async fn handle(&self, request: McpRequest) -> McpResponse;
}

/// Simple function-based MCP handler
pub struct FnHandler<F>
where
    F: Fn(McpRequest) -> McpResponse + Send + Sync,
{
    handler: F,
}

impl<F> FnHandler<F>
where
    F: Fn(McpRequest) -> McpResponse + Send + Sync,
{
    pub fn new(handler: F) -> Self {
        Self { handler }
    }
}

#[async_trait::async_trait]
impl<F> McpHandler for FnHandler<F>
where
    F: Fn(McpRequest) -> McpResponse + Send + Sync,
{
    async fn handle(&self, request: McpRequest) -> McpResponse {
        (self.handler)(request)
    }
}

/// Async function-based MCP handler
pub struct AsyncFnHandler<F, Fut>
where
    F: Fn(McpRequest) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = McpResponse> + Send + Sync,
{
    handler: F,
    _phantom: std::marker::PhantomData<fn() -> Fut>,
}

impl<F, Fut> AsyncFnHandler<F, Fut>
where
    F: Fn(McpRequest) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = McpResponse> + Send + Sync,
{
    pub fn new(handler: F) -> Self {
        Self {
            handler,
            _phantom: std::marker::PhantomData,
        }
    }
}

#[async_trait::async_trait]
impl<F, Fut> McpHandler for AsyncFnHandler<F, Fut>
where
    F: Fn(McpRequest) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = McpResponse> + Send + Sync,
{
    async fn handle(&self, request: McpRequest) -> McpResponse {
        (self.handler)(request).await
    }
}

/// Router for dispatching requests to different tool handlers
pub struct McpRouter {
    /// Tool handlers by method name
    handlers: std::collections::HashMap<String, Box<dyn McpHandler>>,

    /// Default handler for unknown tools
    default_handler: Option<Box<dyn McpHandler>>,
}

impl McpRouter {
    /// Create new empty router
    pub fn new() -> Self {
        Self {
            handlers: std::collections::HashMap::new(),
            default_handler: None,
        }
    }

    /// Register a handler for a tool
    pub fn route(mut self, method: impl Into<String>, handler: impl McpHandler + 'static) -> Self {
        self.handlers.insert(method.into(), Box::new(handler));
        self
    }

    /// Set default handler for unknown tools
    pub fn default(mut self, handler: impl McpHandler + 'static) -> Self {
        self.default_handler = Some(Box::new(handler));
        self
    }
}

impl Default for McpRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl McpHandler for McpRouter {
    async fn handle(&self, request: McpRequest) -> McpResponse {
        let handler = self.handlers.get(&request.method)
            .or(self.default_handler.as_ref());

        match handler {
            Some(h) => h.handle(request).await,
            None => McpResponse::error(
                &request.id,
                -32601,
                format!("Method not found: {}", request.method),
            ),
        }
    }
}

/// Secure STDIO transport adapter for command-line MCP servers
pub struct SecureStdioAdapter {
    transport: Arc<SecureMcpTransport<McpRouter>>,
}

impl SecureStdioAdapter {
    /// Create new STDIO adapter
    pub fn new(guard: Arc<McpAuthGuard>, router: McpRouter) -> Self {
        Self {
            transport: Arc::new(SecureMcpTransport::new(guard, router)),
        }
    }

    /// Run the STDIO adapter (reads from stdin, writes to stdout)
    pub async fn run(&self) -> std::io::Result<()> {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let mut reader = BufReader::new(stdin);
        let mut line = String::new();

        loop {
            line.clear();
            let bytes_read = reader.read_line(&mut line).await?;

            if bytes_read == 0 {
                break; // EOF
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let response = self.transport.process_json(trimmed).await;
            stdout.write_all(response.as_bytes()).await?;
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;
        }

        Ok(())
    }
}

/// Builder for creating secure MCP transports
pub struct SecureTransportBuilder {
    config: crate::SecurityConfig,
    sign_responses: bool,
}

impl SecureTransportBuilder {
    /// Create new builder with default config
    pub fn new() -> Self {
        Self {
            config: crate::SecurityConfig::default(),
            sign_responses: true,
        }
    }

    /// Use maximum security configuration
    pub fn maximum_security(mut self) -> Self {
        self.config = crate::SecurityConfig::maximum_security();
        self
    }

    /// Disable response signing
    pub fn without_response_signing(mut self) -> Self {
        self.sign_responses = false;
        self
    }

    /// Set custom configuration
    pub fn with_config(mut self, config: crate::SecurityConfig) -> Self {
        self.config = config;
        self
    }

    /// Build the secure transport with a handler
    pub fn build<H: McpHandler>(self, handler: H) -> McpAuthResult<SecureMcpTransport<H>> {
        let guard = Arc::new(McpAuthGuard::new(self.config)?);

        Ok(if self.sign_responses {
            SecureMcpTransport::new(guard, handler)
        } else {
            SecureMcpTransport::without_response_signing(guard, handler)
        })
    }

    /// Build the secure transport with a router
    pub fn build_with_router(self, router: McpRouter) -> McpAuthResult<SecureMcpTransport<McpRouter>> {
        self.build(router)
    }
}

impl Default for SecureTransportBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoHandler;

    #[async_trait::async_trait]
    impl McpHandler for EchoHandler {
        async fn handle(&self, request: McpRequest) -> McpResponse {
            McpResponse::success(&request.id, request.params)
        }
    }

    #[tokio::test]
    async fn test_secure_transport() {
        let guard = Arc::new(McpAuthGuard::default_security().unwrap());
        let transport = SecureMcpTransport::new(guard.clone(), EchoHandler);

        // Register a client
        let client_id = guard.register_client("test").unwrap();

        // Create and sign a request
        let request = McpRequest::new("echo", serde_json::json!({"msg": "hello"}));
        let signed = guard.sign_request(&client_id, request).unwrap();

        // Process through secure transport
        let response = transport.process(signed).await;

        // Should succeed
        assert!(response.result.is_some());
        assert!(response.error.is_none());
    }

    #[tokio::test]
    async fn test_router() {
        let router = McpRouter::new()
            .route("echo", EchoHandler)
            .default(FnHandler::new(|req| {
                McpResponse::error(&req.id, -32601, "Not found")
            }));

        // Test known route
        let request = McpRequest::new("echo", serde_json::json!({}));
        let response = router.handle(request).await;
        assert!(response.result.is_some());

        // Test unknown route
        let request = McpRequest::new("unknown", serde_json::json!({}));
        let response = router.handle(request).await;
        assert!(response.error.is_some());
    }
}
