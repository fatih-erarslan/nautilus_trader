//! STDIO transport for MCP server
//!
//! This transport enables `npx neural-trader mcp start` to communicate
//! via standard input/output, the primary MCP transport method.

use tokio::io::{self, AsyncBufReadExt, AsyncWriteExt, BufReader};
use neural_trader_mcp_protocol::{JsonRpcRequest, JsonRpcResponse, ProtocolError};
use tracing::{info, error, debug};

/// STDIO transport handler
pub struct StdioTransport;

impl StdioTransport {
    /// Create a new STDIO transport
    pub fn new() -> Self {
        Self
    }

    /// Start the STDIO transport server
    pub async fn run(&self) -> Result<(), ProtocolError> {
        info!("Starting MCP STDIO transport");

        let stdin = io::stdin();
        let stdout = io::stdout();
        let mut reader = BufReader::new(stdin);
        let mut writer = stdout;
        let mut line = String::new();

        loop {
            line.clear();
            match reader.read_line(&mut line).await {
                Ok(0) => {
                    debug!("EOF reached, shutting down");
                    break;
                }
                Ok(_) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    debug!("Received request: {}", trimmed);

                    // Parse JSON-RPC request
                    match serde_json::from_str::<JsonRpcRequest>(trimmed) {
                        Ok(request) => {
                            // Handle request
                            let response = self.handle_request(request).await;

                            // Send response
                            match serde_json::to_string(&response) {
                                Ok(json) => {
                                    if let Err(e) = writer.write_all(json.as_bytes()).await {
                                        error!("Failed to write response: {}", e);
                                        break;
                                    }
                                    if let Err(e) = writer.write_all(b"\n").await {
                                        error!("Failed to write newline: {}", e);
                                        break;
                                    }
                                    if let Err(e) = writer.flush().await {
                                        error!("Failed to flush output: {}", e);
                                        break;
                                    }
                                }
                                Err(e) => {
                                    error!("Failed to serialize response: {}", e);
                                }
                            }
                        }
                        Err(e) => {
                            error!("Failed to parse request: {}", e);
                            // Send parse error response
                            let error_response = JsonRpcResponse::error(
                                neural_trader_mcp_protocol::JsonRpcError {
                                    code: -32700,
                                    message: format!("Parse error: {}", e),
                                    data: None,
                                },
                                None,
                            );
                            if let Ok(json) = serde_json::to_string(&error_response) {
                                let _ = writer.write_all(json.as_bytes()).await;
                                let _ = writer.write_all(b"\n").await;
                                let _ = writer.flush().await;
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to read from stdin: {}", e);
                    break;
                }
            }
        }

        info!("MCP STDIO transport shut down");
        Ok(())
    }

    /// Handle a JSON-RPC request
    async fn handle_request(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        // Call handler
        match crate::handlers::tools::handle_tool_call(&request).await {
            Ok(result) => JsonRpcResponse::success(result, request.id),
            Err(error) => JsonRpcResponse::error(error.to_json_rpc_error(), request.id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transport_creation() {
        let transport = StdioTransport::new();
        // Basic smoke test
        assert!(std::mem::size_of_val(&transport) == 0); // Zero-sized type
    }
}
