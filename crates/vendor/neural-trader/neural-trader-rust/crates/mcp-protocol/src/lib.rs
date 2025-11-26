//! MCP (Model Context Protocol) JSON-RPC 2.0 Protocol Types
//!
//! This crate provides the core protocol types for the Model Context Protocol,
//! implementing JSON-RPC 2.0 as specified by Anthropic.
//!
//! # Features
//! - Type-safe JSON-RPC 2.0 requests and responses
//! - Standard error codes
//! - Async-ready traits
//! - Zero-copy where possible

mod types;
mod error;

pub use types::*;
pub use error::*;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_request_serialization() {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: "list_strategies".to_string(),
            params: Some(json!({})),
            id: Some(RequestId::String("req-1".to_string())),
        };

        let serialized = serde_json::to_string(&request).unwrap();
        assert!(serialized.contains("\"jsonrpc\":\"2.0\""));
        assert!(serialized.contains("\"method\":\"list_strategies\""));
    }

    #[test]
    fn test_response_serialization() {
        let response = JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            result: Some(json!({"status": "ok"})),
            error: None,
            id: Some(RequestId::Number(1)),
        };

        let serialized = serde_json::to_string(&response).unwrap();
        assert!(serialized.contains("\"jsonrpc\":\"2.0\""));
        assert!(serialized.contains("\"result\""));
    }

    #[test]
    fn test_error_response() {
        let error = JsonRpcError {
            code: ErrorCode::MethodNotFound.as_i32(),
            message: "Method not found".to_string(),
            data: None,
        };

        let response = JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            result: None,
            error: Some(error),
            id: Some(RequestId::String("req-1".to_string())),
        };

        let serialized = serde_json::to_string(&response).unwrap();
        assert!(serialized.contains("\"error\""));
        assert!(serialized.contains("-32601"));
    }
}
