//! Error types and codes for MCP protocol

use thiserror::Error;

/// Standard JSON-RPC 2.0 error codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    /// Invalid JSON was received by the server
    ParseError = -32700,
    /// The JSON sent is not a valid Request object
    InvalidRequest = -32600,
    /// The method does not exist / is not available
    MethodNotFound = -32601,
    /// Invalid method parameter(s)
    InvalidParams = -32602,
    /// Internal JSON-RPC error
    InternalError = -32603,

    // MCP-specific error codes (custom range: -32000 to -32099)
    /// Model not found
    ModelNotFound = -32001,
    /// Strategy execution error
    StrategyError = -32002,
    /// Data retrieval/processing error
    DataError = -32003,
    /// Authentication error
    AuthError = -32004,
    /// Resource not found
    ResourceNotFound = -32005,
    /// Tool execution error
    ToolError = -32006,
    /// GPU acceleration error
    GpuError = -32007,
    /// Broker API error
    BrokerError = -32008,
    /// Market data error
    MarketDataError = -32009,
    /// Position limit error
    PositionLimitError = -32010,
    /// Risk threshold exceeded
    RiskError = -32011,
}

impl ErrorCode {
    /// Convert to i32 for JSON-RPC
    pub fn as_i32(self) -> i32 {
        self as i32
    }

    /// Get human-readable message for error code
    pub fn message(&self) -> &'static str {
        match self {
            ErrorCode::ParseError => "Parse error",
            ErrorCode::InvalidRequest => "Invalid request",
            ErrorCode::MethodNotFound => "Method not found",
            ErrorCode::InvalidParams => "Invalid params",
            ErrorCode::InternalError => "Internal error",
            ErrorCode::ModelNotFound => "Model not found",
            ErrorCode::StrategyError => "Strategy error",
            ErrorCode::DataError => "Data error",
            ErrorCode::AuthError => "Authentication error",
            ErrorCode::ResourceNotFound => "Resource not found",
            ErrorCode::ToolError => "Tool execution error",
            ErrorCode::GpuError => "GPU error",
            ErrorCode::BrokerError => "Broker API error",
            ErrorCode::MarketDataError => "Market data error",
            ErrorCode::PositionLimitError => "Position limit exceeded",
            ErrorCode::RiskError => "Risk threshold exceeded",
        }
    }
}

/// MCP Protocol Errors
#[derive(Error, Debug)]
pub enum ProtocolError {
    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Method not found: {0}")]
    MethodNotFound(String),

    #[error("Invalid parameters: {0}")]
    InvalidParams(String),

    #[error("Internal error: {0}")]
    InternalError(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Strategy error: {0}")]
    StrategyError(String),

    #[error("Data error: {0}")]
    DataError(String),

    #[error("Authentication error: {0}")]
    AuthError(String),

    #[error("Resource not found: {0}")]
    ResourceNotFound(String),

    #[error("Tool error: {0}")]
    ToolError(String),

    #[error("GPU error: {0}")]
    GpuError(String),

    #[error("Broker error: {0}")]
    BrokerError(String),

    #[error("Market data error: {0}")]
    MarketDataError(String),

    #[error("Position limit exceeded: {0}")]
    PositionLimitError(String),

    #[error("Risk threshold exceeded: {0}")]
    RiskError(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
}

impl ProtocolError {
    /// Get the error code for this error
    pub fn code(&self) -> ErrorCode {
        match self {
            ProtocolError::ParseError(_) | ProtocolError::SerializationError(_) => {
                ErrorCode::ParseError
            }
            ProtocolError::InvalidRequest(_) => ErrorCode::InvalidRequest,
            ProtocolError::MethodNotFound(_) => ErrorCode::MethodNotFound,
            ProtocolError::InvalidParams(_) => ErrorCode::InvalidParams,
            ProtocolError::InternalError(_) | ProtocolError::IoError(_) => {
                ErrorCode::InternalError
            }
            ProtocolError::ModelNotFound(_) => ErrorCode::ModelNotFound,
            ProtocolError::StrategyError(_) => ErrorCode::StrategyError,
            ProtocolError::DataError(_) => ErrorCode::DataError,
            ProtocolError::AuthError(_) => ErrorCode::AuthError,
            ProtocolError::ResourceNotFound(_) => ErrorCode::ResourceNotFound,
            ProtocolError::ToolError(_) => ErrorCode::ToolError,
            ProtocolError::GpuError(_) => ErrorCode::GpuError,
            ProtocolError::BrokerError(_) => ErrorCode::BrokerError,
            ProtocolError::MarketDataError(_) => ErrorCode::MarketDataError,
            ProtocolError::PositionLimitError(_) => ErrorCode::PositionLimitError,
            ProtocolError::RiskError(_) => ErrorCode::RiskError,
        }
    }

    /// Convert to JSON-RPC error object
    pub fn to_json_rpc_error(&self) -> super::JsonRpcError {
        super::JsonRpcError {
            code: self.code().as_i32(),
            message: self.to_string(),
            data: None,
        }
    }
}

pub type Result<T> = std::result::Result<T, ProtocolError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes() {
        assert_eq!(ErrorCode::ParseError.as_i32(), -32700);
        assert_eq!(ErrorCode::InvalidRequest.as_i32(), -32600);
        assert_eq!(ErrorCode::MethodNotFound.as_i32(), -32601);
    }

    #[test]
    fn test_error_messages() {
        assert_eq!(ErrorCode::ParseError.message(), "Parse error");
        assert_eq!(ErrorCode::MethodNotFound.message(), "Method not found");
    }

    #[test]
    fn test_protocol_error_conversion() {
        let error = ProtocolError::MethodNotFound("test_method".to_string());
        assert_eq!(error.code(), ErrorCode::MethodNotFound);

        let json_error = error.to_json_rpc_error();
        assert_eq!(json_error.code, -32601);
    }
}
