//! MCP request types with cryptographic signing

use hyperphysics_dilithium::DilithiumSignature;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Unsigned MCP request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpRequest {
    /// Unique request identifier
    pub id: String,

    /// JSON-RPC version (always "2.0")
    pub jsonrpc: String,

    /// Tool method name
    pub method: String,

    /// Tool parameters
    pub params: serde_json::Value,

    /// Request timestamp
    pub timestamp: DateTime<Utc>,

    /// Cryptographic nonce for replay protection
    pub nonce: String,
}

impl McpRequest {
    /// Create a new MCP request
    pub fn new(method: impl Into<String>, params: serde_json::Value) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            jsonrpc: "2.0".to_string(),
            method: method.into(),
            params,
            timestamp: Utc::now(),
            nonce: generate_nonce(),
        }
    }

    /// Create request with custom ID
    pub fn with_id(id: impl Into<String>, method: impl Into<String>, params: serde_json::Value) -> Self {
        Self {
            id: id.into(),
            jsonrpc: "2.0".to_string(),
            method: method.into(),
            params,
            timestamp: Utc::now(),
            nonce: generate_nonce(),
        }
    }

    /// Get canonical bytes for signing
    pub fn canonical_bytes(&self) -> Vec<u8> {
        // Create deterministic representation for signing
        let canonical = CanonicalRequest {
            id: &self.id,
            method: &self.method,
            params_hash: hash_params(&self.params),
            timestamp: self.timestamp.timestamp(),
            nonce: &self.nonce,
        };

        // Use bincode for deterministic serialization
        bincode::serialize(&canonical).unwrap_or_else(|_| {
            // Fallback to JSON if bincode fails
            serde_json::to_vec(&canonical).unwrap_or_default()
        })
    }
}

/// Canonical request structure for deterministic signing
#[derive(Serialize)]
struct CanonicalRequest<'a> {
    id: &'a str,
    method: &'a str,
    params_hash: String,
    timestamp: i64,
    nonce: &'a str,
}

/// Signed MCP request with cryptographic authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedMcpRequest {
    /// The underlying request
    pub request: McpRequest,

    /// Client identifier
    pub client_id: String,

    /// Dilithium signature over canonical request bytes
    pub signature: DilithiumSignature,

    /// BLAKE3 hash of the request for integrity verification
    pub content_hash: String,
}

impl SignedMcpRequest {
    /// Create a signed request (signature must be provided externally)
    pub fn new(
        request: McpRequest,
        client_id: String,
        signature: DilithiumSignature,
    ) -> Self {
        let content_hash = compute_content_hash(&request);
        Self {
            request,
            client_id,
            signature,
            content_hash,
        }
    }

    /// Verify content hash integrity
    pub fn verify_content_hash(&self) -> bool {
        let computed = compute_content_hash(&self.request);
        self.content_hash == computed
    }
}

/// MCP response with optional signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResponse {
    /// Request ID this response corresponds to
    pub id: String,

    /// JSON-RPC version
    pub jsonrpc: String,

    /// Result on success
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,

    /// Error on failure
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<McpError>,

    /// Response timestamp
    pub timestamp: DateTime<Utc>,

    /// Optional server signature for authenticated responses
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<DilithiumSignature>,
}

impl McpResponse {
    /// Create successful response
    pub fn success(id: impl Into<String>, result: serde_json::Value) -> Self {
        Self {
            id: id.into(),
            jsonrpc: "2.0".to_string(),
            result: Some(result),
            error: None,
            timestamp: Utc::now(),
            signature: None,
        }
    }

    /// Create error response
    pub fn error(id: impl Into<String>, code: i32, message: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            jsonrpc: "2.0".to_string(),
            result: None,
            error: Some(McpError {
                code,
                message: message.into(),
                data: None,
            }),
            timestamp: Utc::now(),
            signature: None,
        }
    }

    /// Create authentication error response
    pub fn auth_error(id: impl Into<String>, message: impl Into<String>) -> Self {
        Self::error(id, -32001, message)
    }

    /// Create unauthorized error response
    pub fn unauthorized(id: impl Into<String>) -> Self {
        Self::error(id, -32002, "Unauthorized: Invalid or missing authentication")
    }

    /// Create rate limit error response
    pub fn rate_limited(id: impl Into<String>) -> Self {
        Self::error(id, -32003, "Rate limit exceeded")
    }
}

/// MCP error structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

/// Generate cryptographic nonce
fn generate_nonce() -> String {
    let timestamp = Utc::now().timestamp_nanos_opt().unwrap_or(0);
    let random_bytes: [u8; 16] = rand_bytes();

    let mut hasher = blake3::Hasher::new();
    hasher.update(&timestamp.to_le_bytes());
    hasher.update(&random_bytes);

    hasher.finalize().to_hex().to_string()
}

/// Generate random bytes (using system RNG)
fn rand_bytes<const N: usize>() -> [u8; N] {
    let mut bytes = [0u8; N];
    getrandom::getrandom(&mut bytes).unwrap_or_else(|_| {
        // Fallback: use timestamp-based entropy (less secure but functional)
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        for (i, byte) in bytes.iter_mut().enumerate() {
            *byte = ((ts >> (i * 8)) & 0xFF) as u8;
        }
    });
    bytes
}

/// Hash parameters for canonical representation
fn hash_params(params: &serde_json::Value) -> String {
    let bytes = serde_json::to_vec(params).unwrap_or_default();
    blake3::hash(&bytes).to_hex().to_string()
}

/// Compute BLAKE3 hash of request content
fn compute_content_hash(request: &McpRequest) -> String {
    let bytes = request.canonical_bytes();
    blake3::hash(&bytes).to_hex().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_request() {
        let request = McpRequest::new("vector_db_search", serde_json::json!({
            "query": [0.1, 0.2, 0.3],
            "k": 10
        }));

        assert_eq!(request.jsonrpc, "2.0");
        assert_eq!(request.method, "vector_db_search");
        assert!(!request.nonce.is_empty());
    }

    #[test]
    fn test_canonical_bytes_deterministic() {
        let request = McpRequest::with_id(
            "test-id",
            "test_method",
            serde_json::json!({"key": "value"})
        );

        let bytes1 = request.canonical_bytes();
        let bytes2 = request.canonical_bytes();

        assert_eq!(bytes1, bytes2);
    }

    #[test]
    fn test_response_creation() {
        let success = McpResponse::success("1", serde_json::json!({"result": "ok"}));
        assert!(success.result.is_some());
        assert!(success.error.is_none());

        let error = McpResponse::error("2", -32600, "Invalid request");
        assert!(error.result.is_none());
        assert!(error.error.is_some());
    }
}
