//! HTTP+SSE transport for MCP server (stub)

#[cfg(feature = "http")]
pub struct HttpTransport;

#[cfg(feature = "http")]
impl HttpTransport {
    pub fn new() -> Self {
        Self
    }
}
