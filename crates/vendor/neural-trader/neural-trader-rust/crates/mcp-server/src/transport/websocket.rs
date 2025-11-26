//! WebSocket transport for MCP server (stub)

#[cfg(feature = "websocket")]
pub struct WebSocketTransport;

#[cfg(feature = "websocket")]
impl WebSocketTransport {
    pub fn new() -> Self {
        Self
    }
}
