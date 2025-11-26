# mcp-protocol

Model Context Protocol (MCP) definitions and types for Neural Trader.

## Features

- MCP message types
- Protocol serialization/deserialization
- Request/response handling
- WebSocket and stdio transport

## Usage

```rust
use mcp_protocol::{Request, Response, Tool};

let request = Request::CallTool {
    name: "execute_trade".to_string(),
    arguments: serde_json::json!({}),
};
```

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
