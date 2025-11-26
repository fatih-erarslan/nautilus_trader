# mcp-server

Model Context Protocol (MCP) server implementation for Neural Trader.

## Features

- MCP tool registration and execution
- Multi-transport support (stdio, HTTP, WebSocket)
- Async request handling
- Trading operation tools

## Usage

```rust
use mcp_server::{Server, Tool};

let server = Server::new();
server.register_tool(my_tool);
server.start().await?;
```

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
