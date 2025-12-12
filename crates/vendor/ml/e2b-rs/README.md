# E2B Rust Client

A Rust client for [E2B](https://e2b.dev) - secure cloud sandboxes for AI-generated code execution.

## Overview

E2B is an open-source infrastructure for running AI-generated code in secure isolated sandboxes in the cloud. This crate provides a Rust interface to the E2B API.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
e2b-rs = { path = "../vendor/ml/e2b-rs" }
```

## Usage

```rust
use e2b_rs::{E2BClient, SandboxConfig};

#[tokio::main]
async fn main() -> Result<(), e2b_rs::Error> {
    // Create client with API key
    let client = E2BClient::new(std::env::var("E2B_API_KEY")?)?;
    
    // Create a sandbox
    let sandbox = client.create_sandbox(SandboxConfig::default()).await?;
    
    // Execute Python code
    let result = sandbox.execute("print('Hello from E2B!')").await?;
    println!("Output: {}", result.stdout);
    
    // Run shell commands
    let ls_result = sandbox.run_command("ls -la").await?;
    println!("Files: {}", ls_result.stdout);
    
    // File operations
    sandbox.write_file("/tmp/test.txt", b"Hello, World!").await?;
    let content = sandbox.read_file("/tmp/test.txt").await?;
    
    // Clean up
    sandbox.close().await?;
    Ok(())
}
```

## Configuration

```rust
let config = SandboxConfig {
    template: "python".to_string(),  // or "nodejs", "base", etc.
    timeout_ms: Some(300_000),       // 5 minutes
    keep_alive: true,
    metadata: Some(HashMap::from([
        ("project".to_string(), "my-project".to_string()),
    ])),
};
```

## API Key

Get your API key from [E2B Dashboard](https://e2b.dev/dashboard).

Set via environment variable:
```bash
export E2B_API_KEY=your-api-key
```

## Features

- **Code Execution**: Run Python, JavaScript, or shell commands
- **File System**: Read/write files in the sandbox
- **Streaming**: Stream execution output (coming soon)
- **Templates**: Use pre-built templates or custom ones

## License

MIT
