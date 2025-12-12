# Wolfram MCP Server v2.0 Setup

High-performance MCP server for Wolfram Alpha with **Bun.js** runtime and native **Rust/Swift** bindings.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Wolfram MCP Server                        │
│                       (Bun.js)                               │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Wolfram API │  │ WolframScript│  │  Native Modules    │  │
│  │  (Remote)   │  │   (Local)    │  │  (Rust + Swift)    │  │
│  └──────┬──────┘  └──────┬──────┘  └─────────┬───────────┘  │
│         │                │                   │               │
│         └────────────────┴───────────────────┘               │
│                          │                                   │
│              Automatic Fallback Chain                        │
│         API → WolframScript → Native Compute                 │
└─────────────────────────────────────────────────────────────┘
```

## Native Bindings

### Rust (NAPI-RS)
- Hyperbolic geometry (Poincaré disk, geodesics, Möbius transforms)
- STDP weight updates
- Shannon entropy, KL divergence
- LMSR cost function
- Ising Hamiltonian
- Landauer bound

### Swift (macOS)
- Native macOS performance
- Same mathematical primitives
- CLI tool for subprocess integration

## Quick Setup

### 1. Get Your Wolfram API Key

1. Go to [Wolfram Alpha Developer Portal](https://developer.wolframalpha.com)
2. Sign in with your Wolfram ID (Pro subscription)
3. Click "Get an AppID" and create a new app
4. Copy your AppID

### 2. Configure for Windsurf

Add to your Windsurf settings (`~/.codeium/windsurf/mcp_config.json`):

```json
{
  "mcpServers": {
    "wolfram": {
      "command": "node",
      "args": ["/Volumes/Tengritek/Ashina/HyperPhysics/tools/wolfram-mcp/dist/index.js"],
      "env": {
        "WOLFRAM_APP_ID": "YOUR_APP_ID_HERE"
      }
    }
  }
}
```

### 3. Configure for Claude Code (claude_desktop_config.json)

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "wolfram": {
      "command": "node",
      "args": ["/Volumes/Tengritek/Ashina/HyperPhysics/tools/wolfram-mcp/dist/index.js"],
      "env": {
        "WOLFRAM_APP_ID": "YOUR_APP_ID_HERE"
      }
    }
  }
}
```

### 4. Test the Server

```bash
# Set your API key
export WOLFRAM_APP_ID="your-app-id"

# Run the server directly for testing
node /Volumes/Tengritek/Ashina/HyperPhysics/tools/wolfram-mcp/dist/index.js
```

## Available Tools

Once configured, the following tools will be available to Claude/Windsurf:

| Tool | Description |
|------|-------------|
| `wolfram_llm_query` | Natural language queries optimized for AI |
| `wolfram_compute` | Mathematical computations (integrals, derivatives, etc.) |
| `wolfram_validate` | Validate mathematical expressions |
| `wolfram_unit_convert` | Convert between units |
| `wolfram_data_query` | Query scientific/geographic/financial data |
| `wolfram_full_query` | Full structured results with all pods |

## Example Usage in Claude/Windsurf

Once connected, you can ask:

- "Use Wolfram to compute the integral of x^2 * sin(x)"
- "Ask Wolfram for the population of Japan"
- "Validate with Wolfram that sin²(x) + cos²(x) = 1"
- "Convert 100 miles per hour to meters per second using Wolfram"
- "Query Wolfram for the atomic mass of gold"

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `WOLFRAM_APP_ID` | Yes | Your Wolfram Alpha API AppID |

## Troubleshooting

### Server won't start
- Ensure Node.js 18+ is installed
- Check that `WOLFRAM_APP_ID` is set correctly
- Run `npm install` and `npm run build` in the `wolfram-mcp` directory

### API errors
- Verify your AppID is valid at https://developer.wolframalpha.com
- Check your API usage limits (Free: 2000/month, Pro: more)

### Connection issues in Windsurf/Claude
- Restart the IDE after updating MCP config
- Check the path to the built `index.js` is correct
