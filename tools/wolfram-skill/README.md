# Wolfram Scientific Computing Plugin

A Claude Code plugin that provides expert guidance for using Wolfram Alpha API and WolframScript for scientific computing.

## Features

- **Mathematical Validation**: Verify implementations of mathematical algorithms
- **Symbolic Computation**: Integrals, derivatives, differential equations
- **Scientific Data**: Physical constants, chemical properties
- **Hyperbolic Geometry**: Poincaré disk, Möbius transforms, geodesics
- **HyperPhysics Integration**: Native Rust crate support

## Installation

### For Claude Code

```bash
# From the HyperPhysics directory
claude /plugin install ./tools/wolfram-skill
```

### Environment Variables

Set `WOLFRAM_APP_ID` with your Wolfram Alpha API key:

```bash
export WOLFRAM_APP_ID="your-app-id"
```

## Usage

Once installed, Claude will automatically use this skill when you ask about:

- Mathematical computations
- Scientific validation
- Hyperbolic geometry
- Physics simulations
- Algorithm verification

## Examples

```
"Validate my hyperbolic distance implementation"
"Compute the integral of x³eˣ"
"What is the atomic mass of gold?"
"Verify that sin²(x) + cos²(x) = 1"
```

## Related

- `tools/wolfram-mcp/` - MCP server for Wolfram integration
- `crates/hyperphysics-wolfram/` - Rust crate for Wolfram bridge
