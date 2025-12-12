# HyperPhysics MCP Physics Lab

Enterprise-grade physics and mathematics computation server for AI agents via Model Context Protocol (MCP).

## System Specifications

| Component | Value |
|-----------|-------|
| **Machine** | Mac Pro 7,1 |
| **CPU** | Intel 8-core @ 3 GHz |
| **RAM** | 96 GB |
| **GPU 1** | AMD Radeon RX 6800 XT (16 GB VRAM) |
| **GPU 2** | AMD Radeon RX 5500 XT |
| **Graphics API** | Metal 3 |

## Quick Setup

```bash
cd /Volumes/Tengritek/Ashina/HyperPhysics/mcp-physics-lab
chmod +x setup.sh
./setup.sh
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Claude Code / AI Agent                        │
└─────────────────────────────────────────────────────────────────┘
                              │ MCP Protocol
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    physics_mcp_server.py                         │
│  ┌───────────────┐  ┌───────────────┐  ┌─────────────────────┐  │
│  │ SymbolicEngine│  │ PhysicsEngine │  │ NeuralEngine        │  │
│  │ (SymPy/       │  │ (MuJoCo/      │  │ (HH/Brian2)         │  │
│  │  Wolfram/Sage)│  │  Taichi/Bullet│  │                     │  │
│  └───────────────┘  └───────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐
│ WolframScript│    │   SageMath   │    │  HyperPhysics Crates │
│   (Symbolic) │    │  (Symbolic)  │    │  (Rust Physics)      │
└──────────────┘    └──────────────┘    └──────────────────────┘
```

## Available Tools

### Symbolic Mathematics

| Tool | Description | Engine |
|------|-------------|--------|
| `solve` | Solve equations symbolically | SymPy |
| `integrate` | Symbolic/definite integration | SymPy |
| `differentiate` | Symbolic differentiation | SymPy |
| `simplify` | Expression simplification | SymPy |
| `series` | Taylor series expansion | SymPy |
| `wolfram` | Execute Wolfram Language code | WolframScript |
| `sage` | Execute SageMath code | SageMath |

### Physics Simulation

| Tool | Description | Engine |
|------|-------------|--------|
| `mujoco_sim` | Robotics simulation from MJCF | MuJoCo |
| `bullet_sim` | Quick rigid body physics | PyBullet |
| `taichi_sim` | GPU-accelerated MPM/SPH | Taichi (Metal) |

### Neural Simulation

| Tool | Description | Engine |
|------|-------------|--------|
| `hodgkin_huxley` | Full ion channel neuron | NumPy |
| `izhikevich` | Efficient spiking model | NumPy |
| `brian_network` | Spiking neural network | Brian2 |

### Consciousness Metrics

| Tool | Description | Engine |
|------|-------------|--------|
| `compute_phi` | IIT Integrated Information Φ | NumPy |
| `global_workspace` | GWT activation | NumPy |
| `free_energy` | Active inference VFE | NumPy |

## MCP Configuration

### For Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "physics-lab": {
      "command": "/Volumes/Tengritek/Ashina/.venv/bin/python",
      "args": ["/Volumes/Tengritek/Ashina/HyperPhysics/mcp-physics-lab/physics_mcp_server.py"]
    }
  }
}
```

### For Windsurf/Cascade

Add to MCP settings in the IDE.

## Usage Examples

### Solve Equation

```python
# Claude can call:
solve(equation="x**2 - 4*x + 4", variable="x")
# Returns: {"solutions": ["2"], "engine": "sympy"}
```

### Integrate

```python
integrate(expression="sin(x)*exp(-x)", variable="x")
# Returns: {"result": "-exp(-x)*(sin(x) + cos(x))/2", "latex": "..."}
```

### Wolfram Language

```python
wolfram(code="Integrate[Sin[x]^2, {x, 0, Pi}]")
# Returns: {"result": "Pi/2", "engine": "wolfram"}
```

### MuJoCo Simulation

```python
mujoco_sim(model_xml="""
<mujoco>
  <worldbody>
    <body name="ball" pos="0 0 1">
      <joint type="free"/>
      <geom type="sphere" size="0.1"/>
    </body>
    <geom type="plane" size="5 5 0.1"/>
  </worldbody>
</mujoco>
""", steps=100)
```

### Hodgkin-Huxley Neuron

```python
hodgkin_huxley(duration_ms=100, current_ua=10, dt=0.01)
# Returns: {"spike_times": [...], "spike_count": 5, "voltage_trace": [...]}
```

### Compute Φ (Consciousness)

```python
compute_phi(tpm=[[0.9, 0.1], [0.1, 0.9]])
# Returns: {"phi": 0.531, "whole_entropy": 1.0}
```

## HyperPhysics Vendor Integration

The following physics engines from `/Volumes/Tengritek/Ashina/HyperPhysics/crates/vendor/physics/` can be exposed:

| Engine | Status | Integration Path |
|--------|--------|------------------|
| **Rapier** | 🟢 Via Rust bindings | `rapier-hyperphysics` crate |
| **MuJoCo** | 🟢 Direct Python | `mujoco-hyperphysics` crate |
| **Genesis** | 🟡 Python import | Direct `genesis` package |
| **Warp** | 🟡 CPU only on Mac | Direct `warp` package |
| **Taichi** | 🟢 Metal GPU | Direct `taichi` package |
| **JoltPhysics** | 🔴 Needs bindings | `jolt-hyperphysics` crate |
| **Avian** | 🟢 Via Bevy | `avian3d` crate |

## Extending the Server

### Add New Tool

```python
# In physics_mcp_server.py

@server.list_tools()
async def list_tools() -> List[Tool]:
    return [
        # ... existing tools ...
        Tool(
            name="my_new_tool",
            description="Description",
            inputSchema={
                "type": "object",
                "properties": {
                    "param1": {"type": "string"}
                },
                "required": ["param1"]
            }
        ),
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    if name == "my_new_tool":
        result = my_function(arguments["param1"])
        return [TextContent(type="text", text=json.dumps(result))]
```

### Add New Engine

```python
class MyEngine:
    def __init__(self):
        # Initialize
        pass
    
    def compute(self, params):
        # Computation
        return {"result": ...}

# Register in create_mcp_server()
my_engine = MyEngine()
```

## Files

```
mcp-physics-lab/
├── physics_mcp_server.py   # Main MCP server
├── requirements.txt        # Python dependencies
├── setup.sh               # Installation script
├── mcp_config.json        # MCP configuration template
└── README.md              # This file
```

## Dependencies

### Already Installed (via Homebrew)
- SageMath 10.7
- ParaView 6.0.1
- Wolfram Engine (if available)

### Python (via setup.sh)
- NumPy, SciPy, SymPy, mpmath
- MuJoCo, PyBullet, Taichi
- Brian2, NEURON
- PyTorch, JAX, MLX
- OpenMM, MDAnalysis
- PyVista, Open3D
- CasADi, CVXPY, Optuna

## License

MIT
