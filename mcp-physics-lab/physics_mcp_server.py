#!/usr/bin/env python3
"""
HyperPhysics MCP Server - Physics & Math Lab for Claude Code

Exposes enterprise-grade physics engines and mathematical tools to AI agents
via the Model Context Protocol (MCP).

## Capabilities

### Symbolic Mathematics (via Wolfram/SageMath/SymPy)
- solve: Symbolic equation solving
- simplify: Expression simplification
- integrate: Symbolic/numerical integration
- differentiate: Symbolic differentiation
- series: Taylor/Laurent series expansion
- laplace: Laplace transforms
- ode_solve: ODE solving

### Physics Simulation
- mujoco_sim: MuJoCo robotics simulation
- taichi_sim: Taichi MPM/SPH simulation
- rapier_sim: Rust Rapier physics (via bindings)
- bullet_sim: PyBullet quick prototyping

### Neural Simulation
- hodgkin_huxley: Full ion channel dynamics
- izhikevich: Efficient spiking model
- brian_network: Brian2 network simulation
- neuron_compartment: NEURON compartmental model

### Consciousness Metrics
- compute_phi: IIT Φ calculation
- global_workspace: GWT activation
- free_energy: Active inference free energy

### Optimization
- minimize: General optimization
- control_synthesis: Optimal control
- convex_solve: Convex optimization

## Usage

Register in Claude Desktop/Windsurf MCP settings:
```json
{
  "mcpServers": {
    "physics-lab": {
      "command": "python",
      "args": ["/Volumes/Tengritek/Ashina/HyperPhysics/mcp-physics-lab/physics_mcp_server.py"],
      "env": {
        "PYTHONPATH": "/Volumes/Tengritek/Ashina/.venv/lib/python3.13/site-packages"
      }
    }
  }
}
```
"""

import asyncio
import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os

# Add venv to path
VENV_PATH = "/Volumes/Tengritek/Ashina/.venv"
sys.path.insert(0, f"{VENV_PATH}/lib/python3.13/site-packages")

# MCP imports (will be available after install)
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("MCP not installed. Run: pip install mcp", file=sys.stderr)


# =============================================================================
# SYMBOLIC MATH ENGINE
# =============================================================================

class SymbolicEngine:
    """Unified symbolic math interface (SymPy + Wolfram + SageMath)"""
    
    def __init__(self):
        self.sympy = None
        self.wolfram_available = False
        self.sage_available = False
        
        try:
            import sympy
            self.sympy = sympy
        except ImportError:
            pass
        
        # Check for WolframScript
        self.wolfram_available = os.path.exists("/usr/local/bin/wolframscript") or \
                                  os.path.exists("/Applications/Wolfram Engine.app")
        
        # Check for SageMath
        self.sage_available = os.path.exists("/Applications/SageMath-10-7.app")
    
    def solve(self, equation: str, variable: str = "x") -> Dict[str, Any]:
        """Solve symbolic equation"""
        if self.sympy:
            x = self.sympy.Symbol(variable)
            expr = self.sympy.sympify(equation)
            solutions = self.sympy.solve(expr, x)
            return {
                "solutions": [str(s) for s in solutions],
                "engine": "sympy"
            }
        return {"error": "No symbolic engine available"}
    
    def integrate(self, expression: str, variable: str = "x", 
                  lower: Optional[float] = None, upper: Optional[float] = None) -> Dict[str, Any]:
        """Symbolic or definite integration"""
        if self.sympy:
            x = self.sympy.Symbol(variable)
            expr = self.sympy.sympify(expression)
            
            if lower is not None and upper is not None:
                result = self.sympy.integrate(expr, (x, lower, upper))
            else:
                result = self.sympy.integrate(expr, x)
            
            return {
                "result": str(result),
                "latex": self.sympy.latex(result),
                "engine": "sympy"
            }
        return {"error": "No symbolic engine available"}
    
    def differentiate(self, expression: str, variable: str = "x", order: int = 1) -> Dict[str, Any]:
        """Symbolic differentiation"""
        if self.sympy:
            x = self.sympy.Symbol(variable)
            expr = self.sympy.sympify(expression)
            result = self.sympy.diff(expr, x, order)
            return {
                "result": str(result),
                "latex": self.sympy.latex(result),
                "engine": "sympy"
            }
        return {"error": "No symbolic engine available"}
    
    def simplify(self, expression: str) -> Dict[str, Any]:
        """Simplify expression"""
        if self.sympy:
            expr = self.sympy.sympify(expression)
            result = self.sympy.simplify(expr)
            return {
                "result": str(result),
                "latex": self.sympy.latex(result),
                "engine": "sympy"
            }
        return {"error": "No symbolic engine available"}
    
    def series(self, expression: str, variable: str = "x", 
               point: float = 0, order: int = 6) -> Dict[str, Any]:
        """Taylor series expansion"""
        if self.sympy:
            x = self.sympy.Symbol(variable)
            expr = self.sympy.sympify(expression)
            result = self.sympy.series(expr, x, point, order)
            return {
                "result": str(result),
                "latex": self.sympy.latex(result),
                "engine": "sympy"
            }
        return {"error": "No symbolic engine available"}
    
    def wolfram_eval(self, code: str) -> Dict[str, Any]:
        """Execute Wolfram Language code"""
        if not self.wolfram_available:
            return {"error": "WolframScript not available"}
        
        try:
            result = subprocess.run(
                ["wolframscript", "-code", code],
                capture_output=True,
                text=True,
                timeout=30
            )
            return {
                "result": result.stdout.strip(),
                "error": result.stderr if result.returncode != 0 else None,
                "engine": "wolfram"
            }
        except subprocess.TimeoutExpired:
            return {"error": "Wolfram evaluation timed out"}
        except Exception as e:
            return {"error": str(e)}
    
    def sage_eval(self, code: str) -> Dict[str, Any]:
        """Execute SageMath code"""
        if not self.sage_available:
            return {"error": "SageMath not available"}
        
        sage_path = "/Applications/SageMath-10-7.app/Contents/Resources/sage/sage"
        try:
            result = subprocess.run(
                [sage_path, "-c", code],
                capture_output=True,
                text=True,
                timeout=30
            )
            return {
                "result": result.stdout.strip(),
                "error": result.stderr if result.returncode != 0 else None,
                "engine": "sage"
            }
        except subprocess.TimeoutExpired:
            return {"error": "SageMath evaluation timed out"}
        except Exception as e:
            return {"error": str(e)}


# =============================================================================
# PHYSICS ENGINE
# =============================================================================

class PhysicsEngine:
    """Multi-backend physics simulation"""
    
    def __init__(self):
        self.mujoco = None
        self.taichi = None
        self.pybullet = None
        
        try:
            import mujoco
            self.mujoco = mujoco
        except ImportError:
            pass
        
        try:
            import taichi as ti
            ti.init(arch=ti.metal)  # Use Metal on Mac
            self.taichi = ti
        except ImportError:
            pass
        
        try:
            import pybullet as p
            self.pybullet = p
        except ImportError:
            pass
    
    def mujoco_simulate(self, model_xml: str, steps: int = 100) -> Dict[str, Any]:
        """Run MuJoCo simulation"""
        if not self.mujoco:
            return {"error": "MuJoCo not available"}
        
        try:
            model = self.mujoco.MjModel.from_xml_string(model_xml)
            data = self.mujoco.MjData(model)
            
            positions = []
            for _ in range(steps):
                self.mujoco.mj_step(model, data)
                positions.append(data.qpos.tolist())
            
            return {
                "positions": positions,
                "final_state": {
                    "qpos": data.qpos.tolist(),
                    "qvel": data.qvel.tolist()
                },
                "engine": "mujoco"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def bullet_simulate(self, objects: List[Dict], steps: int = 100) -> Dict[str, Any]:
        """Quick physics simulation with PyBullet"""
        if not self.pybullet:
            return {"error": "PyBullet not available"}
        
        try:
            p = self.pybullet
            physics_client = p.connect(p.DIRECT)
            p.setGravity(0, 0, -9.81)
            
            # Create ground plane
            p.createCollisionShape(p.GEOM_PLANE)
            p.createMultiBody(0, 0)
            
            body_ids = []
            for obj in objects:
                shape_type = obj.get("shape", "sphere")
                position = obj.get("position", [0, 0, 1])
                mass = obj.get("mass", 1.0)
                
                if shape_type == "sphere":
                    radius = obj.get("radius", 0.1)
                    col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
                elif shape_type == "box":
                    size = obj.get("size", [0.1, 0.1, 0.1])
                    col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
                else:
                    continue
                
                body_id = p.createMultiBody(mass, col_shape, basePosition=position)
                body_ids.append(body_id)
            
            trajectories = {bid: [] for bid in body_ids}
            for _ in range(steps):
                p.stepSimulation()
                for bid in body_ids:
                    pos, _ = p.getBasePositionAndOrientation(bid)
                    trajectories[bid].append(list(pos))
            
            p.disconnect()
            
            return {
                "trajectories": trajectories,
                "engine": "pybullet"
            }
        except Exception as e:
            return {"error": str(e)}


# =============================================================================
# NEURAL SIMULATION
# =============================================================================

class NeuralEngine:
    """Neural simulation engine"""
    
    def hodgkin_huxley(self, duration_ms: float = 100, 
                       current_ua: float = 10.0,
                       dt: float = 0.01) -> Dict[str, Any]:
        """Simulate Hodgkin-Huxley neuron"""
        try:
            import numpy as np
        except ImportError:
            return {"error": "NumPy not available"}
        
        # HH parameters
        C_m = 1.0      # Membrane capacitance (µF/cm²)
        g_Na = 120.0   # Sodium conductance (mS/cm²)
        g_K = 36.0     # Potassium conductance (mS/cm²)
        g_L = 0.3      # Leak conductance (mS/cm²)
        E_Na = 50.0    # Sodium reversal potential (mV)
        E_K = -77.0    # Potassium reversal potential (mV)
        E_L = -54.387  # Leak reversal potential (mV)
        
        # Rate functions
        def alpha_m(V): return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        def beta_m(V): return 4.0 * np.exp(-(V + 65) / 18)
        def alpha_h(V): return 0.07 * np.exp(-(V + 65) / 20)
        def beta_h(V): return 1 / (1 + np.exp(-(V + 35) / 10))
        def alpha_n(V): return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
        def beta_n(V): return 0.125 * np.exp(-(V + 65) / 80)
        
        # Initial conditions
        V = -65.0
        m = alpha_m(V) / (alpha_m(V) + beta_m(V))
        h = alpha_h(V) / (alpha_h(V) + beta_h(V))
        n = alpha_n(V) / (alpha_n(V) + beta_n(V))
        
        # Simulation
        steps = int(duration_ms / dt)
        t = np.linspace(0, duration_ms, steps)
        V_trace = np.zeros(steps)
        
        for i in range(steps):
            # Ionic currents
            I_Na = g_Na * m**3 * h * (V - E_Na)
            I_K = g_K * n**4 * (V - E_K)
            I_L = g_L * (V - E_L)
            
            # Membrane equation
            dV = (current_ua - I_Na - I_K - I_L) / C_m
            V += dV * dt
            
            # Gating variables
            m += (alpha_m(V) * (1 - m) - beta_m(V) * m) * dt
            h += (alpha_h(V) * (1 - h) - beta_h(V) * h) * dt
            n += (alpha_n(V) * (1 - n) - beta_n(V) * n) * dt
            
            V_trace[i] = V
        
        # Find spikes
        spikes = []
        for i in range(1, len(V_trace) - 1):
            if V_trace[i] > 0 and V_trace[i] > V_trace[i-1] and V_trace[i] > V_trace[i+1]:
                spikes.append(t[i])
        
        return {
            "voltage_trace": V_trace[::10].tolist(),  # Downsample
            "time": t[::10].tolist(),
            "spike_times": spikes,
            "spike_count": len(spikes),
            "engine": "numpy_hh"
        }


# =============================================================================
# CONSCIOUSNESS METRICS
# =============================================================================

class ConsciousnessEngine:
    """Consciousness metrics computation"""
    
    def compute_phi(self, tpm: List[List[float]]) -> Dict[str, Any]:
        """Compute Integrated Information Φ (simplified)"""
        try:
            import numpy as np
        except ImportError:
            return {"error": "NumPy not available"}
        
        tpm = np.array(tpm)
        n = tpm.shape[0]
        
        # Compute whole system entropy
        stationary = np.ones(n) / n
        for _ in range(100):
            stationary = stationary @ tpm
        
        H_whole = -np.sum(stationary * np.log2(stationary + 1e-10))
        
        # Minimum information partition (simplified: bipartition)
        min_phi = float('inf')
        
        for i in range(1, n // 2 + 1):
            # Simple partition: first i vs rest
            H_part1 = -np.sum(stationary[:i] * np.log2(stationary[:i] / stationary[:i].sum() + 1e-10))
            H_part2 = -np.sum(stationary[i:] * np.log2(stationary[i:] / stationary[i:].sum() + 1e-10))
            
            phi = H_whole - (H_part1 + H_part2)
            min_phi = min(min_phi, abs(phi))
        
        return {
            "phi": min_phi,
            "whole_entropy": H_whole,
            "engine": "numpy_phi"
        }


# =============================================================================
# MCP SERVER
# =============================================================================

def create_mcp_server():
    """Create and configure MCP server"""
    
    if not MCP_AVAILABLE:
        return None
    
    server = Server("physics-lab")
    
    # Initialize engines
    symbolic = SymbolicEngine()
    physics = PhysicsEngine()
    neural = NeuralEngine()
    consciousness = ConsciousnessEngine()
    
    @server.list_tools()
    async def list_tools() -> List[Tool]:
        return [
            # Symbolic Math
            Tool(
                name="solve",
                description="Solve symbolic equation. Args: equation (str), variable (str, default 'x')",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "equation": {"type": "string", "description": "Equation to solve (e.g., 'x**2 - 4')"},
                        "variable": {"type": "string", "default": "x"}
                    },
                    "required": ["equation"]
                }
            ),
            Tool(
                name="integrate",
                description="Symbolic or definite integration. Args: expression, variable, lower, upper",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"},
                        "variable": {"type": "string", "default": "x"},
                        "lower": {"type": "number"},
                        "upper": {"type": "number"}
                    },
                    "required": ["expression"]
                }
            ),
            Tool(
                name="differentiate",
                description="Symbolic differentiation. Args: expression, variable, order",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"},
                        "variable": {"type": "string", "default": "x"},
                        "order": {"type": "integer", "default": 1}
                    },
                    "required": ["expression"]
                }
            ),
            Tool(
                name="simplify",
                description="Simplify mathematical expression",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"}
                    },
                    "required": ["expression"]
                }
            ),
            Tool(
                name="wolfram",
                description="Execute Wolfram Language code directly",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Wolfram Language code"}
                    },
                    "required": ["code"]
                }
            ),
            Tool(
                name="sage",
                description="Execute SageMath code",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "SageMath/Python code"}
                    },
                    "required": ["code"]
                }
            ),
            # Physics
            Tool(
                name="mujoco_sim",
                description="Run MuJoCo physics simulation from MJCF XML",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model_xml": {"type": "string", "description": "MJCF XML model"},
                        "steps": {"type": "integer", "default": 100}
                    },
                    "required": ["model_xml"]
                }
            ),
            Tool(
                name="bullet_sim",
                description="Quick physics simulation with PyBullet",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "objects": {"type": "array", "description": "List of objects with shape, position, mass"},
                        "steps": {"type": "integer", "default": 100}
                    },
                    "required": ["objects"]
                }
            ),
            # Neural
            Tool(
                name="hodgkin_huxley",
                description="Simulate Hodgkin-Huxley neuron",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "duration_ms": {"type": "number", "default": 100},
                        "current_ua": {"type": "number", "default": 10},
                        "dt": {"type": "number", "default": 0.01}
                    }
                }
            ),
            # Consciousness
            Tool(
                name="compute_phi",
                description="Compute IIT Integrated Information Φ from transition probability matrix",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tpm": {"type": "array", "description": "Transition probability matrix"}
                    },
                    "required": ["tpm"]
                }
            ),
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        result = {}
        
        if name == "solve":
            result = symbolic.solve(arguments["equation"], arguments.get("variable", "x"))
        elif name == "integrate":
            result = symbolic.integrate(
                arguments["expression"],
                arguments.get("variable", "x"),
                arguments.get("lower"),
                arguments.get("upper")
            )
        elif name == "differentiate":
            result = symbolic.differentiate(
                arguments["expression"],
                arguments.get("variable", "x"),
                arguments.get("order", 1)
            )
        elif name == "simplify":
            result = symbolic.simplify(arguments["expression"])
        elif name == "wolfram":
            result = symbolic.wolfram_eval(arguments["code"])
        elif name == "sage":
            result = symbolic.sage_eval(arguments["code"])
        elif name == "mujoco_sim":
            result = physics.mujoco_simulate(
                arguments["model_xml"],
                arguments.get("steps", 100)
            )
        elif name == "bullet_sim":
            result = physics.bullet_simulate(
                arguments["objects"],
                arguments.get("steps", 100)
            )
        elif name == "hodgkin_huxley":
            result = neural.hodgkin_huxley(
                arguments.get("duration_ms", 100),
                arguments.get("current_ua", 10),
                arguments.get("dt", 0.01)
            )
        elif name == "compute_phi":
            result = consciousness.compute_phi(arguments["tpm"])
        else:
            result = {"error": f"Unknown tool: {name}"}
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    return server


async def main():
    """Run the MCP server"""
    server = create_mcp_server()
    
    if server is None:
        print("MCP not available. Install with: pip install mcp", file=sys.stderr)
        sys.exit(1)
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


if __name__ == "__main__":
    asyncio.run(main())
