#!/bin/bash
# HyperPhysics MCP Physics Lab Setup
# Run from: /Volumes/Tengritek/Ashina/HyperPhysics/mcp-physics-lab

set -e

VENV_PATH="/Volumes/Tengritek/Ashina/.venv"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     HyperPhysics MCP Physics Lab Setup                       ║"
echo "║     Mac Pro 7,1 | 96GB RAM | RX 6800 XT | Metal 3            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Activate conda environment
echo "→ Activating conda environment..."
source "$VENV_PATH/bin/activate" 2>/dev/null || {
    # If not conda, try regular venv
    source "$VENV_PATH/bin/activate"
}

echo "→ Python: $(python --version)"
echo "→ Location: $(which python)"
echo ""

# Core packages (required)
echo "═══════════════════════════════════════════════════════════════"
echo "Phase 1: Core Scientific Computing"
echo "═══════════════════════════════════════════════════════════════"
pip install --upgrade pip
pip install numpy scipy sympy mpmath

# MCP SDK
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Phase 2: MCP SDK"
echo "═══════════════════════════════════════════════════════════════"
pip install mcp fastapi uvicorn httpx aiohttp

# Physics Engines
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Phase 3: Physics Engines"
echo "═══════════════════════════════════════════════════════════════"
pip install mujoco pybullet
pip install taichi  # Metal backend for GPU physics

# Neural Simulation
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Phase 4: Neural Simulation"
echo "═══════════════════════════════════════════════════════════════"
pip install brian2
# NEURON requires special handling
pip install neuron || echo "⚠ NEURON install may need: brew install neuron"

# ML Frameworks (Apple Silicon optimized)
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Phase 5: ML Frameworks"
echo "═══════════════════════════════════════════════════════════════"
pip install torch torchvision  # MPS backend for AMD GPU
pip install jax jaxlib
pip install mlx || echo "⚠ MLX is Apple Silicon only"

# Geometry & Visualization
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Phase 6: Geometry & Visualization"
echo "═══════════════════════════════════════════════════════════════"
pip install pygmsh meshio trimesh pyvista open3d
pip install matplotlib plotly

# Optimization
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Phase 7: Optimization"
echo "═══════════════════════════════════════════════════════════════"
pip install casadi cvxpy nlopt optuna

# Molecular Dynamics (Optional)
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Phase 8: Biophysics (Optional)"
echo "═══════════════════════════════════════════════════════════════"
pip install openmm MDAnalysis ase || echo "⚠ Some biophysics packages may need conda"

# Jupyter
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Phase 9: Jupyter Environment"
echo "═══════════════════════════════════════════════════════════════"
pip install jupyterlab ipywidgets

# Utilities
pip install rich typer pydantic python-dotenv tqdm

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    ✓ Installation Complete                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo ""
echo "1. Add MCP server to Claude Desktop/Windsurf:"
echo "   Edit: ~/Library/Application Support/Claude/claude_desktop_config.json"
echo ""
echo '   {
     "mcpServers": {
       "physics-lab": {
         "command": "'$VENV_PATH'/bin/python",
         "args": ["'$SCRIPT_DIR'/physics_mcp_server.py"]
       }
     }
   }'
echo ""
echo "2. Test the server:"
echo "   python $SCRIPT_DIR/physics_mcp_server.py"
echo ""
echo "3. Start Jupyter Lab:"
echo "   jupyter lab"
echo ""
echo "4. Available tools:"
echo "   - solve, integrate, differentiate, simplify (SymPy)"
echo "   - wolfram (WolframScript)"
echo "   - sage (SageMath)"
echo "   - mujoco_sim, bullet_sim (Physics)"
echo "   - hodgkin_huxley (Neural)"
echo "   - compute_phi (Consciousness)"
