import warp as wp

# Initialize Warp
wp.init()

@wp.struct
class AgentState:
    position: wp.vec3
    velocity: wp.vec3
    capital: float
    inventory: float
    risk_aversion: float

@wp.struct
class MarketState:
    price: float
    volume: float
    volatility: float
    trend: float

@wp.kernel
def market_dynamics_kernel(
    agents: wp.array(dtype=AgentState),
    market: wp.array(dtype=MarketState),
    dt: float,
    num_agents: int
):
    # Simple differentiable market physics
    # Price update based on aggregate agent action (simplified proxy)
    
    tid = wp.tid()
    
    # In a real kernel, we would use atomic adds to aggregate orders
    # For this prototype, we simulate local impact
    
    # Example: Agent position affects local "field" of price
    # This is a placeholder for the actual order book dynamics
    
    agent = agents[tid]
    
    # Differentiable logic:
    # If agent moves "up" (buy), price pressure increases
    price_pressure = agent.velocity[1] * agent.capital * 0.001
    
    # We can't write to global market state from parallel threads without atomics
    # But for DiffPhys, we often compute per-agent gradients
    
    # Let's update agent state based on market
    # (Agents react to price)
    
    current_price = market[0].price
    
    # Force = -Gradient of Potential
    # Potential = Risk Aversion * (Price - Target)^2
    
    force = wp.vec3(0.0, 0.0, 0.0)
    if current_price > 100.0:
        force = wp.vec3(0.0, -1.0, 0.0) * agent.risk_aversion
    else:
        force = wp.vec3(0.0, 1.0, 0.0) * (1.0 / agent.risk_aversion)
        
    # Symplectic Euler integration
    agent.velocity = agent.velocity + force * dt
    agent.position = agent.position + agent.velocity * dt

@wp.kernel
def price_update_kernel(
    agent_velocities: wp.array(dtype=wp.vec3),
    agent_capitals: wp.array(dtype=float),
    market_price: wp.array(dtype=float),
    market_volume: wp.array(dtype=float),
    dt: float
):
    """Aggregate agent actions to update market price"""
    tid = wp.tid()

    # Each agent's buy/sell pressure
    velocity_y = agent_velocities[tid][1]  # Y-component = buy/sell direction
    capital = agent_capitals[tid]

    # Price impact proportional to capital and direction
    impact = velocity_y * capital * 0.0001 * dt

    # Atomic add to aggregate all impacts
    wp.atomic_add(market_price, 0, impact)
    wp.atomic_add(market_volume, 0, wp.abs(impact) * 1000.0)


class WarpSimulation:
    """Differentiable market simulation using NVIDIA Warp"""

    def __init__(self, num_agents: int, device: str = "cuda:0"):
        self.num_agents = num_agents
        self.device = device

        # Initialize agent arrays
        self.positions = wp.zeros(num_agents, dtype=wp.vec3, device=device)
        self.velocities = wp.zeros(num_agents, dtype=wp.vec3, device=device)
        self.capitals = wp.full(num_agents, 100000.0, dtype=float, device=device)
        self.risk_aversions = wp.full(num_agents, 0.5, dtype=float, device=device)

        # Market state
        self.price = wp.full(1, 100.0, dtype=float, device=device)
        self.volume = wp.zeros(1, dtype=float, device=device)
        self.volatility = wp.full(1, 0.02, dtype=float, device=device)

        # Enable gradient computation
        self.tape = None

    def step(self, dt: float, record_gradients: bool = False):
        """Step simulation forward by dt seconds"""
        if record_gradients:
            self.tape = wp.Tape()
            self.tape.begin()

        # Launch agent dynamics kernel
        wp.launch(
            kernel=market_dynamics_kernel,
            dim=self.num_agents,
            inputs=[
                self.positions,
                self.velocities,
                self.capitals,
                self.risk_aversions,
                self.price,
                dt
            ],
            device=self.device
        )

        # Launch price update kernel
        wp.launch(
            kernel=price_update_kernel,
            dim=self.num_agents,
            inputs=[
                self.velocities,
                self.capitals,
                self.price,
                self.volume,
                dt
            ],
            device=self.device
        )

        if record_gradients and self.tape:
            self.tape.end()

        wp.synchronize()

    def backward(self, loss: wp.array):
        """Compute gradients via autodiff"""
        if self.tape:
            self.tape.backward(loss)
            return {
                'position_grads': self.positions.grad,
                'velocity_grads': self.velocities.grad,
                'capital_grads': self.capitals.grad,
            }
        return None

    def get_price(self) -> float:
        """Get current market price"""
        return self.price.numpy()[0]

    def get_volume(self) -> float:
        """Get current trading volume"""
        return self.volume.numpy()[0]


# Global simulation instance (created lazily)
_simulation = None


def init_simulation(num_agents: int, device: str = "cuda:0") -> str:
    """Initialize the global simulation"""
    global _simulation
    try:
        _simulation = WarpSimulation(num_agents, device)
        return f"Initialized Warp simulation with {num_agents} agents on {device}"
    except Exception as e:
        return f"Failed to initialize Warp: {e}"


def step_simulation(agents_ptr, market_ptr, num_agents, dt):
    """Step simulation - called from Rust via PyO3"""
    global _simulation

    # Lazy initialization
    if _simulation is None or _simulation.num_agents != num_agents:
        try:
            # Try GPU first, fall back to CPU
            try:
                _simulation = WarpSimulation(num_agents, "cuda:0")
            except:
                _simulation = WarpSimulation(num_agents, "cpu")
        except Exception as e:
            # Ultimate fallback: no simulation
            return f"Simulation unavailable: {e}. Stepped {num_agents} agents with dt={dt} (stub)"

    try:
        _simulation.step(dt)
        price = _simulation.get_price()
        volume = _simulation.get_volume()
        return f"Stepped {num_agents} agents: price={price:.4f}, volume={volume:.2f}, dt={dt}"
    except Exception as e:
        return f"Step failed: {e}"


def compute_gradients(loss_value: float) -> dict:
    """Compute gradients for optimization"""
    global _simulation

    if _simulation is None:
        return {"error": "Simulation not initialized"}

    try:
        # Create loss array
        loss = wp.full(1, loss_value, dtype=float, device=_simulation.device)
        grads = _simulation.backward(loss)

        if grads:
            return {
                "position_grads": grads['position_grads'].numpy().tolist() if grads['position_grads'] else [],
                "velocity_grads": grads['velocity_grads'].numpy().tolist() if grads['velocity_grads'] else [],
            }
        return {"error": "No gradients recorded (call step with record_gradients=True)"}
    except Exception as e:
        return {"error": str(e)}
