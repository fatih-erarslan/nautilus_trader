#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Quantum Trading System - Main Orchestration

This module brings together all quantum-biological trading agents with real messaging
to create a unified decision-making system for algorithmic trading.
"""

import asyncio
import logging
import signal
import sys
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import os
from dataclasses import dataclass, field

# Import all the agents and systems
try:
    from pads import PanarchyAdaptiveDecisionSystem
    PADS_AVAILABLE = True
except ImportError:
    PADS_AVAILABLE = False

try:
    from quasar import QUASAR
    QUASAR_AVAILABLE = True
except ImportError:
    QUASAR_AVAILABLE = False

try:
    from quantum_amos import QuantumAmosAgent, QuantumAmosNetwork
    QUANTUM_AMOS_AVAILABLE = True
except ImportError:
    QUANTUM_AMOS_AVAILABLE = False

# Import messaging adapters
try:
    from pads_messaging_integration import integrate_pads_messaging
    from quasar_messaging_adapter import integrate_quasar_messaging
    from quantum_amos_messaging_adapter import integrate_quantum_amos_messaging
    MESSAGING_ADAPTERS_AVAILABLE = True
except ImportError:
    MESSAGING_ADAPTERS_AVAILABLE = False

# Import QBMIA with PADS connector
try:
    import sys
    sys.path.append('./qbmia')
    from qbmia.core.agent import QBMIAAgent
    from qbmia.integration.pads_connector import PADSConnector
    QBMIA_AVAILABLE = True
except ImportError:
    QBMIA_AVAILABLE = False

# Import unified messaging
try:
    from unified_messaging import AgentType, MessageType, Message
    UNIFIED_MESSAGING_AVAILABLE = True
except ImportError:
    UNIFIED_MESSAGING_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IntegratedQuantumSystem")

@dataclass
class SystemConfiguration:
    """Configuration for the integrated quantum trading system."""
    
    # Messaging configuration
    redis_url: str = "redis://localhost:6379"
    use_real_messaging: bool = True
    zmq_ports: Dict[str, int] = field(default_factory=lambda: {
        'pads': 9090,
        'qbmia': 9091,
        'quasar': 9092,
        'quantum_amos': 9093
    })
    
    # Agent configurations
    enable_pads: bool = True
    enable_qbmia: bool = True
    enable_quasar: bool = True
    enable_quantum_amos: bool = True
    
    # System settings
    decision_timeout: float = 5.0
    health_check_interval: float = 30.0
    auto_restart_failed: bool = True
    
    # Market data simulation
    simulate_market_data: bool = True
    market_update_interval: float = 1.0

class IntegratedQuantumTradingSystem:
    """
    Main orchestrator for the integrated quantum trading system.
    
    Coordinates PADS, QBMIA, QUASAR, and Quantum AMOS agents with
    real messaging for unified decision making.
    """
    
    def __init__(self, config: Optional[SystemConfiguration] = None):
        self.config = config or SystemConfiguration()
        
        # System state
        self.running = False
        self.agents = {}
        self.messaging_adapters = {}
        self.health_status = {}
        
        # Agent instances
        self.pads = None
        self.qbmia = None
        self.quasar = None
        self.quantum_amos = None
        
        # Background tasks
        self.background_tasks = []
        
        logger.info("Integrated Quantum Trading System initialized")
    
    async def initialize(self) -> bool:
        """Initialize all agents and messaging systems."""
        logger.info("Starting system initialization...")
        
        # Check dependencies
        if not self._check_dependencies():
            logger.error("Required dependencies not available")
            return False
        
        # Initialize agents
        success = True
        
        if self.config.enable_pads and PADS_AVAILABLE:
            success &= await self._initialize_pads()
        
        if self.config.enable_qbmia and QBMIA_AVAILABLE:
            success &= await self._initialize_qbmia()
        
        if self.config.enable_quasar and QUASAR_AVAILABLE:
            success &= await self._initialize_quasar()
        
        if self.config.enable_quantum_amos and QUANTUM_AMOS_AVAILABLE:
            success &= await self._initialize_quantum_amos()
        
        if not success:
            logger.error("Failed to initialize some agents")
            return False
        
        # Setup messaging between agents
        if MESSAGING_ADAPTERS_AVAILABLE and self.config.use_real_messaging:
            success &= await self._setup_messaging()
        
        if success:
            logger.info("System initialization completed successfully")
        else:
            logger.error("System initialization failed")
        
        return success
    
    def _check_dependencies(self) -> bool:
        """Check that required dependencies are available."""
        required = []
        
        if self.config.enable_pads and not PADS_AVAILABLE:
            required.append("PADS")
        
        if self.config.enable_qbmia and not QBMIA_AVAILABLE:
            required.append("QBMIA")
        
        if self.config.enable_quasar and not QUASAR_AVAILABLE:
            required.append("QUASAR")
        
        if self.config.enable_quantum_amos and not QUANTUM_AMOS_AVAILABLE:
            required.append("Quantum AMOS")
        
        if self.config.use_real_messaging and not UNIFIED_MESSAGING_AVAILABLE:
            required.append("Unified Messaging")
        
        if required:
            logger.error(f"Missing required components: {', '.join(required)}")
            return False
        
        return True
    
    async def _initialize_pads(self) -> bool:
        """Initialize PADS orchestrator."""
        try:
            logger.info("Initializing PADS (Panarchy Adaptive Decision System)...")
            
            # Create PADS instance with configuration
            pads_config = {
                'board_members': {
                    'qar': 0.25,        # QBMIA representation
                    'qstar': 0.25,     # QUASAR representation  
                    'antifragility': 0.20,  # Quantum AMOS representation
                    'consensus': 0.15,
                    'adaptability': 0.10,
                    'market_maker': 0.05
                },
                'decision_styles': ['analytical', 'intuitive', 'consensus', 'innovative'],
                'confidence_thresholds': {
                    'low': 0.3,
                    'medium': 0.6,
                    'high': 0.8
                }
            }
            
            self.pads = PanarchyAdaptiveDecisionSystem(
                config=pads_config,
                logger=logger.getChild("PADS")
            )
            
            # Initialize PADS
            if hasattr(self.pads, 'initialize'):
                await self.pads.initialize()
            
            self.agents['pads'] = self.pads
            self.health_status['pads'] = 'healthy'
            
            logger.info("PADS initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize PADS: {e}")
            return False
    
    async def _initialize_qbmia(self) -> bool:
        """Initialize QBMIA agent."""
        try:
            logger.info("Initializing QBMIA (Quantum-Biological Market Intuition Agent)...")
            
            # Create QBMIA configuration
            qbmia_config = {
                'agent_name': 'QBMIA_PRIMARY',
                'strategies': {
                    'quantum_nash': {'enabled': True, 'weight': 0.25},
                    'machiavellian': {'enabled': True, 'weight': 0.20},
                    'robin_hood': {'enabled': True, 'weight': 0.20},
                    'temporal_nash': {'enabled': True, 'weight': 0.20},
                    'antifragile': {'enabled': True, 'weight': 0.15}
                },
                'hardware_optimization': True,
                'memory_management': {'enabled': True, 'max_cache_size': 1000},
                'learning_rate': 0.01
            }
            
            self.qbmia = QBMIAAgent(
                config=qbmia_config,
                logger=logger.getChild("QBMIA")
            )
            
            # Initialize QBMIA
            if hasattr(self.qbmia, 'initialize'):
                await self.qbmia.initialize()
            
            self.agents['qbmia'] = self.qbmia
            self.health_status['qbmia'] = 'healthy'
            
            logger.info("QBMIA initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize QBMIA: {e}")
            return False
    
    async def _initialize_quasar(self) -> bool:
        """Initialize QUASAR system."""
        try:
            logger.info("Initializing QUASAR (Quantum Unified Star Agentic Reasoning)...")
            
            # Create QUASAR configuration
            quasar_config = {
                'qstar_config': {
                    'learning_rate': 0.01,
                    'exploration_rate': 0.1,
                    'memory_size': 10000
                },
                'qar_config': {
                    'risk_tolerance': 0.1,
                    'confidence_threshold': 0.6
                },
                'decision_threshold': 0.7,
                'integration_weights': {
                    'qstar': 0.6,
                    'qar': 0.4
                }
            }
            
            self.quasar = QUASAR(
                config=quasar_config,
                logger=logger.getChild("QUASAR")
            )
            
            # Initialize QUASAR
            if hasattr(self.quasar, 'initialize'):
                await self.quasar.initialize()
            
            self.agents['quasar'] = self.quasar
            self.health_status['quasar'] = 'healthy'
            
            logger.info("QUASAR initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize QUASAR: {e}")
            return False
    
    async def _initialize_quantum_amos(self) -> bool:
        """Initialize Quantum AMOS agent."""
        try:
            logger.info("Initializing Quantum AMOS (Quantum Hybrid CADM-BDIA Agent)...")
            
            # Create Quantum AMOS configuration
            amos_config = {
                'agent_name': 'QuantumAMOS_001',
                'desire_threshold': 0.7,
                'hardware_acceleration': True,
                'quantum_circuits': True,
                'network_mode': False  # Start with single agent
            }
            
            self.quantum_amos = QuantumAmosAgent(
                name=amos_config['agent_name'],
                config=amos_config,
                logger=logger.getChild("QuantumAMOS")
            )
            
            # Initialize Quantum AMOS
            if hasattr(self.quantum_amos, 'initialize'):
                await self.quantum_amos.initialize()
            
            self.agents['quantum_amos'] = self.quantum_amos
            self.health_status['quantum_amos'] = 'healthy'
            
            logger.info("Quantum AMOS initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Quantum AMOS: {e}")
            return False
    
    async def _setup_messaging(self) -> bool:
        """Setup messaging between all agents."""
        try:
            logger.info("Setting up inter-agent messaging...")
            
            messaging_config = {
                'redis_url': self.config.redis_url,
                'use_real_messaging': self.config.use_real_messaging,
                'zmq_ports': self.config.zmq_ports
            }
            
            # Setup PADS messaging (central coordinator)
            if self.pads:
                pads_integration = integrate_pads_messaging(self.pads, messaging_config)
                self.messaging_adapters['pads'] = pads_integration
                await pads_integration.connect()
            
            # Setup QBMIA messaging
            if self.qbmia:
                qbmia_connector = PADSConnector(self.qbmia, messaging_config)
                self.messaging_adapters['qbmia'] = qbmia_connector
                await qbmia_connector.connect_to_pads()
            
            # Setup QUASAR messaging
            if self.quasar:
                quasar_adapter = integrate_quasar_messaging(self.quasar, messaging_config)
                self.messaging_adapters['quasar'] = quasar_adapter
                await quasar_adapter.connect()
            
            # Setup Quantum AMOS messaging
            if self.quantum_amos:
                amos_adapter = integrate_quantum_amos_messaging(self.quantum_amos, messaging_config)
                self.messaging_adapters['quantum_amos'] = amos_adapter
                await amos_adapter.connect()
            
            logger.info("Inter-agent messaging setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup messaging: {e}")
            return False
    
    async def start(self) -> None:
        """Start the integrated system."""
        if not await self.initialize():
            raise RuntimeError("Failed to initialize system")
        
        logger.info("Starting Integrated Quantum Trading System...")
        self.running = True
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._health_monitor()),
            asyncio.create_task(self._market_simulation()) if self.config.simulate_market_data else None,
            asyncio.create_task(self._decision_coordinator())
        ]
        
        # Filter out None tasks
        self.background_tasks = [task for task in self.background_tasks if task is not None]
        
        logger.info("System started successfully")
    
    async def stop(self) -> None:
        """Stop the integrated system gracefully."""
        logger.info("Stopping Integrated Quantum Trading System...")
        self.running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Disconnect messaging
        for adapter in self.messaging_adapters.values():
            try:
                if hasattr(adapter, 'disconnect'):
                    await adapter.disconnect()
                elif hasattr(adapter, 'disconnect_from_pads'):
                    await adapter.disconnect_from_pads()
            except Exception as e:
                logger.error(f"Error disconnecting adapter: {e}")
        
        logger.info("System stopped")
    
    async def _health_monitor(self) -> None:
        """Monitor the health of all agents."""
        while self.running:
            try:
                for agent_name, agent in self.agents.items():
                    try:
                        # Check agent health
                        if hasattr(agent, 'get_status'):
                            status = agent.get_status()
                            if status.get('healthy', True):
                                self.health_status[agent_name] = 'healthy'
                            else:
                                self.health_status[agent_name] = 'unhealthy'
                                logger.warning(f"Agent {agent_name} is unhealthy: {status}")
                        else:
                            # Assume healthy if no status method
                            self.health_status[agent_name] = 'healthy'
                            
                    except Exception as e:
                        logger.error(f"Health check failed for {agent_name}: {e}")
                        self.health_status[agent_name] = 'error'
                
                # Log system health summary
                healthy = sum(1 for status in self.health_status.values() if status == 'healthy')
                total = len(self.health_status)
                logger.debug(f"System health: {healthy}/{total} agents healthy")
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(self.config.health_check_interval)
    
    async def _market_simulation(self) -> None:
        """Simulate market data updates for testing."""
        import random
        import math
        
        base_price = 50000  # Base BTC price
        time_step = 0
        
        while self.running:
            try:
                # Generate simulated market data
                time_step += 1
                price_change = math.sin(time_step * 0.1) * 1000 + random.gauss(0, 500)
                current_price = base_price + price_change
                
                market_data = {
                    'symbol': 'BTC/USDT',
                    'timestamp': datetime.utcnow().isoformat(),
                    'price': current_price,
                    'volume': random.uniform(100, 1000),
                    'volatility': abs(price_change) / base_price,
                    'trend': 'up' if price_change > 0 else 'down'
                }
                
                # Send market updates to all agents via PADS
                if self.pads and hasattr(self.pads, 'messaging_integration'):
                    from unified_messaging import Message, MessageType, AgentType
                    
                    market_message = Message(
                        message_type=MessageType.MARKET_UPDATE,
                        sender=AgentType.PADS,
                        recipient=None,  # Broadcast
                        data=market_data,
                        priority=2
                    )
                    
                    if hasattr(self.pads.messaging_integration, 'messenger'):
                        await self.pads.messaging_integration.messenger.send_message(market_message)
                
                await asyncio.sleep(self.config.market_update_interval)
                
            except Exception as e:
                logger.error(f"Market simulation error: {e}")
                await asyncio.sleep(self.config.market_update_interval)
    
    async def _decision_coordinator(self) -> None:
        """Coordinate decisions between agents."""
        while self.running:
            try:
                # Periodic decision coordination
                if self.pads and hasattr(self.pads, 'request_agent_decisions'):
                    # Create sample market data for decision request
                    market_data = {
                        'timestamp': datetime.utcnow().isoformat(),
                        'symbol': 'BTC/USDT',
                        'action_needed': True
                    }
                    
                    # Request decisions from all agents
                    agent_types = []
                    if 'qbmia' in self.agents:
                        agent_types.append(AgentType.QBMIA)
                    if 'quasar' in self.agents:
                        agent_types.append(AgentType.QUASAR)
                    if 'quantum_amos' in self.agents:
                        agent_types.append(AgentType.QUANTUM_AMOS)
                    
                    if agent_types:
                        responses = await self.pads.request_agent_decisions(
                            market_data=market_data,
                            agents=agent_types,
                            timeout=self.config.decision_timeout
                        )
                        
                        if responses:
                            logger.info(f"Received {len(responses)} agent decisions")
                            
                            # Make final PADS decision based on agent inputs
                            final_decision = self.pads.make_decision(market_data, {}, {})
                            if final_decision:
                                logger.info(f"PADS final decision: {final_decision.decision_type.name} (confidence: {final_decision.confidence:.2f})")
                
                # Wait before next coordination cycle
                await asyncio.sleep(10.0)  # Coordinate every 10 seconds
                
            except Exception as e:
                logger.error(f"Decision coordination error: {e}")
                await asyncio.sleep(10.0)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'running': self.running,
            'agents': {
                name: {
                    'available': name in self.agents,
                    'health': self.health_status.get(name, 'unknown'),
                    'connected': hasattr(self.messaging_adapters.get(name), 'connected') and 
                               self.messaging_adapters[name].connected if name in self.messaging_adapters else False
                }
                for name in ['pads', 'qbmia', 'quasar', 'quantum_amos']
            },
            'messaging': {
                'enabled': self.config.use_real_messaging,
                'adapters_count': len(self.messaging_adapters)
            },
            'background_tasks': len([t for t in self.background_tasks if not t.done()]),
            'timestamp': datetime.utcnow().isoformat()
        }

async def main():
    """Main entry point for the integrated system."""
    
    # Setup signal handlers for graceful shutdown
    system = None
    
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        if system:
            asyncio.create_task(system.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Load configuration
        config_file = os.environ.get('QUANTUM_SYSTEM_CONFIG', 'quantum_system_config.json')
        config = SystemConfiguration()
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    # Update config with loaded data
                    for key, value in config_data.items():
                        if hasattr(config, key):
                            setattr(config, key, value)
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.warning(f"Could not load config file {config_file}: {e}")
        
        # Create and start system
        system = IntegratedQuantumTradingSystem(config)
        
        logger.info("=" * 60)
        logger.info("INTEGRATED QUANTUM TRADING SYSTEM")
        logger.info("=" * 60)
        logger.info("Initializing quantum-biological decision agents...")
        
        await system.start()
        
        # Keep running until interrupted
        logger.info("System running. Press Ctrl+C to stop.")
        
        # Print status every 30 seconds
        while system.running:
            await asyncio.sleep(30)
            status = system.get_system_status()
            healthy_agents = sum(1 for agent in status['agents'].values() if agent['health'] == 'healthy')
            total_agents = len(status['agents'])
            logger.info(f"System Status: {healthy_agents}/{total_agents} agents healthy, {status['background_tasks']} tasks running")
    
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        if system:
            await system.stop()
        logger.info("System shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())