"""
MCP Resources Handler

Handles access to model parameters, market data, and strategy configurations
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class ResourcesHandler:
    """Handles MCP resource operations"""
    
    def __init__(self, server):
        self.server = server
        self.resource_cache: Dict[str, Any] = {}
        self.resource_base_path = Path("/workspaces/ai-news-trader")
        
        # Load optimized parameters
        self._load_optimized_parameters()
        
    def _load_optimized_parameters(self):
        """Load optimized strategy parameters from files"""
        try:
            # Load mirror trader optimization results
            optimization_file = self.resource_base_path / "optimization_results.json"
            if optimization_file.exists():
                with open(optimization_file, 'r') as f:
                    self.resource_cache['mirror_trader_params'] = json.load(f)
                    logger.info("Loaded mirror trader optimized parameters")
            
            # Load other strategy parameters if available
            strategy_files = {
                'momentum_trader': 'momentum_optimization_results.json',
                'swing_trader': 'swing_optimization_results.json',
                'mean_reversion_trader': 'mean_reversion_optimization_results.json'
            }
            
            for strategy, filename in strategy_files.items():
                filepath = self.resource_base_path / filename
                if filepath.exists():
                    with open(filepath, 'r') as f:
                        self.resource_cache[f'{strategy}_params'] = json.load(f)
                        logger.info(f"Loaded {strategy} optimized parameters")
                        
        except Exception as e:
            logger.error(f"Error loading optimized parameters: {str(e)}")
    
    async def handle_list_resources(self, params: Dict) -> Dict:
        """List available resources"""
        resources = []
        
        # Model parameters resources
        for strategy in ['mirror_trader', 'momentum_trader', 'swing_trader', 'mean_reversion_trader']:
            resources.append({
                'uri': f'mcp://parameters/{strategy}',
                'name': f'{strategy}_parameters',
                'description': f'Optimized parameters for {strategy.replace("_", " ").title()}',
                'mimeType': 'application/json',
                'metadata': {
                    'optimized': True,
                    'last_updated': datetime.now().isoformat()
                }
            })
        
        # Strategy configuration resources
        resources.append({
            'uri': 'mcp://config/strategies',
            'name': 'strategy_configurations',
            'description': 'Configuration for all available trading strategies',
            'mimeType': 'application/json'
        })
        
        # Market data resources
        resources.append({
            'uri': 'mcp://data/market',
            'name': 'market_data',
            'description': 'Real-time and historical market data',
            'mimeType': 'application/json',
            'metadata': {
                'streaming': True,
                'data_types': ['quotes', 'trades', 'news']
            }
        })
        
        # Model state resources
        resources.append({
            'uri': 'mcp://state/models',
            'name': 'model_states',
            'description': 'Current state of all trading models',
            'mimeType': 'application/json'
        })
        
        return {
            'resources': resources,
            'count': len(resources)
        }
    
    async def handle_read_resource(self, params: Dict) -> Dict:
        """Read a specific resource"""
        uri = params.get('uri')
        
        if not uri:
            raise ValueError("Resource URI is required")
        
        # Parse URI
        if not uri.startswith('mcp://'):
            raise ValueError("Invalid MCP URI format")
        
        uri_parts = uri[6:].split('/')  # Remove 'mcp://' prefix
        resource_type = uri_parts[0]
        resource_name = uri_parts[1] if len(uri_parts) > 1 else None
        
        # Route to appropriate handler
        if resource_type == 'parameters':
            return await self._read_parameters(resource_name)
        elif resource_type == 'config':
            return await self._read_config(resource_name)
        elif resource_type == 'data':
            return await self._read_data(resource_name)
        elif resource_type == 'state':
            return await self._read_state(resource_name)
        else:
            raise ValueError(f"Unknown resource type: {resource_type}")
    
    async def _read_parameters(self, strategy: str) -> Dict:
        """Read strategy parameters"""
        cache_key = f'{strategy}_params'
        
        if cache_key in self.resource_cache:
            params = self.resource_cache[cache_key]
            return {
                'contents': [
                    {
                        'uri': f'mcp://parameters/{strategy}',
                        'mimeType': 'application/json',
                        'text': json.dumps(params, indent=2)
                    }
                ]
            }
        
        # Default parameters if not optimized
        default_params = self._get_default_parameters(strategy)
        return {
            'contents': [
                {
                    'uri': f'mcp://parameters/{strategy}',
                    'mimeType': 'application/json',
                    'text': json.dumps(default_params, indent=2),
                    'metadata': {
                        'default': True,
                        'message': 'Using default parameters - no optimization results found'
                    }
                }
            ]
        }
    
    async def _read_config(self, config_type: str) -> Dict:
        """Read configuration"""
        if config_type == 'strategies':
            strategies_config = {
                'available_strategies': [
                    {
                        'name': 'mirror_trader',
                        'description': 'Mirror institutional investor trades',
                        'optimized': 'mirror_trader_params' in self.resource_cache,
                        'risk_level': 'medium',
                        'time_horizon': 'medium-long',
                        'min_capital': 10000
                    },
                    {
                        'name': 'momentum_trader',
                        'description': 'Follow price momentum and trends',
                        'optimized': 'momentum_trader_params' in self.resource_cache,
                        'risk_level': 'high',
                        'time_horizon': 'short-medium',
                        'min_capital': 5000
                    },
                    {
                        'name': 'swing_trader',
                        'description': 'Capture short-term price swings',
                        'optimized': 'swing_trader_params' in self.resource_cache,
                        'risk_level': 'medium-high',
                        'time_horizon': 'short',
                        'min_capital': 5000
                    },
                    {
                        'name': 'mean_reversion_trader',
                        'description': 'Trade on price mean reversion',
                        'optimized': 'mean_reversion_trader_params' in self.resource_cache,
                        'risk_level': 'medium',
                        'time_horizon': 'short-medium',
                        'min_capital': 10000
                    }
                ],
                'global_settings': {
                    'max_positions': 10,
                    'max_position_size': 0.2,  # 20% of portfolio
                    'stop_loss_default': 0.05,  # 5%
                    'take_profit_default': 0.15,  # 15%
                    'rebalance_frequency': 'weekly'
                }
            }
            
            return {
                'contents': [
                    {
                        'uri': 'mcp://config/strategies',
                        'mimeType': 'application/json',
                        'text': json.dumps(strategies_config, indent=2)
                    }
                ]
            }
        
        raise ValueError(f"Unknown config type: {config_type}")
    
    async def _read_data(self, data_type: str) -> Dict:
        """Read market data"""
        if data_type == 'market':
            # This would connect to real market data sources
            # For now, return sample structure
            market_data = {
                'timestamp': datetime.now().isoformat(),
                'quotes': {
                    'AAPL': {'bid': 189.45, 'ask': 189.47, 'last': 189.46},
                    'GOOGL': {'bid': 141.23, 'ask': 141.25, 'last': 141.24},
                    'MSFT': {'bid': 377.89, 'ask': 377.91, 'last': 377.90}
                },
                'market_status': 'open',
                'next_update': (datetime.now().timestamp() + 1) * 1000  # 1 second
            }
            
            return {
                'contents': [
                    {
                        'uri': 'mcp://data/market',
                        'mimeType': 'application/json',
                        'text': json.dumps(market_data, indent=2),
                        'metadata': {
                            'streaming': True,
                            'update_frequency': 1000  # milliseconds
                        }
                    }
                ]
            }
        
        raise ValueError(f"Unknown data type: {data_type}")
    
    async def _read_state(self, state_type: str) -> Dict:
        """Read model/system state"""
        if state_type == 'models':
            # Get state from strategy manager
            strategy_manager = await self.server.tools_handler._get_strategy_manager()
            model_states = await strategy_manager.get_all_model_states()
            
            return {
                'contents': [
                    {
                        'uri': 'mcp://state/models',
                        'mimeType': 'application/json',
                        'text': json.dumps(model_states, indent=2)
                    }
                ]
            }
        
        raise ValueError(f"Unknown state type: {state_type}")
    
    def _get_default_parameters(self, strategy: str) -> Dict:
        """Get default parameters for a strategy"""
        defaults = {
            'mirror_trader': {
                'berkshire_confidence': 0.7,
                'bridgewater_confidence': 0.8,
                'renaissance_confidence': 0.9,
                'max_position_pct': 0.05,
                'stop_loss_threshold': -0.1,
                'take_profit_threshold': 0.2
            },
            'momentum_trader': {
                'lookback_period': 20,
                'momentum_threshold': 0.02,
                'volume_filter': 1.5,
                'rsi_overbought': 70,
                'rsi_oversold': 30
            },
            'swing_trader': {
                'swing_threshold': 0.03,
                'support_resistance_lookback': 50,
                'breakout_volume_multiplier': 2.0,
                'atr_multiplier': 2.0
            },
            'mean_reversion_trader': {
                'lookback_period': 20,
                'std_dev_threshold': 2.0,
                'mean_reversion_speed': 0.5,
                'min_volatility': 0.01
            }
        }
        
        return defaults.get(strategy, {})
    
    async def handle_resource_updated(self, params: Dict) -> Dict:
        """Handle resource update notifications"""
        uri = params.get('uri')
        
        if not uri:
            raise ValueError("Resource URI is required")
        
        # Broadcast update to WebSocket clients
        await self.server.broadcast_update('resource_updated', {
            'uri': uri,
            'timestamp': datetime.now().isoformat()
        })
        
        return {
            'status': 'acknowledged',
            'uri': uri
        }
    
    async def handle_subscribe_resource(self, params: Dict) -> Dict:
        """Subscribe to resource updates"""
        uri = params.get('uri')
        
        if not uri:
            raise ValueError("Resource URI is required")
        
        # Add to subscription list (implementation depends on transport)
        # For WebSocket connections, this would be tracked per connection
        
        return {
            'status': 'subscribed',
            'uri': uri,
            'message': 'You will receive updates for this resource'
        }