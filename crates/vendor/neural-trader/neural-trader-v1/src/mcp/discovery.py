"""
MCP Discovery and Registration Handler

Handles service discovery, capability advertisement, and client authentication
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import jwt
import secrets
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ServiceInfo:
    """Service information structure"""
    name: str
    version: str
    description: str
    endpoints: List[str]
    capabilities: Dict[str, Any]
    status: str = "active"
    registered_at: str = ""
    last_health_check: str = ""


class DiscoveryHandler:
    """Handles MCP service discovery and registration"""
    
    def __init__(self, server):
        self.server = server
        self.registered_services: Dict[str, ServiceInfo] = {}
        self.auth_tokens: Dict[str, Dict] = {}
        self.secret_key = secrets.token_urlsafe(32)
        
        # Register this server
        self._register_self()
        
    def _register_self(self):
        """Register the MCP server itself"""
        self.registered_services['ai-news-trader-mcp'] = ServiceInfo(
            name='ai-news-trader-mcp',
            version='1.0.0',
            description='MCP server for AI News Trading platform',
            endpoints=[
                f'http://{self.server.host}:{self.server.http_port}/mcp',
                f'ws://{self.server.host}:{self.server.ws_port}'
            ],
            capabilities={
                'tools': ['execute_trade', 'backtest', 'optimize'],
                'resources': ['model_parameters', 'market_data', 'strategy_configs'],
                'prompts': ['strategy_recommendation', 'risk_analysis'],
                'sampling': ['monte_carlo', 'historical_replay'],
                'strategies': [
                    'mirror_trader',
                    'momentum_trader',
                    'swing_trader',
                    'mean_reversion_trader'
                ]
            },
            registered_at=datetime.now().isoformat(),
            last_health_check=datetime.now().isoformat()
        )
    
    async def handle_discover(self, params: Dict) -> Dict:
        """
        Discover available MCP services
        
        Args:
            params: {
                'filter': Optional filter criteria
            }
        """
        filter_criteria = params.get('filter', {})
        services = []
        
        for service_id, service_info in self.registered_services.items():
            # Apply filters if provided
            if filter_criteria:
                if 'name' in filter_criteria and filter_criteria['name'] not in service_info.name:
                    continue
                if 'capabilities' in filter_criteria:
                    required_caps = filter_criteria['capabilities']
                    if not all(cap in service_info.capabilities for cap in required_caps):
                        continue
            
            services.append({
                'id': service_id,
                **asdict(service_info)
            })
        
        return {
            'services': services,
            'count': len(services),
            'timestamp': datetime.now().isoformat()
        }
    
    async def handle_register(self, params: Dict) -> Dict:
        """
        Register a new MCP service
        
        Args:
            params: Service registration information
        """
        try:
            service_info = ServiceInfo(**params)
            service_id = f"{service_info.name}-{secrets.token_hex(4)}"
            
            service_info.registered_at = datetime.now().isoformat()
            service_info.last_health_check = datetime.now().isoformat()
            
            self.registered_services[service_id] = service_info
            
            logger.info(f"Registered new service: {service_id}")
            
            return {
                'service_id': service_id,
                'status': 'registered',
                'message': 'Service registered successfully'
            }
            
        except Exception as e:
            logger.error(f"Service registration error: {str(e)}")
            raise ValueError(f"Invalid service registration: {str(e)}")
    
    async def handle_unregister(self, params: Dict) -> Dict:
        """
        Unregister an MCP service
        
        Args:
            params: {
                'service_id': Service ID to unregister
            }
        """
        service_id = params.get('service_id')
        
        if not service_id:
            raise ValueError("service_id is required")
        
        if service_id not in self.registered_services:
            raise ValueError(f"Service not found: {service_id}")
        
        del self.registered_services[service_id]
        
        logger.info(f"Unregistered service: {service_id}")
        
        return {
            'status': 'unregistered',
            'message': f'Service {service_id} unregistered successfully'
        }
    
    async def handle_health_check(self, params: Dict) -> Dict:
        """
        Perform health check on a service
        
        Args:
            params: {
                'service_id': Optional service ID (defaults to self)
            }
        """
        service_id = params.get('service_id', 'ai-news-trader-mcp')
        
        if service_id not in self.registered_services:
            return {
                'service_id': service_id,
                'status': 'unknown',
                'message': 'Service not found'
            }
        
        # Update last health check time
        self.registered_services[service_id].last_health_check = datetime.now().isoformat()
        
        # Check various health indicators
        health_status = {
            'service_id': service_id,
            'status': 'healthy',
            'checks': {
                'server': 'ok',
                'handlers': 'ok',
                'connections': len(self.server.websocket_connections),
                'gpu': 'available' if self.server.gpu_available else 'not available'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return health_status
    
    async def handle_authenticate(self, params: Dict) -> Dict:
        """
        Authenticate a client and issue JWT token
        
        Args:
            params: {
                'client_id': Client identifier
                'client_secret': Client secret (for service accounts)
            }
        """
        client_id = params.get('client_id')
        client_secret = params.get('client_secret')
        
        if not client_id:
            raise ValueError("client_id is required")
        
        # In production, verify client_secret against database
        # For now, we'll create a token for any valid client_id
        
        # Generate JWT token
        payload = {
            'client_id': client_id,
            'exp': datetime.utcnow() + timedelta(hours=24),
            'iat': datetime.utcnow(),
            'capabilities': ['tools', 'resources', 'prompts', 'sampling']
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        
        self.auth_tokens[token] = {
            'client_id': client_id,
            'issued_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(hours=24)).isoformat()
        }
        
        logger.info(f"Issued auth token for client: {client_id}")
        
        return {
            'token': token,
            'expires_in': 86400,  # 24 hours
            'token_type': 'Bearer',
            'capabilities': payload['capabilities']
        }
    
    async def handle_verify_token(self, params: Dict) -> Dict:
        """
        Verify an authentication token
        
        Args:
            params: {
                'token': JWT token to verify
            }
        """
        token = params.get('token')
        
        if not token:
            raise ValueError("token is required")
        
        try:
            # Decode and verify token
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            return {
                'valid': True,
                'client_id': payload['client_id'],
                'capabilities': payload.get('capabilities', []),
                'expires_at': datetime.fromtimestamp(payload['exp']).isoformat()
            }
            
        except jwt.ExpiredSignatureError:
            return {
                'valid': False,
                'error': 'Token expired'
            }
        except jwt.InvalidTokenError:
            return {
                'valid': False,
                'error': 'Invalid token'
            }
    
    async def handle_get_capabilities(self, params: Dict) -> Dict:
        """
        Get detailed capability information
        
        Args:
            params: {
                'service_id': Optional service ID (defaults to self)
            }
        """
        service_id = params.get('service_id', 'ai-news-trader-mcp')
        
        if service_id not in self.registered_services:
            raise ValueError(f"Service not found: {service_id}")
        
        service = self.registered_services[service_id]
        
        # Detailed capability information
        capabilities = {
            'service_id': service_id,
            'name': service.name,
            'version': service.version,
            'tools': {
                'execute_trade': {
                    'description': 'Execute trading orders',
                    'parameters': ['strategy', 'symbol', 'quantity', 'order_type'],
                    'requires_auth': True
                },
                'backtest': {
                    'description': 'Run historical backtesting',
                    'parameters': ['strategy', 'start_date', 'end_date', 'symbols'],
                    'requires_auth': True
                },
                'optimize': {
                    'description': 'Optimize strategy parameters',
                    'parameters': ['strategy', 'objective', 'constraints'],
                    'requires_auth': True
                }
            },
            'resources': {
                'model_parameters': {
                    'description': 'Access optimized model parameters',
                    'formats': ['json', 'binary'],
                    'strategies': service.capabilities.get('strategies', [])
                },
                'market_data': {
                    'description': 'Real-time and historical market data',
                    'types': ['quotes', 'trades', 'news'],
                    'streaming': True
                }
            },
            'prompts': {
                'strategy_recommendation': {
                    'description': 'Get AI-powered strategy recommendations',
                    'inputs': ['market_conditions', 'risk_profile', 'objectives']
                },
                'risk_analysis': {
                    'description': 'Analyze portfolio risk',
                    'inputs': ['positions', 'market_data', 'correlations']
                }
            },
            'sampling': {
                'monte_carlo': {
                    'description': 'Monte Carlo simulation for risk assessment',
                    'parameters': ['iterations', 'confidence_level']
                },
                'historical_replay': {
                    'description': 'Replay historical market scenarios',
                    'parameters': ['scenario', 'speed', 'symbols']
                }
            }
        }
        
        return capabilities