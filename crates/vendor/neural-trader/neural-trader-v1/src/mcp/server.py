"""
MCP (Model Context Protocol) Server for AI News Trading Platform

This server provides JSON-RPC 2.0 compliant endpoints for:
- Trading strategy execution
- Model parameter management
- Real-time market data streaming
- Backtesting and optimization
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime
import websockets
import aiohttp
from aiohttp import web
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum
import traceback
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPErrorCode(Enum):
    """Standard JSON-RPC 2.0 error codes"""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    
    # Custom MCP error codes
    MODEL_NOT_FOUND = -32001
    STRATEGY_ERROR = -32002
    DATA_ERROR = -32003
    AUTH_ERROR = -32004


@dataclass
class MCPRequest:
    """MCP request structure"""
    jsonrpc: str
    method: str
    params: Optional[Union[Dict, List]] = None
    id: Optional[Union[str, int]] = None


@dataclass
class MCPResponse:
    """MCP response structure"""
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[Dict] = None
    id: Optional[Union[str, int]] = None


@dataclass
class MCPError:
    """MCP error structure"""
    code: int
    message: str
    data: Optional[Any] = None


class MCPServer:
    """Main MCP Server implementation"""
    
    def __init__(self, host: str = "0.0.0.0", http_port: int = 8080, ws_port: int = 8081):
        self.host = host
        self.http_port = http_port
        self.ws_port = ws_port
        self.app = web.Application()
        self.handlers: Dict[str, Callable] = {}
        self.websocket_connections = set()
        self.session_data: Dict[str, Dict] = {}
        
        # GPU availability check
        self.gpu_available = self._check_gpu_availability()
        
        # Initialize handlers
        self._register_core_handlers()
        
        # Setup routes
        self._setup_routes()
        
    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            logger.warning("PyTorch not available - GPU acceleration disabled")
            return False
    
    def _register_core_handlers(self):
        """Register core MCP handlers"""
        from .handlers.tools import ToolsHandler
        from .handlers.resources import ResourcesHandler
        from .handlers.prompts import PromptsHandler
        from .handlers.sampling import SamplingHandler
        from .discovery import DiscoveryHandler
        
        # Initialize handlers
        self.tools_handler = ToolsHandler(self)
        self.resources_handler = ResourcesHandler(self)
        self.prompts_handler = PromptsHandler(self)
        self.sampling_handler = SamplingHandler(self)
        self.discovery_handler = DiscoveryHandler(self)
        
        # Register handler methods
        self._register_handler_methods(self.tools_handler)
        self._register_handler_methods(self.resources_handler)
        self._register_handler_methods(self.prompts_handler)
        self._register_handler_methods(self.sampling_handler)
        self._register_handler_methods(self.discovery_handler)
        
    def _register_handler_methods(self, handler):
        """Register all methods from a handler class"""
        for method_name in dir(handler):
            if method_name.startswith('handle_'):
                method = getattr(handler, method_name)
                rpc_method_name = method_name.replace('handle_', '')
                self.register_handler(rpc_method_name, method)
    
    def register_handler(self, method: str, handler: Callable):
        """Register a method handler"""
        self.handlers[method] = handler
        logger.info(f"Registered handler for method: {method}")
    
    def _setup_routes(self):
        """Setup HTTP routes"""
        self.app.router.add_post('/mcp', self.handle_http_request)
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/capabilities', self.get_capabilities)
        
    async def handle_http_request(self, request: web.Request) -> web.Response:
        """Handle HTTP JSON-RPC requests"""
        try:
            data = await request.json()
            response = await self.process_request(data)
            return web.json_response(response)
        except json.JSONDecodeError:
            error_response = MCPResponse(
                error=asdict(MCPError(
                    code=MCPErrorCode.PARSE_ERROR.value,
                    message="Parse error"
                ))
            )
            return web.json_response(asdict(error_response), status=400)
        except Exception as e:
            logger.error(f"HTTP request error: {str(e)}")
            error_response = MCPResponse(
                error=asdict(MCPError(
                    code=MCPErrorCode.INTERNAL_ERROR.value,
                    message="Internal error",
                    data=str(e)
                ))
            )
            return web.json_response(asdict(error_response), status=500)
    
    async def process_request(self, data: Dict) -> Dict:
        """Process a JSON-RPC request"""
        try:
            # Validate request structure
            if not isinstance(data, dict) or 'jsonrpc' not in data or data['jsonrpc'] != '2.0':
                raise ValueError("Invalid JSON-RPC 2.0 request")
            
            request = MCPRequest(**data)
            
            # Check if method exists
            if request.method not in self.handlers:
                return asdict(MCPResponse(
                    id=request.id,
                    error=asdict(MCPError(
                        code=MCPErrorCode.METHOD_NOT_FOUND.value,
                        message=f"Method not found: {request.method}"
                    ))
                ))
            
            # Execute handler
            handler = self.handlers[request.method]
            result = await handler(request.params or {})
            
            return asdict(MCPResponse(
                id=request.id,
                result=result
            ))
            
        except Exception as e:
            logger.error(f"Request processing error: {str(e)}\n{traceback.format_exc()}")
            return asdict(MCPResponse(
                id=data.get('id'),
                error=asdict(MCPError(
                    code=MCPErrorCode.INTERNAL_ERROR.value,
                    message="Internal error",
                    data=str(e)
                ))
            ))
    
    async def handle_websocket(self, websocket, path):
        """Handle WebSocket connections"""
        connection_id = str(uuid.uuid4())
        self.websocket_connections.add(websocket)
        self.session_data[connection_id] = {
            'connected_at': datetime.now().isoformat(),
            'subscriptions': set()
        }
        
        logger.info(f"WebSocket client connected: {connection_id}")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    response = await self.process_request(data)
                    await websocket.send(json.dumps(response))
                except json.JSONDecodeError:
                    error_response = MCPResponse(
                        error=asdict(MCPError(
                            code=MCPErrorCode.PARSE_ERROR.value,
                            message="Parse error"
                        ))
                    )
                    await websocket.send(json.dumps(asdict(error_response)))
                except Exception as e:
                    logger.error(f"WebSocket message error: {str(e)}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket client disconnected: {connection_id}")
        finally:
            self.websocket_connections.remove(websocket)
            del self.session_data[connection_id]
    
    async def broadcast_update(self, event_type: str, data: Any):
        """Broadcast updates to all connected WebSocket clients"""
        if self.websocket_connections:
            message = json.dumps({
                'jsonrpc': '2.0',
                'method': 'update',
                'params': {
                    'event_type': event_type,
                    'data': data,
                    'timestamp': datetime.now().isoformat()
                }
            })
            
            # Send to all connected clients
            await asyncio.gather(
                *[ws.send(message) for ws in self.websocket_connections],
                return_exceptions=True
            )
    
    async def health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint"""
        return web.json_response({
            'status': 'healthy',
            'version': '1.0.0',
            'gpu_available': self.gpu_available,
            'active_connections': len(self.websocket_connections),
            'timestamp': datetime.now().isoformat()
        })
    
    async def get_capabilities(self, request: web.Request) -> web.Response:
        """Get server capabilities"""
        return web.json_response({
            'capabilities': {
                'tools': True,
                'resources': True,
                'prompts': True,
                'sampling': True,
                'streaming': True,
                'gpu_acceleration': self.gpu_available,
                'supported_strategies': [
                    'mirror_trader',
                    'momentum_trader',
                    'swing_trader',
                    'mean_reversion_trader'
                ],
                'transport': ['http', 'websocket'],
                'version': '1.0.0'
            },
            'methods': list(self.handlers.keys())
        })
    
    async def start(self):
        """Start the MCP server"""
        # Start HTTP server
        runner = web.AppRunner(self.app)
        await runner.setup()
        http_site = web.TCPSite(runner, self.host, self.http_port)
        await http_site.start()
        
        logger.info(f"MCP HTTP server started on {self.host}:{self.http_port}")
        
        # Start WebSocket server
        ws_server = await websockets.serve(
            self.handle_websocket,
            self.host,
            self.ws_port
        )
        
        logger.info(f"MCP WebSocket server started on {self.host}:{self.ws_port}")
        logger.info(f"GPU acceleration: {'enabled' if self.gpu_available else 'disabled'}")
        
        # Keep servers running
        await asyncio.Future()


async def main():
    """Main entry point"""
    server = MCPServer()
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())