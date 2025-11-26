"""
MCP (Model Context Protocol) Server for AI News Trading Platform

This package provides a complete MCP server implementation with:
- JSON-RPC 2.0 protocol support
- WebSocket and HTTP transports
- Trading strategy integration
- GPU acceleration support
- Real-time market data streaming
"""

from .server import MCPServer
from .discovery import DiscoveryHandler
from .handlers import (
    ToolsHandler,
    ResourcesHandler,
    PromptsHandler,
    SamplingHandler
)
from .trading import StrategyManager, ModelLoader

__version__ = '1.0.0'

__all__ = [
    'MCPServer',
    'DiscoveryHandler',
    'ToolsHandler',
    'ResourcesHandler',
    'PromptsHandler',
    'SamplingHandler',
    'StrategyManager',
    'ModelLoader'
]