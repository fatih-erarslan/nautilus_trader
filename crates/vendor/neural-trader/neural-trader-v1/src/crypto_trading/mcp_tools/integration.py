"""
Integration module to connect Beefy Finance tools with the MCP server
"""

import logging
from typing import Dict, Any
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from crypto_trading.mcp_tools.beefy_tools import BeefyToolHandlers

logger = logging.getLogger(__name__)

# Initialize handlers
beefy_handlers = BeefyToolHandlers()

# Tool definitions for MCP server registration
BEEFY_TOOL_DEFINITIONS = [
    {
        "name": "beefy_get_vaults",
        "description": "Get available Beefy Finance vaults with advanced filtering",
        "parameters": {
            "type": "object",
            "properties": {
                "chain": {
                    "type": "string",
                    "enum": ["bsc", "polygon", "ethereum", "arbitrum", "optimism", "avalanche", "fantom"],
                    "description": "Filter by blockchain network"
                },
                "min_apy": {
                    "type": "number",
                    "description": "Minimum APY filter (e.g., 10 for 10%)"
                },
                "max_tvl": {
                    "type": "number",
                    "description": "Maximum TVL filter in USD"
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["apy", "tvl", "created", "name"],
                    "default": "apy",
                    "description": "Sort criteria"
                },
                "limit": {
                    "type": "integer",
                    "default": 50,
                    "description": "Number of results to return"
                }
            }
        }
    },
    {
        "name": "beefy_analyze_vault",
        "description": "Deep analysis of a specific Beefy vault with risk metrics",
        "parameters": {
            "type": "object",
            "properties": {
                "vault_id": {
                    "type": "string",
                    "description": "The vault ID to analyze"
                },
                "include_history": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include historical performance data"
                }
            },
            "required": ["vault_id"]
        }
    },
    {
        "name": "beefy_invest",
        "description": "Execute investment into a Beefy Finance vault",
        "parameters": {
            "type": "object",
            "properties": {
                "vault_id": {
                    "type": "string",
                    "description": "The vault ID to invest in"
                },
                "amount": {
                    "type": "number",
                    "description": "Amount to invest in USD"
                },
                "slippage": {
                    "type": "number",
                    "default": 0.01,
                    "description": "Maximum slippage tolerance (0.01 = 1%)"
                },
                "simulate": {
                    "type": "boolean",
                    "default": False,
                    "description": "Simulate without executing"
                }
            },
            "required": ["vault_id", "amount"]
        }
    },
    {
        "name": "beefy_harvest_yields",
        "description": "Harvest yields from Beefy vaults",
        "parameters": {
            "type": "object",
            "properties": {
                "vault_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Vault IDs to harvest from (empty for all)"
                },
                "auto_compound": {
                    "type": "boolean",
                    "default": True,
                    "description": "Auto-compound harvested yields"
                },
                "simulate": {
                    "type": "boolean",
                    "default": False,
                    "description": "Simulate without executing"
                }
            }
        }
    },
    {
        "name": "beefy_rebalance_portfolio",
        "description": "Rebalance Beefy portfolio based on strategy",
        "parameters": {
            "type": "object",
            "properties": {
                "strategy": {
                    "type": "string",
                    "enum": ["equal_weight", "risk_parity", "max_apy", "custom"],
                    "description": "Rebalancing strategy"
                },
                "target_allocations": {
                    "type": "object",
                    "description": "Target allocations for custom strategy (vault_id: percentage)"
                },
                "simulate": {
                    "type": "boolean",
                    "default": False,
                    "description": "Simulate without executing"
                }
            },
            "required": ["strategy"]
        }
    }
]

# Register tool handlers for MCP server
async def register_beefy_tools(mcp_server):
    """Register Beefy Finance tools with the MCP server"""
    
    @mcp_server.tool()
    async def beefy_get_vaults(
        chain: str = None,
        min_apy: float = None,
        max_tvl: float = None,
        sort_by: str = "apy",
        limit: int = 50
    ) -> Dict[str, Any]:
        """Get available Beefy Finance vaults with advanced filtering"""
        return await beefy_handlers.handle_beefy_get_vaults({
            "chain": chain,
            "min_apy": min_apy,
            "max_tvl": max_tvl,
            "sort_by": sort_by,
            "limit": limit
        })
    
    @mcp_server.tool()
    async def beefy_analyze_vault(
        vault_id: str,
        include_history: bool = True
    ) -> Dict[str, Any]:
        """Deep analysis of a specific Beefy vault with risk metrics"""
        return await beefy_handlers.handle_beefy_analyze_vault({
            "vault_id": vault_id,
            "include_history": include_history
        })
    
    @mcp_server.tool()
    async def beefy_invest(
        vault_id: str,
        amount: float,
        slippage: float = 0.01,
        simulate: bool = False
    ) -> Dict[str, Any]:
        """Execute investment into a Beefy Finance vault"""
        return await beefy_handlers.handle_beefy_invest({
            "vault_id": vault_id,
            "amount": amount,
            "slippage": slippage,
            "simulate": simulate
        })
    
    @mcp_server.tool()
    async def beefy_harvest_yields(
        vault_ids: list = None,
        auto_compound: bool = True,
        simulate: bool = False
    ) -> Dict[str, Any]:
        """Harvest yields from Beefy vaults"""
        return await beefy_handlers.handle_beefy_harvest_yields({
            "vault_ids": vault_ids,
            "auto_compound": auto_compound,
            "simulate": simulate
        })
    
    @mcp_server.tool()
    async def beefy_rebalance_portfolio(
        strategy: str,
        target_allocations: Dict[str, float] = None,
        simulate: bool = False
    ) -> Dict[str, Any]:
        """Rebalance Beefy portfolio based on strategy"""
        return await beefy_handlers.handle_beefy_rebalance_portfolio({
            "strategy": strategy,
            "target_allocations": target_allocations,
            "simulate": simulate
        })
    
    logger.info("Beefy Finance tools registered successfully")

# Cleanup function
async def cleanup_beefy_tools():
    """Cleanup resources when shutting down"""
    await beefy_handlers.close()
    logger.info("Beefy Finance tools cleaned up")