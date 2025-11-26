"""
Beefy Finance MCP Tool Handlers
Real-time yield farming integration with WebSocket support
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from handlers.vault_handler import VaultHandler
from handlers.investment_handler import InvestmentHandler
from handlers.portfolio_handler import PortfolioHandler
from handlers.analytics_handler import AnalyticsHandler
from schemas import (
    GetVaultsInput, GetVaultsOutput,
    AnalyzeVaultInput, AnalyzeVaultOutput,
    InvestInput, InvestOutput,
    HarvestInput, HarvestOutput,
    RebalanceInput, RebalanceOutput
)

logger = logging.getLogger(__name__)

class BeefyToolHandlers:
    """MCP tool handlers for Beefy Finance integration"""
    
    def __init__(self, db_path: str = "./beefy_trading.db"):
        """Initialize all handler modules"""
        self.vault_handler = VaultHandler()
        self.investment_handler = InvestmentHandler(db_path)
        self.portfolio_handler = PortfolioHandler(db_path)
        self.analytics_handler = AnalyticsHandler(db_path)
        
        # WebSocket connections for real-time APY updates
        self.ws_connections = {}
        
    async def handle_beefy_get_vaults(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get available Beefy vaults with filtering"""
        try:
            # Validate input
            input_data = GetVaultsInput(**params)
            
            # Get vaults from API
            vaults = await self.vault_handler.get_vaults(
                chain=input_data.chain,
                min_apy=input_data.min_apy,
                max_tvl=input_data.max_tvl,
                sort_by=input_data.sort_by,
                limit=input_data.limit
            )
            
            # Format response
            output = GetVaultsOutput(
                vaults=vaults,
                total_count=len(vaults),
                timestamp=datetime.now()
            )
            
            return output.dict()
            
        except Exception as e:
            logger.error(f"Error in get_vaults: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def handle_beefy_analyze_vault(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Deep analysis of a specific vault"""
        try:
            # Validate input
            input_data = AnalyzeVaultInput(**params)
            
            # Get vault analysis
            analysis = await self.vault_handler.analyze_vault(
                vault_id=input_data.vault_id,
                include_history=input_data.include_history
            )
            
            # Get risk metrics
            risk_metrics = await self.analytics_handler.calculate_vault_risk(
                vault_id=input_data.vault_id
            )
            
            # Combine results
            output = AnalyzeVaultOutput(
                vault_id=input_data.vault_id,
                analysis=analysis,
                risk_metrics=risk_metrics,
                timestamp=datetime.now()
            )
            
            return output.dict()
            
        except Exception as e:
            logger.error(f"Error in analyze_vault: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def handle_beefy_invest(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute investment into a Beefy vault"""
        try:
            # Validate input
            input_data = InvestInput(**params)
            
            # Prepare transaction
            tx_data = await self.investment_handler.prepare_investment(
                vault_id=input_data.vault_id,
                amount=input_data.amount,
                slippage=input_data.slippage
            )
            
            # Execute if not simulation
            if not input_data.simulate:
                result = await self.investment_handler.execute_investment(tx_data)
                
                # Store position in database
                await self.portfolio_handler.add_position(
                    vault_id=input_data.vault_id,
                    amount=input_data.amount,
                    tx_hash=result['tx_hash']
                )
            else:
                result = {
                    "simulation": True,
                    "estimated_shares": tx_data['estimated_shares'],
                    "estimated_gas": tx_data['gas_estimate']
                }
            
            output = InvestOutput(
                vault_id=input_data.vault_id,
                amount_invested=input_data.amount,
                result=result,
                timestamp=datetime.now()
            )
            
            return output.dict()
            
        except Exception as e:
            logger.error(f"Error in invest: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def handle_beefy_harvest_yields(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Harvest yields from vaults"""
        try:
            # Validate input
            input_data = HarvestInput(**params)
            
            # Get harvestable yields
            yields = await self.portfolio_handler.get_harvestable_yields(
                vault_ids=input_data.vault_ids
            )
            
            # Execute harvest if not simulation
            if not input_data.simulate:
                results = []
                for vault_id, yield_info in yields.items():
                    if yield_info['harvestable'] > 0:
                        tx_result = await self.investment_handler.harvest_yield(
                            vault_id=vault_id,
                            auto_compound=input_data.auto_compound
                        )
                        results.append({
                            "vault_id": vault_id,
                            "harvested": yield_info['harvestable'],
                            "tx_hash": tx_result['tx_hash']
                        })
                        
                        # Update database
                        await self.portfolio_handler.record_harvest(
                            vault_id=vault_id,
                            amount=yield_info['harvestable'],
                            tx_hash=tx_result['tx_hash']
                        )
            else:
                results = [
                    {
                        "vault_id": vid,
                        "harvestable": info['harvestable'],
                        "simulation": True
                    }
                    for vid, info in yields.items()
                ]
            
            output = HarvestOutput(
                yields_harvested=results,
                total_harvested=sum(r.get('harvested', r.get('harvestable', 0)) for r in results),
                timestamp=datetime.now()
            )
            
            return output.dict()
            
        except Exception as e:
            logger.error(f"Error in harvest_yields: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def handle_beefy_rebalance_portfolio(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Rebalance portfolio based on strategy"""
        try:
            # Validate input
            input_data = RebalanceInput(**params)
            
            # Get current portfolio
            current_positions = await self.portfolio_handler.get_all_positions()
            
            # Calculate optimal allocation
            rebalance_plan = await self.analytics_handler.calculate_rebalance(
                current_positions=current_positions,
                target_allocations=input_data.target_allocations,
                strategy=input_data.strategy
            )
            
            # Execute rebalance if not simulation
            if not input_data.simulate:
                results = []
                for action in rebalance_plan['actions']:
                    if action['type'] == 'withdraw':
                        tx_result = await self.investment_handler.withdraw(
                            vault_id=action['vault_id'],
                            amount=action['amount']
                        )
                    else:  # deposit
                        tx_result = await self.investment_handler.execute_investment({
                            'vault_id': action['vault_id'],
                            'amount': action['amount']
                        })
                    
                    results.append({
                        "action": action,
                        "tx_hash": tx_result['tx_hash']
                    })
                    
                    # Update database
                    await self.portfolio_handler.update_position(
                        vault_id=action['vault_id'],
                        action=action
                    )
            else:
                results = rebalance_plan['actions']
            
            output = RebalanceOutput(
                rebalance_actions=results,
                new_allocations=rebalance_plan['new_allocations'],
                timestamp=datetime.now()
            )
            
            return output.dict()
            
        except Exception as e:
            logger.error(f"Error in rebalance_portfolio: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def start_apy_websocket(self, vault_ids: List[str]):
        """Start WebSocket connections for real-time APY updates"""
        try:
            for vault_id in vault_ids:
                if vault_id not in self.ws_connections:
                    ws = await self.vault_handler.connect_websocket(vault_id)
                    self.ws_connections[vault_id] = ws
                    
                    # Start listening for updates
                    async def handle_apy_update(data):
                        await self.analytics_handler.process_apy_update(
                            vault_id=vault_id,
                            apy=data['apy'],
                            timestamp=data['timestamp']
                        )
                    
                    ws.on_message = handle_apy_update
                    
            return {"status": "websocket_started", "vaults": vault_ids}
            
        except Exception as e:
            logger.error(f"Error starting WebSocket: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def close(self):
        """Close all connections and resources"""
        # Close WebSocket connections
        for ws in self.ws_connections.values():
            await ws.close()
        
        # Close database connections
        await self.portfolio_handler.close()
        await self.analytics_handler.close()