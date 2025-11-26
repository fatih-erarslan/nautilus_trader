"""
Trading Bots Client
==================

Python client for managing trading bots with Supabase persistence.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from uuid import UUID
from decimal import Decimal

from ..client import AsyncSupabaseClient, SupabaseError
from ..models.database_models import (
    TradingBot, BotExecution, SandboxDeployment, TradingAccount,
    CreateBotRequest, UpdateBotRequest, BotStatus
)

logger = logging.getLogger(__name__)

class TradingBotsClient:
    """
    Client for managing trading bots and their executions.
    """
    
    def __init__(self, supabase_client: AsyncSupabaseClient):
        """
        Initialize trading bots client.
        
        Args:
            supabase_client: Async Supabase client instance
        """
        self.client = supabase_client
    
    async def create_bot(
        self, 
        user_id: UUID, 
        bot_data: CreateBotRequest
    ) -> Tuple[Optional[TradingBot], Optional[str]]:
        """
        Create a new trading bot.
        
        Args:
            user_id: ID of the user creating the bot
            bot_data: Bot creation request data
            
        Returns:
            Tuple of (created bot, error message if any)
        """
        try:
            # Verify account exists and belongs to user
            account_result = await self.client.select(
                "trading_accounts",
                "id",
                filter_dict={
                    "id": str(bot_data.account_id),
                    "user_id": str(user_id)
                },
                limit=1
            )
            
            if not account_result:
                return None, "Trading account not found or access denied"
            
            data = {
                "user_id": str(user_id),
                "account_id": str(bot_data.account_id),
                "name": bot_data.name,
                "strategy_type": bot_data.strategy_type,
                "configuration": bot_data.configuration,
                "model_ids": [str(mid) for mid in bot_data.model_ids],
                "symbols": bot_data.symbols,
                "max_position_size": float(bot_data.max_position_size),
                "risk_limit": float(bot_data.risk_limit),
                "status": BotStatus.PAUSED.value,
                "performance_metrics": {}
            }
            
            result = await self.client.insert("trading_bots", data)
            
            if result:
                bot = TradingBot.from_db(result[0])
                logger.info(f"Created trading bot: {bot.id}")
                return bot, None
            else:
                return None, "Failed to create bot"
                
        except Exception as e:
            logger.error(f"Error creating trading bot: {e}")
            return None, str(e)
    
    async def get_user_bots(
        self,
        user_id: UUID,
        status: Optional[BotStatus] = None,
        account_id: Optional[UUID] = None,
        strategy_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[TradingBot], Optional[str]]:
        """
        Get trading bots for a user with optional filters.
        
        Args:
            user_id: ID of the user
            status: Optional status filter
            account_id: Optional account filter
            strategy_type: Optional strategy type filter
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            Tuple of (list of bots, error message if any)
        """
        try:
            filters = {"user_id": str(user_id)}
            
            if status:
                filters["status"] = status.value
            
            if account_id:
                filters["account_id"] = str(account_id)
                
            if strategy_type:
                filters["strategy_type"] = strategy_type
            
            result = await self.client.select(
                "trading_bots",
                "*",
                filter_dict=filters,
                order_by="-created_at",
                limit=limit,
                offset=offset
            )
            
            bots = [TradingBot.from_db(record) for record in result]
            logger.debug(f"Retrieved {len(bots)} bots for user {user_id}")
            return bots, None
            
        except Exception as e:
            logger.error(f"Error retrieving bots: {e}")
            return [], str(e)
    
    async def get_bot_by_id(
        self, 
        bot_id: UUID, 
        user_id: Optional[UUID] = None
    ) -> Tuple[Optional[TradingBot], Optional[str]]:
        """
        Get a specific bot by ID.
        
        Args:
            bot_id: ID of the bot
            user_id: Optional user ID for access control
            
        Returns:
            Tuple of (bot, error message if any)
        """
        try:
            filters = {"id": str(bot_id)}
            
            if user_id:
                filters["user_id"] = str(user_id)
            
            result = await self.client.select(
                "trading_bots",
                "*",
                filter_dict=filters,
                limit=1
            )
            
            if result:
                bot = TradingBot.from_db(result[0])
                return bot, None
            else:
                return None, "Bot not found"
                
        except Exception as e:
            logger.error(f"Error retrieving bot {bot_id}: {e}")
            return None, str(e)
    
    async def update_bot(
        self, 
        bot_id: UUID, 
        updates: UpdateBotRequest,
        user_id: Optional[UUID] = None
    ) -> Tuple[Optional[TradingBot], Optional[str]]:
        """
        Update a trading bot.
        
        Args:
            bot_id: ID of the bot
            updates: Update request data
            user_id: Optional user ID for access control
            
        Returns:
            Tuple of (updated bot, error message if any)
        """
        try:
            # Build update data
            update_data = {"updated_at": datetime.utcnow().isoformat()}
            
            if updates.name is not None:
                update_data["name"] = updates.name
            if updates.configuration is not None:
                update_data["configuration"] = updates.configuration
            if updates.model_ids is not None:
                update_data["model_ids"] = [str(mid) for mid in updates.model_ids]
            if updates.symbols is not None:
                update_data["symbols"] = updates.symbols
            if updates.max_position_size is not None:
                update_data["max_position_size"] = float(updates.max_position_size)
            if updates.risk_limit is not None:
                update_data["risk_limit"] = float(updates.risk_limit)
            if updates.status is not None:
                update_data["status"] = updates.status.value
            
            # Set up filters
            filters = {"id": str(bot_id)}
            if user_id:
                filters["user_id"] = str(user_id)
            
            result = await self.client.update("trading_bots", update_data, filters)
            
            if result:
                bot = TradingBot.from_db(result[0])
                logger.info(f"Updated trading bot: {bot.id}")
                return bot, None
            else:
                return None, "Failed to update bot or bot not found"
                
        except Exception as e:
            logger.error(f"Error updating bot: {e}")
            return None, str(e)
    
    async def set_bot_status(
        self, 
        bot_id: UUID, 
        status: BotStatus,
        user_id: Optional[UUID] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Update bot status.
        
        Args:
            bot_id: ID of the bot
            status: New status
            user_id: Optional user ID for access control
            
        Returns:
            Tuple of (success, error message if any)
        """
        try:
            filters = {"id": str(bot_id)}
            if user_id:
                filters["user_id"] = str(user_id)
            
            result = await self.client.update(
                "trading_bots",
                {
                    "status": status.value,
                    "updated_at": datetime.utcnow().isoformat()
                },
                filters
            )
            
            if result:
                logger.info(f"Updated bot {bot_id} status to {status.value}")
                return True, None
            else:
                return False, "Failed to update bot status or bot not found"
                
        except Exception as e:
            logger.error(f"Error updating bot status: {e}")
            return False, str(e)
    
    async def record_execution(
        self,
        bot_id: UUID,
        symbol_id: UUID,
        action: str,
        signal_strength: Optional[float] = None,
        reasoning: Optional[str] = None,
        order_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[BotExecution], Optional[str]]:
        """
        Record a bot execution.
        
        Args:
            bot_id: ID of the bot
            symbol_id: ID of the symbol
            action: Action taken ('buy', 'sell', 'hold')
            signal_strength: Strength of the signal (0.0 to 1.0)
            reasoning: Reasoning for the action
            order_id: ID of related order if any
            metadata: Additional metadata
            
        Returns:
            Tuple of (execution record, error message if any)
        """
        try:
            data = {
                "bot_id": str(bot_id),
                "symbol_id": str(symbol_id),
                "action": action,
                "metadata": metadata or {}
            }
            
            if signal_strength is not None:
                data["signal_strength"] = signal_strength
            if reasoning is not None:
                data["reasoning"] = reasoning
            if order_id is not None:
                data["order_id"] = str(order_id)
            
            result = await self.client.insert("bot_executions", data)
            
            if result:
                execution = BotExecution.from_db(result[0])
                logger.debug(f"Recorded bot execution: {execution.id}")
                return execution, None
            else:
                return None, "Failed to record execution"
                
        except Exception as e:
            logger.error(f"Error recording execution: {e}")
            return None, str(e)
    
    async def get_bot_executions(
        self,
        bot_id: UUID,
        symbol_id: Optional[UUID] = None,
        action: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[BotExecution], Optional[str]]:
        """
        Get execution history for a bot.
        
        Args:
            bot_id: ID of the bot
            symbol_id: Optional symbol filter
            action: Optional action filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            Tuple of (list of executions, error message if any)
        """
        try:
            filters = {"bot_id": str(bot_id)}
            
            if symbol_id:
                filters["symbol_id"] = str(symbol_id)
            if action:
                filters["action"] = action
            
            # Note: Date filtering would need custom SQL for proper range queries
            result = await self.client.select(
                "bot_executions",
                "*",
                filter_dict=filters,
                order_by="-executed_at",
                limit=limit,
                offset=offset
            )
            
            executions = [BotExecution.from_db(record) for record in result]
            logger.debug(f"Retrieved {len(executions)} executions for bot {bot_id}")
            return executions, None
            
        except Exception as e:
            logger.error(f"Error retrieving executions: {e}")
            return [], str(e)
    
    async def calculate_bot_performance(
        self,
        bot_id: UUID,
        time_range_hours: int = 24
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Calculate bot performance metrics.
        
        Args:
            bot_id: ID of the bot
            time_range_hours: Time range for calculation in hours
            
        Returns:
            Tuple of (performance metrics, error message if any)
        """
        try:
            # Get bot details
            bot, error = await self.get_bot_by_id(bot_id)
            if error or not bot:
                return {}, error or "Bot not found"
            
            # Calculate time range
            start_time = datetime.utcnow() - timedelta(hours=time_range_hours)
            
            # Get executions in time range (simplified - would need better date filtering)
            executions, error = await self.get_bot_executions(
                bot_id, limit=1000  # Large limit to get recent executions
            )
            
            if error:
                return {}, error
            
            # Filter executions by time (since we can't do it in query)
            recent_executions = [
                ex for ex in executions 
                if ex.executed_at >= start_time
            ]
            
            # Calculate metrics
            total_executions = len(recent_executions)
            buy_executions = len([ex for ex in recent_executions if ex.action == 'buy'])
            sell_executions = len([ex for ex in recent_executions if ex.action == 'sell'])
            hold_decisions = len([ex for ex in recent_executions if ex.action == 'hold'])
            
            # Calculate average signal strength
            signals_with_strength = [
                ex for ex in recent_executions 
                if ex.signal_strength is not None
            ]
            avg_signal_strength = (
                sum(ex.signal_strength for ex in signals_with_strength) / len(signals_with_strength)
                if signals_with_strength else 0
            )
            
            # Calculate execution rate (non-hold actions)
            execution_rate = (
                (buy_executions + sell_executions) / total_executions
                if total_executions > 0 else 0
            )
            
            metrics = {
                "total_executions": total_executions,
                "buy_executions": buy_executions,
                "sell_executions": sell_executions,
                "hold_decisions": hold_decisions,
                "avg_signal_strength": avg_signal_strength,
                "execution_rate": execution_rate,
                "period_start": start_time.isoformat(),
                "period_end": datetime.utcnow().isoformat(),
                "calculated_at": datetime.utcnow().isoformat()
            }
            
            # Update bot performance metrics
            await self.client.update(
                "trading_bots",
                {
                    "performance_metrics": metrics,
                    "updated_at": datetime.utcnow().isoformat()
                },
                {"id": str(bot_id)}
            )
            
            return metrics, None
            
        except Exception as e:
            logger.error(f"Error calculating bot performance: {e}")
            return {}, str(e)
    
    async def get_bot_portfolio_performance(
        self, 
        bot_id: UUID
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Get portfolio performance for a bot using database function.
        
        Args:
            bot_id: ID of the bot
            
        Returns:
            Tuple of (performance data, error message if any)
        """
        try:
            bot, error = await self.get_bot_by_id(bot_id)
            if error or not bot:
                return {}, error or "Bot not found"
            
            # Use database function to calculate portfolio performance
            result = await self.client.rpc(
                "calculate_portfolio_performance",
                {"account_id_param": str(bot.account_id)}
            )
            
            if result and len(result) > 0:
                return result[0], None
            else:
                return {}, None
                
        except Exception as e:
            logger.error(f"Error getting portfolio performance: {e}")
            return {}, str(e)
    
    async def delete_bot(self, bot_id: UUID, user_id: Optional[UUID] = None) -> Tuple[bool, Optional[str]]:
        """
        Delete a trading bot and associated data.
        
        Args:
            bot_id: ID of the bot
            user_id: Optional user ID for access control
            
        Returns:
            Tuple of (success, error message if any)
        """
        try:
            # Verify ownership if user_id provided
            if user_id:
                bot, error = await self.get_bot_by_id(bot_id, user_id)
                if error or not bot:
                    return False, error or "Bot not found or access denied"
            
            # Delete associated data first (foreign key constraints)
            await self.client.delete("sandbox_deployments", {"bot_id": str(bot_id)})
            await self.client.delete("bot_executions", {"bot_id": str(bot_id)})
            
            # Delete the bot
            result = await self.client.delete("trading_bots", {"id": str(bot_id)})
            
            if result:
                logger.info(f"Deleted trading bot: {bot_id}")
                return True, None
            else:
                return False, "Failed to delete bot"
                
        except Exception as e:
            logger.error(f"Error deleting bot: {e}")
            return False, str(e)
    
    async def clone_bot(
        self, 
        bot_id: UUID, 
        new_name: str,
        user_id: Optional[UUID] = None
    ) -> Tuple[Optional[TradingBot], Optional[str]]:
        """
        Clone an existing trading bot.
        
        Args:
            bot_id: ID of the bot to clone
            new_name: Name for the new bot
            user_id: Optional user ID for access control
            
        Returns:
            Tuple of (cloned bot, error message if any)
        """
        try:
            # Get original bot
            original_bot, error = await self.get_bot_by_id(bot_id, user_id)
            if error or not original_bot:
                return None, error or "Original bot not found"
            
            # Create clone data
            clone_data = {
                "user_id": str(original_bot.user_id),
                "account_id": str(original_bot.account_id),
                "name": new_name,
                "strategy_type": original_bot.strategy_type,
                "configuration": original_bot.configuration,
                "model_ids": [str(mid) for mid in original_bot.model_ids],
                "symbols": original_bot.symbols,
                "max_position_size": float(original_bot.max_position_size),
                "risk_limit": float(original_bot.risk_limit),
                "status": BotStatus.PAUSED.value,  # Always start clones as paused
                "performance_metrics": {}
            }
            
            result = await self.client.insert("trading_bots", clone_data)
            
            if result:
                cloned_bot = TradingBot.from_db(result[0])
                logger.info(f"Cloned bot {bot_id} to {cloned_bot.id}")
                return cloned_bot, None
            else:
                return None, "Failed to clone bot"
                
        except Exception as e:
            logger.error(f"Error cloning bot: {e}")
            return None, str(e)
    
    async def get_active_bots_for_symbol(
        self, 
        symbol: str
    ) -> Tuple[List[TradingBot], Optional[str]]:
        """
        Get all active bots trading a specific symbol.
        
        Args:
            symbol: Symbol to filter by
            
        Returns:
            Tuple of (list of active bots, error message if any)
        """
        try:
            # Note: This is a simplified query. In production, you'd use a more sophisticated
            # array contains query for the symbols field
            result = await self.client.select(
                "trading_bots",
                "*",
                filter_dict={"status": BotStatus.ACTIVE.value},
                limit=1000
            )
            
            # Filter by symbol in Python (ideally done in database)
            active_bots = [
                TradingBot.from_db(record) 
                for record in result 
                if symbol in record.get("symbols", [])
            ]
            
            return active_bots, None
            
        except Exception as e:
            logger.error(f"Error getting active bots for symbol: {e}")
            return [], str(e)
    
    async def get_bot_statistics(
        self, 
        user_id: UUID
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Get statistics about user's trading bots.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Tuple of (statistics, error message if any)
        """
        try:
            bots, error = await self.get_user_bots(user_id, limit=1000)
            if error:
                return {}, error
            
            # Calculate statistics
            total_bots = len(bots)
            by_status = {}
            by_strategy = {}
            
            for bot in bots:
                # Count by status
                status = bot.status.value if isinstance(bot.status, BotStatus) else bot.status
                by_status[status] = by_status.get(status, 0) + 1
                
                # Count by strategy
                by_strategy[bot.strategy_type] = by_strategy.get(bot.strategy_type, 0) + 1
            
            statistics = {
                "total_bots": total_bots,
                "by_status": by_status,
                "by_strategy": by_strategy,
                "active_bots": by_status.get("active", 0),
                "paused_bots": by_status.get("paused", 0)
            }
            
            return statistics, None
            
        except Exception as e:
            logger.error(f"Error getting bot statistics: {e}")
            return {}, str(e)
    
    async def bulk_update_bot_status(
        self,
        bot_ids: List[UUID],
        status: BotStatus,
        user_id: Optional[UUID] = None
    ) -> Tuple[int, Optional[str]]:
        """
        Update status for multiple bots.
        
        Args:
            bot_ids: List of bot IDs
            status: New status for all bots
            user_id: Optional user ID for access control
            
        Returns:
            Tuple of (number of bots updated, error message if any)
        """
        try:
            updated_count = 0
            
            for bot_id in bot_ids:
                success, error = await self.set_bot_status(bot_id, status, user_id)
                if success:
                    updated_count += 1
                elif error:
                    logger.warning(f"Failed to update bot {bot_id}: {error}")
            
            return updated_count, None
            
        except Exception as e:
            logger.error(f"Error bulk updating bot status: {e}")
            return 0, str(e)