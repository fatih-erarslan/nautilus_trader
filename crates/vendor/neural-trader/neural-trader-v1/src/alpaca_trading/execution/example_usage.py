"""Example usage of the Alpaca execution engine.

Demonstrates low-latency order execution with comprehensive
monitoring and performance tracking.
"""

import asyncio
import os
from datetime import datetime
import logging
from typing import List

from execution_engine import ExecutionEngine, Signal
from order_manager import OrderStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main execution example."""
    
    # Initialize execution engine
    api_key = os.getenv('ALPACA_API_KEY', 'your-api-key')
    api_secret = os.getenv('ALPACA_API_SECRET', 'your-api-secret')
    
    engine = ExecutionEngine(
        api_key=api_key,
        api_secret=api_secret,
        base_url="https://paper-api.alpaca.markets",  # Paper trading
        max_position_size=1000,
        max_order_value=10000,
        max_daily_trades=50
    )
    
    # Initialize components
    await engine.initialize()
    
    # Register order callbacks
    engine.order_manager.register_callback('order_submitted', on_order_submitted)
    engine.order_manager.register_callback('order_filled', on_order_filled)
    engine.order_manager.register_callback('order_rejected', on_order_rejected)
    
    try:
        # Example 1: High urgency market order
        signal1 = Signal(
            signal_id="sig_001",
            symbol="AAPL",
            action="buy",
            quantity=100,
            urgency="high",
            strategy_id="momentum_strategy",
            confidence=0.85,
            metadata={"reason": "Breakout detected"}
        )
        
        logger.info("\n=== Executing high urgency order ===")
        result1 = await engine.execute_signal(signal1)
        log_execution_result(result1)
        
        # Example 2: Passive limit order
        signal2 = Signal(
            signal_id="sig_002",
            symbol="MSFT",
            action="sell",
            quantity=50,
            urgency="low",
            strategy_id="mean_reversion",
            confidence=0.75,
            target_price=350.50,
            stop_loss=345.00
        )
        
        logger.info("\n=== Executing passive limit order ===")
        result2 = await engine.execute_signal(signal2)
        log_execution_result(result2)
        
        # Example 3: Standard order with slippage control
        signal3 = Signal(
            signal_id="sig_003",
            symbol="GOOGL",
            action="buy",
            quantity=25,
            urgency="medium",
            strategy_id="pairs_trading",
            confidence=0.80
        )
        
        logger.info("\n=== Executing standard order ===")
        result3 = await engine.execute_signal(signal3)
        log_execution_result(result3)
        
        # Wait for fills (in real trading)
        await asyncio.sleep(5)
        
        # Check active orders
        logger.info("\n=== Active Orders ===")
        active_orders = await engine.order_manager.get_active_orders()
        for order in active_orders:
            logger.info(f"Order {order.client_order_id}: {order.symbol} {order.side} "
                       f"{order.qty} @ {order.order_type.value} - Status: {order.status.value}")
        
        # Example 4: Cancel an order
        if active_orders:
            order_to_cancel = active_orders[0]
            logger.info(f"\n=== Cancelling order {order_to_cancel.client_order_id} ===")
            cancelled = await engine.cancel_order(order_to_cancel.client_order_id)
            logger.info(f"Cancellation {'successful' if cancelled else 'failed'}")
        
        # Example 5: Replace an order
        if len(active_orders) > 1:
            order_to_replace = active_orders[1]
            logger.info(f"\n=== Replacing order {order_to_replace.client_order_id} ===")
            replaced = await engine.replace_order(
                order_to_replace.client_order_id,
                new_qty=order_to_replace.qty * 0.5,
                new_limit_price=order_to_replace.limit_price * 1.01 if order_to_replace.limit_price else None
            )
            logger.info(f"Replacement {'successful' if replaced else 'failed'}")
        
        # Performance metrics
        logger.info("\n=== Performance Metrics ===")
        metrics = engine.get_metrics()
        
        logger.info("Execution Engine Metrics:")
        logger.info(f"  Signals processed: {metrics['signals_processed']}")
        logger.info(f"  Orders submitted: {metrics['orders_submitted']}")
        logger.info(f"  Orders rejected: {metrics['orders_rejected']}")
        logger.info(f"  Avg total latency: {metrics['avg_total_latency_ms']:.1f}ms")
        logger.info(f"  Avg risk check: {metrics['avg_risk_check_ms']:.1f}ms")
        logger.info(f"  Avg routing: {metrics['avg_routing_ms']:.1f}ms")
        logger.info(f"  Avg submission: {metrics['avg_submission_ms']:.1f}ms")
        
        logger.info("\nOrder Manager Metrics:")
        om_metrics = metrics['order_manager_metrics']
        logger.info(f"  Total orders: {om_metrics['total_orders']}")
        logger.info(f"  Filled orders: {om_metrics['filled_orders']}")
        logger.info(f"  Rejected orders: {om_metrics['rejected_orders']}")
        logger.info(f"  Avg fill latency: {om_metrics['avg_fill_latency_ms']:.1f}ms")
        
        logger.info("\nSlippage Metrics:")
        slippage_metrics = metrics['slippage_metrics']
        logger.info(f"  Total executions: {slippage_metrics['total_executions']}")
        logger.info(f"  Avg slippage: {slippage_metrics['avg_slippage_bps']:.1f}bps")
        logger.info(f"  Execution quality: {slippage_metrics['execution_quality_score']:.1f}%")
        
        # Smart router metrics
        router_metrics = engine.smart_router.get_metrics()
        logger.info("\nSmart Router Metrics:")
        logger.info(f"  Orders routed: {router_metrics['orders_routed']}")
        logger.info(f"  Market orders: {router_metrics['market_orders']} "
                   f"({router_metrics.get('market_order_pct', 0):.1f}%)")
        logger.info(f"  Limit orders: {router_metrics['limit_orders']} "
                   f"({router_metrics.get('limit_order_pct', 0):.1f}%)")
        
    finally:
        # Clean up
        await engine.close()


def log_execution_result(result):
    """Log execution result details."""
    if result.success:
        logger.info(f"✅ Order executed successfully")
        logger.info(f"   Order ID: {result.order.client_order_id}")
        logger.info(f"   Symbol: {result.order.symbol}")
        logger.info(f"   Side: {result.order.side}")
        logger.info(f"   Quantity: {result.order.qty}")
        logger.info(f"   Type: {result.order.order_type.value}")
        if result.order.limit_price:
            logger.info(f"   Limit Price: ${result.order.limit_price:.2f}")
        logger.info(f"   Total Latency: {result.total_latency_ms:.1f}ms")
        logger.info(f"   Breakdown: {result.latency_breakdown}")
    else:
        logger.error(f"❌ Order execution failed: {result.error}")
        logger.error(f"   Latency: {result.total_latency_ms:.1f}ms")


async def on_order_submitted(order):
    """Callback for order submission."""
    logger.info(f"📤 Order submitted: {order.client_order_id} - {order.symbol}")


async def on_order_filled(order):
    """Callback for order fill."""
    logger.info(f"✅ Order filled: {order.client_order_id} - {order.symbol} "
               f"{order.filled_qty} @ ${order.avg_fill_price:.2f}")
    if order.fill_latency_ms:
        logger.info(f"   Fill latency: {order.fill_latency_ms:.1f}ms")


async def on_order_rejected(order):
    """Callback for order rejection."""
    logger.error(f"❌ Order rejected: {order.client_order_id} - {order.symbol}")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
