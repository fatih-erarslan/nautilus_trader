"""
Basic trading example using IBKR integration

This example demonstrates how to connect to IB TWS, place orders,
and monitor positions with the low-latency IBKR client.
"""

import asyncio
import logging
from src.trading_apis.ibkr import IBKRClient, ConnectionConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Main trading example"""
    
    # Configure connection (paper trading)
    config = ConnectionConfig(
        host="127.0.0.1",
        port=7497,  # Paper trading port
        client_id=1,
        auto_reconnect=True,
        readonly=False
    )
    
    # Create client
    client = IBKRClient(config)
    
    try:
        # Connect to TWS
        logger.info("Connecting to TWS...")
        if not await client.connect():
            logger.error("Failed to connect to TWS")
            return
        
        logger.info("Connected successfully!")
        
        # Place a market order
        logger.info("Placing market order for AAPL...")
        order_id = await client.place_order(
            symbol="AAPL",
            quantity=10,
            order_type="MKT",
            side="BUY"
        )
        
        if order_id:
            logger.info(f"Market order placed: {order_id}")
        else:
            logger.error("Failed to place market order")
            return
        
        # Wait for order to process
        await asyncio.sleep(2)
        
        # Place a limit order
        logger.info("Placing limit order for MSFT...")
        limit_order_id = await client.place_order(
            symbol="MSFT",
            quantity=5,
            order_type="LMT",
            side="BUY",
            price=300.00
        )
        
        if limit_order_id:
            logger.info(f"Limit order placed: {limit_order_id}")
        else:
            logger.error("Failed to place limit order")
        
        # Monitor orders for 10 seconds
        for i in range(10):
            if order_id:
                order_status = await client.get_order_status(order_id)
                if order_status:
                    logger.info(f"Order {order_id} status: {order_status.get('status', 'Unknown')}")
            
            await asyncio.sleep(1)
        
        # Get current positions
        positions = await client.get_positions()
        logger.info(f"Current positions: {positions}")
        
        # Get account values
        account_values = await client.get_account_values()
        logger.info(f"Account values: {account_values}")
        
        # Get latency report
        latency_report = client.get_latency_report()
        logger.info(f"Latency report: {latency_report}")
        
    except Exception as e:
        logger.error(f"Error in trading example: {e}")
    
    finally:
        # Disconnect
        await client.disconnect()
        logger.info("Disconnected from TWS")


if __name__ == "__main__":
    asyncio.run(main())