#!/usr/bin/env python3
"""
Test script for market_data_fetcher.py
"""

import asyncio
import logging
from tengri.pairlist_app.market_data_fetcher import (
    TengriPairlistData, 
    load_data_source_configs,
    DataSourceManager,
    MarketDataAcquisition,
    BlockchainDataAcquisition,
    MultiSourceDataAggregator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_market_data_fetcher")

async def test_tengri_pairlist_data():
    """Test the TengriPairlistData class."""
    try:
        # Create a TengriPairlistData instance
        logger.info("Creating TengriPairlistData instance...")
        pairlist_data = TengriPairlistData()
        
        # Initialize the data provider
        logger.info("Initializing data provider...")
        await pairlist_data.initialize()
        
        # Get top pairs
        logger.info("Getting top pairs...")
        top_pairs = await pairlist_data.get_top_pairs(limit=10)
        logger.info(f"Retrieved {len(top_pairs.get('pairs', []))} top pairs")
        
        # Export pairlist
        logger.info("Exporting pairlist...")
        pairlist = await pairlist_data.export_pairlist()
        logger.info(f"Exported pairlist with {len(pairlist.get('pairs', []))} pairs")
        
        # Close connections
        logger.info("Closing connections...")
        await pairlist_data.close()
        
        logger.info("Test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("Starting market_data_fetcher test...")
    result = asyncio.run(test_tengri_pairlist_data())
    if result:
        logger.info("All tests passed!")
    else:
        logger.error("Tests failed!")