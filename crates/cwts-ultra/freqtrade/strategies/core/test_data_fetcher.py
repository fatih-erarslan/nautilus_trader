#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for AdaptiveMarketDataFetcher

A simple script to test if the AdaptiveMarketDataFetcher is working correctly.
Tests basic functionality by fetching data for a few common cryptocurrency pairs.

Usage:
    python test_data_fetcher.py [pair1 pair2 ...]
    
Examples:
    python test_data_fetcher.py  # Uses default pairs
    python test_data_fetcher.py BTC/USDT ETH/USDT  # Uses specified pairs
"""

import logging
import pandas as pd
import sys
import os
import traceback
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("DataFetcherTest")

# Import the AdaptiveMarketDataFetcher
try:
    from adaptive_market_data_fetcher import AdaptiveMarketDataFetcher
    logger.info("Successfully imported AdaptiveMarketDataFetcher")
except ImportError as e:
    logger.error(f"Failed to import AdaptiveMarketDataFetcher: {e}")
    sys.exit(1)

def test_data_fetcher(pairs: List[str] = None):
    """
    Test the AdaptiveMarketDataFetcher by fetching data for the specified pairs.
    
    Args:
        pairs: List of pairs to test (e.g., ["BTC/USDT", "ETH/USDT"])
              If None, uses a default set of common pairs
              
    Returns:
        Dictionary with test results
    """
    # Default test pairs if none provided
    if pairs is None:
        pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT"]
    
    logger.info(f"Testing AdaptiveMarketDataFetcher with pairs: {pairs}")
    
    # Create the fetcher with minimal configuration
    config = {
        "max_workers": 2,  # Limit workers for testing
        "update_interval": 86400,  # Daily update
        "enable_auto_discovery": False,  # Disable auto-discovery for testing
        "log_level": logging.INFO,
    }
    
    # Dictionary to store test results
    results = {
        "pairs_tested": pairs,
        "initialization": False,
        "data_fetch": {},
        "pair_rankings": None,
        "freqtrade_export": None,
        "errors": []
    }
    
    try:
        # Initialize fetcher
        logger.info("Initializing AdaptiveMarketDataFetcher...")
        fetcher = AdaptiveMarketDataFetcher(config)
        logger.info("Fetcher initialized successfully")
        results["initialization"] = True
        
        # Test basic properties
        active_pairs = fetcher.get_active_pairs()
        universe_pairs = fetcher.get_universe_pairs()
        logger.info(f"Active pairs: {active_pairs}")
        logger.info(f"Universe pairs: {universe_pairs}")
        results["active_pairs"] = active_pairs
        results["universe_pairs"] = universe_pairs
        
        # Test fetching data directly from MarketDataFetcher
        logger.info("Fetching data using the base MarketDataFetcher...")
        
        # Try different timeframes
        timeframes = ["1d", "4h"]
        
        for timeframe in timeframes:
            logger.info(f"Testing timeframe: {timeframe}")
            try:
                # Use the base_fetcher which is the MarketDataFetcher instance
                data_dict = fetcher.base_fetcher.fetch_data_for_cdfa(
                    pairs, 
                    source='yahoo', 
                    period='30d', 
                    interval=timeframe
                )
                
                results["data_fetch"][timeframe] = {
                    "success": bool(data_dict),
                    "pairs_with_data": list(data_dict.keys()) if data_dict else [],
                    "details": {}
                }
                
                if not data_dict:
                    logger.warning(f"No data retrieved for timeframe {timeframe}")
                    continue
                
                # Print summary of retrieved data
                logger.info(f"Successfully retrieved data for {len(data_dict)} pairs with timeframe {timeframe}")
                
                for symbol, df in data_dict.items():
                    if df is None or df.empty:
                        logger.warning(f"Empty dataframe for {symbol}")
                        results["data_fetch"][timeframe]["details"][symbol] = {
                            "success": False,
                            "error": "Empty dataframe"
                        }
                        continue
                    
                    # Print dataframe info
                    shape = df.shape
                    date_range = (df.index.min(), df.index.max())
                    columns = df.columns.tolist()
                    
                    logger.info(f"Data for {symbol}:")
                    logger.info(f"  Shape: {shape}")
                    logger.info(f"  Date range: {date_range[0]} to {date_range[1]}")
                    logger.info(f"  Columns: {columns}")
                    
                    # Show a few rows for verification
                    logger.info(f"  Recent data sample:")
                    print(df.tail(3))
                    print("\n")
                    
                    # Store results
                    results["data_fetch"][timeframe]["details"][symbol] = {
                        "success": True,
                        "shape": shape,
                        "date_range": [str(date_range[0]), str(date_range[1])],
                        "columns": columns,
                        "has_ohlcv": all(col in columns for col in ['open', 'high', 'low', 'close', 'volume']),
                        "has_technical_indicators": any(col in columns for col in ['bb_upper', 'bb_lower', 'volatility_20d'])
                    }
                    
            except Exception as e:
                error_msg = f"Error fetching data for timeframe {timeframe}: {e}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                results["errors"].append(error_msg)
                results["data_fetch"][timeframe] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Test the pair rankings functionality
        try:
            logger.info("Testing pair rankings...")
            rankings = fetcher.get_pair_rankings(limit=10)
            logger.info(f"Top ranked pairs: {rankings}")
            results["pair_rankings"] = rankings
        except Exception as e:
            error_msg = f"Error getting pair rankings: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            results["errors"].append(error_msg)
            
        # Test FreqTrade pairlist export
        try:
            logger.info("Testing FreqTrade pairlist export...")
            pairlist = fetcher.export_to_freqtrade_pairlist(limit=10)
            logger.info(f"FreqTrade pairlist result: {pairlist}")
            results["freqtrade_export"] = pairlist
        except Exception as e:
            error_msg = f"Error exporting to FreqTrade pairlist: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            results["errors"].append(error_msg)
                
        logger.info("All tests completed")
        
        # Print a summary
        print("\n" + "=" * 50)
        print("AdaptiveMarketDataFetcher Test Summary")
        print("=" * 50)
        print(f"Initialization successful: {results['initialization']}")
        print(f"Pairs tested: {results['pairs_tested']}")
        print(f"Active pairs: {len(results['active_pairs'])}")
        print(f"Universe pairs: {len(results['universe_pairs'])}")
        
        for timeframe, tf_results in results["data_fetch"].items():
            print(f"\nTimeframe {timeframe} data fetch: {'Success' if tf_results['success'] else 'Failed'}")
            if tf_results['success']:
                print(f"  Pairs with data: {len(tf_results['pairs_with_data'])}/{len(pairs)}")
                
                details = tf_results['details']
                successes = sum(1 for d in details.values() if d['success'])
                print(f"  Successful fetches: {successes}/{len(details)}")
                
                ohlcv_complete = sum(1 for d in details.values() if d.get('has_ohlcv', False))
                print(f"  Pairs with complete OHLCV data: {ohlcv_complete}/{len(details)}")
                
                indicators = sum(1 for d in details.values() if d.get('has_technical_indicators', False))
                print(f"  Pairs with technical indicators: {indicators}/{len(details)}")
                
        if results["pair_rankings"]:
            print(f"\nPair rankings test: Success")
            print(f"  Top 3 ranked pairs: {[p[0] for p in results['pair_rankings'][:3]]}")
        else:
            print(f"\nPair rankings test: Failed")
            
        if results["freqtrade_export"]:
            print(f"\nFreqTrade export test: Success")
            print(f"  Pairs in export: {len(results['freqtrade_export'].get('pairs', []))}")
        else:
            print(f"\nFreqTrade export test: Failed")
            
        if results["errors"]:
            print(f"\nErrors encountered: {len(results['errors'])}")
            for i, error in enumerate(results["errors"]):
                print(f"  {i+1}. {error}")
        else:
            print(f"\nAll tests passed successfully!")
        
        print("=" * 50)
        return results
        
    except Exception as e:
        error_msg = f"Error testing data fetcher: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        results["errors"].append(error_msg)
        return results
        
if __name__ == "__main__":
    # Allow passing pairs as command-line arguments
    custom_pairs = sys.argv[1:] if len(sys.argv) > 1 else None
    results = test_data_fetcher(custom_pairs)
    
    # Return exit code based on success/failure
    success = results["initialization"] and not results["errors"]
    sys.exit(0 if success else 1)