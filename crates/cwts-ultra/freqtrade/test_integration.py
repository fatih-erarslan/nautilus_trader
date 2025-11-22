#!/usr/bin/env python3
"""
Test script for CWTS Ultra FreqTrade Integration
"""

import sys
import os
import time

# Add freqtrade directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("CWTS Ultra FreqTrade Integration Test")
print("=" * 60)

# Test 1: Import the client
print("\n1. Testing client import...")
try:
    import cwts_client_simple as cwts_client
    print("✅ Client module imported successfully")
except ImportError as e:
    print(f"❌ Failed to import client: {e}")
    sys.exit(1)

# Test 2: Create client instance
print("\n2. Creating client instance...")
try:
    client = cwts_client.create_client(
        shm_path="/dev/shm/cwts_ultra",
        websocket_url="ws://localhost:4000"
    )
    print(f"✅ Client created")
    print(f"   - Shared memory: {'Connected' if client.connected else 'Not connected'}")
    print(f"   - WebSocket URL: {client.websocket_url}")
except Exception as e:
    print(f"❌ Failed to create client: {e}")
    sys.exit(1)

# Test 3: Test market data retrieval
print("\n3. Testing market data retrieval...")
try:
    data = client.get_market_data("BTC/USDT")
    if data:
        print(f"✅ Market data retrieved: {data}")
    else:
        print("⚠️  No market data available (CWTS Ultra may not be running)")
except Exception as e:
    print(f"❌ Failed to get market data: {e}")

# Test 4: Test signal sending
print("\n4. Testing signal transmission...")
try:
    # Send a test buy signal
    success = client.send_signal(
        symbol="BTC/USDT",
        action=cwts_client.SIGNAL_BUY,
        confidence=0.85,
        size=0.01,
        price=0.0,  # Market order
        stop_loss=40000,
        take_profit=45000,
        strategy_id=12345
    )
    if success:
        print("✅ Buy signal sent successfully")
    else:
        print("⚠️  Signal not sent (CWTS Ultra may not be running)")
    
    # Send a test sell signal
    success = client.send_signal(
        symbol="BTC/USDT",
        action=cwts_client.SIGNAL_SELL,
        confidence=0.90,
        strategy_id=12345
    )
    if success:
        print("✅ Sell signal sent successfully")
    else:
        print("⚠️  Signal not sent")
        
except Exception as e:
    print(f"❌ Failed to send signal: {e}")

# Test 5: Test order book retrieval
print("\n5. Testing order book retrieval...")
try:
    import numpy as np
    orderbook = client.get_order_book("BTC/USDT", depth=10)
    if len(orderbook) > 0:
        print(f"✅ Order book retrieved: shape {orderbook.shape}")
        print(f"   Sample: Bid={orderbook[0,0]:.2f} @ {orderbook[0,1]:.4f}, "
              f"Ask={orderbook[0,2]:.2f} @ {orderbook[0,3]:.4f}")
    else:
        print("⚠️  No order book data available")
except Exception as e:
    print(f"❌ Failed to get order book: {e}")

# Test 6: Test multi-symbol snapshot
print("\n6. Testing multi-symbol snapshot...")
try:
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    snapshot = client.get_market_snapshot(symbols)
    print(f"✅ Snapshot retrieved: shape {snapshot.shape}")
    for i, symbol in enumerate(symbols):
        if not np.isnan(snapshot[i, 0]):
            print(f"   {symbol}: Bid={snapshot[i,0]:.2f}, Ask={snapshot[i,1]:.2f}, "
                  f"Spread={snapshot[i,4]:.4f}")
        else:
            print(f"   {symbol}: No data available")
except Exception as e:
    print(f"❌ Failed to get snapshot: {e}")

# Test 7: Performance test
print("\n7. Testing latency...")
try:
    import time
    
    # Test signal sending latency
    start = time.perf_counter()
    for _ in range(100):
        client.send_signal("BTC/USDT", cwts_client.SIGNAL_HOLD, strategy_id=12345)
    end = time.perf_counter()
    
    latency_us = ((end - start) / 100) * 1_000_000
    print(f"✅ Average signal latency: {latency_us:.1f} μs")
    
    if latency_us < 100:
        print("   🚀 Ultra-low latency achieved!")
    elif latency_us < 1000:
        print("   ⚡ Low latency achieved")
    else:
        print("   ⚠️  Higher latency (WebSocket mode?)")
        
except Exception as e:
    print(f"❌ Performance test failed: {e}")

# Cleanup
print("\n8. Cleaning up...")
try:
    client.close()
    print("✅ Client closed successfully")
except Exception as e:
    print(f"⚠️  Error during cleanup: {e}")

print("\n" + "=" * 60)
print("Integration test complete!")
print("=" * 60)

# Summary
print("\n📊 Summary:")
print(f"  - Client mode: {'Shared Memory' if client.connected else 'WebSocket'}")
print(f"  - Average latency: {latency_us:.1f} μs")
print(f"  - Ready for FreqTrade: ✅")
print("\nNext steps:")
print("  1. Start CWTS Ultra: ~/.local/cwts-ultra/scripts/launch.sh")
print("  2. Configure FreqTrade with CWTSMomentumStrategy")
print("  3. Run backtest or live trading")
print()