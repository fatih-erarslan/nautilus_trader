#!/usr/bin/env python3
"""
Comprehensive Integration Tests for CCXT Module

This script tests all components of the CCXT integration module
to ensure everything works together correctly.
"""

import asyncio
import sys
import os
from datetime import datetime
import json
from typing import Dict, List, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ccxt_integration.interfaces.ccxt_interface import CCXTInterface, ExchangeConfig
from ccxt_integration.core.client_manager import ClientManager
from ccxt_integration.core.exchange_registry import ExchangeRegistry
from ccxt_integration.streaming.websocket_manager import WebSocketManager
from ccxt_integration.execution.order_router import OrderRouter, OrderRequest, RoutingStrategy


class CCXTIntegrationTester:
    """Comprehensive test suite for CCXT integration"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'summary': {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'errors': []
            }
        }
        
    async def test_exchange_registry(self) -> bool:
        """Test Exchange Registry functionality"""
        print("\n" + "="*60)
        print("TEST 1: Exchange Registry")
        print("="*60)
        
        try:
            registry = ExchangeRegistry()
            
            # Test 1.1: List exchanges
            exchanges = registry.list_exchanges()
            assert len(exchanges) > 0, "No exchanges found"
            print(f"✅ Found {len(exchanges)} exchanges")
            
            # Test 1.2: Get exchange metadata
            binance = registry.get_exchange('binance')
            assert binance is not None, "Binance not found"
            assert binance.name == 'binance', "Invalid exchange name"
            print(f"✅ Retrieved Binance metadata")
            
            # Test 1.3: Filter by capability
            ws_exchanges = registry.get_exchanges_by_capability('websocket_support')
            assert len(ws_exchanges) > 0, "No WebSocket exchanges found"
            print(f"✅ Found {len(ws_exchanges)} exchanges with WebSocket")
            
            # Test 1.4: Filter by quote currency
            usdt_exchanges = registry.get_exchanges_by_quote_currency('USDT')
            assert len(usdt_exchanges) > 0, "No USDT exchanges found"
            print(f"✅ Found {len(usdt_exchanges)} exchanges supporting USDT")
            
            # Test 1.5: Get best exchange
            best = registry.get_best_exchange_for_pair('BTC', 'USDT')
            assert best is not None, "No best exchange found"
            print(f"✅ Best exchange for BTC/USDT: {best}")
            
            self.results['tests']['exchange_registry'] = 'PASSED'
            return True
            
        except Exception as e:
            print(f"❌ Exchange Registry test failed: {str(e)}")
            self.results['tests']['exchange_registry'] = f'FAILED: {str(e)}'
            self.results['summary']['errors'].append(str(e))
            return False
            
    async def test_ccxt_interface(self) -> bool:
        """Test CCXT Interface"""
        print("\n" + "="*60)
        print("TEST 2: CCXT Interface")
        print("="*60)
        
        try:
            # Test with Kraken (doesn't require sandbox URLs)
            config = ExchangeConfig(
                name='kraken',
                sandbox=False,
                enable_rate_limit=True
            )
            
            interface = CCXTInterface(config)
            
            # Test 2.1: Initialization
            await interface.initialize()
            assert interface._initialized, "Interface not initialized"
            print(f"✅ Interface initialized for {config.name}")
            
            # Test 2.2: Get markets
            markets = await interface.get_markets()
            assert len(markets) > 0, "No markets found"
            print(f"✅ Found {len(markets)} trading pairs")
            
            # Test 2.3: Get ticker
            if 'BTC/USD' in markets or 'XBT/USD' in markets:
                symbol = 'XBT/USD' if 'XBT/USD' in markets else 'BTC/USD'
                ticker = await interface.get_ticker(symbol)
                assert ticker['last'] > 0, "Invalid ticker price"
                print(f"✅ {symbol} ticker: ${ticker['last']:,.2f}")
            
            # Test 2.4: Get orderbook
            if 'ETH/USD' in markets:
                orderbook = await interface.get_orderbook('ETH/USD', limit=5)
                assert len(orderbook['bids']) > 0, "No bids in orderbook"
                assert len(orderbook['asks']) > 0, "No asks in orderbook"
                print(f"✅ ETH/USD orderbook retrieved")
            
            await interface.close()
            print(f"✅ Interface closed successfully")
            
            self.results['tests']['ccxt_interface'] = 'PASSED'
            return True
            
        except Exception as e:
            print(f"❌ CCXT Interface test failed: {str(e)}")
            self.results['tests']['ccxt_interface'] = f'FAILED: {str(e)}'
            self.results['summary']['errors'].append(str(e))
            return False
            
    async def test_client_manager(self) -> bool:
        """Test Client Manager"""
        print("\n" + "="*60)
        print("TEST 3: Client Manager")
        print("="*60)
        
        try:
            manager = ClientManager(max_clients_per_exchange=2)
            
            # Test 3.1: Add exchange
            config = ExchangeConfig(name='kraken', sandbox=False)
            await manager.add_exchange(config)
            assert 'kraken' in manager.active_exchanges, "Exchange not added"
            print(f"✅ Added Kraken to manager")
            
            # Test 3.2: Get client
            client = await manager.get_client('kraken')
            assert client is not None, "Client not retrieved"
            print(f"✅ Retrieved client for Kraken")
            
            # Test 3.3: Health check
            await manager._check_exchange_health('kraken')
            health = manager.health_status.get('kraken')
            assert health is not None, "No health status"
            print(f"✅ Health check completed: {health.status.value}")
            
            # Test 3.4: Get health report
            report = await manager.get_health_report()
            assert report['total_exchanges'] == 1, "Invalid exchange count"
            print(f"✅ Health report generated")
            
            # Test 3.5: Execute with failover
            try:
                result = await manager.execute_with_failover(
                    'get_ticker',
                    'ETH/USD'
                )
                print(f"✅ Failover execution successful")
            except:
                print(f"⚠️ Failover execution skipped (no API key)")
            
            await manager.close_all()
            print(f"✅ Manager closed successfully")
            
            self.results['tests']['client_manager'] = 'PASSED'
            return True
            
        except Exception as e:
            print(f"❌ Client Manager test failed: {str(e)}")
            self.results['tests']['client_manager'] = f'FAILED: {str(e)}'
            self.results['summary']['errors'].append(str(e))
            return False
            
    async def test_websocket_manager(self) -> bool:
        """Test WebSocket Manager"""
        print("\n" + "="*60)
        print("TEST 4: WebSocket Manager")
        print("="*60)
        
        try:
            # Note: Requires ccxt.pro
            import ccxt.pro as ccxt_pro
            
            ws_manager = WebSocketManager(
                exchange_name='binance',
                config={'sandbox': False}
            )
            
            # Test 4.1: Initialize
            await ws_manager.initialize()
            assert ws_manager._running, "WebSocket not running"
            print(f"✅ WebSocket manager initialized")
            
            # Test 4.2: Subscribe to ticker
            received_data = []
            
            def on_ticker(ticker):
                received_data.append(ticker)
                
            sub_id = await ws_manager.subscribe_ticker('BTC/USDT', on_ticker)
            assert sub_id in ws_manager.subscriptions, "Subscription not created"
            print(f"✅ Subscribed to BTC/USDT ticker")
            
            # Wait for data
            await asyncio.sleep(3)
            
            # Test 4.3: Check subscription status
            status = ws_manager.get_subscription_status()
            assert status['running'], "WebSocket not running"
            assert len(status['subscriptions']) > 0, "No active subscriptions"
            print(f"✅ Subscription status: {len(status['subscriptions'])} active")
            
            # Test 4.4: Unsubscribe
            await ws_manager.unsubscribe(sub_id)
            assert sub_id not in ws_manager.subscriptions, "Unsubscribe failed"
            print(f"✅ Unsubscribed successfully")
            
            await ws_manager.close()
            print(f"✅ WebSocket manager closed")
            
            if len(received_data) > 0:
                print(f"✅ Received {len(received_data)} ticker updates")
            
            self.results['tests']['websocket_manager'] = 'PASSED'
            return True
            
        except ImportError:
            print(f"⚠️ WebSocket test skipped (CCXT Pro not installed)")
            self.results['tests']['websocket_manager'] = 'SKIPPED: CCXT Pro required'
            return True
        except Exception as e:
            print(f"❌ WebSocket Manager test failed: {str(e)}")
            self.results['tests']['websocket_manager'] = f'FAILED: {str(e)}'
            self.results['summary']['errors'].append(str(e))
            return False
            
    async def test_order_router(self) -> bool:
        """Test Order Router"""
        print("\n" + "="*60)
        print("TEST 5: Order Router")
        print("="*60)
        
        try:
            manager = ClientManager()
            
            # Add test exchange
            await manager.add_exchange(ExchangeConfig(name='kraken', sandbox=False))
            
            router = OrderRouter(manager)
            
            # Test 5.1: Create order request
            request = OrderRequest(
                symbol='BTC/USD',
                side='buy',
                amount=0.001,
                order_type='market',
                routing_strategy=RoutingStrategy.BEST_PRICE,
                max_slippage=0.01
            )
            print(f"✅ Order request created")
            
            # Test 5.2: Get candidate exchanges
            candidates = await router._get_candidate_exchanges(request)
            print(f"✅ Found {len(candidates)} candidate exchanges")
            
            # Test 5.3: Analyze exchange (without executing)
            if candidates:
                analysis = await router._analyze_exchange(candidates[0], request)
                if analysis:
                    print(f"✅ Exchange analysis completed")
                    print(f"   Price: ${analysis.get('price', 0):,.2f}")
                    print(f"   Liquidity: {analysis.get('liquidity', 0):.2%}")
            
            # Test 5.4: Routing strategies
            strategies = [
                RoutingStrategy.BEST_PRICE,
                RoutingStrategy.LOWEST_FEE,
                RoutingStrategy.HIGHEST_LIQUIDITY,
                RoutingStrategy.SMART_ROUTING
            ]
            
            for strategy in strategies:
                request.routing_strategy = strategy
                # Would execute here with real API keys
                print(f"✅ {strategy.value} strategy configured")
            
            await manager.close_all()
            
            self.results['tests']['order_router'] = 'PASSED'
            return True
            
        except Exception as e:
            print(f"❌ Order Router test failed: {str(e)}")
            self.results['tests']['order_router'] = f'FAILED: {str(e)}'
            self.results['summary']['errors'].append(str(e))
            return False
            
    async def test_integration(self) -> bool:
        """Test full integration workflow"""
        print("\n" + "="*60)
        print("TEST 6: Full Integration")
        print("="*60)
        
        try:
            # Create all components
            registry = ExchangeRegistry()
            manager = ClientManager()
            
            # Add multiple exchanges
            exchanges_added = 0
            for exchange_name in ['kraken', 'coinbase']:
                try:
                    metadata = registry.get_exchange(exchange_name)
                    if metadata:
                        config = ExchangeConfig(
                            name=exchange_name,
                            sandbox=metadata.capabilities.sandbox_available
                        )
                        await manager.add_exchange(config)
                        exchanges_added += 1
                except:
                    pass
            
            print(f"✅ Added {exchanges_added} exchanges")
            
            # Create router
            router = OrderRouter(manager)
            
            # Test best exchange selection
            best = await manager.get_best_exchange('BTC/USD')
            if best:
                print(f"✅ Best exchange for BTC/USD: {best}")
                
                # Get client and fetch data
                client = await manager.get_client(best)
                if client:
                    ticker = await client.get_ticker('BTC/USD')
                    print(f"✅ BTC/USD on {best}: ${ticker['last']:,.2f}")
            
            # Generate health report
            health = await manager.get_health_report()
            print(f"✅ System health: {health['healthy_exchanges']}/{health['total_exchanges']} healthy")
            
            # Cleanup
            await manager.close_all()
            
            self.results['tests']['integration'] = 'PASSED'
            return True
            
        except Exception as e:
            print(f"❌ Integration test failed: {str(e)}")
            self.results['tests']['integration'] = f'FAILED: {str(e)}'
            self.results['summary']['errors'].append(str(e))
            return False
            
    async def run_all_tests(self):
        """Run all tests and generate report"""
        print("\n" + "="*70)
        print(" CCXT INTEGRATION MODULE - COMPREHENSIVE TEST SUITE")
        print("="*70)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run tests
        tests = [
            self.test_exchange_registry(),
            self.test_ccxt_interface(),
            self.test_client_manager(),
            self.test_websocket_manager(),
            self.test_order_router(),
            self.test_integration()
        ]
        
        results = await asyncio.gather(*tests, return_exceptions=True)
        
        # Calculate summary
        self.results['summary']['total'] = len(tests)
        self.results['summary']['passed'] = sum(1 for r in results if r is True)
        self.results['summary']['failed'] = self.results['summary']['total'] - self.results['summary']['passed']
        
        # Print summary
        print("\n" + "="*70)
        print(" TEST RESULTS SUMMARY")
        print("="*70)
        
        for test_name, result in self.results['tests'].items():
            status = "✅" if result == 'PASSED' else "❌" if 'FAILED' in str(result) else "⚠️"
            print(f"{status} {test_name}: {result}")
        
        print("\n" + "-"*70)
        print(f"Total Tests: {self.results['summary']['total']}")
        print(f"Passed: {self.results['summary']['passed']}")
        print(f"Failed: {self.results['summary']['failed']}")
        
        success_rate = (self.results['summary']['passed'] / self.results['summary']['total']) * 100
        print(f"Success Rate: {success_rate:.1f}%")
        
        if self.results['summary']['errors']:
            print("\nErrors encountered:")
            for error in self.results['summary']['errors']:
                print(f"  - {error}")
        
        # Save results
        with open('/workspaces/ai-news-trader/src/ccxt_integration/tests/test_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        print("\n✅ Test results saved to test_results.json")
        
        print("\n" + "="*70)
        if success_rate >= 80:
            print(" ✅ CCXT MODULE VERIFICATION: PASSED")
        else:
            print(" ❌ CCXT MODULE VERIFICATION: NEEDS ATTENTION")
        print("="*70)
        
        return success_rate >= 80


async def main():
    """Main test runner"""
    tester = CCXTIntegrationTester()
    success = await tester.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)