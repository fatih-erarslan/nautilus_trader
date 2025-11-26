"""
Performance benchmarks for crypto trading system

Tests system performance under various load conditions.
"""

import pytest
import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from unittest.mock import Mock, patch, AsyncMock

from crypto_trading.beefy.beefy_client import BeefyFinanceAPI
from crypto_trading.strategies.yield_chaser import YieldChaserStrategy
from crypto_trading.strategies.stable_farmer import StableFarmerStrategy
from crypto_trading.database.connection import DatabaseManager
from crypto_trading.database.models import VaultPosition, YieldHistory
from crypto_trading.mcp_tools.beefy_tools import BeefyMCPTools

from ..fixtures.vault_data import SAMPLE_VAULTS, SAMPLE_APY_DATA


class PerformanceBenchmark:
    """Base class for performance benchmarks"""
    
    def __init__(self):
        self.results = []
    
    def time_operation(self, operation_name: str):
        """Context manager to time operations"""
        class TimingContext:
            def __init__(self, benchmark, name):
                self.benchmark = benchmark
                self.name = name
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.perf_counter()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                end_time = time.perf_counter()
                duration = end_time - self.start_time
                self.benchmark.results.append({
                    'operation': self.name,
                    'duration': duration,
                    'timestamp': time.time()
                })
        
        return TimingContext(self, operation_name)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.results:
            return {}
        
        durations = [r['duration'] for r in self.results]
        
        return {
            'count': len(durations),
            'mean': statistics.mean(durations),
            'median': statistics.median(durations),
            'std_dev': statistics.stdev(durations) if len(durations) > 1 else 0,
            'min': min(durations),
            'max': max(durations),
            'total_time': sum(durations),
            'operations_per_second': len(durations) / sum(durations) if sum(durations) > 0 else 0
        }


class TestAPIPerformance:
    """Test API client performance"""
    
    @pytest.fixture
    async def mock_beefy_client(self):
        """Create mocked Beefy client for performance testing"""
        client = BeefyFinanceAPI()
        client.session = AsyncMock()
        
        # Mock fast responses
        with patch.object(client, '_make_request') as mock_request:
            def fast_response(endpoint, *args, **kwargs):
                # Simulate small processing delay
                time.sleep(0.001)
                
                if endpoint == "/vaults":
                    return SAMPLE_VAULTS
                elif endpoint == "/apy":
                    return {k: v["totalApy"] for k, v in SAMPLE_APY_DATA.items()}
                elif endpoint == "/prices":
                    return {"BNB": 245.67, "CAKE": 2.15, "ETH": 1650.0}
                else:
                    return {}
            
            mock_request.side_effect = fast_response
            yield client
        
        if client.session:
            await client.session.close()
    
    @pytest.mark.asyncio
    async def test_sequential_api_calls(self, mock_beefy_client):
        """Benchmark sequential API calls"""
        benchmark = PerformanceBenchmark()
        
        iterations = 100
        
        for i in range(iterations):
            with benchmark.time_operation(f"get_vaults_{i}"):
                await mock_beefy_client.get_vaults()
        
        stats = benchmark.get_stats()
        
        # Performance assertions
        assert stats['mean'] < 0.1, f"Average API call took {stats['mean']:.4f}s, expected < 0.1s"
        assert stats['operations_per_second'] > 10, f"Only {stats['operations_per_second']:.2f} ops/sec, expected > 10"
        
        print(f"Sequential API Performance: {stats['operations_per_second']:.2f} ops/sec")
    
    @pytest.mark.asyncio
    async def test_concurrent_api_calls(self, mock_beefy_client):
        """Benchmark concurrent API calls"""
        benchmark = PerformanceBenchmark()
        
        concurrent_requests = 50
        
        async def single_request(request_id):
            with benchmark.time_operation(f"concurrent_request_{request_id}"):
                return await mock_beefy_client.get_vaults()
        
        start_time = time.perf_counter()
        
        # Execute concurrent requests
        tasks = [single_request(i) for i in range(concurrent_requests)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        stats = benchmark.get_stats()
        
        # All requests should complete
        assert len(results) == concurrent_requests
        
        # Concurrent performance should be better than sequential
        concurrent_ops_per_sec = concurrent_requests / total_time
        assert concurrent_ops_per_sec > 50, f"Concurrent performance: {concurrent_ops_per_sec:.2f} ops/sec"
        
        print(f"Concurrent API Performance: {concurrent_ops_per_sec:.2f} ops/sec")
    
    @pytest.mark.asyncio
    async def test_api_caching_performance(self, mock_beefy_client):
        """Test performance improvement from caching"""
        benchmark = PerformanceBenchmark()
        
        # First call (should be slow due to API call)
        with benchmark.time_operation("first_call"):
            await mock_beefy_client.get_vaults()
        
        # Subsequent calls (should be fast due to caching)
        for i in range(10):
            with benchmark.time_operation(f"cached_call_{i}"):
                await mock_beefy_client.get_vaults()
        
        stats = benchmark.get_stats()
        
        # Cached calls should be significantly faster
        first_call_time = benchmark.results[0]['duration']
        cached_call_times = [r['duration'] for r in benchmark.results[1:]]
        avg_cached_time = statistics.mean(cached_call_times)
        
        # Cache should provide at least 2x speedup
        speedup_ratio = first_call_time / avg_cached_time
        assert speedup_ratio > 2, f"Cache speedup only {speedup_ratio:.2f}x, expected > 2x"
        
        print(f"Cache speedup: {speedup_ratio:.2f}x")


class TestStrategyPerformance:
    """Test trading strategy performance"""
    
    @pytest.fixture
    def large_opportunity_set(self):
        """Create large set of opportunities for performance testing"""
        from crypto_trading.strategies.base_strategy import VaultOpportunity, ChainType
        
        opportunities = []
        
        for i in range(1000):  # Large number of opportunities
            opportunity = VaultOpportunity(
                vault_id=f"test-vault-{i}",
                chain=ChainType.BSC if i % 2 == 0 else ChainType.ETHEREUM,
                protocol=f"protocol-{i % 10}",
                token_pair=(f"TOKEN{i}", f"TOKEN{i+1}"),
                apy=10.0 + (i % 50),  # APY range 10-60%
                daily_apy=(10.0 + (i % 50)) / 365,
                tvl=1000000.0 + (i * 10000),
                platform_fee=0.045,
                withdraw_fee=0.001,
                is_paused=False,
                has_boost=i % 10 == 0,  # 10% have boost
                boost_apy=5.0 if i % 10 == 0 else None,
                risk_factors={"smart_contract": 0.1 + (i % 5) * 0.1},
                created_at=time.time() - (i * 86400),  # Different creation times
                last_harvest=time.time() - (i * 3600)  # Different harvest times
            )
            opportunities.append(opportunity)
        
        return opportunities
    
    @pytest.fixture
    def portfolio_state(self):
        """Create portfolio state for testing"""
        from crypto_trading.strategies.base_strategy import PortfolioState
        
        return PortfolioState(
            positions=[],
            total_value=100000.0,
            available_capital=50000.0,
            timestamp=time.time()
        )
    
    def test_yield_chaser_performance(self, large_opportunity_set, portfolio_state):
        """Benchmark yield chaser strategy performance"""
        strategy = YieldChaserStrategy(min_apy_threshold=15.0)
        benchmark = PerformanceBenchmark()
        
        iterations = 10
        
        for i in range(iterations):
            with benchmark.time_operation(f"evaluate_opportunities_{i}"):
                evaluations = strategy.evaluate_opportunities(large_opportunity_set, portfolio_state)
        
        stats = benchmark.get_stats()
        
        # Strategy should handle 1000 opportunities quickly
        assert stats['mean'] < 1.0, f"Strategy evaluation took {stats['mean']:.4f}s, expected < 1.0s"
        assert len(evaluations) > 0, "Strategy should return some evaluations"
        
        # Performance should be consistent
        assert stats['std_dev'] < stats['mean'] * 0.5, "Performance should be consistent"
        
        print(f"Yield Chaser Performance: {stats['mean']:.4f}s avg, {stats['operations_per_second']:.2f} ops/sec")
    
    def test_stable_farmer_performance(self, large_opportunity_set, portfolio_state):
        """Benchmark stable farmer strategy performance"""
        strategy = StableFarmerStrategy(preferred_stablecoins=["USDC", "USDT", "DAI"])
        benchmark = PerformanceBenchmark()
        
        iterations = 10
        
        for i in range(iterations):
            with benchmark.time_operation(f"evaluate_opportunities_{i}"):
                evaluations = strategy.evaluate_opportunities(large_opportunity_set, portfolio_state)
        
        stats = benchmark.get_stats()
        
        # Strategy should handle large opportunity set efficiently
        assert stats['mean'] < 1.0, f"Strategy evaluation took {stats['mean']:.4f}s, expected < 1.0s"
        
        print(f"Stable Farmer Performance: {stats['mean']:.4f}s avg, {stats['operations_per_second']:.2f} ops/sec")
    
    def test_strategy_comparison_performance(self, large_opportunity_set, portfolio_state):
        """Compare performance of different strategies"""
        strategies = {
            'yield_chaser': YieldChaserStrategy(min_apy_threshold=15.0),
            'stable_farmer': StableFarmerStrategy(preferred_stablecoins=["USDC", "USDT", "DAI"])
        }
        
        performance_results = {}
        
        for name, strategy in strategies.items():
            benchmark = PerformanceBenchmark()
            
            for i in range(5):
                with benchmark.time_operation(f"{name}_eval_{i}"):
                    strategy.evaluate_opportunities(large_opportunity_set, portfolio_state)
            
            performance_results[name] = benchmark.get_stats()
        
        # Compare performance
        for name, stats in performance_results.items():
            print(f"{name}: {stats['mean']:.4f}s avg, {stats['operations_per_second']:.2f} ops/sec")
        
        # All strategies should be reasonably fast
        for stats in performance_results.values():
            assert stats['mean'] < 2.0, "Strategy should complete within 2 seconds"


class TestDatabasePerformance:
    """Test database operation performance"""
    
    @pytest.fixture
    def db_manager(self):
        """Create in-memory database for performance testing"""
        db_manager = DatabaseManager("sqlite:///:memory:")
        db_manager.create_tables()
        return db_manager
    
    def test_bulk_insert_performance(self, db_manager):
        """Test bulk database insert performance"""
        benchmark = PerformanceBenchmark()
        
        with db_manager.get_session() as session:
            # Test bulk insert of positions
            positions = []
            for i in range(1000):
                position = VaultPosition(
                    vault_id=f"vault-{i}",
                    vault_name=f"Vault {i}",
                    chain="bsc" if i % 2 == 0 else "ethereum",
                    amount_deposited=1000.0 + i,
                    shares_owned=950.0 + i,
                    current_value=1050.0 + i,
                    entry_price=1.05,
                    entry_apy=20.0 + (i % 30),
                    status="active"
                )
                positions.append(position)
            
            with benchmark.time_operation("bulk_insert_1000_positions"):
                session.add_all(positions)
                session.commit()
        
        stats = benchmark.get_stats()
        
        # Bulk insert should be fast
        assert stats['mean'] < 5.0, f"Bulk insert took {stats['mean']:.4f}s, expected < 5.0s"
        
        print(f"Bulk Insert Performance: {stats['mean']:.4f}s for 1000 records")
    
    def test_query_performance(self, db_manager):
        """Test database query performance"""
        benchmark = PerformanceBenchmark()
        
        # First populate database
        with db_manager.get_session() as session:
            positions = []
            for i in range(10000):  # Large dataset
                position = VaultPosition(
                    vault_id=f"vault-{i}",
                    vault_name=f"Vault {i}",
                    chain="bsc" if i % 3 == 0 else ("ethereum" if i % 3 == 1 else "polygon"),
                    amount_deposited=1000.0 + i,
                    shares_owned=950.0 + i,
                    current_value=1050.0 + i,
                    entry_price=1.05,
                    entry_apy=20.0 + (i % 30),
                    status="active" if i % 10 != 0 else "closed"
                )
                positions.append(position)
            
            session.add_all(positions)
            session.commit()
            
            # Test various query patterns
            query_tests = [
                ("simple_select", lambda: session.query(VaultPosition).all()),
                ("filtered_by_chain", lambda: session.query(VaultPosition).filter_by(chain="bsc").all()),
                ("filtered_by_status", lambda: session.query(VaultPosition).filter_by(status="active").all()),
                ("complex_filter", lambda: session.query(VaultPosition).filter(
                    VaultPosition.chain == "ethereum",
                    VaultPosition.status == "active",
                    VaultPosition.entry_apy > 25.0
                ).all()),
                ("count_query", lambda: session.query(VaultPosition).count()),
                ("aggregate_query", lambda: session.query(
                    VaultPosition.chain,
                    session.query(VaultPosition.current_value).filter(
                        VaultPosition.chain == VaultPosition.chain
                    ).scalar()
                ).group_by(VaultPosition.chain).all())
            ]
            
            for query_name, query_func in query_tests:
                for i in range(5):  # Run each query 5 times
                    with benchmark.time_operation(f"{query_name}_{i}"):
                        result = query_func()
        
        stats = benchmark.get_stats()
        
        # Queries should be reasonably fast
        assert stats['mean'] < 1.0, f"Average query took {stats['mean']:.4f}s, expected < 1.0s"
        
        # Group results by query type
        query_performance = {}
        for result in benchmark.results:
            query_type = result['operation'].rsplit('_', 1)[0]
            if query_type not in query_performance:
                query_performance[query_type] = []
            query_performance[query_type].append(result['duration'])
        
        for query_type, durations in query_performance.items():
            avg_duration = statistics.mean(durations)
            print(f"{query_type}: {avg_duration:.4f}s avg")
    
    def test_concurrent_database_access(self, db_manager):
        """Test concurrent database access performance"""
        benchmark = PerformanceBenchmark()
        
        def database_operation(thread_id):
            """Database operation to run in thread"""
            with db_manager.get_session() as session:
                start_time = time.perf_counter()
                
                # Insert some data
                position = VaultPosition(
                    vault_id=f"concurrent-vault-{thread_id}",
                    vault_name=f"Concurrent Vault {thread_id}",
                    chain="bsc",
                    amount_deposited=1000.0,
                    shares_owned=950.0,
                    current_value=1050.0,
                    entry_price=1.05,
                    entry_apy=20.0,
                    status="active"
                )
                session.add(position)
                session.commit()
                
                # Query data
                positions = session.query(VaultPosition).filter_by(chain="bsc").all()
                
                end_time = time.perf_counter()
                return end_time - start_time
        
        # Run concurrent operations
        num_threads = 10
        
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(database_operation, i) for i in range(num_threads)]
            
            durations = []
            for future in as_completed(futures):
                duration = future.result()
                durations.append(duration)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # All operations should complete successfully
        assert len(durations) == num_threads
        
        # Average operation time should be reasonable
        avg_duration = statistics.mean(durations)
        assert avg_duration < 2.0, f"Average concurrent operation took {avg_duration:.4f}s"
        
        # Total throughput
        throughput = num_threads / total_time
        print(f"Concurrent Database Performance: {throughput:.2f} ops/sec, {avg_duration:.4f}s avg")


class TestMCPToolsPerformance:
    """Test MCP tools performance"""
    
    @pytest.fixture
    def mcp_tools(self):
        """Create MCP tools for performance testing"""
        return BeefyMCPTools()
    
    @pytest.mark.asyncio
    async def test_mcp_tool_response_time(self, mcp_tools):
        """Test MCP tool response times"""
        benchmark = PerformanceBenchmark()
        
        # Mock the underlying handlers for consistent performance
        with patch.object(mcp_tools.vault_handler, 'search_vaults') as mock_search:
            mock_search.return_value = [
                {"id": "vault-1", "name": "Test Vault", "apy": 25.0, "chain": "bsc"}
            ]
            
            # Test multiple tool calls
            for i in range(50):
                search_params = {
                    "query": "test",
                    "chain": "bsc",
                    "min_apy": 15.0,
                    "max_results": 10
                }
                
                with benchmark.time_operation(f"search_vaults_{i}"):
                    await mcp_tools.search_vaults(search_params)
        
        stats = benchmark.get_stats()
        
        # MCP tools should respond quickly
        assert stats['mean'] < 0.1, f"MCP tool took {stats['mean']:.4f}s, expected < 0.1s"
        assert stats['operations_per_second'] > 100, f"MCP throughput: {stats['operations_per_second']:.2f} ops/sec"
        
        print(f"MCP Tools Performance: {stats['operations_per_second']:.2f} ops/sec")
    
    @pytest.mark.asyncio
    async def test_mcp_concurrent_requests(self, mcp_tools):
        """Test MCP tools under concurrent load"""
        
        # Mock handlers for consistent performance
        with patch.object(mcp_tools.vault_handler, 'search_vaults') as mock_search:
            with patch.object(mcp_tools.vault_handler, 'get_vault_details') as mock_details:
                with patch.object(mcp_tools.portfolio_handler, 'get_portfolio_summary') as mock_summary:
                    
                    mock_search.return_value = [{"id": "vault-1", "apy": 25.0}]
                    mock_details.return_value = {"id": "vault-1", "name": "Test"}
                    mock_summary.return_value = {"total_value": 5000.0}
                    
                    # Different types of requests
                    async def search_request():
                        return await mcp_tools.search_vaults({"query": "test"})
                    
                    async def analyze_request():
                        return await mcp_tools.analyze_vault({"vault_id": "vault-1"})
                    
                    async def portfolio_request():
                        return await mcp_tools.get_portfolio_summary({})
                    
                    # Mix of request types
                    tasks = []
                    for i in range(30):
                        if i % 3 == 0:
                            tasks.append(search_request())
                        elif i % 3 == 1:
                            tasks.append(analyze_request())
                        else:
                            tasks.append(portfolio_request())
                    
                    start_time = time.perf_counter()
                    results = await asyncio.gather(*tasks)
                    end_time = time.perf_counter()
                    
                    total_time = end_time - start_time
                    throughput = len(tasks) / total_time
                    
                    # All requests should complete
                    assert len(results) == len(tasks)
                    
                    # Concurrent throughput should be good
                    assert throughput > 50, f"Concurrent MCP throughput: {throughput:.2f} ops/sec"
                    
                    print(f"Concurrent MCP Performance: {throughput:.2f} ops/sec")


class TestMemoryUsage:
    """Test memory usage under load"""
    
    def test_memory_usage_scaling(self):
        """Test memory usage as data scales"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large data structures
        large_vault_list = []
        for i in range(10000):
            vault_data = {
                "id": f"vault-{i}",
                "name": f"Test Vault {i}",
                "chain": "bsc",
                "apy": 20.0 + (i % 30),
                "tvl": 1000000 + (i * 1000),
                "assets": [f"TOKEN{i}", f"TOKEN{i+1}"],
                "metadata": {
                    "risk_score": i % 100,
                    "platform": f"platform-{i % 10}",
                    "description": f"Description for vault {i}" * 10  # Make it larger
                }
            }
            large_vault_list.append(vault_data)
        
        # Memory after large data creation
        after_creation_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = after_creation_memory - baseline_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 100, f"Memory increased by {memory_increase:.2f}MB for 10k vaults"
        
        print(f"Memory usage: {baseline_memory:.2f}MB -> {after_creation_memory:.2f}MB (+{memory_increase:.2f}MB)")
        
        # Clean up
        del large_vault_list
    
    def test_memory_leak_detection(self):
        """Test for memory leaks in repeated operations"""
        import psutil
        import os
        import gc
        
        process = psutil.Process(os.getpid())
        
        memory_measurements = []
        
        for iteration in range(10):
            # Perform memory-intensive operations
            temp_data = []
            for i in range(1000):
                temp_data.append({
                    "data": f"test_data_{i}" * 100,
                    "numbers": list(range(100))
                })
            
            # Process the data
            processed = [d for d in temp_data if len(d["data"]) > 500]
            
            # Clean up
            del temp_data
            del processed
            gc.collect()
            
            # Measure memory
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_measurements.append(current_memory)
        
        # Memory should not continuously increase
        first_half_avg = statistics.mean(memory_measurements[:5])
        second_half_avg = statistics.mean(memory_measurements[5:])
        
        memory_growth = second_half_avg - first_half_avg
        
        # Small growth acceptable, but not significant leaks
        assert memory_growth < 10, f"Potential memory leak detected: {memory_growth:.2f}MB growth"
        
        print(f"Memory stability: {first_half_avg:.2f}MB -> {second_half_avg:.2f}MB ({memory_growth:+.2f}MB)")


def run_all_benchmarks():
    """Run all performance benchmarks and generate report"""
    print("=" * 60)
    print("CRYPTO TRADING SYSTEM PERFORMANCE BENCHMARKS")
    print("=" * 60)
    
    # Run pytest with specific markers
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure
    ])


if __name__ == "__main__":
    run_all_benchmarks()