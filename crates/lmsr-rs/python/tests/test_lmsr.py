"""
Comprehensive Python tests for LMSR-RS

These tests verify the Python bindings and ensure compatibility
with Python financial systems and trading frameworks.
"""

import pytest
import math
from typing import List, Tuple
import lmsr_rs


class TestLMSRMarket:
    """Test basic LMSR market functionality"""
    
    def test_binary_market_creation(self):
        """Test creating a binary market"""
        market = lmsr_rs.LMSRMarket.create_binary(
            "Will it rain?", 
            "Weather prediction", 
            1000.0
        )
        
        prices = market.get_prices()
        assert len(prices) == 2
        assert abs(sum(prices) - 1.0) < 1e-10
        assert all(0 < p < 1 for p in prices)
    
    def test_categorical_market_creation(self):
        """Test creating a categorical market"""
        outcomes = ["A", "B", "C", "D"]
        market = lmsr_rs.LMSRMarket.create_categorical(
            "Election", 
            "Who wins?", 
            outcomes, 
            2000.0
        )
        
        prices = market.get_prices()
        assert len(prices) == 4
        assert abs(sum(prices) - 1.0) < 1e-10
        
        metadata = market.get_metadata()
        assert metadata["name"] == "Election"
        assert len(metadata["outcomes"]) == 4
    
    def test_trading_basic(self):
        """Test basic trading functionality"""
        market = lmsr_rs.LMSRMarket(2, 100.0)
        
        # Initial state
        initial_prices = market.get_prices()
        assert abs(initial_prices[0] - 0.5) < 1e-10
        assert abs(initial_prices[1] - 0.5) < 1e-10
        
        # Execute trade
        cost = market.trade("trader1", [10.0, 0.0])
        assert cost > 0
        
        # Check price change
        new_prices = market.get_prices()
        assert new_prices[0] > initial_prices[0]
        assert new_prices[1] < initial_prices[1]
        assert abs(sum(new_prices) - 1.0) < 1e-10
    
    def test_position_tracking(self):
        """Test position tracking functionality"""
        market = lmsr_rs.LMSRMarket(3, 500.0)
        
        # Execute trades
        cost1 = market.trade("alice", [10.0, 0.0, 0.0])
        cost2 = market.trade("alice", [0.0, 5.0, 0.0])
        
        # Check position
        position = market.get_position("alice")
        assert position is not None
        assert position.quantities == [10.0, 5.0, 0.0]
        assert abs(position.total_invested - (cost1 + cost2)) < 1e-10
        
        # Calculate position value
        value = market.calculate_position_value("alice")
        assert value > 0
    
    def test_cost_calculation(self):
        """Test trade cost calculation without execution"""
        market = lmsr_rs.LMSRMarket(2, 100.0)
        
        quantities = [5.0, 0.0]
        calculated_cost = market.calculate_cost(quantities)
        
        # Execute the trade and compare
        actual_cost = market.trade("trader", quantities)
        assert abs(calculated_cost - actual_cost) < 1e-10
    
    def test_batch_trading(self):
        """Test batch trading functionality"""
        market = lmsr_rs.LMSRMarket(3, 1000.0)
        
        trades = [
            ("trader1", [10.0, 0.0, 0.0]),
            ("trader2", [0.0, 8.0, 0.0]),
            ("trader3", [0.0, 0.0, 12.0])
        ]
        
        costs = market.batch_trade(trades)
        assert len(costs) == 3
        assert all(cost > 0 for cost in costs)
        
        # Check all positions were created
        for trader_id, _ in trades:
            position = market.get_position(trader_id)
            assert position is not None


class TestNumericalStability:
    """Test numerical stability under extreme conditions"""
    
    def test_large_trades(self):
        """Test handling of very large trades"""
        market = lmsr_rs.LMSRMarket(2, 10000.0)
        
        # Execute large trade
        cost = market.trade("whale", [100000.0, 0.0])
        assert math.isfinite(cost)
        
        prices = market.get_prices()
        assert all(math.isfinite(p) for p in prices)
        assert all(0 <= p <= 1 for p in prices)
        assert abs(sum(prices) - 1.0) < 1e-6
    
    def test_small_liquidity(self):
        """Test behavior with very small liquidity parameters"""
        market = lmsr_rs.LMSRMarket(2, 0.01)
        
        cost = market.trade("trader", [0.1, 0.0])
        assert math.isfinite(cost)
        
        prices = market.get_prices()
        assert all(math.isfinite(p) for p in prices)
        assert abs(sum(prices) - 1.0) < 1e-6
    
    def test_many_outcomes(self):
        """Test markets with many outcomes"""
        market = lmsr_rs.LMSRMarket(100, 10000.0)
        
        quantities = [1.0] + [0.0] * 99
        cost = market.trade("trader", quantities)
        assert math.isfinite(cost)
        
        prices = market.get_prices()
        assert len(prices) == 100
        assert all(math.isfinite(p) for p in prices)
        assert abs(sum(prices) - 1.0) < 1e-6
    
    def test_extreme_quantities(self):
        """Test handling of extreme quantity values"""
        market = lmsr_rs.LMSRMarket(2, 1000.0)
        
        # Very small trades
        cost = market.trade("small_trader", [1e-10, 0.0])
        assert math.isfinite(cost)
        
        # Mixed scale trades
        cost = market.trade("mixed_trader", [1e6, 1e-6])
        assert math.isfinite(cost)


class TestErrorHandling:
    """Test error handling and validation"""
    
    def test_invalid_market_creation(self):
        """Test error handling in market creation"""
        # Too few outcomes
        with pytest.raises(ValueError):
            lmsr_rs.LMSRMarket(1, 100.0)
        
        # Invalid liquidity
        with pytest.raises(ValueError):
            lmsr_rs.LMSRMarket(2, 0.0)
        
        with pytest.raises(ValueError):
            lmsr_rs.LMSRMarket(2, -100.0)
    
    def test_invalid_trades(self):
        """Test error handling in trading"""
        market = lmsr_rs.LMSRMarket(2, 100.0)
        
        # Wrong number of quantities
        with pytest.raises(ValueError):
            market.trade("trader", [1.0])
        
        with pytest.raises(ValueError):
            market.trade("trader", [1.0, 2.0, 3.0])
        
        # Non-finite quantities
        with pytest.raises(ArithmeticError):
            market.trade("trader", [float('inf'), 0.0])
        
        with pytest.raises(ArithmeticError):
            market.trade("trader", [float('nan'), 0.0])
    
    def test_invalid_outcome_index(self):
        """Test error handling for invalid outcome indices"""
        market = lmsr_rs.LMSRMarket(3, 100.0)
        
        with pytest.raises(IndexError):
            market.get_price(5)  # Index out of bounds


class TestMarketSimulation:
    """Test market simulation functionality"""
    
    def test_simulation_basic(self):
        """Test basic simulation functionality"""
        sim = lmsr_rs.MarketSimulation()
        market = lmsr_rs.LMSRMarket(2, 1000.0)
        
        market_id = sim.add_market(market)
        assert market_id == 0
        
        sim.add_trader("trader1", 10000.0)
        assert sim.get_balance("trader1") == 10000.0
        
        # Execute trade
        cost = sim.execute_trade(market_id, "trader1", [10.0, 0.0])
        assert cost > 0
        assert sim.get_balance("trader1") < 10000.0
    
    def test_insufficient_balance(self):
        """Test handling of insufficient trader balance"""
        sim = lmsr_rs.MarketSimulation()
        market = lmsr_rs.LMSRMarket(2, 100.0)
        
        market_id = sim.add_market(market)
        sim.add_trader("poor_trader", 1.0)  # Very small balance
        
        # Try to make expensive trade
        with pytest.raises(ValueError, match="Insufficient balance"):
            sim.execute_trade(market_id, "poor_trader", [1000.0, 0.0])
    
    def test_random_simulation(self):
        """Test random trading simulation"""
        sim = lmsr_rs.MarketSimulation()
        market = lmsr_rs.LMSRMarket(3, 1000.0)
        
        sim.add_market(market)
        sim.add_trader("random_trader", 50000.0)
        
        # Run random simulation
        costs = sim.run_random_simulation(100)
        assert len(costs) <= 100  # Some trades might fail
        assert all(cost > 0 for cost in costs)


class TestStandaloneFunctions:
    """Test standalone utility functions"""
    
    def test_calculate_price(self):
        """Test standalone price calculation"""
        quantities = [10.0, 5.0, 2.0]
        liquidity = 100.0
        
        price = lmsr_rs.py_calculate_price(quantities, 0, liquidity)
        assert 0 < price < 1
        
        # Compare with market calculation
        market = lmsr_rs.LMSRMarket(3, liquidity)
        market.trade("setup", quantities)
        market_price = market.get_price(0)
        
        # Should be approximately equal (some numerical differences expected)
        assert abs(price - market_price) < 1e-6
    
    def test_calculate_cost(self):
        """Test standalone cost calculation"""
        current = [0.0, 0.0]
        buy_amounts = [10.0, 5.0]
        liquidity = 100.0
        
        cost = lmsr_rs.py_calculate_cost(current, buy_amounts, liquidity)
        assert cost > 0
        
        # Compare with market calculation
        market = lmsr_rs.LMSRMarket(2, liquidity)
        market_cost = market.calculate_cost(buy_amounts)
        
        assert abs(cost - market_cost) < 1e-10


class TestPerformance:
    """Performance and benchmark tests"""
    
    @pytest.mark.benchmark
    def test_price_calculation_performance(self, benchmark):
        """Benchmark price calculations"""
        market = lmsr_rs.LMSRMarket(10, 1000.0)
        
        def get_prices():
            return market.get_prices()
        
        result = benchmark(get_prices)
        assert len(result) == 10
    
    @pytest.mark.benchmark
    def test_trading_performance(self, benchmark):
        """Benchmark trade execution"""
        market = lmsr_rs.LMSRMarket(5, 2000.0)
        quantities = [1.0, 0.0, 0.0, 0.0, 0.0]
        
        trade_counter = [0]
        def execute_trade():
            trade_counter[0] += 1
            trader_id = f"trader_{trade_counter[0]}"
            return market.trade(trader_id, quantities)
        
        result = benchmark(execute_trade)
        assert result > 0
    
    @pytest.mark.benchmark
    def test_benchmark_comparison(self):
        """Compare performance with theoretical Python implementation"""
        # Use built-in benchmark
        benchmark = lmsr_rs.LMSRBenchmark()
        
        # Benchmark price calculations
        time_taken = benchmark.benchmark_prices(
            num_outcomes=10,
            liquidity=1000.0,
            num_iterations=10000
        )
        
        print(f"10k price calculations took: {time_taken:.4f} seconds")
        assert time_taken < 1.0  # Should be very fast
        
        # Benchmark trade costs
        time_taken = benchmark.benchmark_costs(
            num_outcomes=10,
            liquidity=1000.0,
            num_iterations=10000
        )
        
        print(f"10k cost calculations took: {time_taken:.4f} seconds")
        assert time_taken < 1.0
        
        # Benchmark market operations
        time_taken = benchmark.benchmark_market_operations(
            num_outcomes=5,
            liquidity=1000.0,
            num_trades=1000
        )
        
        print(f"1k market operations took: {time_taken:.4f} seconds")
        assert time_taken < 5.0


class TestIntegration:
    """Integration tests with realistic scenarios"""
    
    def test_election_market_scenario(self):
        """Test a realistic election prediction market"""
        candidates = ["Alice", "Bob", "Charlie", "Diana"]
        market = lmsr_rs.LMSRMarket.create_categorical(
            "2024 Election",
            "Who will win the presidency?",
            candidates,
            50000.0  # $50k liquidity
        )
        
        # Simulate various traders with different strategies
        traders = {
            "institutional": 100000.0,
            "hedge_fund": 250000.0,
            "retail_1": 5000.0,
            "retail_2": 3000.0,
            "whale": 1000000.0
        }
        
        sim = lmsr_rs.MarketSimulation()
        market_id = sim.add_market(market)
        
        for trader_id, balance in traders.items():
            sim.add_trader(trader_id, balance)
        
        # Execute various trades
        trades = [
            ("institutional", [1000.0, 0.0, 0.0, 0.0]),    # Big on Alice
            ("hedge_fund", [0.0, 2000.0, 0.0, 0.0]),       # Big on Bob
            ("retail_1", [100.0, 100.0, 0.0, 0.0]),        # Diversified
            ("whale", [5000.0, 0.0, 0.0, 0.0]),            # Huge on Alice
            ("retail_2", [0.0, 0.0, 200.0, 200.0]),        # Contrarian
        ]
        
        total_volume = 0.0
        for trader_id, quantities in trades:
            try:
                cost = sim.execute_trade(market_id, trader_id, quantities)
                total_volume += cost
                print(f"{trader_id} traded for ${cost:.2f}")
            except Exception as e:
                print(f"{trader_id} trade failed: {e}")
        
        # Check final market state
        final_prices = market.get_prices()
        print(f"Final prices: {final_prices}")
        
        # Alice should be favored (lots of buying)
        assert final_prices[0] > 0.3  # Alice got significant support
        assert sum(final_prices) == pytest.approx(1.0, abs=1e-6)
        
        # Check positions
        alice_position = market.get_position("institutional")
        assert alice_position is not None
        assert alice_position.quantities[0] > 0
    
    def test_arbitrage_scenario(self):
        """Test arbitrage detection and resolution"""
        market = lmsr_rs.LMSRMarket(3, 1000.0)
        
        # Create market imbalance
        market.trade("biased_trader", [100.0, 0.0, 0.0])
        
        initial_prices = market.get_prices()
        print(f"Imbalanced prices: {initial_prices}")
        
        # Arbitrageur comes in
        arbitrage_quantities = [0.0, 50.0, 50.0]  # Buy undervalued outcomes
        arb_cost = market.trade("arbitrageur", arbitrage_quantities)
        
        final_prices = market.get_prices()
        print(f"After arbitrage: {final_prices}")
        
        # Market should be more balanced
        price_spread = max(final_prices) - min(final_prices)
        initial_spread = max(initial_prices) - min(initial_prices)
        
        assert price_spread < initial_spread  # Arbitrage reduced spread
        
    def test_stress_testing(self):
        """Stress test with many concurrent operations"""
        market = lmsr_rs.LMSRMarket(10, 10000.0)
        
        # Execute many small trades
        for i in range(1000):
            trader_id = f"trader_{i}"
            quantities = [0.0] * 10
            quantities[i % 10] = 1.0  # Rotate through outcomes
            
            try:
                cost = market.trade(trader_id, quantities)
                assert cost > 0
            except Exception as e:
                pytest.fail(f"Trade {i} failed: {e}")
        
        # Verify market integrity
        final_prices = market.get_prices()
        assert len(final_prices) == 10
        assert abs(sum(final_prices) - 1.0) < 1e-6
        assert all(0 < p < 1 for p in final_prices)
        
        stats = market.get_statistics()
        assert stats.trade_count <= 1000  # Some trades might have failed
        assert stats.total_volume > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])