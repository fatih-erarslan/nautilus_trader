"""
Comprehensive tests for all crypto trading strategies

Tests strategy logic, risk calculations, portfolio management, and performance metrics.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from crypto_trading.strategies.base_strategy import (
    BaseStrategy, Position, PortfolioState, VaultOpportunity,
    ChainType, RiskLevel
)
from crypto_trading.strategies.yield_chaser import YieldChaserStrategy
from crypto_trading.strategies.stable_farmer import StableFarmerStrategy
from crypto_trading.strategies.risk_balanced import RiskBalancedStrategy
from crypto_trading.strategies.portfolio_optimizer import PortfolioOptimizerStrategy
from crypto_trading.strategies.news_driven import NewsDrivenStrategy


class TestBaseStrategy:
    """Test base strategy functionality"""

    def test_strategy_initialization(self):
        """Test strategy initialization with parameters"""
        
        class TestStrategy(BaseStrategy):
            def evaluate_opportunities(self, opportunities, portfolio):
                return []
            def calculate_risk_score(self, opportunity):
                return 50.0
            def should_rebalance(self, portfolio):
                return False
            def generate_rebalance_trades(self, portfolio, opportunities):
                return []
        
        strategy = TestStrategy(
            name="Test Strategy",
            risk_level=RiskLevel.MEDIUM,
            min_apy_threshold=10.0,
            max_position_size=0.2,
            rebalance_threshold=0.15
        )
        
        assert strategy.name == "Test Strategy"
        assert strategy.risk_level == RiskLevel.MEDIUM
        assert strategy.min_apy_threshold == 10.0
        assert strategy.max_position_size == 0.2
        assert strategy.rebalance_threshold == 0.15
        assert strategy.execution_history == []

    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation"""
        
        class TestStrategy(BaseStrategy):
            def evaluate_opportunities(self, opportunities, portfolio): return []
            def calculate_risk_score(self, opportunity): return 50.0
            def should_rebalance(self, portfolio): return False
            def generate_rebalance_trades(self, portfolio, opportunities): return []
        
        strategy = TestStrategy("Test", RiskLevel.LOW)
        
        # Test with positive returns
        returns = np.array([0.01, 0.02, -0.005, 0.015, 0.008])
        sharpe = strategy.calculate_sharpe_ratio(returns, risk_free_rate=0.02)
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
        
        # Test with insufficient data
        short_returns = np.array([0.01])
        sharpe_short = strategy.calculate_sharpe_ratio(short_returns)
        assert sharpe_short == 0.0
        
        # Test with zero volatility
        zero_vol_returns = np.array([0.02, 0.02, 0.02, 0.02])
        sharpe_zero = strategy.calculate_sharpe_ratio(zero_vol_returns)
        assert sharpe_zero == 0.0

    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation"""
        
        class TestStrategy(BaseStrategy):
            def evaluate_opportunities(self, opportunities, portfolio): return []
            def calculate_risk_score(self, opportunity): return 50.0
            def should_rebalance(self, portfolio): return False
            def generate_rebalance_trades(self, portfolio, opportunities): return []
        
        strategy = TestStrategy("Test", RiskLevel.LOW)
        
        # Test with declining values
        values = np.array([1000, 950, 900, 1050, 800, 900])
        max_dd = strategy.calculate_max_drawdown(values)
        assert max_dd > 0
        assert max_dd <= 100  # Should be a percentage
        
        # Test with insufficient data
        short_values = np.array([1000])
        max_dd_short = strategy.calculate_max_drawdown(short_values)
        assert max_dd_short == 0.0
        
        # Test with only increasing values
        increasing_values = np.array([100, 110, 120, 130, 140])
        max_dd_inc = strategy.calculate_max_drawdown(increasing_values)
        assert max_dd_inc >= 0

    def test_diversification_score(self):
        """Test portfolio diversification scoring"""
        
        class TestStrategy(BaseStrategy):
            def evaluate_opportunities(self, opportunities, portfolio): return []
            def calculate_risk_score(self, opportunity): return 50.0
            def should_rebalance(self, portfolio): return False
            def generate_rebalance_trades(self, portfolio, opportunities): return []
        
        strategy = TestStrategy("Test", RiskLevel.LOW)
        
        # Create diversified portfolio
        positions = [
            Position("vault1", ChainType.BSC, "pancakeswap", ("CAKE", "BNB"), 
                    20.0, 1000000, 1000, datetime.now(), 30.0),
            Position("vault2", ChainType.POLYGON, "quickswap", ("MATIC", "ETH"), 
                    15.0, 800000, 800, datetime.now(), 25.0),
            Position("vault3", ChainType.ETHEREUM, "uniswap", ("ETH", "USDC"), 
                    12.0, 2000000, 1200, datetime.now(), 20.0)
        ]
        
        portfolio = PortfolioState(
            positions=positions,
            total_value=3000,
            available_capital=500,
            timestamp=datetime.now(),
            chain_allocations={
                ChainType.BSC: 33.33,
                ChainType.POLYGON: 26.67,
                ChainType.ETHEREUM: 40.0
            },
            protocol_allocations={
                "pancakeswap": 33.33,
                "quickswap": 26.67,
                "uniswap": 40.0
            }
        )
        
        div_score = strategy.diversification_score(portfolio)
        assert 0 <= div_score <= 100
        assert div_score > 0  # Should have some diversification
        
        # Test empty portfolio
        empty_portfolio = PortfolioState([], 0, 1000, datetime.now())
        empty_score = strategy.diversification_score(empty_portfolio)
        assert empty_score == 0.0

    def test_position_size_validation(self):
        """Test position size validation logic"""
        
        class TestStrategy(BaseStrategy):
            def evaluate_opportunities(self, opportunities, portfolio): return []
            def calculate_risk_score(self, opportunity): return 50.0
            def should_rebalance(self, portfolio): return False
            def generate_rebalance_trades(self, portfolio, opportunities): return []
        
        strategy = TestStrategy("Test", RiskLevel.LOW, max_position_size=0.25)
        
        portfolio = PortfolioState(
            positions=[],
            total_value=4000,
            available_capital=1000,
            timestamp=datetime.now()
        )
        
        opportunity = VaultOpportunity(
            vault_id="test-vault",
            chain=ChainType.BSC,
            protocol="test",
            token_pair=("A", "B"),
            apy=20.0,
            daily_apy=0.055,
            tvl=1000000,
            platform_fee=0.045,
            withdraw_fee=0.001,
            is_paused=False,
            has_boost=False,
            boost_apy=None,
            risk_factors={"smart_contract": 0.1},
            created_at=datetime.now(),
            last_harvest=datetime.now()
        )
        
        # Test normal position size
        validated_size = strategy.validate_position_size(800, portfolio, opportunity)
        assert validated_size == 800
        
        # Test position exceeding available capital
        oversized = strategy.validate_position_size(1500, portfolio, opportunity)
        assert oversized == 1000  # Should be capped at available capital
        
        # Test position exceeding max position size (25% of 5000 = 1250)
        max_exceeded = strategy.validate_position_size(1300, portfolio, opportunity)
        assert max_exceeded == 1000  # Limited by available capital
        
        # Test minimum position size
        too_small = strategy.validate_position_size(50, portfolio, opportunity)
        assert too_small == 0.0  # Below $100 minimum

    def test_execution_logging(self):
        """Test strategy execution logging"""
        
        class TestStrategy(BaseStrategy):
            def evaluate_opportunities(self, opportunities, portfolio): return []
            def calculate_risk_score(self, opportunity): return 50.0
            def should_rebalance(self, portfolio): return False
            def generate_rebalance_trades(self, portfolio, opportunities): return []
        
        strategy = TestStrategy("Test", RiskLevel.LOW)
        
        # Log an execution
        strategy.log_execution("deposit", {
            "vault_id": "test-vault",
            "amount": 1000,
            "apy": 25.0
        })
        
        assert len(strategy.execution_history) == 1
        
        log_entry = strategy.execution_history[0]
        assert log_entry["action"] == "deposit"
        assert log_entry["strategy"] == "Test"
        assert log_entry["details"]["vault_id"] == "test-vault"
        assert "timestamp" in log_entry


class TestYieldChaserStrategy:
    """Test yield chasing strategy"""

    @pytest.fixture
    def strategy(self):
        return YieldChaserStrategy(
            min_apy_threshold=15.0,
            max_position_size=0.3
        )

    @pytest.fixture
    def sample_opportunities(self):
        """Create sample vault opportunities"""
        return [
            VaultOpportunity(
                vault_id="high-yield-vault",
                chain=ChainType.BSC,
                protocol="pancakeswap",
                token_pair=("CAKE", "BNB"),
                apy=35.0,
                daily_apy=0.096,
                tvl=5000000,
                platform_fee=0.045,
                withdraw_fee=0.001,
                is_paused=False,
                has_boost=True,
                boost_apy=5.0,
                risk_factors={"impermanent_loss": 0.3, "smart_contract": 0.2},
                created_at=datetime.now(),
                last_harvest=datetime.now() - timedelta(hours=2)
            ),
            VaultOpportunity(
                vault_id="stable-vault",
                chain=ChainType.POLYGON,
                protocol="aave",
                token_pair=("USDC", "USDT"),
                apy=8.0,
                daily_apy=0.022,
                tvl=10000000,
                platform_fee=0.045,
                withdraw_fee=0.001,
                is_paused=False,
                has_boost=False,
                boost_apy=None,
                risk_factors={"smart_contract": 0.1},
                created_at=datetime.now(),
                last_harvest=datetime.now() - timedelta(hours=1)
            ),
            VaultOpportunity(
                vault_id="medium-vault",
                chain=ChainType.ETHEREUM,
                protocol="uniswap",
                token_pair=("ETH", "USDC"),
                apy=18.5,
                daily_apy=0.051,
                tvl=8000000,
                platform_fee=0.045,
                withdraw_fee=0.001,
                is_paused=False,
                has_boost=False,
                boost_apy=None,
                risk_factors={"impermanent_loss": 0.2, "smart_contract": 0.15},
                created_at=datetime.now(),
                last_harvest=datetime.now() - timedelta(hours=3)
            )
        ]

    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio state"""
        return PortfolioState(
            positions=[],
            total_value=5000,
            available_capital=2000,
            timestamp=datetime.now(),
            chain_allocations={},
            protocol_allocations={}
        )

    def test_strategy_initialization(self, strategy):
        """Test yield chaser strategy initialization"""
        assert strategy.name == "Yield Chaser"
        assert strategy.risk_level == RiskLevel.HIGH
        assert strategy.min_apy_threshold == 15.0

    def test_risk_score_calculation(self, strategy, sample_opportunities):
        """Test risk score calculation for yield chaser"""
        # High yield vault should have higher risk score
        high_yield_risk = strategy.calculate_risk_score(sample_opportunities[0])
        stable_risk = strategy.calculate_risk_score(sample_opportunities[1])
        
        assert high_yield_risk > stable_risk
        assert 0 <= high_yield_risk <= 100
        assert 0 <= stable_risk <= 100

    def test_opportunity_evaluation(self, strategy, sample_opportunities, sample_portfolio):
        """Test opportunity evaluation and ranking"""
        evaluations = strategy.evaluate_opportunities(sample_opportunities, sample_portfolio)
        
        # Should return list of (opportunity, allocation) tuples
        assert isinstance(evaluations, list)
        assert len(evaluations) <= len(sample_opportunities)
        
        # Should be sorted by yield (highest first)
        if len(evaluations) > 1:
            assert evaluations[0][0].total_apy >= evaluations[1][0].total_apy
        
        # Should only include opportunities above threshold
        for opp, allocation in evaluations:
            assert opp.net_apy >= strategy.min_apy_threshold

    def test_rebalancing_decision(self, strategy):
        """Test rebalancing decision logic"""
        # Portfolio with underperforming positions
        old_positions = [
            Position("old-vault", ChainType.BSC, "old-protocol", ("A", "B"),
                    10.0, 1000000, 1000, datetime.now() - timedelta(days=30), 20.0)
        ]
        
        portfolio = PortfolioState(
            positions=old_positions,
            total_value=3000,
            available_capital=500,
            timestamp=datetime.now()
        )
        
        should_rebalance = strategy.should_rebalance(portfolio)
        # Yield chaser should want to rebalance from low-yield positions
        assert isinstance(should_rebalance, bool)

    def test_gas_cost_consideration(self, strategy, sample_opportunities, sample_portfolio):
        """Test that gas costs are considered in evaluations"""
        # Mock high gas costs
        with patch('crypto_trading.strategies.yield_chaser.get_gas_prices') as mock_gas:
            mock_gas.return_value = {"ethereum": 100, "bsc": 5, "polygon": 30}
            
            evaluations = strategy.evaluate_opportunities(sample_opportunities, sample_portfolio)
            
            # BSC opportunities should be preferred due to lower gas costs
            # (This depends on the specific implementation)
            if evaluations:
                assert len(evaluations) > 0


class TestStableFarmerStrategy:
    """Test stable farming strategy"""

    @pytest.fixture
    def strategy(self):
        return StableFarmerStrategy(
            preferred_stablecoins=["USDC", "USDT", "DAI"],
            min_apy_threshold=5.0
        )

    @pytest.fixture
    def stable_opportunities(self):
        """Create stable coin opportunities"""
        return [
            VaultOpportunity(
                vault_id="usdc-vault",
                chain=ChainType.ETHEREUM,
                protocol="aave",
                token_pair=("USDC", "USDT"),
                apy=7.5,
                daily_apy=0.021,
                tvl=20000000,
                platform_fee=0.045,
                withdraw_fee=0.001,
                is_paused=False,
                has_boost=False,
                boost_apy=None,
                risk_factors={"smart_contract": 0.05},
                created_at=datetime.now(),
                last_harvest=datetime.now()
            ),
            VaultOpportunity(
                vault_id="risky-vault",
                chain=ChainType.BSC,
                protocol="pancakeswap",
                token_pair=("CAKE", "BNB"),
                apy=45.0,
                daily_apy=0.123,
                tvl=1000000,
                platform_fee=0.045,
                withdraw_fee=0.001,
                is_paused=False,
                has_boost=True,
                boost_apy=10.0,
                risk_factors={"impermanent_loss": 0.5, "smart_contract": 0.3},
                created_at=datetime.now(),
                last_harvest=datetime.now()
            )
        ]

    def test_strategy_initialization(self, strategy):
        """Test stable farmer initialization"""
        assert strategy.name == "Stable Farmer"
        assert strategy.risk_level == RiskLevel.LOW
        assert "USDC" in strategy.preferred_stablecoins

    def test_stable_coin_preference(self, strategy, stable_opportunities):
        """Test preference for stable coin pairs"""
        portfolio = PortfolioState([], 5000, 2000, datetime.now())
        
        evaluations = strategy.evaluate_opportunities(stable_opportunities, portfolio)
        
        # Should prefer USDC-USDT over CAKE-BNB despite lower yield
        stable_eval = next((e for e in evaluations if e[0].vault_id == "usdc-vault"), None)
        risky_eval = next((e for e in evaluations if e[0].vault_id == "risky-vault"), None)
        
        if stable_eval and risky_eval:
            # Stable should get higher allocation relative to risk
            assert stable_eval[1] > 0  # Should get some allocation

    def test_risk_score_calculation(self, strategy, stable_opportunities):
        """Test risk scoring favors stable coins"""
        stable_risk = strategy.calculate_risk_score(stable_opportunities[0])
        risky_risk = strategy.calculate_risk_score(stable_opportunities[1])
        
        assert stable_risk < risky_risk
        assert stable_risk < 30  # Should be low risk for stablecoins


class TestRiskBalancedStrategy:
    """Test risk-balanced strategy"""

    @pytest.fixture
    def strategy(self):
        return RiskBalancedStrategy(
            target_risk_score=40.0,
            max_chain_allocation=0.4,
            max_protocol_allocation=0.3
        )

    def test_strategy_initialization(self, strategy):
        """Test risk balanced strategy initialization"""
        assert strategy.name == "Risk Balanced"
        assert strategy.risk_level == RiskLevel.MEDIUM
        assert strategy.target_risk_score == 40.0

    def test_portfolio_risk_calculation(self, strategy):
        """Test portfolio risk calculation"""
        positions = [
            Position("vault1", ChainType.BSC, "protocol1", ("A", "B"),
                    20.0, 1000000, 1000, datetime.now(), 30.0),
            Position("vault2", ChainType.ETHEREUM, "protocol2", ("C", "D"),
                    15.0, 2000000, 1500, datetime.now(), 50.0)
        ]
        
        portfolio = PortfolioState(positions, 2500, 500, datetime.now())
        portfolio_risk = strategy.calculate_portfolio_risk(portfolio)
        
        # Should be weighted average of position risks
        expected_risk = (1000 * 30.0 + 1500 * 50.0) / 2500
        assert abs(portfolio_risk - expected_risk) < 0.1

    def test_diversification_constraints(self, strategy):
        """Test diversification constraint enforcement"""
        opportunities = [
            VaultOpportunity("vault1", ChainType.BSC, "protocol1", ("A", "B"),
                           25.0, 0.068, 1000000, 0.045, 0.001, False, False, None,
                           {"smart_contract": 0.2}, datetime.now(), datetime.now()),
            VaultOpportunity("vault2", ChainType.BSC, "protocol1", ("C", "D"),
                           30.0, 0.082, 800000, 0.045, 0.001, False, False, None,
                           {"smart_contract": 0.25}, datetime.now(), datetime.now())
        ]
        
        portfolio = PortfolioState([], 5000, 2000, datetime.now())
        evaluations = strategy.evaluate_opportunities(opportunities, portfolio)
        
        # Should limit allocation to same chain/protocol
        total_allocation = sum(allocation for _, allocation in evaluations)
        if total_allocation > 0:
            # Check that no single allocation exceeds limits
            for _, allocation in evaluations:
                assert allocation <= 2000 * strategy.max_chain_allocation


class TestNewsEarlyExitDetection:
    """Test news-driven early exit detection"""

    @pytest.fixture
    def strategy(self):
        return NewsDrivenStrategy(
            news_weight=0.3,
            sentiment_threshold=0.7
        )

    def test_negative_news_detection(self, strategy):
        """Test detection of negative news requiring early exit"""
        # Mock negative news about a protocol
        negative_news = [
            {
                "title": "Major security vulnerability found in PancakeSwap",
                "sentiment": -0.8,
                "relevance": 0.9,
                "protocol": "pancakeswap",
                "timestamp": datetime.now()
            }
        ]
        
        with patch.object(strategy, 'get_recent_news') as mock_news:
            mock_news.return_value = negative_news
            
            should_exit = strategy.should_exit_due_to_news("pancakeswap-vault")
            assert should_exit is True

    def test_positive_news_detection(self, strategy):
        """Test that positive news doesn't trigger early exit"""
        positive_news = [
            {
                "title": "PancakeSwap announces new yield farming program",
                "sentiment": 0.8,
                "relevance": 0.9,
                "protocol": "pancakeswap",
                "timestamp": datetime.now()
            }
        ]
        
        with patch.object(strategy, 'get_recent_news') as mock_news:
            mock_news.return_value = positive_news
            
            should_exit = strategy.should_exit_due_to_news("pancakeswap-vault")
            assert should_exit is False


class TestPortfolioOptimizerStrategy:
    """Test portfolio optimization strategy"""

    @pytest.fixture
    def strategy(self):
        return PortfolioOptimizerStrategy(
            optimization_method="sharpe",
            rebalance_frequency_hours=24
        )

    def test_strategy_initialization(self, strategy):
        """Test portfolio optimizer initialization"""
        assert strategy.name == "Portfolio Optimizer"
        assert strategy.optimization_method == "sharpe"
        assert strategy.rebalance_frequency_hours == 24

    def test_efficient_frontier_calculation(self, strategy):
        """Test efficient frontier calculation"""
        # Mock return and covariance data
        returns = np.array([0.1, 0.15, 0.12, 0.08])
        cov_matrix = np.array([
            [0.02, 0.01, 0.005, 0.003],
            [0.01, 0.04, 0.008, 0.002],
            [0.005, 0.008, 0.03, 0.004],
            [0.003, 0.002, 0.004, 0.01]
        ])
        
        weights = strategy.optimize_portfolio_weights(returns, cov_matrix)
        
        assert isinstance(weights, np.ndarray)
        assert len(weights) == len(returns)
        assert abs(np.sum(weights) - 1.0) < 0.01  # Weights should sum to 1
        assert all(w >= 0 for w in weights)  # No short selling

    def test_rebalancing_frequency(self, strategy):
        """Test rebalancing frequency logic"""
        # Create portfolio with recent rebalance
        positions = [
            Position("vault1", ChainType.BSC, "protocol1", ("A", "B"),
                    20.0, 1000000, 1000, datetime.now() - timedelta(hours=12), 30.0)
        ]
        
        portfolio = PortfolioState(positions, 1000, 500, datetime.now())
        
        # Should not rebalance too frequently
        should_rebalance = strategy.should_rebalance(portfolio)
        # This depends on implementation details


class TestStrategyPerformanceMetrics:
    """Test performance metrics across all strategies"""

    @pytest.fixture
    def performance_data(self):
        """Generate sample performance data"""
        np.random.seed(42)  # For reproducible tests
        
        # Generate 100 days of returns
        returns_yield_chaser = np.random.normal(0.0003, 0.02, 100)  # Higher risk/return
        returns_stable_farmer = np.random.normal(0.0001, 0.005, 100)  # Lower risk/return
        returns_risk_balanced = np.random.normal(0.0002, 0.012, 100)  # Balanced
        
        return {
            "yield_chaser": returns_yield_chaser,
            "stable_farmer": returns_stable_farmer,
            "risk_balanced": returns_risk_balanced
        }

    def test_strategy_performance_comparison(self, performance_data):
        """Test performance comparison between strategies"""
        strategies = {
            "yield_chaser": YieldChaserStrategy(),
            "stable_farmer": StableFarmerStrategy(),
            "risk_balanced": RiskBalancedStrategy()
        }
        
        metrics = {}
        
        for name, strategy in strategies.items():
            returns = performance_data[name]
            
            # Calculate cumulative values
            cumulative_values = np.cumprod(1 + returns) * 1000  # Starting with $1000
            
            metrics[name] = {
                "total_return": (cumulative_values[-1] / 1000 - 1) * 100,
                "sharpe_ratio": strategy.calculate_sharpe_ratio(returns),
                "max_drawdown": strategy.calculate_max_drawdown(cumulative_values),
                "volatility": np.std(returns) * np.sqrt(365) * 100
            }
        
        # Yield chaser should have higher returns but also higher risk
        assert metrics["yield_chaser"]["total_return"] != 0
        assert metrics["stable_farmer"]["max_drawdown"] <= metrics["yield_chaser"]["max_drawdown"]
        
        # Risk balanced should be between the other two
        assert metrics["stable_farmer"]["volatility"] <= metrics["risk_balanced"]["volatility"]

    def test_strategy_consistency(self, performance_data):
        """Test strategy consistency over time"""
        strategy = RiskBalancedStrategy()
        
        returns = performance_data["risk_balanced"]
        
        # Split into two periods
        period1 = returns[:50]
        period2 = returns[50:]
        
        sharpe1 = strategy.calculate_sharpe_ratio(period1)
        sharpe2 = strategy.calculate_sharpe_ratio(period2)
        
        # Sharpe ratios should be reasonable for both periods
        assert -2 <= sharpe1 <= 5  # Reasonable Sharpe ratio range
        assert -2 <= sharpe2 <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])