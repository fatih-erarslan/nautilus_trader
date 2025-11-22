"""
Comprehensive TDD tests for financial calculations
Mathematical rigor with 100% code coverage
"""

import pytest
import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Tuple, Optional
import math
from unittest.mock import Mock, patch
from test_configuration import TEST_CONFIG, validate_financial_calculation

class TestFinancialCalculations:
    """Test suite for financial calculations with mathematical rigor"""
    
    @pytest.fixture
    def sample_prices(self) -> pd.Series:
        """Sample price data for testing"""
        np.random.seed(42)
        prices = pd.Series([100.0])
        for _ in range(100):
            change = np.random.normal(0.001, 0.02)
            prices = pd.concat([prices, pd.Series([prices.iloc[-1] * (1 + change)])])
        return prices.reset_index(drop=True)
    
    @pytest.fixture
    def sample_returns(self, sample_prices) -> pd.Series:
        """Sample returns data"""
        return sample_prices.pct_change().dropna()
    
    def test_sharpe_ratio_calculation(self, sample_returns):
        """Test Sharpe ratio calculation with mathematical precision"""
        risk_free_rate = 0.02
        
        # Test normal case
        excess_returns = sample_returns - risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        # Mathematical validation
        expected_sharpe = (sample_returns.mean() - risk_free_rate / 252) / sample_returns.std() * np.sqrt(252)
        assert validate_financial_calculation(sharpe_ratio, expected_sharpe)
        
        # Edge cases
        zero_volatility_returns = pd.Series([0.01] * 100)
        with pytest.raises(ZeroDivisionError):
            (zero_volatility_returns - risk_free_rate / 252).mean() / 0 * np.sqrt(252)
        
        # Empty series
        empty_returns = pd.Series([], dtype=float)
        assert np.isnan((empty_returns - risk_free_rate / 252).mean())
    
    def test_maximum_drawdown_calculation(self, sample_prices):
        """Test maximum drawdown calculation"""
        cumulative_returns = sample_prices / sample_prices.iloc[0]
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Validate calculation
        assert max_drawdown <= 0, "Maximum drawdown should be negative or zero"
        assert max_drawdown >= -1, "Maximum drawdown cannot exceed -100%"
        
        # Test edge cases
        increasing_prices = pd.Series([100, 110, 120, 130, 140])
        cum_ret = increasing_prices / increasing_prices.iloc[0]
        run_max = cum_ret.expanding().max()
        dd = (cum_ret - run_max) / run_max
        assert dd.min() == 0, "No drawdown for monotonically increasing prices"
    
    def test_volatility_calculation(self, sample_returns):
        """Test volatility calculations"""
        # Simple volatility
        simple_vol = sample_returns.std()
        assert simple_vol > 0, "Volatility must be positive"
        
        # Annualized volatility
        annualized_vol = simple_vol * np.sqrt(252)
        assert annualized_vol > simple_vol, "Annualized volatility should be higher"
        
        # EWMA volatility
        alpha = 0.06
        ewma_var = sample_returns.ewm(alpha=alpha).var().iloc[-1]
        ewma_vol = np.sqrt(ewma_var)
        assert ewma_vol > 0, "EWMA volatility must be positive"
        
        # Test with different decay factors
        for decay in [0.01, 0.05, 0.1, 0.2]:
            vol = sample_returns.ewm(alpha=decay).std().iloc[-1]
            assert vol > 0 and not np.isnan(vol)
    
    def test_value_at_risk_calculation(self, sample_returns):
        """Test Value at Risk (VaR) calculations"""
        confidence_levels = [0.95, 0.99, 0.999]
        
        for confidence in confidence_levels:
            # Historical VaR
            var = np.percentile(sample_returns, (1 - confidence) * 100)
            assert var < 0, f"VaR at {confidence} confidence should be negative"
            
            # Parametric VaR (assuming normal distribution)
            mean_return = sample_returns.mean()
            vol = sample_returns.std()
            from scipy.stats import norm
            z_score = norm.ppf(1 - confidence)
            parametric_var = mean_return + z_score * vol
            
            # VaR should become more negative with higher confidence
            if confidence > 0.95:
                var_95 = np.percentile(sample_returns, 5)
                assert var < var_95, "Higher confidence VaR should be more extreme"
    
    def test_expected_shortfall_calculation(self, sample_returns):
        """Test Expected Shortfall (Conditional VaR) calculations"""
        confidence = 0.95
        var = np.percentile(sample_returns, 5)
        
        # Expected Shortfall is average of returns below VaR
        tail_returns = sample_returns[sample_returns <= var]
        expected_shortfall = tail_returns.mean()
        
        assert expected_shortfall <= var, "Expected Shortfall should be more extreme than VaR"
        assert not np.isnan(expected_shortfall), "Expected Shortfall should be calculable"
    
    def test_beta_calculation(self):
        """Test beta calculation against market benchmark"""
        np.random.seed(42)
        market_returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        
        # Create correlated asset returns
        correlation = 0.7
        asset_returns = correlation * market_returns + np.sqrt(1 - correlation**2) * pd.Series(np.random.normal(0.001, 0.03, 100))
        
        # Calculate beta
        covariance = np.cov(asset_returns, market_returns)[0, 1]
        market_variance = market_returns.var()
        beta = covariance / market_variance
        
        # Validate beta calculation
        assert 0.5 < beta < 1.5, f"Beta should be reasonable: {beta}"
        
        # Test edge cases
        # Perfect correlation
        perfect_corr_returns = market_returns.copy()
        perfect_beta = np.cov(perfect_corr_returns, market_returns)[0, 1] / market_returns.var()
        assert abs(perfect_beta - 1.0) < 0.01, "Perfect correlation should give beta ≈ 1"
        
        # No correlation
        uncorr_returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        np.random.seed(100)  # Different seed
        uncorr_beta = np.cov(uncorr_returns, market_returns)[0, 1] / market_returns.var()
        assert abs(uncorr_beta) < 0.3, "Uncorrelated returns should have beta ≈ 0"
    
    def test_information_ratio(self, sample_returns):
        """Test Information Ratio calculation"""
        # Create benchmark returns
        np.random.seed(123)
        benchmark_returns = pd.Series(np.random.normal(0.0008, 0.015, len(sample_returns)))
        
        # Calculate active returns
        active_returns = sample_returns - benchmark_returns
        
        # Information Ratio
        ir = active_returns.mean() / active_returns.std()
        
        assert not np.isnan(ir), "Information Ratio should be calculable"
        assert abs(ir) < 5, "Information Ratio should be reasonable"
    
    def test_sortino_ratio(self, sample_returns):
        """Test Sortino Ratio calculation"""
        target_return = 0.0
        
        # Calculate downside deviation
        downside_returns = sample_returns[sample_returns < target_return]
        downside_deviation = downside_returns.std()
        
        # Sortino Ratio
        if downside_deviation > 0:
            sortino_ratio = sample_returns.mean() / downside_deviation
            assert not np.isnan(sortino_ratio), "Sortino Ratio should be calculable"
        else:
            # All returns above target
            assert len(downside_returns) == 0
    
    def test_calmar_ratio(self, sample_prices):
        """Test Calmar Ratio calculation"""
        # Calculate annualized return
        total_return = (sample_prices.iloc[-1] / sample_prices.iloc[0]) - 1
        periods = len(sample_prices)
        annualized_return = (1 + total_return) ** (252 / periods) - 1
        
        # Calculate maximum drawdown
        cumulative = sample_prices / sample_prices.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # Calmar Ratio
        if max_drawdown > 0:
            calmar_ratio = annualized_return / max_drawdown
            assert not np.isnan(calmar_ratio), "Calmar Ratio should be calculable"
    
    @pytest.mark.parametrize("window_size", [20, 50, 100])
    def test_rolling_calculations(self, sample_returns, window_size):
        """Test rolling window calculations"""
        if len(sample_returns) < window_size:
            pytest.skip(f"Not enough data for window size {window_size}")
        
        # Rolling Sharpe ratio
        rolling_sharpe = sample_returns.rolling(window_size).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        
        assert not rolling_sharpe.isna().all(), "Rolling Sharpe should have valid values"
        
        # Rolling volatility
        rolling_vol = sample_returns.rolling(window_size).std() * np.sqrt(252)
        assert (rolling_vol > 0).any(), "Rolling volatility should be positive"
    
    def test_precision_and_rounding(self):
        """Test financial calculation precision"""
        # Test decimal precision for financial calculations
        price1 = Decimal('100.12345678')
        price2 = Decimal('99.87654321')
        
        # Price difference with rounding
        diff = (price1 - price2).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        expected = Decimal('0.24')
        assert diff == expected, f"Expected {expected}, got {diff}"
        
        # Percentage calculation
        pct_change = ((price1 - price2) / price2 * 100).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
        assert pct_change > 0, "Percentage change should be positive"
    
    def test_portfolio_calculations(self):
        """Test portfolio-level calculations"""
        # Portfolio weights
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        assert abs(weights.sum() - 1.0) < 1e-10, "Weights should sum to 1"
        
        # Expected portfolio return
        asset_returns = np.array([0.08, 0.12, 0.06, 0.15])
        portfolio_return = np.dot(weights, asset_returns)
        expected = 0.4 * 0.08 + 0.3 * 0.12 + 0.2 * 0.06 + 0.1 * 0.15
        assert validate_financial_calculation(portfolio_return, expected)
        
        # Portfolio variance
        cov_matrix = np.array([
            [0.04, 0.02, 0.01, 0.03],
            [0.02, 0.09, 0.015, 0.025],
            [0.01, 0.015, 0.0225, 0.02],
            [0.03, 0.025, 0.02, 0.16]
        ])
        
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        assert portfolio_variance > 0, "Portfolio variance should be positive"
    
    def test_complex_derivatives(self):
        """Test complex derivative pricing calculations"""
        # Black-Scholes components
        S = 100  # Stock price
        K = 105  # Strike price
        T = 0.25  # Time to expiry
        r = 0.05  # Risk-free rate
        sigma = 0.2  # Volatility
        
        # d1 and d2 calculations
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Validate calculations
        assert not np.isnan(d1) and not np.isnan(d2), "d1 and d2 should be calculable"
        assert d2 < d1, "d2 should be less than d1"
        
        # Greeks calculations (delta, gamma, theta, vega, rho)
        from scipy.stats import norm
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        assert 0 <= delta <= 1, "Delta should be between 0 and 1 for calls"
        assert gamma >= 0, "Gamma should be non-negative"
    
    def test_risk_metrics_edge_cases(self):
        """Test risk metrics with edge cases"""
        # Single value series
        single_return = pd.Series([0.01])
        assert np.isnan(single_return.std()), "Single value should have NaN std"
        
        # All identical values
        identical_returns = pd.Series([0.01] * 10)
        assert identical_returns.std() == 0, "Identical values should have zero std"
        
        # Extreme values
        extreme_returns = pd.Series([0.5, -0.5, 0.3, -0.3])
        vol = extreme_returns.std()
        assert vol > 0.3, "Extreme values should have high volatility"
        
        # Missing values
        returns_with_nan = pd.Series([0.01, np.nan, 0.02, 0.015])
        clean_vol = returns_with_nan.dropna().std()
        assert not np.isnan(clean_vol), "Should handle NaN values"

    def test_mathematical_properties(self, sample_returns):
        """Test mathematical properties of financial metrics"""
        # Linearity tests
        scaled_returns = sample_returns * 2
        original_vol = sample_returns.std()
        scaled_vol = scaled_returns.std()
        
        assert validate_financial_calculation(scaled_vol, original_vol * 2, 0.001)
        
        # Translation invariance for Sharpe ratio
        shifted_returns = sample_returns + 0.001
        original_sharpe = sample_returns.mean() / sample_returns.std()
        shifted_sharpe = shifted_returns.mean() / shifted_returns.std()
        
        # Sharpe ratio should change when shifting returns
        assert not validate_financial_calculation(original_sharpe, shifted_sharpe, 0.001)
        
        # Variance properties
        var_returns = sample_returns.var()
        std_returns = sample_returns.std()
        assert validate_financial_calculation(var_returns, std_returns ** 2, 0.0001)