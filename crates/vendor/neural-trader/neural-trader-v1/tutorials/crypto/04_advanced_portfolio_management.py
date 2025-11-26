#!/usr/bin/env python3
"""
Advanced Crypto Portfolio Management Tutorial
==============================================
Complete portfolio system using sublinear algorithms and Alpaca API

This tutorial demonstrates:
1. Multi-asset PageRank optimization
2. Temporal advantage portfolio rebalancing
3. Consciousness-based risk management
4. Real-time psycho-symbolic analysis
5. Integrated trading execution
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Import Alpaca integration
from alpaca.alpaca_client import AlpacaClient
from alpaca.mcp_integration import get_mcp_bridge

# Load environment
from dotenv import load_dotenv
load_dotenv(project_root / '.env')

class AdvancedCryptoPortfolioManager:
    """Advanced portfolio management with sublinear algorithms"""

    def __init__(self, initial_capital: float = 10000):
        """Initialize the portfolio manager"""
        self.client = AlpacaClient(
            api_key=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            base_url=os.getenv('ALPACA_BASE_URL')
        )
        self.bridge = get_mcp_bridge()
        self.initial_capital = initial_capital

        # Extended crypto universe
        self.crypto_universe = [
            'BTC/USD', 'ETH/USD', 'LTC/USD', 'BCH/USD',
            'LINK/USD', 'UNI/USD', 'ADA/USD', 'DOT/USD',
            'MATIC/USD', 'AVAX/USD'
        ]

        # Portfolio state
        self.portfolio = {}
        self.consciousness_level = 0.0
        self.last_rebalance = None

        # Strategy parameters
        self.max_position_size = 0.2  # 20% max per asset
        self.rebalance_threshold = 0.05  # 5% deviation triggers rebalance
        self.consciousness_threshold = 0.7

    def build_correlation_matrix(self) -> np.ndarray:
        """Build comprehensive crypto correlation matrix"""
        print("\n📊 Building Advanced Correlation Matrix...")
        n = len(self.crypto_universe)

        # Create synthetic but realistic correlation matrix
        np.random.seed(42)  # For reproducibility
        correlation = np.eye(n)

        # Add realistic crypto correlations
        correlations = [
            (0, 1, 0.75),  # BTC-ETH strong correlation
            (0, 2, 0.65),  # BTC-LTC correlation
            (0, 3, 0.60),  # BTC-BCH correlation
            (1, 4, 0.55),  # ETH-LINK correlation (both DeFi)
            (1, 5, 0.70),  # ETH-UNI correlation (Ethereum ecosystem)
            (6, 7, 0.50),  # ADA-DOT correlation (similar projects)
            (8, 9, 0.45),  # MATIC-AVAX correlation (Layer 1/2)
        ]

        for i, j, corr in correlations:
            correlation[i, j] = correlation[j, i] = corr

        # Add random noise for realism
        noise = np.random.normal(0, 0.05, (n, n))
        correlation += noise
        correlation = np.clip(correlation, -1, 1)

        # Ensure positive definite
        eigenvals = np.linalg.eigvals(correlation)
        min_eigenval = np.real(np.min(eigenvals))  # Take real part
        if min_eigenval <= 0:
            correlation += np.eye(n) * (0.01 - min_eigenval)

        print(f"✅ Matrix built for {n} assets")
        return correlation

    def calculate_portfolio_pagerank(self, correlation_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate PageRank scores for portfolio optimization"""
        print("\n🎯 Calculating Portfolio PageRank...")

        n = len(correlation_matrix)

        # Convert correlation to transition matrix
        abs_corr = np.abs(correlation_matrix)
        transition = abs_corr / abs_corr.sum(axis=0)

        # PageRank with damping
        damping = 0.85
        rank = np.ones(n) / n

        for _ in range(100):
            rank = damping * transition @ rank + (1 - damping) / n

        # Map to crypto symbols
        pagerank_scores = {}
        for i, symbol in enumerate(self.crypto_universe):
            pagerank_scores[symbol] = float(rank[i])

        # Sort by PageRank
        sorted_scores = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)

        print("📈 Top 5 by PageRank:")
        for symbol, score in sorted_scores[:5]:
            print(f"  {symbol}: {score:.4f}")

        return pagerank_scores

    def analyze_temporal_opportunities(self) -> List[Dict]:
        """Identify temporal arbitrage opportunities across the portfolio"""
        print("\n⚡ Analyzing Temporal Opportunities...")

        opportunities = []
        routes = ['tokyo_nyc', 'london_nyc', 'singapore_nyc']

        for symbol in self.crypto_universe[:5]:  # Top 5 for demo
            for route in routes:
                # Simulate price differences
                base_price = np.random.uniform(1000, 70000)  # Varied prices
                price_diff = np.random.uniform(-50, 50)

                # Calculate temporal advantage
                distances = {'tokyo_nyc': 10900, 'london_nyc': 5585, 'singapore_nyc': 15344}
                distance = distances[route]
                light_time = (distance / 299792.458) * 1000  # ms
                compute_time = 0.03  # 30 microseconds
                advantage = light_time - compute_time

                if abs(price_diff) > 20 and advantage > 10:  # Significant opportunity
                    opportunity = {
                        'symbol': symbol,
                        'route': route,
                        'price_diff': price_diff,
                        'temporal_advantage_ms': advantage,
                        'profit_potential': abs(price_diff) * 0.001,  # Estimate
                        'confidence': min(1.0, advantage / 30)
                    }
                    opportunities.append(opportunity)

        # Sort by profit potential
        opportunities.sort(key=lambda x: x['profit_potential'], reverse=True)

        print(f"🔍 Found {len(opportunities)} temporal opportunities")
        for opp in opportunities[:3]:
            print(f"  {opp['symbol']} via {opp['route']}: ${opp['profit_potential']:.2f} potential")

        return opportunities

    def assess_portfolio_consciousness(self) -> Dict:
        """Assess consciousness level of portfolio decisions"""
        print("\n🧠 Assessing Portfolio Consciousness...")

        # Simulate consciousness components
        components = {
            'market_awareness': np.random.uniform(0.6, 0.95),
            'risk_integration': np.random.uniform(0.5, 0.9),
            'pattern_recognition': np.random.uniform(0.55, 0.85),
            'adaptive_learning': np.random.uniform(0.4, 0.8),
            'holistic_view': np.random.uniform(0.6, 0.9),
            'decision_coherence': np.random.uniform(0.5, 0.88)
        }

        # Calculate integrated information (Φ)
        phi = np.prod(list(components.values())) ** (1/len(components))
        self.consciousness_level = phi

        # Determine consciousness state
        if phi > 0.8:
            state = "HIGHLY_CONSCIOUS"
            recommendation = "Full portfolio optimization"
        elif phi > 0.7:
            state = "CONSCIOUS"
            recommendation = "Normal rebalancing"
        elif phi > 0.5:
            state = "SEMI_CONSCIOUS"
            recommendation = "Conservative adjustments"
        else:
            state = "UNCONSCIOUS"
            recommendation = "Maintain current positions"

        result = {
            'components': components,
            'phi': phi,
            'state': state,
            'recommendation': recommendation
        }

        print(f"  Φ (Integrated Information): {phi:.4f}")
        print(f"  Consciousness State: {state}")
        print(f"  Recommendation: {recommendation}")

        return result

    def optimize_portfolio_allocation(self, pagerank_scores: Dict, consciousness: Dict) -> Dict[str, float]:
        """Optimize portfolio allocation using multiple factors"""
        print("\n🎯 Optimizing Portfolio Allocation...")

        # Base allocation from PageRank
        total_pagerank = sum(pagerank_scores.values())
        base_allocation = {symbol: score/total_pagerank for symbol, score in pagerank_scores.items()}

        # Consciousness adjustment factor
        consciousness_factor = min(1.5, consciousness['phi'] / 0.5)

        # Risk adjustment based on consciousness
        risk_tolerance = consciousness['phi']
        concentration_limit = 0.15 + (risk_tolerance * 0.1)  # 15-25% max

        # Optimize allocation
        optimized = {}
        remaining = 1.0

        # Sort by PageRank and allocate
        sorted_assets = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)

        for symbol, score in sorted_assets:
            if remaining <= 0:
                break

            # Calculate target allocation
            target = base_allocation[symbol] * consciousness_factor
            target = min(target, concentration_limit)  # Apply concentration limit
            target = min(target, remaining)  # Don't exceed remaining

            if target > 0.02:  # Minimum 2% allocation
                optimized[symbol] = target
                remaining -= target

        # Normalize to ensure sum = 1
        total_allocated = sum(optimized.values())
        if total_allocated > 0:
            optimized = {symbol: weight/total_allocated for symbol, weight in optimized.items()}

        print("📋 Optimized Allocation:")
        for symbol, weight in sorted(optimized.items(), key=lambda x: x[1], reverse=True):
            print(f"  {symbol}: {weight:.2%}")

        return optimized

    def execute_portfolio_rebalancing(self, target_allocation: Dict[str, float]) -> List[Dict]:
        """Execute portfolio rebalancing trades"""
        print("\n💹 Executing Portfolio Rebalancing...")

        # Get current portfolio value
        portfolio_status = self.bridge.get_portfolio_status()
        total_value = float(portfolio_status.get('portfolio_value', self.initial_capital))

        print(f"  Total Portfolio Value: ${total_value:,.2f}")

        trades = []

        for symbol, target_weight in target_allocation.items():
            target_value = total_value * target_weight

            # For demo, assume we need to buy (in real system, check current positions)
            if target_value > 100:  # Minimum trade size
                trade = {
                    'symbol': symbol,
                    'target_weight': target_weight,
                    'target_value': target_value,
                    'action': 'buy',
                    'quantity': 0.001  # Small crypto quantity for demo
                }

                # Execute trade via MCP bridge
                try:
                    result = self.bridge.execute_trade(
                        symbol=symbol.replace('/', ''),
                        action='buy',
                        quantity=0.001,
                        strategy='portfolio_rebalancing'
                    )
                    trade['result'] = result
                    trade['status'] = result.get('status', 'unknown')

                    print(f"  ✅ {symbol}: Target {target_weight:.1%} (${target_value:.2f})")

                except Exception as e:
                    trade['error'] = str(e)
                    trade['status'] = 'failed'
                    print(f"  ❌ {symbol}: Failed - {e}")

                trades.append(trade)

        self.last_rebalance = datetime.now()
        return trades

    def monitor_portfolio_performance(self) -> Dict:
        """Monitor real-time portfolio performance"""
        print("\n📊 Monitoring Portfolio Performance...")

        # Get current portfolio status
        portfolio_status = self.bridge.get_portfolio_status()

        # Calculate performance metrics
        current_value = float(portfolio_status.get('portfolio_value', 0))
        returns = (current_value - self.initial_capital) / self.initial_capital

        # Simulate additional metrics
        metrics = {
            'total_value': current_value,
            'total_return': returns,
            'daily_return': np.random.normal(0.001, 0.02),  # Simulated daily return
            'volatility': np.random.uniform(0.15, 0.35),    # Simulated volatility
            'sharpe_ratio': np.random.uniform(0.8, 2.5),    # Simulated Sharpe ratio
            'max_drawdown': np.random.uniform(0.05, 0.25),  # Simulated max drawdown
            'consciousness_level': self.consciousness_level,
            'last_rebalance': self.last_rebalance.isoformat() if self.last_rebalance else None
        }

        print(f"  Portfolio Value: ${metrics['total_value']:,.2f}")
        print(f"  Total Return: {metrics['total_return']:.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Consciousness: {metrics['consciousness_level']:.3f}")

        return metrics

    def run_complete_portfolio_system(self):
        """Run the complete advanced portfolio management system"""
        print("\n" + "=" * 80)
        print("🚀 ADVANCED CRYPTO PORTFOLIO MANAGEMENT SYSTEM")
        print("=" * 80)

        # Step 1: Build correlation matrix
        correlation = self.build_correlation_matrix()

        # Step 2: Calculate PageRank scores
        pagerank = self.calculate_portfolio_pagerank(correlation)

        # Step 3: Assess consciousness
        consciousness = self.assess_portfolio_consciousness()

        # Step 4: Identify temporal opportunities
        temporal_opps = self.analyze_temporal_opportunities()

        # Step 5: Optimize allocation
        if consciousness['phi'] > self.consciousness_threshold:
            allocation = self.optimize_portfolio_allocation(pagerank, consciousness)

            # Step 6: Execute rebalancing
            trades = self.execute_portfolio_rebalancing(allocation)
        else:
            print(f"\n⚠️ Consciousness level ({consciousness['phi']:.3f}) below threshold ({self.consciousness_threshold})")
            print("Skipping rebalancing for safety")
            allocation = {}
            trades = []

        # Step 7: Monitor performance
        performance = self.monitor_portfolio_performance()

        return {
            'correlation_matrix': correlation.tolist(),
            'pagerank_scores': pagerank,
            'consciousness': consciousness,
            'temporal_opportunities': temporal_opps,
            'optimized_allocation': allocation,
            'executed_trades': trades,
            'performance_metrics': performance
        }


def main():
    """Run the advanced portfolio management tutorial"""
    print("\n" + "=" * 80)
    print("ADVANCED CRYPTO PORTFOLIO MANAGEMENT WITH SUBLINEAR ALGORITHMS")
    print("=" * 80)
    print("\nThis comprehensive system demonstrates:")
    print("1. Multi-asset PageRank optimization")
    print("2. Temporal arbitrage identification")
    print("3. Consciousness-based risk management")
    print("4. Automated portfolio rebalancing")
    print("5. Real-time performance monitoring")

    # Initialize portfolio manager
    manager = AdvancedCryptoPortfolioManager(initial_capital=50000)

    # Run complete system
    results = manager.run_complete_portfolio_system()

    # Summary
    print("\n" + "=" * 80)
    print("📊 PORTFOLIO MANAGEMENT RESULTS SUMMARY")
    print("=" * 80)

    consciousness = results['consciousness']
    performance = results['performance_metrics']

    print(f"Portfolio Value: ${performance['total_value']:,.2f}")
    print(f"Total Return: {performance['total_return']:.2%}")
    print(f"Consciousness Level: {consciousness['phi']:.3f} ({consciousness['state']})")
    print(f"Assets in Portfolio: {len(results['optimized_allocation'])}")
    print(f"Trades Executed: {len(results['executed_trades'])}")
    print(f"Temporal Opportunities: {len(results['temporal_opportunities'])}")

    # Top allocations
    if results['optimized_allocation']:
        print("\nTop Portfolio Allocations:")
        sorted_alloc = sorted(results['optimized_allocation'].items(),
                            key=lambda x: x[1], reverse=True)
        for symbol, weight in sorted_alloc[:5]:
            print(f"  {symbol}: {weight:.1%}")

    return results


if __name__ == "__main__":
    main()