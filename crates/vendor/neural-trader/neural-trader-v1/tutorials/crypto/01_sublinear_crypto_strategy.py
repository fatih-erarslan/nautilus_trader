#!/usr/bin/env python3
"""
Sublinear Crypto Trading Strategy Tutorial
===========================================
Advanced crypto trading using sublinear algorithms and Alpaca API

This tutorial demonstrates:
1. PageRank-based crypto asset ranking
2. Temporal advantage for predictive trading
3. Psycho-symbolic market sentiment analysis
4. Consciousness-based adaptive strategies
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Import Alpaca integration
from alpaca.alpaca_client import AlpacaClient
from alpaca.mcp_integration import get_mcp_bridge

# Load environment
from dotenv import load_dotenv
load_dotenv(project_root / '.env')

class SublinearCryptoStrategy:
    """Advanced crypto trading using sublinear algorithms"""

    def __init__(self):
        """Initialize the strategy with Alpaca and sublinear tools"""
        self.client = AlpacaClient(
            api_key=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            base_url=os.getenv('ALPACA_BASE_URL')
        )
        self.bridge = get_mcp_bridge()

        # Crypto pairs to track
        self.crypto_pairs = [
            'BTC/USD', 'ETH/USD', 'LTC/USD',
            'BCH/USD', 'LINK/USD', 'UNI/USD'
        ]

        # Initialize strategy parameters
        self.pagerank_alpha = 0.85
        self.temporal_distance_km = 10900  # Tokyo to NYC
        self.consciousness_threshold = 0.7

    def build_crypto_correlation_matrix(self) -> np.ndarray:
        """Build correlation matrix for crypto assets"""
        print("\n📊 Building Crypto Correlation Matrix...")

        # For demo, create a synthetic correlation matrix
        n = len(self.crypto_pairs)
        correlation = np.eye(n)

        # Add some correlations
        correlation[0, 1] = correlation[1, 0] = 0.8  # BTC-ETH correlation
        correlation[0, 2] = correlation[2, 0] = 0.6  # BTC-LTC correlation
        correlation[1, 3] = correlation[3, 1] = 0.5  # ETH-BCH correlation

        print(f"✅ Correlation matrix built for {n} crypto assets")
        return correlation

    def calculate_crypto_pagerank(self, correlation_matrix: np.ndarray) -> Dict[str, float]:
        """
        Use PageRank algorithm to identify influential crypto assets

        The idea: Cryptos that are highly correlated with many others
        are more "influential" in the market
        """
        print("\n🎯 Calculating Crypto PageRank...")

        # Convert correlation to transition matrix
        n = len(correlation_matrix)
        transition = np.abs(correlation_matrix)
        transition = transition / transition.sum(axis=0)

        # PageRank iteration
        rank = np.ones(n) / n
        for _ in range(100):
            rank = self.pagerank_alpha * transition @ rank + (1 - self.pagerank_alpha) / n

        # Map to crypto pairs
        pagerank_scores = {}
        for i, pair in enumerate(self.crypto_pairs):
            pagerank_scores[pair] = float(rank[i])
            print(f"  {pair}: PageRank = {rank[i]:.4f}")

        return pagerank_scores

    def calculate_temporal_advantage(self, symbol: str) -> Dict[str, Any]:
        """
        Calculate temporal advantage for ultra-fast trading

        Concept: Solve the trading decision before market data arrives
        using sublinear prediction
        """
        print(f"\n⚡ Calculating Temporal Advantage for {symbol}...")

        # Speed of light calculation
        speed_of_light_ms = 299792.458  # km/s
        light_travel_time = self.temporal_distance_km / speed_of_light_ms

        # Sublinear computation time (microseconds)
        computation_time = 0.000050  # 50 microseconds

        # Temporal advantage
        advantage = light_travel_time - computation_time

        result = {
            "symbol": symbol,
            "distance_km": self.temporal_distance_km,
            "light_travel_ms": light_travel_time * 1000,
            "computation_ms": computation_time * 1000,
            "advantage_ms": advantage * 1000,
            "can_predict": advantage > 0
        }

        print(f"  Light travel: {result['light_travel_ms']:.3f}ms")
        print(f"  Computation: {result['computation_ms']:.3f}ms")
        print(f"  Advantage: {result['advantage_ms']:.3f}ms")
        print(f"  Can predict: {result['can_predict']}")

        return result

    def analyze_market_psychology(self, symbol: str) -> Dict[str, Any]:
        """
        Use psycho-symbolic reasoning for market sentiment
        """
        print(f"\n🧠 Analyzing Market Psychology for {symbol}...")

        # Simulated psychological factors
        psychological_factors = {
            "fear_greed_index": np.random.uniform(20, 80),
            "social_sentiment": np.random.uniform(-1, 1),
            "institutional_confidence": np.random.uniform(0.3, 0.9),
            "retail_fomo": np.random.uniform(0, 1),
            "whale_accumulation": np.random.uniform(-0.5, 0.5)
        }

        # Calculate composite psychological score
        weights = [0.3, 0.2, 0.25, 0.15, 0.1]
        normalized_factors = [
            psychological_factors["fear_greed_index"] / 100,
            (psychological_factors["social_sentiment"] + 1) / 2,
            psychological_factors["institutional_confidence"],
            psychological_factors["retail_fomo"],
            (psychological_factors["whale_accumulation"] + 0.5)
        ]

        composite_score = sum(w * f for w, f in zip(weights, normalized_factors))

        # Determine action based on psychology
        if composite_score > 0.7:
            action = "STRONG_BUY"
        elif composite_score > 0.55:
            action = "BUY"
        elif composite_score > 0.45:
            action = "HOLD"
        elif composite_score > 0.3:
            action = "SELL"
        else:
            action = "STRONG_SELL"

        result = {
            **psychological_factors,
            "composite_score": composite_score,
            "recommended_action": action
        }

        for key, value in psychological_factors.items():
            print(f"  {key}: {value:.3f}")
        print(f"  Composite Score: {composite_score:.3f}")
        print(f"  Recommendation: {action}")

        return result

    def calculate_consciousness_score(self) -> float:
        """
        Calculate consciousness/awareness score for adaptive strategy
        Uses integrated information theory concepts
        """
        print("\n🌟 Calculating Strategy Consciousness Level...")

        # Simulate integrated information (Φ)
        phi_components = {
            "market_awareness": np.random.uniform(0.5, 1.0),
            "pattern_recognition": np.random.uniform(0.4, 0.9),
            "adaptive_capacity": np.random.uniform(0.3, 0.8),
            "prediction_accuracy": np.random.uniform(0.4, 0.85),
            "self_correction": np.random.uniform(0.2, 0.7)
        }

        # Calculate Φ (phi) - integrated information
        phi = np.prod(list(phi_components.values())) ** (1/len(phi_components))

        print(f"  Integrated Information (Φ): {phi:.4f}")
        for component, value in phi_components.items():
            print(f"    {component}: {value:.3f}")

        return phi

    def execute_sublinear_trade(self, symbol: str, action: str, quantity: int) -> Dict:
        """Execute trade using sublinear decision making"""
        print(f"\n💹 Executing Sublinear Trade: {action} {quantity} {symbol}")

        # Calculate all sublinear metrics
        temporal = self.calculate_temporal_advantage(symbol)
        psychology = self.analyze_market_psychology(symbol)
        consciousness = self.calculate_consciousness_score()

        # Decision matrix
        should_trade = (
            temporal['can_predict'] and
            psychology['composite_score'] > 0.5 and
            consciousness > self.consciousness_threshold
        )

        if should_trade:
            print(f"✅ All conditions met - Executing trade")

            # Use MCP bridge for real trading
            result = self.bridge.execute_trade(
                symbol=symbol.replace('/', ''),  # Remove slash for Alpaca
                action=action.lower(),
                quantity=quantity,
                strategy='sublinear_crypto'
            )

            print(f"  Order ID: {result.get('order_id', 'DEMO')}")
            print(f"  Status: {result.get('status')}")
            print(f"  Demo Mode: {result.get('demo_mode')}")

            return result
        else:
            print("❌ Conditions not met - Trade skipped")
            print(f"  Temporal OK: {temporal['can_predict']}")
            print(f"  Psychology OK: {psychology['composite_score'] > 0.5}")
            print(f"  Consciousness OK: {consciousness > self.consciousness_threshold}")

            return {"status": "skipped", "reason": "Conditions not met"}

    def run_complete_strategy(self):
        """Run the complete sublinear crypto trading strategy"""
        print("\n" + "="*60)
        print("🚀 SUBLINEAR CRYPTO TRADING STRATEGY")
        print("="*60)

        # Step 1: Build correlation matrix
        correlation = self.build_crypto_correlation_matrix()

        # Step 2: Calculate PageRank
        pagerank = self.calculate_crypto_pagerank(correlation)

        # Step 3: Select top crypto by PageRank
        top_crypto = max(pagerank, key=pagerank.get)
        print(f"\n🏆 Top Crypto by PageRank: {top_crypto}")

        # Step 4: Analyze and potentially trade
        analysis = self.execute_sublinear_trade(
            symbol=top_crypto,
            action="buy",
            quantity=1
        )

        # Step 5: Portfolio check
        print("\n📈 Checking Portfolio Status...")
        portfolio = self.bridge.get_portfolio_status()
        print(f"  Portfolio Value: ${portfolio.get('portfolio_value', 0):,.2f}")
        print(f"  Cash: ${portfolio.get('cash', 0):,.2f}")
        print(f"  Demo Mode: {portfolio.get('demo_mode')}")

        return {
            "pagerank": pagerank,
            "top_crypto": top_crypto,
            "trade_result": analysis,
            "portfolio": portfolio
        }


def main():
    """Run the tutorial"""
    print("\n" + "="*60)
    print("ALPACA CRYPTO TRADING WITH SUBLINEAR ALGORITHMS")
    print("="*60)
    print("\nThis tutorial demonstrates advanced crypto trading using:")
    print("1. PageRank for asset influence ranking")
    print("2. Temporal advantage for predictive trading")
    print("3. Psycho-symbolic market analysis")
    print("4. Consciousness-based adaptive strategies")

    # Initialize strategy
    strategy = SublinearCryptoStrategy()

    # Run complete strategy
    results = strategy.run_complete_strategy()

    print("\n" + "="*60)
    print("📊 STRATEGY RESULTS SUMMARY")
    print("="*60)
    print(f"Top Crypto: {results['top_crypto']}")
    print(f"Trade Status: {results['trade_result'].get('status')}")
    print(f"Portfolio Value: ${results['portfolio'].get('portfolio_value', 0):,.2f}")

    return results


if __name__ == "__main__":
    main()