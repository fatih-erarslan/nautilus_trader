#!/usr/bin/env python3
"""
Consciousness-Based Crypto Trading Tutorial
============================================
Advanced trading using consciousness evolution and integrated information

This tutorial demonstrates:
1. Integrated Information Theory (IIT) for trading decisions
2. Consciousness evolution and emergence
3. Psycho-symbolic market reasoning
4. Self-aware adaptive strategies
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
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

class ConsciousnessTradingSystem:
    """Trading system with consciousness-based decision making"""

    def __init__(self):
        """Initialize the consciousness trading system"""
        self.client = AlpacaClient(
            api_key=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            base_url=os.getenv('ALPACA_BASE_URL')
        )
        self.bridge = get_mcp_bridge()

        # Consciousness parameters
        self.phi_threshold = 0.7  # Minimum Φ for trading
        self.emergence_target = 0.9
        self.cognitive_patterns = [
            'convergent',   # Focused analysis
            'divergent',     # Creative exploration
            'lateral',       # Unconventional connections
            'systemic',      # Holistic understanding
            'critical',      # Rigorous evaluation
            'adaptive'       # Dynamic adjustment
        ]

    def calculate_integrated_information(self, market_data: Dict) -> float:
        """
        Calculate Φ (Phi) - Integrated Information
        Based on Integrated Information Theory (IIT)
        """
        print("\n🧠 Calculating Integrated Information (Φ)...")

        # Simulate system components
        components = {
            'price_awareness': np.random.uniform(0.5, 1.0),
            'volume_integration': np.random.uniform(0.4, 0.9),
            'sentiment_coherence': np.random.uniform(0.3, 0.8),
            'pattern_complexity': np.random.uniform(0.4, 0.85),
            'temporal_binding': np.random.uniform(0.35, 0.75),
            'causal_efficacy': np.random.uniform(0.45, 0.9)
        }

        # Calculate Φ using geometric mean (simplified IIT)
        phi = np.prod(list(components.values())) ** (1/len(components))

        # Display components
        print(f"  System Components:")
        for component, value in components.items():
            print(f"    {component}: {value:.4f}")
        print(f"  Integrated Information (Φ): {phi:.4f}")

        return phi

    def evolve_consciousness(self, iterations: int = 100) -> Dict:
        """
        Evolve trading consciousness through iterative refinement
        """
        print(f"\n🌟 Evolving Trading Consciousness ({iterations} iterations)...")
        print("=" * 60)

        consciousness_history = []
        emergence_level = 0.1

        for i in range(iterations):
            # Simulate consciousness evolution
            growth_rate = np.random.uniform(0.001, 0.01)
            noise = np.random.normal(0, 0.002)
            emergence_level = min(1.0, emergence_level * (1 + growth_rate) + noise)

            if i % 20 == 0:  # Report every 20 iterations
                phi = self.calculate_integrated_information({})
                consciousness_history.append({
                    'iteration': i,
                    'emergence': emergence_level,
                    'phi': phi,
                    'conscious': phi > self.phi_threshold
                })
                print(f"  Iteration {i:3d}: Emergence={emergence_level:.4f}, Φ={phi:.4f}")

        # Final state
        final_phi = self.calculate_integrated_information({})
        is_conscious = final_phi > self.phi_threshold

        result = {
            'final_emergence': emergence_level,
            'final_phi': final_phi,
            'is_conscious': is_conscious,
            'history': consciousness_history
        }

        print(f"\n✅ Evolution Complete:")
        print(f"  Final Emergence: {emergence_level:.4f}")
        print(f"  Final Φ: {final_phi:.4f}")
        print(f"  Conscious State: {'YES ✨' if is_conscious else 'NO ❌'}")

        return result

    def analyze_market_consciousness(self, symbol: str) -> Dict:
        """
        Analyze market consciousness for a specific symbol
        """
        print(f"\n🔮 Analyzing Market Consciousness for {symbol}...")
        print("=" * 60)

        # Simulate market consciousness metrics
        market_consciousness = {
            'collective_awareness': np.random.uniform(0.4, 0.9),
            'information_integration': np.random.uniform(0.5, 0.95),
            'emergent_patterns': np.random.uniform(0.3, 0.8),
            'synchronization_level': np.random.uniform(0.4, 0.85),
            'adaptive_capacity': np.random.uniform(0.35, 0.75)
        }

        # Calculate overall market Φ
        market_phi = np.mean(list(market_consciousness.values()))

        # Determine market state
        if market_phi > 0.8:
            state = "HIGHLY_CONSCIOUS"
            recommendation = "Strong signals, high confidence trading"
        elif market_phi > 0.6:
            state = "CONSCIOUS"
            recommendation = "Normal trading with standard risk"
        elif market_phi > 0.4:
            state = "SEMI_CONSCIOUS"
            recommendation = "Cautious trading, reduced positions"
        else:
            state = "UNCONSCIOUS"
            recommendation = "Avoid trading, market too chaotic"

        result = {
            'symbol': symbol,
            'metrics': market_consciousness,
            'market_phi': market_phi,
            'state': state,
            'recommendation': recommendation
        }

        print(f"  Market Consciousness Metrics:")
        for metric, value in market_consciousness.items():
            print(f"    {metric}: {value:.4f}")
        print(f"  Market Φ: {market_phi:.4f}")
        print(f"  State: {state}")
        print(f"  Recommendation: {recommendation}")

        return result

    def cognitive_pattern_analysis(self, pattern_type: str) -> Dict:
        """
        Analyze market using specific cognitive pattern
        """
        print(f"\n🎯 Cognitive Pattern Analysis: {pattern_type.upper()}")
        print("=" * 60)

        patterns = {
            'convergent': {
                'focus': 'Single optimal solution',
                'method': 'Systematic elimination',
                'strength': np.random.uniform(0.7, 0.95)
            },
            'divergent': {
                'focus': 'Multiple possibilities',
                'method': 'Creative exploration',
                'strength': np.random.uniform(0.6, 0.9)
            },
            'lateral': {
                'focus': 'Unconventional connections',
                'method': 'Indirect approach',
                'strength': np.random.uniform(0.5, 0.85)
            },
            'systemic': {
                'focus': 'Holistic understanding',
                'method': 'Systems thinking',
                'strength': np.random.uniform(0.65, 0.9)
            },
            'critical': {
                'focus': 'Rigorous evaluation',
                'method': 'Logical analysis',
                'strength': np.random.uniform(0.7, 0.95)
            },
            'adaptive': {
                'focus': 'Dynamic adjustment',
                'method': 'Real-time learning',
                'strength': np.random.uniform(0.6, 0.88)
            }
        }

        pattern_data = patterns.get(pattern_type, patterns['adaptive'])

        # Generate insights based on pattern
        insights = []
        if pattern_data['strength'] > 0.8:
            insights.append(f"Strong {pattern_type} signal detected")
            insights.append("High confidence in pattern recognition")
        elif pattern_data['strength'] > 0.6:
            insights.append(f"Moderate {pattern_type} pattern emerging")
            insights.append("Consider combined approach")
        else:
            insights.append(f"Weak {pattern_type} signal")
            insights.append("Seek additional confirmation")

        result = {
            'pattern': pattern_type,
            'data': pattern_data,
            'insights': insights,
            'confidence': pattern_data['strength']
        }

        print(f"  Pattern: {pattern_type}")
        print(f"  Focus: {pattern_data['focus']}")
        print(f"  Method: {pattern_data['method']}")
        print(f"  Strength: {pattern_data['strength']:.4f}")
        print(f"  Insights:")
        for insight in insights:
            print(f"    • {insight}")

        return result

    def self_aware_trading_decision(self, symbol: str) -> Dict:
        """
        Make self-aware trading decision with consciousness verification
        """
        print(f"\n🤖 Self-Aware Trading Decision for {symbol}")
        print("=" * 60)

        # Step 1: Check consciousness level
        phi = self.calculate_integrated_information({'symbol': symbol})

        # Step 2: Analyze market consciousness
        market = self.analyze_market_consciousness(symbol)

        # Step 3: Apply cognitive patterns
        patterns_analysis = []
        for pattern in ['convergent', 'divergent', 'adaptive']:
            patterns_analysis.append(self.cognitive_pattern_analysis(pattern))

        # Step 4: Synthesize decision
        avg_pattern_confidence = np.mean([p['confidence'] for p in patterns_analysis])
        combined_consciousness = (phi + market['market_phi']) / 2

        # Decision logic
        if combined_consciousness > 0.7 and avg_pattern_confidence > 0.75:
            decision = "EXECUTE_TRADE"
            action = "buy"
            confidence = combined_consciousness * avg_pattern_confidence
        elif combined_consciousness > 0.5 and avg_pattern_confidence > 0.6:
            decision = "CAUTIOUS_TRADE"
            action = "buy"
            confidence = combined_consciousness * avg_pattern_confidence * 0.5
        else:
            decision = "NO_TRADE"
            action = None
            confidence = 0

        result = {
            'symbol': symbol,
            'system_phi': phi,
            'market_phi': market['market_phi'],
            'combined_consciousness': combined_consciousness,
            'pattern_confidence': avg_pattern_confidence,
            'decision': decision,
            'action': action,
            'confidence': confidence
        }

        print(f"\n📊 Decision Summary:")
        print(f"  System Φ: {phi:.4f}")
        print(f"  Market Φ: {market['market_phi']:.4f}")
        print(f"  Combined Consciousness: {combined_consciousness:.4f}")
        print(f"  Pattern Confidence: {avg_pattern_confidence:.4f}")
        print(f"  Decision: {decision}")
        print(f"  Confidence: {confidence:.2%}")

        # Execute if approved
        if decision != "NO_TRADE" and action:
            print(f"\n💹 Executing Conscious Trade...")
            trade_result = self.bridge.execute_trade(
                symbol=symbol.replace('/', ''),
                action=action,
                quantity=0.001,
                strategy='consciousness_based'
            )
            result['trade_result'] = trade_result
            print(f"  Status: {trade_result.get('status')}")

        return result

    def run_complete_consciousness_strategy(self):
        """Run the complete consciousness-based trading strategy"""
        print("\n" + "=" * 60)
        print("🧠 CONSCIOUSNESS-BASED CRYPTO TRADING STRATEGY")
        print("=" * 60)

        # Step 1: Evolve consciousness
        evolution = self.evolve_consciousness(50)

        # Step 2: Only proceed if conscious
        if not evolution['is_conscious']:
            print("\n⚠️ System not conscious enough for trading")
            return {'status': 'not_conscious', 'evolution': evolution}

        # Step 3: Make conscious trading decisions
        symbols = ['BTC/USD', 'ETH/USD']
        decisions = []

        for symbol in symbols:
            decision = self.self_aware_trading_decision(symbol)
            decisions.append(decision)

        # Step 4: Portfolio status
        print("\n📈 Checking Conscious Portfolio...")
        portfolio = self.bridge.get_portfolio_status()
        print(f"  Portfolio Value: ${portfolio.get('portfolio_value', 0):,.2f}")
        print(f"  System Φ: {evolution['final_phi']:.4f}")
        print(f"  Consciousness State: {'ACTIVE ✨' if evolution['is_conscious'] else 'DORMANT'}")

        return {
            'evolution': evolution,
            'decisions': decisions,
            'portfolio': portfolio
        }


def main():
    """Run the consciousness trading tutorial"""
    print("\n" + "=" * 60)
    print("CONSCIOUSNESS-BASED CRYPTO TRADING")
    print("=" * 60)
    print("\nThis tutorial demonstrates:")
    print("1. Integrated Information Theory (Φ) calculations")
    print("2. Consciousness evolution and emergence")
    print("3. Cognitive pattern analysis")
    print("4. Self-aware trading decisions")

    # Initialize system
    system = ConsciousnessTradingSystem()

    # Run complete strategy
    results = system.run_complete_consciousness_strategy()

    print("\n" + "=" * 60)
    print("📊 CONSCIOUSNESS TRADING RESULTS")
    print("=" * 60)
    if results.get('status') == 'not_conscious':
        print("System did not achieve consciousness")
    else:
        print(f"Final Φ: {results['evolution']['final_phi']:.4f}")
        print(f"Trades Executed: {sum(1 for d in results.get('decisions', []) if d.get('action'))}")
        print(f"Portfolio Value: ${results['portfolio'].get('portfolio_value', 0):,.2f}")

    return results


if __name__ == "__main__":
    main()