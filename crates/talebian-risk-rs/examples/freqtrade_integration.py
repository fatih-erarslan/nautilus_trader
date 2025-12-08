#!/usr/bin/env python3
"""
FreqTrade Integration Example for Talebian Risk Management

This example demonstrates how to integrate the aggressive Machiavellian
Talebian risk management system into a FreqTrade strategy.

Usage:
    python freqtrade_integration.py

Requirements:
    - talebian-risk-rs compiled with Python bindings
    - FreqTrade development environment
"""

import sys
import time
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Import the Talebian Risk management library
try:
    import talebian_risk_rs as tr
except ImportError:
    print("Error: talebian_risk_rs not found. Please compile with Python bindings:")
    print("cd talebian-risk-rs && maturin develop --features python-bindings")
    sys.exit(1)

class TalebianRiskFreqTradeStrategy:
    """
    FreqTrade strategy using aggressive Machiavellian Talebian risk management
    """
    
    def __init__(self, config_type: str = "aggressive"):
        """
        Initialize with specified configuration type
        
        Args:
            config_type: "aggressive", "conservative", or "extreme"
        """
        # Initialize Talebian risk configuration
        if config_type == "aggressive":
            self.config = tr.MacchiavelianConfig.aggressive_defaults()
        elif config_type == "conservative":
            self.config = tr.MacchiavelianConfig.conservative_baseline()
        elif config_type == "extreme":
            self.config = tr.MacchiavelianConfig.extreme_machiavellian()
        else:
            raise ValueError(f"Unknown config type: {config_type}")
        
        # Initialize risk engine
        self.risk_engine = tr.TalebianRiskEngine(self.config)
        
        # Performance tracking
        self.total_trades = 0
        self.successful_trades = 0
        self.total_return = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = 0.0
        
        print(f"✅ Initialized Talebian Risk Strategy ({config_type})")
        print(f"   Antifragility threshold: {self.config.antifragility_threshold:.3f}")
        print(f"   Kelly fraction: {self.config.kelly_fraction:.3f}")
        print(f"   Black swan threshold: {self.config.black_swan_threshold:.3f}")
        print(f"   Barbell safe ratio: {self.config.barbell_safe_ratio:.3f}")

    def analyze_market_conditions(self, dataframe) -> Dict:
        """
        Analyze current market conditions using Talebian risk framework
        
        Args:
            dataframe: FreqTrade OHLCV dataframe
            
        Returns:
            Dict with risk analysis results
        """
        try:
            # Extract latest market data
            latest = dataframe.iloc[-1]
            
            # Calculate returns from recent price data
            returns = []
            if len(dataframe) >= 20:
                prices = dataframe['close'].tail(20).values
                returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            
            # Calculate volume history
            volume_history = []
            if len(dataframe) >= 10:
                volume_history = dataframe['volume'].tail(10).tolist()
            
            # Estimate volatility from recent returns
            volatility = 0.02  # Default
            if len(returns) >= 5:
                import statistics
                volatility = statistics.stdev(returns) if len(set(returns)) > 1 else 0.02
            
            # Prepare market data for Talebian analysis
            market_data = {
                'timestamp': int(latest.name.timestamp()) if hasattr(latest.name, 'timestamp') else int(time.time()),
                'price': float(latest['close']),
                'volume': float(latest['volume']),
                'bid': float(latest['low']),  # Approximate bid as low
                'ask': float(latest['high']), # Approximate ask as high
                'bid_volume': float(latest['volume'] * 0.5),  # Estimate
                'ask_volume': float(latest['volume'] * 0.5),  # Estimate
                'volatility': volatility,
                'returns': returns[-10:] if len(returns) >= 10 else returns,
                'volume_history': volume_history
            }
            
            # Perform Talebian risk assessment
            assessment = self.risk_engine.assess_risk(market_data)
            
            return {
                'assessment': assessment,
                'market_data': market_data,
                'analysis_timestamp': datetime.now().isoformat(),
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Error in market analysis: {e}")
            return {
                'error': str(e),
                'success': False
            }

    def generate_trading_signal(self, dataframe) -> Tuple[str, float, Dict]:
        """
        Generate trading signal based on Talebian risk analysis
        
        Args:
            dataframe: FreqTrade OHLCV dataframe
            
        Returns:
            Tuple of (signal, position_size, metadata)
            signal: "BUY", "SELL", or "HOLD"
            position_size: Recommended position size (0.0 to 1.0)
            metadata: Additional information about the decision
        """
        analysis = self.analyze_market_conditions(dataframe)
        
        if not analysis['success']:
            return "HOLD", 0.0, {"error": analysis['error']}
        
        assessment = analysis['assessment']
        
        # Extract key metrics
        opportunity_score = assessment['parasitic_opportunity']['opportunity_score']
        whale_detected = assessment['whale_detection']['is_detected']
        whale_confidence = assessment['whale_detection']['confidence']
        recommended_size = assessment['recommended_position_size']
        confidence = assessment['confidence']
        antifragility_score = assessment['antifragility_score']
        
        # Decision logic based on Talebian principles
        signal = "HOLD"
        position_size = 0.0
        
        # Check if opportunity meets threshold
        if opportunity_score > self.config.parasitic_opportunity_threshold:
            if whale_detected and whale_confidence > 0.7:
                # High-confidence whale following (parasitic strategy)
                signal = "BUY"
                position_size = min(recommended_size * 1.2, 0.75)  # Boost for whale following
                reason = f"🐋 Whale following (confidence: {whale_confidence:.1%})"
                
            elif antifragility_score > self.config.antifragility_threshold:
                # Antifragile opportunity in volatile market
                signal = "BUY"
                position_size = recommended_size
                reason = f"📈 Antifragile opportunity (score: {antifragility_score:.3f})"
                
            elif opportunity_score > 0.8 and confidence > 0.8:
                # High-confidence opportunity
                signal = "BUY"
                position_size = recommended_size * 0.9  # Slightly reduced for safety
                reason = f"🎯 High opportunity (score: {opportunity_score:.3f})"
                
            else:
                # Moderate opportunity
                signal = "BUY"
                position_size = recommended_size * 0.6  # Conservative sizing
                reason = f"📊 Moderate opportunity (score: {opportunity_score:.3f})"
        
        else:
            # Below opportunity threshold
            reason = f"⏳ Below opportunity threshold ({opportunity_score:.3f} < {self.config.parasitic_opportunity_threshold:.3f})"
        
        # Risk controls
        if assessment['black_swan_probability'] > self.config.black_swan_threshold:
            if position_size > 0:
                position_size *= 0.5  # Reduce position due to black swan risk
                reason += f" | ⚠️ Black swan risk detected"
        
        # Final position size bounds
        position_size = max(0.0, min(position_size, 0.8))  # 0% to 80% max
        
        metadata = {
            'reason': reason,
            'opportunity_score': opportunity_score,
            'whale_detected': whale_detected,
            'whale_confidence': whale_confidence,
            'antifragility_score': antifragility_score,
            'confidence': confidence,
            'black_swan_probability': assessment['black_swan_probability'],
            'kelly_fraction': assessment['kelly_fraction'],
            'barbell_allocation': assessment['barbell_allocation'],
            'analysis_timestamp': analysis['analysis_timestamp']
        }
        
        return signal, position_size, metadata

    def record_trade_result(self, return_pct: float, was_whale_trade: bool = False, momentum_score: float = 0.5):
        """
        Record trade result for learning and adaptation
        
        Args:
            return_pct: Trade return as percentage (e.g., 0.02 for 2%)
            was_whale_trade: Whether this was a whale-following trade
            momentum_score: Momentum strength at trade time (0.0 to 1.0)
        """
        try:
            self.risk_engine.record_trade_outcome(return_pct, was_whale_trade, momentum_score)
            
            # Update performance tracking
            self.total_trades += 1
            self.total_return += return_pct
            
            if return_pct > 0:
                self.successful_trades += 1
            
            # Update equity tracking
            current_equity = 1.0 + self.total_return
            self.peak_equity = max(self.peak_equity, current_equity)
            current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            print(f"📊 Trade recorded: {return_pct:+.1%} | Win rate: {self.get_win_rate():.1%} | Total return: {self.total_return:+.1%}")
            
        except Exception as e:
            print(f"❌ Error recording trade: {e}")

    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        try:
            status = self.risk_engine.get_status()
            
            return {
                'total_trades': self.total_trades,
                'win_rate': self.get_win_rate(),
                'total_return': self.total_return,
                'max_drawdown': self.max_drawdown,
                'sharpe_estimate': self.estimate_sharpe_ratio(),
                'engine_stats': status,
                'config_summary': {
                    'antifragility_threshold': self.config.antifragility_threshold,
                    'kelly_fraction': self.config.kelly_fraction,
                    'black_swan_threshold': self.config.black_swan_threshold,
                    'whale_volume_threshold': self.config.whale_volume_threshold,
                }
            }
        except Exception as e:
            return {'error': str(e)}

    def get_win_rate(self) -> float:
        """Calculate current win rate"""
        return (self.successful_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0

    def estimate_sharpe_ratio(self) -> float:
        """Estimate Sharpe ratio from current performance"""
        if self.total_trades < 10:
            return 0.0
        
        avg_return = self.total_return / self.total_trades
        # Simplified Sharpe estimation
        return avg_return / 0.02 if avg_return > 0 else 0.0

    def optimize_parameters(self, historical_returns: List[float]) -> Dict:
        """
        Suggest parameter optimizations based on historical performance
        
        Args:
            historical_returns: List of historical trade returns
            
        Returns:
            Dict with optimization suggestions
        """
        if len(historical_returns) < 20:
            return {'message': 'Insufficient data for optimization (need at least 20 trades)'}
        
        import statistics
        
        # Calculate performance metrics
        win_rate = sum(1 for r in historical_returns if r > 0) / len(historical_returns)
        avg_return = statistics.mean(historical_returns)
        volatility = statistics.stdev(historical_returns) if len(set(historical_returns)) > 1 else 0.01
        
        suggestions = []
        
        # Kelly fraction optimization
        optimal_kelly = avg_return / (volatility ** 2) if volatility > 0 else 0.25
        if abs(optimal_kelly - self.config.kelly_fraction) > 0.1:
            suggestions.append({
                'parameter': 'kelly_fraction',
                'current': self.config.kelly_fraction,
                'suggested': max(0.1, min(0.8, optimal_kelly)),
                'reason': f'Optimize for current win rate ({win_rate:.1%}) and volatility'
            })
        
        # Antifragility threshold adjustment
        if win_rate > 0.7 and avg_return > 0.01:
            # Performing well, could be more aggressive
            if self.config.antifragility_threshold > 0.3:
                suggestions.append({
                    'parameter': 'antifragility_threshold',
                    'current': self.config.antifragility_threshold,
                    'suggested': max(0.25, self.config.antifragility_threshold - 0.05),
                    'reason': 'High performance suggests room for more aggression'
                })
        elif win_rate < 0.5 or avg_return < 0:
            # Underperforming, should be more conservative
            suggestions.append({
                'parameter': 'antifragility_threshold',
                'current': self.config.antifragility_threshold,
                'suggested': min(0.6, self.config.antifragility_threshold + 0.1),
                'reason': 'Poor performance suggests need for more selectivity'
            })
        
        return {
            'performance_metrics': {
                'win_rate': win_rate,
                'avg_return': avg_return,
                'volatility': volatility,
                'sharpe_estimate': avg_return / volatility if volatility > 0 else 0
            },
            'suggestions': suggestions,
            'total_trades_analyzed': len(historical_returns)
        }


def demonstrate_freqtrade_integration():
    """
    Demonstrate the Talebian risk management integration with sample data
    """
    print("🚀 Talebian Risk Management - FreqTrade Integration Demo")
    print("=" * 60)
    
    # Initialize strategy
    strategy = TalebianRiskFreqTradeStrategy("aggressive")
    
    # Simulate sample market data (normally comes from FreqTrade)
    import pandas as pd
    import numpy as np
    
    # Create sample OHLCV data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    prices = 50000 + np.cumsum(np.random.randn(100) * 100)  # Random walk
    volumes = 1000 + np.random.exponential(500, 100)  # Exponential volume
    
    sample_data = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.random.rand(100) * 0.02),
        'low': prices * (1 - np.random.rand(100) * 0.02),
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    print("\n📊 Sample Market Data:")
    print(f"   Price range: ${prices.min():,.0f} - ${prices.max():,.0f}")
    print(f"   Average volume: {volumes.mean():,.0f}")
    
    # Analyze market conditions
    print("\n🔍 Analyzing Market Conditions...")
    analysis = strategy.analyze_market_conditions(sample_data)
    
    if analysis['success']:
        assessment = analysis['assessment']
        print(f"   ✅ Analysis successful")
        print(f"   Opportunity score: {assessment['parasitic_opportunity']['opportunity_score']:.3f}")
        print(f"   Whale detected: {assessment['whale_detection']['is_detected']}")
        print(f"   Antifragility score: {assessment['antifragility_score']:.3f}")
        print(f"   Recommended position: {assessment['recommended_position_size']:.1%}")
        print(f"   Confidence: {assessment['confidence']:.1%}")
    else:
        print(f"   ❌ Analysis failed: {analysis['error']}")
        return
    
    # Generate trading signal
    print("\n🎯 Generating Trading Signal...")
    signal, position_size, metadata = strategy.generate_trading_signal(sample_data)
    
    print(f"   Signal: {signal}")
    print(f"   Position size: {position_size:.1%}")
    print(f"   Reason: {metadata['reason']}")
    
    # Simulate some trades
    print("\n💰 Simulating Trade Results...")
    trade_returns = []
    
    for i in range(10):
        # Simulate random trade returns (normally based on actual trades)
        if np.random.rand() > 0.4:  # 60% win rate
            trade_return = np.random.uniform(0.005, 0.03)  # 0.5% to 3% profit
        else:
            trade_return = np.random.uniform(-0.02, -0.005)  # 0.5% to 2% loss
        
        was_whale = np.random.rand() > 0.7  # 30% whale trades
        momentum = np.random.uniform(0.3, 0.9)
        
        strategy.record_trade_result(trade_return, was_whale, momentum)
        trade_returns.append(trade_return)
    
    # Show performance summary
    print("\n📈 Performance Summary:")
    summary = strategy.get_performance_summary()
    
    print(f"   Total trades: {summary['total_trades']}")
    print(f"   Win rate: {summary['win_rate']:.1f}%")
    print(f"   Total return: {summary['total_return']:+.1%}")
    print(f"   Max drawdown: {summary['max_drawdown']:.1%}")
    
    # Parameter optimization suggestions
    print("\n🔧 Parameter Optimization:")
    optimization = strategy.optimize_parameters(trade_returns)
    
    if 'suggestions' in optimization:
        if optimization['suggestions']:
            for suggestion in optimization['suggestions']:
                print(f"   📝 {suggestion['parameter']}: {suggestion['current']:.3f} → {suggestion['suggested']:.3f}")
                print(f"      Reason: {suggestion['reason']}")
        else:
            print("   ✅ Parameters appear optimal for current performance")
    else:
        print(f"   ℹ️ {optimization['message']}")
    
    print("\n🎉 Demo completed successfully!")
    
    # Show quick-use functions
    print("\n⚡ Quick Assessment Functions:")
    
    # Quick risk assessment
    quick_result = tr.quick_risk_assessment(
        price=50000.0,
        volume=1500.0,
        volatility=0.03,
        returns=[0.01, 0.02, -0.005, 0.015],
        config=strategy.config
    )
    print(f"   Quick assessment position size: {quick_result['recommended_position_size']:.1%}")
    
    # Check aggressive conditions
    is_aggressive = tr.is_aggressive_conditions(
        volatility=0.04,
        volume_ratio=2.5,
        momentum=0.025
    )
    print(f"   Aggressive conditions detected: {is_aggressive}")
    
    # Calculate position size
    position = tr.calculate_position_size(
        opportunity_score=0.7,
        whale_confidence=0.8,
        volatility=0.03
    )
    print(f"   Calculated position size: {position:.1%}")


if __name__ == "__main__":
    try:
        demonstrate_freqtrade_integration()
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()