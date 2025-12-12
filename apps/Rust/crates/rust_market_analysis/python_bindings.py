#!/usr/bin/env python3
"""
Python bindings for Rust Market Analysis System

This module provides a Python interface to the high-performance Rust-based
market analysis system, including:
- Antifragility detection
- Whale activity monitoring  
- Self-Organized Criticality analysis
- Panarchy cycle detection
- Fibonacci pattern recognition
- Black Swan event detection
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Try to import the compiled Rust module
try:
    import market_analysis
    RUST_AVAILABLE = True
    print("✅ Rust market analysis module loaded successfully")
except ImportError as e:
    RUST_AVAILABLE = False
    print(f"⚠️  Rust module not available: {e}")
    print("   Falling back to Python implementation")

class RustMarketAnalyzer:
    """
    Python wrapper for the Rust-based market analysis system
    """
    
    def __init__(self):
        if RUST_AVAILABLE:
            self.analyzer = market_analysis.MarketAnalysisEngine()
            self.engine = self.analyzer  # Backward compatibility
            self.backend = "rust"
        else:
            self.analyzer = None
            self.engine = None
            self.backend = "python_fallback"
            logging.warning("Rust backend not available, using Python fallback")
    
    def analyze_market(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Comprehensive market analysis combining all detection systems
        
        Args:
            data: Dictionary containing market data arrays:
                - 'close': Closing prices
                - 'volume': Trading volumes  
                - 'high': High prices
                - 'low': Low prices
                
        Returns:
            Dictionary with analysis results
        """
        if RUST_AVAILABLE and self.engine:
            # Convert numpy arrays to Python lists for Rust FFI
            rust_data = {
                'close': data['close'].tolist() if isinstance(data['close'], np.ndarray) else data['close'],
                'volume': data['volume'].tolist() if isinstance(data['volume'], np.ndarray) else data['volume'],
                'high': data['high'].tolist() if isinstance(data['high'], np.ndarray) else data['high'],
                'low': data['low'].tolist() if isinstance(data['low'], np.ndarray) else data['low'],
            }
            
            try:
                return self.engine.analyze_market(rust_data)
            except Exception as e:
                logging.error(f"Rust analysis failed: {e}")
                return self._python_fallback_analysis(data)
        else:
            return self._python_fallback_analysis(data)
    
    def run_comprehensive_analysis(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Direct access to the Rust run_comprehensive_analysis method
        
        Args:
            data: Dictionary containing market data arrays:
                - 'close': Closing prices
                - 'volume': Trading volumes  
                - 'high': High prices
                - 'low': Low prices
                
        Returns:
            Dictionary with detailed analysis results
        """
        if RUST_AVAILABLE and self.analyzer:
            # Convert numpy arrays to Python lists for Rust FFI
            rust_data = {
                'close': data['close'].tolist() if isinstance(data['close'], np.ndarray) else data['close'],
                'volume': data['volume'].tolist() if isinstance(data['volume'], np.ndarray) else data['volume'],
                'high': data['high'].tolist() if isinstance(data['high'], np.ndarray) else data['high'],
                'low': data['low'].tolist() if isinstance(data['low'], np.ndarray) else data['low'],
            }
            
            try:
                return self.analyzer.run_comprehensive_analysis(rust_data)
            except Exception as e:
                logging.error(f"Rust run_comprehensive_analysis failed: {e}")
                return self._python_fallback_analysis(data)
        else:
            return self._python_fallback_analysis(data)
    
    def get_profitable_pairs(self, pairs_data: Dict[str, Dict[str, np.ndarray]]) -> List[str]:
        """
        Get profitable pairs ranked by comprehensive analysis
        
        Args:
            pairs_data: Dictionary mapping pair names to market data
            
        Returns:
            List of pair names ranked by profitability score
        """
        if RUST_AVAILABLE and self.engine:
            # Convert data format for Rust
            rust_pairs_data = {}
            for pair, data in pairs_data.items():
                rust_pairs_data[pair] = {
                    'close': data['close'].tolist() if isinstance(data['close'], np.ndarray) else data['close'],
                    'volume': data['volume'].tolist() if isinstance(data['volume'], np.ndarray) else data['volume'],
                    'high': data['high'].tolist() if isinstance(data['high'], np.ndarray) else data['high'],
                    'low': data['low'].tolist() if isinstance(data['low'], np.ndarray) else data['low'],
                }
            
            try:
                return self.engine.get_profitable_pairs(rust_pairs_data)
            except Exception as e:
                logging.error(f"Rust pair ranking failed: {e}")
                return self._python_fallback_pair_ranking(pairs_data)
        else:
            return self._python_fallback_pair_ranking(pairs_data)
    
    def generate_trade_signals(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Real-time trade signal generation
        
        Args:
            data: Market data dictionary
            
        Returns:
            Dictionary with trade signals and risk metrics
        """
        if RUST_AVAILABLE and self.engine:
            rust_data = {
                'close': data['close'].tolist() if isinstance(data['close'], np.ndarray) else data['close'],
                'volume': data['volume'].tolist() if isinstance(data['volume'], np.ndarray) else data['volume'],
                'high': data['high'].tolist() if isinstance(data['high'], np.ndarray) else data['high'],
                'low': data['low'].tolist() if isinstance(data['low'], np.ndarray) else data['low'],
            }
            
            try:
                return self.engine.py_generate_trade_signals(rust_data)
            except Exception as e:
                logging.error(f"Rust signal generation failed: {e}")
                return self._python_fallback_signals(data)
        else:
            return self._python_fallback_signals(data)
    
    def _python_fallback_analysis(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Python fallback implementation for market analysis
        """
        prices = np.array(data['close'])
        volumes = np.array(data['volume'])
        
        # Simple fallback implementations
        antifragility_score = self._calculate_simple_antifragility(prices)
        whale_activity = self._detect_simple_whale_activity(prices, volumes)
        criticality_level = self._calculate_simple_criticality(prices)
        panarchy_phase = self._detect_simple_panarchy_phase(prices)
        fibonacci_levels = self._calculate_simple_fibonacci_levels(prices)
        black_swan_probability = self._calculate_simple_black_swan_prob(prices)
        
        overall_score = (
            antifragility_score * 0.3 +
            whale_activity * 0.2 +
            (1.0 - criticality_level) * 0.2 +
            (1.0 - black_swan_probability) * 0.3
        )
        
        trade_action = "BUY" if overall_score > 0.6 else "SELL" if overall_score < 0.4 else "HOLD"
        
        return {
            'antifragility': antifragility_score,
            'whale_activity': {'major_whale_detected': whale_activity > 0.7, 'whale_direction': 1.0 if whale_activity > 0.5 else -1.0},
            'criticality_level': criticality_level,
            'panarchy_phase': panarchy_phase,
            'fibonacci_levels': fibonacci_levels,
            'black_swan_probability': black_swan_probability,
            'overall_score': overall_score,
            'trade_recommendation': trade_action,
        }
    
    def _python_fallback_pair_ranking(self, pairs_data: Dict[str, Dict[str, np.ndarray]]) -> List[str]:
        """
        Python fallback for pair ranking
        """
        pair_scores = []
        
        for pair, data in pairs_data.items():
            analysis = self._python_fallback_analysis(data)
            score = analysis['overall_score']
            pair_scores.append((pair, score))
        
        # Sort by score descending
        pair_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [pair for pair, score in pair_scores[:10]]
    
    def _python_fallback_signals(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Python fallback for signal generation
        """
        analysis = self._python_fallback_analysis(data)
        overall_score = analysis['overall_score']
        
        return {
            'enter_long': overall_score > 0.7,
            'enter_short': overall_score < 0.3,
            'confidence': overall_score * 100.0 if overall_score > 0.5 else (1.0 - overall_score) * 100.0,
            'exit_long': overall_score < 0.4,
            'exit_short': overall_score > 0.6,
            'urgency': 'HIGH' if analysis['black_swan_probability'] > 0.7 else 'NORMAL',
            'position_size_multiplier': max(0.1, min(2.0, overall_score * 1.5)),
            'risk_level': 'HIGH' if analysis['black_swan_probability'] > 0.5 else 'MEDIUM' if overall_score < 0.5 else 'LOW',
        }
    
    # Simple fallback implementations
    def _calculate_simple_antifragility(self, prices: np.ndarray) -> float:
        """Simple antifragility calculation"""
        if len(prices) < 20:
            return 0.5
        
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        
        # Higher volatility tolerance indicates antifragility
        return min(1.0, volatility * 50)
    
    def _detect_simple_whale_activity(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Simple whale detection"""
        if len(volumes) < 10:
            return 0.0
        
        recent_volume = np.mean(volumes[-5:])
        historical_volume = np.mean(volumes[:-5])
        
        volume_ratio = recent_volume / historical_volume if historical_volume > 0 else 1.0
        return min(1.0, max(0.0, (volume_ratio - 1.0) / 2.0))
    
    def _calculate_simple_criticality(self, prices: np.ndarray) -> float:
        """Simple criticality calculation"""
        if len(prices) < 20:
            return 0.5
        
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        
        # Higher volatility indicates higher criticality
        return min(1.0, volatility * 30)
    
    def _detect_simple_panarchy_phase(self, prices: np.ndarray) -> str:
        """Simple panarchy phase detection"""
        if len(prices) < 20:
            return "unknown"
        
        short_ma = np.mean(prices[-10:])
        long_ma = np.mean(prices[-20:])
        trend = (short_ma - long_ma) / long_ma
        
        if trend > 0.02:
            return "growth"
        elif trend > -0.02:
            return "conservation"
        elif trend > -0.05:
            return "release"
        else:
            return "reorganization"
    
    def _calculate_simple_fibonacci_levels(self, prices: np.ndarray) -> List[float]:
        """Simple Fibonacci level calculation"""
        if len(prices) < 10:
            return []
        
        high = np.max(prices)
        low = np.min(prices)
        diff = high - low
        
        fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
        levels = [high - diff * ratio for ratio in fib_ratios]
        
        return levels
    
    def _calculate_simple_black_swan_prob(self, prices: np.ndarray) -> float:
        """Simple Black Swan probability calculation"""
        if len(prices) < 20:
            return 0.1
        
        returns = np.diff(prices) / prices[:-1]
        
        # Look for extreme movements
        extreme_threshold = 3 * np.std(returns)
        extreme_count = np.sum(np.abs(returns) > extreme_threshold)
        
        return min(1.0, extreme_count / len(returns) * 10)

def build_rust_module():
    """
    Build the Rust module using maturin
    """
    try:
        import subprocess
        import sys
        
        print("🔧 Building Rust market analysis module...")
        
        # Change to rust directory
        rust_dir = Path(__file__).parent
        
        # Install maturin if not available
        try:
            import maturin
        except ImportError:
            print("   Installing maturin...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "maturin"])
        
        # Build the module
        result = subprocess.run(
            ["maturin", "develop", "--release"],
            cwd=rust_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Rust module built successfully!")
            return True
        else:
            print(f"❌ Build failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Build error: {e}")
        return False

# Example usage and testing
if __name__ == "__main__":
    print("🚀 Testing Rust Market Analysis System")
    
    # Create test data
    np.random.seed(42)
    n_points = 100
    
    prices = 100 + np.cumsum(np.random.normal(0, 1, n_points))
    volumes = np.random.uniform(1000, 5000, n_points)
    highs = prices * (1 + np.random.uniform(0, 0.02, n_points))
    lows = prices * (1 - np.random.uniform(0, 0.02, n_points))
    
    test_data = {
        'close': prices,
        'volume': volumes,
        'high': highs,
        'low': lows,
    }
    
    # Initialize analyzer
    analyzer = RustMarketAnalyzer()
    print(f"Using backend: {analyzer.backend}")
    
    # Run analysis
    print("\n📊 Running comprehensive analysis...")
    results = analyzer.analyze_market(test_data)
    
    print(f"   Antifragility: {results.get('antifragility', 0):.3f}")
    print(f"   Criticality: {results.get('criticality_level', 0):.3f}")
    print(f"   Panarchy Phase: {results.get('panarchy_phase', 'unknown')}")
    print(f"   Black Swan Prob: {results.get('black_swan_probability', 0):.3f}")
    print(f"   Overall Score: {results.get('overall_score', 0):.3f}")
    print(f"   Trade Recommendation: {results.get('trade_recommendation', 'UNKNOWN')}")
    
    # Test signal generation
    print("\n⚡ Generating trade signals...")
    signals = analyzer.generate_trade_signals(test_data)
    
    print(f"   Enter Long: {signals.get('enter_long', False)}")
    print(f"   Enter Short: {signals.get('enter_short', False)}")
    print(f"   Confidence: {signals.get('confidence', 0):.1f}%")
    print(f"   Risk Level: {signals.get('risk_level', 'UNKNOWN')}")
    
    print("\n✅ Testing completed successfully!")