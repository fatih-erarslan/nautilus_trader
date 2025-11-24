#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 12:16:11 2025

@author: ashina
"""

"""
Example usage of the advanced market detectors and analyzers.
This shows how to integrate them with your existing Black Swan, Whale Detector
and CDFA system.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import the detection modules
# Assuming the detectors are saved in a file named 'advanced_detectors.py'
from advanced_detectors import (
    AccumulationDetector, DistributionDetector, 
    ConfluenceAreaDetector, BubbleDetector,
    TopologicalDataAnalyzer, TemporalPatternAnalyzer
)

# Simulated import of your existing components
# Replace these with your actual imports
from cdfa_extensions.detectors import BlackSwanDetector, WhaleDetector
from advanced_cdfa import AdvancedCDFA, AdvancedCDFAConfig

# Load sample data
# Replace this with your actual data loading code
def load_sample_data(symbol='BTC/USD', timeframe='1h', limit=500):
    """Load or generate sample OHLCV data for testing."""
    # This is a placeholder - in a real implementation you would load actual data
    dates = pd.date_range(end=datetime.now(), periods=limit, freq=timeframe)
    
    # Generate synthetic price data with some patterns
    np.random.seed(42)  # For reproducibility
    close = np.random.normal(size=limit).cumsum() + 10000
    high = close * (1 + np.random.random(size=limit) * 0.02)
    low = close * (1 - np.random.random(size=limit) * 0.02)
    open_price = low + (high - low) * np.random.random(size=limit)
    volume = np.random.exponential(scale=1000, size=limit)
    
    # Add some volatility patterns
    for i in range(10, limit, 50):
        close[i:i+10] = close[i-1] * (1 + np.cumsum(np.random.normal(0, 0.02, 10)))
    
    # Simulate Bitcoin market trends
    # Add a bubble pattern
    bubble_start = 300
    bubble_length = 50
    bubble_factor = np.exp(np.linspace(0, 0.5, bubble_length))
    close[bubble_start:bubble_start+bubble_length] *= bubble_factor
    high[bubble_start:bubble_start+bubble_length] *= bubble_factor
    low[bubble_start:bubble_start+bubble_length] *= bubble_factor
    open_price[bubble_start:bubble_start+bubble_length] *= bubble_factor
    volume[bubble_start:bubble_start+bubble_length] *= np.linspace(1, 3, bubble_length)
    
    # Add an accumulation pattern
    accum_start = 150
    accum_length = 40
    close[accum_start:accum_start+accum_length] = np.linspace(close[accum_start-1], close[accum_start-1]*1.05, accum_length)
    volume[accum_start:accum_start+accum_length] = volume[accum_start:accum_start+accum_length] * np.linspace(1.5, 0.5, accum_length)
    
    # Create DataFrame
    data = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    # Add some technical indicators
    data['rsi'] = compute_rsi(data['close'])
    data['volatility'] = compute_volatility(data['close'])
    
    return data

def compute_rsi(prices, window=14):
    """Calculate RSI."""
    deltas = np.diff(prices)
    seed = deltas[:window+1]
    up = seed[seed >= 0].sum()/window
    down = -seed[seed < 0].sum()/window
    rs = up/down if down != 0 else np.inf
    rsi = np.zeros_like(prices)
    rsi[:window] = 100. - 100./(1. + rs)
    
    for i in range(window, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
            
        up = (up * (window - 1) + upval) / window
        down = (down * (window - 1) + downval) / window
        
        rs = up/down if down != 0 else np.inf
        rsi[i] = 100. - 100./(1. + rs)
    
    return rsi

def compute_volatility(prices, window=20):
    """Calculate price volatility."""
    volatility = np.zeros_like(prices)
    for i in range(window, len(prices)):
        volatility[i] = np.std(prices[i-window+1:i+1])
    return volatility

def main():
    # Step 1: Load data
    print("Loading data...")
    data = load_sample_data()
    print(f"Loaded {len(data)} data points")
    
    # Step 2: Initialize detectors
    print("\nInitializing detectors...")
    accumulation_detector = AccumulationDetector(lookback_period=30, sensitivity=0.6)
    distribution_detector = DistributionDetector(lookback_period=30, sensitivity=0.6)
    confluence_detector = ConfluenceAreaDetector(num_indicators=2, window_size=20)
    bubble_detector = BubbleDetector(lookback_period=60, sensitivity=0.6)
    topological_analyzer = TopologicalDataAnalyzer(max_dimension=1, window_size=20)
    temporal_analyzer = TemporalPatternAnalyzer(max_cycle_length=50, min_cycle_length=5)
    
    # Also initialize your existing detectors
    blackswan_detector = BlackSwanDetector()  # Replace with your actual initialization
    whale_detector = WhaleDetector()  # Replace with your actual initialization
    
    # Step 3: Run detection algorithms
    print("\nRunning detection algorithms...")
    
    # Detect accumulation zones
    accumulation_zones = accumulation_detector.detect(data)
    print(f"Detected {accumulation_zones.sum()} accumulation zones")
    
    # Detect distribution zones
    distribution_zones = distribution_detector.detect(data)
    print(f"Detected {distribution_zones.sum()} distribution zones")
    
    # Find confluence areas
    confluence_areas = confluence_detector.detect(data, current_price=data['close'].iloc[-1])
    print(f"Found {len(confluence_areas)} confluence areas")
    
    # Detect bubble conditions
    bubble_indicator, bubble_probability = bubble_detector.detect(data)
    print(f"Detected bubble conditions in {bubble_indicator.sum()} periods")
    
    # Run topological analysis
    topo_features = topological_analyzer.analyze(data, columns=['close'])
    print(f"Extracted {len(topo_features.columns)} topological features")
    
    # Run temporal pattern analysis
    temporal_results = temporal_analyzer.analyze(data)
    print(f"Found {len(temporal_results.get('dominant_cycles', []))} dominant cycles")
    
    # Run your existing detectors
    blackswan_signals = blackswan_detector.detect(data)
    whale_signals = whale_detector.detect(data)
    
    # Step 4: Combine signals using CDFA
    print("\nCombining signals using CDFA...")
    
    # Create a combined DataFrame with all signals
    signals = pd.DataFrame(index=data.index)
    signals['accumulation'] = accumulation_zones
    signals['distribution'] = distribution_zones
    signals['bubble'] = bubble_indicator
    signals['blackswan'] = blackswan_signals
    signals['whale'] = whale_signals
    
    # Add temporal and topological features if available
    if len(topo_features) > 0:
        # Select key topological features
        key_topo_features = ['close_total_persistence_0', 'close_persistence_entropy_0']
        for feature in key_topo_features:
            if feature in topo_features.columns:
                signals[feature] = topo_features[feature]
    
    # Add temporal features
    signals['hurst_exponent'] = temporal_results.get('hurst_exponent', 0)
    signals['self_similarity'] = np.mean(temporal_results.get('similarity_scores', [0]))
    
    # Run CDFA to combine signals
    cdfa_system = AdvancedCDFA()  # Replace with your actual CDFA system
    fused_signal = cdfa_system.fuse_signals(signals)
    
    # Step 5: Visualize results
    print("\nVisualizing results...")
    
    # Plot accumulation and distribution zones
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['close'], label='Price', color='blue', alpha=0.6)
    
    # Highlight accumulation zones
    for i in range(len(accumulation_zones)):
        if accumulation_zones.iloc[i] == 1:
            plt.axvspan(accumulation_zones.index[i], 
                       accumulation_zones.index[min(i+1, len(accumulation_zones)-1)], 
                       alpha=0.2, color='green')
    
    # Highlight distribution zones
    for i in range(len(distribution_zones)):
        if distribution_zones.iloc[i] == 1:
            plt.axvspan(distribution_zones.index[i], 
                       distribution_zones.index[min(i+1, len(distribution_zones)-1)], 
                       alpha=0.2, color='red')
    
    plt.title('Price with Accumulation and Distribution Zones')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot bubble probability
    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['close'], label='Price', color='blue')
    plt.title('Price Chart')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(bubble_probability.index, bubble_probability, color='red', label='Bubble Probability')
    plt.axhline(y=bubble_detector.sensitivity, linestyle='--', color='black', alpha=0.5)
    plt.fill_between(bubble_probability.index, bubble_probability, color='red', alpha=0.3)
    plt.title('Bubble Probability')
    plt.tight_layout()
    plt.show()
    
    # Plot fused CDFA signal
    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['close'], label='Price', color='blue')
    plt.title('Price Chart')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(fused_signal.index, fused_signal, color='purple', label='CDFA Fused Signal')
    plt.title('CDFA Fused Signal')
    plt.tight_layout()
    plt.show()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()