#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical LSTM Trading Integration Example
Shows how to use the enhanced LSTM models for cryptocurrency trading
"""

import numpy as np
import torch
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our enhanced LSTM
from enhanced_lstm_integration import create_enhanced_lstm, EnhancedLSTMConfig

class LSTMTradingPredictor:
    """Trading predictor using enhanced LSTM models"""
    
    def __init__(self, config: Optional[EnhancedLSTMConfig] = None):
        """Initialize the trading predictor"""
        
        if config is None:
            # Default configuration optimized for crypto trading
            self.config = EnhancedLSTMConfig(
                input_size=50,  # OHLCV + technical indicators
                hidden_size=64,
                num_layers=2,
                use_biological_activation=True,  # Better market dynamics
                use_multi_timeframe=False,  # Disabled to avoid hanging
                use_advanced_attention=True,  # Cached attention for speed
                use_quantum=False,  # Start with classical
                cache_size=1000
            )
        else:
            self.config = config
        
        # Create model
        self.model = create_enhanced_lstm(self.config)
        self.model.eval()  # Set to evaluation mode
        
        # Feature scaler parameters (would be fitted on training data)
        self.feature_mean = None
        self.feature_std = None
        
        logger.info(f"LSTM Trading Predictor initialized with {self.config.hidden_size} hidden units")
    
    def prepare_features(self, ohlcv_data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features from OHLCV data
        
        Args:
            ohlcv_data: DataFrame with columns [open, high, low, close, volume]
            
        Returns:
            Feature matrix with technical indicators
        """
        features = []
        
        # Price features
        features.append(ohlcv_data['close'].values)
        features.append(ohlcv_data['volume'].values)
        
        # Price changes
        features.append(ohlcv_data['close'].pct_change().fillna(0).values)
        features.append(ohlcv_data['volume'].pct_change().fillna(0).values)
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            ma = ohlcv_data['close'].rolling(period).mean().fillna(method='bfill')
            features.append(ma.values)
            features.append((ohlcv_data['close'] - ma).values)  # Distance from MA
        
        # RSI
        delta = ohlcv_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi.fillna(50).values)
        
        # Bollinger Bands
        sma20 = ohlcv_data['close'].rolling(20).mean()
        std20 = ohlcv_data['close'].rolling(20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        features.append(((ohlcv_data['close'] - bb_lower) / (bb_upper - bb_lower)).fillna(0.5).values)
        
        # MACD
        ema12 = ohlcv_data['close'].ewm(span=12).mean()
        ema26 = ohlcv_data['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        features.append(macd.fillna(0).values)
        features.append(signal.fillna(0).values)
        features.append((macd - signal).fillna(0).values)  # MACD histogram
        
        # Volume indicators
        features.append((ohlcv_data['volume'] / ohlcv_data['volume'].rolling(20).mean()).fillna(1).values)
        
        # Volatility
        features.append(ohlcv_data['close'].pct_change().rolling(20).std().fillna(0).values)
        
        # High-Low spread
        features.append(((ohlcv_data['high'] - ohlcv_data['low']) / ohlcv_data['close']).values)
        
        # Stack all features
        feature_matrix = np.column_stack(features)
        
        # Handle any remaining NaN values
        feature_matrix = np.nan_to_num(feature_matrix, 0)
        
        # Pad or truncate to match expected input size
        n_features = feature_matrix.shape[1]
        if n_features < self.config.input_size:
            # Pad with zeros
            padding = np.zeros((len(feature_matrix), self.config.input_size - n_features))
            feature_matrix = np.hstack([feature_matrix, padding])
        elif n_features > self.config.input_size:
            # Use first input_size features
            feature_matrix = feature_matrix[:, :self.config.input_size]
        
        return feature_matrix
    
    def create_sequences(self, features: np.ndarray, seq_len: int = 60) -> np.ndarray:
        """Create sequences for LSTM input"""
        sequences = []
        
        for i in range(len(features) - seq_len + 1):
            sequences.append(features[i:i+seq_len])
        
        return np.array(sequences)
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using stored statistics"""
        if self.feature_mean is None:
            # First time - compute statistics
            self.feature_mean = np.mean(features, axis=0)
            self.feature_std = np.std(features, axis=0) + 1e-8
        
        return (features - self.feature_mean) / self.feature_std
    
    def predict(self, ohlcv_data: pd.DataFrame, seq_len: int = 60) -> Dict[str, np.ndarray]:
        """
        Make predictions on OHLCV data
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            seq_len: Sequence length for LSTM
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        # Prepare features
        features = self.prepare_features(ohlcv_data)
        features = self.normalize_features(features)
        
        # Create sequences
        sequences = self.create_sequences(features, seq_len)
        
        if len(sequences) == 0:
            logger.warning("Not enough data for prediction")
            return {"predictions": np.array([]), "confidence": np.array([])}
        
        # Convert to tensor
        x = torch.FloatTensor(sequences)
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(x)
            
            # Get predictions from the last timestep
            predictions = outputs[:, -1, 0].numpy()
            
            # Calculate confidence based on model statistics
            if hasattr(self.model, 'get_performance_stats'):
                stats = self.model.get_performance_stats()
                # Simple confidence based on biological activation usage
                base_confidence = 0.7 if stats.get('biological_activation', False) else 0.5
            else:
                base_confidence = 0.6
            
            # Adjust confidence based on prediction strength
            confidence = base_confidence + 0.3 * np.abs(predictions)
            confidence = np.clip(confidence, 0, 1)
        
        return {
            "predictions": predictions,
            "confidence": confidence,
            "timestamps": ohlcv_data.index[-len(predictions):].tolist()
        }
    
    def generate_signals(self, predictions: Dict[str, np.ndarray], threshold: float = 0.02) -> List[Dict]:
        """
        Generate trading signals from predictions
        
        Args:
            predictions: Output from predict()
            threshold: Minimum prediction strength for signal
            
        Returns:
            List of trading signals
        """
        signals = []
        
        pred_values = predictions["predictions"]
        confidence = predictions["confidence"]
        timestamps = predictions["timestamps"]
        
        for i, (pred, conf, ts) in enumerate(zip(pred_values, confidence, timestamps)):
            if abs(pred) > threshold and conf > 0.6:
                signal = {
                    "timestamp": ts,
                    "action": "BUY" if pred > 0 else "SELL",
                    "strength": abs(pred),
                    "confidence": conf,
                    "index": i
                }
                signals.append(signal)
        
        return signals


def demonstrate_trading_example():
    """Demonstrate the LSTM trading predictor"""
    print("\n" + "="*60)
    print("LSTM TRADING INTEGRATION EXAMPLE")
    print("="*60 + "\n")
    
    # Generate synthetic OHLCV data for demonstration
    dates = pd.date_range(start='2024-01-01', periods=500, freq='1H')
    
    # Simulate price with trend and noise
    trend = np.linspace(30000, 35000, 500)
    seasonal = 1000 * np.sin(np.linspace(0, 4*np.pi, 500))
    noise = np.random.randn(500) * 500
    
    close_prices = trend + seasonal + noise
    
    ohlcv_data = pd.DataFrame({
        'open': close_prices + np.random.randn(500) * 100,
        'high': close_prices + np.abs(np.random.randn(500) * 200),
        'low': close_prices - np.abs(np.random.randn(500) * 200),
        'close': close_prices,
        'volume': np.abs(np.random.randn(500) * 1000000 + 5000000)
    }, index=dates)
    
    print("1. Created synthetic OHLCV data")
    print(f"   Date range: {dates[0]} to {dates[-1]}")
    print(f"   Price range: ${close_prices.min():.2f} to ${close_prices.max():.2f}")
    
    # Create predictor
    print("\n2. Initializing LSTM predictor...")
    predictor = LSTMTradingPredictor()
    
    # Make predictions
    print("\n3. Making predictions...")
    predictions = predictor.predict(ohlcv_data, seq_len=60)
    
    print(f"   Generated {len(predictions['predictions'])} predictions")
    print(f"   Average confidence: {np.mean(predictions['confidence']):.2%}")
    
    # Generate trading signals
    print("\n4. Generating trading signals...")
    signals = predictor.generate_signals(predictions, threshold=0.01)
    
    print(f"   Found {len(signals)} trading signals")
    
    # Show first few signals
    if signals:
        print("\n   First 5 signals:")
        for signal in signals[:5]:
            print(f"   - {signal['timestamp']}: {signal['action']} "
                  f"(strength: {signal['strength']:.3f}, confidence: {signal['confidence']:.2%})")
    
    # Calculate simple performance metrics
    print("\n5. Performance summary:")
    buy_signals = [s for s in signals if s['action'] == 'BUY']
    sell_signals = [s for s in signals if s['action'] == 'SELL']
    
    print(f"   Buy signals: {len(buy_signals)}")
    print(f"   Sell signals: {len(sell_signals)}")
    print(f"   Avg signal confidence: {np.mean([s['confidence'] for s in signals]):.2%}" if signals else "   No signals")
    
    # Show how to use with real data
    print("\n" + "="*60)
    print("USAGE WITH REAL DATA")
    print("="*60)
    
    usage_example = '''
# Load real market data (e.g., from ccxt)
import ccxt
exchange = ccxt.binance()
ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=500)
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# Create predictor with custom config
config = EnhancedLSTMConfig(
    input_size=50,
    hidden_size=128,  # Larger for complex patterns
    use_biological_activation=True,
    use_advanced_attention=True,
    use_quantum=False  # Set True for experimental quantum features
)

predictor = LSTMTradingPredictor(config)

# Make predictions
predictions = predictor.predict(df, seq_len=60)
signals = predictor.generate_signals(predictions)

# Use signals in your trading strategy
for signal in signals:
    if signal['confidence'] > 0.8:  # High confidence only
        execute_trade(signal)
'''
    
    print(usage_example)
    
    return predictor, predictions, signals


if __name__ == "__main__":
    # Run demonstration
    predictor, predictions, signals = demonstrate_trading_example()
    
    # Save example signals
    if signals:
        with open("lstm_trading_signals_example.txt", "w") as f:
            f.write("LSTM Trading Signals Example\n")
            f.write(f"Generated at: {datetime.now()}\n\n")
            for signal in signals[:20]:  # First 20 signals
                f.write(f"{signal['timestamp']}: {signal['action']} "
                       f"(strength: {signal['strength']:.3f}, confidence: {signal['confidence']:.2%})\n")
        print("\nExample signals saved to lstm_trading_signals_example.txt")
    
    print("\n✅ LSTM trading integration example complete!")