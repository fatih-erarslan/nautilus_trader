# mcp__ai-news-trader__quick_analysis

## Description
Get quick market analysis for a symbol with optional GPU acceleration. This tool provides rapid technical analysis, market indicators, and trading signals for any given symbol, optimized for real-time decision making.

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| symbol | string | Yes | N/A | Trading symbol to analyze (e.g., "AAPL", "BTC-USD", "EUR/USD") |
| use_gpu | boolean | No | false | Enable GPU acceleration for faster analysis |

```typescript
{
  symbol: string,     // Required: Trading symbol
  use_gpu?: boolean   // Optional: GPU acceleration (default: false)
}
```

## Return Value Structure
```json
{
  "symbol": "string",
  "timestamp": "string",         // ISO timestamp
  "current_price": number,
  "price_change": {
    "absolute": number,
    "percentage": number,
    "period": "string"           // "1d", "1h", etc.
  },
  "technical_indicators": {
    "rsi": {
      "value": number,           // 0-100
      "signal": "string"         // "oversold", "neutral", "overbought"
    },
    "macd": {
      "macd_line": number,
      "signal_line": number,
      "histogram": number,
      "signal": "string"         // "bullish", "bearish", "neutral"
    },
    "moving_averages": {
      "sma_20": number,
      "sma_50": number,
      "sma_200": number,
      "ema_12": number,
      "ema_26": number
    },
    "bollinger_bands": {
      "upper": number,
      "middle": number,
      "lower": number,
      "position": "string"       // "above_upper", "in_band", "below_lower"
    },
    "atr": number,               // Average True Range
    "volume": {
      "current": number,
      "average": number,
      "ratio": number            // Current/Average
    }
  },
  "support_resistance": {
    "support_levels": [number],
    "resistance_levels": [number],
    "pivot_point": number,
    "nearest_support": number,
    "nearest_resistance": number
  },
  "trend_analysis": {
    "primary_trend": "string",   // "bullish", "bearish", "sideways"
    "trend_strength": number,    // 0-100
    "trend_duration": "string",  // "short", "medium", "long"
    "momentum": "string"         // "increasing", "decreasing", "stable"
  },
  "trading_signals": {
    "composite_signal": "string", // "strong_buy", "buy", "neutral", "sell", "strong_sell"
    "confidence": number,        // 0-100
    "signals": [
      {
        "indicator": "string",
        "signal": "string",
        "strength": number
      }
    ]
  },
  "market_context": {
    "sector_performance": number,
    "market_sentiment": "string", // "bullish", "bearish", "neutral"
    "correlation_sp500": number,
    "volatility_rank": number,   // 0-100 percentile
    "volume_profile": "string"   // "accumulation", "distribution", "neutral"
  },
  "risk_metrics": {
    "volatility": number,
    "beta": number,
    "sharpe_ratio": number,
    "risk_score": number         // 0-100
  },
  "performance_stats": {
    "processing_time_ms": number,
    "gpu_utilized": boolean,
    "data_points_analyzed": number
  }
}
```

## Examples

### Example 1: Basic Symbol Analysis
```python
# Quick analysis of Apple stock
analysis = mcp__ai_news_trader__quick_analysis(symbol="AAPL")

print(f"=== {analysis['symbol']} Quick Analysis ===")
print(f"Current Price: ${analysis['current_price']:.2f}")
print(f"Change: {analysis['price_change']['percentage']:.2f}%")
print(f"Trading Signal: {analysis['trading_signals']['composite_signal']}")
print(f"Confidence: {analysis['trading_signals']['confidence']}%")
print(f"Primary Trend: {analysis['trend_analysis']['primary_trend']}")
```

### Example 2: GPU-Accelerated Multi-Symbol Analysis
```python
# Analyze multiple symbols with GPU acceleration
symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
analyses = []

import time
start_time = time.time()

for symbol in symbols:
    analysis = mcp__ai_news_trader__quick_analysis(
        symbol=symbol,
        use_gpu=True
    )
    analyses.append(analysis)

gpu_time = time.time() - start_time
print(f"GPU Analysis completed in {gpu_time:.2f}s")

# Rank by trading signal strength
ranked = sorted(analyses, 
                key=lambda x: x['trading_signals']['confidence'], 
                reverse=True)

print("\nTop Trading Opportunities:")
for analysis in ranked[:3]:
    print(f"{analysis['symbol']}: {analysis['trading_signals']['composite_signal']} "
          f"(Confidence: {analysis['trading_signals']['confidence']}%)")
```

### Example 3: Technical Indicator Dashboard
```python
# Comprehensive technical analysis display
analysis = mcp__ai_news_trader__quick_analysis(symbol="BTC-USD", use_gpu=True)

tech = analysis['technical_indicators']
print(f"=== Technical Indicators for {analysis['symbol']} ===")

# RSI Analysis
rsi = tech['rsi']
print(f"\nRSI: {rsi['value']:.2f} - {rsi['signal'].upper()}")
if rsi['value'] < 30:
    print("  ⚠️ Potentially oversold - consider buying opportunity")
elif rsi['value'] > 70:
    print("  ⚠️ Potentially overbought - consider taking profits")

# MACD Analysis
macd = tech['macd']
print(f"\nMACD Signal: {macd['signal'].upper()}")
print(f"  MACD Line: {macd['macd_line']:.4f}")
print(f"  Signal Line: {macd['signal_line']:.4f}")
print(f"  Histogram: {macd['histogram']:.4f}")

# Moving Averages
ma = tech['moving_averages']
price = analysis['current_price']
print(f"\nMoving Averages:")
print(f"  Price: ${price:.2f}")
print(f"  SMA 20: ${ma['sma_20']:.2f} ({((price/ma['sma_20'])-1)*100:+.2f}%)")
print(f"  SMA 50: ${ma['sma_50']:.2f} ({((price/ma['sma_50'])-1)*100:+.2f}%)")
print(f"  SMA 200: ${ma['sma_200']:.2f} ({((price/ma['sma_200'])-1)*100:+.2f}%)")

# Volume Analysis
vol = tech['volume']
print(f"\nVolume: {vol['current']:,} ({vol['ratio']:.2f}x average)")
if vol['ratio'] > 2:
    print("  📊 High volume - significant interest")
```

### Example 4: Support/Resistance Trading Strategy
```python
# Identify trading opportunities based on support/resistance
analysis = mcp__ai_news_trader__quick_analysis(symbol="EUR/USD", use_gpu=True)

sr = analysis['support_resistance']
current_price = analysis['current_price']

print(f"=== Support/Resistance Analysis for {analysis['symbol']} ===")
print(f"Current Price: {current_price:.4f}")
print(f"Pivot Point: {sr['pivot_point']:.4f}")
print(f"Nearest Support: {sr['nearest_support']:.4f}")
print(f"Nearest Resistance: {sr['nearest_resistance']:.4f}")

# Calculate distances
support_distance = (current_price - sr['nearest_support']) / current_price * 100
resistance_distance = (sr['nearest_resistance'] - current_price) / current_price * 100

print(f"\nDistance to Support: {support_distance:.2f}%")
print(f"Distance to Resistance: {resistance_distance:.2f}%")

# Trading decision logic
if support_distance < 1.0:
    print("⚠️ NEAR SUPPORT - Potential bounce opportunity")
    print(f"   Entry: {current_price:.4f}")
    print(f"   Stop Loss: {sr['nearest_support'] * 0.995:.4f}")
    print(f"   Target: {sr['pivot_point']:.4f}")
    
elif resistance_distance < 1.0:
    print("⚠️ NEAR RESISTANCE - Potential reversal")
    print(f"   Consider taking profits or shorting")
    
# Display all levels
print("\nAll Support Levels:", [f"{s:.4f}" for s in sr['support_levels']])
print("All Resistance Levels:", [f"{r:.4f}" for r in sr['resistance_levels']])
```

### Example 5: Real-Time Monitoring with Alerts
```python
# Monitor symbol with automated alerts
import time
from datetime import datetime

def monitor_symbol_realtime(symbol, threshold_confidence=70, duration_minutes=10):
    """Monitor symbol and alert on high-confidence signals"""
    end_time = time.time() + (duration_minutes * 60)
    previous_signal = None
    alerts = []
    
    print(f"Monitoring {symbol} for {duration_minutes} minutes...")
    print(f"Alert threshold: {threshold_confidence}% confidence\n")
    
    while time.time() < end_time:
        try:
            # Get analysis with GPU for speed
            analysis = mcp__ai_news_trader__quick_analysis(
                symbol=symbol,
                use_gpu=True
            )
            
            current_signal = analysis['trading_signals']['composite_signal']
            confidence = analysis['trading_signals']['confidence']
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            # Check for signal change
            if current_signal != previous_signal:
                print(f"[{timestamp}] Signal Change: {current_signal} "
                      f"(Confidence: {confidence}%)")
                previous_signal = current_signal
            
            # High confidence alerts
            if confidence >= threshold_confidence:
                alert_msg = (f"🚨 HIGH CONFIDENCE ALERT: {symbol} - "
                           f"{current_signal} @ ${analysis['current_price']:.2f}")
                print(f"[{timestamp}] {alert_msg}")
                alerts.append({
                    'time': timestamp,
                    'signal': current_signal,
                    'price': analysis['current_price'],
                    'confidence': confidence
                })
            
            # Risk alerts
            risk_score = analysis['risk_metrics']['risk_score']
            if risk_score > 80:
                print(f"[{timestamp}] ⚠️ HIGH RISK WARNING: Risk score {risk_score}/100")
            
            # Performance stats
            proc_time = analysis['performance_stats']['processing_time_ms']
            print(f"[{timestamp}] Price: ${analysis['current_price']:.2f} | "
                  f"Signal: {current_signal} | "
                  f"Processing: {proc_time}ms", end='\r')
            
            time.sleep(5)  # Check every 5 seconds
            
        except Exception as e:
            print(f"\nError during monitoring: {e}")
            time.sleep(30)  # Wait longer on error
    
    print(f"\n\nMonitoring complete. {len(alerts)} high-confidence alerts generated.")
    return alerts

# Usage
alerts = monitor_symbol_realtime("TSLA", threshold_confidence=75, duration_minutes=2)
```

## Common Use Cases

1. **Pre-Trade Analysis**: Quick assessment before entering positions
2. **Multi-Symbol Screening**: Rapid analysis of multiple symbols
3. **Technical Indicator Dashboard**: Real-time technical analysis
4. **Support/Resistance Trading**: Identify key price levels
5. **Risk Assessment**: Quick risk evaluation before trading
6. **Alert Generation**: Automated monitoring and notifications

## Error Handling Notes

- **Invalid Symbol**: Symbol not found or not supported
- **Data Unavailable**: Market closed or data feed issues
- **GPU Errors**: GPU not available when requested
- **Timeout Issues**: Analysis taking too long (especially without GPU)
- **Insufficient Data**: Not enough historical data for indicators

### Error Handling Example:
```python
def safe_quick_analysis(symbol, use_gpu=False, retry_count=3):
    """Perform quick analysis with comprehensive error handling"""
    
    for attempt in range(retry_count):
        try:
            # Attempt analysis
            result = mcp__ai_news_trader__quick_analysis(
                symbol=symbol,
                use_gpu=use_gpu
            )
            
            # Validate critical fields
            if not result.get('current_price'):
                raise ValueError(f"No price data for {symbol}")
            
            if not result.get('trading_signals'):
                raise ValueError(f"No trading signals generated for {symbol}")
            
            # Warn on degraded data
            if result['performance_stats']['data_points_analyzed'] < 100:
                print(f"Warning: Limited data available ({result['performance_stats']['data_points_analyzed']} points)")
            
            return result
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            
            # Try without GPU on GPU errors
            if use_gpu and "GPU" in str(e) and attempt < retry_count - 1:
                print("Retrying without GPU...")
                use_gpu = False
                continue
            
            # Wait before retry
            if attempt < retry_count - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
    
    print(f"Failed to analyze {symbol} after {retry_count} attempts")
    return None
```

## Performance Tips

1. **GPU Acceleration**: Always use GPU for multi-symbol analysis
2. **Batch Processing**: Analyze multiple symbols in parallel
3. **Caching**: Cache results for 30-60 seconds for identical requests
4. **Selective Indicators**: Request only needed indicators if supported
5. **Async Operations**: Use async/await for non-blocking analysis

### Performance Example:
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
import statistics

class HighPerformanceAnalyzer:
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.cache = {}
        self.cache_duration = 30  # seconds
    
    async def analyze_portfolio(self, symbols):
        """Analyze entire portfolio with maximum performance"""
        tasks = []
        
        for symbol in symbols:
            task = asyncio.create_task(self._analyze_with_cache(symbol))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Portfolio-level analysis
        portfolio_analysis = self._aggregate_portfolio_signals(results)
        return portfolio_analysis
    
    async def _analyze_with_cache(self, symbol):
        """Check cache before analyzing"""
        cache_key = f"{symbol}_{self.use_gpu}"
        
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_duration:
                return cached_data
        
        # Perform analysis
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            lambda: mcp__ai_news_trader__quick_analysis(
                symbol=symbol,
                use_gpu=self.use_gpu
            )
        )
        
        # Cache result
        self.cache[cache_key] = (result, time.time())
        return result
    
    def _aggregate_portfolio_signals(self, analyses):
        """Aggregate individual analyses into portfolio view"""
        
        # Calculate portfolio metrics
        avg_confidence = statistics.mean(
            [a['trading_signals']['confidence'] for a in analyses]
        )
        
        risk_scores = [a['risk_metrics']['risk_score'] for a in analyses]
        portfolio_risk = statistics.mean(risk_scores)
        
        # Determine overall signal
        signal_counts = {}
        for analysis in analyses:
            signal = analysis['trading_signals']['composite_signal']
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
        
        dominant_signal = max(signal_counts, key=signal_counts.get)
        
        return {
            'portfolio_signal': dominant_signal,
            'average_confidence': avg_confidence,
            'portfolio_risk_score': portfolio_risk,
            'individual_analyses': analyses,
            'signal_distribution': signal_counts,
            'timestamp': datetime.now().isoformat()
        }
```

## Related Tools
- `simulate_trade`: Test trading decisions based on analysis
- `get_strategy_info`: Get strategies that match analysis signals
- `analyze_news`: Combine with news sentiment for comprehensive view
- `run_backtest`: Validate analysis signals historically
- `execute_trade`: Execute trades based on analysis signals