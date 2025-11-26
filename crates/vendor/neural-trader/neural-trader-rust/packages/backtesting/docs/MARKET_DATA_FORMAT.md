# Market Data Format Documentation

## Overview

The `BacktestEngine.run()` method requires historical market data in a standardized CSV (Comma-Separated Values) format. This document specifies the exact format, column requirements, data validation rules, and provides examples for different asset types and timeframes.

**Key Features:**
- **CSV Format**: Industry-standard comma-separated format
- **OHLCV Data**: Open, High, Low, Close, Volume (standard candlestick data)
- **Flexible Timeframes**: Supports daily, intraday (minute/hourly), and tick data
- **Multi-Symbol**: Support for single or multiple symbols in one file
- **Validation**: Built-in validation with clear error messages
- **High Performance**: Efficient parsing optimized for large datasets (millions of bars)

---

## CSV Format Specification

### Header Row (Required)

The first line of the CSV file MUST contain column headers:

```
timestamp,open,high,low,close,volume
```

**Column Names:**
- `timestamp` - Market data timestamp (required)
- `open` - Opening price (required)
- `high` - Highest price in period (required)
- `low` - Lowest price in period (required)
- `close` - Closing price (required)
- `volume` - Trading volume in shares (required)

**Optional Columns:**
- `symbol` - Stock ticker symbol (optional, useful for multi-symbol data)
- `bid` - Bid price (optional, for additional analysis)
- `ask` - Ask price (optional, for additional analysis)

### Data Row Format

Each subsequent row contains one bar of market data:

```
timestamp,open,high,low,close,volume
2024-01-01 09:30:00,150.00,151.50,149.75,151.00,1000000
2024-01-01 09:31:00,151.00,152.00,150.50,151.75,950000
2024-01-01 09:32:00,151.75,152.50,151.00,151.50,1100000
```

### Timestamp Formats

The engine accepts multiple timestamp formats:

**1. ISO 8601 Format (Recommended)**
```
2024-01-01T09:30:00Z
2024-01-01T09:30:00+00:00
2024-01-01 09:30:00
```

**2. Date Only (Daily Data)**
```
2024-01-01
2024/01/01
01/01/2024
```

**3. Unix Timestamp (Seconds)**
```
1704110400
```

**4. Unix Timestamp (Milliseconds)**
```
1704110400000
```

**5. Unix Timestamp (Nanoseconds)**
```
1704110400000000000
```

### Data Types and Ranges

| Column | Type | Range | Example | Notes |
|--------|------|-------|---------|-------|
| timestamp | String/Number | Any valid datetime | `2024-01-01 09:30:00` | Must be in chronological order |
| open | Float | 0 < value | 150.00 | Must be positive, non-zero |
| high | Float | high >= max(open, close) | 151.50 | Must be >= open and close |
| low | Float | low <= min(open, close) | 149.75 | Must be <= open and close |
| close | Float | 0 < value | 151.00 | Must be positive, non-zero |
| volume | Integer | 0 <= value | 1000000 | Non-negative, can be 0 |

### Validation Rules

The engine enforces these validation rules:

1. **High-Low Relationship**: `low <= open, close <= high`
   - High must be >= all prices in the bar
   - Low must be <= all prices in the bar

2. **Chronological Order**: Timestamps must be in ascending order
   - Each timestamp must be > previous timestamp
   - Duplicate timestamps are rejected

3. **Positive Prices**: Open, high, low, close must be > 0
   - Negative or zero prices are invalid
   - Exception: volume can be 0

4. **Data Continuity**:
   - First row's timestamp must match engine startDate
   - Last row's timestamp must match engine endDate
   - Missing data periods trigger warnings

5. **Numeric Format**:
   - Prices must be valid floating-point numbers
   - Volume must be a valid integer
   - Scientific notation is accepted (1e6 = 1,000,000)

---

## Examples by Asset Type

### Example 1: Daily Stock Data

**File: `AAPL-daily-2024.csv`**

```csv
timestamp,open,high,low,close,volume
2024-01-01,186.00,187.50,185.50,187.00,50000000
2024-01-02,187.00,188.25,186.75,187.50,45000000
2024-01-03,187.50,189.00,187.00,188.50,55000000
2024-01-04,188.50,189.50,188.00,189.00,48000000
2024-01-05,189.00,190.00,188.75,189.75,52000000
```

**Characteristics:**
- One bar per trading day
- Large round volumes (millions of shares)
- Timestamps are dates only
- Typical for long-term strategies

### Example 2: Intraday Minute Data

**File: `MSFT-intraday-2024-01-01.csv`**

```csv
timestamp,open,high,low,close,volume
2024-01-01 09:30:00,420.00,420.50,419.75,420.25,1500000
2024-01-01 09:31:00,420.25,420.75,420.00,420.50,1200000
2024-01-01 09:32:00,420.50,421.00,420.25,420.75,1400000
2024-01-01 09:33:00,420.75,421.50,420.50,421.00,1300000
2024-01-01 09:34:00,421.00,421.75,420.75,421.50,1600000
2024-01-01 09:35:00,421.50,422.00,421.25,421.75,1450000
```

**Characteristics:**
- One bar per minute (or configurable interval)
- Full timestamp with time component
- Smaller volumes (thousands per minute)
- Typical for short-term day trading strategies

### Example 3: Hourly Cryptocurrency Data

**File: `BTC-hourly-2024.csv`**

```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,42500.00,43100.00,42300.00,42800.00,1500
2024-01-01 01:00:00,42800.00,43200.00,42700.00,43050.00,1600
2024-01-01 02:00:00,43050.00,43500.00,42950.00,43300.00,1750
2024-01-01 03:00:00,43300.00,43600.00,43100.00,43400.00,1620
2024-01-01 04:00:00,43400.00,43800.00,43250.00,43550.00,1700
```

**Characteristics:**
- One bar per hour (24x7 trading)
- Hourly OHLCV data
- Typical for cryptocurrency day trading
- Can run 24/7 unlike stock markets

### Example 4: Multi-Symbol Portfolio Data

**File: `portfolio-daily-2024.csv`**

```csv
timestamp,symbol,open,high,low,close,volume
2024-01-01,AAPL,186.00,187.50,185.50,187.00,50000000
2024-01-01,MSFT,420.00,421.50,419.50,420.50,30000000
2024-01-01,GOOGL,139.00,140.50,138.75,140.00,35000000
2024-01-01,AMZN,180.00,181.00,179.50,180.50,55000000
2024-01-02,AAPL,187.00,188.25,186.75,187.50,45000000
2024-01-02,MSFT,420.50,422.00,420.00,421.00,28000000
2024-01-02,GOOGL,140.00,141.00,139.50,140.75,32000000
2024-01-02,AMZN,180.50,181.50,180.00,181.00,52000000
```

**Characteristics:**
- Multiple symbols in single file
- `symbol` column identifies asset
- All symbols must have same timestamps
- Enables portfolio-level backtesting

### Example 5: Futures Data with Tick Information

**File: `ES-5min-2024.csv`**

```csv
timestamp,symbol,open,high,low,close,volume,bid,ask
2024-01-01 09:30:00,ES,5000.00,5010.00,4995.00,5008.00,500000,5007.75,5008.25
2024-01-01 09:35:00,ES,5008.00,5015.00,5005.00,5012.00,520000,5011.75,5012.25
2024-01-01 09:40:00,ES,5012.00,5020.00,5010.00,5018.00,510000,5017.75,5018.25
```

**Characteristics:**
- 5-minute bars (standard for futures)
- Optional bid/ask data
- Large volume (100s of thousands)
- Typical for day trading S&P 500 E-mini futures

---

## Timeframe Support

The engine supports any regular timeframe. Common timeframes:

| Timeframe | Typical Use | Data Points/Year |
|-----------|------------|------------------|
| 1-minute | High-frequency day trading | ~252,000 |
| 5-minute | Active day trading | ~50,000 |
| 15-minute | Day trading, swing trading | ~16,000 |
| Hourly | Swing trading, position trading | ~8,760 |
| Daily | Position trading, medium-term | ~252 |
| Weekly | Long-term investing | ~52 |
| Monthly | Very long-term analysis | ~12 |

**Important**: The engine does NOT require regular intervals. You can mix:
- Different trading days with different numbers of bars
- Market gaps (weekends, holidays)
- Extended hours trading (pre-market, after-hours)

---

## Data Validation Examples

### Valid Data

```csv
timestamp,open,high,low,close,volume
2024-01-01 09:30:00,150.00,151.50,149.75,151.00,1000000
2024-01-01 09:31:00,151.00,152.00,150.50,151.75,950000
2024-01-01 09:32:00,151.75,152.50,151.00,151.50,1100000
```

✅ All validations pass:
- High >= max(open, close)
- Low <= min(open, close)
- All prices positive
- Chronological order
- Consistent columns

### Invalid Data Examples

**❌ Example 1: High-Low Violation**
```csv
timestamp,open,high,low,close,volume
2024-01-01,150.00,151.00,152.00,151.00,1000000
```
Error: `low (152.00) > high (151.00)` - Invalid

**❌ Example 2: Negative Price**
```csv
timestamp,open,high,low,close,volume
2024-01-01,150.00,151.50,-1.00,151.00,1000000
```
Error: `low (-1.00) must be positive` - Invalid

**❌ Example 3: Out of Order**
```csv
timestamp,open,high,low,close,volume
2024-01-02,150.00,151.50,149.75,151.00,1000000
2024-01-01,151.00,152.00,150.50,151.75,950000
```
Error: `timestamp not in chronological order` - Invalid

**❌ Example 4: Missing Header**
```csv
2024-01-01,150.00,151.50,149.75,151.00,1000000
```
Error: `Invalid CSV header - expected: timestamp,open,high,low,close,volume` - Invalid

**❌ Example 5: Missing Column**
```csv
timestamp,open,high,low,volume
2024-01-01,150.00,151.50,149.75,1000000
```
Error: `Missing required column: close` - Invalid

---

## Data Sources

### Recommended Free Data Sources

1. **Yahoo Finance**
   - Daily OHLCV data for stocks, ETFs, crypto
   - Via `yfinance` Python library
   - Example: `yfinance.download('AAPL', start='2024-01-01', end='2024-12-31')`

2. **Alpha Vantage**
   - Intraday and daily data
   - Free tier available (500 requests/day)
   - API: https://www.alphavantage.co

3. **IEX Cloud**
   - Real-time and historical market data
   - Free tier available
   - API: https://iexcloud.io

4. **Polygon.io**
   - Comprehensive historical data
   - Free tier available
   - API: https://polygon.io

5. **CoinGecko (Cryptocurrency)**
   - Free historical cryptocurrency data
   - Minute to daily resolution
   - API: https://www.coingecko.com/api

### Commercial Data Providers

- **Bloomberg Terminal** - Professional grade data
- **FactSet** - Comprehensive market data
- **Refinitiv** - Professional-grade feeds
- **Interactive Brokers** - Historical data API

---

## CSV Generation Tools

### Python Example

```python
import pandas as pd
import yfinance as yf

# Download daily data
data = yf.download('AAPL', start='2024-01-01', end='2024-12-31')

# Format for backtesting engine
data_formatted = data.reset_index()
data_formatted.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
data_formatted['timestamp'] = data_formatted['timestamp'].dt.strftime('%Y-%m-%d')

# Save to CSV
data_formatted.to_csv('AAPL-daily-2024.csv', index=False)

print(f"Generated {len(data_formatted)} bars of data")
```

### Node.js Example

```javascript
const axios = require('axios');
const fs = require('fs');

async function fetchData() {
  const response = await axios.get('https://api.example.com/historical', {
    params: {
      symbol: 'AAPL',
      from: '2024-01-01',
      to: '2024-12-31'
    }
  });

  const csv = 'timestamp,open,high,low,close,volume\n' +
    response.data.map(bar =>
      `${bar.timestamp},${bar.open},${bar.high},${bar.low},${bar.close},${bar.volume}`
    ).join('\n');

  fs.writeFileSync('AAPL-daily-2024.csv', csv);
  console.log(`Generated ${response.data.length} bars of data`);
}

fetchData().catch(console.error);
```

---

## Performance Considerations

### File Size Impact

| Data Size | Processing Time | Memory | Typical Period |
|-----------|-----------------|--------|-----------------|
| 1 MB | ~5ms | 2 MB | 1 month daily + 1 year intraday |
| 10 MB | ~40ms | 15 MB | 1 year daily + 5 years minute |
| 100 MB | ~400ms | 150 MB | 10 years daily + 50 years minute |
| 1 GB | ~4 seconds | 1.5 GB | Very large datasets (parallel processing) |

### Optimization Tips

1. **Use Appropriate Timeframe**
   - Daily data for long-term strategies
   - Minute/hourly for short-term trading
   - Don't use unnecessary high-frequency data

2. **Compress Data**
   - GZIP compression: 80% reduction
   - Parquet format: 90% reduction
   - Archive old data separately

3. **Filter Time Periods**
   - Only include trading hours
   - Exclude weekends/holidays
   - Use relevant date ranges

4. **Parallel Processing**
   - Process multiple symbols concurrently
   - Use walk-forward analysis
   - Leverage multi-threading (set `NT_THREADS` environment variable)

---

## Common Issues and Solutions

### Issue 1: "Invalid CSV Format" Error

**Cause**: Column headers don't match exactly

**Solution**: Verify exact column names:
```csv
timestamp,open,high,low,close,volume
```

**Not acceptable:**
```csv
Timestamp,Open,High,Low,Close,Volume  # Case mismatch
date,o,h,l,c,vol  # Wrong names
```

### Issue 2: "Timestamp Out of Range" Error

**Cause**: Signal or data timestamps don't align

**Solution**: Ensure data covers backtest period
```javascript
// ✅ Correct
const engine = new BacktestEngine({
  startDate: '2024-01-01',
  endDate: '2024-12-31'
});
// CSV data must cover full date range

// ❌ Wrong
const engine = new BacktestEngine({
  startDate: '2024-01-01',
  endDate: '2024-12-31'
});
// CSV data only covers 2024-06-01 to 2024-12-31
```

### Issue 3: "High-Low Violation" Error

**Cause**: Bar data is invalid

**Solution**: Verify OHLC relationships
```csv
# ❌ Invalid: high < low
2024-01-01,150,151,152,151,1000000

# ✅ Valid: low <= min(o,c) and high >= max(o,c)
2024-01-01,150,152,149,151,1000000
```

### Issue 4: Large File Performance

**Cause**: File is very large (>500MB)

**Solution**:
1. Split into smaller periods
2. Use higher timeframe (hourly instead of 1-minute)
3. Enable parallel processing with environment variable
```bash
export NT_THREADS=8
node backtest.js
```

### Issue 5: Missing Data Points

**Cause**: Market gaps (weekends, holidays, halts)

**Solution**: This is normal and expected
```csv
2024-01-03,150,152,149,151,1000000
2024-01-04,151,153,150,152,1050000
2024-01-05,152,154,151,153,1100000
# January 6-7 = weekend (no trading)
2024-01-08,153,155,152,154,1150000
```

---

## Migration from Other Formats

### From Excel (.xlsx)

```python
import pandas as pd

# Read Excel
df = pd.read_excel('market_data.xlsx', sheet_name=0)

# Format columns
df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

# Sort by timestamp
df = df.sort_values('timestamp')

# Save as CSV
df.to_csv('market_data.csv', index=False)
```

### From JSON

```javascript
const fs = require('fs');

// Read JSON
const data = JSON.parse(fs.readFileSync('market_data.json', 'utf8'));

// Convert to CSV
const csv = 'timestamp,open,high,low,close,volume\n' +
  data.map(bar =>
    `${bar.timestamp},${bar.open},${bar.high},${bar.low},${bar.close},${bar.volume}`
  ).join('\n');

fs.writeFileSync('market_data.csv', csv);
```

### From Database (SQL)

```sql
-- Export from PostgreSQL
COPY (
  SELECT
    timestamp,
    open,
    high,
    low,
    close,
    volume
  FROM market_data
  WHERE symbol = 'AAPL'
  ORDER BY timestamp
) TO STDOUT WITH CSV HEADER;

-- Then save to file
-- psql -d database -c "COPY ... TO STDOUT WITH CSV HEADER" > market_data.csv
```

---

## API Reference

### CSV Parameter in BacktestEngine.run()

```typescript
// Method signature
run(signals: Signal[], marketData: string): Promise<BacktestResult>

// Parameter: marketData: string
// Can be either:
// 1. File path (string)
const result = await engine.run(signals, 'data/AAPL-2024.csv');

// 2. Raw CSV content (string)
const csvContent = `timestamp,open,high,low,close,volume
2024-01-01,150,152,149,151,1000000`;
const result = await engine.run(signals, csvContent);
```

---

## Best Practices

### ✅ DO:

1. **Use consistent timestamp format**
   - ISO 8601 throughout entire file
   - All timestamps in UTC timezone

2. **Validate data before backtesting**
   - Run validation script first
   - Check for data gaps
   - Verify OHLC relationships

3. **Include sufficient data history**
   - For momentum strategies: +100 bars before signal
   - For mean reversion: +250 bars (1 trading year)
   - For AI models: +1000+ bars if training

4. **Document data source**
   - Include data provider
   - Note any adjustments (splits, dividends)
   - Record download date

5. **Version control data**
   - Track data modifications
   - Keep raw and processed versions
   - Enable reproducibility

### ❌ DON'T:

1. **Mix different timeframes in one file**
   - Use 1 file per timeframe
   - Don't mix daily + hourly data

2. **Use unrealistic prices**
   - Avoid penny stock anomalies
   - Validate against data provider
   - Check for obvious outliers

3. **Include dividends/splits unadjusted**
   - Use split/dividend adjusted prices
   - Document any adjustments
   - Use consistent methodology

4. **Leave gaps unexplained**
   - Document market holidays
   - Note trading halts
   - Explain extended hours inclusion

5. **Over-optimize backtest period**
   - Use multiple time periods
   - Validate out-of-sample
   - Test parameter robustness

---

## Example Files Included

This package includes several example CSV files in `examples/data/`:

1. **AAPL-daily-2024.csv** - Apple daily stock data (entire year)
2. **MSFT-intraday-2024-01-01.csv** - Microsoft minute data (1 day)
3. **BTC-hourly-2024.csv** - Bitcoin hourly cryptocurrency data (1 month)
4. **portfolio-daily-2024.csv** - Multi-symbol portfolio data
5. **AAPL-validation-test.csv** - Test data with edge cases

All example files are ready to use for backtesting and can serve as templates for your own data.

---

## See Also

- [README.md](../README.md) - Main package documentation
- [Troubleshooting Guide](../README.md#troubleshooting) - Common issues and solutions
- [Performance Guide](../README.md#performance-tips) - Optimization techniques
- [Integration Examples](../README.md#integration-examples) - Usage with other packages

---

## Support

For questions about market data format:

1. Check this documentation first
2. Review example files in `examples/data/`
3. Run validation script: `node examples/validate-data.js`
4. Open GitHub issue with error and sample data
5. Contact support@neural-trader.io

---

**Last Updated:** 2024-11-17
**Version:** 1.0.0
**Status:** Production Ready
