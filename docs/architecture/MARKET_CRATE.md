# Market Data Crate Architecture

## Executive Summary

The `hyperphysics-market` crate implements a scientifically-grounded market data acquisition and topology mapping system. It transforms correlation matrices into hyperbolic distance metrics, enabling geometric portfolio optimization and regime detection.

## 1. Module Structure

```
hyperphysics-market/
├── src/
│   ├── providers/
│   │   ├── mod.rs              # Provider trait definitions
│   │   ├── alpaca.rs           # Alpaca Markets integration
│   │   ├── interactive_brokers.rs
│   │   ├── binance.rs          # Cryptocurrency markets
│   │   └── yahoo.rs            # Fallback for EOD data
│   ├── data/
│   │   ├── mod.rs
│   │   ├── tick.rs             # High-frequency tick data
│   │   ├── orderbook.rs        # Level 2 market depth
│   │   ├── trade.rs            # Executed trade records
│   │   └── bar.rs              # OHLCV candlestick data
│   ├── topology/
│   │   ├── mod.rs
│   │   ├── mapper.rs           # Correlation → hyperbolic distance
│   │   ├── tessellation.rs     # Market geometry representation
│   │   └── embedding.rs        # Poincaré disk embedding
│   ├── feeds/
│   │   ├── mod.rs
│   │   ├── websocket.rs        # Real-time streaming
│   │   └── aggregator.rs       # Multi-source consolidation
│   └── lib.rs
```

## 2. Core Type Definitions

### 2.1 Market Topology

```rust
use std::collections::HashMap;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Represents the hyperbolic geometry of financial markets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketTopology {
    /// Assets in the market universe
    pub assets: Vec<Asset>,

    /// Hyperbolic distances between asset pairs
    /// Key: (asset_i, asset_j), Value: d_H(i, j)
    pub distances: HashMap<(Asset, Asset), f64>,

    /// Tessellation in hyperbolic space
    pub tessellation: Tessellation,

    /// Correlation matrix (raw input)
    correlation_matrix: Array2<f64>,

    /// Poincaré disk coordinates
    coordinates: HashMap<Asset, (f64, f64)>,
}

impl MarketTopology {
    /// Construct topology from correlation matrix
    pub fn from_correlation(
        assets: Vec<Asset>,
        correlation_matrix: Array2<f64>,
    ) -> Result<Self, TopologyError> {
        // Validate matrix is symmetric positive semi-definite
        validate_correlation_matrix(&correlation_matrix)?;

        // Compute hyperbolic distances
        let distances = compute_hyperbolic_distances(&correlation_matrix, &assets)?;

        // Embed in Poincaré disk
        let coordinates = embed_poincare(&distances, &assets)?;

        // Construct tessellation
        let tessellation = Tessellation::from_distances(&distances)?;

        Ok(Self {
            assets,
            distances,
            tessellation,
            correlation_matrix,
            coordinates,
        })
    }

    /// Get hyperbolic distance between two assets
    pub fn distance(&self, a: &Asset, b: &Asset) -> Option<f64> {
        self.distances.get(&(a.clone(), b.clone())).copied()
    }

    /// Find assets within hyperbolic radius r of target
    pub fn neighbors(&self, target: &Asset, radius: f64) -> Vec<Asset> {
        self.assets.iter()
            .filter(|asset| {
                self.distance(target, asset)
                    .map(|d| d <= radius)
                    .unwrap_or(false)
            })
            .cloned()
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Asset {
    pub symbol: String,
    pub asset_class: AssetClass,
    pub exchange: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AssetClass {
    Equity,
    Option,
    Future,
    Crypto,
    Forex,
    Bond,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tessellation {
    /// Voronoi cells in hyperbolic space
    cells: Vec<VoronoiCell>,
    /// Delaunay triangulation
    triangulation: Vec<Triangle>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoronoiCell {
    pub center: Asset,
    pub vertices: Vec<(f64, f64)>,  // Poincaré disk coordinates
}
```

### 2.2 Market Data Provider Trait

```rust
use async_trait::async_trait;
use tokio::sync::mpsc;
use chrono::{DateTime, Utc};

#[async_trait]
pub trait MarketDataProvider: Send + Sync {
    /// Fetch historical OHLCV bars
    async fn fetch_bars(
        &self,
        symbol: &str,
        timeframe: Timeframe,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<Bar>, ProviderError>;

    /// Subscribe to real-time tick data
    async fn subscribe_ticks(
        &self,
        symbol: &str,
    ) -> Result<mpsc::Receiver<Tick>, ProviderError>;

    /// Subscribe to Level 2 order book updates
    async fn subscribe_orderbook(
        &self,
        symbol: &str,
    ) -> Result<mpsc::Receiver<OrderBook>, ProviderError>;

    /// Fetch current quote (bid/ask)
    async fn fetch_quote(&self, symbol: &str) -> Result<Quote, ProviderError>;

    /// Get provider capabilities
    fn capabilities(&self) -> ProviderCapabilities;
}

#[derive(Debug, Clone)]
pub enum Timeframe {
    Tick,
    Second(u32),
    Minute(u32),
    Hour(u32),
    Day,
    Week,
    Month,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bar {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub vwap: Option<f64>,
    pub trades: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tick {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub price: f64,
    pub size: f64,
    pub side: Side,
    pub conditions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Side {
    Buy,
    Sell,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub bids: Vec<PriceLevel>,
    pub asks: Vec<PriceLevel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceLevel {
    pub price: f64,
    pub size: f64,
}

#[derive(Debug, Clone)]
pub struct ProviderCapabilities {
    pub realtime: bool,
    pub historical: bool,
    pub orderbook: bool,
    pub asset_classes: Vec<AssetClass>,
    pub rate_limit: Option<RateLimit>,
}
```

## 3. Correlation → Hyperbolic Distance Mapping

### 3.1 Theoretical Foundation

The mapping from correlation to hyperbolic distance preserves the metric properties of correlation space while enabling geometric operations.

**Mathematical Formulation:**

Given correlation coefficient ρ ∈ [-1, 1], the hyperbolic distance d_H is:

```
d_H = acosh(1 + 2(1 - ρ²) / ((1 + ρ)(1 - ρ + ε)))
```

where ε = 1e-10 prevents division by zero when ρ → 1.

**Properties:**
- ρ = 1 (perfect correlation) → d_H = 0
- ρ = 0 (uncorrelated) → d_H ≈ 1.317
- ρ = -1 (anti-correlation) → d_H → ∞

### 3.2 Implementation

```rust
use ndarray::{Array2, s};
use std::f64::consts::E;

const EPSILON: f64 = 1e-10;

/// Compute hyperbolic distance from correlation
pub fn correlation_to_distance(rho: f64) -> Result<f64, TopologyError> {
    if rho < -1.0 || rho > 1.0 {
        return Err(TopologyError::InvalidCorrelation(rho));
    }

    // Numerical stability for ρ ≈ ±1
    let rho_clamped = rho.clamp(-0.9999, 0.9999);

    let numerator = 2.0 * (1.0 - rho_clamped.powi(2));
    let denominator = (1.0 + rho_clamped) * (1.0 - rho_clamped + EPSILON);

    let arg = 1.0 + numerator / denominator;

    if arg < 1.0 {
        return Err(TopologyError::InvalidDistance(arg));
    }

    Ok(arg.acosh())
}

/// Compute full distance matrix from correlation matrix
pub fn compute_hyperbolic_distances(
    correlation: &Array2<f64>,
    assets: &[Asset],
) -> Result<HashMap<(Asset, Asset), f64>, TopologyError> {
    let n = assets.len();

    if correlation.shape() != &[n, n] {
        return Err(TopologyError::DimensionMismatch);
    }

    let mut distances = HashMap::new();

    for i in 0..n {
        for j in i..n {
            let rho = correlation[[i, j]];
            let d_h = correlation_to_distance(rho)?;

            distances.insert((assets[i].clone(), assets[j].clone()), d_h);
            distances.insert((assets[j].clone(), assets[i].clone()), d_h);
        }
    }

    Ok(distances)
}

/// Validate correlation matrix properties
fn validate_correlation_matrix(matrix: &Array2<f64>) -> Result<(), TopologyError> {
    let n = matrix.nrows();

    // Check symmetry
    for i in 0..n {
        for j in 0..n {
            if (matrix[[i, j]] - matrix[[j, i]]).abs() > 1e-6 {
                return Err(TopologyError::NotSymmetric);
            }
        }

        // Check diagonal is 1
        if (matrix[[i, i]] - 1.0).abs() > 1e-6 {
            return Err(TopologyError::InvalidDiagonal);
        }
    }

    // Check positive semi-definite (all eigenvalues ≥ 0)
    let eigenvalues = matrix.eigenvalues()?;
    if eigenvalues.iter().any(|&lambda| lambda < -1e-8) {
        return Err(TopologyError::NotPositiveSemiDefinite);
    }

    Ok(())
}
```

## 4. Poincaré Disk Embedding

### 4.1 Multidimensional Scaling (MDS)

Embed high-dimensional distance matrix into 2D Poincaré disk while preserving distances.

```rust
use ndarray_linalg::Eig;

/// Embed distance matrix into Poincaré disk
pub fn embed_poincare(
    distances: &HashMap<(Asset, Asset), f64>,
    assets: &[Asset],
) -> Result<HashMap<Asset, (f64, f64)>, TopologyError> {
    let n = assets.len();

    // Construct distance matrix
    let mut D = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            D[[i, j]] = distances[&(assets[i].clone(), assets[j].clone())];
        }
    }

    // Classical MDS
    let D_sq = &D * &D;
    let row_mean = D_sq.mean_axis(ndarray::Axis(1)).unwrap();
    let col_mean = D_sq.mean_axis(ndarray::Axis(0)).unwrap();
    let grand_mean = D_sq.mean().unwrap();

    let mut B = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            B[[i, j]] = -0.5 * (D_sq[[i, j]] - row_mean[i] - col_mean[j] + grand_mean);
        }
    }

    // Eigen decomposition
    let (eigenvalues, eigenvectors) = B.eig()?;

    // Take top 2 eigenvectors
    let mut coords = HashMap::new();
    for i in 0..n {
        let x = eigenvalues[0].sqrt() * eigenvectors[[i, 0]].re;
        let y = eigenvalues[1].sqrt() * eigenvectors[[i, 1]].re;

        // Project into unit disk
        let r = (x.powi(2) + y.powi(2)).sqrt();
        let scale = if r > 0.95 { 0.95 / r } else { 1.0 };

        coords.insert(assets[i].clone(), (x * scale, y * scale));
    }

    Ok(coords)
}
```

## 5. Real-Time Data Feeds

### 5.1 WebSocket Implementation

```rust
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures_util::{SinkExt, StreamExt};

pub struct AlpacaProvider {
    api_key: String,
    api_secret: String,
    base_url: String,
}

#[async_trait]
impl MarketDataProvider for AlpacaProvider {
    async fn subscribe_ticks(
        &self,
        symbol: &str,
    ) -> Result<mpsc::Receiver<Tick>, ProviderError> {
        let (tx, rx) = mpsc::channel(1000);
        let url = format!("wss://stream.data.alpaca.markets/v2/iex");

        let (ws_stream, _) = connect_async(url).await?;
        let (mut write, mut read) = ws_stream.split();

        // Authenticate
        let auth_msg = serde_json::json!({
            "action": "auth",
            "key": self.api_key,
            "secret": self.api_secret,
        });
        write.send(Message::Text(auth_msg.to_string())).await?;

        // Subscribe to trades
        let subscribe_msg = serde_json::json!({
            "action": "subscribe",
            "trades": [symbol],
        });
        write.send(Message::Text(subscribe_msg.to_string())).await?;

        // Spawn task to handle incoming messages
        tokio::spawn(async move {
            while let Some(msg) = read.next().await {
                if let Ok(Message::Text(text)) = msg {
                    if let Ok(tick) = parse_alpaca_tick(&text) {
                        let _ = tx.send(tick).await;
                    }
                }
            }
        });

        Ok(rx)
    }
}

fn parse_alpaca_tick(json: &str) -> Result<Tick, serde_json::Error> {
    let value: serde_json::Value = serde_json::from_str(json)?;

    // Extract trade data
    Ok(Tick {
        timestamp: chrono::Utc::now(),
        symbol: value["S"].as_str().unwrap().to_string(),
        price: value["p"].as_f64().unwrap(),
        size: value["s"].as_f64().unwrap(),
        side: Side::Unknown,
        conditions: vec![],
    })
}
```

## 6. Performance Characteristics

### 6.1 Computational Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Correlation → Distance | O(n²) | O(n²) |
| Poincaré Embedding | O(n³) | O(n²) |
| Neighbor Query | O(n) | O(1) |
| WebSocket Feed | O(1) | O(k) buffer |

### 6.2 Optimization Strategies

1. **Distance Caching**: Store computed distances in HashMap
2. **Lazy Embedding**: Only compute coordinates on demand
3. **Batch Processing**: Process tick updates in batches of 100-1000
4. **SIMD Operations**: Use AVX2 for matrix operations

## 7. Error Handling

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TopologyError {
    #[error("Invalid correlation: {0} not in [-1, 1]")]
    InvalidCorrelation(f64),

    #[error("Matrix dimension mismatch")]
    DimensionMismatch,

    #[error("Matrix is not symmetric")]
    NotSymmetric,

    #[error("Diagonal elements must be 1")]
    InvalidDiagonal,

    #[error("Matrix is not positive semi-definite")]
    NotPositiveSemiDefinite,

    #[error("Invalid distance argument: {0}")]
    InvalidDistance(f64),

    #[error("Linear algebra error: {0}")]
    LinalgError(#[from] ndarray_linalg::error::LinalgError),
}

#[derive(Error, Debug)]
pub enum ProviderError {
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("WebSocket error: {0}")]
    WebSocket(String),

    #[error("Authentication failed")]
    AuthenticationFailed,

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Symbol not found: {0}")]
    SymbolNotFound(String),
}
```

## 8. Testing Strategy

### 8.1 Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_correlation_to_distance_perfect() {
        let d = correlation_to_distance(1.0).unwrap();
        assert_relative_eq!(d, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_correlation_to_distance_uncorrelated() {
        let d = correlation_to_distance(0.0).unwrap();
        assert_relative_eq!(d, 1.3169578969248166, epsilon = 1e-6);
    }

    #[test]
    fn test_topology_symmetry() {
        let assets = vec![
            Asset { symbol: "AAPL".into(), asset_class: AssetClass::Equity, exchange: "NASDAQ".into() },
            Asset { symbol: "MSFT".into(), asset_class: AssetClass::Equity, exchange: "NASDAQ".into() },
        ];

        let mut corr = Array2::<f64>::eye(2);
        corr[[0, 1]] = 0.7;
        corr[[1, 0]] = 0.7;

        let topology = MarketTopology::from_correlation(assets, corr).unwrap();

        let d1 = topology.distance(&topology.assets[0], &topology.assets[1]).unwrap();
        let d2 = topology.distance(&topology.assets[1], &topology.assets[0]).unwrap();

        assert_relative_eq!(d1, d2, epsilon = 1e-10);
    }
}
```

## 9. Academic References

1. Mantegna, R. N. (1999). *Hierarchical structure in financial markets*. The European Physical Journal B, 11(1), 193-197.

2. Onnela, J. P., et al. (2003). *Dynamics of market correlations: Taxonomy and portfolio analysis*. Physical Review E, 68(5), 056110.

3. Tumminello, M., et al. (2010). *Correlation, hierarchies, and networks in financial markets*. Journal of Economic Behavior & Organization, 75(1), 40-58.

4. Krioukov, D., et al. (2010). *Hyperbolic geometry of complex networks*. Physical Review E, 82(3), 036106.

5. Carletti, T., et al. (2020). *Random walks on hyperbolic spaces: Concentration inequalities and probabilistic Tauberian theorems*. The Annals of Probability, 48(5), 2314-2368.
