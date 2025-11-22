//! Market data aggregator implementation

use crate::prelude::*;
use crate::models::MarketData;
use chrono::{DateTime, Duration, Utc};
use rust_decimal::Decimal;
use num_traits::ToPrimitive;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Market data aggregator that combines and processes data from multiple sources
#[derive(Debug)]
pub struct DataAggregator {
    /// Aggregator configuration
    config: DataAggregatorConfig,
    
    /// Raw data buffers for each source and symbol
    raw_data_buffers: Arc<RwLock<HashMap<String, HashMap<String, VecDeque<MarketData>>>>>,
    
    /// Aggregated data cache
    aggregated_cache: Arc<RwLock<HashMap<String, AggregatedData>>>,
    
    /// Data source weights and priorities
    source_weights: Arc<RwLock<HashMap<String, SourceWeight>>>,
    
    /// Aggregation metrics
    metrics: Arc<RwLock<AggregationMetrics>>,
}

#[derive(Debug, Clone)]
pub struct DataAggregatorConfig {
    /// Maximum age of data to consider in aggregation
    pub max_data_age_seconds: u64,
    
    /// Minimum number of sources required for aggregation
    pub min_sources: usize,
    
    /// Aggregation methods by data type
    pub aggregation_methods: AggregationMethods,
    
    /// Outlier detection settings
    pub outlier_detection: OutlierDetectionConfig,
    
    /// Buffer sizes for each source
    pub buffer_size: usize,
    
    /// Update frequency for aggregation
    pub update_frequency_ms: u64,
}

#[derive(Debug, Clone)]
pub struct AggregationMethods {
    pub price_method: PriceAggregationMethod,
    pub volume_method: VolumeAggregationMethod,
    pub spread_method: SpreadAggregationMethod,
}

#[derive(Debug, Clone)]
pub enum PriceAggregationMethod {
    WeightedAverage,
    MedianPrice,
    VolumeWeightedAverage,
    HighestLiquidity,
}

#[derive(Debug, Clone)]
pub enum VolumeAggregationMethod {
    Sum,
    WeightedSum,
    Average,
    Maximum,
}

#[derive(Debug, Clone)]
pub enum SpreadAggregationMethod {
    TightestSpread,
    WeightedAverage,
    VolumeWeighted,
}

#[derive(Debug, Clone)]
pub struct OutlierDetectionConfig {
    pub enabled: bool,
    pub price_deviation_threshold: f64,
    pub volume_deviation_threshold: f64,
    pub z_score_threshold: f64,
}

#[derive(Debug, Clone)]
struct SourceWeight {
    reliability_score: f64,
    latency_score: f64,
    volume_score: f64,
    overall_weight: f64,
}

#[derive(Debug, Clone)]
struct AggregatedData {
    symbol: String,
    timestamp: DateTime<Utc>,
    aggregated_price: AggregatedPrice,
    aggregated_volume: AggregatedVolume,
    data_quality: DataQuality,
    source_count: usize,
}

#[derive(Debug, Clone)]
struct AggregatedPrice {
    bid: Decimal,
    ask: Decimal,
    mid: Decimal,
    last: Decimal,
    confidence_score: f64,
}

#[derive(Debug, Clone)]
struct AggregatedVolume {
    total_volume: Decimal,
    weighted_volume: Decimal,
    volume_confidence: f64,
}

#[derive(Debug, Clone)]
struct DataQuality {
    freshness_score: f64,
    consistency_score: f64,
    coverage_score: f64,
    overall_score: f64,
}

#[derive(Debug, Clone, Default)]
struct AggregationMetrics {
    total_aggregations: u64,
    successful_aggregations: u64,
    failed_aggregations: u64,
    average_sources_per_aggregation: f64,
    average_confidence_score: f64,
    outliers_detected: u64,
}

impl Default for DataAggregatorConfig {
    fn default() -> Self {
        Self {
            max_data_age_seconds: 30,
            min_sources: 2,
            aggregation_methods: AggregationMethods {
                price_method: PriceAggregationMethod::VolumeWeightedAverage,
                volume_method: VolumeAggregationMethod::WeightedSum,
                spread_method: SpreadAggregationMethod::TightestSpread,
            },
            outlier_detection: OutlierDetectionConfig {
                enabled: true,
                price_deviation_threshold: 0.05,
                volume_deviation_threshold: 3.0,
                z_score_threshold: 2.5,
            },
            buffer_size: 1000,
            update_frequency_ms: 500,
        }
    }
}

impl DataAggregator {
    /// Create a new data aggregator
    pub fn new(config: DataAggregatorConfig) -> Self {
        Self {
            config,
            raw_data_buffers: Arc::new(RwLock::new(HashMap::new())),
            aggregated_cache: Arc::new(RwLock::new(HashMap::new())),
            source_weights: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(AggregationMetrics::default())),
        }
    }

    /// Add market data from a specific source
    pub async fn add_data(&self, source_id: &str, market_data: MarketData) -> Result<()> {
        let mut buffers = self.raw_data_buffers.write().await;
        
        // Initialize source buffer if it doesn't exist
        buffers.entry(source_id.to_string()).or_insert_with(HashMap::new);
        
        // Initialize symbol buffer if it doesn't exist
        let source_buffer = buffers.get_mut(source_id).unwrap();
        source_buffer.entry(market_data.symbol.clone()).or_insert_with(VecDeque::new);
        
        // Add data to buffer
        let symbol_buffer = source_buffer.get_mut(&market_data.symbol).unwrap();
        symbol_buffer.push_back(market_data.clone());
        
        // Maintain buffer size
        while symbol_buffer.len() > self.config.buffer_size {
            symbol_buffer.pop_front();
        }

        // Update source weights based on data quality
        self.update_source_weights(source_id, &market_data).await;

        Ok(())
    }

    /// Get aggregated market data for a symbol
    pub async fn get_aggregated_data(&self, symbol: &str) -> Result<Option<MarketData>> {
        // Check cache first
        let cache = self.aggregated_cache.read().await;
        if let Some(cached_data) = cache.get(symbol) {
            // Check if cache is still fresh
            let age = (Utc::now() - cached_data.timestamp).num_seconds();
            if age < self.config.max_data_age_seconds as i64 {
                return Ok(Some(self.convert_to_market_data(cached_data)));
            }
        }
        drop(cache);

        // Perform fresh aggregation
        self.aggregate_symbol_data(symbol).await
    }

    /// Perform aggregation for all symbols
    pub async fn aggregate_all(&self) -> Result<HashMap<String, MarketData>> {
        let mut results = HashMap::new();
        let buffers = self.raw_data_buffers.read().await;
        
        // Collect all unique symbols across all sources
        let mut symbols = std::collections::HashSet::new();
        for source_buffer in buffers.values() {
            symbols.extend(source_buffer.keys().cloned());
        }
        
        // Aggregate each symbol
        for symbol in symbols {
            if let Ok(Some(aggregated)) = self.aggregate_symbol_data(&symbol).await {
                results.insert(symbol, aggregated);
            }
        }

        Ok(results)
    }

    /// Get aggregation metrics
    pub async fn metrics(&self) -> AggregationMetrics {
        self.metrics.read().await.clone()
    }

    /// Update source weights and priorities
    pub async fn update_source_priority(&self, source_id: &str, weight: SourceWeight) -> Result<()> {
        let mut weights = self.source_weights.write().await;
        weights.insert(source_id.to_string(), weight);
        Ok(())
    }

    async fn aggregate_symbol_data(&self, symbol: &str) -> Result<Option<MarketData>> {
        let buffers = self.raw_data_buffers.read().await;
        let weights = self.source_weights.read().await;

        // Collect fresh data from all sources
        let mut fresh_data = Vec::new();
        let cutoff_time = Utc::now() - Duration::seconds(self.config.max_data_age_seconds as i64);

        for (source_id, source_buffer) in buffers.iter() {
            if let Some(symbol_buffer) = source_buffer.get(symbol) {
                if let Some(latest) = symbol_buffer.back() {
                    if latest.timestamp > cutoff_time {
                        fresh_data.push((source_id.clone(), latest.clone()));
                    }
                }
            }
        }

        // Check minimum source requirement
        if fresh_data.len() < self.config.min_sources {
            self.update_metrics(false, 0).await;
            return Ok(None);
        }

        // Detect and remove outliers
        let filtered_data = if self.config.outlier_detection.enabled {
            self.detect_outliers(fresh_data).await?
        } else {
            fresh_data
        };

        if filtered_data.is_empty() {
            self.update_metrics(false, 0).await;
            return Ok(None);
        }

        // Perform aggregation
        let aggregated = self.perform_aggregation(symbol, &filtered_data, &weights).await?;
        
        // Cache the result
        let mut cache = self.aggregated_cache.write().await;
        cache.insert(symbol.to_string(), aggregated.clone());

        // Update metrics
        self.update_metrics(true, filtered_data.len()).await;

        Ok(Some(self.convert_to_market_data(&aggregated)))
    }

    async fn detect_outliers(&self, data: Vec<(String, MarketData)>) -> Result<Vec<(String, MarketData)>> {
        if data.len() < 3 {
            return Ok(data); // Not enough data for outlier detection
        }

        let prices: Vec<f64> = data.iter()
            .map(|(_, d)| d.mid.to_f64().unwrap_or(0.0))
            .collect();

        let mean = prices.iter().sum::<f64>() / prices.len() as f64;
        let variance = prices.iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>() / prices.len() as f64;
        let std_dev = variance.sqrt();

        let mut filtered = Vec::new();
        let threshold = self.config.outlier_detection.z_score_threshold;

        for (source_id, market_data) in data {
            let price = market_data.mid.to_f64().unwrap_or(0.0);
            let z_score = if std_dev > 0.0 { 
                (price - mean).abs() / std_dev 
            } else { 
                0.0 
            };

            if z_score <= threshold {
                filtered.push((source_id, market_data));
            } else {
                // Log outlier detection
                warn!("Detected price outlier for {}: {} (z-score: {})", 
                      market_data.symbol, price, z_score);
                
                // Update outlier metrics
                let mut metrics = self.metrics.write().await;
                metrics.outliers_detected += 1;
            }
        }

        Ok(filtered)
    }

    async fn perform_aggregation(
        &self,
        symbol: &str,
        data: &[(String, MarketData)],
        weights: &HashMap<String, SourceWeight>,
    ) -> Result<AggregatedData> {
        let timestamp = Utc::now();

        // Calculate aggregated prices
        let aggregated_price = self.aggregate_prices(data, weights).await?;
        
        // Calculate aggregated volume
        let aggregated_volume = self.aggregate_volumes(data, weights).await?;
        
        // Calculate data quality
        let data_quality = self.calculate_data_quality(data).await?;

        Ok(AggregatedData {
            symbol: symbol.to_string(),
            timestamp,
            aggregated_price,
            aggregated_volume,
            data_quality,
            source_count: data.len(),
        })
    }

    async fn aggregate_prices(
        &self,
        data: &[(String, MarketData)],
        weights: &HashMap<String, SourceWeight>,
    ) -> Result<AggregatedPrice> {
        match self.config.aggregation_methods.price_method {
            PriceAggregationMethod::WeightedAverage => {
                let mut weighted_sum = 0.0;
                let mut weight_sum = 0.0;

                for (source_id, market_data) in data {
                    let weight = weights.get(source_id)
                        .map(|w| w.overall_weight)
                        .unwrap_or(1.0);
                    
                    let price = market_data.mid.to_f64().unwrap_or(0.0);
                    weighted_sum += price * weight;
                    weight_sum += weight;
                }

                let weighted_avg = if weight_sum > 0.0 { 
                    weighted_sum / weight_sum 
                } else { 
                    0.0 
                };

                Ok(AggregatedPrice {
                    bid: Decimal::from_f64_retain(weighted_avg * 0.9995).unwrap_or_default(),
                    ask: Decimal::from_f64_retain(weighted_avg * 1.0005).unwrap_or_default(),
                    mid: Decimal::from_f64_retain(weighted_avg).unwrap_or_default(),
                    last: Decimal::from_f64_retain(weighted_avg).unwrap_or_default(),
                    confidence_score: self.calculate_price_confidence(data).await,
                })
            }
            
            PriceAggregationMethod::VolumeWeightedAverage => {
                let mut volume_weighted_sum = 0.0;
                let mut total_volume = 0.0;

                for (_, market_data) in data {
                    let price = market_data.mid.to_f64().unwrap_or(0.0);
                    let volume = market_data.volume_24h.to_f64().unwrap_or(0.0);
                    
                    volume_weighted_sum += price * volume;
                    total_volume += volume;
                }

                let vwap = if total_volume > 0.0 { 
                    volume_weighted_sum / total_volume 
                } else { 
                    data.first().map(|(_, d)| d.mid.to_f64().unwrap_or(0.0)).unwrap_or(0.0) 
                };

                Ok(AggregatedPrice {
                    bid: Decimal::from_f64_retain(vwap * 0.9995).unwrap_or_default(),
                    ask: Decimal::from_f64_retain(vwap * 1.0005).unwrap_or_default(),
                    mid: Decimal::from_f64_retain(vwap).unwrap_or_default(),
                    last: Decimal::from_f64_retain(vwap).unwrap_or_default(),
                    confidence_score: self.calculate_price_confidence(data).await,
                })
            }
            
            _ => {
                // Fallback to simple average
                let avg_price = data.iter()
                    .map(|(_, d)| d.mid.to_f64().unwrap_or(0.0))
                    .sum::<f64>() / data.len() as f64;

                Ok(AggregatedPrice {
                    bid: Decimal::from_f64_retain(avg_price * 0.9995).unwrap_or_default(),
                    ask: Decimal::from_f64_retain(avg_price * 1.0005).unwrap_or_default(),
                    mid: Decimal::from_f64_retain(avg_price).unwrap_or_default(),
                    last: Decimal::from_f64_retain(avg_price).unwrap_or_default(),
                    confidence_score: self.calculate_price_confidence(data).await,
                })
            }
        }
    }

    async fn aggregate_volumes(
        &self,
        data: &[(String, MarketData)],
        weights: &HashMap<String, SourceWeight>,
    ) -> Result<AggregatedVolume> {
        match self.config.aggregation_methods.volume_method {
            VolumeAggregationMethod::Sum => {
                let total_volume = data.iter()
                    .map(|(_, d)| d.volume_24h)
                    .sum();

                Ok(AggregatedVolume {
                    total_volume,
                    weighted_volume: total_volume,
                    volume_confidence: 0.8, // Default confidence
                })
            }
            
            VolumeAggregationMethod::WeightedSum => {
                let mut weighted_sum = Decimal::ZERO;
                let mut weight_sum = 0.0;

                for (source_id, market_data) in data {
                    let weight = weights.get(source_id)
                        .map(|w| w.overall_weight)
                        .unwrap_or(1.0);
                    
                    weighted_sum += market_data.volume_24h * Decimal::from_f64_retain(weight).unwrap_or_default();
                    weight_sum += weight;
                }

                let normalized_volume = if weight_sum > 0.0 {
                    weighted_sum / Decimal::from_f64_retain(weight_sum).unwrap_or(Decimal::ONE)
                } else {
                    Decimal::ZERO
                };

                Ok(AggregatedVolume {
                    total_volume: weighted_sum,
                    weighted_volume: normalized_volume,
                    volume_confidence: 0.9,
                })
            }
            
            _ => {
                let avg_volume = data.iter()
                    .map(|(_, d)| d.volume_24h)
                    .sum::<Decimal>() / Decimal::from(data.len());

                Ok(AggregatedVolume {
                    total_volume: avg_volume,
                    weighted_volume: avg_volume,
                    volume_confidence: 0.7,
                })
            }
        }
    }

    async fn calculate_data_quality(&self, data: &[(String, MarketData)]) -> Result<DataQuality> {
        let now = Utc::now();
        
        // Calculate freshness score (based on average age)
        let avg_age = data.iter()
            .map(|(_, d)| (now - d.timestamp).num_seconds() as f64)
            .sum::<f64>() / data.len() as f64;
        let freshness_score = (1.0 - (avg_age / self.config.max_data_age_seconds as f64)).max(0.0);

        // Calculate consistency score (based on price variance)
        let prices: Vec<f64> = data.iter()
            .map(|(_, d)| d.mid.to_f64().unwrap_or(0.0))
            .collect();
        let mean_price = prices.iter().sum::<f64>() / prices.len() as f64;
        let price_variance = prices.iter()
            .map(|p| ((p - mean_price) / mean_price).powi(2))
            .sum::<f64>() / prices.len() as f64;
        let consistency_score = (1.0 - price_variance).max(0.0);

        // Calculate coverage score (based on number of sources)
        let coverage_score = (data.len() as f64 / 5.0).min(1.0); // Assume 5 is ideal

        let overall_score = (freshness_score + consistency_score + coverage_score) / 3.0;

        Ok(DataQuality {
            freshness_score,
            consistency_score,
            coverage_score,
            overall_score,
        })
    }

    async fn calculate_price_confidence(&self, data: &[(String, MarketData)]) -> f64 {
        if data.len() < 2 {
            return 0.5; // Low confidence with single source
        }

        let prices: Vec<f64> = data.iter()
            .map(|(_, d)| d.mid.to_f64().unwrap_or(0.0))
            .collect();

        let mean = prices.iter().sum::<f64>() / prices.len() as f64;
        let variance = prices.iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>() / prices.len() as f64;
        let coefficient_of_variation = if mean > 0.0 { 
            variance.sqrt() / mean 
        } else { 
            1.0 
        };

        // Lower coefficient of variation = higher confidence
        (1.0 - coefficient_of_variation).max(0.0).min(1.0)
    }

    async fn update_source_weights(&self, source_id: &str, market_data: &MarketData) {
        let mut weights = self.source_weights.write().await;
        
        let current_weight = weights.get(source_id).cloned().unwrap_or(SourceWeight {
            reliability_score: 0.8,
            latency_score: 0.8,
            volume_score: 0.8,
            overall_weight: 0.8,
        });

        // Update scores based on data quality (simplified)
        let data_age = (Utc::now() - market_data.timestamp).num_seconds() as f64;
        let latency_score = (1.0 - (data_age / 60.0)).max(0.0).min(1.0);
        
        let updated_weight = SourceWeight {
            reliability_score: current_weight.reliability_score * 0.9 + 0.1,
            latency_score,
            volume_score: current_weight.volume_score,
            overall_weight: (current_weight.reliability_score + latency_score + current_weight.volume_score) / 3.0,
        };

        weights.insert(source_id.to_string(), updated_weight);
    }

    async fn update_metrics(&self, success: bool, source_count: usize) {
        let mut metrics = self.metrics.write().await;
        metrics.total_aggregations += 1;
        
        if success {
            metrics.successful_aggregations += 1;
            metrics.average_sources_per_aggregation = 
                (metrics.average_sources_per_aggregation * (metrics.successful_aggregations - 1) as f64 + source_count as f64) 
                / metrics.successful_aggregations as f64;
        } else {
            metrics.failed_aggregations += 1;
        }
    }

    fn convert_to_market_data(&self, aggregated: &AggregatedData) -> MarketData {
        MarketData {
            symbol: aggregated.symbol.clone(),
            timestamp: aggregated.timestamp,
            bid: aggregated.aggregated_price.bid,
            ask: aggregated.aggregated_price.ask,
            mid: aggregated.aggregated_price.mid,
            last: aggregated.aggregated_price.last,
            volume_24h: aggregated.aggregated_volume.total_volume,
            bid_size: Decimal::from(100), // Placeholder
            ask_size: Decimal::from(100), // Placeholder
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[tokio::test]
    async fn test_aggregator_creation() {
        let config = DataAggregatorConfig::default();
        let aggregator = DataAggregator::new(config);
        
        let metrics = aggregator.metrics().await;
        assert_eq!(metrics.total_aggregations, 0);
    }

    #[tokio::test]
    async fn test_data_addition() {
        let config = DataAggregatorConfig::default();
        let aggregator = DataAggregator::new(config);
        
        let market_data = MarketData {
            symbol: "BTC/USD".to_string(),
            timestamp: Utc::now(),
            bid: dec!(50000),
            ask: dec!(50001),
            mid: dec!(50000.5),
            last: dec!(50000),
            volume_24h: dec!(1000),
            bid_size: dec!(10),
            ask_size: dec!(10),
        };

        let result = aggregator.add_data("source1", market_data).await;
        assert!(result.is_ok());
    }
}