//! Market data normalizer implementation

use crate::prelude::*;
use crate::models::MarketData;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use num_traits::ToPrimitive;
use std::collections::HashMap;

/// Market data normalizer that standardizes data from different sources
#[derive(Debug, Clone)]
pub struct DataNormalizer {
    /// Normalizer configuration
    config: DataNormalizerConfig,
    
    /// Currency conversion rates
    conversion_rates: HashMap<String, Decimal>,
    
    /// Symbol mappings for different exchanges
    symbol_mappings: HashMap<String, String>,
    
    /// Normalization statistics
    stats: NormalizationStats,
}

#[derive(Debug, Clone)]
pub struct DataNormalizerConfig {
    /// Base currency for normalization
    pub base_currency: String,
    
    /// Decimal precision for prices
    pub price_precision: u32,
    
    /// Decimal precision for volumes
    pub volume_precision: u32,
    
    /// Time zone for timestamp normalization
    pub timezone: String,
    
    /// Exchange-specific configurations
    pub exchange_configs: HashMap<String, ExchangeConfig>,
    
    /// Data validation rules
    pub validation_rules: ValidationConfig,
}

#[derive(Debug, Clone)]
pub struct ExchangeConfig {
    /// Price scaling factor
    pub price_scale: Decimal,
    
    /// Volume scaling factor
    pub volume_scale: Decimal,
    
    /// Symbol format (e.g., "BTC-USD", "BTCUSD", "BTC/USD")
    pub symbol_format: SymbolFormat,
    
    /// Timestamp format
    pub timestamp_format: TimestampFormat,
}

#[derive(Debug, Clone)]
pub enum SymbolFormat {
    Slash,      // BTC/USD
    Dash,       // BTC-USD
    Underscore, // BTC_USD
    Concat,     // BTCUSD
    Custom(String), // Custom format string
}

#[derive(Debug, Clone)]
pub enum TimestampFormat {
    Unix,           // Unix timestamp
    UnixMillis,     // Unix timestamp in milliseconds
    ISO8601,        // ISO 8601 format
    RFC3339,        // RFC 3339 format
    Custom(String), // Custom format string
}

#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Minimum valid price
    pub min_price: Decimal,
    
    /// Maximum valid price
    pub max_price: Decimal,
    
    /// Minimum valid volume
    pub min_volume: Decimal,
    
    /// Maximum valid volume
    pub max_volume: Decimal,
    
    /// Maximum allowed spread percentage
    pub max_spread_pct: f64,
    
    /// Maximum age for data to be considered valid
    pub max_age_seconds: u64,
}

#[derive(Debug, Clone, Default)]
struct NormalizationStats {
    total_processed: u64,
    successful_normalizations: u64,
    validation_failures: u64,
    conversion_failures: u64,
    format_corrections: u64,
}

impl Default for DataNormalizerConfig {
    fn default() -> Self {
        Self {
            base_currency: "USD".to_string(),
            price_precision: 8,
            volume_precision: 6,
            timezone: "UTC".to_string(),
            exchange_configs: HashMap::new(),
            validation_rules: ValidationConfig {
                min_price: Decimal::from_f64_retain(0.0001).unwrap(),
                max_price: Decimal::from_f64_retain(1000000.0).unwrap(),
                min_volume: Decimal::ZERO,
                max_volume: Decimal::from_f64_retain(1000000000.0).unwrap(),
                max_spread_pct: 0.1,
                max_age_seconds: 300,
            },
        }
    }
}

impl DataNormalizer {
    /// Create a new data normalizer
    pub fn new(config: DataNormalizerConfig) -> Self {
        let mut normalizer = Self {
            config,
            conversion_rates: HashMap::new(),
            symbol_mappings: HashMap::new(),
            stats: NormalizationStats::default(),
        };

        // Initialize default conversion rates
        normalizer.initialize_conversion_rates();
        
        // Initialize symbol mappings
        normalizer.initialize_symbol_mappings();

        normalizer
    }

    /// Normalize market data from a specific exchange
    pub async fn normalize(&mut self, exchange_id: &str, raw_data: MarketData) -> Result<MarketData> {
        self.stats.total_processed += 1;

        // Get exchange configuration
        let exchange_config = self.config.exchange_configs
            .get(exchange_id)
            .cloned()
            .unwrap_or_else(|| self.default_exchange_config());

        // Normalize symbol format
        let normalized_symbol = self.normalize_symbol(&raw_data.symbol, &exchange_config)?;

        // Normalize timestamp
        let normalized_timestamp = self.normalize_timestamp(raw_data.timestamp)?;

        // Normalize prices
        let normalized_prices = self.normalize_prices(&raw_data, &exchange_config).await?;

        // Normalize volumes
        let normalized_volumes = self.normalize_volumes(&raw_data, &exchange_config).await?;

        // Validate normalized data
        let normalized_data = MarketData {
            symbol: normalized_symbol,
            timestamp: normalized_timestamp,
            bid: normalized_prices.bid,
            ask: normalized_prices.ask,
            mid: normalized_prices.mid,
            last: normalized_prices.last,
            volume_24h: normalized_volumes.volume_24h,
            bid_size: normalized_volumes.bid_size,
            ask_size: normalized_volumes.ask_size,
        };

        if self.validate_data(&normalized_data)? {
            self.stats.successful_normalizations += 1;
            Ok(normalized_data)
        } else {
            self.stats.validation_failures += 1;
            Err(Error::MarketData("Data validation failed after normalization".to_string()))
        }
    }

    /// Normalize a batch of market data
    pub async fn normalize_batch(&mut self, exchange_id: &str, raw_data: Vec<MarketData>) -> Result<Vec<MarketData>> {
        let mut normalized_batch = Vec::with_capacity(raw_data.len());

        for data in raw_data {
            match self.normalize(exchange_id, data).await {
                Ok(normalized) => normalized_batch.push(normalized),
                Err(e) => {
                    warn!("Failed to normalize data: {}", e);
                    continue;
                }
            }
        }

        Ok(normalized_batch)
    }

    /// Update conversion rates
    pub fn update_conversion_rates(&mut self, rates: HashMap<String, Decimal>) {
        self.conversion_rates.extend(rates);
    }

    /// Add symbol mapping
    pub fn add_symbol_mapping(&mut self, exchange_symbol: String, standard_symbol: String) {
        self.symbol_mappings.insert(exchange_symbol, standard_symbol);
    }

    /// Get normalization statistics
    pub fn stats(&self) -> &NormalizationStats {
        &self.stats
    }

    fn initialize_conversion_rates(&mut self) {
        // Initialize with common USD pairs (in a real system, these would be fetched from an API)
        self.conversion_rates.insert("USD".to_string(), Decimal::ONE);
        self.conversion_rates.insert("EUR".to_string(), Decimal::from_f64_retain(1.1).unwrap());
        self.conversion_rates.insert("GBP".to_string(), Decimal::from_f64_retain(1.25).unwrap());
        self.conversion_rates.insert("JPY".to_string(), Decimal::from_f64_retain(0.007).unwrap());
        self.conversion_rates.insert("CAD".to_string(), Decimal::from_f64_retain(0.75).unwrap());
        self.conversion_rates.insert("AUD".to_string(), Decimal::from_f64_retain(0.65).unwrap());
    }

    fn initialize_symbol_mappings(&mut self) {
        // Common symbol mappings
        self.symbol_mappings.insert("BTCUSD".to_string(), "BTC/USD".to_string());
        self.symbol_mappings.insert("BTC-USD".to_string(), "BTC/USD".to_string());
        self.symbol_mappings.insert("BTC_USD".to_string(), "BTC/USD".to_string());
        self.symbol_mappings.insert("ETHUSD".to_string(), "ETH/USD".to_string());
        self.symbol_mappings.insert("ETH-USD".to_string(), "ETH/USD".to_string());
        self.symbol_mappings.insert("ETH_USD".to_string(), "ETH/USD".to_string());
    }

    fn default_exchange_config(&self) -> ExchangeConfig {
        ExchangeConfig {
            price_scale: Decimal::ONE,
            volume_scale: Decimal::ONE,
            symbol_format: SymbolFormat::Slash,
            timestamp_format: TimestampFormat::RFC3339,
        }
    }

    fn normalize_symbol(&mut self, symbol: &str, config: &ExchangeConfig) -> Result<String> {
        // First check if we have a direct mapping
        if let Some(mapped) = self.symbol_mappings.get(symbol) {
            return Ok(mapped.clone());
        }

        // Try to convert based on format
        let normalized = match config.symbol_format {
            SymbolFormat::Slash => {
                if symbol.contains('/') {
                    symbol.to_string()
                } else {
                    self.convert_to_slash_format(symbol)?
                }
            }
            SymbolFormat::Dash => {
                symbol.replace('-', "/")
            }
            SymbolFormat::Underscore => {
                symbol.replace('_', "/")
            }
            SymbolFormat::Concat => {
                self.split_concatenated_symbol(symbol)?
            }
            SymbolFormat::Custom(_) => {
                // Would implement custom format parsing here
                symbol.to_string()
            }
        };

        // Cache the mapping for future use
        self.symbol_mappings.insert(symbol.to_string(), normalized.clone());
        self.stats.format_corrections += 1;

        Ok(normalized)
    }

    fn convert_to_slash_format(&self, symbol: &str) -> Result<String> {
        // Common crypto pairs
        let common_pairs = [
            ("BTCUSD", "BTC/USD"), ("ETHUSD", "ETH/USD"), ("ADAUSD", "ADA/USD"),
            ("XRPUSD", "XRP/USD"), ("DOTUSD", "DOT/USD"), ("LINKUSD", "LINK/USD"),
            ("LTCUSD", "LTC/USD"), ("BCHUSD", "BCH/USD"), ("XLMUSD", "XLM/USD"),
        ];

        for (concat, slash) in &common_pairs {
            if symbol.eq_ignore_ascii_case(concat) {
                return Ok(slash.to_string());
            }
        }

        // Fallback: assume first 3-4 chars are base, rest is quote
        if symbol.len() >= 6 {
            let (base, quote) = if symbol.len() == 6 {
                symbol.split_at(3)
            } else {
                // Try to be smart about longer symbols
                if symbol.ends_with("USD") {
                    (&symbol[..symbol.len()-3], "USD")
                } else if symbol.ends_with("BTC") {
                    (&symbol[..symbol.len()-3], "BTC")
                } else if symbol.ends_with("ETH") {
                    (&symbol[..symbol.len()-3], "ETH")
                } else {
                    symbol.split_at(3)
                }
            };
            Ok(format!("{}/{}", base, quote))
        } else {
            Err(Error::MarketData(format!("Cannot parse symbol format: {}", symbol)))
        }
    }

    fn split_concatenated_symbol(&self, symbol: &str) -> Result<String> {
        self.convert_to_slash_format(symbol)
    }

    fn normalize_timestamp(&self, timestamp: DateTime<Utc>) -> Result<DateTime<Utc>> {
        // Ensure timestamp is in UTC (it already is based on the type)
        // Additional validation could be added here
        let now = Utc::now();
        let max_age = chrono::Duration::seconds(self.config.validation_rules.max_age_seconds as i64);
        
        if timestamp < now - max_age {
            warn!("Timestamp is too old: {}", timestamp);
        }
        
        if timestamp > now + chrono::Duration::minutes(5) {
            warn!("Timestamp is in the future: {}", timestamp);
        }

        Ok(timestamp)
    }

    async fn normalize_prices(&self, data: &MarketData, config: &ExchangeConfig) -> Result<NormalizedPrices> {
        // Apply scaling factor
        let bid = self.apply_price_scaling(data.bid, config)?;
        let ask = self.apply_price_scaling(data.ask, config)?;
        let mid = self.apply_price_scaling(data.mid, config)?;
        let last = self.apply_price_scaling(data.last, config)?;

        // Apply currency conversion if needed
        let converted_prices = if self.needs_currency_conversion(&data.symbol) {
            self.convert_currency_prices(NormalizedPrices { bid, ask, mid, last }).await?
        } else {
            NormalizedPrices { bid, ask, mid, last }
        };

        // Apply precision rounding
        Ok(NormalizedPrices {
            bid: self.round_to_precision(converted_prices.bid, self.config.price_precision),
            ask: self.round_to_precision(converted_prices.ask, self.config.price_precision),
            mid: self.round_to_precision(converted_prices.mid, self.config.price_precision),
            last: self.round_to_precision(converted_prices.last, self.config.price_precision),
        })
    }

    async fn normalize_volumes(&self, data: &MarketData, config: &ExchangeConfig) -> Result<NormalizedVolumes> {
        // Apply scaling factor
        let volume_24h = data.volume_24h * config.volume_scale;
        let bid_size = data.bid_size * config.volume_scale;
        let ask_size = data.ask_size * config.volume_scale;

        // Apply precision rounding
        Ok(NormalizedVolumes {
            volume_24h: self.round_to_precision(volume_24h, self.config.volume_precision),
            bid_size: self.round_to_precision(bid_size, self.config.volume_precision),
            ask_size: self.round_to_precision(ask_size, self.config.volume_precision),
        })
    }

    fn apply_price_scaling(&self, price: Decimal, config: &ExchangeConfig) -> Result<Decimal> {
        Ok(price * config.price_scale)
    }

    fn needs_currency_conversion(&self, symbol: &str) -> bool {
        // Check if the quote currency is not our base currency
        if let Some(slash_pos) = symbol.rfind('/') {
            let quote_currency = &symbol[slash_pos + 1..];
            quote_currency != self.config.base_currency
        } else {
            false
        }
    }

    async fn convert_currency_prices(&self, prices: NormalizedPrices) -> Result<NormalizedPrices> {
        // In a real implementation, this would use current exchange rates
        // For now, we'll just return the prices as-is since we assume USD base
        Ok(prices)
    }

    fn round_to_precision(&self, value: Decimal, precision: u32) -> Decimal {
        let scale_factor = Decimal::from(10_u64.pow(precision));
        (value * scale_factor).round() / scale_factor
    }

    fn validate_data(&mut self, data: &MarketData) -> Result<bool> {
        let rules = &self.config.validation_rules;

        // Check price ranges
        if data.bid < rules.min_price || data.bid > rules.max_price {
            return Ok(false);
        }
        if data.ask < rules.min_price || data.ask > rules.max_price {
            return Ok(false);
        }

        // Check bid <= ask
        if data.bid > data.ask {
            return Ok(false);
        }

        // Check spread
        let spread_pct = ((data.ask - data.bid) / data.mid).to_f64().unwrap_or(0.0);
        if spread_pct > rules.max_spread_pct {
            return Ok(false);
        }

        // Check volume ranges
        if data.volume_24h < rules.min_volume || data.volume_24h > rules.max_volume {
            return Ok(false);
        }

        // Check timestamp age
        let age = (Utc::now() - data.timestamp).num_seconds() as u64;
        if age > rules.max_age_seconds {
            return Ok(false);
        }

        Ok(true)
    }
}

#[derive(Debug, Clone)]
struct NormalizedPrices {
    bid: Decimal,
    ask: Decimal,
    mid: Decimal,
    last: Decimal,
}

#[derive(Debug, Clone)]
struct NormalizedVolumes {
    volume_24h: Decimal,
    bid_size: Decimal,
    ask_size: Decimal,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[tokio::test]
    async fn test_normalizer_creation() {
        let config = DataNormalizerConfig::default();
        let normalizer = DataNormalizer::new(config);
        
        assert_eq!(normalizer.stats.total_processed, 0);
    }

    #[tokio::test]
    async fn test_symbol_normalization() {
        let config = DataNormalizerConfig::default();
        let mut normalizer = DataNormalizer::new(config);
        
        let exchange_config = ExchangeConfig {
            price_scale: Decimal::ONE,
            volume_scale: Decimal::ONE,
            symbol_format: SymbolFormat::Concat,
            timestamp_format: TimestampFormat::RFC3339,
        };

        let result = normalizer.normalize_symbol("BTCUSD", &exchange_config);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "BTC/USD");
    }

    #[tokio::test]
    async fn test_data_normalization() {
        let config = DataNormalizerConfig::default();
        let mut normalizer = DataNormalizer::new(config);
        
        let raw_data = MarketData {
            symbol: "BTCUSD".to_string(),
            timestamp: Utc::now(),
            bid: dec!(50000),
            ask: dec!(50001),
            mid: dec!(50000.5),
            last: dec!(50000),
            volume_24h: dec!(1000),
            bid_size: dec!(10),
            ask_size: dec!(10),
        };

        let result = normalizer.normalize("test_exchange", raw_data).await;
        assert!(result.is_ok());
        
        let normalized = result.unwrap();
        assert_eq!(normalized.symbol, "BTC/USD");
    }
}