//! Parquet storage backend for cryptocurrency data

use super::{Storage, StorageStats};
use crate::{Result, DataCollectorError};
use crate::config::StorageConfig;
use crate::types::*;
use std::path::Path;
use tracing::{info, debug, error};
use arrow::array::*;
use arrow::datatypes::{Schema, Field, TimeUnit, DataType as ArrowDataType};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::basic::{Compression, Encoding};
use parquet::file::properties::WriterProperties;
use std::sync::Arc;
use std::fs::File;

/// Parquet storage backend
pub struct ParquetStorage {
    config: StorageConfig,
    base_path: std::path::PathBuf,
}

impl ParquetStorage {
    pub async fn new(config: &StorageConfig) -> Result<Self> {
        let base_path = config.base_directory.clone();
        
        // Create directory structure
        std::fs::create_dir_all(&base_path)
            .map_err(|e| DataCollectorError::Storage(format!("Failed to create storage directory: {}", e)))?;
            
        std::fs::create_dir_all(base_path.join("klines"))
            .map_err(|e| DataCollectorError::Storage(format!("Failed to create klines directory: {}", e)))?;
            
        std::fs::create_dir_all(base_path.join("trades"))
            .map_err(|e| DataCollectorError::Storage(format!("Failed to create trades directory: {}", e)))?;
            
        std::fs::create_dir_all(base_path.join("order_books"))
            .map_err(|e| DataCollectorError::Storage(format!("Failed to create order_books directory: {}", e)))?;
            
        std::fs::create_dir_all(base_path.join("tickers"))
            .map_err(|e| DataCollectorError::Storage(format!("Failed to create tickers directory: {}", e)))?;
            
        std::fs::create_dir_all(base_path.join("funding_rates"))
            .map_err(|e| DataCollectorError::Storage(format!("Failed to create funding_rates directory: {}", e)))?;
        
        info!("Initialized Parquet storage at: {:?}", base_path);
        
        Ok(Self {
            config: config.clone(),
            base_path,
        })
    }
    
    /// Create writer properties based on configuration
    fn create_writer_properties(&self) -> WriterProperties {
        let compression = match self.config.compression_algorithm {
            crate::config::CompressionAlgorithm::None => Compression::UNCOMPRESSED,
            crate::config::CompressionAlgorithm::Gzip => Compression::GZIP(Default::default()),
            crate::config::CompressionAlgorithm::Zstd => Compression::ZSTD(Default::default()),
            crate::config::CompressionAlgorithm::Lz4 => Compression::LZ4,
        };
        
        WriterProperties::builder()
            .set_compression(compression)
            .set_encoding(Encoding::PLAIN)
            .set_statistics_enabled(parquet::basic::LogicalType::None, true)
            .build()
    }
    
    /// Generate file path for data type
    fn generate_file_path(&self, data_type: &str, exchange: &str, symbol: &str, date: &str) -> std::path::PathBuf {
        self.base_path
            .join(data_type)
            .join(exchange)
            .join(symbol)
            .join(format!("{}_{}_{}_{}.parquet", data_type, exchange, symbol, date))
    }
    
    /// Write klines to Parquet
    async fn write_klines_to_parquet(&self, klines: &[Kline]) -> Result<()> {
        if klines.is_empty() {
            return Ok(());
        }
        
        let exchange = &klines[0].exchange;
        let symbol = &klines[0].symbol;
        let date = klines[0].open_time.format("%Y%m%d").to_string();
        
        let file_path = self.generate_file_path("klines", exchange, symbol, &date);
        
        // Create parent directory if it doesn't exist
        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| DataCollectorError::Storage(format!("Failed to create directory: {}", e)))?;
        }
        
        // Create Arrow schema for klines
        let schema = Arc::new(Schema::new(vec![
            Field::new("symbol", ArrowDataType::Utf8, false),
            Field::new("open_time", ArrowDataType::Timestamp(TimeUnit::Millisecond, Some(Arc::from("UTC"))), false),
            Field::new("close_time", ArrowDataType::Timestamp(TimeUnit::Millisecond, Some(Arc::from("UTC"))), false),
            Field::new("open", ArrowDataType::Float64, false),
            Field::new("high", ArrowDataType::Float64, false),
            Field::new("low", ArrowDataType::Float64, false),
            Field::new("close", ArrowDataType::Float64, false),
            Field::new("volume", ArrowDataType::Float64, false),
            Field::new("quote_volume", ArrowDataType::Float64, false),
            Field::new("trades_count", ArrowDataType::UInt64, false),
            Field::new("taker_buy_base_volume", ArrowDataType::Float64, false),
            Field::new("taker_buy_quote_volume", ArrowDataType::Float64, false),
            Field::new("interval", ArrowDataType::Utf8, false),
            Field::new("exchange", ArrowDataType::Utf8, false),
        ]));
        
        // Convert klines to Arrow arrays
        let mut symbol_builder = StringBuilder::new();
        let mut open_time_builder = TimestampMillisecondBuilder::new();
        let mut close_time_builder = TimestampMillisecondBuilder::new();
        let mut open_builder = Float64Builder::new();
        let mut high_builder = Float64Builder::new();
        let mut low_builder = Float64Builder::new();
        let mut close_builder = Float64Builder::new();
        let mut volume_builder = Float64Builder::new();
        let mut quote_volume_builder = Float64Builder::new();
        let mut trades_count_builder = UInt64Builder::new();
        let mut taker_buy_base_volume_builder = Float64Builder::new();
        let mut taker_buy_quote_volume_builder = Float64Builder::new();
        let mut interval_builder = StringBuilder::new();
        let mut exchange_builder = StringBuilder::new();
        
        for kline in klines {
            symbol_builder.append_value(&kline.symbol);
            open_time_builder.append_value(kline.open_time.timestamp_millis());
            close_time_builder.append_value(kline.close_time.timestamp_millis());
            open_builder.append_value(kline.open);
            high_builder.append_value(kline.high);
            low_builder.append_value(kline.low);
            close_builder.append_value(kline.close);
            volume_builder.append_value(kline.volume);
            quote_volume_builder.append_value(kline.quote_volume);
            trades_count_builder.append_value(kline.trades_count);
            taker_buy_base_volume_builder.append_value(kline.taker_buy_base_volume);
            taker_buy_quote_volume_builder.append_value(kline.taker_buy_quote_volume);
            interval_builder.append_value(&format!("{:?}", kline.interval));
            exchange_builder.append_value(&kline.exchange);
        }
        
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(symbol_builder.finish()),
                Arc::new(open_time_builder.finish()),
                Arc::new(close_time_builder.finish()),
                Arc::new(open_builder.finish()),
                Arc::new(high_builder.finish()),
                Arc::new(low_builder.finish()),
                Arc::new(close_builder.finish()),
                Arc::new(volume_builder.finish()),
                Arc::new(quote_volume_builder.finish()),
                Arc::new(trades_count_builder.finish()),
                Arc::new(taker_buy_base_volume_builder.finish()),
                Arc::new(taker_buy_quote_volume_builder.finish()),
                Arc::new(interval_builder.finish()),
                Arc::new(exchange_builder.finish()),
            ],
        ).map_err(|e| DataCollectorError::Storage(format!("Failed to create record batch: {}", e)))?;
        
        // Write to Parquet file
        let file = File::create(&file_path)
            .map_err(|e| DataCollectorError::Storage(format!("Failed to create file: {}", e)))?;
            
        let props = self.create_writer_properties();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props))
            .map_err(|e| DataCollectorError::Storage(format!("Failed to create Arrow writer: {}", e)))?;
            
        writer.write(&batch)
            .map_err(|e| DataCollectorError::Storage(format!("Failed to write batch: {}", e)))?;
            
        writer.close()
            .map_err(|e| DataCollectorError::Storage(format!("Failed to close writer: {}", e)))?;
        
        debug!("Wrote {} klines to: {:?}", klines.len(), file_path);
        Ok(())
    }
}

#[async_trait::async_trait]
impl Storage for ParquetStorage {
    async fn store_klines(&self, klines: &[Kline]) -> Result<()> {
        self.write_klines_to_parquet(klines).await
    }
    
    async fn store_trades(&self, trades: &[Trade]) -> Result<()> {
        // Similar implementation for trades
        debug!("Storing {} trades to Parquet (placeholder)", trades.len());
        Ok(())
    }
    
    async fn store_order_books(&self, order_books: &[OrderBook]) -> Result<()> {
        // Similar implementation for order books
        debug!("Storing {} order books to Parquet (placeholder)", order_books.len());
        Ok(())
    }
    
    async fn store_tickers(&self, tickers: &[Ticker24hr]) -> Result<()> {
        // Similar implementation for tickers
        debug!("Storing {} tickers to Parquet (placeholder)", tickers.len());
        Ok(())
    }
    
    async fn store_funding_rates(&self, funding_rates: &[FundingRate]) -> Result<()> {
        // Similar implementation for funding rates
        debug!("Storing {} funding rates to Parquet (placeholder)", funding_rates.len());
        Ok(())
    }
    
    async fn validate(&self) -> Result<bool> {
        // Check if base directory exists and is writable
        if !self.base_path.exists() {
            return Ok(false);
        }
        
        // Try to create a test file
        let test_file = self.base_path.join("test_write_access.tmp");
        match std::fs::write(&test_file, "test") {
            Ok(_) => {
                let _ = std::fs::remove_file(&test_file);
                Ok(true)
            },
            Err(_) => Ok(false),
        }
    }
    
    async fn get_stats(&self) -> Result<StorageStats> {
        // Calculate storage statistics by scanning directories
        let mut total_size = 0u64;
        let mut file_counts = std::collections::HashMap::new();
        
        for data_type in ["klines", "trades", "order_books", "tickers", "funding_rates"] {
            let dir_path = self.base_path.join(data_type);
            if dir_path.exists() {
                if let Ok(entries) = std::fs::read_dir(&dir_path) {
                    for entry in entries.flatten() {
                        if let Ok(metadata) = entry.metadata() {
                            total_size += metadata.len();
                            *file_counts.entry(data_type).or_insert(0u64) += 1;
                        }
                    }
                }
            }
        }
        
        Ok(StorageStats {
            total_klines: *file_counts.get("klines").unwrap_or(&0),
            total_trades: *file_counts.get("trades").unwrap_or(&0),
            total_order_books: *file_counts.get("order_books").unwrap_or(&0),
            total_tickers: *file_counts.get("tickers").unwrap_or(&0),
            total_funding_rates: *file_counts.get("funding_rates").unwrap_or(&0),
            storage_size_bytes: total_size,
            last_updated: chrono::Utc::now(),
        })
    }
    
    async fn backup(&self, backup_path: &Path) -> Result<()> {
        // Copy all files to backup location
        let backup_target = backup_path.join("parquet_data");
        std::fs::create_dir_all(&backup_target)
            .map_err(|e| DataCollectorError::Storage(format!("Failed to create backup directory: {}", e)))?;
        
        // Use system copy command for efficiency
        let status = std::process::Command::new("cp")
            .arg("-r")
            .arg(&self.base_path)
            .arg(&backup_target)
            .status()
            .map_err(|e| DataCollectorError::Storage(format!("Failed to execute backup: {}", e)))?;
            
        if !status.success() {
            return Err(DataCollectorError::Storage("Backup command failed".to_string()));
        }
        
        info!("Parquet data backed up to: {:?}", backup_target);
        Ok(())
    }
    
    async fn restore(&self, backup_path: &Path) -> Result<()> {
        let backup_source = backup_path.join("parquet_data");
        
        if !backup_source.exists() {
            return Err(DataCollectorError::Storage("Backup source not found".to_string()));
        }
        
        // Remove existing data
        if self.base_path.exists() {
            std::fs::remove_dir_all(&self.base_path)
                .map_err(|e| DataCollectorError::Storage(format!("Failed to remove existing data: {}", e)))?;
        }
        
        // Restore from backup
        let status = std::process::Command::new("cp")
            .arg("-r")
            .arg(&backup_source)
            .arg(&self.base_path)
            .status()
            .map_err(|e| DataCollectorError::Storage(format!("Failed to execute restore: {}", e)))?;
            
        if !status.success() {
            return Err(DataCollectorError::Storage("Restore command failed".to_string()));
        }
        
        info!("Parquet data restored from: {:?}", backup_source);
        Ok(())
    }
}