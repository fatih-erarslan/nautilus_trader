//! CSV storage backend for cryptocurrency data

use super::{Storage, StorageStats};
use crate::{Result, DataCollectorError};
use crate::config::StorageConfig;
use crate::types::*;
use std::path::Path;
use tracing::{info, debug};
use csv::Writer;
use std::fs::OpenOptions;

/// CSV storage backend
pub struct CsvStorage {
    config: StorageConfig,
    base_path: std::path::PathBuf,
}

impl CsvStorage {
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
        
        info!("Initialized CSV storage at: {:?}", base_path);
        
        Ok(Self {
            config: config.clone(),
            base_path,
        })
    }
    
    /// Generate file path for data type
    fn generate_file_path(&self, data_type: &str, exchange: &str, symbol: &str, date: &str) -> std::path::PathBuf {
        let filename = if self.config.enable_compression {
            format!("{}_{}_{}_{}.csv.gz", data_type, exchange, symbol, date)
        } else {
            format!("{}_{}_{}_{}.csv", data_type, exchange, symbol, date)
        };
        
        self.base_path
            .join(data_type)
            .join(exchange)
            .join(symbol)
            .join(filename)
    }
    
    /// Write klines to CSV
    async fn write_klines_to_csv(&self, klines: &[Kline]) -> Result<()> {
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
        
        // Check if file exists to determine if we need headers
        let file_exists = file_path.exists();
        
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)
            .map_err(|e| DataCollectorError::Storage(format!("Failed to open file: {}", e)))?;
        
        let mut writer = Writer::from_writer(file);
        
        // Write headers if this is a new file
        if !file_exists {
            writer.write_record(&[
                "symbol", "open_time", "close_time", "open", "high", "low", "close",
                "volume", "quote_volume", "trades_count", "taker_buy_base_volume",
                "taker_buy_quote_volume", "interval", "exchange"
            ]).map_err(|e| DataCollectorError::Storage(format!("Failed to write headers: {}", e)))?;
        }
        
        // Write kline data
        for kline in klines {
            writer.write_record(&[
                &kline.symbol,
                &kline.open_time.to_rfc3339(),
                &kline.close_time.to_rfc3339(),
                &kline.open.to_string(),
                &kline.high.to_string(),
                &kline.low.to_string(),
                &kline.close.to_string(),
                &kline.volume.to_string(),
                &kline.quote_volume.to_string(),
                &kline.trades_count.to_string(),
                &kline.taker_buy_base_volume.to_string(),
                &kline.taker_buy_quote_volume.to_string(),
                &format!("{:?}", kline.interval),
                &kline.exchange,
            ]).map_err(|e| DataCollectorError::Storage(format!("Failed to write record: {}", e)))?;
        }
        
        writer.flush()
            .map_err(|e| DataCollectorError::Storage(format!("Failed to flush writer: {}", e)))?;
        
        debug!("Wrote {} klines to: {:?}", klines.len(), file_path);
        Ok(())
    }
    
    /// Write trades to CSV
    async fn write_trades_to_csv(&self, trades: &[Trade]) -> Result<()> {
        if trades.is_empty() {
            return Ok(());
        }
        
        let exchange = &trades[0].exchange;
        let symbol = &trades[0].symbol;
        let date = trades[0].timestamp.format("%Y%m%d").to_string();
        
        let file_path = self.generate_file_path("trades", exchange, symbol, &date);
        
        // Create parent directory if it doesn't exist
        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| DataCollectorError::Storage(format!("Failed to create directory: {}", e)))?;
        }
        
        let file_exists = file_path.exists();
        
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)
            .map_err(|e| DataCollectorError::Storage(format!("Failed to open file: {}", e)))?;
        
        let mut writer = Writer::from_writer(file);
        
        if !file_exists {
            writer.write_record(&[
                "symbol", "trade_id", "price", "quantity", "quote_quantity",
                "timestamp", "is_buyer_maker", "exchange"
            ]).map_err(|e| DataCollectorError::Storage(format!("Failed to write headers: {}", e)))?;
        }
        
        for trade in trades {
            writer.write_record(&[
                &trade.symbol,
                &trade.trade_id.to_string(),
                &trade.price.to_string(),
                &trade.quantity.to_string(),
                &trade.quote_quantity.to_string(),
                &trade.timestamp.to_rfc3339(),
                &trade.is_buyer_maker.to_string(),
                &trade.exchange,
            ]).map_err(|e| DataCollectorError::Storage(format!("Failed to write record: {}", e)))?;
        }
        
        writer.flush()
            .map_err(|e| DataCollectorError::Storage(format!("Failed to flush writer: {}", e)))?;
        
        debug!("Wrote {} trades to: {:?}", trades.len(), file_path);
        Ok(())
    }
}

#[async_trait::async_trait]
impl Storage for CsvStorage {
    async fn store_klines(&self, klines: &[Kline]) -> Result<()> {
        self.write_klines_to_csv(klines).await
    }
    
    async fn store_trades(&self, trades: &[Trade]) -> Result<()> {
        self.write_trades_to_csv(trades).await
    }
    
    async fn store_order_books(&self, order_books: &[OrderBook]) -> Result<()> {
        debug!("Storing {} order books to CSV (placeholder)", order_books.len());
        Ok(())
    }
    
    async fn store_tickers(&self, tickers: &[Ticker24hr]) -> Result<()> {
        debug!("Storing {} tickers to CSV (placeholder)", tickers.len());
        Ok(())
    }
    
    async fn store_funding_rates(&self, funding_rates: &[FundingRate]) -> Result<()> {
        debug!("Storing {} funding rates to CSV (placeholder)", funding_rates.len());
        Ok(())
    }
    
    async fn validate(&self) -> Result<bool> {
        if !self.base_path.exists() {
            return Ok(false);
        }
        
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
        let backup_target = backup_path.join("csv_data");
        std::fs::create_dir_all(&backup_target)
            .map_err(|e| DataCollectorError::Storage(format!("Failed to create backup directory: {}", e)))?;
        
        let status = std::process::Command::new("cp")
            .arg("-r")
            .arg(&self.base_path)
            .arg(&backup_target)
            .status()
            .map_err(|e| DataCollectorError::Storage(format!("Failed to execute backup: {}", e)))?;
            
        if !status.success() {
            return Err(DataCollectorError::Storage("Backup command failed".to_string()));
        }
        
        info!("CSV data backed up to: {:?}", backup_target);
        Ok(())
    }
    
    async fn restore(&self, backup_path: &Path) -> Result<()> {
        let backup_source = backup_path.join("csv_data");
        
        if !backup_source.exists() {
            return Err(DataCollectorError::Storage("Backup source not found".to_string()));
        }
        
        if self.base_path.exists() {
            std::fs::remove_dir_all(&self.base_path)
                .map_err(|e| DataCollectorError::Storage(format!("Failed to remove existing data: {}", e)))?;
        }
        
        let status = std::process::Command::new("cp")
            .arg("-r")
            .arg(&backup_source)
            .arg(&self.base_path)
            .status()
            .map_err(|e| DataCollectorError::Storage(format!("Failed to execute restore: {}", e)))?;
            
        if !status.success() {
            return Err(DataCollectorError::Storage("Restore command failed".to_string()));
        }
        
        info!("CSV data restored from: {:?}", backup_source);
        Ok(())
    }
}