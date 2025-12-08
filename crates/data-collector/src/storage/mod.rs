//! Storage backends for cryptocurrency data

use crate::{Result, DataCollectorError};
use crate::types::*;
use std::path::Path;
use tracing::{info, debug, warn};

pub mod parquet;
pub mod csv;
pub mod sqlite;
pub mod postgresql;

use crate::config::{StorageBackend, StorageConfig};

/// Storage trait for different backends
#[async_trait::async_trait]
pub trait Storage: Send + Sync {
    /// Store kline data
    async fn store_klines(&self, klines: &[Kline]) -> Result<()>;
    
    /// Store trade data
    async fn store_trades(&self, trades: &[Trade]) -> Result<()>;
    
    /// Store order book data
    async fn store_order_books(&self, order_books: &[OrderBook]) -> Result<()>;
    
    /// Store ticker data
    async fn store_tickers(&self, tickers: &[Ticker24hr]) -> Result<()>;
    
    /// Store funding rate data
    async fn store_funding_rates(&self, funding_rates: &[FundingRate]) -> Result<()>;
    
    /// Validate storage connection
    async fn validate(&self) -> Result<bool>;
    
    /// Get storage statistics
    async fn get_stats(&self) -> Result<StorageStats>;
    
    /// Backup data
    async fn backup(&self, backup_path: &Path) -> Result<()>;
    
    /// Restore data from backup
    async fn restore(&self, backup_path: &Path) -> Result<()>;
}

/// Storage statistics
#[derive(Debug, Clone)]
pub struct StorageStats {
    pub total_klines: u64,
    pub total_trades: u64,
    pub total_order_books: u64,
    pub total_tickers: u64,
    pub total_funding_rates: u64,
    pub storage_size_bytes: u64,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Multi-backend storage manager
pub struct StorageManager {
    storages: Vec<Box<dyn Storage>>,
    config: StorageConfig,
}

impl StorageManager {
    /// Create a new storage manager
    pub async fn new(config: StorageConfig) -> Result<Self> {
        let mut storages: Vec<Box<dyn Storage>> = Vec::new();
        
        match &config.backend {
            StorageBackend::Parquet => {
                let storage = parquet::ParquetStorage::new(&config).await?;
                storages.push(Box::new(storage));
            },
            StorageBackend::Csv => {
                let storage = csv::CsvStorage::new(&config).await?;
                storages.push(Box::new(storage));
            },
            StorageBackend::Sqlite => {
                let storage = sqlite::SqliteStorage::new(&config).await?;
                storages.push(Box::new(storage));
            },
            StorageBackend::Postgresql => {
                let storage = postgresql::PostgresqlStorage::new(&config).await?;
                storages.push(Box::new(storage));
            },
            StorageBackend::Combined(backends) => {
                for backend in backends {
                    let mut backend_config = config.clone();
                    backend_config.backend = backend.clone();
                    
                    match backend {
                        StorageBackend::Parquet => {
                            let storage = parquet::ParquetStorage::new(&backend_config).await?;
                            storages.push(Box::new(storage));
                        },
                        StorageBackend::Csv => {
                            let storage = csv::CsvStorage::new(&backend_config).await?;
                            storages.push(Box::new(storage));
                        },
                        StorageBackend::Sqlite => {
                            let storage = sqlite::SqliteStorage::new(&backend_config).await?;
                            storages.push(Box::new(storage));
                        },
                        StorageBackend::Postgresql => {
                            let storage = postgresql::PostgresqlStorage::new(&backend_config).await?;
                            storages.push(Box::new(storage));
                        },
                        StorageBackend::Combined(_) => {
                            return Err(DataCollectorError::Config(
                                "Nested combined storage backends not supported".to_string()
                            ));
                        }
                    }
                }
            }
        }
        
        info!("Initialized storage manager with {} backend(s)", storages.len());
        
        Ok(Self {
            storages,
            config,
        })
    }
    
    /// Store data to all configured backends
    pub async fn store_klines(&self, klines: &[Kline]) -> Result<()> {
        for storage in &self.storages {
            storage.store_klines(klines).await?;
        }
        debug!("Stored {} klines to {} backend(s)", klines.len(), self.storages.len());
        Ok(())
    }
    
    pub async fn store_trades(&self, trades: &[Trade]) -> Result<()> {
        for storage in &self.storages {
            storage.store_trades(trades).await?;
        }
        debug!("Stored {} trades to {} backend(s)", trades.len(), self.storages.len());
        Ok(())
    }
    
    pub async fn store_order_books(&self, order_books: &[OrderBook]) -> Result<()> {
        for storage in &self.storages {
            storage.store_order_books(order_books).await?;
        }
        debug!("Stored {} order books to {} backend(s)", order_books.len(), self.storages.len());
        Ok(())
    }
    
    pub async fn store_tickers(&self, tickers: &[Ticker24hr]) -> Result<()> {
        for storage in &self.storages {
            storage.store_tickers(tickers).await?;
        }
        debug!("Stored {} tickers to {} backend(s)", tickers.len(), self.storages.len());
        Ok(())
    }
    
    pub async fn store_funding_rates(&self, funding_rates: &[FundingRate]) -> Result<()> {
        for storage in &self.storages {
            storage.store_funding_rates(funding_rates).await?;
        }
        debug!("Stored {} funding rates to {} backend(s)", funding_rates.len(), self.storages.len());
        Ok(())
    }
    
    /// Validate all storage backends
    pub async fn validate_all(&self) -> Result<bool> {
        for (i, storage) in self.storages.iter().enumerate() {
            if !storage.validate().await? {
                warn!("Storage backend {} validation failed", i);
                return Ok(false);
            }
        }
        info!("All {} storage backend(s) validated successfully", self.storages.len());
        Ok(true)
    }
    
    /// Get aggregated storage statistics
    pub async fn get_aggregated_stats(&self) -> Result<StorageStats> {
        if self.storages.is_empty() {
            return Ok(StorageStats {
                total_klines: 0,
                total_trades: 0,
                total_order_books: 0,
                total_tickers: 0,
                total_funding_rates: 0,
                storage_size_bytes: 0,
                last_updated: chrono::Utc::now(),
            });
        }
        
        // Use stats from first storage backend
        let first_stats = self.storages[0].get_stats().await?;
        
        // For combined storage, we could aggregate here
        // For now, just return the first backend's stats
        Ok(first_stats)
    }
    
    /// Create backup of all data
    pub async fn backup_all(&self, backup_path: &Path) -> Result<()> {
        for (i, storage) in self.storages.iter().enumerate() {
            let backend_backup_path = backup_path.join(format!("backend_{}", i));
            std::fs::create_dir_all(&backend_backup_path)
                .map_err(|e| DataCollectorError::Storage(format!("Failed to create backup directory: {}", e)))?;
            
            storage.backup(&backend_backup_path).await?;
        }
        
        info!("Backup completed for all {} storage backend(s)", self.storages.len());
        Ok(())
    }
    
    /// Restore from backup
    pub async fn restore_all(&self, backup_path: &Path) -> Result<()> {
        for (i, storage) in self.storages.iter().enumerate() {
            let backend_backup_path = backup_path.join(format!("backend_{}", i));
            storage.restore(&backend_backup_path).await?;
        }
        
        info!("Restore completed for all {} storage backend(s)", self.storages.len());
        Ok(())
    }
}
