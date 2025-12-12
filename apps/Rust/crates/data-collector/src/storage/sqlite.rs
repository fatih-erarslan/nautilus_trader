//! SQLite storage backend for cryptocurrency data

use super::{Storage, StorageStats};
use crate::{Result, DataCollectorError};
use crate::config::StorageConfig;
use crate::types::*;
use std::path::Path;
use tracing::{info, debug};
use sqlx::{sqlite::{SqlitePool, SqlitePoolOptions}, Row};

/// SQLite storage backend
pub struct SqliteStorage {
    pool: SqlitePool,
    config: StorageConfig,
}

impl SqliteStorage {
    pub async fn new(config: &StorageConfig) -> Result<Self> {
        let db_path = config.base_directory.join("crypto_data.db");
        
        // Create parent directory if it doesn't exist
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| DataCollectorError::Storage(format!("Failed to create storage directory: {}", e)))?;
        }
        
        let database_url = format!("sqlite:{}", db_path.to_string_lossy());
        
        let pool = SqlitePoolOptions::new()
            .max_connections(10)
            .connect(&database_url)
            .await
            .map_err(|e| DataCollectorError::Storage(format!("Failed to connect to SQLite: {}", e)))?;
        
        // Create tables
        let storage = Self {
            pool,
            config: config.clone(),
        };
        
        storage.create_tables().await?;
        
        info!("Initialized SQLite storage at: {:?}", db_path);
        Ok(storage)
    }
    
    async fn create_tables(&self) -> Result<()> {
        // Create klines table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS klines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                open_time INTEGER NOT NULL,
                close_time INTEGER NOT NULL,
                open_price REAL NOT NULL,
                high_price REAL NOT NULL,
                low_price REAL NOT NULL,
                close_price REAL NOT NULL,
                volume REAL NOT NULL,
                quote_volume REAL NOT NULL,
                trades_count INTEGER NOT NULL,
                taker_buy_base_volume REAL NOT NULL,
                taker_buy_quote_volume REAL NOT NULL,
                interval_type TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, exchange, open_time, interval_type)
            )
        "#)
        .execute(&self.pool)
        .await
        .map_err(|e| DataCollectorError::Storage(format!("Failed to create klines table: {}", e)))?;
        
        // Create trades table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                trade_id INTEGER NOT NULL,
                price REAL NOT NULL,
                quantity REAL NOT NULL,
                quote_quantity REAL NOT NULL,
                timestamp INTEGER NOT NULL,
                is_buyer_maker BOOLEAN NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(exchange, trade_id)
            )
        "#)
        .execute(&self.pool)
        .await
        .map_err(|e| DataCollectorError::Storage(format!("Failed to create trades table: {}", e)))?;
        
        // Create order_books table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS order_books (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                bids TEXT NOT NULL,  -- JSON array
                asks TEXT NOT NULL,  -- JSON array
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        "#)
        .execute(&self.pool)
        .await
        .map_err(|e| DataCollectorError::Storage(format!("Failed to create order_books table: {}", e)))?;
        
        // Create tickers table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS tickers_24hr (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                price_change REAL NOT NULL,
                price_change_percent REAL NOT NULL,
                weighted_avg_price REAL NOT NULL,
                prev_close_price REAL NOT NULL,
                last_price REAL NOT NULL,
                last_qty REAL NOT NULL,
                bid_price REAL NOT NULL,
                bid_qty REAL NOT NULL,
                ask_price REAL NOT NULL,
                ask_qty REAL NOT NULL,
                open_price REAL NOT NULL,
                high_price REAL NOT NULL,
                low_price REAL NOT NULL,
                volume REAL NOT NULL,
                quote_volume REAL NOT NULL,
                open_time INTEGER NOT NULL,
                close_time INTEGER NOT NULL,
                first_id INTEGER NOT NULL,
                last_id INTEGER NOT NULL,
                count INTEGER NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        "#)
        .execute(&self.pool)
        .await
        .map_err(|e| DataCollectorError::Storage(format!("Failed to create tickers table: {}", e)))?;
        
        // Create funding_rates table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS funding_rates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                funding_rate REAL NOT NULL,
                funding_time INTEGER NOT NULL,
                mark_price REAL NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, exchange, funding_time)
            )
        "#)
        .execute(&self.pool)
        .await
        .map_err(|e| DataCollectorError::Storage(format!("Failed to create funding_rates table: {}", e)))?;
        
        // Create indexes for better performance
        let indexes = [
            "CREATE INDEX IF NOT EXISTS idx_klines_symbol_time ON klines(symbol, open_time)",
            "CREATE INDEX IF NOT EXISTS idx_klines_exchange ON klines(exchange)",
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_trades_exchange ON trades(exchange)",
            "CREATE INDEX IF NOT EXISTS idx_tickers_symbol ON tickers_24hr(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_funding_symbol_time ON funding_rates(symbol, funding_time)",
        ];
        
        for index_sql in &indexes {
            sqlx::query(index_sql)
                .execute(&self.pool)
                .await
                .map_err(|e| DataCollectorError::Storage(format!("Failed to create index: {}", e)))?;
        }
        
        debug!("Created SQLite tables and indexes");
        Ok(())
    }
}

#[async_trait::async_trait]
impl Storage for SqliteStorage {
    async fn store_klines(&self, klines: &[Kline]) -> Result<()> {
        if klines.is_empty() {
            return Ok(());
        }
        
        let mut tx = self.pool.begin()
            .await
            .map_err(|e| DataCollectorError::Storage(format!("Failed to begin transaction: {}", e)))?;
        
        for kline in klines {
            sqlx::query(r#"
                INSERT OR REPLACE INTO klines (
                    symbol, exchange, open_time, close_time, open_price, high_price,
                    low_price, close_price, volume, quote_volume, trades_count,
                    taker_buy_base_volume, taker_buy_quote_volume, interval_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#)
            .bind(&kline.symbol)
            .bind(&kline.exchange)
            .bind(kline.open_time.timestamp_millis())
            .bind(kline.close_time.timestamp_millis())
            .bind(kline.open)
            .bind(kline.high)
            .bind(kline.low)
            .bind(kline.close)
            .bind(kline.volume)
            .bind(kline.quote_volume)
            .bind(kline.trades_count as i64)
            .bind(kline.taker_buy_base_volume)
            .bind(kline.taker_buy_quote_volume)
            .bind(format!("{:?}", kline.interval))
            .execute(&mut *tx)
            .await
            .map_err(|e| DataCollectorError::Storage(format!("Failed to insert kline: {}", e)))?;
        }
        
        tx.commit()
            .await
            .map_err(|e| DataCollectorError::Storage(format!("Failed to commit transaction: {}", e)))?;
        
        debug!("Stored {} klines to SQLite", klines.len());
        Ok(())
    }
    
    async fn store_trades(&self, trades: &[Trade]) -> Result<()> {
        if trades.is_empty() {
            return Ok(());
        }
        
        let mut tx = self.pool.begin()
            .await
            .map_err(|e| DataCollectorError::Storage(format!("Failed to begin transaction: {}", e)))?;
        
        for trade in trades {
            sqlx::query(r#"
                INSERT OR REPLACE INTO trades (
                    symbol, exchange, trade_id, price, quantity, quote_quantity,
                    timestamp, is_buyer_maker
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            "#)
            .bind(&trade.symbol)
            .bind(&trade.exchange)
            .bind(trade.trade_id as i64)
            .bind(trade.price)
            .bind(trade.quantity)
            .bind(trade.quote_quantity)
            .bind(trade.timestamp.timestamp_millis())
            .bind(trade.is_buyer_maker)
            .execute(&mut *tx)
            .await
            .map_err(|e| DataCollectorError::Storage(format!("Failed to insert trade: {}", e)))?;
        }
        
        tx.commit()
            .await
            .map_err(|e| DataCollectorError::Storage(format!("Failed to commit transaction: {}", e)))?;
        
        debug!("Stored {} trades to SQLite", trades.len());
        Ok(())
    }
    
    async fn store_order_books(&self, order_books: &[OrderBook]) -> Result<()> {
        debug!("Storing {} order books to SQLite (placeholder)", order_books.len());
        Ok(())
    }
    
    async fn store_tickers(&self, tickers: &[Ticker24hr]) -> Result<()> {
        debug!("Storing {} tickers to SQLite (placeholder)", tickers.len());
        Ok(())
    }
    
    async fn store_funding_rates(&self, funding_rates: &[FundingRate]) -> Result<()> {
        debug!("Storing {} funding rates to SQLite (placeholder)", funding_rates.len());
        Ok(())
    }
    
    async fn validate(&self) -> Result<bool> {
        match sqlx::query("SELECT 1").fetch_one(&self.pool).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }
    
    async fn get_stats(&self) -> Result<StorageStats> {
        let klines_count: i64 = sqlx::query("SELECT COUNT(*) as count FROM klines")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| DataCollectorError::Storage(format!("Failed to count klines: {}", e)))?
            .get("count");
            
        let trades_count: i64 = sqlx::query("SELECT COUNT(*) as count FROM trades")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| DataCollectorError::Storage(format!("Failed to count trades: {}", e)))?
            .get("count");
            
        let order_books_count: i64 = sqlx::query("SELECT COUNT(*) as count FROM order_books")
            .fetch_one(&self.pool)
            .await
            .unwrap_or_else(|_| sqlx::Row::from(sqlx::sqlite::SqliteRow::from(vec![])))
            .try_get("count")
            .unwrap_or(0);
            
        let tickers_count: i64 = sqlx::query("SELECT COUNT(*) as count FROM tickers_24hr")
            .fetch_one(&self.pool)
            .await
            .unwrap_or_else(|_| sqlx::Row::from(sqlx::sqlite::SqliteRow::from(vec![])))
            .try_get("count")
            .unwrap_or(0);
            
        let funding_rates_count: i64 = sqlx::query("SELECT COUNT(*) as count FROM funding_rates")
            .fetch_one(&self.pool)
            .await
            .unwrap_or_else(|_| sqlx::Row::from(sqlx::sqlite::SqliteRow::from(vec![])))
            .try_get("count")
            .unwrap_or(0);
        
        Ok(StorageStats {
            total_klines: klines_count as u64,
            total_trades: trades_count as u64,
            total_order_books: order_books_count as u64,
            total_tickers: tickers_count as u64,
            total_funding_rates: funding_rates_count as u64,
            storage_size_bytes: 0, // TODO: Get actual database file size
            last_updated: chrono::Utc::now(),
        })
    }
    
    async fn backup(&self, backup_path: &Path) -> Result<()> {
        let backup_target = backup_path.join("crypto_data.db.backup");
        let db_path = self.config.base_directory.join("crypto_data.db");
        
        std::fs::copy(&db_path, &backup_target)
            .map_err(|e| DataCollectorError::Storage(format!("Failed to backup database: {}", e)))?;
        
        info!("SQLite database backed up to: {:?}", backup_target);
        Ok(())
    }
    
    async fn restore(&self, backup_path: &Path) -> Result<()> {
        let backup_source = backup_path.join("crypto_data.db.backup");
        let db_path = self.config.base_directory.join("crypto_data.db");
        
        if !backup_source.exists() {
            return Err(DataCollectorError::Storage("Backup source not found".to_string()));
        }
        
        std::fs::copy(&backup_source, &db_path)
            .map_err(|e| DataCollectorError::Storage(format!("Failed to restore database: {}", e)))?;
        
        info!("SQLite database restored from: {:?}", backup_source);
        Ok(())
    }
}