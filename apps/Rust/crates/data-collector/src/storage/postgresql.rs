//! PostgreSQL storage backend for cryptocurrency data

use super::{Storage, StorageStats};
use crate::{Result, DataCollectorError};
use crate::config::StorageConfig;
use crate::types::*;
use std::path::Path;
use tracing::{info, debug};
use sqlx::{postgres::{PgPool, PgPoolOptions}, Row};

/// PostgreSQL storage backend
pub struct PostgresqlStorage {
    pool: PgPool,
    config: StorageConfig,
}

impl PostgresqlStorage {
    pub async fn new(config: &StorageConfig) -> Result<Self> {
        let database_url = config.database_url
            .as_ref()
            .ok_or_else(|| DataCollectorError::Config("PostgreSQL database URL not configured".to_string()))?;
        
        let pool = PgPoolOptions::new()
            .max_connections(20)
            .connect(database_url)
            .await
            .map_err(|e| DataCollectorError::Storage(format!("Failed to connect to PostgreSQL: {}", e)))?;
        
        let storage = Self {
            pool,
            config: config.clone(),
        };
        
        storage.create_tables().await?;
        
        info!("Initialized PostgreSQL storage");
        Ok(storage)
    }
    
    async fn create_tables(&self) -> Result<()> {
        // Create klines table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS klines (
                id BIGSERIAL PRIMARY KEY,
                symbol VARCHAR(50) NOT NULL,
                exchange VARCHAR(50) NOT NULL,
                open_time BIGINT NOT NULL,
                close_time BIGINT NOT NULL,
                open_price DOUBLE PRECISION NOT NULL,
                high_price DOUBLE PRECISION NOT NULL,
                low_price DOUBLE PRECISION NOT NULL,
                close_price DOUBLE PRECISION NOT NULL,
                volume DOUBLE PRECISION NOT NULL,
                quote_volume DOUBLE PRECISION NOT NULL,
                trades_count BIGINT NOT NULL,
                taker_buy_base_volume DOUBLE PRECISION NOT NULL,
                taker_buy_quote_volume DOUBLE PRECISION NOT NULL,
                interval_type VARCHAR(20) NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                CONSTRAINT unique_kline UNIQUE(symbol, exchange, open_time, interval_type)
            )
        "#)
        .execute(&self.pool)
        .await
        .map_err(|e| DataCollectorError::Storage(format!("Failed to create klines table: {}", e)))?;
        
        // Create trades table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS trades (
                id BIGSERIAL PRIMARY KEY,
                symbol VARCHAR(50) NOT NULL,
                exchange VARCHAR(50) NOT NULL,
                trade_id BIGINT NOT NULL,
                price DOUBLE PRECISION NOT NULL,
                quantity DOUBLE PRECISION NOT NULL,
                quote_quantity DOUBLE PRECISION NOT NULL,
                timestamp BIGINT NOT NULL,
                is_buyer_maker BOOLEAN NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                CONSTRAINT unique_trade UNIQUE(exchange, trade_id)
            )
        "#)
        .execute(&self.pool)
        .await
        .map_err(|e| DataCollectorError::Storage(format!("Failed to create trades table: {}", e)))?;
        
        // Create order_books table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS order_books (
                id BIGSERIAL PRIMARY KEY,
                symbol VARCHAR(50) NOT NULL,
                exchange VARCHAR(50) NOT NULL,
                timestamp BIGINT NOT NULL,
                bids JSONB NOT NULL,
                asks JSONB NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        "#)
        .execute(&self.pool)
        .await
        .map_err(|e| DataCollectorError::Storage(format!("Failed to create order_books table: {}", e)))?;
        
        // Create tickers table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS tickers_24hr (
                id BIGSERIAL PRIMARY KEY,
                symbol VARCHAR(50) NOT NULL,
                exchange VARCHAR(50) NOT NULL,
                price_change DOUBLE PRECISION NOT NULL,
                price_change_percent DOUBLE PRECISION NOT NULL,
                weighted_avg_price DOUBLE PRECISION NOT NULL,
                prev_close_price DOUBLE PRECISION NOT NULL,
                last_price DOUBLE PRECISION NOT NULL,
                last_qty DOUBLE PRECISION NOT NULL,
                bid_price DOUBLE PRECISION NOT NULL,
                bid_qty DOUBLE PRECISION NOT NULL,
                ask_price DOUBLE PRECISION NOT NULL,
                ask_qty DOUBLE PRECISION NOT NULL,
                open_price DOUBLE PRECISION NOT NULL,
                high_price DOUBLE PRECISION NOT NULL,
                low_price DOUBLE PRECISION NOT NULL,
                volume DOUBLE PRECISION NOT NULL,
                quote_volume DOUBLE PRECISION NOT NULL,
                open_time BIGINT NOT NULL,
                close_time BIGINT NOT NULL,
                first_id BIGINT NOT NULL,
                last_id BIGINT NOT NULL,
                count BIGINT NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        "#)
        .execute(&self.pool)
        .await
        .map_err(|e| DataCollectorError::Storage(format!("Failed to create tickers table: {}", e)))?;
        
        // Create funding_rates table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS funding_rates (
                id BIGSERIAL PRIMARY KEY,
                symbol VARCHAR(50) NOT NULL,
                exchange VARCHAR(50) NOT NULL,
                funding_rate DOUBLE PRECISION NOT NULL,
                funding_time BIGINT NOT NULL,
                mark_price DOUBLE PRECISION NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                CONSTRAINT unique_funding_rate UNIQUE(symbol, exchange, funding_time)
            )
        "#)
        .execute(&self.pool)
        .await
        .map_err(|e| DataCollectorError::Storage(format!("Failed to create funding_rates table: {}", e)))?;
        
        // Create indexes for better performance
        let indexes = [
            "CREATE INDEX IF NOT EXISTS idx_klines_symbol_time ON klines(symbol, open_time)",
            "CREATE INDEX IF NOT EXISTS idx_klines_exchange ON klines(exchange)",
            "CREATE INDEX IF NOT EXISTS idx_klines_created_at ON klines(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_trades_exchange ON trades(exchange)",
            "CREATE INDEX IF NOT EXISTS idx_trades_created_at ON trades(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_order_books_symbol ON order_books(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_tickers_symbol ON tickers_24hr(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_funding_symbol_time ON funding_rates(symbol, funding_time)",
        ];
        
        for index_sql in &indexes {
            sqlx::query(index_sql)
                .execute(&self.pool)
                .await
                .map_err(|e| DataCollectorError::Storage(format!("Failed to create index: {}", e)))?;
        }
        
        debug!("Created PostgreSQL tables and indexes");
        Ok(())
    }
}

#[async_trait::async_trait]
impl Storage for PostgresqlStorage {
    async fn store_klines(&self, klines: &[Kline]) -> Result<()> {
        if klines.is_empty() {
            return Ok(());
        }
        
        let mut tx = self.pool.begin()
            .await
            .map_err(|e| DataCollectorError::Storage(format!("Failed to begin transaction: {}", e)))?;
        
        for kline in klines {
            sqlx::query(r#"
                INSERT INTO klines (
                    symbol, exchange, open_time, close_time, open_price, high_price,
                    low_price, close_price, volume, quote_volume, trades_count,
                    taker_buy_base_volume, taker_buy_quote_volume, interval_type
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                ON CONFLICT (symbol, exchange, open_time, interval_type) DO UPDATE SET
                    close_time = EXCLUDED.close_time,
                    open_price = EXCLUDED.open_price,
                    high_price = EXCLUDED.high_price,
                    low_price = EXCLUDED.low_price,
                    close_price = EXCLUDED.close_price,
                    volume = EXCLUDED.volume,
                    quote_volume = EXCLUDED.quote_volume,
                    trades_count = EXCLUDED.trades_count,
                    taker_buy_base_volume = EXCLUDED.taker_buy_base_volume,
                    taker_buy_quote_volume = EXCLUDED.taker_buy_quote_volume
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
        
        debug!("Stored {} klines to PostgreSQL", klines.len());
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
                INSERT INTO trades (
                    symbol, exchange, trade_id, price, quantity, quote_quantity,
                    timestamp, is_buyer_maker
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (exchange, trade_id) DO UPDATE SET
                    symbol = EXCLUDED.symbol,
                    price = EXCLUDED.price,
                    quantity = EXCLUDED.quantity,
                    quote_quantity = EXCLUDED.quote_quantity,
                    timestamp = EXCLUDED.timestamp,
                    is_buyer_maker = EXCLUDED.is_buyer_maker
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
        
        debug!("Stored {} trades to PostgreSQL", trades.len());
        Ok(())
    }
    
    async fn store_order_books(&self, order_books: &[OrderBook]) -> Result<()> {
        debug!("Storing {} order books to PostgreSQL (placeholder)", order_books.len());
        Ok(())
    }
    
    async fn store_tickers(&self, tickers: &[Ticker24hr]) -> Result<()> {
        debug!("Storing {} tickers to PostgreSQL (placeholder)", tickers.len());
        Ok(())
    }
    
    async fn store_funding_rates(&self, funding_rates: &[FundingRate]) -> Result<()> {
        debug!("Storing {} funding rates to PostgreSQL (placeholder)", funding_rates.len());
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
            .unwrap_or_else(|_| {
                use sqlx::postgres::PgRow;
                PgRow::from(vec![])
            })
            .try_get("count")
            .unwrap_or(0);
            
        let tickers_count: i64 = sqlx::query("SELECT COUNT(*) as count FROM tickers_24hr")
            .fetch_one(&self.pool)
            .await
            .unwrap_or_else(|_| {
                use sqlx::postgres::PgRow;
                PgRow::from(vec![])
            })
            .try_get("count")
            .unwrap_or(0);
            
        let funding_rates_count: i64 = sqlx::query("SELECT COUNT(*) as count FROM funding_rates")
            .fetch_one(&self.pool)
            .await
            .unwrap_or_else(|_| {
                use sqlx::postgres::PgRow;
                PgRow::from(vec![])
            })
            .try_get("count")
            .unwrap_or(0);
        
        // Get database size
        let db_size: i64 = sqlx::query("SELECT pg_database_size(current_database()) as size")
            .fetch_one(&self.pool)
            .await
            .map(|row| row.get("size"))
            .unwrap_or(0);
        
        Ok(StorageStats {
            total_klines: klines_count as u64,
            total_trades: trades_count as u64,
            total_order_books: order_books_count as u64,
            total_tickers: tickers_count as u64,
            total_funding_rates: funding_rates_count as u64,
            storage_size_bytes: db_size as u64,
            last_updated: chrono::Utc::now(),
        })
    }
    
    async fn backup(&self, backup_path: &Path) -> Result<()> {
        // PostgreSQL backup would typically use pg_dump
        // For now, we'll create a simple data export
        let backup_file = backup_path.join("postgresql_backup.sql");
        
        info!("PostgreSQL backup functionality not fully implemented");
        info!("Backup would be created at: {:?}", backup_file);
        
        // TODO: Implement pg_dump or custom backup logic
        Ok(())
    }
    
    async fn restore(&self, backup_path: &Path) -> Result<()> {
        let backup_file = backup_path.join("postgresql_backup.sql");
        
        info!("PostgreSQL restore functionality not fully implemented");
        info!("Restore would be from: {:?}", backup_file);
        
        // TODO: Implement pg_restore or custom restore logic
        Ok(())
    }
}