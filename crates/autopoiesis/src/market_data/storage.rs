//! Market data storage implementation

use crate::prelude::*;
use crate::models::MarketData;
use chrono::{DateTime, Utc, Duration};
use rust_decimal::Decimal;
use sqlx::{PgPool, Row};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Market data storage that handles persistence and retrieval of market data
#[derive(Debug)]
pub struct DataStorage {
    /// Database connection pool
    pool: PgPool,
    
    /// Storage configuration
    config: DataStorageConfig,
    
    /// In-memory cache for recent data
    cache: Arc<RwLock<HashMap<String, CachedData>>>,
    
    /// Storage metrics
    metrics: Arc<RwLock<StorageMetrics>>,
}

#[derive(Debug, Clone)]
pub struct DataStorageConfig {
    /// Database connection string
    pub database_url: String,
    
    /// Maximum connections in pool
    pub max_connections: u32,
    
    /// Cache configuration
    pub cache_config: CacheConfig,
    
    /// Retention policy
    pub retention_policy: RetentionPolicy,
    
    /// Partitioning strategy
    pub partitioning: PartitioningConfig,
    
    /// Compression settings
    pub compression: CompressionConfig,
}

#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum items in cache per symbol
    pub max_items_per_symbol: usize,
    
    /// Cache TTL in seconds
    pub ttl_seconds: u64,
    
    /// Enable write-through caching
    pub write_through: bool,
    
    /// Enable read-through caching
    pub read_through: bool,
}

#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Raw data retention period
    pub raw_data_days: u32,
    
    /// Minute aggregates retention period
    pub minute_aggregates_days: u32,
    
    /// Hour aggregates retention period
    pub hour_aggregates_days: u32,
    
    /// Daily aggregates retention period
    pub daily_aggregates_days: u32,
    
    /// Auto-cleanup enabled
    pub auto_cleanup: bool,
}

#[derive(Debug, Clone)]
pub struct PartitioningConfig {
    /// Partition by time interval
    pub time_interval: PartitionInterval,
    
    /// Partition by symbol
    pub partition_by_symbol: bool,
    
    /// Maximum partition size in MB
    pub max_partition_size_mb: u64,
}

#[derive(Debug, Clone)]
pub enum PartitionInterval {
    Daily,
    Weekly,
    Monthly,
    Yearly,
}

#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Enable compression
    pub enabled: bool,
    
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    
    /// Compression level (1-9)
    pub level: u8,
    
    /// Minimum age for compression in hours
    pub min_age_hours: u32,
}

#[derive(Debug, Clone)]
pub enum CompressionAlgorithm {
    Gzip,
    Lz4,
    Zstd,
}

#[derive(Debug, Clone)]
struct CachedData {
    data: Vec<MarketData>,
    last_updated: DateTime<Utc>,
    access_count: u64,
}

#[derive(Debug, Clone, Default)]
struct StorageMetrics {
    total_writes: u64,
    total_reads: u64,
    cache_hits: u64,
    cache_misses: u64,
    database_errors: u64,
    average_write_time_ms: f64,
    average_read_time_ms: f64,
    storage_size_mb: f64,
}

impl Default for DataStorageConfig {
    fn default() -> Self {
        Self {
            database_url: std::env::var("DATABASE_URL")
                .unwrap_or_else(|_| {
                    eprintln!("SECURITY ERROR: DATABASE_URL environment variable is required");
                    eprintln!("Set DATABASE_URL=postgresql://username:password@host:port/database");
                    std::process::exit(1);
                }),
            max_connections: 10,
            cache_config: CacheConfig {
                max_items_per_symbol: 1000,
                ttl_seconds: 300,
                write_through: true,
                read_through: true,
            },
            retention_policy: RetentionPolicy {
                raw_data_days: 30,
                minute_aggregates_days: 90,
                hour_aggregates_days: 365,
                daily_aggregates_days: 1825, // 5 years
                auto_cleanup: true,
            },
            partitioning: PartitioningConfig {
                time_interval: PartitionInterval::Daily,
                partition_by_symbol: true,
                max_partition_size_mb: 1024,
            },
            compression: CompressionConfig {
                enabled: true,
                algorithm: CompressionAlgorithm::Zstd,
                level: 3,
                min_age_hours: 24,
            },
        }
    }
}

impl DataStorage {
    /// Create a new data storage instance
    pub async fn new(config: DataStorageConfig) -> Result<Self> {
        // Create database connection pool
        let pool = PgPool::connect(&config.database_url).await?;
        
        let storage = Self {
            pool,
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(StorageMetrics::default())),
        };

        // Initialize database schema
        storage.initialize_schema().await?;
        
        // Start background tasks
        storage.start_background_tasks().await?;

        Ok(storage)
    }

    /// Store market data
    pub async fn store(&self, market_data: MarketData) -> Result<()> {
        let start_time = std::time::Instant::now();
        
        // Update cache if write-through is enabled
        if self.config.cache_config.write_through {
            self.update_cache(&market_data).await;
        }

        // Store in database
        let result = self.store_in_database(&market_data).await;
        
        // Update metrics
        let write_time = start_time.elapsed().as_millis() as f64;
        self.update_write_metrics(result.is_ok(), write_time).await;

        result
    }

    /// Store multiple market data points
    pub async fn store_batch(&self, market_data: Vec<MarketData>) -> Result<()> {
        let start_time = std::time::Instant::now();
        
        // Update cache for each item if write-through is enabled
        if self.config.cache_config.write_through {
            for data in &market_data {
                self.update_cache(data).await;
            }
        }

        // Store in database as batch
        let result = self.store_batch_in_database(&market_data).await;
        
        // Update metrics
        let write_time = start_time.elapsed().as_millis() as f64;
        self.update_write_metrics(result.is_ok(), write_time).await;

        result
    }

    /// Retrieve market data for a symbol within a time range
    pub async fn retrieve(
        &self,
        symbol: &str,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<Vec<MarketData>> {
        let cache_start_time = std::time::Instant::now();
        
        // Check cache first if read-through is enabled
        if self.config.cache_config.read_through {
            if let Some(cached_data) = self.get_from_cache(symbol, start_time, end_time).await {
                let read_time = cache_start_time.elapsed().as_millis() as f64;
                self.update_read_metrics(true, true, read_time).await;
                return Ok(cached_data);
            }
        }

        // Retrieve from database
        let db_start_time = std::time::Instant::now();
        let result = self.retrieve_from_database(symbol, start_time, end_time).await;
        let read_time = db_start_time.elapsed().as_millis() as f64;
        
        match &result {
            Ok(data) => {
                // Update cache with retrieved data
                if self.config.cache_config.read_through && !data.is_empty() {
                    self.cache_retrieved_data(symbol, data.clone()).await;
                }
                self.update_read_metrics(true, false, read_time).await;
            }
            Err(_) => {
                self.update_read_metrics(false, false, read_time).await;
            }
        }

        result
    }

    /// Get latest market data for a symbol
    pub async fn get_latest(&self, symbol: &str) -> Result<Option<MarketData>> {
        // Check cache first
        if let Some(cached) = self.get_latest_from_cache(symbol).await {
            return Ok(Some(cached));
        }

        // Retrieve from database
        self.get_latest_from_database(symbol).await
    }

    /// Get market data aggregated by time intervals
    pub async fn get_aggregated(
        &self,
        symbol: &str,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
        interval: AggregationInterval,
    ) -> Result<Vec<AggregatedMarketData>> {
        match interval {
            AggregationInterval::Minute => {
                self.get_minute_aggregates(symbol, start_time, end_time).await
            }
            AggregationInterval::Hour => {
                self.get_hour_aggregates(symbol, start_time, end_time).await
            }
            AggregationInterval::Day => {
                self.get_daily_aggregates(symbol, start_time, end_time).await
            }
        }
    }

    /// Get storage metrics
    pub async fn metrics(&self) -> StorageMetrics {
        self.metrics.read().await.clone()
    }

    /// Perform database maintenance
    pub async fn maintenance(&self) -> Result<MaintenanceReport> {
        let mut report = MaintenanceReport::default();

        // Run cleanup if enabled
        if self.config.retention_policy.auto_cleanup {
            report.cleanup_stats = Some(self.cleanup_old_data().await?);
        }

        // Compress old data if enabled
        if self.config.compression.enabled {
            report.compression_stats = Some(self.compress_old_data().await?);
        }

        // Update partition statistics
        report.partition_stats = Some(self.update_partition_stats().await?);

        // Vacuum and analyze tables
        report.vacuum_stats = Some(self.vacuum_analyze().await?);

        Ok(report)
    }

    async fn initialize_schema(&self) -> Result<()> {
        // Create main market data table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS market_data (
                id BIGSERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                bid DECIMAL(20,8) NOT NULL,
                ask DECIMAL(20,8) NOT NULL,
                mid DECIMAL(20,8) NOT NULL,
                last DECIMAL(20,8) NOT NULL,
                volume_24h DECIMAL(20,8) NOT NULL,
                bid_size DECIMAL(20,8) NOT NULL,
                ask_size DECIMAL(20,8) NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        "#)
        .execute(&self.pool)
        .await?;

        // Create indexes
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data (symbol, timestamp)")
            .execute(&self.pool)
            .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data (timestamp)")
            .execute(&self.pool)
            .await?;

        // Create aggregated tables
        self.create_aggregate_tables().await?;

        Ok(())
    }

    async fn create_aggregate_tables(&self) -> Result<()> {
        // Minute aggregates
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS market_data_1m (
                symbol VARCHAR(20) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                open DECIMAL(20,8) NOT NULL,
                high DECIMAL(20,8) NOT NULL,
                low DECIMAL(20,8) NOT NULL,
                close DECIMAL(20,8) NOT NULL,
                volume DECIMAL(20,8) NOT NULL,
                vwap DECIMAL(20,8) NOT NULL,
                count INTEGER NOT NULL,
                PRIMARY KEY (symbol, timestamp)
            )
        "#)
        .execute(&self.pool)
        .await?;

        // Hour aggregates
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS market_data_1h (
                symbol VARCHAR(20) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                open DECIMAL(20,8) NOT NULL,
                high DECIMAL(20,8) NOT NULL,
                low DECIMAL(20,8) NOT NULL,
                close DECIMAL(20,8) NOT NULL,
                volume DECIMAL(20,8) NOT NULL,
                vwap DECIMAL(20,8) NOT NULL,
                count INTEGER NOT NULL,
                PRIMARY KEY (symbol, timestamp)
            )
        "#)
        .execute(&self.pool)
        .await?;

        // Daily aggregates
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS market_data_1d (
                symbol VARCHAR(20) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                open DECIMAL(20,8) NOT NULL,
                high DECIMAL(20,8) NOT NULL,
                low DECIMAL(20,8) NOT NULL,
                close DECIMAL(20,8) NOT NULL,
                volume DECIMAL(20,8) NOT NULL,
                vwap DECIMAL(20,8) NOT NULL,
                count INTEGER NOT NULL,
                PRIMARY KEY (symbol, timestamp)
            )
        "#)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    async fn start_background_tasks(&self) -> Result<()> {
        // Start cache cleanup task
        let cache = self.cache.clone();
        let ttl_seconds = self.config.cache_config.ttl_seconds;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(60));
            loop {
                interval.tick().await;
                Self::cleanup_cache(&cache, ttl_seconds).await;
            }
        });

        // Start aggregation task
        let pool = self.pool.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(300)); // Every 5 minutes
            loop {
                interval.tick().await;
                if let Err(e) = Self::generate_aggregates(&pool).await {
                    error!("Failed to generate aggregates: {}", e);
                }
            }
        });

        Ok(())
    }

    async fn store_in_database(&self, market_data: &MarketData) -> Result<()> {
        sqlx::query(r#"
            INSERT INTO market_data (symbol, timestamp, bid, ask, mid, last, volume_24h, bid_size, ask_size)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        "#)
        .bind(&market_data.symbol)
        .bind(market_data.timestamp)
        .bind(market_data.bid)
        .bind(market_data.ask)
        .bind(market_data.mid)
        .bind(market_data.last)
        .bind(market_data.volume_24h)
        .bind(market_data.bid_size)
        .bind(market_data.ask_size)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    async fn store_batch_in_database(&self, market_data: &[MarketData]) -> Result<()> {
        let mut tx = self.pool.begin().await?;

        for data in market_data {
            sqlx::query(r#"
                INSERT INTO market_data (symbol, timestamp, bid, ask, mid, last, volume_24h, bid_size, ask_size)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            "#)
            .bind(&data.symbol)
            .bind(data.timestamp)
            .bind(data.bid)
            .bind(data.ask)
            .bind(data.mid)
            .bind(data.last)
            .bind(data.volume_24h)
            .bind(data.bid_size)
            .bind(data.ask_size)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        Ok(())
    }

    async fn retrieve_from_database(
        &self,
        symbol: &str,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<Vec<MarketData>> {
        let rows = sqlx::query(r#"
            SELECT symbol, timestamp, bid, ask, mid, last, volume_24h, bid_size, ask_size
            FROM market_data
            WHERE symbol = $1 AND timestamp >= $2 AND timestamp <= $3
            ORDER BY timestamp ASC
        "#)
        .bind(symbol)
        .bind(start_time)
        .bind(end_time)
        .fetch_all(&self.pool)
        .await?;

        let mut result = Vec::with_capacity(rows.len());
        for row in rows {
            result.push(MarketData {
                symbol: row.get("symbol"),
                timestamp: row.get("timestamp"),
                bid: row.get("bid"),
                ask: row.get("ask"),
                mid: row.get("mid"),
                last: row.get("last"),
                volume_24h: row.get("volume_24h"),
                bid_size: row.get("bid_size"),
                ask_size: row.get("ask_size"),
            });
        }

        Ok(result)
    }

    async fn get_latest_from_database(&self, symbol: &str) -> Result<Option<MarketData>> {
        let row = sqlx::query(r#"
            SELECT symbol, timestamp, bid, ask, mid, last, volume_24h, bid_size, ask_size
            FROM market_data
            WHERE symbol = $1
            ORDER BY timestamp DESC
            LIMIT 1
        "#)
        .bind(symbol)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            Ok(Some(MarketData {
                symbol: row.get("symbol"),
                timestamp: row.get("timestamp"),
                bid: row.get("bid"),
                ask: row.get("ask"),
                mid: row.get("mid"),
                last: row.get("last"),
                volume_24h: row.get("volume_24h"),
                bid_size: row.get("bid_size"),
                ask_size: row.get("ask_size"),
            }))
        } else {
            Ok(None)
        }
    }

    async fn update_cache(&self, market_data: &MarketData) {
        let mut cache = self.cache.write().await;
        let entry = cache.entry(market_data.symbol.clone()).or_insert_with(|| CachedData {
            data: Vec::new(),
            last_updated: Utc::now(),
            access_count: 0,
        });

        entry.data.push(market_data.clone());
        entry.last_updated = Utc::now();

        // Maintain cache size
        let max_items = self.config.cache_config.max_items_per_symbol;
        if entry.data.len() > max_items {
            entry.data.drain(0..entry.data.len() - max_items);
        }
    }

    async fn get_from_cache(
        &self,
        symbol: &str,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Option<Vec<MarketData>> {
        let mut cache = self.cache.write().await;
        
        if let Some(cached) = cache.get_mut(symbol) {
            // Check if cache is still valid
            let age = (Utc::now() - cached.last_updated).num_seconds() as u64;
            if age <= self.config.cache_config.ttl_seconds {
                cached.access_count += 1;
                
                // Filter data by time range
                let filtered: Vec<MarketData> = cached.data
                    .iter()
                    .filter(|d| d.timestamp >= start_time && d.timestamp <= end_time)
                    .cloned()
                    .collect();
                
                if !filtered.is_empty() {
                    return Some(filtered);
                }
            }
        }

        None
    }

    async fn get_latest_from_cache(&self, symbol: &str) -> Option<MarketData> {
        let cache = self.cache.read().await;
        
        if let Some(cached) = cache.get(symbol) {
            let age = (Utc::now() - cached.last_updated).num_seconds() as u64;
            if age <= self.config.cache_config.ttl_seconds {
                return cached.data.last().cloned();
            }
        }

        None
    }

    async fn cache_retrieved_data(&self, symbol: &str, data: Vec<MarketData>) {
        let mut cache = self.cache.write().await;
        cache.insert(symbol.to_string(), CachedData {
            data,
            last_updated: Utc::now(),
            access_count: 1,
        });
    }

    async fn cleanup_cache(cache: &RwLock<HashMap<String, CachedData>>, ttl_seconds: u64) {
        let mut cache = cache.write().await;
        let now = Utc::now();
        let ttl_duration = Duration::seconds(ttl_seconds as i64);
        
        cache.retain(|_, cached| now - cached.last_updated < ttl_duration);
    }

    async fn generate_aggregates(pool: &PgPool) -> Result<()> {
        // Generate minute aggregates
        sqlx::query(r#"
            INSERT INTO market_data_1m (symbol, timestamp, open, high, low, close, volume, vwap, count)
            SELECT 
                symbol,
                date_trunc('minute', timestamp) as timestamp,
                (array_agg(mid ORDER BY timestamp ASC))[1] as open,
                max(mid) as high,
                min(mid) as low,
                (array_agg(mid ORDER BY timestamp DESC))[1] as close,
                avg(volume_24h) as volume,
                sum(mid * volume_24h) / sum(volume_24h) as vwap,
                count(*) as count
            FROM market_data
            WHERE timestamp >= NOW() - INTERVAL '2 minutes'
                AND timestamp < date_trunc('minute', NOW())
            GROUP BY symbol, date_trunc('minute', timestamp)
            ON CONFLICT (symbol, timestamp) DO NOTHING
        "#)
        .execute(pool)
        .await?;

        Ok(())
    }

    async fn update_write_metrics(&self, success: bool, write_time_ms: f64) {
        let mut metrics = self.metrics.write().await;
        metrics.total_writes += 1;
        
        if !success {
            metrics.database_errors += 1;
        }
        
        // Update average write time (exponential moving average)
        metrics.average_write_time_ms = if metrics.total_writes == 1 {
            write_time_ms
        } else {
            0.9 * metrics.average_write_time_ms + 0.1 * write_time_ms
        };
    }

    async fn update_read_metrics(&self, success: bool, cache_hit: bool, read_time_ms: f64) {
        let mut metrics = self.metrics.write().await;
        metrics.total_reads += 1;
        
        if cache_hit {
            metrics.cache_hits += 1;
        } else {
            metrics.cache_misses += 1;
        }
        
        if !success {
            metrics.database_errors += 1;
        }
        
        // Update average read time (exponential moving average)
        metrics.average_read_time_ms = if metrics.total_reads == 1 {
            read_time_ms
        } else {
            0.9 * metrics.average_read_time_ms + 0.1 * read_time_ms
        };
    }

    async fn get_minute_aggregates(
        &self,
        symbol: &str,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<Vec<AggregatedMarketData>> {
        let rows = sqlx::query(r#"
            SELECT symbol, timestamp, open, high, low, close, volume, vwap, count
            FROM market_data_1m
            WHERE symbol = $1 AND timestamp >= $2 AND timestamp <= $3
            ORDER BY timestamp ASC
        "#)
        .bind(symbol)
        .bind(start_time)
        .bind(end_time)
        .fetch_all(&self.pool)
        .await?;

        let mut result = Vec::with_capacity(rows.len());
        for row in rows {
            result.push(AggregatedMarketData {
                symbol: row.get("symbol"),
                timestamp: row.get("timestamp"),
                open: row.get("open"),
                high: row.get("high"),
                low: row.get("low"),
                close: row.get("close"),
                volume: row.get("volume"),
                vwap: row.get("vwap"),
                count: row.get("count"),
            });
        }

        Ok(result)
    }

    async fn get_hour_aggregates(
        &self,
        symbol: &str,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<Vec<AggregatedMarketData>> {
        // Similar to minute aggregates but from 1h table
        let rows = sqlx::query(r#"
            SELECT symbol, timestamp, open, high, low, close, volume, vwap, count
            FROM market_data_1h
            WHERE symbol = $1 AND timestamp >= $2 AND timestamp <= $3
            ORDER BY timestamp ASC
        "#)
        .bind(symbol)
        .bind(start_time)
        .bind(end_time)
        .fetch_all(&self.pool)
        .await?;

        let mut result = Vec::with_capacity(rows.len());
        for row in rows {
            result.push(AggregatedMarketData {
                symbol: row.get("symbol"),
                timestamp: row.get("timestamp"),
                open: row.get("open"),
                high: row.get("high"),
                low: row.get("low"),
                close: row.get("close"),
                volume: row.get("volume"),
                vwap: row.get("vwap"),
                count: row.get("count"),
            });
        }

        Ok(result)
    }

    async fn get_daily_aggregates(
        &self,
        symbol: &str,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<Vec<AggregatedMarketData>> {
        // Similar to minute aggregates but from 1d table
        let rows = sqlx::query(r#"
            SELECT symbol, timestamp, open, high, low, close, volume, vwap, count
            FROM market_data_1d
            WHERE symbol = $1 AND timestamp >= $2 AND timestamp <= $3
            ORDER BY timestamp ASC
        "#)
        .bind(symbol)
        .bind(start_time)
        .bind(end_time)
        .fetch_all(&self.pool)
        .await?;

        let mut result = Vec::with_capacity(rows.len());
        for row in rows {
            result.push(AggregatedMarketData {
                symbol: row.get("symbol"),
                timestamp: row.get("timestamp"),
                open: row.get("open"),
                high: row.get("high"),
                low: row.get("low"),
                close: row.get("close"),
                volume: row.get("volume"),
                vwap: row.get("vwap"),
                count: row.get("count"),
            });
        }

        Ok(result)
    }

    async fn cleanup_old_data(&self) -> Result<CleanupStats> {
        let mut stats = CleanupStats::default();
        let policy = &self.config.retention_policy;

        // Cleanup raw data
        let raw_cutoff = Utc::now() - Duration::days(policy.raw_data_days as i64);
        let raw_result = sqlx::query("DELETE FROM market_data WHERE timestamp < $1")
            .bind(raw_cutoff)
            .execute(&self.pool)
            .await?;
        stats.raw_data_deleted = raw_result.rows_affected();

        // Cleanup aggregates
        let minute_cutoff = Utc::now() - Duration::days(policy.minute_aggregates_days as i64);
        let minute_result = sqlx::query("DELETE FROM market_data_1m WHERE timestamp < $1")
            .bind(minute_cutoff)
            .execute(&self.pool)
            .await?;
        stats.minute_aggregates_deleted = minute_result.rows_affected();

        Ok(stats)
    }

    async fn compress_old_data(&self) -> Result<CompressionStats> {
        // Placeholder for compression logic
        Ok(CompressionStats::default())
    }

    async fn update_partition_stats(&self) -> Result<PartitionStats> {
        // Placeholder for partition statistics
        Ok(PartitionStats::default())
    }

    async fn vacuum_analyze(&self) -> Result<VacuumStats> {
        // Run VACUUM ANALYZE on main tables
        sqlx::query("VACUUM ANALYZE market_data").execute(&self.pool).await?;
        sqlx::query("VACUUM ANALYZE market_data_1m").execute(&self.pool).await?;
        sqlx::query("VACUUM ANALYZE market_data_1h").execute(&self.pool).await?;
        sqlx::query("VACUUM ANALYZE market_data_1d").execute(&self.pool).await?;

        Ok(VacuumStats {
            tables_vacuumed: 4,
            space_reclaimed_mb: 0.0, // Would calculate in real implementation
        })
    }
}

#[derive(Debug, Clone)]
pub enum AggregationInterval {
    Minute,
    Hour,
    Day,
}

#[derive(Debug, Clone)]
pub struct AggregatedMarketData {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: Decimal,
    pub vwap: Decimal,
    pub count: i32,
}

#[derive(Debug, Clone, Default)]
pub struct MaintenanceReport {
    pub cleanup_stats: Option<CleanupStats>,
    pub compression_stats: Option<CompressionStats>,
    pub partition_stats: Option<PartitionStats>,
    pub vacuum_stats: Option<VacuumStats>,
}

#[derive(Debug, Clone, Default)]
pub struct CleanupStats {
    pub raw_data_deleted: u64,
    pub minute_aggregates_deleted: u64,
    pub hour_aggregates_deleted: u64,
    pub daily_aggregates_deleted: u64,
}

#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    pub tables_compressed: u32,
    pub space_saved_mb: f64,
    pub compression_ratio: f64,
}

#[derive(Debug, Clone, Default)]
pub struct PartitionStats {
    pub total_partitions: u32,
    pub average_partition_size_mb: f64,
    pub partitions_created: u32,
    pub partitions_dropped: u32,
}

#[derive(Debug, Clone, Default)]
pub struct VacuumStats {
    pub tables_vacuumed: u32,
    pub space_reclaimed_mb: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = DataStorageConfig::default();
        assert_eq!(config.max_connections, 10);
        assert!(config.retention_policy.auto_cleanup);
    }

    #[test]
    fn test_metrics_initialization() {
        let metrics = StorageMetrics::default();
        assert_eq!(metrics.total_writes, 0);
        assert_eq!(metrics.total_reads, 0);
    }
}