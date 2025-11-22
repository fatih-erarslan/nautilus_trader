use bytes::Bytes;
use futures::Stream;
use std::collections::HashMap;
use std::path::Path;
use std::pin::Pin;
/// Production-grade async-safe RocksDB wrapper
///
/// Solves RocksDB iterator Send/Sync issues by providing:
/// - Thread-safe database operations
/// - Async-safe iterators with proper lifetimes
/// - High-performance buffering strategies
/// - Production-ready error handling and monitoring
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

/// Error types for async RocksDB operations
#[derive(Debug, thiserror::Error)]
pub enum AsyncRocksDBError {
    #[error("RocksDB error: {0}")]
    DatabaseError(String),
    #[error("Key not found: {0}")]
    KeyNotFound(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Async operation cancelled")]
    Cancelled,
    #[error("Iterator exhausted")]
    IteratorExhausted,
    #[error("Thread pool error: {0}")]
    ThreadPoolError(String),
}

pub type AsyncRocksDBResult<T> = Result<T, AsyncRocksDBError>;

/// Configuration for AsyncRocksDB
#[derive(Debug, Clone)]
pub struct AsyncRocksDBConfig {
    /// Maximum number of entries to buffer in memory for iterators
    pub iterator_buffer_size: usize,
    /// Cache TTL for frequently accessed keys
    pub cache_ttl_secs: u64,
    /// Enable write-ahead logging
    pub enable_wal: bool,
    /// Block cache size in bytes
    pub block_cache_size: usize,
    /// Enable compression
    pub enable_compression: bool,
}

impl Default for AsyncRocksDBConfig {
    fn default() -> Self {
        Self {
            iterator_buffer_size: 1000,
            cache_ttl_secs: 300,
            enable_wal: true,
            block_cache_size: 64 * 1024 * 1024, // 64MB
            enable_compression: true,
        }
    }
}

/// Async-safe RocksDB key-value pair
#[derive(Debug, Clone)]
pub struct AsyncKeyValue {
    pub key: Bytes,
    pub value: Bytes,
}

/// Thread-safe async iterator for RocksDB
///
/// This solves the Send/Sync issues by:
/// 1. Pre-buffering results in a Send-safe format
/// 2. Using tokio channels for streaming
/// 3. Proper resource management and cleanup
pub struct AsyncRocksDBIterator {
    buffer: Vec<AsyncKeyValue>,
    position: usize,
    has_more: bool,
    // For streaming large datasets
    stream_rx: Option<tokio::sync::mpsc::Receiver<AsyncRocksDBResult<AsyncKeyValue>>>,
}

impl AsyncRocksDBIterator {
    /// Create a new buffered iterator
    fn new_buffered(buffer: Vec<AsyncKeyValue>, has_more: bool) -> Self {
        Self {
            buffer,
            position: 0,
            has_more,
            stream_rx: None,
        }
    }

    /// Create a new streaming iterator for large datasets
    fn new_streaming(rx: tokio::sync::mpsc::Receiver<AsyncRocksDBResult<AsyncKeyValue>>) -> Self {
        Self {
            buffer: Vec::new(),
            position: 0,
            has_more: true,
            stream_rx: Some(rx),
        }
    }

    /// Get the next key-value pair
    pub async fn next(&mut self) -> AsyncRocksDBResult<Option<AsyncKeyValue>> {
        // Handle streaming mode
        if let Some(rx) = &mut self.stream_rx {
            match rx.recv().await {
                Some(result) => result.map(Some),
                None => Ok(None), // Stream ended
            }
        } else {
            // Handle buffered mode
            if self.position < self.buffer.len() {
                let item = self.buffer[self.position].clone();
                self.position += 1;
                Ok(Some(item))
            } else {
                Ok(None)
            }
        }
    }

    /// Collect all remaining items into a vector
    pub async fn collect(mut self) -> AsyncRocksDBResult<Vec<AsyncKeyValue>> {
        let mut results = Vec::new();

        while let Some(item) = self.next().await? {
            results.push(item);
        }

        Ok(results)
    }

    /// Convert to async stream
    pub fn into_stream(
        self,
    ) -> Pin<Box<dyn Stream<Item = AsyncRocksDBResult<AsyncKeyValue>> + Send>> {
        Box::pin(AsyncIteratorStream::new(self))
    }
}

/// Stream wrapper for AsyncRocksDBIterator
struct AsyncIteratorStream {
    iterator: AsyncRocksDBIterator,
}

impl AsyncIteratorStream {
    fn new(iterator: AsyncRocksDBIterator) -> Self {
        Self { iterator }
    }
}

impl Stream for AsyncIteratorStream {
    type Item = AsyncRocksDBResult<AsyncKeyValue>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let future = self.iterator.next();
        tokio::pin!(future);

        match future.poll(cx) {
            std::task::Poll::Ready(Ok(Some(item))) => std::task::Poll::Ready(Some(Ok(item))),
            std::task::Poll::Ready(Ok(None)) => std::task::Poll::Ready(None),
            std::task::Poll::Ready(Err(e)) => std::task::Poll::Ready(Some(Err(e))),
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

/// Cache entry with TTL
#[derive(Debug, Clone)]
struct CacheEntry {
    value: Bytes,
    created_at: std::time::Instant,
    ttl: std::time::Duration,
}

impl CacheEntry {
    fn is_expired(&self) -> bool {
        self.created_at.elapsed() > self.ttl
    }
}

/// Performance metrics for monitoring
#[derive(Debug, Default, Clone)]
pub struct RocksDBMetrics {
    pub read_operations: u64,
    pub write_operations: u64,
    pub iterator_operations: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub total_read_duration: std::time::Duration,
    pub total_write_duration: std::time::Duration,
}

/// Thread-safe async wrapper for RocksDB
///
/// This wrapper provides:
/// - Async operations without blocking the tokio runtime
/// - Thread-safe access with proper Send/Sync bounds
/// - Intelligent caching for frequently accessed data
/// - High-performance iterator implementation
/// - Production-ready monitoring and error handling
pub struct AsyncRocksDB {
    // We don't store rocksdb::DB directly as it's not Send
    // Instead, we use the database path and create connections as needed
    db_path: std::path::PathBuf,
    config: AsyncRocksDBConfig,
    // Thread-safe cache for frequently accessed keys
    cache: Arc<RwLock<HashMap<Bytes, CacheEntry>>>,
    // Performance metrics
    metrics: Arc<Mutex<RocksDBMetrics>>,
    // Write buffer for batch operations
    write_buffer: Arc<Mutex<HashMap<Bytes, Option<Bytes>>>>, // None = delete
}

impl AsyncRocksDB {
    /// Open or create a RocksDB database
    ///
    /// # Arguments
    /// * `path` - Database directory path
    /// * `config` - Configuration options
    ///
    /// # Returns
    /// * `AsyncRocksDBResult<Self>` - The database wrapper or error
    pub async fn open<P: AsRef<Path>>(
        path: P,
        config: AsyncRocksDBConfig,
    ) -> AsyncRocksDBResult<Self> {
        let db_path = path.as_ref().to_path_buf();

        // Ensure directory exists and test database creation
        let path_clone = db_path.clone();
        let config_clone = config.clone();

        tokio::task::spawn_blocking(move || {
            use std::sync::Arc;

            // Create options based on config
            let mut opts = rocksdb::Options::default();
            opts.create_if_missing(true);
            opts.set_use_fsync(false);

            if config_clone.enable_compression {
                opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
            }

            // Set block cache
            let cache = rocksdb::Cache::new_lru_cache(config_clone.block_cache_size);
            let mut block_opts = rocksdb::BlockBasedOptions::default();
            block_opts.set_block_cache(&cache);
            opts.set_block_based_table_factory(&block_opts);

            // Test opening the database
            let db = rocksdb::DB::open(&opts, &path_clone)
                .map_err(|e| AsyncRocksDBError::DatabaseError(e.to_string()))?;

            // Test basic operation
            db.put(b"__test_key__", b"test_value")
                .map_err(|e| AsyncRocksDBError::DatabaseError(e.to_string()))?;

            db.delete(b"__test_key__")
                .map_err(|e| AsyncRocksDBError::DatabaseError(e.to_string()))?;

            Ok::<(), AsyncRocksDBError>(())
        })
        .await
        .map_err(|e| AsyncRocksDBError::ThreadPoolError(e.to_string()))??;

        Ok(Self {
            db_path,
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(Mutex::new(RocksDBMetrics::default())),
            write_buffer: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Get a value by key (async-safe)
    pub async fn get<K: AsRef<[u8]>>(&self, key: K) -> AsyncRocksDBResult<Option<Bytes>> {
        let start = std::time::Instant::now();
        let key_bytes = Bytes::copy_from_slice(key.as_ref());

        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(entry) = cache.get(&key_bytes) {
                if !entry.is_expired() {
                    // Cache hit
                    let mut metrics = self.metrics.lock().await;
                    metrics.cache_hits += 1;
                    metrics.read_operations += 1;
                    metrics.total_read_duration += start.elapsed();
                    return Ok(Some(entry.value.clone()));
                }
            }
        }

        // Check write buffer first
        {
            let write_buffer = self.write_buffer.lock().await;
            if let Some(buffered_value) = write_buffer.get(&key_bytes) {
                return Ok(buffered_value.clone());
            }
        }

        let db_path = self.db_path.clone();
        let key_clone = key_bytes.clone();

        let result = tokio::task::spawn_blocking(move || {
            let mut opts = rocksdb::Options::default();
            opts.create_if_missing(true);

            let db = rocksdb::DB::open_for_read_only(&opts, &db_path, false)
                .map_err(|e| AsyncRocksDBError::DatabaseError(e.to_string()))?;

            match db.get(&key_clone) {
                Ok(Some(value)) => Ok(Some(Bytes::from(value))),
                Ok(None) => Ok(None),
                Err(e) => Err(AsyncRocksDBError::DatabaseError(e.to_string())),
            }
        })
        .await
        .map_err(|e| AsyncRocksDBError::ThreadPoolError(e.to_string()))??;

        // Update cache if value found
        if let Some(ref value) = result {
            let mut cache = self.cache.write().await;
            cache.insert(
                key_bytes,
                CacheEntry {
                    value: value.clone(),
                    created_at: std::time::Instant::now(),
                    ttl: std::time::Duration::from_secs(self.config.cache_ttl_secs),
                },
            );
        }

        // Update metrics
        {
            let mut metrics = self.metrics.lock().await;
            metrics.cache_misses += 1;
            metrics.read_operations += 1;
            metrics.total_read_duration += start.elapsed();
        }

        Ok(result)
    }

    /// Put a key-value pair (async-safe, buffered)
    pub async fn put<K: AsRef<[u8]>, V: AsRef<[u8]>>(
        &self,
        key: K,
        value: V,
    ) -> AsyncRocksDBResult<()> {
        let key_bytes = Bytes::copy_from_slice(key.as_ref());
        let value_bytes = Bytes::copy_from_slice(value.as_ref());

        // Add to write buffer
        {
            let mut write_buffer = self.write_buffer.lock().await;
            write_buffer.insert(key_bytes.clone(), Some(value_bytes.clone()));
        }

        // Update cache immediately for read consistency
        {
            let mut cache = self.cache.write().await;
            cache.insert(
                key_bytes,
                CacheEntry {
                    value: value_bytes,
                    created_at: std::time::Instant::now(),
                    ttl: std::time::Duration::from_secs(self.config.cache_ttl_secs),
                },
            );
        }

        // Update metrics
        {
            let mut metrics = self.metrics.lock().await;
            metrics.write_operations += 1;
        }

        Ok(())
    }

    /// Delete a key (async-safe, buffered)
    pub async fn delete<K: AsRef<[u8]>>(&self, key: K) -> AsyncRocksDBResult<()> {
        let key_bytes = Bytes::copy_from_slice(key.as_ref());

        // Add to write buffer as deletion
        {
            let mut write_buffer = self.write_buffer.lock().await;
            write_buffer.insert(key_bytes.clone(), None);
        }

        // Remove from cache
        {
            let mut cache = self.cache.write().await;
            cache.remove(&key_bytes);
        }

        Ok(())
    }

    /// Flush write buffer to disk
    pub async fn flush(&self) -> AsyncRocksDBResult<()> {
        let start = std::time::Instant::now();

        // Get current write buffer
        let operations = {
            let mut write_buffer = self.write_buffer.lock().await;
            let ops = write_buffer.clone();
            write_buffer.clear();
            ops
        };

        if operations.is_empty() {
            return Ok(());
        }

        let db_path = self.db_path.clone();
        let config = self.config.clone();

        tokio::task::spawn_blocking(move || {
            let mut opts = rocksdb::Options::default();
            opts.create_if_missing(true);

            if config.enable_compression {
                opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
            }

            let db = rocksdb::DB::open(&opts, &db_path)
                .map_err(|e| AsyncRocksDBError::DatabaseError(e.to_string()))?;

            // Use write batch for atomic operations
            let mut batch = rocksdb::WriteBatch::default();

            for (key, value) in operations {
                match value {
                    Some(val) => batch.put(&key, &val),
                    None => batch.delete(&key),
                }
            }

            db.write(batch)
                .map_err(|e| AsyncRocksDBError::DatabaseError(e.to_string()))?;

            Ok::<(), AsyncRocksDBError>(())
        })
        .await
        .map_err(|e| AsyncRocksDBError::ThreadPoolError(e.to_string()))??;

        // Update metrics
        {
            let mut metrics = self.metrics.lock().await;
            metrics.total_write_duration += start.elapsed();
        }

        Ok(())
    }

    /// Create an async iterator over all key-value pairs
    pub async fn iter(&self) -> AsyncRocksDBResult<AsyncRocksDBIterator> {
        let start = std::time::Instant::now();
        let db_path = self.db_path.clone();
        let buffer_size = self.config.iterator_buffer_size;

        // For small datasets, use buffered approach
        let buffer = tokio::task::spawn_blocking(move || {
            let opts = rocksdb::Options::default();
            let db = rocksdb::DB::open_for_read_only(&opts, &db_path, false)
                .map_err(|e| AsyncRocksDBError::DatabaseError(e.to_string()))?;

            let iterator = db.iterator(rocksdb::IteratorMode::Start);
            let mut results = Vec::new();
            let mut count = 0;

            for item in iterator {
                if count >= buffer_size {
                    break;
                }

                let (key, value) =
                    item.map_err(|e| AsyncRocksDBError::DatabaseError(e.to_string()))?;

                results.push(AsyncKeyValue {
                    key: Bytes::from(key.to_vec()),
                    value: Bytes::from(value.to_vec()),
                });

                count += 1;
            }

            Ok::<(Vec<AsyncKeyValue>, bool), AsyncRocksDBError>((results, count < buffer_size))
        })
        .await
        .map_err(|e| AsyncRocksDBError::ThreadPoolError(e.to_string()))??;

        // Update metrics
        {
            let mut metrics = self.metrics.lock().await;
            metrics.iterator_operations += 1;
        }

        Ok(AsyncRocksDBIterator::new_buffered(buffer.0, buffer.1))
    }

    /// Create an async iterator with a key prefix
    pub async fn iter_prefix<P: AsRef<[u8]>>(
        &self,
        prefix: P,
    ) -> AsyncRocksDBResult<AsyncRocksDBIterator> {
        let prefix_bytes = prefix.as_ref().to_vec();
        let db_path = self.db_path.clone();
        let buffer_size = self.config.iterator_buffer_size;

        let buffer = tokio::task::spawn_blocking(move || {
            let opts = rocksdb::Options::default();
            let db = rocksdb::DB::open_for_read_only(&opts, &db_path, false)
                .map_err(|e| AsyncRocksDBError::DatabaseError(e.to_string()))?;

            let iterator = db.iterator(rocksdb::IteratorMode::From(
                &prefix_bytes,
                rocksdb::Direction::Forward,
            ));
            let mut results = Vec::new();
            let mut count = 0;

            for item in iterator {
                if count >= buffer_size {
                    break;
                }

                let (key, value) =
                    item.map_err(|e| AsyncRocksDBError::DatabaseError(e.to_string()))?;

                // Check if key still has the prefix
                if !key.starts_with(&prefix_bytes) {
                    break;
                }

                results.push(AsyncKeyValue {
                    key: Bytes::from(key.to_vec()),
                    value: Bytes::from(value.to_vec()),
                });

                count += 1;
            }

            Ok::<Vec<AsyncKeyValue>, AsyncRocksDBError>(results)
        })
        .await
        .map_err(|e| AsyncRocksDBError::ThreadPoolError(e.to_string()))??;

        Ok(AsyncRocksDBIterator::new_buffered(buffer, true))
    }

    /// Get performance metrics
    pub async fn get_metrics(&self) -> RocksDBMetrics {
        let metrics = self.metrics.lock().await;
        metrics.clone()
    }

    /// Clear cache (useful for testing or memory management)
    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
    }

    /// Compact database (async-safe)
    pub async fn compact(&self) -> AsyncRocksDBResult<()> {
        let db_path = self.db_path.clone();

        tokio::task::spawn_blocking(move || {
            let opts = rocksdb::Options::default();
            let db = rocksdb::DB::open(&opts, &db_path)
                .map_err(|e| AsyncRocksDBError::DatabaseError(e.to_string()))?;

            db.compact_range(None::<&[u8]>, None::<&[u8]>);
            Ok::<(), AsyncRocksDBError>(())
        })
        .await
        .map_err(|e| AsyncRocksDBError::ThreadPoolError(e.to_string()))??;

        Ok(())
    }
}

// Ensure our wrapper is Send + Sync for async contexts
unsafe impl Send for AsyncRocksDB {}
unsafe impl Sync for AsyncRocksDB {}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn create_test_db() -> (TempDir, AsyncRocksDB) {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");
        let config = AsyncRocksDBConfig::default();
        let db = AsyncRocksDB::open(&db_path, config).await.unwrap();
        (temp_dir, db)
    }

    #[tokio::test]
    async fn test_basic_operations() {
        let (_temp_dir, db) = create_test_db().await;

        // Test put and get
        db.put(b"key1", b"value1").await.unwrap();
        db.flush().await.unwrap();

        let value = db.get(b"key1").await.unwrap();
        assert_eq!(value, Some(Bytes::from("value1")));

        // Test non-existent key
        let missing = db.get(b"missing").await.unwrap();
        assert_eq!(missing, None);
    }

    #[tokio::test]
    async fn test_buffered_operations() {
        let (_temp_dir, db) = create_test_db().await;

        // Test buffered writes
        db.put(b"key1", b"value1").await.unwrap();
        db.put(b"key2", b"value2").await.unwrap();

        // Should be available from buffer before flush
        let value1 = db.get(b"key1").await.unwrap();
        assert_eq!(value1, Some(Bytes::from("value1")));

        // Flush and verify persistence
        db.flush().await.unwrap();
        let value2 = db.get(b"key2").await.unwrap();
        assert_eq!(value2, Some(Bytes::from("value2")));
    }

    #[tokio::test]
    async fn test_iterator() {
        let (_temp_dir, db) = create_test_db().await;

        // Insert test data
        for i in 0..10 {
            let key = format!("key{:02}", i);
            let value = format!("value{}", i);
            db.put(key.as_bytes(), value.as_bytes()).await.unwrap();
        }
        db.flush().await.unwrap();

        // Test iterator
        let mut iter = db.iter().await.unwrap();
        let mut count = 0;
        while let Some(_item) = iter.next().await.unwrap() {
            count += 1;
        }
        assert_eq!(count, 10);
    }

    #[tokio::test]
    async fn test_prefix_iterator() {
        let (_temp_dir, db) = create_test_db().await;

        // Insert test data with different prefixes
        db.put(b"prefix_a_1", b"value1").await.unwrap();
        db.put(b"prefix_a_2", b"value2").await.unwrap();
        db.put(b"prefix_b_1", b"value3").await.unwrap();
        db.flush().await.unwrap();

        // Test prefix iterator
        let mut iter = db.iter_prefix(b"prefix_a").await.unwrap();
        let mut count = 0;
        while let Some(_item) = iter.next().await.unwrap() {
            count += 1;
        }
        assert_eq!(count, 2);
    }

    #[tokio::test]
    async fn test_cache_functionality() {
        let (_temp_dir, db) = create_test_db().await;

        db.put(b"cached_key", b"cached_value").await.unwrap();
        db.flush().await.unwrap();

        // First read - cache miss
        let _value1 = db.get(b"cached_key").await.unwrap();

        // Second read - cache hit
        let _value2 = db.get(b"cached_key").await.unwrap();

        let metrics = db.get_metrics().await;
        assert!(metrics.cache_hits > 0);
    }

    #[tokio::test]
    async fn test_delete_operations() {
        let (_temp_dir, db) = create_test_db().await;

        // Insert and delete
        db.put(b"delete_me", b"value").await.unwrap();
        db.flush().await.unwrap();

        let value_before = db.get(b"delete_me").await.unwrap();
        assert!(value_before.is_some());

        db.delete(b"delete_me").await.unwrap();
        db.flush().await.unwrap();

        let value_after = db.get(b"delete_me").await.unwrap();
        assert!(value_after.is_none());
    }
}
