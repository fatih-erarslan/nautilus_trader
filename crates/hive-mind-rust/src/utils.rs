//! Utility functions and helpers for the hive mind system

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use std::sync::Arc;
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
#[cfg(feature = "crypto")]
use ring::rand::{SecureRandom, SystemRandom};
use tracing::{debug, warn, info};

use crate::error::{HiveMindError, Result};

/// Time utilities
pub mod time {
    // Remove unused wildcard import
    
    /// Get current timestamp as seconds since Unix epoch
    pub fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
    
    /// Get current timestamp as milliseconds since Unix epoch
    pub fn current_timestamp_millis() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
    
    /// Get current timestamp as microseconds since Unix epoch
    pub fn current_timestamp_micros() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64
    }
    
    /// Convert SystemTime to Unix timestamp
    pub fn system_time_to_timestamp(time: SystemTime) -> u64 {
        time.duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
    
    /// Convert Unix timestamp to SystemTime
    pub fn timestamp_to_system_time(timestamp: u64) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(timestamp)
    }
    
    /// Format duration as human-readable string
    pub fn format_duration(duration: Duration) -> String {
        let total_seconds = duration.as_secs();
        let days = total_seconds / 86400;
        let hours = (total_seconds % 86400) / 3600;
        let minutes = (total_seconds % 3600) / 60;
        let seconds = total_seconds % 60;
        let millis = duration.subsec_millis();
        
        if days > 0 {
            format!("{}d {}h {}m {}s", days, hours, minutes, seconds)
        } else if hours > 0 {
            format!("{}h {}m {}s", hours, minutes, seconds)
        } else if minutes > 0 {
            format!("{}m {}s", minutes, seconds)
        } else if seconds > 0 {
            format!("{}.{}s", seconds, millis / 100)
        } else {
            format!("{}ms", millis)
        }
    }
    
    /// Parse duration from human-readable string
    pub fn parse_duration(s: &str) -> Result<Duration> {
        let s = s.trim().to_lowercase();
        
        if s.ends_with("ms") {
            let ms: u64 = s[..s.len()-2].parse()
                .map_err(|_| HiveMindError::InvalidState { 
                    message: "Invalid duration format".to_string() 
                })?;
            Ok(Duration::from_millis(ms))
        } else if s.ends_with('s') {
            let secs: u64 = s[..s.len()-1].parse()
                .map_err(|_| HiveMindError::InvalidState { 
                    message: "Invalid duration format".to_string() 
                })?;
            Ok(Duration::from_secs(secs))
        } else if s.ends_with('m') {
            let mins: u64 = s[..s.len()-1].parse()
                .map_err(|_| HiveMindError::InvalidState { 
                    message: "Invalid duration format".to_string() 
                })?;
            Ok(Duration::from_secs(mins * 60))
        } else if s.ends_with('h') {
            let hours: u64 = s[..s.len()-1].parse()
                .map_err(|_| HiveMindError::InvalidState { 
                    message: "Invalid duration format".to_string() 
                })?;
            Ok(Duration::from_secs(hours * 3600))
        } else if s.ends_with('d') {
            let days: u64 = s[..s.len()-1].parse()
                .map_err(|_| HiveMindError::InvalidState { 
                    message: "Invalid duration format".to_string() 
                })?;
            Ok(Duration::from_secs(days * 86400))
        } else {
            // Default to seconds
            let secs: u64 = s.parse()
                .map_err(|_| HiveMindError::InvalidState { 
                    message: "Invalid duration format".to_string() 
                })?;
            Ok(Duration::from_secs(secs))
        }
    }
}

/// Cryptographic utilities
pub mod crypto {
    // Remove unused wildcard import
    
    /// Generate a secure random UUID
    pub fn generate_secure_uuid() -> Uuid {
        Uuid::new_v4()
    }
    
    /// Generate secure random bytes
    #[cfg(feature = "crypto")]
    pub fn generate_random_bytes(length: usize) -> Result<Vec<u8>> {
        let rng = SystemRandom::new();
        let mut bytes = vec![0u8; length];
        rng.fill(&mut bytes)
            .map_err(|_| HiveMindError::Internal("Failed to generate random bytes".to_string()))?;
        Ok(bytes)
    }
    
    /// Generate secure random bytes (fallback without ring)
    #[cfg(not(feature = "crypto"))]
    pub fn generate_random_bytes(length: usize) -> Result<Vec<u8>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        Ok((0..length).map(|_| rng.gen::<u8>()).collect())
    }
    
    /// Generate a secure random string
    pub fn generate_random_string(length: usize) -> Result<String> {
        const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        let bytes = generate_random_bytes(length)?;
        let result: String = bytes
            .iter()
            .map(|&byte| CHARSET[(byte as usize) % CHARSET.len()] as char)
            .collect();
        Ok(result)
    }
    
    /// Hash data using SHA-256
    pub fn hash_sha256(data: &[u8]) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.finalize().to_vec()
    }
    
    /// Hash string using SHA-256 and return as hex
    pub fn hash_string(data: &str) -> String {
        let hash = hash_sha256(data.as_bytes());
        hex::encode(hash)
    }
    
    /// Generate a deterministic ID from multiple inputs
    pub fn generate_deterministic_id(inputs: &[&str]) -> String {
        let combined = inputs.join("|");
        hash_string(&combined)
    }
    
    /// Verify hash integrity
    pub fn verify_hash(data: &[u8], expected_hash: &[u8]) -> bool {
        let actual_hash = hash_sha256(data);
        actual_hash == expected_hash
    }
}

/// String utilities
pub mod string {
    // Remove unused wildcard import
    
    /// Truncate string to maximum length with ellipsis
    pub fn truncate(s: &str, max_len: usize) -> String {
        if s.len() <= max_len {
            s.to_string()
        } else if max_len <= 3 {
            "...".to_string()
        } else {
            format!("{}...", &s[..max_len-3])
        }
    }
    
    /// Sanitize string for safe usage in identifiers
    pub fn sanitize_identifier(s: &str) -> String {
        s.chars()
            .map(|c| if c.is_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
            .collect()
    }
    
    /// Convert string to snake_case
    pub fn to_snake_case(s: &str) -> String {
        let mut result = String::new();
        let mut chars = s.chars().peekable();
        
        while let Some(ch) = chars.next() {
            if ch.is_uppercase() && !result.is_empty() {
                if let Some(&next_ch) = chars.peek() {
                    if next_ch.is_lowercase() {
                        result.push('_');
                    }
                }
            }
            result.push(ch.to_lowercase().next().unwrap_or(ch));
        }
        
        result
    }
    
    /// Convert string to kebab-case
    pub fn to_kebab_case(s: &str) -> String {
        to_snake_case(s).replace('_', "-")
    }
    
    /// Check if string is valid JSON
    pub fn is_valid_json(s: &str) -> bool {
        serde_json::from_str::<serde_json::Value>(s).is_ok()
    }
    
    /// Escape string for safe logging
    pub fn escape_for_log(s: &str) -> String {
        s.chars()
            .map(|c| match c {
                '"' => "\\\"".to_string(),
                '\\' => "\\\\".to_string(),
                '\n' => "\\n".to_string(),
                '\r' => "\\r".to_string(),
                '\t' => "\\t".to_string(),
                c if c.is_control() => format!("\\u{:04x}", c as u32),
                c => c.to_string(),
            })
            .collect()
    }
}

/// Network utilities
pub mod network {
    // Remove unused wildcard import
    use std::net::{IpAddr, SocketAddr, ToSocketAddrs};
    
    /// Check if an IP address is private
    pub fn is_private_ip(ip: &IpAddr) -> bool {
        match ip {
            IpAddr::V4(ipv4) => {
                ipv4.is_private() || ipv4.is_loopback() || ipv4.is_link_local()
            }
            IpAddr::V6(ipv6) => {
                ipv6.is_loopback() || ipv6.is_unicast_link_local() || ipv6.is_unique_local()
            }
        }
    }
    
    /// Parse socket address with default port
    pub fn parse_socket_addr(addr: &str, default_port: u16) -> Result<SocketAddr> {
        if addr.contains(':') {
            addr.parse()
                .map_err(|_| HiveMindError::InvalidState {
                    message: format!("Invalid socket address: {}", addr),
                })
        } else {
            format!("{}:{}", addr, default_port)
                .parse()
                .map_err(|_| HiveMindError::InvalidState {
                    message: format!("Invalid socket address: {}", addr),
                })
        }
    }
    
    /// Resolve hostname to IP addresses
    pub fn resolve_hostname(hostname: &str, port: u16) -> Result<Vec<SocketAddr>> {
        let addr_str = format!("{}:{}", hostname, port);
        addr_str.to_socket_addrs()
            .map(|addrs| addrs.collect())
            .map_err(|e| HiveMindError::Internal(format!("Failed to resolve hostname: {}", e)))
    }
    
    /// Get local IP addresses
    pub fn get_local_ip_addresses() -> Result<Vec<IpAddr>> {
        // This is a simplified implementation
        // In a real implementation, you would use platform-specific APIs
        Ok(vec![IpAddr::V4(std::net::Ipv4Addr::new(127, 0, 0, 1))])
    }
    
    /// Check if port is available
    pub fn is_port_available(port: u16) -> bool {
        match std::net::TcpListener::bind(("127.0.0.1", port)) {
            Ok(_) => true,
            Err(_) => false,
        }
    }
    
    /// Find available port in range
    pub fn find_available_port(start: u16, end: u16) -> Option<u16> {
        (start..=end).find(|&port| is_port_available(port))
    }
}

/// Data structure utilities
pub mod collections {
    // Remove unused wildcard import
    use std::collections::{BTreeMap, VecDeque};
    
    /// Circular buffer with fixed capacity
    #[derive(Debug, Clone)]
    pub struct CircularBuffer<T> {
        buffer: VecDeque<T>,
        capacity: usize,
    }
    
    impl<T> CircularBuffer<T> {
        /// Create new circular buffer with given capacity
        pub fn new(capacity: usize) -> Self {
            Self {
                buffer: VecDeque::with_capacity(capacity),
                capacity,
            }
        }
        
        /// Push item to buffer
        pub fn push(&mut self, item: T) {
            if self.buffer.len() >= self.capacity {
                self.buffer.pop_front();
            }
            self.buffer.push_back(item);
        }
        
        /// Get current length
        pub fn len(&self) -> usize {
            self.buffer.len()
        }
        
        /// Check if buffer is empty
        pub fn is_empty(&self) -> bool {
            self.buffer.is_empty()
        }
        
        /// Get item at index
        pub fn get(&self, index: usize) -> Option<&T> {
            self.buffer.get(index)
        }
        
        /// Iterate over items
        pub fn iter(&self) -> impl Iterator<Item = &T> {
            self.buffer.iter()
        }
        
        /// Clear buffer
        pub fn clear(&mut self) {
            self.buffer.clear();
        }
    }
    
    /// Time-series data structure
    #[derive(Debug, Clone)]
    pub struct TimeSeries<T> {
        data: BTreeMap<u64, T>,
        max_points: usize,
    }
    
    impl<T> TimeSeries<T> {
        /// Create new time series with maximum points
        pub fn new(max_points: usize) -> Self {
            Self {
                data: BTreeMap::new(),
                max_points,
            }
        }
        
        /// Insert data point
        pub fn insert(&mut self, timestamp: u64, value: T) {
            self.data.insert(timestamp, value);
            
            // Remove oldest points if exceeding capacity
            while self.data.len() > self.max_points {
                if let Some(oldest_key) = self.data.keys().next().cloned() {
                    self.data.remove(&oldest_key);
                }
            }
        }
        
        /// Get value at timestamp
        pub fn get(&self, timestamp: u64) -> Option<&T> {
            self.data.get(&timestamp)
        }
        
        /// Get range of values
        pub fn range(&self, start: u64, end: u64) -> impl Iterator<Item = (&u64, &T)> {
            self.data.range(start..=end)
        }
        
        /// Get latest value
        pub fn latest(&self) -> Option<(&u64, &T)> {
            self.data.iter().next_back()
        }
        
        /// Get number of data points
        pub fn len(&self) -> usize {
            self.data.len()
        }
        
        /// Check if empty
        pub fn is_empty(&self) -> bool {
            self.data.is_empty()
        }
    }
    
    /// LRU cache implementation
    #[derive(Debug)]
    pub struct LruCache<K, V>
    where
        K: std::hash::Hash + Eq + Clone,
    {
        map: HashMap<K, V>,
        order: VecDeque<K>,
        capacity: usize,
    }
    
    impl<K, V> LruCache<K, V>
    where
        K: std::hash::Hash + Eq + Clone,
    {
        /// Create new LRU cache with given capacity
        pub fn new(capacity: usize) -> Self {
            Self {
                map: HashMap::with_capacity(capacity),
                order: VecDeque::with_capacity(capacity),
                capacity,
            }
        }
        
        /// Get value by key
        pub fn get(&mut self, key: &K) -> Option<&V> {
            if self.map.contains_key(key) {
                self.move_to_front(key);
                self.map.get(key)
            } else {
                None
            }
        }
        
        /// Insert key-value pair
        pub fn insert(&mut self, key: K, value: V) {
            if self.map.contains_key(&key) {
                self.map.insert(key.clone(), value);
                self.move_to_front(&key);
            } else {
                if self.map.len() >= self.capacity {
                    self.evict_oldest();
                }
                self.map.insert(key.clone(), value);
                self.order.push_front(key);
            }
        }
        
        /// Remove key-value pair
        pub fn remove(&mut self, key: &K) -> Option<V> {
            if let Some(value) = self.map.remove(key) {
                self.order.retain(|k| k != key);
                Some(value)
            } else {
                None
            }
        }
        
        /// Get current size
        pub fn len(&self) -> usize {
            self.map.len()
        }
        
        /// Check if empty
        pub fn is_empty(&self) -> bool {
            self.map.is_empty()
        }
        
        /// Move key to front of order
        fn move_to_front(&mut self, key: &K) {
            self.order.retain(|k| k != key);
            self.order.push_front(key.clone());
        }
        
        /// Evict oldest item
        fn evict_oldest(&mut self) {
            if let Some(oldest_key) = self.order.pop_back() {
                self.map.remove(&oldest_key);
            }
        }
    }
}

/// Math utilities
pub mod math {
    // Remove unused wildcard import
    
    /// Calculate moving average
    pub fn moving_average(values: &[f64], window_size: usize) -> Vec<f64> {
        if values.is_empty() || window_size == 0 {
            return Vec::new();
        }
        
        let mut result = Vec::new();
        for i in 0..values.len() {
            let start = if i + 1 >= window_size { i + 1 - window_size } else { 0 };
            let end = i + 1;
            let sum: f64 = values[start..end].iter().sum();
            let avg = sum / (end - start) as f64;
            result.push(avg);
        }
        result
    }
    
    /// Calculate exponential moving average
    pub fn exponential_moving_average(values: &[f64], alpha: f64) -> Vec<f64> {
        if values.is_empty() {
            return Vec::new();
        }
        
        let mut result = Vec::with_capacity(values.len());
        let mut ema = values[0];
        result.push(ema);
        
        for &value in &values[1..] {
            ema = alpha * value + (1.0 - alpha) * ema;
            result.push(ema);
        }
        result
    }
    
    /// Calculate standard deviation
    pub fn standard_deviation(values: &[f64]) -> f64 {
        if values.len() <= 1 {
            return 0.0;
        }
        
        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        let variance: f64 = values
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (values.len() - 1) as f64;
        variance.sqrt()
    }
    
    /// Calculate percentile
    pub fn percentile(values: &[f64], p: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = (p / 100.0) * (sorted.len() - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;
        
        if lower == upper {
            sorted[lower]
        } else {
            let weight = index - lower as f64;
            sorted[lower] * (1.0 - weight) + sorted[upper] * weight
        }
    }
    
    /// Calculate correlation coefficient
    pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }
        
        let n = x.len() as f64;
        let mean_x: f64 = x.iter().sum::<f64>() / n;
        let mean_y: f64 = y.iter().sum::<f64>() / n;
        
        let numerator: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum();
        
        let sum_sq_x: f64 = x.iter().map(|xi| (xi - mean_x).powi(2)).sum();
        let sum_sq_y: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();
        
        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
    
    /// Normalize values to 0-1 range
    pub fn normalize(values: &[f64]) -> Vec<f64> {
        if values.is_empty() {
            return Vec::new();
        }
        
        let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        if (max - min).abs() < f64::EPSILON {
            vec![0.5; values.len()]
        } else {
            values.iter().map(|&x| (x - min) / (max - min)).collect()
        }
    }
    
    /// Calculate z-score
    pub fn z_score(values: &[f64]) -> Vec<f64> {
        if values.is_empty() {
            return Vec::new();
        }
        
        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        let std_dev = standard_deviation(values);
        
        if std_dev == 0.0 {
            vec![0.0; values.len()]
        } else {
            values.iter().map(|&x| (x - mean) / std_dev).collect()
        }
    }
}

/// Validation utilities
pub mod validation {
    // Remove unused wildcard import
    use std::net::IpAddr;
    use uuid::Uuid;
    
    /// Validate email address format
    pub fn is_valid_email(email: &str) -> bool {
        // Simple email validation
        email.contains('@') && email.contains('.') && !email.starts_with('@') && !email.ends_with('@')
    }
    
    /// Validate UUID format
    pub fn is_valid_uuid(uuid_str: &str) -> bool {
        Uuid::parse_str(uuid_str).is_ok()
    }
    
    /// Validate IP address format
    pub fn is_valid_ip(ip_str: &str) -> bool {
        ip_str.parse::<IpAddr>().is_ok()
    }
    
    /// Validate port number
    pub fn is_valid_port(port: u32) -> bool {
        port > 0 && port <= 65535
    }
    
    /// Validate URL format
    pub fn is_valid_url(url: &str) -> bool {
        // Simple URL validation
        url.starts_with("http://") || url.starts_with("https://")
    }
    
    /// Validate configuration value range
    pub fn validate_range<T>(value: T, min: T, max: T) -> Result<T>
    where
        T: PartialOrd + std::fmt::Display + Copy,
    {
        if value < min || value > max {
            Err(HiveMindError::InvalidState {
                message: format!("Value {} is out of range [{}, {}]", value, min, max),
            })
        } else {
            Ok(value)
        }
    }
    
    /// Validate string length
    pub fn validate_string_length(s: &str, min_len: usize, max_len: usize) -> Result<()> {
        if s.len() < min_len {
            Err(HiveMindError::InvalidState {
                message: format!("String too short: {} < {}", s.len(), min_len),
            })
        } else if s.len() > max_len {
            Err(HiveMindError::InvalidState {
                message: format!("String too long: {} > {}", s.len(), max_len),
            })
        } else {
            Ok(())
        }
    }
}

/// Concurrency utilities
pub mod concurrency {
    // Remove unused wildcard import
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Arc;
    use tokio::sync::{Mutex, RwLock, Semaphore};
    use tokio::time::{sleep, Duration};
    
    /// Simple atomic counter
    #[derive(Debug, Default)]
    pub struct AtomicCounter {
        value: AtomicU64,
    }
    
    impl AtomicCounter {
        /// Create new counter
        pub fn new(initial: u64) -> Self {
            Self {
                value: AtomicU64::new(initial),
            }
        }
        
        /// Increment counter
        pub fn increment(&self) -> u64 {
            self.value.fetch_add(1, Ordering::SeqCst)
        }
        
        /// Decrement counter
        pub fn decrement(&self) -> u64 {
            self.value.fetch_sub(1, Ordering::SeqCst)
        }
        
        /// Get current value
        pub fn get(&self) -> u64 {
            self.value.load(Ordering::SeqCst)
        }
        
        /// Set value
        pub fn set(&self, value: u64) {
            self.value.store(value, Ordering::SeqCst);
        }
    }
    
    /// Circuit breaker for fault tolerance
    #[derive(Debug)]
    pub struct CircuitBreaker {
        state: Arc<Mutex<CircuitBreakerState>>,
        failure_threshold: u32,
        timeout: Duration,
        failure_count: AtomicU64,
    }
    
    #[derive(Debug, Clone, PartialEq)]
    enum CircuitBreakerState {
        Closed,
        Open,
        HalfOpen,
    }
    
    impl CircuitBreaker {
        /// Create new circuit breaker
        pub fn new(failure_threshold: u32, timeout: Duration) -> Self {
            Self {
                state: Arc::new(Mutex::new(CircuitBreakerState::Closed)),
                failure_threshold,
                timeout,
                failure_count: AtomicU64::new(0),
            }
        }
        
        /// Execute function with circuit breaker protection
        pub async fn execute<F, T, E>(&self, f: F) -> Result<T>
        where
            F: futures::Future<Output = std::result::Result<T, E>>,
            E: std::fmt::Display,
        {
            let state = {
                let state_guard = self.state.lock().await;
                state_guard.clone()
            };
            
            match state {
                CircuitBreakerState::Open => {
                    Err(HiveMindError::Internal("Circuit breaker is open".to_string()))
                }
                CircuitBreakerState::Closed | CircuitBreakerState::HalfOpen => {
                    match f.await {
                        Ok(result) => {
                            self.on_success().await;
                            Ok(result)
                        }
                        Err(e) => {
                            self.on_failure().await;
                            Err(HiveMindError::Internal(format!("Operation failed: {}", e)))
                        }
                    }
                }
            }
        }
        
        /// Handle successful operation
        async fn on_success(&self) {
            self.failure_count.store(0, Ordering::SeqCst);
            let mut state = self.state.lock().await;
            *state = CircuitBreakerState::Closed;
        }
        
        /// Handle failed operation
        async fn on_failure(&self) {
            let failures = self.failure_count.fetch_add(1, Ordering::SeqCst) + 1;
            
            if failures >= self.failure_threshold as u64 {
                let mut state = self.state.lock().await;
                *state = CircuitBreakerState::Open;
                
                // Schedule state transition to half-open
                let state_clone = self.state.clone();
                let timeout = self.timeout;
                tokio::spawn(async move {
                    sleep(timeout).await;
                    let mut state_guard = state_clone.lock().await;
                    if *state_guard == CircuitBreakerState::Open {
                        *state_guard = CircuitBreakerState::HalfOpen;
                    }
                });
            }
        }
    }
    
    /// Rate limiter for controlling request rates
    #[derive(Debug)]
    pub struct RateLimiter {
        semaphore: Arc<Semaphore>,
        rate: u32,
        window: Duration,
    }
    
    impl RateLimiter {
        /// Create new rate limiter
        pub fn new(rate: u32, window: Duration) -> Self {
            Self {
                semaphore: Arc::new(Semaphore::new(rate as usize)),
                rate,
                window,
            }
        }
        
        /// Acquire permission (blocking)
        pub async fn acquire(&self) -> Result<()> {
            let _permit = self.semaphore.acquire().await
                .map_err(|_| HiveMindError::Internal("Rate limiter error".to_string()))?;
            
            // Release permit after window duration
            let semaphore = self.semaphore.clone();
            let window = self.window;
            tokio::spawn(async move {
                sleep(window).await;
                semaphore.add_permits(1);
            });
            
            Ok(())
        }
        
        /// Try to acquire permission (non-blocking)
        pub fn try_acquire(&self) -> bool {
            if let Ok(_permit) = self.semaphore.try_acquire() {
                let semaphore = self.semaphore.clone();
                let window = self.window;
                tokio::spawn(async move {
                    sleep(window).await;
                    semaphore.add_permits(1);
                });
                true
            } else {
                false
            }
        }
    }

    /// Health monitoring utility
    #[derive(Debug)]
    pub struct HealthMonitor {
        config: crate::config::HiveMindConfig,
        last_check: Arc<RwLock<std::time::Instant>>,
        health_status: Arc<RwLock<bool>>,
    }
    
    impl HealthMonitor {
        /// Create new health monitor
        pub fn new(config: crate::config::HiveMindConfig) -> Self {
            Self {
                config,
                last_check: Arc::new(RwLock::new(std::time::Instant::now())),
                health_status: Arc::new(RwLock::new(true)),
            }
        }
        
        /// Check system health
        pub async fn check_health(&self) -> Result<bool> {
            let mut last_check = self.last_check.write().await;
            let mut health_status = self.health_status.write().await;
            
            *last_check = std::time::Instant::now();
            *health_status = true; // Simplified health check
            
            Ok(*health_status)
        }
        
        /// Get current health status
        pub async fn is_healthy(&self) -> bool {
            *self.health_status.read().await
        }
    }

    /// Recovery manager for system failures
    #[derive(Debug)]
    pub struct RecoveryManager {
        config: crate::config::HiveMindConfig,
        recovery_attempts: Arc<RwLock<HashMap<String, u32>>>,
        max_attempts: u32,
    }
    
    impl RecoveryManager {
        /// Create new recovery manager
        pub fn new(config: crate::config::HiveMindConfig) -> Self {
            Self {
                config,
                recovery_attempts: Arc::new(RwLock::new(HashMap::new())),
                max_attempts: 3,
            }
        }
        
        /// Attempt recovery for a component
        pub async fn attempt_recovery(&self, component: &str) -> Result<bool> {
            let mut attempts = self.recovery_attempts.write().await;
            let current_attempts = attempts.get(component).unwrap_or(&0);
            
            if *current_attempts >= self.max_attempts {
                return Ok(false);
            }
            
            attempts.insert(component.to_string(), current_attempts + 1);
            
            // Simplified recovery logic
            info!("Attempting recovery for component: {}", component);
            Ok(true)
        }
        
        /// Reset recovery attempts for a component
        pub async fn reset_attempts(&self, component: &str) {
            let mut attempts = self.recovery_attempts.write().await;
            attempts.remove(component);
        }
    }
}

/// Testing utilities
#[cfg(test)]
pub mod testing {
    // Remove unused wildcard import
    use std::sync::Arc;
    use tokio::sync::RwLock;
    
    /// Mock data generator
    pub struct MockDataGenerator {
        seed: u64,
    }
    
    impl MockDataGenerator {
        pub fn new(seed: u64) -> Self {
            Self { seed }
        }
        
        /// Generate mock metrics data
        pub fn generate_metrics(&mut self, count: usize) -> Vec<(String, f64)> {
            (0..count)
                .map(|i| {
                    let value = (self.seed as f64 + i as f64) % 100.0;
                    self.seed = self.seed.wrapping_mul(1103515245).wrapping_add(12345);
                    (format!("metric_{}", i), value)
                })
                .collect()
        }
        
        /// Generate mock time series data
        pub fn generate_time_series(&mut self, count: usize) -> Vec<(u64, f64)> {
            let base_time = time::current_timestamp();
            (0..count)
                .map(|i| {
                    let timestamp = base_time + i as u64;
                    let value = (self.seed as f64 + i as f64) % 100.0;
                    self.seed = self.seed.wrapping_mul(1103515245).wrapping_add(12345);
                    (timestamp, value)
                })
                .collect()
        }
    }
    
    /// Test helpers
    pub async fn wait_for_condition<F>(condition: F, timeout: Duration) -> bool
    where
        F: Fn() -> bool,
    {
        let start = std::time::Instant::now();
        while start.elapsed() < timeout {
            if condition() {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    // Remove unused wildcard import
    
    #[test]
    fn test_time_formatting() {
        let duration = Duration::from_secs(3661); // 1h 1m 1s
        let formatted = time::format_duration(duration);
        assert!(formatted.contains("1h"));
        assert!(formatted.contains("1m"));
        assert!(formatted.contains("1s"));
    }
    
    #[test]
    fn test_duration_parsing() {
        assert_eq!(time::parse_duration("30s").unwrap(), Duration::from_secs(30));
        assert_eq!(time::parse_duration("5m").unwrap(), Duration::from_secs(300));
        assert_eq!(time::parse_duration("2h").unwrap(), Duration::from_secs(7200));
        assert_eq!(time::parse_duration("1d").unwrap(), Duration::from_secs(86400));
    }
    
    #[test]
    fn test_string_utilities() {
        assert_eq!(string::truncate("hello world", 5), "he...");
        assert_eq!(string::to_snake_case("CamelCase"), "camel_case");
        assert_eq!(string::to_kebab_case("CamelCase"), "camel-case");
        assert_eq!(string::sanitize_identifier("hello-world!"), "hello-world_");
    }
    
    #[test]
    fn test_crypto_utilities() {
        let uuid = crypto::generate_secure_uuid();
        assert_ne!(uuid, Uuid::nil());
        
        let random_bytes = crypto::generate_random_bytes(16).unwrap();
        assert_eq!(random_bytes.len(), 16);
        
        let hash = crypto::hash_string("test");
        assert_eq!(hash.len(), 64); // SHA-256 produces 64-character hex string
    }
    
    #[test]
    fn test_math_utilities() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let moving_avg = math::moving_average(&values, 3);
        assert_eq!(moving_avg.len(), 5);
        
        let std_dev = math::standard_deviation(&values);
        assert!(std_dev > 0.0);
        
        let p50 = math::percentile(&values, 50.0);
        assert_eq!(p50, 3.0);
        
        let normalized = math::normalize(&values);
        assert!(normalized.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }
    
    #[test]
    fn test_validation_utilities() {
        assert!(validation::is_valid_email("test@example.com"));
        assert!(!validation::is_valid_email("invalid-email"));
        
        assert!(validation::is_valid_port(8080));
        assert!(!validation::is_valid_port(0));
        assert!(!validation::is_valid_port(70000));
        
        assert!(validation::is_valid_url("https://example.com"));
        assert!(!validation::is_valid_url("not-a-url"));
    }
    
    #[test]
    fn test_circular_buffer() {
        let mut buffer = collections::CircularBuffer::new(3);
        
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);
        assert_eq!(buffer.len(), 3);
        
        buffer.push(4); // Should evict 1
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.get(0), Some(&2));
    }
    
    #[test]
    fn test_lru_cache() {
        let mut cache = collections::LruCache::new(2);
        
        cache.insert("a", 1);
        cache.insert("b", 2);
        assert_eq!(cache.get(&"a"), Some(&1));
        
        cache.insert("c", 3); // Should evict "b"
        assert_eq!(cache.get(&"b"), None);
        assert_eq!(cache.get(&"a"), Some(&1));
        assert_eq!(cache.get(&"c"), Some(&3));
    }
    
    #[tokio::test]
    async fn test_atomic_counter() {
        let counter = concurrency::AtomicCounter::new(0);
        
        assert_eq!(counter.get(), 0);
        counter.increment();
        assert_eq!(counter.get(), 1);
        counter.decrement();
        assert_eq!(counter.get(), 0);
    }
}