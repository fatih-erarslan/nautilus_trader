use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, mpsc};
use tracing::{info, warn, error, debug, instrument, Level};
use regex::Regex;

/// Log Level Enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl From<Level> for LogLevel {
    fn from(level: Level) -> Self {
        match level {
            Level::ERROR => LogLevel::Error,
            Level::WARN => LogLevel::Warn,
            Level::INFO => LogLevel::Info,
            Level::DEBUG => LogLevel::Debug,
            Level::TRACE => LogLevel::Trace,
        }
    }
}

/// Structured Log Entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: u64,
    pub level: LogLevel,
    pub component: String,
    pub message: String,
    pub metadata: HashMap<String, String>,
    pub trace_id: Option<String>,
    pub span_id: Option<String>,
    pub thread_id: String,
    pub source_file: Option<String>,
    pub source_line: Option<u32>,
}

/// Log Search Query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogSearchQuery {
    pub start_time: Option<u64>,
    pub end_time: Option<u64>,
    pub levels: Vec<LogLevel>,
    pub components: Vec<String>,
    pub text_pattern: Option<String>,
    pub trace_id: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// Log Analytics Result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogAnalytics {
    pub total_entries: usize,
    pub level_distribution: HashMap<LogLevel, usize>,
    pub component_distribution: HashMap<String, usize>,
    pub error_patterns: Vec<ErrorPattern>,
    pub performance_insights: Vec<PerformanceInsight>,
    pub anomaly_detections: Vec<LogAnomaly>,
    pub timeline: Vec<TimelineBucket>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPattern {
    pub pattern: String,
    pub occurrences: usize,
    pub first_seen: u64,
    pub last_seen: u64,
    pub affected_components: Vec<String>,
    pub severity_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceInsight {
    pub component: String,
    pub operation: String,
    pub avg_duration_ms: f64,
    pub max_duration_ms: f64,
    pub occurrences: usize,
    pub performance_trend: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogAnomaly {
    pub anomaly_type: String,
    pub component: String,
    pub description: String,
    pub confidence_score: f64,
    pub detected_at: u64,
    pub related_entries: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineBucket {
    pub timestamp: u64,
    pub error_count: usize,
    pub warn_count: usize,
    pub info_count: usize,
    pub debug_count: usize,
}

/// Log Aggregation and Analysis Engine
pub struct LogAggregationEngine {
    log_buffer: Arc<RwLock<VecDeque<LogEntry>>>,
    log_index: Arc<RwLock<HashMap<String, Vec<usize>>>>,
    component_index: Arc<RwLock<HashMap<String, Vec<usize>>>>,
    error_patterns: Arc<RwLock<HashMap<String, ErrorPattern>>>,
    performance_cache: Arc<RwLock<HashMap<String, PerformanceInsight>>>,
    max_buffer_size: usize,
    log_receiver: Arc<RwLock<Option<mpsc::Receiver<LogEntry>>>>,
    analytics_cache: Arc<RwLock<Option<LogAnalytics>>>,
    
    // Compiled regex patterns for common errors
    error_regex_patterns: Vec<(String, Regex)>,
    performance_regex_patterns: Vec<(String, Regex)>,
}

impl LogAggregationEngine {
    pub fn new(max_buffer_size: usize) -> Result<Self> {
        // Compile common error patterns
        let error_patterns = vec![
            ("network_timeout".to_string(), Regex::new(r"(?i)timeout|timed out|connection.*timeout")?),
            ("memory_error".to_string(), Regex::new(r"(?i)out of memory|memory allocation|oom")?),
            ("database_error".to_string(), Regex::new(r"(?i)database.*error|sql.*error|connection.*refused")?),
            ("authentication_error".to_string(), Regex::new(r"(?i)authentication.*failed|unauthorized|invalid.*credentials")?),
            ("api_error".to_string(), Regex::new(r"(?i)api.*error|http.*[45]\d\d|internal.*server.*error")?),
            ("neural_error".to_string(), Regex::new(r"(?i)neural.*error|inference.*failed|model.*error")?),
            ("trading_error".to_string(), Regex::new(r"(?i)order.*failed|trade.*rejected|market.*error")?),
        ];
        
        // Compile performance patterns
        let performance_patterns = vec![
            ("slow_query".to_string(), Regex::new(r"(?i)query.*took|slow.*query|duration.*(\d+)ms")?),
            ("high_latency".to_string(), Regex::new(r"(?i)latency.*(\d+)ms|response.*time.*(\d+)ms")?),
            ("inference_time".to_string(), Regex::new(r"(?i)inference.*(\d+\.?\d*)ms|neural.*processing.*(\d+\.?\d*)ms")?),
            ("order_processing".to_string(), Regex::new(r"(?i)order.*processed.*(\d+\.?\d*)ms|trade.*execution.*(\d+\.?\d*)ms")?),
        ];
        
        Ok(Self {
            log_buffer: Arc::new(RwLock::new(VecDeque::with_capacity(max_buffer_size))),
            log_index: Arc::new(RwLock::new(HashMap::new())),
            component_index: Arc::new(RwLock::new(HashMap::new())),
            error_patterns: Arc::new(RwLock::new(HashMap::new())),
            performance_cache: Arc::new(RwLock::new(HashMap::new())),
            max_buffer_size,
            log_receiver: Arc::new(RwLock::new(None)),
            analytics_cache: Arc::new(RwLock::new(None)),
            error_regex_patterns: error_patterns,
            performance_regex_patterns: performance_patterns,
        })
    }
    
    /// Ingest a new log entry
    #[instrument(skip(self, entry))]
    pub async fn ingest_log(&self, mut entry: LogEntry) -> Result<()> {
        // Ensure timestamp is set
        if entry.timestamp == 0 {
            entry.timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;
        }
        
        let mut buffer = self.log_buffer.write().await;
        let entry_index = buffer.len();
        
        // Maintain buffer size limit
        if buffer.len() >= self.max_buffer_size {
            // Remove oldest entry
            if let Some(old_entry) = buffer.pop_front() {
                self.remove_from_indices(&old_entry, 0).await?;
            }
        }
        
        // Add to buffer
        buffer.push_back(entry.clone());
        drop(buffer);
        
        // Update indices
        self.update_indices(&entry, entry_index).await?;
        
        // Analyze for patterns
        self.analyze_entry(&entry).await?;
        
        // Invalidate analytics cache
        *self.analytics_cache.write().await = None;
        
        debug!("Ingested log entry: {} - {}", entry.component, entry.message);
        Ok(())
    }
    
    /// Search logs based on query
    #[instrument(skip(self))]
    pub async fn search_logs(&self, query: LogSearchQuery) -> Result<Vec<LogEntry>> {
        let buffer = self.log_buffer.read().await;
        let mut results = Vec::new();
        
        let start_time = query.start_time.unwrap_or(0);
        let end_time = query.end_time.unwrap_or(u64::MAX);
        let limit = query.limit.unwrap_or(1000);
        let offset = query.offset.unwrap_or(0);
        
        // Compile text pattern if provided
        let text_regex = if let Some(pattern) = &query.text_pattern {
            Some(Regex::new(pattern)?)
        } else {
            None
        };
        
        let mut matched_count = 0;
        let mut skipped_count = 0;
        
        for entry in buffer.iter().rev() { // Search newest first
            // Time range filter
            if entry.timestamp < start_time || entry.timestamp > end_time {
                continue;
            }
            
            // Level filter
            if !query.levels.is_empty() && !query.levels.contains(&entry.level) {
                continue;
            }
            
            // Component filter
            if !query.components.is_empty() && !query.components.contains(&entry.component) {
                continue;
            }
            
            // Trace ID filter
            if let Some(trace_id) = &query.trace_id {
                if entry.trace_id.as_ref() != Some(trace_id) {
                    continue;
                }
            }
            
            // Text pattern filter
            if let Some(regex) = &text_regex {
                if !regex.is_match(&entry.message) {
                    continue;
                }
            }
            
            // Apply offset
            if skipped_count < offset {
                skipped_count += 1;
                continue;
            }
            
            results.push(entry.clone());
            matched_count += 1;
            
            if matched_count >= limit {
                break;
            }
        }
        
        info!("Log search completed: {} results, query: {:?}", results.len(), query);
        Ok(results)
    }
    
    /// Generate comprehensive log analytics
    #[instrument(skip(self))]
    pub async fn generate_analytics(&self) -> Result<LogAnalytics> {
        // Check cache first
        if let Some(cached) = self.analytics_cache.read().await.as_ref() {
            return Ok(cached.clone());
        }
        
        let buffer = self.log_buffer.read().await;
        let mut level_distribution = HashMap::new();
        let mut component_distribution = HashMap::new();
        let mut timeline = Vec::new();
        
        // Initialize counters
        for level in [LogLevel::Error, LogLevel::Warn, LogLevel::Info, LogLevel::Debug, LogLevel::Trace] {
            level_distribution.insert(level, 0);
        }
        
        // Analyze all entries
        for entry in buffer.iter() {
            // Level distribution
            *level_distribution.entry(entry.level.clone()).or_insert(0) += 1;
            
            // Component distribution
            *component_distribution.entry(entry.component.clone()).or_insert(0) += 1;
        }
        
        // Generate timeline (hourly buckets for last 24 hours)
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;
        let hour_ms = 3600 * 1000;
        
        for i in 0..24 {
            let bucket_start = now - (24 - i) * hour_ms;
            let bucket_end = bucket_start + hour_ms;
            
            let mut error_count = 0;
            let mut warn_count = 0;
            let mut info_count = 0;
            let mut debug_count = 0;
            
            for entry in buffer.iter() {
                if entry.timestamp >= bucket_start && entry.timestamp < bucket_end {
                    match entry.level {
                        LogLevel::Error => error_count += 1,
                        LogLevel::Warn => warn_count += 1,
                        LogLevel::Info => info_count += 1,
                        LogLevel::Debug | LogLevel::Trace => debug_count += 1,
                    }
                }
            }
            
            timeline.push(TimelineBucket {
                timestamp: bucket_start,
                error_count,
                warn_count,
                info_count,
                debug_count,
            });
        }
        
        // Get error patterns and performance insights
        let error_patterns = self.error_patterns.read().await.values().cloned().collect();
        let performance_insights = self.performance_cache.read().await.values().cloned().collect();
        
        // Detect anomalies
        let anomaly_detections = self.detect_anomalies(&buffer).await?;
        
        let analytics = LogAnalytics {
            total_entries: buffer.len(),
            level_distribution,
            component_distribution,
            error_patterns,
            performance_insights,
            anomaly_detections,
            timeline,
        };
        
        // Cache the results
        *self.analytics_cache.write().await = Some(analytics.clone());
        
        info!("Generated log analytics: {} total entries, {} error patterns", 
              analytics.total_entries, analytics.error_patterns.len());
        
        Ok(analytics)
    }
    
    /// Update search indices
    async fn update_indices(&self, entry: &LogEntry, index: usize) -> Result<()> {
        // Update text index
        let mut log_index = self.log_index.write().await;
        let words: Vec<String> = entry.message
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();
        
        for word in words {
            log_index.entry(word).or_insert_with(Vec::new).push(index);
        }
        
        // Update component index
        let mut component_index = self.component_index.write().await;
        component_index
            .entry(entry.component.clone())
            .or_insert_with(Vec::new)
            .push(index);
        
        Ok(())
    }
    
    /// Remove entry from indices (when buffer rotates)
    async fn remove_from_indices(&self, entry: &LogEntry, index: usize) -> Result<()> {
        // Remove from text index
        let mut log_index = self.log_index.write().await;
        let words: Vec<String> = entry.message
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();
        
        for word in words {
            if let Some(indices) = log_index.get_mut(&word) {
                indices.retain(|&i| i != index);
                if indices.is_empty() {
                    log_index.remove(&word);
                }
            }
        }
        
        // Remove from component index
        let mut component_index = self.component_index.write().await;
        if let Some(indices) = component_index.get_mut(&entry.component) {
            indices.retain(|&i| i != index);
            if indices.is_empty() {
                component_index.remove(&entry.component);
            }
        }
        
        Ok(())
    }
    
    /// Analyze log entry for patterns
    async fn analyze_entry(&self, entry: &LogEntry) -> Result<()> {
        // Skip analysis for debug/trace levels
        if matches!(entry.level, LogLevel::Debug | LogLevel::Trace) {
            return Ok(());
        }
        
        // Analyze error patterns
        if matches!(entry.level, LogLevel::Error | LogLevel::Warn) {
            for (pattern_name, regex) in &self.error_regex_patterns {
                if regex.is_match(&entry.message) {
                    let mut patterns = self.error_patterns.write().await;
                    let pattern = patterns.entry(pattern_name.clone()).or_insert_with(|| {
                        ErrorPattern {
                            pattern: pattern_name.clone(),
                            occurrences: 0,
                            first_seen: entry.timestamp,
                            last_seen: entry.timestamp,
                            affected_components: Vec::new(),
                            severity_score: 0.0,
                        }
                    });
                    
                    pattern.occurrences += 1;
                    pattern.last_seen = entry.timestamp;
                    pattern.severity_score = match entry.level {
                        LogLevel::Error => pattern.severity_score + 1.0,
                        LogLevel::Warn => pattern.severity_score + 0.5,
                        _ => pattern.severity_score,
                    };
                    
                    if !pattern.affected_components.contains(&entry.component) {
                        pattern.affected_components.push(entry.component.clone());
                    }
                    
                    break;
                }
            }
        }
        
        // Analyze performance patterns
        for (pattern_name, regex) in &self.performance_regex_patterns {
            if let Some(captures) = regex.captures(&entry.message) {
                if let Some(duration_str) = captures.get(1).or_else(|| captures.get(2)) {
                    if let Ok(duration) = duration_str.as_str().parse::<f64>() {
                        let mut cache = self.performance_cache.write().await;
                        let key = format!("{}_{}", entry.component, pattern_name);
                        
                        let insight = cache.entry(key).or_insert_with(|| {
                            PerformanceInsight {
                                component: entry.component.clone(),
                                operation: pattern_name.clone(),
                                avg_duration_ms: duration,
                                max_duration_ms: duration,
                                occurrences: 0,
                                performance_trend: "stable".to_string(),
                            }
                        });
                        
                        insight.occurrences += 1;
                        insight.avg_duration_ms = (insight.avg_duration_ms * (insight.occurrences - 1) as f64 + duration) / insight.occurrences as f64;
                        insight.max_duration_ms = insight.max_duration_ms.max(duration);
                        
                        // Determine trend
                        if duration > insight.avg_duration_ms * 1.5 {
                            insight.performance_trend = "degrading".to_string();
                        } else if duration < insight.avg_duration_ms * 0.8 {
                            insight.performance_trend = "improving".to_string();
                        }
                        
                        break;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Detect log anomalies
    async fn detect_anomalies(&self, buffer: &VecDeque<LogEntry>) -> Result<Vec<LogAnomaly>> {
        let mut anomalies = Vec::new();
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;
        let recent_window = 3600 * 1000; // 1 hour
        
        // Analyze error rate spikes
        let recent_errors: Vec<_> = buffer
            .iter()
            .filter(|e| e.timestamp > now - recent_window && e.level == LogLevel::Error)
            .collect();
        
        if recent_errors.len() > 50 { // Threshold for error spike
            anomalies.push(LogAnomaly {
                anomaly_type: "error_spike".to_string(),
                component: "system".to_string(),
                description: format!("Detected {} errors in the last hour", recent_errors.len()),
                confidence_score: 0.9,
                detected_at: now,
                related_entries: recent_errors.iter().take(5).map(|e| e.message.clone()).collect(),
            });
        }
        
        // Analyze component silence (no logs for extended period)
        let mut component_last_seen = HashMap::new();
        for entry in buffer.iter().rev().take(1000) { // Check last 1000 entries
            component_last_seen.entry(entry.component.clone()).or_insert(entry.timestamp);
        }
        
        for (component, last_seen) in component_last_seen {
            if now - last_seen > 2 * 3600 * 1000 && component != "system" { // 2 hours silence
                anomalies.push(LogAnomaly {
                    anomaly_type: "component_silence".to_string(),
                    component: component.clone(),
                    description: format!("Component {} has been silent for over 2 hours", component),
                    confidence_score: 0.7,
                    detected_at: now,
                    related_entries: vec![format!("Last seen: {}", last_seen)],
                });
            }
        }
        
        // Analyze repeated error patterns
        let patterns = self.error_patterns.read().await;
        for pattern in patterns.values() {
            if pattern.occurrences > 20 && now - pattern.last_seen < 3600 * 1000 {
                anomalies.push(LogAnomaly {
                    anomaly_type: "repeated_error_pattern".to_string(),
                    component: pattern.affected_components.join(", "),
                    description: format!("Pattern '{}' occurred {} times recently", pattern.pattern, pattern.occurrences),
                    confidence_score: 0.8,
                    detected_at: now,
                    related_entries: vec![format!("Affected components: {:?}", pattern.affected_components)],
                });
            }
        }
        
        Ok(anomalies)
    }
    
    /// Export logs to external system (Elasticsearch, etc.)
    #[instrument(skip(self))]
    pub async fn export_logs(&self, start_time: u64, end_time: u64) -> Result<Vec<LogEntry>> {
        let query = LogSearchQuery {
            start_time: Some(start_time),
            end_time: Some(end_time),
            levels: vec![],
            components: vec![],
            text_pattern: None,
            trace_id: None,
            limit: None,
            offset: None,
        };
        
        self.search_logs(query).await
    }
    
    /// Get real-time log statistics
    pub async fn get_realtime_stats(&self) -> Result<HashMap<String, u64>> {
        let buffer = self.log_buffer.read().await;
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;
        let recent_window = 300 * 1000; // 5 minutes
        
        let mut stats = HashMap::new();
        
        let recent_logs: Vec<_> = buffer
            .iter()
            .filter(|e| e.timestamp > now - recent_window)
            .collect();
        
        stats.insert("total_recent".to_string(), recent_logs.len() as u64);
        stats.insert("errors_recent".to_string(), 
                     recent_logs.iter().filter(|e| e.level == LogLevel::Error).count() as u64);
        stats.insert("warnings_recent".to_string(), 
                     recent_logs.iter().filter(|e| e.level == LogLevel::Warn).count() as u64);
        stats.insert("buffer_size".to_string(), buffer.len() as u64);
        stats.insert("buffer_capacity".to_string(), self.max_buffer_size as u64);
        
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_log_ingestion() {
        let engine = LogAggregationEngine::new(1000).unwrap();
        
        let entry = LogEntry {
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64,
            level: LogLevel::Info,
            component: "test".to_string(),
            message: "Test log message".to_string(),
            metadata: HashMap::new(),
            trace_id: None,
            span_id: None,
            thread_id: "main".to_string(),
            source_file: None,
            source_line: None,
        };
        
        let result = engine.ingest_log(entry).await;
        assert!(result.is_ok());
        
        let stats = engine.get_realtime_stats().await.unwrap();
        assert_eq!(stats.get("total_recent"), Some(&1));
    }
    
    #[tokio::test]
    async fn test_log_search() {
        let engine = LogAggregationEngine::new(1000).unwrap();
        
        // Ingest test logs
        for i in 0..10 {
            let entry = LogEntry {
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64,
                level: if i % 2 == 0 { LogLevel::Error } else { LogLevel::Info },
                component: format!("component{}", i % 3),
                message: format!("Test message {}", i),
                metadata: HashMap::new(),
                trace_id: None,
                span_id: None,
                thread_id: "main".to_string(),
                source_file: None,
                source_line: None,
            };
            engine.ingest_log(entry).await.unwrap();
        }
        
        // Search for error logs
        let query = LogSearchQuery {
            start_time: None,
            end_time: None,
            levels: vec![LogLevel::Error],
            components: vec![],
            text_pattern: None,
            trace_id: None,
            limit: Some(10),
            offset: None,
        };
        
        let results = engine.search_logs(query).await.unwrap();
        assert_eq!(results.len(), 5); // Should find 5 error logs
    }
}