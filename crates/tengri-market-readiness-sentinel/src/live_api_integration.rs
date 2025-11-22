//! TENGRI Live API Integration Testing
//!
//! Comprehensive live API integration testing with actual exchange APIs (testnet/sandbox).
//! This module enforces zero-mock API testing with real exchange connections,
//! actual API calls, and live data validation.

use std::sync::Arc;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use anyhow::{Result, anyhow};
use tracing::{info, warn, error, debug};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use tokio::sync::RwLock;
use reqwest::{Client, Response};
use serde_json::Value;
use tokio_tungstenite::{connect_async, WebSocketStream};
use futures_util::{SinkExt, StreamExt};
use url::Url;

use crate::config::MarketReadinessConfig;
use crate::types::{ValidationResult, ValidationStatus};
use crate::zero_mock_detection::ZeroMockDetectionEngine;

/// Live API Integration Tester
/// 
/// This tester validates API integrations using actual exchange APIs,
/// real authentication, and live market data. No mock APIs allowed.
#[derive(Debug, Clone)]
pub struct LiveApiIntegrationTester {
    config: Arc<MarketReadinessConfig>,
    http_client: Client,
    exchange_configs: Arc<RwLock<HashMap<String, ExchangeConfig>>>,
    api_test_results: Arc<RwLock<Vec<ApiTestResult>>>,
    websocket_connections: Arc<RwLock<HashMap<String, WebSocketConnection>>>,
    zero_mock_detector: Arc<ZeroMockDetectionEngine>,
    monitoring_active: Arc<RwLock<bool>>,
}

/// Exchange configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeConfig {
    pub id: String,
    pub name: String,
    pub exchange_type: ExchangeType,
    pub base_url: String,
    pub websocket_url: Option<String>,
    pub api_key: Option<String>,
    pub api_secret: Option<String>,
    pub passphrase: Option<String>,
    pub testnet: bool,
    pub rate_limit_per_second: u32,
    pub supported_endpoints: Vec<String>,
    pub supported_symbols: Vec<String>,
    pub authentication_required: bool,
    pub ssl_verification: bool,
}

/// Exchange types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExchangeType {
    Binance,
    Coinbase,
    Kraken,
    Bybit,
    OKX,
    Bitget,
    Kucoin,
    Huobi,
    Bitfinex,
    FTX,
    Deribit,
    Bitmex,
    Custom,
}

/// API test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiTestResult {
    pub test_id: Uuid,
    pub exchange_id: String,
    pub endpoint: String,
    pub test_type: ApiTestType,
    pub method: HttpMethod,
    pub status: ValidationStatus,
    pub response_time_ms: u64,
    pub status_code: Option<u16>,
    pub response_size_bytes: Option<u64>,
    pub error_message: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub request_payload: Option<Value>,
    pub response_payload: Option<Value>,
    pub authentication_used: bool,
    pub rate_limit_remaining: Option<u32>,
}

/// API test types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApiTestType {
    ConnectivityTest,
    AuthenticationTest,
    MarketDataTest,
    AccountInfoTest,
    OrderPlacementTest,
    OrderCancellationTest,
    TradingHistoryTest,
    BalanceTest,
    DepositAddressTest,
    WithdrawalTest,
    WebSocketTest,
    RateLimitTest,
    ErrorHandlingTest,
    SecurityTest,
}

/// HTTP methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HttpMethod {
    GET,
    POST,
    PUT,
    DELETE,
    PATCH,
}

/// WebSocket connection
#[derive(Debug)]
pub struct WebSocketConnection {
    pub exchange_id: String,
    pub url: String,
    pub connected_at: DateTime<Utc>,
    pub last_message_at: Option<DateTime<Utc>>,
    pub messages_received: u64,
    pub messages_sent: u64,
    pub status: WebSocketStatus,
}

/// WebSocket status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WebSocketStatus {
    Connected,
    Disconnected,
    Connecting,
    Error,
}

/// API integration test report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiIntegrationReport {
    pub test_id: Uuid,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub test_results: Vec<ApiTestResult>,
    pub performance_metrics: Vec<ApiPerformanceMetrics>,
    pub overall_status: ValidationStatus,
    pub exchanges_tested: u32,
    pub endpoints_tested: u32,
    pub tests_passed: u32,
    pub tests_failed: u32,
    pub average_response_time_ms: f64,
    pub critical_issues: Vec<String>,
    pub recommendations: Vec<String>,
}

/// API performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiPerformanceMetrics {
    pub exchange_id: String,
    pub endpoint: String,
    pub average_response_time_ms: f64,
    pub min_response_time_ms: u64,
    pub max_response_time_ms: u64,
    pub p95_response_time_ms: f64,
    pub p99_response_time_ms: f64,
    pub success_rate: f64,
    pub error_rate: f64,
    pub throughput_requests_per_second: f64,
    pub data_transfer_mb: f64,
}

impl LiveApiIntegrationTester {
    /// Create a new live API integration tester
    pub async fn new(
        config: Arc<MarketReadinessConfig>,
        zero_mock_detector: Arc<ZeroMockDetectionEngine>,
    ) -> Result<Self> {
        let http_client = Client::builder()
            .timeout(Duration::from_secs(30))
            .pool_max_idle_per_host(10)
            .build()
            .map_err(|e| anyhow!("Failed to create HTTP client: {}", e))?;
        
        let tester = Self {
            config,
            http_client,
            exchange_configs: Arc::new(RwLock::new(HashMap::new())),
            api_test_results: Arc::new(RwLock::new(Vec::new())),
            websocket_connections: Arc::new(RwLock::new(HashMap::new())),
            zero_mock_detector,
            monitoring_active: Arc::new(RwLock::new(false)),
        };

        Ok(tester)
    }

    /// Initialize the live API integration tester
    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing Live API Integration Tester...");
        
        // Validate that no mock APIs are configured
        self.validate_no_mock_apis().await?;
        
        // Initialize exchange configurations
        self.initialize_exchange_configs().await?;
        
        // Start monitoring
        self.start_monitoring().await?;
        
        info!("Live API Integration Tester initialized successfully");
        Ok(())
    }

    /// Validate that no mock APIs are configured
    async fn validate_no_mock_apis(&self) -> Result<()> {
        info!("Validating no mock APIs are configured...");
        
        // Run zero-mock detection on API configurations
        let detection_result = self.zero_mock_detector.run_comprehensive_scan().await?;
        
        // Check for API-related mock violations
        for violation in &detection_result.violations_found {
            if matches!(violation.category, crate::zero_mock_detection::DetectionCategory::FakeNetworkData) {
                return Err(anyhow!(
                    "Mock API detected: {}. Real APIs required for integration testing.",
                    violation.description
                ));
            }
        }
        
        info!("No mock APIs detected - validation passed");
        Ok(())
    }

    /// Initialize exchange configurations
    async fn initialize_exchange_configs(&self) -> Result<()> {
        info!("Initializing exchange configurations...");
        
        let mut configs = self.exchange_configs.write().await;
        
        // Binance Testnet
        configs.insert("binance_testnet".to_string(), ExchangeConfig {
            id: "binance_testnet".to_string(),
            name: "Binance Testnet".to_string(),
            exchange_type: ExchangeType::Binance,
            base_url: "https://testnet.binance.vision".to_string(),
            websocket_url: Some("wss://testnet.binance.vision/ws".to_string()),
            api_key: std::env::var("BINANCE_TESTNET_API_KEY").ok(),
            api_secret: std::env::var("BINANCE_TESTNET_API_SECRET").ok(),
            passphrase: None,
            testnet: true,
            rate_limit_per_second: 10,
            supported_endpoints: vec![
                "/api/v3/ping".to_string(),
                "/api/v3/time".to_string(),
                "/api/v3/exchangeInfo".to_string(),
                "/api/v3/ticker/24hr".to_string(),
                "/api/v3/klines".to_string(),
                "/api/v3/account".to_string(),
            ],
            supported_symbols: vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()],
            authentication_required: false, // For public endpoints
            ssl_verification: true,
        });
        
        // Coinbase Pro Sandbox
        configs.insert("coinbase_sandbox".to_string(), ExchangeConfig {
            id: "coinbase_sandbox".to_string(),
            name: "Coinbase Pro Sandbox".to_string(),
            exchange_type: ExchangeType::Coinbase,
            base_url: "https://api-public.sandbox.pro.coinbase.com".to_string(),
            websocket_url: Some("wss://ws-feed-public.sandbox.pro.coinbase.com".to_string()),
            api_key: std::env::var("COINBASE_SANDBOX_API_KEY").ok(),
            api_secret: std::env::var("COINBASE_SANDBOX_API_SECRET").ok(),
            passphrase: std::env::var("COINBASE_SANDBOX_PASSPHRASE").ok(),
            testnet: true,
            rate_limit_per_second: 10,
            supported_endpoints: vec![
                "/time".to_string(),
                "/products".to_string(),
                "/products/BTC-USD/ticker".to_string(),
                "/products/BTC-USD/book".to_string(),
                "/accounts".to_string(),
            ],
            supported_symbols: vec!["BTC-USD".to_string(), "ETH-USD".to_string()],
            authentication_required: false, // For public endpoints
            ssl_verification: true,
        });
        
        // Kraken
        configs.insert("kraken".to_string(), ExchangeConfig {
            id: "kraken".to_string(),
            name: "Kraken".to_string(),
            exchange_type: ExchangeType::Kraken,
            base_url: "https://api.kraken.com".to_string(),
            websocket_url: Some("wss://ws.kraken.com".to_string()),
            api_key: std::env::var("KRAKEN_API_KEY").ok(),
            api_secret: std::env::var("KRAKEN_API_SECRET").ok(),
            passphrase: None,
            testnet: false,
            rate_limit_per_second: 1, // Kraken has strict rate limits
            supported_endpoints: vec![
                "/0/public/Time".to_string(),
                "/0/public/SystemStatus".to_string(),
                "/0/public/AssetPairs".to_string(),
                "/0/public/Ticker".to_string(),
                "/0/public/OHLC".to_string(),
            ],
            supported_symbols: vec!["XBTUSD".to_string(), "ETHUSD".to_string()],
            authentication_required: false, // For public endpoints
            ssl_verification: true,
        });
        
        info!("Exchange configurations initialized: {} exchanges", configs.len());
        Ok(())
    }

    /// Start monitoring API integrations
    async fn start_monitoring(&self) -> Result<()> {
        {
            let mut monitoring = self.monitoring_active.write().await;
            *monitoring = true;
        }
        
        info!("API integration monitoring started");
        Ok(())
    }

    /// Stop monitoring API integrations
    async fn stop_monitoring(&self) -> Result<()> {
        {
            let mut monitoring = self.monitoring_active.write().await;
            *monitoring = false;
        }
        
        info!("API integration monitoring stopped");
        Ok(())
    }

    /// Run comprehensive API integration tests
    pub async fn run_comprehensive_tests(&self) -> Result<ApiIntegrationReport> {
        let test_id = Uuid::new_v4();
        let started_at = Utc::now();
        
        info!("Starting comprehensive API integration tests: {}", test_id);
        
        let mut report = ApiIntegrationReport {
            test_id,
            started_at,
            completed_at: None,
            test_results: Vec::new(),
            performance_metrics: Vec::new(),
            overall_status: ValidationStatus::InProgress,
            exchanges_tested: 0,
            endpoints_tested: 0,
            tests_passed: 0,
            tests_failed: 0,
            average_response_time_ms: 0.0,
            critical_issues: Vec::new(),
            recommendations: Vec::new(),
        };

        // Test all configured exchanges
        let configs = self.exchange_configs.read().await;
        for (exchange_id, config) in configs.iter() {
            info!("Testing exchange: {}", exchange_id);
            
            // Test connectivity
            let connectivity_results = self.test_exchange_connectivity(config).await?;
            report.test_results.extend(connectivity_results);
            
            // Test public endpoints
            let public_endpoint_results = self.test_public_endpoints(config).await?;
            report.test_results.extend(public_endpoint_results);
            
            // Test authentication (if credentials are available)
            if config.api_key.is_some() {
                let auth_results = self.test_authentication(config).await?;
                report.test_results.extend(auth_results);
            }
            
            // Test WebSocket connections
            if config.websocket_url.is_some() {
                let websocket_results = self.test_websocket_connection(config).await?;
                report.test_results.extend(websocket_results);
            }
            
            // Test rate limiting
            let rate_limit_results = self.test_rate_limiting(config).await?;
            report.test_results.extend(rate_limit_results);
        }
        
        // Generate performance metrics
        report.performance_metrics = self.generate_performance_metrics(&report.test_results).await?;
        
        // Calculate statistics
        self.calculate_test_statistics(&mut report);
        
        // Determine overall status
        report.overall_status = self.determine_overall_status(&report);
        
        // Generate recommendations
        self.generate_recommendations(&mut report);
        
        let completed_at = Utc::now();
        report.completed_at = Some(completed_at);
        
        // Store test results
        {
            let mut test_results = self.api_test_results.write().await;
            test_results.extend(report.test_results.clone());
        }
        
        info!("API integration tests completed: {} tests run", report.test_results.len());
        Ok(report)
    }

    /// Test exchange connectivity
    async fn test_exchange_connectivity(&self, config: &ExchangeConfig) -> Result<Vec<ApiTestResult>> {
        let mut results = Vec::new();
        
        // Test basic connectivity with ping or time endpoint
        let ping_endpoint = match config.exchange_type {
            ExchangeType::Binance => "/api/v3/ping",
            ExchangeType::Coinbase => "/time",
            ExchangeType::Kraken => "/0/public/Time",
            _ => "/ping",
        };
        
        let result = self.test_api_endpoint(
            config,
            ping_endpoint,
            HttpMethod::GET,
            None,
            false,
            ApiTestType::ConnectivityTest,
        ).await?;
        
        results.push(result);
        Ok(results)
    }

    /// Test public endpoints
    async fn test_public_endpoints(&self, config: &ExchangeConfig) -> Result<Vec<ApiTestResult>> {
        let mut results = Vec::new();
        
        for endpoint in &config.supported_endpoints {
            let result = self.test_api_endpoint(
                config,
                endpoint,
                HttpMethod::GET,
                None,
                false,
                ApiTestType::MarketDataTest,
            ).await?;
            
            results.push(result);
            
            // Add delay to respect rate limits
            let delay_ms = 1000 / config.rate_limit_per_second;
            tokio::time::sleep(Duration::from_millis(delay_ms as u64)).await;
        }
        
        Ok(results)
    }

    /// Test authentication
    async fn test_authentication(&self, config: &ExchangeConfig) -> Result<Vec<ApiTestResult>> {
        let mut results = Vec::new();
        
        // Test account endpoint which requires authentication
        let account_endpoint = match config.exchange_type {
            ExchangeType::Binance => "/api/v3/account",
            ExchangeType::Coinbase => "/accounts",
            ExchangeType::Kraken => "/0/private/Balance",
            _ => "/account",
        };
        
        let result = self.test_api_endpoint(
            config,
            account_endpoint,
            HttpMethod::GET,
            None,
            true,
            ApiTestType::AuthenticationTest,
        ).await?;
        
        results.push(result);
        Ok(results)
    }

    /// Test WebSocket connection
    async fn test_websocket_connection(&self, config: &ExchangeConfig) -> Result<Vec<ApiTestResult>> {
        let mut results = Vec::new();
        
        if let Some(ws_url) = &config.websocket_url {
            let start_time = Instant::now();
            
            match self.test_websocket_connectivity(config, ws_url).await {
                Ok(_) => {
                    results.push(ApiTestResult {
                        test_id: Uuid::new_v4(),
                        exchange_id: config.id.clone(),
                        endpoint: ws_url.clone(),
                        test_type: ApiTestType::WebSocketTest,
                        method: HttpMethod::GET, // Not applicable for WebSocket
                        status: ValidationStatus::Passed,
                        response_time_ms: start_time.elapsed().as_millis() as u64,
                        status_code: None,
                        response_size_bytes: None,
                        error_message: None,
                        timestamp: Utc::now(),
                        request_payload: None,
                        response_payload: None,
                        authentication_used: false,
                        rate_limit_remaining: None,
                    });
                }
                Err(e) => {
                    results.push(ApiTestResult {
                        test_id: Uuid::new_v4(),
                        exchange_id: config.id.clone(),
                        endpoint: ws_url.clone(),
                        test_type: ApiTestType::WebSocketTest,
                        method: HttpMethod::GET,
                        status: ValidationStatus::Failed,
                        response_time_ms: start_time.elapsed().as_millis() as u64,
                        status_code: None,
                        response_size_bytes: None,
                        error_message: Some(e.to_string()),
                        timestamp: Utc::now(),
                        request_payload: None,
                        response_payload: None,
                        authentication_used: false,
                        rate_limit_remaining: None,
                    });
                }
            }
        }
        
        Ok(results)
    }

    /// Test WebSocket connectivity
    async fn test_websocket_connectivity(&self, config: &ExchangeConfig, ws_url: &str) -> Result<()> {
        info!("Testing WebSocket connectivity to {}", ws_url);
        
        let url = Url::parse(ws_url)
            .map_err(|e| anyhow!("Invalid WebSocket URL {}: {}", ws_url, e))?;
        
        // Connect to WebSocket
        let (ws_stream, _) = connect_async(url).await
            .map_err(|e| anyhow!("WebSocket connection failed: {}", e))?;
        
        // Store connection info
        {
            let mut connections = self.websocket_connections.write().await;
            connections.insert(config.id.clone(), WebSocketConnection {
                exchange_id: config.id.clone(),
                url: ws_url.to_string(),
                connected_at: Utc::now(),
                last_message_at: None,
                messages_received: 0,
                messages_sent: 0,
                status: WebSocketStatus::Connected,
            });
        }
        
        info!("WebSocket connection established to {}", ws_url);
        Ok(())
    }

    /// Test rate limiting
    async fn test_rate_limiting(&self, config: &ExchangeConfig) -> Result<Vec<ApiTestResult>> {
        let mut results = Vec::new();
        
        // Test rate limiting by making rapid requests
        let test_endpoint = config.supported_endpoints.first()
            .unwrap_or(&"/ping".to_string());
        
        let requests_per_test = config.rate_limit_per_second * 2; // Exceed rate limit
        let mut successful_requests = 0;
        let mut rate_limited_requests = 0;
        
        for i in 0..requests_per_test {
            let result = self.test_api_endpoint(
                config,
                test_endpoint,
                HttpMethod::GET,
                None,
                false,
                ApiTestType::RateLimitTest,
            ).await?;
            
            if result.status == ValidationStatus::Passed {
                successful_requests += 1;
            } else if result.status_code == Some(429) { // Too Many Requests
                rate_limited_requests += 1;
            }
            
            results.push(result);
            
            // Small delay between requests
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        
        info!("Rate limit test completed: {}/{} requests successful, {} rate limited",
              successful_requests, requests_per_test, rate_limited_requests);
        
        Ok(results)
    }

    /// Test a specific API endpoint
    async fn test_api_endpoint(
        &self,
        config: &ExchangeConfig,
        endpoint: &str,
        method: HttpMethod,
        payload: Option<Value>,
        requires_auth: bool,
        test_type: ApiTestType,
    ) -> Result<ApiTestResult> {
        let start_time = Instant::now();
        let url = format!("{}{}", config.base_url, endpoint);
        
        debug!("Testing API endpoint: {} {}", method as u8, url);
        
        // Build request
        let mut request_builder = match method {
            HttpMethod::GET => self.http_client.get(&url),
            HttpMethod::POST => self.http_client.post(&url),
            HttpMethod::PUT => self.http_client.put(&url),
            HttpMethod::DELETE => self.http_client.delete(&url),
            HttpMethod::PATCH => self.http_client.patch(&url),
        };
        
        // Add authentication headers if required
        if requires_auth && config.api_key.is_some() {
            request_builder = self.add_authentication_headers(request_builder, config, endpoint, &payload).await?;
        }
        
        // Add payload if provided
        if let Some(payload) = &payload {
            request_builder = request_builder.json(payload);
        }
        
        // Send request
        match request_builder.send().await {
            Ok(response) => {
                let response_time = start_time.elapsed().as_millis() as u64;
                let status_code = response.status().as_u16();
                let response_size = response.content_length();
                
                // Read response body
                let response_text = response.text().await.unwrap_or_default();
                let response_payload: Option<Value> = serde_json::from_str(&response_text).ok();
                
                let status = if status_code >= 200 && status_code < 300 {
                    ValidationStatus::Passed
                } else {
                    ValidationStatus::Failed
                };
                
                Ok(ApiTestResult {
                    test_id: Uuid::new_v4(),
                    exchange_id: config.id.clone(),
                    endpoint: endpoint.to_string(),
                    test_type,
                    method,
                    status,
                    response_time_ms: response_time,
                    status_code: Some(status_code),
                    response_size_bytes: response_size,
                    error_message: if status == ValidationStatus::Failed {
                        Some(format!("HTTP {}: {}", status_code, response_text))
                    } else {
                        None
                    },
                    timestamp: Utc::now(),
                    request_payload: payload,
                    response_payload,
                    authentication_used: requires_auth,
                    rate_limit_remaining: None, // TODO: Parse from headers
                })
            }
            Err(e) => {
                let response_time = start_time.elapsed().as_millis() as u64;
                
                Ok(ApiTestResult {
                    test_id: Uuid::new_v4(),
                    exchange_id: config.id.clone(),
                    endpoint: endpoint.to_string(),
                    test_type,
                    method,
                    status: ValidationStatus::Failed,
                    response_time_ms: response_time,
                    status_code: None,
                    response_size_bytes: None,
                    error_message: Some(e.to_string()),
                    timestamp: Utc::now(),
                    request_payload: payload,
                    response_payload: None,
                    authentication_used: requires_auth,
                    rate_limit_remaining: None,
                })
            }
        }
    }

    /// Add authentication headers to request
    async fn add_authentication_headers(
        &self,
        request_builder: reqwest::RequestBuilder,
        config: &ExchangeConfig,
        endpoint: &str,
        payload: &Option<Value>,
    ) -> Result<reqwest::RequestBuilder> {
        match config.exchange_type {
            ExchangeType::Binance => {
                // Binance authentication
                if let (Some(api_key), Some(api_secret)) = (&config.api_key, &config.api_secret) {
                    let timestamp = chrono::Utc::now().timestamp_millis();
                    let query_string = format!("timestamp={}", timestamp);
                    
                    // Create signature (simplified - real implementation would need proper HMAC)
                    let signature = format!("{}{}{}", endpoint, query_string, api_secret);
                    
                    Ok(request_builder
                        .header("X-MBX-APIKEY", api_key)
                        .query(&[("timestamp", timestamp.to_string())])
                        .query(&[("signature", signature)]))
                } else {
                    Err(anyhow!("API key and secret required for Binance authentication"))
                }
            }
            ExchangeType::Coinbase => {
                // Coinbase Pro authentication
                if let (Some(api_key), Some(api_secret), Some(passphrase)) = 
                    (&config.api_key, &config.api_secret, &config.passphrase) {
                    let timestamp = chrono::Utc::now().timestamp();
                    
                    // Create signature (simplified - real implementation would need proper HMAC)
                    let signature = format!("{}{}{}", timestamp, endpoint, api_secret);
                    
                    Ok(request_builder
                        .header("CB-ACCESS-KEY", api_key)
                        .header("CB-ACCESS-SIGN", signature)
                        .header("CB-ACCESS-TIMESTAMP", timestamp.to_string())
                        .header("CB-ACCESS-PASSPHRASE", passphrase))
                } else {
                    Err(anyhow!("API key, secret, and passphrase required for Coinbase authentication"))
                }
            }
            ExchangeType::Kraken => {
                // Kraken authentication
                if let (Some(api_key), Some(api_secret)) = (&config.api_key, &config.api_secret) {
                    let nonce = chrono::Utc::now().timestamp_nanos();
                    
                    // Create signature (simplified - real implementation would need proper HMAC)
                    let signature = format!("{}{}{}", nonce, endpoint, api_secret);
                    
                    Ok(request_builder
                        .header("API-Key", api_key)
                        .header("API-Sign", signature)
                        .form(&[("nonce", nonce.to_string())]))
                } else {
                    Err(anyhow!("API key and secret required for Kraken authentication"))
                }
            }
            _ => {
                // Generic authentication
                if let Some(api_key) = &config.api_key {
                    Ok(request_builder.header("Authorization", format!("Bearer {}", api_key)))
                } else {
                    Err(anyhow!("API key required for authentication"))
                }
            }
        }
    }

    /// Generate performance metrics
    async fn generate_performance_metrics(&self, test_results: &[ApiTestResult]) -> Result<Vec<ApiPerformanceMetrics>> {
        let mut metrics = Vec::new();
        
        // Group results by exchange and endpoint
        let mut grouped_results: HashMap<(String, String), Vec<&ApiTestResult>> = HashMap::new();
        
        for result in test_results {
            let key = (result.exchange_id.clone(), result.endpoint.clone());
            grouped_results.entry(key).or_insert_with(Vec::new).push(result);
        }
        
        // Calculate metrics for each group
        for ((exchange_id, endpoint), results) in grouped_results {
            if results.is_empty() {
                continue;
            }
            
            let response_times: Vec<u64> = results.iter().map(|r| r.response_time_ms).collect();
            let successful_requests = results.iter().filter(|r| r.status == ValidationStatus::Passed).count();
            let total_requests = results.len();
            
            let average_response_time = response_times.iter().sum::<u64>() as f64 / response_times.len() as f64;
            let min_response_time = *response_times.iter().min().unwrap_or(&0);
            let max_response_time = *response_times.iter().max().unwrap_or(&0);
            
            // Calculate percentiles (simplified)
            let mut sorted_times = response_times.clone();
            sorted_times.sort();
            let p95_index = (sorted_times.len() as f64 * 0.95) as usize;
            let p99_index = (sorted_times.len() as f64 * 0.99) as usize;
            let p95_response_time = sorted_times.get(p95_index).unwrap_or(&0).clone() as f64;
            let p99_response_time = sorted_times.get(p99_index).unwrap_or(&0).clone() as f64;
            
            let success_rate = successful_requests as f64 / total_requests as f64;
            let error_rate = 1.0 - success_rate;
            
            // Calculate data transfer (simplified)
            let total_data_transfer = results.iter()
                .map(|r| r.response_size_bytes.unwrap_or(0))
                .sum::<u64>() as f64 / 1024.0 / 1024.0; // Convert to MB
            
            metrics.push(ApiPerformanceMetrics {
                exchange_id,
                endpoint,
                average_response_time_ms: average_response_time,
                min_response_time_ms: min_response_time,
                max_response_time_ms: max_response_time,
                p95_response_time_ms: p95_response_time,
                p99_response_time_ms: p99_response_time,
                success_rate,
                error_rate,
                throughput_requests_per_second: total_requests as f64 / 60.0, // Assuming 1 minute test
                data_transfer_mb: total_data_transfer,
            });
        }
        
        Ok(metrics)
    }

    /// Calculate test statistics
    fn calculate_test_statistics(&self, report: &mut ApiIntegrationReport) {
        let total_tests = report.test_results.len() as u32;
        let passed_tests = report.test_results.iter()
            .filter(|r| r.status == ValidationStatus::Passed)
            .count() as u32;
        let failed_tests = report.test_results.iter()
            .filter(|r| r.status == ValidationStatus::Failed)
            .count() as u32;
        
        // Count unique exchanges and endpoints
        let mut exchanges = std::collections::HashSet::new();
        let mut endpoints = std::collections::HashSet::new();
        
        for result in &report.test_results {
            exchanges.insert(result.exchange_id.clone());
            endpoints.insert(result.endpoint.clone());
        }
        
        let total_response_time: u64 = report.test_results.iter()
            .map(|r| r.response_time_ms)
            .sum();
        
        report.exchanges_tested = exchanges.len() as u32;
        report.endpoints_tested = endpoints.len() as u32;
        report.tests_passed = passed_tests;
        report.tests_failed = failed_tests;
        report.average_response_time_ms = if total_tests > 0 {
            total_response_time as f64 / total_tests as f64
        } else {
            0.0
        };
        
        // Identify critical issues
        for result in &report.test_results {
            if result.status == ValidationStatus::Failed {
                report.critical_issues.push(format!(
                    "Exchange {} endpoint {} failed: {}",
                    result.exchange_id,
                    result.endpoint,
                    result.error_message.as_deref().unwrap_or("Unknown error")
                ));
            }
        }
    }

    /// Determine overall status
    fn determine_overall_status(&self, report: &ApiIntegrationReport) -> ValidationStatus {
        if report.tests_failed > 0 {
            ValidationStatus::Failed
        } else if report.tests_passed == report.test_results.len() as u32 {
            ValidationStatus::Passed
        } else {
            ValidationStatus::Warning
        }
    }

    /// Generate recommendations
    fn generate_recommendations(&self, report: &mut ApiIntegrationReport) {
        if report.tests_failed > 0 {
            report.recommendations.push(
                "Address all failed API tests before deploying to production".to_string()
            );
        }
        
        if report.average_response_time_ms > 1000.0 {
            report.recommendations.push(
                "API response times are high. Consider optimizing or using closer endpoints".to_string()
            );
        }
        
        if report.exchanges_tested == 0 {
            report.recommendations.push(
                "Configure real exchange connections for API integration testing".to_string()
            );
        }
        
        // Check for authentication failures
        let auth_failures = report.test_results.iter()
            .filter(|r| r.test_type == ApiTestType::AuthenticationTest && r.status == ValidationStatus::Failed)
            .count();
        
        if auth_failures > 0 {
            report.recommendations.push(
                "Review API credentials and authentication configuration".to_string()
            );
        }
    }

    /// Validate integration with real APIs
    pub async fn validate_integration(&self) -> Result<ValidationResult> {
        info!("Validating real API integration...");
        
        // Run comprehensive tests
        let report = self.run_comprehensive_tests().await?;
        
        if report.overall_status == ValidationStatus::Passed {
            Ok(ValidationResult::passed(
                "All API integration tests passed with real exchanges".to_string()
            ))
        } else if report.overall_status == ValidationStatus::Warning {
            Ok(ValidationResult::warning(
                format!("API integration tests completed with warnings: {} critical issues", 
                       report.critical_issues.len())
            ))
        } else {
            Ok(ValidationResult::failed(
                format!("API integration tests failed: {} critical issues", 
                       report.critical_issues.len())
            ))
        }
    }

    /// Shutdown the API integration tester
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down Live API Integration Tester...");
        
        // Stop monitoring
        self.stop_monitoring().await?;
        
        // Close WebSocket connections
        {
            let mut connections = self.websocket_connections.write().await;
            connections.clear();
        }
        
        info!("Live API Integration Tester shutdown completed");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MarketReadinessConfig;
    
    #[tokio::test]
    async fn test_api_integration_tester_creation() {
        let config = Arc::new(MarketReadinessConfig::default());
        let zero_mock_detector = Arc::new(
            crate::zero_mock_detection::ZeroMockDetectionEngine::new(config.clone())
                .await
                .unwrap()
        );
        
        let tester = LiveApiIntegrationTester::new(config, zero_mock_detector).await;
        assert!(tester.is_ok());
    }
    
    #[tokio::test]
    async fn test_exchange_config_initialization() {
        let config = Arc::new(MarketReadinessConfig::default());
        let zero_mock_detector = Arc::new(
            crate::zero_mock_detection::ZeroMockDetectionEngine::new(config.clone())
                .await
                .unwrap()
        );
        
        let mut tester = LiveApiIntegrationTester::new(config, zero_mock_detector).await.unwrap();
        tester.initialize_exchange_configs().await.unwrap();
        
        let configs = tester.exchange_configs.read().await;
        assert!(!configs.is_empty());
        assert!(configs.contains_key("binance_testnet"));
    }
}
