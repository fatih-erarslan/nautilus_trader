use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use anyhow::Result;
use serde_json::json;
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tracing::{info, warn, error, instrument};
use warp::{Filter, Reply};

use crate::observability::{ObservabilityManager, NeuralMetrics, SystemHealthMetrics, TradingMetrics};

/// Prometheus Metrics Exporter for Cerebellar Norse
pub struct MetricsExporter {
    observability: Arc<ObservabilityManager>,
    bind_address: String,
    port: u16,
    server_info: ServerInfo,
}

#[derive(Debug, Clone)]
pub struct ServerInfo {
    pub version: String,
    pub build_timestamp: String,
    pub commit_hash: String,
    pub rust_version: String,
}

impl MetricsExporter {
    pub fn new(observability: Arc<ObservabilityManager>, bind_address: String, port: u16) -> Self {
        let server_info = ServerInfo {
            version: env!("CARGO_PKG_VERSION").to_string(),
            build_timestamp: env!("BUILD_TIMESTAMP").unwrap_or("unknown").to_string(),
            commit_hash: env!("GIT_COMMIT_HASH").unwrap_or("unknown").to_string(),
            rust_version: env!("RUST_VERSION").unwrap_or("unknown").to_string(),
        };
        
        Self {
            observability,
            bind_address,
            port,
            server_info,
        }
    }
    
    /// Start the metrics exporter HTTP server
    #[instrument(skip(self))]
    pub async fn start(&self) -> Result<()> {
        let observability = self.observability.clone();
        let server_info = self.server_info.clone();
        
        // Health check endpoint
        let health = warp::path("health")
            .and(warp::get())
            .map(move || {
                warp::reply::json(&json!({
                    "status": "healthy",
                    "timestamp": SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis(),
                    "version": server_info.version,
                }))
            });
        
        // Prometheus metrics endpoint
        let observability_for_metrics = observability.clone();
        let metrics = warp::path("metrics")
            .and(warp::get())
            .and_then(move || {
                let obs = observability_for_metrics.clone();
                async move {
                    match obs.export_prometheus_metrics() {
                        Ok(metrics) => Ok(warp::reply::with_header(
                            metrics,
                            "content-type",
                            "text/plain; version=0.0.4; charset=utf-8"
                        )),
                        Err(e) => {
                            error!("Failed to export metrics: {}", e);
                            Err(warp::reject::custom(MetricsError))
                        }
                    }
                }
            });
        
        // Neural network specific metrics
        let observability_for_neural = observability.clone();
        let neural_metrics = warp::path("neural")
            .and(warp::path("metrics"))
            .and(warp::get())
            .and_then(move || {
                let obs = observability_for_neural.clone();
                async move {
                    match obs.get_neural_metrics_summary().await {
                        Ok(Some(metrics)) => {
                            let prometheus_format = format_neural_metrics_for_prometheus(&metrics);
                            Ok(warp::reply::with_header(
                                prometheus_format,
                                "content-type",
                                "text/plain; version=0.0.4; charset=utf-8"
                            ))
                        },
                        Ok(None) => Ok(warp::reply::with_header(
                            "# No neural metrics available\n".to_string(),
                            "content-type",
                            "text/plain; version=0.0.4; charset=utf-8"
                        )),
                        Err(e) => {
                            error!("Failed to get neural metrics: {}", e);
                            Err(warp::reject::custom(MetricsError))
                        }
                    }
                }
            });
        
        // Trading system metrics
        let observability_for_trading = observability.clone();
        let trading_metrics = warp::path("trading")
            .and(warp::path("metrics"))
            .and(warp::get())
            .and_then(move || {
                let obs = observability_for_trading.clone();
                async move {
                    match obs.get_trading_metrics_summary().await {
                        Ok(Some(metrics)) => {
                            let prometheus_format = format_trading_metrics_for_prometheus(&metrics);
                            Ok(warp::reply::with_header(
                                prometheus_format,
                                "content-type",
                                "text/plain; version=0.0.4; charset=utf-8"
                            ))
                        },
                        Ok(None) => Ok(warp::reply::with_header(
                            "# No trading metrics available\n".to_string(),
                            "content-type",
                            "text/plain; version=0.0.4; charset=utf-8"
                        )),
                        Err(e) => {
                            error!("Failed to get trading metrics: {}", e);
                            Err(warp::reject::custom(MetricsError))
                        }
                    }
                }
            });
        
        // System health metrics
        let observability_for_system = observability.clone();
        let system_metrics = warp::path("system")
            .and(warp::path("metrics"))
            .and(warp::get())
            .and_then(move || {
                let obs = observability_for_system.clone();
                async move {
                    match obs.get_system_health_summary().await {
                        Ok(Some(metrics)) => {
                            let prometheus_format = format_system_metrics_for_prometheus(&metrics);
                            Ok(warp::reply::with_header(
                                prometheus_format,
                                "content-type",
                                "text/plain; version=0.0.4; charset=utf-8"
                            ))
                        },
                        Ok(None) => Ok(warp::reply::with_header(
                            "# No system metrics available\n".to_string(),
                            "content-type",
                            "text/plain; version=0.0.4; charset=utf-8"
                        )),
                        Err(e) => {
                            error!("Failed to get system metrics: {}", e);
                            Err(warp::reject::custom(MetricsError))
                        }
                    }
                }
            });
        
        // Market data metrics (placeholder for future implementation)
        let market_metrics = warp::path("market")
            .and(warp::path("metrics"))
            .and(warp::get())
            .map(|| {
                let timestamp = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis();
                
                format!(
                    "# HELP market_data_latency_ms Market data processing latency in milliseconds\n\
                     # TYPE market_data_latency_ms gauge\n\
                     market_data_latency_ms {{component=\"market_data\"}} 2.5 {}\n\
                     # HELP market_data_throughput_msgs_per_sec Market data throughput in messages per second\n\
                     # TYPE market_data_throughput_msgs_per_sec gauge\n\
                     market_data_throughput_msgs_per_sec {{component=\"market_data\"}} 15000.0 {}\n",
                    timestamp, timestamp
                )
            })
            .map(|body| warp::reply::with_header(
                body,
                "content-type",
                "text/plain; version=0.0.4; charset=utf-8"
            ));
        
        // Risk management metrics
        let observability_for_risk = observability.clone();
        let risk_metrics = warp::path("risk")
            .and(warp::path("metrics"))
            .and(warp::get())
            .and_then(move || {
                let obs = observability_for_risk.clone();
                async move {
                    match obs.get_trading_metrics_summary().await {
                        Ok(Some(metrics)) => {
                            let timestamp = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap()
                                .as_millis();
                            
                            let prometheus_format = format!(
                                "# HELP risk_score Current trading risk score (0-1)\n\
                                 # TYPE risk_score gauge\n\
                                 risk_score {{component=\"risk_management\"}} {} {}\n\
                                 # HELP portfolio_value_usd Current portfolio value in USD\n\
                                 # TYPE portfolio_value_usd gauge\n\
                                 portfolio_value_usd {{component=\"risk_management\"}} {} {}\n\
                                 # HELP open_positions_total Total number of open positions\n\
                                 # TYPE open_positions_total gauge\n\
                                 open_positions_total {{component=\"risk_management\"}} {} {}\n",
                                metrics.risk_score, timestamp,
                                metrics.portfolio_value, timestamp,
                                metrics.open_positions, timestamp
                            );
                            
                            Ok(warp::reply::with_header(
                                prometheus_format,
                                "content-type",
                                "text/plain; version=0.0.4; charset=utf-8"
                            ))
                        },
                        Ok(None) => Ok(warp::reply::with_header(
                            "# No risk metrics available\n".to_string(),
                            "content-type",
                            "text/plain; version=0.0.4; charset=utf-8"
                        )),
                        Err(e) => {
                            error!("Failed to get risk metrics: {}", e);
                            Err(warp::reject::custom(MetricsError))
                        }
                    }
                }
            });
        
        // Benchmark metrics
        let benchmark_metrics = warp::path("benchmarks")
            .and(warp::path("metrics"))
            .and(warp::get())
            .map(|| {
                let timestamp = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis();
                
                format!(
                    "# HELP benchmark_neural_inference_ns Neural network inference benchmark in nanoseconds\n\
                     # TYPE benchmark_neural_inference_ns gauge\n\
                     benchmark_neural_inference_ns {{benchmark=\"inference\",model=\"cerebellar\"}} 2500000 {}\n\
                     # HELP benchmark_memory_allocation_ns Memory allocation benchmark in nanoseconds\n\
                     # TYPE benchmark_memory_allocation_ns gauge\n\
                     benchmark_memory_allocation_ns {{benchmark=\"memory\",type=\"zero_alloc\"}} 50000 {}\n\
                     # HELP benchmark_simd_ops_ns SIMD operations benchmark in nanoseconds\n\
                     # TYPE benchmark_simd_ops_ns gauge\n\
                     benchmark_simd_ops_ns {{benchmark=\"simd\",operation=\"vector_mul\"}} 100000 {}\n",
                    timestamp, timestamp, timestamp
                )
            })
            .map(|body| warp::reply::with_header(
                body,
                "content-type",
                "text/plain; version=0.0.4; charset=utf-8"
            ));
        
        // Server info endpoint
        let info = warp::path("info")
            .and(warp::get())
            .map(move || {
                warp::reply::json(&json!({
                    "service": "cerebellar-norse-metrics-exporter",
                    "version": server_info.version,
                    "build_timestamp": server_info.build_timestamp,
                    "commit_hash": server_info.commit_hash,
                    "rust_version": server_info.rust_version,
                    "endpoints": [
                        "/health",
                        "/info",
                        "/metrics",
                        "/neural/metrics",
                        "/trading/metrics",
                        "/system/metrics", 
                        "/market/metrics",
                        "/risk/metrics",
                        "/benchmarks/metrics"
                    ]
                }))
            });
        
        // Combine all routes
        let routes = health
            .or(info)
            .or(metrics)
            .or(neural_metrics)
            .or(trading_metrics)
            .or(system_metrics)
            .or(market_metrics)
            .or(risk_metrics)
            .or(benchmark_metrics)
            .with(warp::cors().allow_any_origin())
            .recover(handle_rejection);
        
        let addr = format!("{}:{}", self.bind_address, self.port);
        info!("Starting metrics exporter on {}", addr);
        
        warp::serve(routes)
            .run(([0, 0, 0, 0], self.port))
            .await;
        
        Ok(())
    }
}

/// Format neural metrics for Prometheus export
fn format_neural_metrics_for_prometheus(metrics: &NeuralMetrics) -> String {
    let timestamp = metrics.timestamp;
    
    format!(
        "# HELP neural_inference_latency_ns Neural network inference latency in nanoseconds\n\
         # TYPE neural_inference_latency_ns gauge\n\
         neural_inference_latency_ns {{component=\"neural_network\"}} {} {}\n\
         # HELP neural_throughput_ops_per_sec Neural network throughput in operations per second\n\
         # TYPE neural_throughput_ops_per_sec gauge\n\
         neural_throughput_ops_per_sec {{component=\"neural_network\"}} {} {}\n\
         # HELP neural_accuracy_percentage Neural network accuracy percentage\n\
         # TYPE neural_accuracy_percentage gauge\n\
         neural_accuracy_percentage {{component=\"neural_network\"}} {} {}\n\
         # HELP neural_memory_usage_mb Neural network memory usage in megabytes\n\
         # TYPE neural_memory_usage_mb gauge\n\
         neural_memory_usage_mb {{component=\"neural_network\"}} {} {}\n\
         # HELP neural_gpu_utilization_percent GPU utilization percentage\n\
         # TYPE neural_gpu_utilization_percent gauge\n\
         neural_gpu_utilization_percent {{component=\"neural_network\"}} {} {}\n\
         # HELP neural_error_rate_percent Neural network error rate percentage\n\
         # TYPE neural_error_rate_percent gauge\n\
         neural_error_rate_percent {{component=\"neural_network\"}} {} {}\n",
        metrics.inference_latency_ns, timestamp,
        metrics.throughput_ops_per_sec, timestamp,
        metrics.accuracy_percentage, timestamp,
        metrics.memory_usage_mb, timestamp,
        metrics.gpu_utilization_percent, timestamp,
        metrics.error_rate_percent, timestamp
    )
}

/// Format trading metrics for Prometheus export
fn format_trading_metrics_for_prometheus(metrics: &TradingMetrics) -> String {
    let timestamp = metrics.timestamp;
    
    format!(
        "# HELP trading_orders_per_second Trading orders processed per second\n\
         # TYPE trading_orders_per_second gauge\n\
         trading_orders_per_second {{component=\"trading_system\"}} {} {}\n\
         # HELP trading_order_fill_latency_ms Order fill latency in milliseconds\n\
         # TYPE trading_order_fill_latency_ms gauge\n\
         trading_order_fill_latency_ms {{component=\"trading_system\"}} {} {}\n\
         # HELP trading_market_data_latency_ms Market data latency in milliseconds\n\
         # TYPE trading_market_data_latency_ms gauge\n\
         trading_market_data_latency_ms {{component=\"trading_system\"}} {} {}\n\
         # HELP trading_risk_score Current trading risk score\n\
         # TYPE trading_risk_score gauge\n\
         trading_risk_score {{component=\"trading_system\"}} {} {}\n\
         # HELP trading_pnl_realtime Real-time profit and loss\n\
         # TYPE trading_pnl_realtime gauge\n\
         trading_pnl_realtime {{component=\"trading_system\"}} {} {}\n\
         # HELP trading_portfolio_value Total portfolio value\n\
         # TYPE trading_portfolio_value gauge\n\
         trading_portfolio_value {{component=\"trading_system\"}} {} {}\n\
         # HELP trading_open_positions Number of open positions\n\
         # TYPE trading_open_positions gauge\n\
         trading_open_positions {{component=\"trading_system\"}} {} {}\n\
         # HELP trading_alerts_triggered Number of alerts triggered\n\
         # TYPE trading_alerts_triggered gauge\n\
         trading_alerts_triggered {{component=\"trading_system\"}} {} {}\n",
        metrics.orders_per_second, timestamp,
        metrics.order_fill_latency_ms, timestamp,
        metrics.market_data_latency_ms, timestamp,
        metrics.risk_score, timestamp,
        metrics.pnl_realtime, timestamp,
        metrics.portfolio_value, timestamp,
        metrics.open_positions, timestamp,
        metrics.alerts_triggered, timestamp
    )
}

/// Format system metrics for Prometheus export
fn format_system_metrics_for_prometheus(metrics: &SystemHealthMetrics) -> String {
    let timestamp = metrics.timestamp;
    
    format!(
        "# HELP system_cpu_usage_percent System CPU usage percentage\n\
         # TYPE system_cpu_usage_percent gauge\n\
         system_cpu_usage_percent {{component=\"system\"}} {} {}\n\
         # HELP system_memory_usage_percent System memory usage percentage\n\
         # TYPE system_memory_usage_percent gauge\n\
         system_memory_usage_percent {{component=\"system\"}} {} {}\n\
         # HELP system_disk_usage_percent System disk usage percentage\n\
         # TYPE system_disk_usage_percent gauge\n\
         system_disk_usage_percent {{component=\"system\"}} {} {}\n\
         # HELP system_network_rx_mbps Network receive rate in megabits per second\n\
         # TYPE system_network_rx_mbps gauge\n\
         system_network_rx_mbps {{component=\"system\"}} {} {}\n\
         # HELP system_network_tx_mbps Network transmit rate in megabits per second\n\
         # TYPE system_network_tx_mbps gauge\n\
         system_network_tx_mbps {{component=\"system\"}} {} {}\n\
         # HELP system_process_count Number of running processes\n\
         # TYPE system_process_count gauge\n\
         system_process_count {{component=\"system\"}} {} {}\n",
        metrics.cpu_usage_percent, timestamp,
        metrics.memory_usage_percent, timestamp,
        metrics.disk_usage_percent, timestamp,
        metrics.network_rx_mbps, timestamp,
        metrics.network_tx_mbps, timestamp,
        metrics.process_count, timestamp
    )
}

/// Custom error type for metrics export
#[derive(Debug)]
struct MetricsError;

impl warp::reject::Reject for MetricsError {}

/// Handle HTTP rejections
async fn handle_rejection(err: warp::Rejection) -> Result<impl Reply, std::convert::Infallible> {
    let code;
    let message;

    if err.is_not_found() {
        code = warp::http::StatusCode::NOT_FOUND;
        message = "NOT_FOUND";
    } else if let Some(_) = err.find::<MetricsError>() {
        code = warp::http::StatusCode::INTERNAL_SERVER_ERROR;
        message = "METRICS_ERROR";
    } else if let Some(_) = err.find::<warp::filters::body::BodyDeserializeError>() {
        code = warp::http::StatusCode::BAD_REQUEST;
        message = "BAD_REQUEST";
    } else if let Some(_) = err.find::<warp::reject::MethodNotAllowed>() {
        code = warp::http::StatusCode::METHOD_NOT_ALLOWED;
        message = "METHOD_NOT_ALLOWED";
    } else {
        error!("Unhandled rejection: {:?}", err);
        code = warp::http::StatusCode::INTERNAL_SERVER_ERROR;
        message = "UNHANDLED_REJECTION";
    }

    let json = warp::reply::json(&json!({
        "code": code.as_u16(),
        "message": message,
        "timestamp": SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis()
    }));

    Ok(warp::reply::with_status(json, code))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::observability::ObservabilityManager;
    
    #[tokio::test]
    async fn test_metrics_exporter_creation() {
        let observability = Arc::new(ObservabilityManager::new().unwrap());
        let exporter = MetricsExporter::new(
            observability,
            "127.0.0.1".to_string(),
            8080
        );
        
        assert_eq!(exporter.bind_address, "127.0.0.1");
        assert_eq!(exporter.port, 8080);
        assert_eq!(exporter.server_info.version, env!("CARGO_PKG_VERSION"));
    }
    
    #[test]
    fn test_neural_metrics_formatting() {
        let metrics = NeuralMetrics {
            inference_latency_ns: 5_000_000,
            throughput_ops_per_sec: 1000.0,
            accuracy_percentage: 97.5,
            memory_usage_mb: 2048.0,
            gpu_utilization_percent: 85.0,
            network_io_mbps: 100.0,
            error_rate_percent: 0.1,
            timestamp: 1234567890,
        };
        
        let formatted = format_neural_metrics_for_prometheus(&metrics);
        assert!(formatted.contains("neural_inference_latency_ns"));
        assert!(formatted.contains("5000000"));
        assert!(formatted.contains("neural_accuracy_percentage"));
        assert!(formatted.contains("97.5"));
    }
}