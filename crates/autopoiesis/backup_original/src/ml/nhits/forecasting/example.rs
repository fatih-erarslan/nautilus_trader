//! Example usage of the NHITS Forecasting Pipeline
//! 
//! This example demonstrates how to use the complete forecasting system
//! with real-time data, online learning, and adaptive retraining.

use std::sync::Arc;
use std::collections::HashMap;
use tokio;
use ndarray::{Array1, Array2};
use chrono::{DateTime, Utc};

use crate::ml::nhits::forecasting::{
    ForecastingPipeline, ForecastingConfig, ForecastingEvent,
    RetrainingConfig, PreprocessingConfig, FeatureEngineeringConfig,
};
use crate::ml::nhits::NHITSConfig;
use crate::consciousness::ConsciousnessField;
use crate::core::autopoiesis::AutopoieticSystem;

/// Example: Financial time series forecasting
#[tokio::main]
async fn financial_forecasting_example() -> anyhow::Result<()> {
    // Initialize consciousness and autopoietic systems
    let consciousness = Arc::new(ConsciousnessField::new());
    let autopoietic = Arc::new(AutopoieticSystem::new());
    
    // Configure forecasting pipeline for financial data
    let config = ForecastingConfig {
        horizons: vec![1, 5, 20], // 1-day, 1-week, 1-month
        ensemble_size: 7,
        confidence_levels: vec![0.90, 0.95, 0.99],
        online_window_size: 2000,
        update_frequency: 100,
        anomaly_threshold: 3.5,
        retraining_config: RetrainingConfig {
            performance_threshold: 0.15,
            max_time_between_retraining: chrono::Duration::days(3),
            min_samples: 500,
            detect_concept_drift: true,
        },
        preprocessing_config: PreprocessingConfig {
            normalize: true,
            detrending: crate::ml::nhits::forecasting::DetrendingMethod::Linear,
            seasonal_decomposition: true,
            feature_engineering: FeatureEngineeringConfig {
                lag_features: vec![1, 5, 20, 60],
                rolling_windows: vec![5, 20, 60],
                fourier_features: Some(20),
                calendar_features: true,
            },
            outlier_handling: crate::ml::nhits::forecasting::OutlierHandling::Clip,
        },
        persistence_config: Default::default(),
    };
    
    // Create forecasting pipeline
    let pipeline = ForecastingPipeline::new(config, consciousness, autopoietic).await?;
    
    // Subscribe to events
    let mut event_rx = pipeline.subscribe();
    
    // Spawn event handler
    tokio::spawn(async move {
        while let Ok(event) = event_rx.recv().await {
            match event {
                ForecastingEvent::ForecastGenerated(result) => {
                    println!("📊 Forecast generated at {}", result.timestamp);
                    println!("   Confidence: {:.2}%", result.confidence * 100.0);
                    for (horizon, forecast) in &result.forecasts {
                        println!("   {}-step ahead: {:.4}", horizon, forecast.mean().unwrap());
                    }
                }
                ForecastingEvent::ModelRetrained(version) => {
                    println!("🔄 Model retrained: {}", version);
                }
                ForecastingEvent::AnomalyDetected(score) => {
                    println!("⚠️  Anomaly detected with score: {:.2}", score);
                }
                ForecastingEvent::PerformanceDegraded(degradation) => {
                    println!("📉 Performance degraded by {:.2}%", degradation * 100.0);
                }
                ForecastingEvent::ConceptDriftDetected => {
                    println!("🌊 Concept drift detected!");
                }
                ForecastingEvent::ModelSaved(version) => {
                    println!("💾 Model saved: {}", version);
                }
            }
        }
    });
    
    // Simulate streaming financial data
    println!("Starting financial time series forecasting...\n");
    
    // Historical data for initial training
    let historical_data = generate_financial_data(1000);
    
    // Generate initial forecast
    let forecast = pipeline.forecast(&historical_data, None).await?;
    print_forecast_summary(&forecast);
    
    // Simulate real-time streaming
    for day in 0..100 {
        // Generate new data point
        let new_data = generate_financial_data(100);
        
        // Generate forecast
        let forecast = pipeline.forecast(&new_data, None).await?;
        
        // Simulate actual outcome (with noise)
        let actual = new_data.slice(s![1..]).to_owned() + Array1::random(99, rand_distr::Normal::new(0.0, 0.1)?);
        
        // Update model with actual data
        pipeline.update(&new_data.slice(s![..99]).to_owned(), &actual).await?;
        
        // Calculate and display metrics periodically
        if day % 10 == 0 {
            let mut actuals = HashMap::new();
            actuals.insert(1, actual.slice(s![..1]).to_owned());
            if actual.len() >= 5 {
                actuals.insert(5, actual.slice(s![..5]).to_owned());
            }
            
            let metrics = pipeline.calculate_metrics(&forecast, &actuals).await?;
            print_performance_metrics(&metrics);
        }
        
        // Small delay to simulate real-time
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }
    
    // Save final model
    let version = pipeline.save_models().await?;
    println!("\n✅ Final model saved as: {}", version);
    
    Ok(())
}

/// Example: IoT sensor anomaly detection
async fn iot_anomaly_detection_example() -> anyhow::Result<()> {
    let consciousness = Arc::new(ConsciousnessField::new());
    let autopoietic = Arc::new(AutopoieticSystem::new());
    
    // Configure for anomaly detection focus
    let mut config = ForecastingConfig::default();
    config.horizons = vec![1, 3, 10]; // Short-term focus
    config.anomaly_threshold = 2.5; // More sensitive
    config.preprocessing_config.feature_engineering.rolling_windows = vec![10, 30, 60];
    
    let pipeline = ForecastingPipeline::new(config, consciousness, autopoietic).await?;
    
    println!("Starting IoT sensor anomaly detection...\n");
    
    // Simulate sensor data stream
    for minute in 0..1440 { // 24 hours
        let sensor_data = generate_sensor_data(60); // 60 seconds of data
        
        // Check for anomalies
        let forecast = pipeline.forecast(&sensor_data, None).await?;
        
        if let Some(anomaly_scores) = &forecast.anomaly_scores {
            let max_score = anomaly_scores.iter().cloned().fold(0.0, f64::max);
            if max_score > 2.5 {
                println!("🚨 Anomaly detected at minute {}: score = {:.2}", minute, max_score);
                
                // Analyze anomaly pattern
                let anomaly_indices: Vec<_> = anomaly_scores.iter()
                    .enumerate()
                    .filter(|(_, &score)| score > 2.5)
                    .map(|(i, _)| i)
                    .collect();
                    
                println!("   Anomalous points: {:?}", anomaly_indices);
            }
        }
        
        // Update model with actual sensor readings
        let actual = sensor_data.slice(s![1..]).to_owned();
        pipeline.update(&sensor_data.slice(s![..-1]).to_owned(), &actual).await?;
        
        // Periodic status update
        if minute % 60 == 0 {
            println!("⏰ Hour {}: Model confidence = {:.2}%", 
                minute / 60, forecast.confidence * 100.0);
        }
    }
    
    Ok(())
}

/// Example: Energy demand forecasting with external features
async fn energy_demand_forecasting_example() -> anyhow::Result<()> {
    let consciousness = Arc::new(ConsciousnessField::new());
    let autopoietic = Arc::new(AutopoieticSystem::new());
    
    // Configure for energy forecasting
    let mut config = ForecastingConfig::default();
    config.horizons = vec![24, 168]; // 1-day, 1-week ahead
    config.preprocessing_config.feature_engineering.calendar_features = true;
    
    let pipeline = ForecastingPipeline::new(config, consciousness, autopoietic).await?;
    
    println!("Starting energy demand forecasting with weather data...\n");
    
    // Historical energy demand
    let demand_history = generate_energy_demand(7 * 24); // 1 week of hourly data
    
    // External features: temperature, humidity, day_of_week
    let external_features = generate_weather_features(7 * 24);
    
    // Generate forecast with external features
    let forecast = pipeline.forecast(&demand_history, Some(&external_features)).await?;
    
    println!("📊 Energy Demand Forecast:");
    println!("   24-hour ahead: {:.2} MW", forecast.forecasts[&24].mean().unwrap());
    println!("   Weekly average: {:.2} MW", forecast.forecasts[&168].mean().unwrap());
    
    // Display prediction intervals
    for (horizon, level) in [(24, 0.95), (168, 0.95)] {
        if let Some((lower, upper)) = forecast.intervals.get(&(horizon, level)) {
            println!("   {}-hour {}% interval: [{:.2}, {:.2}] MW",
                horizon, level * 100.0,
                lower.mean().unwrap(),
                upper.mean().unwrap()
            );
        }
    }
    
    Ok(())
}

// Helper functions for data generation
fn generate_financial_data(n: usize) -> Array1<f64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut data = Array1::zeros(n);
    
    // Random walk with drift and volatility clustering
    data[0] = 100.0;
    let mut volatility = 0.01;
    
    for i in 1..n {
        volatility = 0.9 * volatility + 0.1 * rng.gen::<f64>() * 0.02;
        let return_pct = rng.gen_range(-volatility..volatility);
        data[i] = data[i-1] * (1.0 + return_pct);
    }
    
    data
}

fn generate_sensor_data(n: usize) -> Array1<f64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    // Base signal with occasional spikes
    Array1::from_shape_fn(n, |i| {
        let base = 20.0 + 5.0 * (i as f64 * 0.1).sin();
        let noise = rng.gen_range(-0.5..0.5);
        let spike = if rng.gen::<f64>() < 0.01 { rng.gen_range(10.0..20.0) } else { 0.0 };
        base + noise + spike
    })
}

fn generate_energy_demand(n: usize) -> Array1<f64> {
    // Hourly energy demand with daily and weekly patterns
    Array1::from_shape_fn(n, |i| {
        let hour_of_day = i % 24;
        let day_of_week = (i / 24) % 7;
        
        // Base load
        let base = 1000.0;
        
        // Daily pattern (peak at 6pm)
        let daily = 200.0 * (((hour_of_day as f64 - 18.0) / 12.0 * std::f64::consts::PI).cos() + 1.0);
        
        // Weekly pattern (lower on weekends)
        let weekly = if day_of_week >= 5 { -100.0 } else { 50.0 };
        
        // Random variation
        let noise = rand::random::<f64>() * 50.0 - 25.0;
        
        base + daily + weekly + noise
    })
}

fn generate_weather_features(n: usize) -> Array2<f64> {
    let mut features = Array2::zeros((n, 3));
    
    for i in 0..n {
        let hour = i % 24;
        
        // Temperature (Celsius)
        features[[i, 0]] = 20.0 + 10.0 * ((hour as f64 - 14.0) / 12.0 * std::f64::consts::PI).sin() 
            + rand::random::<f64>() * 2.0;
        
        // Humidity (%)
        features[[i, 1]] = 60.0 + 20.0 * ((hour as f64) / 24.0 * std::f64::consts::PI * 2.0).cos()
            + rand::random::<f64>() * 5.0;
        
        // Day of week (encoded)
        features[[i, 2]] = ((i / 24) % 7) as f64 / 7.0;
    }
    
    features
}

fn print_forecast_summary(forecast: &crate::ml::nhits::forecasting::ForecastResult) {
    println!("\n📈 Forecast Summary:");
    println!("   Timestamp: {}", forecast.timestamp);
    println!("   Model version: {}", forecast.model_version);
    println!("   Overall confidence: {:.2}%", forecast.confidence * 100.0);
    
    for (horizon, values) in &forecast.forecasts {
        println!("\n   {}-step ahead forecast:", horizon);
        println!("     Mean: {:.4}", values.mean().unwrap());
        println!("     Std: {:.4}", values.std(0.0));
        
        if let Some(uncertainty) = forecast.uncertainty.get(horizon) {
            println!("     Avg uncertainty: {:.4}", uncertainty.mean().unwrap());
        }
    }
}

fn print_performance_metrics(metrics: &crate::ml::nhits::forecasting::PerformanceMetrics) {
    println!("\n📊 Performance Metrics:");
    println!("   MAE: {:.4}", metrics.mae);
    println!("   RMSE: {:.4}", metrics.rmse);
    println!("   MAPE: {:.2}%", metrics.mape * 100.0);
    println!("   Bias: {:.4}", metrics.bias);
    
    for (level, coverage) in &metrics.interval_coverage {
        println!("   {}% interval coverage: {:.2}%", level * 100.0, coverage * 100.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_financial_forecasting() {
        let result = financial_forecasting_example().await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_anomaly_detection() {
        let result = iot_anomaly_detection_example().await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_energy_forecasting() {
        let result = energy_demand_forecasting_example().await;
        assert!(result.is_ok());
    }
}