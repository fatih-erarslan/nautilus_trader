//! Data Pipeline Processor Binary
//! 
//! Standalone binary for processing cryptocurrency data through
//! the data pipeline for feature extraction and enhancement.

use std::env;
use std::path::Path;
use std::fs;
use serde_json;
use chrono::{DateTime, Utc};
use anyhow::{Result, anyhow};

#[derive(Debug)]
struct ProcessorArgs {
    input_path: String,
    output_path: String,
    symbol: String,
    features: Vec<String>,
    streaming: bool,
}

impl ProcessorArgs {
    fn parse() -> Result<Self> {
        let args: Vec<String> = env::args().collect();
        
        let mut input_path = String::new();
        let mut output_path = String::new();
        let mut symbol = "BTCUSDT".to_string();
        let mut features = vec!["technical_indicators".to_string()];
        let mut streaming = false;
        
        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--input" => {
                    i += 1;
                    if i >= args.len() { return Err(anyhow!("Missing input path")); }
                    input_path = args[i].clone();
                },
                "--output" => {
                    i += 1;
                    if i >= args.len() { return Err(anyhow!("Missing output path")); }
                    output_path = args[i].clone();
                },
                "--symbol" => {
                    i += 1;
                    if i >= args.len() { return Err(anyhow!("Missing symbol")); }
                    symbol = args[i].clone();
                },
                "--features" => {
                    i += 1;
                    if i >= args.len() { return Err(anyhow!("Missing features")); }
                    features = args[i].split(',').map(|s| s.to_string()).collect();
                },
                "--streaming" => {
                    i += 1;
                    if i >= args.len() { return Err(anyhow!("Missing streaming flag")); }
                    streaming = args[i] == "true";
                },
                _ => {}
            }
            i += 1;
        }
        
        if input_path.is_empty() || output_path.is_empty() {
            return Err(anyhow!("Input and output paths are required"));
        }
        
        Ok(ProcessorArgs {
            input_path,
            output_path,
            symbol,
            features,
            streaming,
        })
    }
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct OHLCVData {
    timestamp: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct ProcessedData {
    timestamp: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    // Technical indicators
    sma_20: Option<f64>,
    sma_50: Option<f64>,
    ema_12: Option<f64>,
    ema_26: Option<f64>,
    rsi: Option<f64>,
    macd: Option<f64>,
    macd_signal: Option<f64>,
    bb_upper: Option<f64>,
    bb_lower: Option<f64>,
    volatility: Option<f64>,
    atr: Option<f64>,
    momentum: Option<f64>,
}

fn process_csv_data(input_path: &str, output_path: &str, symbol: &str, features: &[String]) -> Result<()> {
    println!("🚀 Processing {} data from {}", symbol, input_path);
    
    // Read CSV file
    let mut reader = csv::Reader::from_path(input_path)?;
    let mut raw_data: Vec<OHLCVData> = Vec::new();
    
    // Parse CSV rows
    for result in reader.deserialize() {
        match result {
            Ok(record) => {
                let data: OHLCVData = record;
                raw_data.push(data);
            },
            Err(e) => {
                eprintln!("Warning: Failed to parse row: {}", e);
                continue;
            }
        }
    }
    
    println!("📊 Loaded {} records for processing", raw_data.len());
    
    if raw_data.is_empty() {
        return Err(anyhow!("No valid data found in input file"));
    }
    
    // Process data with technical indicators
    let processed_data = enhance_with_features(&raw_data, features)?;
    
    println!("✅ Enhanced {} records with technical indicators", processed_data.len());
    
    // Write processed data to output CSV
    write_processed_csv(&processed_data, output_path)?;
    
    println!("💾 Processed data saved to {}", output_path);
    
    Ok(())
}

fn enhance_with_features(raw_data: &[OHLCVData], features: &[String]) -> Result<Vec<ProcessedData>> {
    let mut processed = Vec::new();
    
    // Extract close prices for calculations
    let closes: Vec<f64> = raw_data.iter().map(|d| d.close).collect();
    let highs: Vec<f64> = raw_data.iter().map(|d| d.high).collect();
    let lows: Vec<f64> = raw_data.iter().map(|d| d.low).collect();
    
    for (i, data) in raw_data.iter().enumerate() {
        let mut processed_record = ProcessedData {
            timestamp: data.timestamp.clone(),
            open: data.open,
            high: data.high,
            low: data.low,
            close: data.close,
            volume: data.volume,
            sma_20: None,
            sma_50: None,
            ema_12: None,
            ema_26: None,
            rsi: None,
            macd: None,
            macd_signal: None,
            bb_upper: None,
            bb_lower: None,
            volatility: None,
            atr: None,
            momentum: None,
        };
        
        // Calculate technical indicators if requested
        if features.contains(&"technical_indicators".to_string()) {
            // Simple Moving Averages
            if i >= 19 {
                processed_record.sma_20 = Some(closes[i-19..=i].iter().sum::<f64>() / 20.0);
            }
            if i >= 49 {
                processed_record.sma_50 = Some(closes[i-49..=i].iter().sum::<f64>() / 50.0);
            }
            
            // Exponential Moving Averages (simplified)
            if i >= 11 {
                processed_record.ema_12 = Some(calculate_ema(&closes[..=i], 12));
            }
            if i >= 25 {
                processed_record.ema_26 = Some(calculate_ema(&closes[..=i], 26));
            }
            
            // RSI (simplified)
            if i >= 14 {
                processed_record.rsi = Some(calculate_rsi(&closes[i-14..=i]));
            }
            
            // Volatility
            if i >= 19 {
                let returns: Vec<f64> = closes[i-19..=i].windows(2)
                    .map(|w| (w[1] / w[0] - 1.0).powi(2))
                    .collect();
                processed_record.volatility = Some((returns.iter().sum::<f64>() / returns.len() as f64).sqrt());
            }
            
            // ATR (simplified)
            if i >= 13 {
                let tr_values: Vec<f64> = (i-13..=i).map(|idx| {
                    let high = highs[idx];
                    let low = lows[idx];
                    let prev_close = if idx > 0 { closes[idx-1] } else { closes[idx] };
                    
                    let tr1 = high - low;
                    let tr2 = (high - prev_close).abs();
                    let tr3 = (low - prev_close).abs();
                    
                    tr1.max(tr2).max(tr3)
                }).collect();
                
                processed_record.atr = Some(tr_values.iter().sum::<f64>() / tr_values.len() as f64);
            }
            
            // Momentum
            if i >= 5 {
                processed_record.momentum = Some(closes[i] - closes[i-5]);
            }
        }
        
        processed.push(processed_record);
    }
    
    Ok(processed)
}

fn calculate_ema(prices: &[f64], period: usize) -> f64 {
    if prices.len() < period {
        return prices.iter().sum::<f64>() / prices.len() as f64;
    }
    
    let multiplier = 2.0 / (period as f64 + 1.0);
    let mut ema = prices[0..period].iter().sum::<f64>() / period as f64;
    
    for price in prices[period..].iter() {
        ema = (price * multiplier) + (ema * (1.0 - multiplier));
    }
    
    ema
}

fn calculate_rsi(prices: &[f64]) -> f64 {
    if prices.len() < 2 {
        return 50.0;
    }
    
    let mut gains = 0.0;
    let mut losses = 0.0;
    
    for window in prices.windows(2) {
        let change = window[1] - window[0];
        if change > 0.0 {
            gains += change;
        } else {
            losses += change.abs();
        }
    }
    
    let avg_gain = gains / (prices.len() - 1) as f64;
    let avg_loss = losses / (prices.len() - 1) as f64;
    
    if avg_loss == 0.0 {
        return 100.0;
    }
    
    let rs = avg_gain / avg_loss;
    100.0 - (100.0 / (1.0 + rs))
}

fn write_processed_csv(data: &[ProcessedData], output_path: &str) -> Result<()> {
    let mut writer = csv::Writer::from_path(output_path)?;
    
    for record in data {
        writer.serialize(record)?;
    }
    
    writer.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    println!("🦸‍♂️ SUPERMAN DATA PIPELINE PROCESSOR");
    println!("=====================================");
    
    let args = ProcessorArgs::parse()?;
    
    println!("📝 Configuration:");
    println!("   Input:     {}", args.input_path);
    println!("   Output:    {}", args.output_path);
    println!("   Symbol:    {}", args.symbol);
    println!("   Features:  {:?}", args.features);
    println!("   Streaming: {}", args.streaming);
    
    // Verify input file exists
    if !Path::new(&args.input_path).exists() {
        return Err(anyhow!("Input file does not exist: {}", args.input_path));
    }
    
    // Process the data
    process_csv_data(&args.input_path, &args.output_path, &args.symbol, &args.features)?;
    
    println!("🎯 Data pipeline processing completed successfully!");
    
    Ok(())
}