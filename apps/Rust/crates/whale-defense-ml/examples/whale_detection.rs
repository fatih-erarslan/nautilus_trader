//! Example of using the whale detection transformer model
//! 
//! This example demonstrates how to use the whale defense ML system
//! for real-time whale detection in trading systems.

use whale_defense_ml::{
    TransformerWhaleDetector, TransformerConfig,
    EnsemblePredictor, PredictionResult,
    FeatureExtractor, MarketFeatures,
    WhaleDataset, WhaleEvent, WhaleEventType,
    PerformanceMetrics, InferenceTimer,
};
use candle_core::{Device, Tensor};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the system
    println!("Initializing Whale Defense ML System...");
    
    // Setup device (CPU for this example, use CUDA in production)
    let device = Device::Cpu;
    
    // Create transformer configuration
    let config = TransformerConfig {
        input_dim: 19,      // Number of market features
        hidden_dim: 256,
        num_heads: 8,
        num_layers: 6,
        dropout_rate: 0.1,
        max_seq_length: 60,  // 60 minutes of history
        num_classes: 2,      // Binary: whale or not
        ff_dim_multiplier: 4,
    };
    
    // Create ensemble predictor
    let predictor = EnsemblePredictor::with_config(device.clone(), config, Default::default())?;
    
    // Simulate market data
    println!("\nSimulating market data stream...");
    let mut feature_extractor = FeatureExtractor::new(20);
    
    // Generate synthetic market data
    let mut feature_sequences = Vec::new();
    for i in 0..60 {
        let price = 45000.0 + (i as f32 * 10.0) + (rand::random::<f32>() - 0.5) * 100.0;
        let volume = 1_000_000.0 * (1.0 + rand::random::<f32>());
        let bid = Some(price - 0.5);
        let ask = Some(price + 0.5);
        
        let features = feature_extractor.extract_features(price, volume, bid, ask)?;
        feature_sequences.push(FeatureExtractor::features_to_array(&features));
    }
    
    // Convert to tensor format
    let sequence_tensor = {
        let flat_features: Vec<f32> = feature_sequences
            .iter()
            .flat_map(|arr| arr.iter().copied())
            .collect();
        
        Tensor::from_vec(
            flat_features,
            (1, feature_sequences.len(), 19),  // batch=1, seq_len=60, features=19
            &device,
        )?
    };
    
    // Make predictions
    println!("\nRunning whale detection...");
    let start = Instant::now();
    
    let prediction = predictor.predict(&sequence_tensor)?;
    
    let total_time = start.elapsed();
    println!("\nPrediction Results:");
    println!("==================");
    println!("Whale Probability: {:.2}%", prediction.whale_probability * 100.0);
    println!("Threat Level: {} / 5", prediction.threat_level);
    println!("Confidence: {:.2}%", prediction.confidence * 100.0);
    println!("Inference Time: {}μs", prediction.inference_time_us);
    println!("Total Time: {:?}", total_time);
    
    // Check performance target
    if prediction.inference_time_us <= 500 {
        println!("✅ Performance target met: <500μs");
    } else {
        println!("⚠️  Performance target missed: {}μs > 500μs", prediction.inference_time_us);
    }
    
    // Display model predictions
    println!("\nModel Predictions:");
    for (model, prob) in &prediction.model_predictions {
        println!("  {}: {:.2}%", model, prob * 100.0);
    }
    
    // Display interpretability info
    println!("\nTop Contributing Features:");
    for (feature, importance) in &prediction.interpretability.top_features {
        println!("  {}: {:.3}", feature, importance);
    }
    
    println!("\nAnomaly Score: {:.3}", prediction.interpretability.anomaly_score);
    
    // Demonstrate batch processing
    println!("\n\nBatch Processing Demo:");
    println!("=====================");
    
    // Create multiple sequences
    let batch_size = 10;
    let mut batch_tensor = Vec::new();
    
    for _ in 0..batch_size {
        // Generate random sequence
        let mut seq = Vec::new();
        for _ in 0..60 {
            for _ in 0..19 {
                seq.push(rand::random::<f32>());
            }
        }
        batch_tensor.extend(seq);
    }
    
    let batch = Tensor::from_vec(batch_tensor, (batch_size, 60, 19), &device)?;
    
    // Time batch inference
    let batch_start = Instant::now();
    let batch_results = whale_defense_ml::ensemble::batch_predict(&predictor, &batch)?;
    let batch_time = batch_start.elapsed();
    
    println!("Processed {} samples in {:?}", batch_size, batch_time);
    println!("Average time per sample: {:?}", batch_time / batch_size as u32);
    
    // Summary statistics
    let whale_count = batch_results.iter()
        .filter(|r| r.whale_probability > 0.5)
        .count();
    
    println!("\nBatch Results:");
    println!("  Whales detected: {} / {}", whale_count, batch_size);
    println!("  Average confidence: {:.2}%", 
        batch_results.iter().map(|r| r.confidence).sum::<f32>() / batch_size as f32 * 100.0
    );
    
    // Performance summary
    let avg_inference = batch_results.iter()
        .map(|r| r.inference_time_us)
        .sum::<u64>() / batch_size as u64;
    
    println!("  Average inference time: {}μs", avg_inference);
    
    println!("\n✅ Whale Defense ML System Test Complete!");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_whale_detection_performance() {
        let device = Device::Cpu;
        let predictor = EnsemblePredictor::new(device.clone()).unwrap();
        
        // Create test input
        let input = Tensor::randn(0f32, 1f32, (1, 60, 19), &device).unwrap();
        
        // Measure inference time
        let timer = InferenceTimer::start();
        let _result = predictor.predict(&input).unwrap();
        let elapsed = timer.stop();
        
        // Check performance constraint
        assert!(elapsed < 1000, "Inference took {}μs, expected <1000μs", elapsed);
    }
}