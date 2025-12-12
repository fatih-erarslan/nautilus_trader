//! Example demonstrating Isolation Forest usage for anomaly detection

use nn_models::isolation_forest::{IsolationForest, AnomalyScore};
use std::time::Instant;

fn main() {
    println!("Isolation Forest Anomaly Detection Demo");
    println!("======================================\n");
    
    // Generate synthetic data with anomalies
    let mut data = Vec::new();
    
    // Normal data cluster around (0, 0)
    println!("Generating normal data...");
    for i in 0..200 {
        let angle = i as f32 * 0.031415;
        let radius = 1.0 + (i as f32 * 0.005);
        data.push(vec![
            radius * angle.cos(),
            radius * angle.sin(),
            angle.sin() * 0.5,
        ]);
    }
    
    // Add some anomalies
    println!("Adding anomalies...");
    data.push(vec![5.0, 5.0, 2.0]);      // Far from normal cluster
    data.push(vec![-5.0, -5.0, -2.0]);   // Far from normal cluster
    data.push(vec![0.0, 10.0, 0.0]);     // Outlier
    data.push(vec![10.0, 0.0, 5.0]);     // Outlier
    
    println!("Total samples: {}", data.len());
    println!("Expected anomalies: 4\n");
    
    // Create and train Isolation Forest
    println!("Training Isolation Forest...");
    let start = Instant::now();
    
    let mut forest = IsolationForest::builder()
        .n_estimators(200)
        .max_samples(256)
        .contamination(0.02)  // Expect 2% anomalies
        .random_seed(42)
        .build();
    
    forest.fit(&data);
    
    let training_time = start.elapsed();
    println!("Training completed in: {:?}\n", training_time);
    
    // Test anomaly detection
    println!("Detecting anomalies...");
    let predictions = forest.predict(&data);
    let scores = forest.decision_function(&data);
    
    // Count anomalies
    let anomaly_count = predictions.iter().filter(|&&p| p == -1).count();
    println!("Detected {} anomalies\n", anomaly_count);
    
    // Show anomalous samples
    println!("Anomalous samples:");
    for (i, (pred, score)) in predictions.iter().zip(scores.iter()).enumerate() {
        if *pred == -1 {
            println!("  Sample {}: {:?} (score: {:.4})", i, data[i], score);
        }
    }
    
    // Feature importance
    println!("\nFeature importances:");
    let importances = forest.feature_importances();
    for (i, importance) in importances.iter().enumerate() {
        println!("  Feature {}: {:.4}", i, importance);
    }
    
    // Performance test
    println!("\nPerformance test:");
    let test_sample = vec![0.5, 0.5, 0.25];
    let n_iterations = 10000;
    
    let start = Instant::now();
    for _ in 0..n_iterations {
        let _ = forest.anomaly_score(&test_sample);
    }
    let total_time = start.elapsed();
    let avg_time = total_time / n_iterations;
    
    println!("Average inference time: {:?} (target: <50μs)", avg_time);
    println!("Inference rate: {:.0} predictions/second", 1_000_000.0 / avg_time.as_micros() as f64);
    
    // Test on new samples
    println!("\nTesting on new samples:");
    let test_samples = vec![
        vec![0.0, 0.0, 0.0],       // Should be normal
        vec![1.0, 1.0, 0.5],       // Should be normal
        vec![8.0, 8.0, 4.0],       // Should be anomaly
        vec![-8.0, 2.0, -3.0],     // Should be anomaly
    ];
    
    for (i, sample) in test_samples.iter().enumerate() {
        let score = forest.anomaly_score(sample);
        let is_anomaly = score > forest.threshold();
        println!(
            "  Sample {}: {:?} -> score: {:.4}, anomaly: {}",
            i, sample, score, is_anomaly
        );
    }
    
    // Demonstrate serialization capability
    println!("\nConfiguration:");
    println!("  Trees: {}", forest.n_estimators());
    println!("  Contamination: {}", forest.contamination());
    println!("  Threshold: {:.4}", forest.threshold());
}