use nn_models::isolation_forest::{IsolationForest, IsolationForestBuilder, IsolationForestConfig};

#[test]
fn test_builder_pattern() {
    let forest = IsolationForest::builder()
        .n_estimators(100)
        .max_samples(128)
        .contamination(0.05)
        .max_depth(10)
        .random_seed(42)
        .n_jobs(4)
        .build();
    
    assert_eq!(forest.n_estimators(), 100);
    assert_eq!(forest.contamination(), 0.05);
}

#[test]
fn test_config_creation() {
    let config = IsolationForestConfig {
        n_estimators: 150,
        max_samples: 200,
        contamination: 0.15,
        max_depth: Some(12),
        random_seed: Some(123),
        n_jobs: Some(8),
    };
    
    let forest = IsolationForest::with_config(config);
    assert_eq!(forest.n_estimators(), 150);
    assert_eq!(forest.contamination(), 0.15);
}

#[test]
fn test_anomaly_detection_simple() {
    // Create clearly separated normal and anomalous data
    let mut data = Vec::new();
    
    // Normal data clustered around origin
    for i in 0..50 {
        data.push(vec![i as f32 * 0.01, i as f32 * 0.01]);
    }
    
    // Clear anomalies far from normal data
    data.push(vec![10.0, 10.0]);
    data.push(vec![-10.0, -10.0]);
    
    let mut forest = IsolationForest::builder()
        .n_estimators(100)
        .contamination(0.04) // ~2 out of 52 samples
        .random_seed(42)
        .build();
    
    forest.fit(&data);
    
    // Check that anomalies are detected
    let scores = forest.decision_function(&data);
    
    // Last two samples should have highest scores
    assert!(scores[50] > scores[0]);
    assert!(scores[51] > scores[0]);
    assert!(scores[50] > scores[25]);
    assert!(scores[51] > scores[25]);
}

#[test]
fn test_empty_data_handling() {
    let mut forest = IsolationForest::new();
    let result = std::panic::catch_unwind(move || {
        forest.fit(&Vec::<Vec<f32>>::new());
    });
    assert!(result.is_err());
}

#[test]
fn test_single_sample_prediction() {
    let data = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.1],
        vec![0.2, 0.2],
        vec![5.0, 5.0], // Anomaly
    ];
    
    let mut forest = IsolationForest::new();
    forest.fit(&data);
    
    let normal_sample = vec![0.15, 0.15];
    let anomaly_sample = vec![4.5, 4.5];
    
    let normal_score = forest.anomaly_score(&normal_sample);
    let anomaly_score = forest.anomaly_score(&anomaly_sample);
    
    assert!(anomaly_score > normal_score);
}

#[test]
fn test_feature_importance_consistency() {
    let data: Vec<Vec<f32>> = (0..100)
        .map(|i| {
            vec![
                i as f32 * 0.1,              // Linear feature
                (i as f32 * 0.1).sin(),       // Non-linear feature
                1.0,                          // Constant feature
                rand::random::<f32>() * 10.0, // Random feature
            ]
        })
        .collect();
    
    let mut forest = IsolationForest::builder()
        .n_estimators(100)
        .random_seed(42)
        .build();
    
    forest.fit(&data);
    
    let importances = forest.feature_importances();
    
    // Check that importances sum to approximately 1.0
    let sum: f32 = importances.iter().sum();
    assert!((sum - 1.0).abs() < 0.1);
    
    // Constant feature should have very low importance
    assert!(importances[2] < 0.05);
}

#[test]
fn test_reproducibility() {
    let data: Vec<Vec<f32>> = (0..50)
        .map(|i| vec![i as f32, (i as f32).sin()])
        .collect();
    
    let mut forest1 = IsolationForest::builder()
        .n_estimators(50)
        .random_seed(42)
        .build();
    
    let mut forest2 = IsolationForest::builder()
        .n_estimators(50)
        .random_seed(42)
        .build();
    
    forest1.fit(&data);
    forest2.fit(&data);
    
    let test_sample = vec![25.0, 0.0];
    let score1 = forest1.anomaly_score(&test_sample);
    let score2 = forest2.anomaly_score(&test_sample);
    
    assert_eq!(score1, score2);
}

#[test]
fn test_batch_prediction() {
    let training_data = vec![
        vec![0.0, 0.0],
        vec![1.0, 1.0],
        vec![2.0, 2.0],
        vec![10.0, 10.0], // Anomaly
    ];
    
    let mut forest = IsolationForest::builder()
        .contamination(0.25)
        .build();
    
    forest.fit(&training_data);
    
    let test_batch = vec![
        vec![1.5, 1.5],   // Normal
        vec![15.0, 15.0], // Anomaly
        vec![0.5, 0.5],   // Normal
        vec![-10.0, -10.0], // Anomaly
    ];
    
    let predictions = forest.predict(&test_batch);
    
    assert_eq!(predictions[0], 1);  // Normal
    assert_eq!(predictions[1], -1); // Anomaly
    assert_eq!(predictions[2], 1);  // Normal
    assert_eq!(predictions[3], -1); // Anomaly
}

#[test]
fn test_performance_requirement() {
    use std::time::Instant;
    
    // Train a model
    let data: Vec<Vec<f32>> = (0..1000)
        .map(|_| (0..10).map(|_| rand::random()).collect())
        .collect();
    
    let mut forest = IsolationForest::builder()
        .n_estimators(200)
        .build();
    
    forest.fit(&data);
    
    // Test inference time
    let test_sample = vec![0.5; 10];
    let warmup_iterations = 100;
    let test_iterations = 1000;
    
    // Warmup
    for _ in 0..warmup_iterations {
        let _ = forest.anomaly_score(&test_sample);
    }
    
    // Actual measurement
    let start = Instant::now();
    for _ in 0..test_iterations {
        let _ = forest.anomaly_score(&test_sample);
    }
    let elapsed = start.elapsed();
    
    let avg_time_micros = elapsed.as_micros() as f64 / test_iterations as f64;
    
    println!("Average inference time: {:.2}μs", avg_time_micros);
    
    // Check that it meets the <50μs requirement
    assert!(avg_time_micros < 50.0, 
            "Inference time {:.2}μs exceeds 50μs requirement", avg_time_micros);
}