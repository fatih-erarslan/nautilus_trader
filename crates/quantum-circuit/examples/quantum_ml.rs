//! Quantum-Enhanced Machine Learning Examples
//!
//! This example demonstrates various quantum-enhanced machine learning
//! techniques including embeddings, quantum kernels, and hybrid networks.

use quantum_circuit::{
    embeddings::{
        AmplitudeEmbedding, AngleEmbedding, ParametricEmbedding, 
        QuantumKernelEmbedding, QuantumPCA, QuantumEmbedding,
        NormalizationMethod, KernelType, EntanglementPattern,
    },
    neural::SimpleHybridNet,
    Circuit, CircuitBuilder,
    optimization::{AdamOptimizer, OptimizerConfig, VariationalOptimizer},
};
use ndarray::{Array1, Array2};
use rand::Rng;
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 Quantum-Enhanced Machine Learning Examples");
    println!("=============================================");
    
    // Example 1: Quantum feature embeddings
    quantum_embedding_examples()?;
    
    // Example 2: Quantum kernel methods  
    quantum_kernel_examples()?;
    
    // Example 3: Quantum dimensionality reduction
    quantum_pca_example()?;
    
    // Example 4: Hybrid quantum-classical neural networks
    hybrid_neural_network_example()?;
    
    // Example 5: Quantum-enhanced classification
    quantum_classification_example()?;
    
    Ok(())
}

fn quantum_embedding_examples() -> quantum_circuit::Result<()> {
    println!("\n🌀 Quantum Feature Embedding Examples");
    println!("-------------------------------------");
    
    // Sample classical data
    let classical_data = vec![
        vec![0.1, 0.2, 0.3, 0.4],
        vec![0.5, 0.6, 0.7, 0.8],
        vec![0.2, 0.4, 0.1, 0.9],
        vec![0.8, 0.3, 0.6, 0.2],
    ];
    
    println!("Classical data samples:");
    for (i, sample) in classical_data.iter().enumerate() {
        println!("  Sample {}: {:?}", i + 1, sample);
    }
    
    // Amplitude Embedding
    println!("\n📊 Amplitude Embedding:");
    let amp_embedding = AmplitudeEmbedding::new(4, NormalizationMethod::L2);
    
    for (i, data) in classical_data.iter().enumerate() {
        let quantum_state = amp_embedding.embed(data)?;
        println!("  Sample {} embedded state (first 4 amplitudes):", i + 1);
        for (j, amplitude) in quantum_state.iter().take(4).enumerate() {
            println!("    |{}⟩: {:.4}", j, amplitude.norm());
        }
        
        // Verify normalization
        let norm: f64 = quantum_state.iter().map(|c| c.norm_sqr()).sum();
        println!("    Normalization: {:.6}", norm);
    }
    
    // Angle Embedding
    println!("\n📐 Angle Embedding:");
    let angle_embedding = AngleEmbedding::new(4).with_scale(PI);
    
    for (i, data) in classical_data.iter().enumerate().take(2) {
        let quantum_state = angle_embedding.embed(data)?;
        println!("  Sample {} quantum state:", i + 1);
        
        // Show the probability distribution
        for (j, amplitude) in quantum_state.iter().enumerate() {
            let prob = amplitude.norm_sqr();
            if prob > 0.01 {
                println!("    |{:04b}⟩: prob = {:.4}", j, prob);
            }
        }
    }
    
    // Parametric Embedding
    println!("\n⚙️ Parametric Embedding:");
    let param_embedding = ParametricEmbedding::new(4, 3, 2); // 4 features, 3 qubits, 2 layers
    
    println!("  Embedding parameters: {}", param_embedding.parameters().len());
    
    for (i, data) in classical_data.iter().enumerate().take(2) {
        let quantum_state = param_embedding.embed(data)?;
        
        // Calculate some quantum features
        let amplitude_sum: f64 = quantum_state.iter().map(|c| c.norm()).sum();
        let phase_variance = calculate_phase_variance(&quantum_state);
        
        println!("  Sample {} quantum features:", i + 1);
        println!("    Amplitude sum: {:.4}", amplitude_sum);
        println!("    Phase variance: {:.4}", phase_variance);
    }
    
    Ok(())
}

fn quantum_kernel_examples() -> quantum_circuit::Result<()> {
    println!("\n🔗 Quantum Kernel Methods");
    println!("-------------------------");
    
    // Generate some sample data for classification
    let data_class_0 = vec![
        vec![0.1, 0.2],
        vec![0.2, 0.1],
        vec![0.15, 0.25],
    ];
    
    let data_class_1 = vec![
        vec![0.8, 0.9],
        vec![0.9, 0.8],
        vec![0.85, 0.95],
    ];
    
    let all_data: Vec<Vec<f64>> = data_class_0.iter()
        .chain(data_class_1.iter())
        .cloned()
        .collect();
    
    // Test different kernel types
    let kernel_types = vec![
        ("Linear", KernelType::Linear),
        ("RBF (γ=1.0)", KernelType::RBF { gamma: 1.0 }),
        ("Polynomial (deg=2)", KernelType::Polynomial { degree: 2, coeff: 1.0 }),
    ];
    
    for (name, kernel_type) in kernel_types {
        println!("\n{} Quantum Kernel:", name);
        
        let kernel_embedding = QuantumKernelEmbedding::new(2, 2, 2, kernel_type);
        
        // Compute kernel matrix
        let kernel_matrix = kernel_embedding.kernel_matrix(&all_data)?;
        
        println!("  Kernel matrix:");
        for i in 0..kernel_matrix.nrows() {
            print!("    ");
            for j in 0..kernel_matrix.ncols() {
                print!("{:6.3} ", kernel_matrix[[i, j]]);
            }
            println!();
        }
        
        // Analyze kernel properties
        println!("  Kernel properties:");
        
        // Intra-class similarities
        let intra_class_0: f64 = (0..3).flat_map(|i| ((i+1)..3).map(move |j| kernel_matrix[[i, j]]))
                                        .sum::<f64>() / 3.0; // 3 pairs in class 0
        let intra_class_1: f64 = (3..6).flat_map(|i| ((i+1)..6).map(move |j| kernel_matrix[[i, j]]))
                                        .sum::<f64>() / 3.0; // 3 pairs in class 1
        
        // Inter-class similarities
        let inter_class: f64 = (0..3).flat_map(|i| (3..6).map(move |j| kernel_matrix[[i, j]]))
                                     .sum::<f64>() / 9.0; // 3x3 = 9 pairs
        
        println!("    Avg intra-class similarity (class 0): {:.4}", intra_class_0);
        println!("    Avg intra-class similarity (class 1): {:.4}", intra_class_1);
        println!("    Avg inter-class similarity: {:.4}", inter_class);
        
        let separability = (intra_class_0 + intra_class_1) / 2.0 - inter_class;
        println!("    Separability score: {:.4}", separability);
    }
    
    Ok(())
}

fn quantum_pca_example() -> quantum_circuit::Result<()> {
    println!("\n📊 Quantum Principal Component Analysis");
    println!("--------------------------------------");
    
    // Generate correlated 2D data
    let mut rng = rand::thread_rng();
    let mut data = Vec::new();
    
    for _ in 0..20 {
        let x = rng.gen_range(-1.0..1.0);
        let y = 0.8 * x + 0.2 * rng.gen_range(-1.0..1.0); // Correlated with noise
        data.push(vec![x, y]);
    }
    
    println!("Original 2D data (first 5 samples):");
    for (i, sample) in data.iter().take(5).enumerate() {
        println!("  Sample {}: ({:.3}, {:.3})", i + 1, sample[0], sample[1]);
    }
    
    // Apply Quantum PCA
    let mut qpca = QuantumPCA::new(2, 1, 2); // 2D input, 1D output, 2 qubits
    
    println!("\nFitting Quantum PCA...");
    qpca.fit(&data)?;
    
    // Transform the data
    let transformed_data = qpca.transform(&data)?;
    
    println!("Transformed 1D data (first 5 samples):");
    for i in 0..5.min(transformed_data.nrows()) {
        println!("  Sample {}: {:.6}", i + 1, transformed_data[[i, 0]]);
    }
    
    // Calculate variance preservation
    let original_variance = calculate_variance_2d(&data);
    let transformed_variance = calculate_variance_1d(&transformed_data);
    
    println!("\nVariance Analysis:");
    println!("  Original data variance: {:.6}", original_variance);
    println!("  Transformed data variance: {:.6}", transformed_variance);
    println!("  Variance preservation: {:.2}%", 100.0 * transformed_variance / original_variance);
    
    Ok(())
}

fn hybrid_neural_network_example() -> quantum_circuit::Result<()> {
    println!("\n🧠 Hybrid Quantum-Classical Neural Network");
    println!("------------------------------------------");
    
    // Create XOR dataset (classic non-linear problem)
    let train_x = Array2::from_shape_vec(
        (4, 2),
        vec![
            0.0, 0.0,  // XOR(0,0) = 0
            0.0, 1.0,  // XOR(0,1) = 1
            1.0, 0.0,  // XOR(1,0) = 1
            1.0, 1.0,  // XOR(1,1) = 0
        ]
    ).unwrap();
    
    let train_y = Array2::from_shape_vec(
        (4, 2), // One-hot encoding
        vec![
            1.0, 0.0,  // Class 0
            0.0, 1.0,  // Class 1
            0.0, 1.0,  // Class 1
            1.0, 0.0,  // Class 0
        ]
    ).unwrap();
    
    println!("XOR Training Dataset:");
    for i in 0..4 {
        println!("  Input: ({:.0}, {:.0}) -> Output: ({:.0}, {:.0})", 
                train_x[[i, 0]], train_x[[i, 1]], 
                train_y[[i, 0]], train_y[[i, 1]]);
    }
    
    // Create hybrid network
    let mut hybrid_net = SimpleHybridNet::new(2, 4, 2, 2); // 2 input, 4 hidden, 2 output, 2 qubits
    
    println!("\nTraining Hybrid Neural Network...");
    let history = hybrid_net.train(&train_x, &train_y, 50)?;
    
    println!("Training completed!");
    println!("  Final loss: {:.6}", history.losses.last().unwrap_or(&0.0));
    println!("  Final accuracy: {:.2}%", 100.0 * history.accuracies.last().unwrap_or(&0.0));
    
    // Test the trained network
    println!("\nTesting trained network:");
    let predictions = hybrid_net.forward(&train_x)?;
    
    for i in 0..4 {
        let pred_class = if predictions[[i, 0]] > predictions[[i, 1]] { 0 } else { 1 };
        let true_class = if train_y[[i, 0]] > train_y[[i, 1]] { 0 } else { 1 };
        let correct = if pred_class == true_class { "✓" } else { "✗" };
        
        println!("  Input: ({:.0}, {:.0}) -> Predicted: {} (confidence: {:.3}) {} True: {}", 
                train_x[[i, 0]], train_x[[i, 1]], pred_class,
                predictions[[i, pred_class]], correct, true_class);
    }
    
    // Show training progress
    println!("\nTraining Progress:");
    let step_size = (history.losses.len() / 10).max(1);
    for (i, (loss, acc)) in history.losses.iter()
        .zip(history.accuracies.iter())
        .enumerate()
        .step_by(step_size) {
        println!("  Epoch {}: Loss = {:.4}, Accuracy = {:.2}%", i, loss, 100.0 * acc);
    }
    
    Ok(())
}

fn quantum_classification_example() -> quantum_circuit::Result<()> {
    println!("\n🎯 Quantum-Enhanced Binary Classification");
    println!("----------------------------------------");
    
    // Generate linearly separable dataset
    let mut data_class_0 = Vec::new();
    let mut data_class_1 = Vec::new();
    
    let mut rng = rand::thread_rng();
    
    // Class 0: centered around (0.3, 0.3)
    for _ in 0..10 {
        let x = 0.3 + 0.2 * rng.gen_range(-1.0..1.0);
        let y = 0.3 + 0.2 * rng.gen_range(-1.0..1.0);
        data_class_0.push(vec![x, y]);
    }
    
    // Class 1: centered around (0.7, 0.7)  
    for _ in 0..10 {
        let x = 0.7 + 0.2 * rng.gen_range(-1.0..1.0);
        let y = 0.7 + 0.2 * rng.gen_range(-1.0..1.0);
        data_class_1.push(vec![x, y]);
    }
    
    // Create quantum feature map
    let feature_map = ParametricEmbedding::new(2, 2, 3)
        .with_entanglement(EntanglementPattern::Circular);
    
    println!("Dataset:");
    println!("  Class 0 samples: {}", data_class_0.len());
    println!("  Class 1 samples: {}", data_class_1.len());
    
    // Quantum feature extraction
    println!("\nExtracting quantum features...");
    
    let mut quantum_features_0 = Vec::new();
    let mut quantum_features_1 = Vec::new();
    
    for sample in &data_class_0 {
        let quantum_state = feature_map.embed(sample)?;
        let features = extract_quantum_features(&quantum_state);
        quantum_features_0.push(features);
    }
    
    for sample in &data_class_1 {
        let quantum_state = feature_map.embed(sample)?;
        let features = extract_quantum_features(&quantum_state);
        quantum_features_1.push(features);
    }
    
    // Analyze quantum feature separability
    let mean_features_0 = calculate_mean_features(&quantum_features_0);
    let mean_features_1 = calculate_mean_features(&quantum_features_1);
    
    println!("Quantum feature analysis:");
    println!("  Class 0 mean features: {:?}", 
             mean_features_0.iter().map(|&x| format!("{:.3}", x)).collect::<Vec<_>>());
    println!("  Class 1 mean features: {:?}", 
             mean_features_1.iter().map(|&x| format!("{:.3}", x)).collect::<Vec<_>>());
    
    // Calculate separation in quantum feature space
    let feature_distance = euclidean_distance(&mean_features_0, &mean_features_1);
    println!("  Feature space separation: {:.4}", feature_distance);
    
    // Simple quantum-inspired classifier using feature distance
    println!("\nTesting quantum-inspired classification:");
    
    let test_samples = vec![
        vec![0.25, 0.35], // Should be class 0
        vec![0.75, 0.65], // Should be class 1
        vec![0.45, 0.55], // Boundary region
        vec![0.2, 0.8],   // Mixed region
    ];
    
    for (i, sample) in test_samples.iter().enumerate() {
        let quantum_state = feature_map.embed(sample)?;
        let features = extract_quantum_features(&quantum_state);
        
        let dist_to_class_0 = euclidean_distance(&features, &mean_features_0);
        let dist_to_class_1 = euclidean_distance(&features, &mean_features_1);
        
        let predicted_class = if dist_to_class_0 < dist_to_class_1 { 0 } else { 1 };
        let confidence = (dist_to_class_1 - dist_to_class_0).abs() / (dist_to_class_0 + dist_to_class_1);
        
        println!("  Test sample {}: {:?} -> Class {} (confidence: {:.3})", 
                i + 1, sample, predicted_class, confidence);
    }
    
    Ok(())
}

// Helper functions
fn calculate_phase_variance(state: &quantum_circuit::StateVector) -> f64 {
    let phases: Vec<f64> = state.iter().map(|c| c.arg()).collect();
    let mean_phase: f64 = phases.iter().sum::<f64>() / phases.len() as f64;
    
    phases.iter()
        .map(|&phase| (phase - mean_phase).powi(2))
        .sum::<f64>() / phases.len() as f64
}

fn calculate_variance_2d(data: &[Vec<f64>]) -> f64 {
    let n = data.len() as f64;
    let mean_x: f64 = data.iter().map(|sample| sample[0]).sum::<f64>() / n;
    let mean_y: f64 = data.iter().map(|sample| sample[1]).sum::<f64>() / n;
    
    let var_x: f64 = data.iter()
        .map(|sample| (sample[0] - mean_x).powi(2))
        .sum::<f64>() / n;
    let var_y: f64 = data.iter()
        .map(|sample| (sample[1] - mean_y).powi(2))
        .sum::<f64>() / n;
        
    var_x + var_y
}

fn calculate_variance_1d(data: &Array2<f64>) -> f64 {
    let values: Vec<f64> = data.column(0).to_vec();
    let n = values.len() as f64;
    let mean: f64 = values.iter().sum::<f64>() / n;
    
    values.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / n
}

fn extract_quantum_features(state: &quantum_circuit::StateVector) -> Vec<f64> {
    let mut features = Vec::new();
    
    // Feature 1: Total amplitude sum
    features.push(state.iter().map(|c| c.norm()).sum::<f64>());
    
    // Feature 2: Maximum amplitude
    features.push(state.iter().map(|c| c.norm()).fold(0.0, f64::max));
    
    // Feature 3: Phase coherence (variance of phases)
    let phases: Vec<f64> = state.iter().map(|c| c.arg()).collect();
    let mean_phase = phases.iter().sum::<f64>() / phases.len() as f64;
    let phase_variance = phases.iter()
        .map(|&p| (p - mean_phase).powi(2))
        .sum::<f64>() / phases.len() as f64;
    features.push(1.0 / (1.0 + phase_variance)); // Coherence measure
    
    // Feature 4: Entropy-like measure
    let probabilities: Vec<f64> = state.iter().map(|c| c.norm_sqr()).collect();
    let entropy = -probabilities.iter()
        .filter(|&&p| p > 1e-10)
        .map(|&p| p * p.ln())
        .sum::<f64>();
    features.push(entropy);
    
    features
}

fn calculate_mean_features(feature_vectors: &[Vec<f64>]) -> Vec<f64> {
    let n_features = feature_vectors[0].len();
    let n_samples = feature_vectors.len() as f64;
    
    (0..n_features)
        .map(|i| {
            feature_vectors.iter()
                .map(|features| features[i])
                .sum::<f64>() / n_samples
        })
        .collect()
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}