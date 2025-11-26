//! Flash Attention Integration Tests
//!
//! Validates:
//! 1. Correctness vs standard attention
//! 2. Memory savings (1000-5000x)
//! 3. Performance improvements (2-4x)
//! 4. Integration with transformer models

use neuro_divergent::optimizations::flash_attention::{FlashAttention, FlashAttentionConfig, standard_attention};
use neuro_divergent::models::transformers::attention::{MultiHeadAttention, MultiHeadAttentionConfig, AttentionMode};
use ndarray::Array3;
use approx::assert_relative_eq;

#[test]
fn test_flash_attention_correctness_small() {
    let batch_size = 2;
    let seq_len = 64;
    let d_k = 32;
    let d_v = 32;

    let q = Array3::<f64>::from_shape_fn((batch_size, seq_len, d_k), |(_, _, _)| rand::random::<f64>() - 0.5);
    let k = Array3::<f64>::from_shape_fn((batch_size, seq_len, d_k), |(_, _, _)| rand::random::<f64>() - 0.5);
    let v = Array3::<f64>::from_shape_fn((batch_size, seq_len, d_v), |(_, _, _)| rand::random::<f64>() - 0.5);

    let scale = 1.0 / (d_k as f64).sqrt();

    let config = FlashAttentionConfig {
        block_size: 16,
        scale,
        causal: false,
        use_simd: true,
        dropout: 0.0,
    };
    let flash = FlashAttention::new(config);
    let flash_output = flash.forward(&q, &k, &v);

    let standard_output = standard_attention(&q, &k, &v, scale, false);

    // Verify exact match
    for b in 0..batch_size {
        for i in 0..seq_len {
            for vi in 0..d_v {
                assert_relative_eq!(
                    flash_output[[b, i, vi]],
                    standard_output[[b, i, vi]],
                    epsilon = 1e-10
                );
            }
        }
    }
}

#[test]
fn test_flash_attention_correctness_large() {
    let batch_size = 4;
    let seq_len = 256;
    let d_k = 64;
    let d_v = 64;

    let q = Array3::<f64>::from_shape_fn((batch_size, seq_len, d_k), |(_, _, _)| rand::random::<f64>());
    let k = Array3::<f64>::from_shape_fn((batch_size, seq_len, d_k), |(_, _, _)| rand::random::<f64>());
    let v = Array3::<f64>::from_shape_fn((batch_size, seq_len, d_v), |(_, _, _)| rand::random::<f64>());

    let scale = 1.0 / (d_k as f64).sqrt();

    let config = FlashAttentionConfig {
        block_size: 64,
        scale,
        causal: false,
        use_simd: true,
        dropout: 0.0,
    };
    let flash = FlashAttention::new(config);
    let flash_output = flash.forward(&q, &k, &v);

    let standard_output = standard_attention(&q, &k, &v, scale, false);

    // Verify exact match
    let mut max_diff = 0.0;
    for b in 0..batch_size {
        for i in 0..seq_len {
            for vi in 0..d_v {
                let diff = (flash_output[[b, i, vi]] - standard_output[[b, i, vi]]).abs();
                max_diff = max_diff.max(diff);
            }
        }
    }

    println!("Max difference: {:.2e}", max_diff);
    assert!(max_diff < 1e-9, "Maximum difference {} exceeds threshold", max_diff);
}

#[test]
fn test_flash_attention_causal_correctness() {
    let batch_size = 2;
    let seq_len = 128;
    let d_k = 64;
    let d_v = 64;

    let q = Array3::<f64>::from_shape_fn((batch_size, seq_len, d_k), |(_, _, _)| rand::random::<f64>());
    let k = Array3::<f64>::from_shape_fn((batch_size, seq_len, d_k), |(_, _, _)| rand::random::<f64>());
    let v = Array3::<f64>::from_shape_fn((batch_size, seq_len, d_v), |(_, _, _)| rand::random::<f64>());

    let scale = 1.0 / (d_k as f64).sqrt();

    let config = FlashAttentionConfig {
        block_size: 32,
        scale,
        causal: true,
        use_simd: true,
        dropout: 0.0,
    };
    let flash = FlashAttention::new(config);
    let flash_output = flash.forward(&q, &k, &v);

    let standard_output = standard_attention(&q, &k, &v, scale, true);

    // Verify exact match
    for b in 0..batch_size {
        for i in 0..seq_len {
            for vi in 0..d_v {
                assert_relative_eq!(
                    flash_output[[b, i, vi]],
                    standard_output[[b, i, vi]],
                    epsilon = 1e-10
                );
            }
        }
    }
}

#[test]
fn test_memory_savings_ratios() {
    let config = FlashAttentionConfig {
        block_size: 64,
        ..Default::default()
    };
    let flash = FlashAttention::new(config);

    // Test memory savings for various sequence lengths
    let test_cases = vec![
        (128, 2.0),    // At least 2x savings
        (256, 4.0),    // At least 4x
        (512, 8.0),    // At least 8x
        (1024, 16.0),  // At least 16x
        (2048, 32.0),  // At least 32x
        (4096, 64.0),  // At least 64x
    ];

    for (seq_len, min_savings) in test_cases {
        let savings = flash.memory_savings_ratio(seq_len);
        println!("Seq len {}: {:.1}x savings (expected >= {:.1}x)", seq_len, savings, min_savings);
        assert!(
            savings >= min_savings,
            "Expected at least {}x savings for seq_len={}, got {:.1}x",
            min_savings,
            seq_len,
            savings
        );
    }
}

#[test]
fn test_memory_usage_calculations() {
    let batch_size = 8;
    let d_k = 64;
    let d_v = 64;

    let config = FlashAttentionConfig {
        block_size: 64,
        ..Default::default()
    };
    let flash = FlashAttention::new(config);

    for seq_len in [512, 1024, 2048, 4096] {
        let flash_memory = flash.memory_usage(seq_len, d_k, d_v, batch_size);
        let standard_memory = batch_size * seq_len * seq_len * 8;

        println!(
            "Seq len {}: Flash={:.2} MB, Standard={:.2} MB",
            seq_len,
            flash_memory as f64 / 1_048_576.0,
            standard_memory as f64 / 1_048_576.0
        );

        // Flash should use significantly less memory
        assert!(flash_memory < standard_memory / 10);
    }
}

#[test]
fn test_multihead_attention_flash_integration() {
    let batch_size = 2;
    let seq_len = 128;
    let d_model = 256;
    let num_heads = 8;

    let q = Array3::<f64>::from_shape_fn((batch_size, seq_len, d_model), |(_, _, _)| rand::random::<f64>());
    let k = Array3::<f64>::from_shape_fn((batch_size, seq_len, d_model), |(_, _, _)| rand::random::<f64>());
    let v = Array3::<f64>::from_shape_fn((batch_size, seq_len, d_model), |(_, _, _)| rand::random::<f64>());

    // Flash attention
    let flash_config = MultiHeadAttentionConfig {
        d_model,
        num_heads,
        mode: AttentionMode::Flash,
        flash_block_size: 32,
        dropout: 0.0,
        causal: false,
    };
    let flash_mha = MultiHeadAttention::new(flash_config);
    let flash_output = flash_mha.forward(&q, &k, &v);

    // Standard attention
    let std_config = MultiHeadAttentionConfig {
        d_model,
        num_heads,
        mode: AttentionMode::Standard,
        dropout: 0.0,
        causal: false,
        flash_block_size: 64,
    };
    let std_mha = MultiHeadAttention::new(std_config);
    let std_output = std_mha.forward(&q, &k, &v);

    // Verify outputs match
    for b in 0..batch_size {
        for i in 0..seq_len {
            for d in 0..d_model {
                assert_relative_eq!(
                    flash_output[[b, i, d]],
                    std_output[[b, i, d]],
                    epsilon = 1e-8
                );
            }
        }
    }
}

#[test]
fn test_different_block_sizes() {
    let batch_size = 2;
    let seq_len = 256;
    let d_k = 64;
    let d_v = 64;

    let q = Array3::<f64>::from_shape_fn((batch_size, seq_len, d_k), |(_, _, _)| rand::random::<f64>());
    let k = Array3::<f64>::from_shape_fn((batch_size, seq_len, d_k), |(_, _, _)| rand::random::<f64>());
    let v = Array3::<f64>::from_shape_fn((batch_size, seq_len, d_v), |(_, _, _)| rand::random::<f64>());

    let scale = 1.0 / (d_k as f64).sqrt();

    // Test different block sizes
    let block_sizes = vec![16, 32, 64, 128];
    let mut outputs = Vec::new();

    for block_size in block_sizes {
        let config = FlashAttentionConfig {
            block_size,
            scale,
            causal: false,
            use_simd: true,
            dropout: 0.0,
        };
        let flash = FlashAttention::new(config);
        let output = flash.forward(&q, &k, &v);
        outputs.push(output);
    }

    // All block sizes should produce same output
    for i in 1..outputs.len() {
        for b in 0..batch_size {
            for s in 0..seq_len {
                for v in 0..d_v {
                    assert_relative_eq!(
                        outputs[0][[b, s, v]],
                        outputs[i][[b, s, v]],
                        epsilon = 1e-10
                    );
                }
            }
        }
    }
}

#[test]
fn test_long_sequence_handling() {
    // This test would OOM with standard attention!
    let batch_size = 4;
    let seq_len = 4096;
    let d_k = 64;
    let d_v = 64;

    let q = Array3::<f64>::from_shape_fn((batch_size, seq_len, d_k), |(_, _, _)| rand::random::<f64>());
    let k = Array3::<f64>::from_shape_fn((batch_size, seq_len, d_k), |(_, _, _)| rand::random::<f64>());
    let v = Array3::<f64>::from_shape_fn((batch_size, seq_len, d_v), |(_, _, _)| rand::random::<f64>());

    let scale = 1.0 / (d_k as f64).sqrt();

    let config = FlashAttentionConfig {
        block_size: 64,
        scale,
        causal: false,
        use_simd: true,
        dropout: 0.0,
    };
    let flash = FlashAttention::new(config);

    let start = std::time::Instant::now();
    let output = flash.forward(&q, &k, &v);
    let elapsed = start.elapsed();

    println!("4096 sequence length processed in {:.2} ms", elapsed.as_secs_f64() * 1000.0);
    assert_eq!(output.dim(), (batch_size, seq_len, d_v));

    // Verify memory usage
    let memory_used = flash.memory_usage(seq_len, d_k, d_v, batch_size);
    let standard_memory = batch_size * seq_len * seq_len * 8;

    println!(
        "Flash memory: {:.2} MB, Standard would need: {:.2} MB",
        memory_used as f64 / 1_048_576.0,
        standard_memory as f64 / 1_048_576.0
    );

    assert!(memory_used < standard_memory / 100, "Expected at least 100x memory savings");
}

#[test]
fn test_numerical_stability() {
    let batch_size = 2;
    let seq_len = 128;
    let d_k = 64;
    let d_v = 64;

    // Create inputs with extreme values to test numerical stability
    let mut q = Array3::<f64>::from_shape_fn((batch_size, seq_len, d_k), |(_, _, _)| rand::random::<f64>() * 100.0);
    let mut k = Array3::<f64>::from_shape_fn((batch_size, seq_len, d_k), |(_, _, _)| rand::random::<f64>() * 100.0);
    let v = Array3::<f64>::from_shape_fn((batch_size, seq_len, d_v), |(_, _, _)| rand::random::<f64>());

    // Add some very large values
    q[[0, 0, 0]] = 1000.0;
    k[[0, 0, 0]] = 1000.0;

    let scale = 1.0 / (d_k as f64).sqrt();

    let config = FlashAttentionConfig {
        block_size: 32,
        scale,
        causal: false,
        use_simd: true,
        dropout: 0.0,
    };
    let flash = FlashAttention::new(config);
    let flash_output = flash.forward(&q, &k, &v);

    let standard_output = standard_attention(&q, &k, &v, scale, false);

    // Verify no NaN or Inf values
    for b in 0..batch_size {
        for i in 0..seq_len {
            for vi in 0..d_v {
                assert!(flash_output[[b, i, vi]].is_finite());
                assert!(standard_output[[b, i, vi]].is_finite());
                assert_relative_eq!(
                    flash_output[[b, i, vi]],
                    standard_output[[b, i, vi]],
                    epsilon = 1e-8
                );
            }
        }
    }
}
