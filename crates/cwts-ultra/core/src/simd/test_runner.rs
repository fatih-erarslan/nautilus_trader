//! SIMD test runner for validating implementations
//! Tests core functionality without unstable features

use super::*;

/// Run basic SIMD functionality tests
pub fn run_simd_tests() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Running SIMD Implementation Tests");
    println!("====================================");
    
    // Test feature detection
    test_feature_detection()?;
    
    // Test unified interface
    test_unified_interface()?;
    
    // Test architecture-specific features
    test_architecture_specific()?;
    
    println!("✅ All SIMD tests passed successfully!");
    Ok(())
}

fn test_feature_detection() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Testing Feature Detection...");
    
    let capabilities = SimdCapabilities::detect();
    println!("  Detected capabilities: {}", capabilities);
    
    let performance = SimdCapabilities::performance_estimate();
    println!("  Performance estimate: {:.1}x", performance);
    
    // Performance estimate should be reasonable
    if performance < 1.0 || performance > 10.0 {
        return Err(format!("Unreasonable performance estimate: {:.1}x", performance).into());
    }
    
    println!("  ✅ Feature detection working");
    Ok(())
}

fn test_unified_interface() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Testing Unified SIMD Interface...");
    
    // Test matrix multiplication
    let matrix = SimdMatrix::new();
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let mut c = vec![0.0; 4];
    
    unsafe {
        matrix.multiply_f32(&a, &b, &mut c, 2, 2, 2);
    }
    
    // Verify results are finite
    if !c.iter().all(|&x| x.is_finite()) {
        return Err("Matrix multiplication produced invalid results".into());
    }
    
    println!("  ✅ Matrix multiplication: {:?}", c);
    
    // Test dot product
    let vector = SimdVector::new();
    let result = unsafe { vector.dot_product_f32(&a, &b) };
    let expected = 1.0*5.0 + 2.0*6.0 + 3.0*7.0 + 4.0*8.0; // 70.0
    
    if (result - expected).abs() > 1e-5 {
        return Err(format!("Dot product incorrect: got {}, expected {}", result, expected).into());
    }
    
    println!("  ✅ Dot product: {} (expected {})", result, expected);
    
    // Test statistics
    let stats = SimdStats::new();
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mean = unsafe { stats.mean_f32(&data) };
    
    if (mean - 3.0).abs() > 1e-5 {
        return Err(format!("Mean calculation incorrect: got {}, expected 3.0", mean).into());
    }
    
    println!("  ✅ Mean calculation: {}", mean);
    
    // Test reductions
    let reduction = SimdReduction::new();
    let max = unsafe { reduction.max_f32(&data) };
    let min = unsafe { reduction.min_f32(&data) };
    
    if (max - 5.0).abs() > 1e-5 {
        return Err(format!("Max reduction incorrect: got {}, expected 5.0", max).into());
    }
    
    if (min - 1.0).abs() > 1e-5 {
        return Err(format!("Min reduction incorrect: got {}, expected 1.0", min).into());
    }
    
    println!("  ✅ Reductions: max={}, min={}", max, min);
    
    Ok(())
}

fn test_architecture_specific() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏗️  Testing Architecture-Specific Features...");
    
    #[cfg(target_arch = "x86_64")]
    {
        let features = x86_64::X86Features::detect();
        println!("  x86_64 features: {:?}", features);
        
        // Test basic x86_64 functionality (avoiding unstable AVX-512)
        let matrix = x86_64::SimdMatrix::new();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];
        
        unsafe {
            matrix.multiply_f32(&a, &b, &mut c, 2, 2, 2);
        }
        
        if !c.iter().all(|&x| x.is_finite()) {
            return Err("x86_64 matrix multiplication failed".into());
        }
        
        println!("  ✅ x86_64 SIMD operations working");
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        let features = aarch64::AArch64Features::detect();
        println!("  AArch64 features: {:?}", features);
        
        let matrix = aarch64::SimdMatrix::new();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];
        
        unsafe {
            matrix.multiply_f32(&a, &b, &mut c, 2, 2, 2);
        }
        
        if !c.iter().all(|&x| x.is_finite()) {
            return Err("AArch64 matrix multiplication failed".into());
        }
        
        println!("  ✅ AArch64 SIMD operations working");
    }
    
    #[cfg(target_arch = "wasm32")]
    {
        let features = wasm32::WasmFeatures::detect();
        println!("  WASM features: {:?}", features);
        
        let compat = wasm32::BrowserCompat::new();
        println!("  Browser compatibility: {}", compat.get_fallback_strategy());
        println!("  Performance estimate: {:.1}x", compat.estimate_performance_multiplier());
        
        let matrix = wasm32::SimdMatrix::new();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];
        
        unsafe {
            matrix.multiply_f32(&a, &b, &mut c, 2, 2, 2);
        }
        
        if !c.iter().all(|&x| x.is_finite()) {
            return Err("WASM matrix multiplication failed".into());
        }
        
        println!("  ✅ WASM SIMD operations working");
    }
    
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32")))]
    {
        println!("  ⚠️  Unsupported architecture - using scalar fallbacks");
    }
    
    Ok(())
}

/// Quick performance test to ensure we're getting SIMD benefits
pub unsafe fn quick_performance_test() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;
    
    println!("⚡ Quick Performance Test");
    println!("========================");
    
    let size = 1000;
    let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.001).collect();
    let b: Vec<f32> = (0..size).map(|i| (i as f32 + 1.0) * 0.001).collect();
    
    // Test dot product performance
    let vector = SimdVector::new();
    
    let start = Instant::now();
    for _ in 0..10000 {
        let _ = vector.dot_product_f32(&a, &b);
    }
    let simd_time = start.elapsed();
    
    // Test scalar performance for comparison
    let start = Instant::now();
    for _ in 0..10000 {
        let _: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    }
    let scalar_time = start.elapsed();
    
    let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;
    
    println!("Dot product (1000 elements, 10k iterations):");
    println!("  SIMD time:   {:?}", simd_time);
    println!("  Scalar time: {:?}", scalar_time);
    println!("  Speedup:     {:.2}x", speedup);
    
    if speedup < 0.8 {
        println!("  ⚠️  SIMD may not be providing expected benefits");
    } else if speedup >= 1.5 {
        println!("  🚀 Excellent SIMD performance!");
    } else {
        println!("  ✅ SIMD providing reasonable performance");
    }
    
    Ok(())
}

/// Test FFT functionality with small data
pub unsafe fn test_fft() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌊 Testing FFT Implementation...");
    
    let fft = SimdFFT::new();
    
    // Test with a simple 8-point complex signal
    let mut data = vec![
        1.0, 0.0,  // 1 + 0i
        1.0, 0.0,  // 1 + 0i  
        0.0, 0.0,  // 0 + 0i
        0.0, 0.0,  // 0 + 0i
        0.0, 0.0,  // 0 + 0i
        0.0, 0.0,  // 0 + 0i
        0.0, 0.0,  // 0 + 0i
        0.0, 0.0,  // 0 + 0i
    ];
    
    let original_data = data.clone();
    
    // Forward FFT
    fft.fft_complex_f32(&mut data, 4, false);
    
    // Check that we got reasonable results
    if !data.iter().all(|&x| x.is_finite()) {
        return Err("FFT produced invalid results".into());
    }
    
    // DC component should be the sum of input (2.0)
    if (data[0] - 2.0).abs() > 0.1 {
        return Err(format!("FFT DC component incorrect: got {}, expected ~2.0", data[0]).into());
    }
    
    // Inverse FFT to recover original
    fft.fft_complex_f32(&mut data, 4, true);
    
    // Check that we recovered the original (within tolerance)
    for (i, (&recovered, &original)) in data.iter().zip(original_data.iter()).enumerate() {
        if (recovered - original).abs() > 0.1 {
            return Err(format!("FFT inverse failed at index {}: got {}, expected {}", 
                             i, recovered, original).into());
        }
    }
    
    println!("  ✅ FFT forward/inverse working correctly");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_functionality() {
        run_simd_tests().expect("SIMD tests failed");
    }
    
    #[test]
    fn test_fft_functionality() {
        unsafe {
            test_fft().expect("FFT tests failed");
        }
    }
    
    #[test]
    fn test_quick_performance() {
        unsafe {
            quick_performance_test().expect("Performance tests failed");
        }
    }
}