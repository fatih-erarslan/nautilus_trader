//! Test CPU feature detection

fn main() {
    println!("Testing CPU Feature Detection");
    println!("=============================");
    
    // Test x86/x86_64 feature detection
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        println!("Architecture: x86/x86_64");
        println!("AVX-512: {}", is_x86_feature_detected!("avx512f"));
        println!("AVX2: {}", is_x86_feature_detected!("avx2"));
        println!("SSE4.1: {}", is_x86_feature_detected!("sse4.1"));
        println!("FMA: {}", is_x86_feature_detected!("fma"));
    }
    
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        println!("Architecture: non-x86");
        println!("All SIMD features disabled on this architecture");
    }
    
    println!("\nThe stdarch_x86_avx512 feature is only enabled on x86/x86_64 architectures.");
}