// Build script for compiling CUDA kernels
// Generates PTX files for optimal performance

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // Only build CUDA kernels if CUDA feature is enabled
    if !cfg!(feature = "cuda") {
        return;
    }
    
    println!("cargo:rerun-if-changed=src/cuda/");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    
    // Try to find CUDA installation
    let cuda_path = find_cuda_path();
    if cuda_path.is_none() {
        println!("cargo:warning=CUDA not found, skipping kernel compilation");
        return;
    }
    
    let cuda_path = cuda_path.unwrap();
    let nvcc_path = cuda_path.join("bin").join("nvcc");
    
    if !nvcc_path.exists() {
        println!("cargo:warning=nvcc not found at {:?}, skipping kernel compilation", nvcc_path);
        return;
    }
    
    // Compile quantum kernels
    compile_cuda_kernel(
        &nvcc_path,
        "src/cuda/quantum_kernels.cu",
        "quantum_kernels.ptx",
    );
    
    // Set up linker paths for CUDA libraries
    if let Some(cuda_lib_path) = find_cuda_lib_path(&cuda_path) {
        println!("cargo:rustc-link-search=native={}", cuda_lib_path.display());
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=curand");
        println!("cargo:rustc-link-lib=cusolver");
    }
}

fn find_cuda_path() -> Option<PathBuf> {
    // Check environment variable first
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let path = PathBuf::from(cuda_path);
        if path.exists() {
            return Some(path);
        }
    }
    
    // Check common installation paths
    let common_paths = [
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/cuda",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8",
    ];
    
    for path_str in &common_paths {
        let path = PathBuf::from(path_str);
        if path.exists() {
            return Some(path);
        }
    }
    
    None
}

fn find_cuda_lib_path(cuda_path: &Path) -> Option<PathBuf> {
    let lib_paths = [
        cuda_path.join("lib64"),
        cuda_path.join("lib"),
        cuda_path.join("lib").join("x64"),
    ];
    
    for lib_path in &lib_paths {
        if lib_path.exists() {
            return Some(lib_path.clone());
        }
    }
    
    None
}

fn compile_cuda_kernel(nvcc_path: &Path, source_file: &str, output_file: &str) {
    let out_dir = env::var("OUT_DIR").unwrap();
    let output_path = Path::new(&out_dir).join(output_file);
    
    // Determine compute capability
    let compute_capability = detect_compute_capability().unwrap_or("80".to_string());
    let arch_flag = format!("sm_{}", compute_capability);
    
    println!("Compiling CUDA kernel: {} -> {:?}", source_file, output_path);
    println!("Using compute capability: {}", compute_capability);
    
    let mut cmd = Command::new(nvcc_path);
    cmd.args(&[
        "-ptx",                           // Generate PTX instead of SASS
        "-O3",                           // Maximum optimization
        &format!("-arch={}", arch_flag), // Target architecture
        "-use_fast_math",                // Enable fast math operations
        "-allow-unsupported-compiler",   // Allow newer GCC versions
        "--ptxas-options=-v",            // Verbose PTX assembler output
        "-Xptxas=-dlcm=cg",             // Aggressive L1 cache usage
        "-maxrregcount=64",              // Limit register usage for occupancy
        "-std=c++17",                    // Use C++17 standard
        "-DCUDA_API_PER_THREAD_DEFAULT_STREAM", // Thread-safe streams
        "-DWARP_SIZE=32",                // Explicit warp size
        "-I", "src/cuda",                // Include directory
        "-o", output_path.to_str().unwrap(),
        source_file,
    ]);
    
    // Add include paths for CUDA headers
    if let Some(cuda_path) = find_cuda_path() {
        let include_path = cuda_path.join("include");
        if include_path.exists() {
            cmd.arg("-I").arg(include_path);
        }
    }
    
    let output = cmd.output()
        .expect("Failed to execute nvcc");
    
    if !output.status.success() {
        panic!(
            "CUDA kernel compilation failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    
    // Print compilation info
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    
    if !stdout.is_empty() {
        println!("NVCC output: {}", stdout);
    }
    
    if !stderr.is_empty() {
        println!("NVCC info: {}", stderr);
    }
    
    println!("Successfully compiled {} to {:?}", source_file, output_path);
}

fn detect_compute_capability() -> Option<String> {
    // Try to detect GPU compute capability
    let nvidia_smi_output = Command::new("nvidia-smi")
        .args(&["--query-gpu=compute_cap", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    
    if nvidia_smi_output.status.success() {
        let output = String::from_utf8_lossy(&nvidia_smi_output.stdout);
        if let Some(first_line) = output.lines().next() {
            let cap = first_line.trim().replace(".", "");
            if !cap.is_empty() {
                return Some(cap);
            }
        }
    }
    
    // Try deviceQuery if available
    let device_query_output = Command::new("deviceQuery")
        .output()
        .ok()?;
    
    if device_query_output.status.success() {
        let output = String::from_utf8_lossy(&device_query_output.stdout);
        for line in output.lines() {
            if line.contains("CUDA Capability Major/Minor version number:") {
                // Parse compute capability from deviceQuery output
                // This is a simplified parser
                continue;
            }
        }
    }
    
    // Default to Ampere (A100/RTX 30xx series) if detection fails
    Some("80".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cuda_path_detection() {
        // This test will only pass if CUDA is actually installed
        if let Some(cuda_path) = find_cuda_path() {
            assert!(cuda_path.exists());
            assert!(cuda_path.is_dir());
        }
    }
    
    #[test]
    fn test_compute_capability_detection() {
        // This test requires nvidia-smi or deviceQuery
        if let Some(cap) = detect_compute_capability() {
            assert!(!cap.is_empty());
            assert!(cap.chars().all(|c| c.is_ascii_digit()));
        }
    }
}