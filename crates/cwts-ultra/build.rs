use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// CWTS Ultra GPU Build Infrastructure
/// Comprehensive build script for CUDA, ROCm/HIP, Metal, and Vulkan support
/// Handles detection, compilation, and linking of GPU kernels across platforms

fn main() {
    println!("cargo:rerun-if-changed=gpu-kernels/");
    println!("cargo:rerun-if-changed=build.rs");
    
    let mut gpu_features = Vec::new();
    
    // Get output directory for compiled kernels
    let out_dir = env::var("OUT_DIR").unwrap();
    let kernel_out_dir = Path::new(&out_dir).join("gpu_kernels");
    fs::create_dir_all(&kernel_out_dir).expect("Failed to create kernel output directory");
    
    // Platform detection and feature configuration
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    
    println!("cargo:rustc-env=TARGET_OS={}", target_os);
    println!("cargo:rustc-env=TARGET_ARCH={}", target_arch);
    
    // CUDA Detection and Compilation
    if detect_cuda() {
        println!("cargo:warning=CUDA toolkit detected, building CUDA kernels");
        if compile_cuda_kernels(&kernel_out_dir) {
            gpu_features.push("cuda");
            link_cuda_libraries();
        } else {
            println!("cargo:warning=CUDA compilation failed, disabling CUDA support");
        }
    } else {
        println!("cargo:warning=CUDA toolkit not found, skipping CUDA compilation");
    }
    
    // ROCm/HIP Detection and Compilation
    if detect_rocm() {
        println!("cargo:warning=ROCm/HIP toolkit detected, building HIP kernels");
        if compile_hip_kernels(&kernel_out_dir) {
            gpu_features.push("rocm");
            link_hip_libraries();
        } else {
            println!("cargo:warning=HIP compilation failed, disabling ROCm support");
        }
    } else {
        println!("cargo:warning=ROCm/HIP toolkit not found, skipping HIP compilation");
    }
    
    // Metal Detection and Compilation (macOS/iOS only)
    if target_os == "macos" || target_os == "ios" {
        if detect_metal() {
            println!("cargo:warning=Metal SDK detected, building Metal kernels");
            if compile_metal_kernels(&kernel_out_dir) {
                gpu_features.push("metal");
                link_metal_libraries();
            } else {
                println!("cargo:warning=Metal compilation failed, disabling Metal support");
            }
        } else {
            println!("cargo:warning=Metal SDK not found, skipping Metal compilation");
        }
    }
    
    // Vulkan Detection and Compilation
    if detect_vulkan() {
        println!("cargo:warning=Vulkan SDK detected, building Vulkan shaders");
        if compile_vulkan_shaders(&kernel_out_dir) {
            gpu_features.push("vulkan");
            link_vulkan_libraries();
        } else {
            println!("cargo:warning=Vulkan compilation failed, disabling Vulkan support");
        }
    } else {
        println!("cargo:warning=Vulkan SDK not found, skipping Vulkan compilation");
    }
    
    // Generate feature configuration
    generate_gpu_config(&gpu_features);
    
    // Set optimization flags
    set_optimization_flags(&target_arch);
    
    // Generate runtime feature detection
    generate_feature_detection(&gpu_features);
    
    println!("cargo:warning=GPU build completed with features: {:?}", gpu_features);
}

/// Detect CUDA toolkit installation
fn detect_cuda() -> bool {
    // Check for CUDA_PATH environment variable
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let nvcc_path = Path::new(&cuda_path).join("bin").join("nvcc");
        if nvcc_path.exists() {
            return true;
        }
    }
    
    // Check for CUDA_HOME
    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        let nvcc_path = Path::new(&cuda_home).join("bin").join("nvcc");
        if nvcc_path.exists() {
            return true;
        }
    }
    
    // Check common CUDA installation paths
    let cuda_paths = [
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/cuda",
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0",
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8",
    ];
    
    for path in &cuda_paths {
        let nvcc_path = Path::new(path).join("bin").join(if cfg!(windows) { "nvcc.exe" } else { "nvcc" });
        if nvcc_path.exists() {
            env::set_var("CUDA_PATH", path);
            return true;
        }
    }
    
    // Check if nvcc is in PATH
    Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

/// Detect ROCm/HIP toolkit installation
fn detect_rocm() -> bool {
    // Check for ROCM_PATH environment variable
    if let Ok(rocm_path) = env::var("ROCM_PATH") {
        let hipcc_path = Path::new(&rocm_path).join("bin").join("hipcc");
        if hipcc_path.exists() {
            return true;
        }
    }
    
    // Check common ROCm installation paths
    let rocm_paths = [
        "/opt/rocm",
        "/usr/local/rocm",
        "/usr/rocm",
    ];
    
    for path in &rocm_paths {
        let hipcc_path = Path::new(path).join("bin").join("hipcc");
        if hipcc_path.exists() {
            env::set_var("ROCM_PATH", path);
            return true;
        }
    }
    
    // Check if hipcc is in PATH
    Command::new("hipcc")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

/// Detect Metal SDK (macOS/iOS only)
fn detect_metal() -> bool {
    // Check for Xcode command line tools
    Command::new("xcrun")
        .args(&["--find", "metal"])
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

/// Detect Vulkan SDK installation
fn detect_vulkan() -> bool {
    // Check for VULKAN_SDK environment variable
    if let Ok(vulkan_sdk) = env::var("VULKAN_SDK") {
        let glslang_path = Path::new(&vulkan_sdk).join("bin").join("glslangValidator");
        if glslang_path.exists() {
            return true;
        }
    }
    
    // Check if glslangValidator is in PATH
    let glslang_cmd = if cfg!(windows) { "glslangValidator.exe" } else { "glslangValidator" };
    Command::new(glslang_cmd)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

/// Compile CUDA kernels
fn compile_cuda_kernels(out_dir: &Path) -> bool {
    let cuda_dir = Path::new("gpu-kernels/cuda");
    if !cuda_dir.exists() {
        println!("cargo:warning=CUDA kernels directory not found");
        return false;
    }
    
    let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let nvcc = Path::new(&cuda_path).join("bin").join(if cfg!(windows) { "nvcc.exe" } else { "nvcc" });
    
    // Find all .cu files
    let cu_files = find_files(&cuda_dir, "cu");
    if cu_files.is_empty() {
        println!("cargo:warning=No CUDA .cu files found");
        return false;
    }
    
    let mut success = true;
    for cu_file in cu_files {
        let output_name = cu_file.file_stem().unwrap().to_str().unwrap();
        let output_path = out_dir.join(format!("{}.ptx", output_name));
        
        let mut cmd = Command::new(&nvcc);
        cmd.args(&[
            "--ptx",
            "--gpu-architecture=sm_60",  // Minimum Pascal architecture
            "--gpu-architecture=sm_70",  // Volta
            "--gpu-architecture=sm_75",  // Turing
            "--gpu-architecture=sm_80",  // Ampere
            "--gpu-architecture=sm_86",  // Ampere (RTX 30xx)
            "--gpu-architecture=sm_89",  // Ada Lovelace (RTX 40xx)
            "--gpu-architecture=sm_90",  // Hopper H100
            "-O3",
            "--use_fast_math",
            "--restrict",
            "--maxrregcount=64",
            "-Xptxas=-O3",
            "-Xptxas=-v",
            "--generate-line-info",
        ]);
        
        // Add CUDA includes
        cmd.arg(format!("-I{}/include", cuda_path));
        
        // Add architecture-specific optimizations
        let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
        match target_arch.as_str() {
            "x86_64" => {
                cmd.args(&["-m64", "--target-cpu-architecture=compute_60"]);
            },
            "aarch64" => {
                cmd.args(&["--target-cpu-architecture=compute_70"]);
            },
            _ => {},
        }
        
        cmd.args(&[
            cu_file.to_str().unwrap(),
            "-o",
            output_path.to_str().unwrap(),
        ]);
        
        println!("cargo:warning=Compiling CUDA kernel: {:?}", cu_file);
        let output = cmd.output().expect("Failed to execute nvcc");
        
        if !output.status.success() {
            println!("cargo:warning=CUDA compilation failed for {:?}", cu_file);
            println!("cargo:warning=stdout: {}", String::from_utf8_lossy(&output.stdout));
            println!("cargo:warning=stderr: {}", String::from_utf8_lossy(&output.stderr));
            success = false;
        } else {
            println!("cargo:warning=Successfully compiled CUDA kernel: {:?}", output_path);
        }
    }
    
    success
}

/// Compile HIP kernels
fn compile_hip_kernels(out_dir: &Path) -> bool {
    let hip_dir = Path::new("gpu-kernels/hip");
    if !hip_dir.exists() {
        println!("cargo:warning=HIP kernels directory not found");
        return false;
    }
    
    let rocm_path = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
    let hipcc = Path::new(&rocm_path).join("bin").join("hipcc");
    
    // Find all .hip.cpp files
    let hip_files = find_files_with_extension(&hip_dir, "hip.cpp");
    if hip_files.is_empty() {
        println!("cargo:warning=No HIP .hip.cpp files found");
        return false;
    }
    
    let mut success = true;
    for hip_file in hip_files {
        let output_name = hip_file.file_stem().unwrap().to_str().unwrap()
            .replace(".hip", "");
        let output_path = out_dir.join(format!("{}_hip.o", output_name));
        
        let mut cmd = Command::new(&hipcc);
        cmd.args(&[
            "-c",
            "-O3",
            "-ffast-math",
            "--amdgpu-target=gfx803",  // Polaris
            "--amdgpu-target=gfx900",  // Vega 10
            "--amdgpu-target=gfx906",  // Vega 20
            "--amdgpu-target=gfx908",  // MI100
            "--amdgpu-target=gfx90a",  // MI200 series
            "--amdgpu-target=gfx940",  // MI300 series
            "--amdgpu-target=gfx1030", // RDNA2
            "--amdgpu-target=gfx1100", // RDNA3
            "-fgpu-rdc",
            "-munsafe-fp-atomics",
        ]);
        
        // Add ROCm includes
        cmd.arg(format!("-I{}/include", rocm_path));
        
        cmd.args(&[
            hip_file.to_str().unwrap(),
            "-o",
            output_path.to_str().unwrap(),
        ]);
        
        println!("cargo:warning=Compiling HIP kernel: {:?}", hip_file);
        let output = cmd.output().expect("Failed to execute hipcc");
        
        if !output.status.success() {
            println!("cargo:warning=HIP compilation failed for {:?}", hip_file);
            println!("cargo:warning=stdout: {}", String::from_utf8_lossy(&output.stdout));
            println!("cargo:warning=stderr: {}", String::from_utf8_lossy(&output.stderr));
            success = false;
        } else {
            println!("cargo:warning=Successfully compiled HIP kernel: {:?}", output_path);
        }
    }
    
    success
}

/// Compile Metal kernels
fn compile_metal_kernels(out_dir: &Path) -> bool {
    let metal_dir = Path::new("gpu-kernels/metal");
    if !metal_dir.exists() {
        println!("cargo:warning=Metal kernels directory not found");
        return false;
    }
    
    // Find all .metal files
    let metal_files = find_files(&metal_dir, "metal");
    if metal_files.is_empty() {
        println!("cargo:warning=No Metal .metal files found");
        return false;
    }
    
    let mut success = true;
    for metal_file in metal_files {
        let output_name = metal_file.file_stem().unwrap().to_str().unwrap();
        let air_path = out_dir.join(format!("{}.air", output_name));
        let metallib_path = out_dir.join(format!("{}.metallib", output_name));
        
        // Compile to AIR (Apple Intermediate Representation)
        let mut cmd = Command::new("xcrun");
        cmd.args(&[
            "-sdk", "macosx",
            "metal",
            "-c",
            "-O3",
            "-ffast-math",
            "-std=metal3.0",  // Metal 3.0 for latest features
        ]);
        
        // Add Metal version flags for different Apple Silicon generations
        let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
        match target_arch.as_str() {
            "aarch64" => {
                // Apple Silicon optimization
                cmd.args(&[
                    "-mmacosx-version-min=12.0",
                    "-arch", "arm64",
                ]);
            },
            "x86_64" => {
                // Intel Mac optimization
                cmd.args(&[
                    "-mmacosx-version-min=10.15",
                    "-arch", "x86_64",
                ]);
            },
            _ => {},
        }
        
        cmd.args(&[
            metal_file.to_str().unwrap(),
            "-o",
            air_path.to_str().unwrap(),
        ]);
        
        println!("cargo:warning=Compiling Metal kernel to AIR: {:?}", metal_file);
        let output = cmd.output().expect("Failed to execute xcrun metal");
        
        if !output.status.success() {
            println!("cargo:warning=Metal AIR compilation failed for {:?}", metal_file);
            println!("cargo:warning=stdout: {}", String::from_utf8_lossy(&output.stdout));
            println!("cargo:warning=stderr: {}", String::from_utf8_lossy(&output.stderr));
            success = false;
            continue;
        }
        
        // Create metallib from AIR
        let mut cmd = Command::new("xcrun");
        cmd.args(&[
            "-sdk", "macosx",
            "metallib",
            air_path.to_str().unwrap(),
            "-o",
            metallib_path.to_str().unwrap(),
        ]);
        
        println!("cargo:warning=Creating Metal library: {:?}", metallib_path);
        let output = cmd.output().expect("Failed to execute xcrun metallib");
        
        if !output.status.success() {
            println!("cargo:warning=Metal library creation failed for {:?}", metal_file);
            println!("cargo:warning=stdout: {}", String::from_utf8_lossy(&output.stdout));
            println!("cargo:warning=stderr: {}", String::from_utf8_lossy(&output.stderr));
            success = false;
        } else {
            println!("cargo:warning=Successfully created Metal library: {:?}", metallib_path);
        }
    }
    
    success
}

/// Compile Vulkan shaders
fn compile_vulkan_shaders(out_dir: &Path) -> bool {
    let vulkan_dir = Path::new("gpu-kernels/vulkan");
    if !vulkan_dir.exists() {
        println!("cargo:warning=Vulkan shaders directory not found");
        return false;
    }
    
    // Find all .comp (compute shader) files
    let comp_files = find_files(&vulkan_dir, "comp");
    if comp_files.is_empty() {
        println!("cargo:warning=No Vulkan .comp files found");
        return false;
    }
    
    let glslang = if cfg!(windows) { "glslangValidator.exe" } else { "glslangValidator" };
    
    let mut success = true;
    for comp_file in comp_files {
        let output_name = comp_file.file_stem().unwrap().to_str().unwrap();
        let spv_path = out_dir.join(format!("{}.spv", output_name));
        
        let mut cmd = Command::new(glslang);
        cmd.args(&[
            "-V",  // Generate SPIR-V
            "-O",  // Optimize
            "--target-env", "vulkan1.3",  // Target Vulkan 1.3
            "-S", "comp",  // Compute shader stage
            "--enhanced-msgs",
            "-g",  // Generate debug info for development builds
        ]);
        
        // Add vendor-specific optimizations
        if env::var("CARGO_CFG_TARGET_OS").unwrap() == "linux" {
            cmd.args(&["--define-macro", "GPU_LINUX=1"]);
        }
        
        cmd.args(&[
            comp_file.to_str().unwrap(),
            "-o",
            spv_path.to_str().unwrap(),
        ]);
        
        println!("cargo:warning=Compiling Vulkan shader: {:?}", comp_file);
        let output = cmd.output().expect("Failed to execute glslangValidator");
        
        if !output.status.success() {
            println!("cargo:warning=Vulkan compilation failed for {:?}", comp_file);
            println!("cargo:warning=stdout: {}", String::from_utf8_lossy(&output.stdout));
            println!("cargo:warning=stderr: {}", String::from_utf8_lossy(&output.stderr));
            success = false;
        } else {
            println!("cargo:warning=Successfully compiled Vulkan shader: {:?}", spv_path);
        }
        
        // Also compile variants for different GPU vendors
        compile_vulkan_variants(&comp_file, out_dir);
    }
    
    success
}

/// Compile GPU vendor-specific Vulkan shader variants
fn compile_vulkan_variants(comp_file: &Path, out_dir: &Path) {
    let output_name = comp_file.file_stem().unwrap().to_str().unwrap();
    let vendors = [
        ("nvidia", "GPU_NVIDIA=1"),
        ("amd", "GPU_AMD=1"),
        ("intel", "GPU_INTEL=1"),
        ("mobile", "GPU_MOBILE=1"),
    ];
    
    let glslang = if cfg!(windows) { "glslangValidator.exe" } else { "glslangValidator" };
    
    for (vendor, define) in &vendors {
        let variant_path = out_dir.join(format!("{}_{}.spv", output_name, vendor));
        
        let mut cmd = Command::new(glslang);
        cmd.args(&[
            "-V",
            "-O",
            "--target-env", "vulkan1.3",
            "-S", "comp",
            "--define-macro", define,
            comp_file.to_str().unwrap(),
            "-o",
            variant_path.to_str().unwrap(),
        ]);
        
        if let Ok(output) = cmd.output() {
            if output.status.success() {
                println!("cargo:warning=Compiled Vulkan {} variant: {:?}", vendor, variant_path);
            }
        }
    }
}

/// Link CUDA runtime libraries
fn link_cuda_libraries() {
    let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    
    // Add CUDA library search paths
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-search=native={}/lib", cuda_path);
    
    if cfg!(windows) {
        println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path);
    }
    
    // Link CUDA runtime libraries
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cudnn");
    println!("cargo:rustc-link-lib=curand");
    println!("cargo:rustc-link-lib=cufft");
    println!("cargo:rustc-link-lib=cusparse");
    
    // Windows-specific libraries
    if cfg!(windows) {
        println!("cargo:rustc-link-lib=advapi32");
        println!("cargo:rustc-link-lib=ws2_32");
    }
    
    println!("cargo:rustc-cfg=feature=\"cuda\"");
}

/// Link HIP/ROCm libraries
fn link_hip_libraries() {
    let rocm_path = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
    
    // Add ROCm library search paths
    println!("cargo:rustc-link-search=native={}/lib", rocm_path);
    println!("cargo:rustc-link-search=native={}/lib64", rocm_path);
    
    // Link HIP and ROCm libraries
    println!("cargo:rustc-link-lib=hip_hcc");
    println!("cargo:rustc-link-lib=hipblas");
    println!("cargo:rustc-link-lib=hiprand");
    println!("cargo:rustc-link-lib=hipfft");
    println!("cargo:rustc-link-lib=hipsparse");
    println!("cargo:rustc-link-lib=rocblas");
    println!("cargo:rustc-link-lib=rocsparse");
    println!("cargo:rustc-link-lib=rocrand");
    println!("cargo:rustc-link-lib=rocfft");
    println!("cargo:rustc-link-lib=miopen");
    
    // AMD-specific runtime
    println!("cargo:rustc-link-lib=amdhip64");
    
    println!("cargo:rustc-cfg=feature=\"rocm\"");
}

/// Link Metal framework
fn link_metal_libraries() {
    // Link Metal framework (macOS/iOS)
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalKit");
    println!("cargo:rustc-link-lib=framework=Foundation");
    
    // Link Accelerate framework for optimized math
    println!("cargo:rustc-link-lib=framework=Accelerate");
    
    // Metal Performance Shaders
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    
    println!("cargo:rustc-cfg=feature=\"metal\"");
}

/// Link Vulkan libraries
fn link_vulkan_libraries() {
    // Platform-specific Vulkan linking
    if cfg!(windows) {
        println!("cargo:rustc-link-lib=vulkan-1");
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=vulkan");
        // MoltenVK for Vulkan on macOS
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=QuartzCore");
        println!("cargo:rustc-link-lib=framework=IOSurface");
    } else {
        println!("cargo:rustc-link-lib=vulkan");
    }
    
    println!("cargo:rustc-cfg=feature=\"vulkan\"");
}

/// Set compiler optimization flags
fn set_optimization_flags(target_arch: &str) {
    // Set architecture-specific optimization flags
    match target_arch {
        "x86_64" => {
            println!("cargo:rustc-env=TARGET_CPU=native");
            println!("cargo:rustc-link-arg=-march=native");
            println!("cargo:rustc-link-arg=-mtune=native");
        },
        "aarch64" => {
            println!("cargo:rustc-env=TARGET_CPU=native");
            println!("cargo:rustc-link-arg=-mcpu=native");
        },
        _ => {},
    }
    
    // Enable fast math and vectorization
    println!("cargo:rustc-link-arg=-ffast-math");
    println!("cargo:rustc-link-arg=-funroll-loops");
    println!("cargo:rustc-link-arg=-fvectorize");
    
    // Link-time optimization for release builds
    if env::var("PROFILE").unwrap() == "release" {
        println!("cargo:rustc-link-arg=-flto");
        println!("cargo:rustc-link-arg=-O3");
    }
}

/// Generate GPU feature configuration
fn generate_gpu_config(features: &[&str]) {
    let config_path = Path::new(&env::var("OUT_DIR").unwrap()).join("gpu_config.rs");
    let mut config_content = String::new();
    
    config_content.push_str("// Auto-generated GPU configuration\n");
    config_content.push_str("// DO NOT EDIT - Generated by build.rs\n\n");
    
    config_content.push_str("pub struct GpuConfig {\n");
    config_content.push_str("    pub cuda_available: bool,\n");
    config_content.push_str("    pub rocm_available: bool,\n");
    config_content.push_str("    pub metal_available: bool,\n");
    config_content.push_str("    pub vulkan_available: bool,\n");
    config_content.push_str("}\n\n");
    
    config_content.push_str("impl Default for GpuConfig {\n");
    config_content.push_str("    fn default() -> Self {\n");
    config_content.push_str("        Self {\n");
    config_content.push_str(&format!("            cuda_available: {},\n", features.contains(&"cuda")));
    config_content.push_str(&format!("            rocm_available: {},\n", features.contains(&"rocm")));
    config_content.push_str(&format!("            metal_available: {},\n", features.contains(&"metal")));
    config_content.push_str(&format!("            vulkan_available: {},\n", features.contains(&"vulkan")));
    config_content.push_str("        }\n");
    config_content.push_str("    }\n");
    config_content.push_str("}\n\n");
    
    // Generate feature constants
    for feature in features {
        config_content.push_str(&format!("pub const {}_ENABLED: bool = true;\n", feature.to_uppercase()));
    }
    
    // Generate disabled features
    let all_features = ["cuda", "rocm", "metal", "vulkan"];
    for feature in &all_features {
        if !features.contains(feature) {
            config_content.push_str(&format!("pub const {}_ENABLED: bool = false;\n", feature.to_uppercase()));
        }
    }
    
    fs::write(config_path, config_content).expect("Failed to write GPU config");
}

/// Generate runtime feature detection
fn generate_feature_detection(features: &[&str]) {
    let detection_path = Path::new(&env::var("OUT_DIR").unwrap()).join("gpu_detection.rs");
    let mut detection_content = String::new();
    
    detection_content.push_str("// Auto-generated GPU runtime detection\n");
    detection_content.push_str("// DO NOT EDIT - Generated by build.rs\n\n");
    
    detection_content.push_str("use std::sync::Once;\n");
    detection_content.push_str("use std::sync::atomic::{AtomicBool, Ordering};\n\n");
    
    // Generate static detection variables
    for feature in features {
        detection_content.push_str(&format!("static {}_DETECTED: AtomicBool = AtomicBool::new(false);\n", feature.to_uppercase()));
        detection_content.push_str(&format!("static {}_INIT: Once = Once::new();\n", feature.to_uppercase()));
    }
    
    detection_content.push_str("\npub fn initialize_gpu_detection() {\n");
    for feature in features {
        detection_content.push_str(&format!("    detect_{}();\n", feature));
    }
    detection_content.push_str("}\n\n");
    
    // Generate detection functions
    for feature in features {
        match *feature {
            "cuda" => {
                detection_content.push_str(&format!("
pub fn detect_cuda() -> bool {{
    {}_INIT.call_once(|| {{
        let detected = cuda_runtime_available();
        {}_DETECTED.store(detected, Ordering::Relaxed);
    }});
    {}_DETECTED.load(Ordering::Relaxed)
}}

fn cuda_runtime_available() -> bool {{
    #[cfg(feature = \"cuda\")]
    {{
        // Try to initialize CUDA runtime
        use std::ptr;
        extern \"C\" {{
            fn cudaRuntimeGetVersion(version: *mut i32) -> i32;
        }}
        
        let mut version: i32 = 0;
        let result = unsafe {{ cudaRuntimeGetVersion(&mut version) }};
        result == 0 && version > 0
    }}
    
    #[cfg(not(feature = \"cuda\"))]
    {{
        false
    }}
}}
", feature.to_uppercase(), feature.to_uppercase(), feature.to_uppercase()));
            },
            "rocm" => {
                detection_content.push_str(&format!("
pub fn detect_rocm() -> bool {{
    {}_INIT.call_once(|| {{
        let detected = rocm_runtime_available();
        {}_DETECTED.store(detected, Ordering::Relaxed);
    }});
    {}_DETECTED.load(Ordering::Relaxed)
}}

fn rocm_runtime_available() -> bool {{
    #[cfg(feature = \"rocm\")]
    {{
        // Try to initialize HIP runtime
        extern \"C\" {{
            fn hipRuntimeGetVersion(version: *mut i32) -> i32;
        }}
        
        let mut version: i32 = 0;
        let result = unsafe {{ hipRuntimeGetVersion(&mut version) }};
        result == 0 && version > 0
    }}
    
    #[cfg(not(feature = \"rocm\"))]
    {{
        false
    }}
}}
", feature.to_uppercase(), feature.to_uppercase(), feature.to_uppercase()));
            },
            "metal" => {
                detection_content.push_str(&format!("
pub fn detect_metal() -> bool {{
    {}_INIT.call_once(|| {{
        let detected = metal_runtime_available();
        {}_DETECTED.store(detected, Ordering::Relaxed);
    }});
    {}_DETECTED.load(Ordering::Relaxed)
}}

fn metal_runtime_available() -> bool {{
    #[cfg(all(feature = \"metal\", any(target_os = \"macos\", target_os = \"ios\")))]
    {{
        // Check if Metal is available on the system
        true // Metal is always available on macOS 10.11+ and iOS 8+
    }}
    
    #[cfg(not(all(feature = \"metal\", any(target_os = \"macos\", target_os = \"ios\"))))]
    {{
        false
    }}
}}
", feature.to_uppercase(), feature.to_uppercase()));
            },
            "vulkan" => {
                detection_content.push_str(&format!("
pub fn detect_vulkan() -> bool {{
    {}_INIT.call_once(|| {{
        let detected = vulkan_runtime_available();
        {}_DETECTED.store(detected, Ordering::Relaxed);
    }});
    {}_DETECTED.load(Ordering::Relaxed)
}}

fn vulkan_runtime_available() -> bool {{
    #[cfg(feature = \"vulkan\")]
    {{
        // Try to enumerate Vulkan instances
        extern \"C\" {{
            fn vkEnumerateInstanceVersion(version: *mut u32) -> i32;
        }}
        
        let mut version: u32 = 0;
        let result = unsafe {{ vkEnumerateInstanceVersion(&mut version) }};
        result == 0 && version > 0
    }}
    
    #[cfg(not(feature = \"vulkan\"))]
    {{
        false
    }}
}}
", feature.to_uppercase(), feature.to_uppercase()));
            },
            _ => {},
        }
    }
    
    fs::write(detection_path, detection_content).expect("Failed to write GPU detection");
}

/// Helper function to find files with specific extension
fn find_files(dir: &Path, extension: &str) -> Vec<PathBuf> {
    let mut files = Vec::new();
    
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some(extension) {
                files.push(path);
            }
        }
    }
    
    files
}

/// Helper function to find files with compound extensions (e.g., .hip.cpp)
fn find_files_with_extension(dir: &Path, extension: &str) -> Vec<PathBuf> {
    let mut files = Vec::new();
    
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(file_name) = path.file_name().and_then(|s| s.to_str()) {
                if file_name.ends_with(extension) {
                    files.push(path);
                }
            }
        }
    }
    
    files
}