use std::env;
use std::path::PathBuf;

fn main() {
    // Re-run build if wrapper changes
    println!("cargo:rerun-if-changed=src/jolt_wrapper.cpp");
    println!("cargo:rerun-if-changed=include/jolt_wrapper.h");

    let jolt_root = PathBuf::from("../../vendor/physics/JoltPhysics");
    let jolt_src = jolt_root.join("Jolt");

    // Check if JoltPhysics source exists
    if !jolt_src.exists() {
        eprintln!("Warning: JoltPhysics source not found at {:?}", jolt_src);
        eprintln!("Building with mock implementation only.");

        // Build mock wrapper only
        cc::Build::new()
            .cpp(true)
            .std("c++17")
            .file("src/jolt_wrapper.cpp")
            .include("include")
            .define("JOLT_MOCK_IMPLEMENTATION", None)
            .compile("jolt_wrapper");

        generate_bindings();
        return;
    }

    println!("cargo:rerun-if-changed={}", jolt_src.display());

    // Collect JoltPhysics core source files
    let jolt_sources = collect_jolt_sources(&jolt_src);

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .warnings(false)  // JoltPhysics has some warnings we can ignore
        .include(&jolt_src.parent().unwrap())  // Include JoltPhysics root
        .include("include")
        .define("JPH_PROFILE_ENABLED", None)
        .define("JPH_DEBUG_RENDERER", None);

    // Platform-specific settings
    #[cfg(target_os = "linux")]
    {
        build.define("JPH_USE_SSE4_1", None);
        build.define("JPH_USE_SSE4_2", None);
        build.define("JPH_USE_AVX", None);
        build.define("JPH_USE_AVX2", None);
        build.define("JPH_USE_F16C", None);
        build.define("JPH_USE_LZCNT", None);
        build.define("JPH_USE_TZCNT", None);
        build.define("JPH_USE_FMADD", None);
        build.flag("-msse4.1");
        build.flag("-msse4.2");
        build.flag("-mavx");
        build.flag("-mavx2");
        build.flag("-mfma");
        build.flag("-mlzcnt");
        build.flag("-mbmi");
        build.flag("-mf16c");
    }

    #[cfg(target_os = "macos")]
    {
        // Apple Silicon uses NEON
        #[cfg(target_arch = "aarch64")]
        {
            build.define("JPH_USE_NEON", None);
        }
        #[cfg(target_arch = "x86_64")]
        {
            build.define("JPH_USE_SSE4_1", None);
            build.define("JPH_USE_SSE4_2", None);
        }
    }

    // Add JoltPhysics source files
    for source in &jolt_sources {
        build.file(source);
    }

    // Add our wrapper
    build.file("src/jolt_wrapper.cpp");

    // Compile everything
    build.compile("jolt_physics");

    // Link pthread on Unix
    #[cfg(unix)]
    println!("cargo:rustc-link-lib=pthread");

    generate_bindings();
}

fn collect_jolt_sources(jolt_src: &PathBuf) -> Vec<PathBuf> {
    let mut sources = Vec::new();

    // Core modules to compile
    let modules = [
        "Core",
        "Math",
        "Geometry",
        "AABBTree",
        "ObjectStream",
        "Physics/Body",
        "Physics/Character",
        "Physics/Collision",
        "Physics/Collision/Shape",
        "Physics/Collision/BroadPhase",
        "Physics/Collision/NarrowPhaseQuery",
        "Physics/Constraints",
        "Physics/Ragdoll",
        "Physics/SoftBody",
        "Physics/Vehicle",
        "Physics",
    ];

    for module in modules {
        let module_path = jolt_src.join(module);
        if module_path.exists() {
            collect_cpp_files(&module_path, &mut sources);
        }
    }

    sources
}

fn collect_cpp_files(dir: &PathBuf, sources: &mut Vec<PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.extension().map_or(false, |e| e == "cpp") {
                sources.push(path);
            } else if path.is_dir() {
                // Don't recurse into subdirectories for top-level modules
                // as they're listed explicitly
            }
        }
    }
}

fn generate_bindings() {
    let bindings = bindgen::Builder::default()
        .header("include/jolt_wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_function("jolt_.*")
        .allowlist_type("Jolt.*")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
