use std::env;
use std::path::PathBuf;

fn main() {
    // Re-run build if wrapper changes
    println!("cargo:rerun-if-changed=src/jolt_wrapper.cpp");
    println!("cargo:rerun-if-changed=include/jolt_wrapper.h");

    // Compile C++ wrapper
    // Note: This assumes JoltPhysics source is available or we are just compiling the wrapper
    // In a real scenario, we would also compile JoltPhysics itself here or link to it.
    // For now, we compile the wrapper which would include Jolt headers.
    // We will add a placeholder for Jolt include path.

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .file("src/jolt_wrapper.cpp")
        .include("include");

    // Use vendored JoltPhysics from crates/vendor/physics/JoltPhysics
    build.include("../../vendor/physics/JoltPhysics/Jolt");

    build.compile("jolt_wrapper");

    // Generate bindings
    let bindings = bindgen::Builder::default()
        .header("include/jolt_wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
