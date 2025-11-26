extern crate napi_build;

fn main() {
    // Configure NAPI build
    napi_build::setup();

    // Print build information
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/");
}
