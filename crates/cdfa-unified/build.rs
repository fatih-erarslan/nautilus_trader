use std::env;
use std::process::Command;

fn main() {
    // Set build date
    let build_date = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string();
    println!("cargo:rustc-env=BUILD_DATE={}", build_date);
    
    // Set git hash
    let git_hash = Command::new("git")
        .args(&["rev-parse", "--short", "HEAD"])
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=GIT_HASH={}", git_hash);
    
    // Set rust version
    let rust_version = env::var("RUSTC_VERSION").unwrap_or_else(|_| {
        Command::new("rustc")
            .args(&["--version"])
            .output()
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
            .unwrap_or_else(|_| "unknown".to_string())
    });
    println!("cargo:rustc-env=RUST_VERSION={}", rust_version);
    
    // Rerun if git changes
    println!("cargo:rerun-if-changed=.git/HEAD");
}