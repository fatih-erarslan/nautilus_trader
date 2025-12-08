use std::env;
use std::path::PathBuf;

fn main() {
    // Only build tree models if feature is enabled
    if cfg!(feature = "tree-models") {
        println!("cargo:rerun-if-changed=build.rs");
        
        // Link to XGBoost and LightGBM libraries
        // These would need to be installed on the system
        println!("cargo:rustc-link-lib=dylib=xgboost");
        println!("cargo:rustc-link-lib=dylib=lightgbm");
        
        // Add library search paths
        if let Ok(xgboost_lib) = env::var("XGBOOST_LIB_DIR") {
            println!("cargo:rustc-link-search=native={}", xgboost_lib);
        }
        
        if let Ok(lgbm_lib) = env::var("LIGHTGBM_LIB_DIR") {
            println!("cargo:rustc-link-search=native={}", lgbm_lib);
        }
        
        // Default paths
        println!("cargo:rustc-link-search=native=/usr/local/lib");
        println!("cargo:rustc-link-search=native=/usr/lib");
    }
}