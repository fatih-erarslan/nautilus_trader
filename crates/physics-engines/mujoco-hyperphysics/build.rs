use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let mujoco_root = PathBuf::from("../../vendor/physics/mujoco");
    let mujoco_include = mujoco_root.join("include");
    let mujoco_src = mujoco_root.join("src");

    // Check if MuJoCo source exists
    if !mujoco_include.exists() {
        eprintln!("Warning: MuJoCo source not found at {:?}", mujoco_include);
        eprintln!("Building with mock implementation.");

        // Generate empty bindings for mock mode
        generate_mock_bindings();
        return;
    }

    println!("cargo:rerun-if-changed={}", mujoco_include.display());

    // For now, we link to pre-built MuJoCo library
    // Building from source requires CMake and many dependencies
    // Users should install MuJoCo and set MUJOCO_PATH

    if let Ok(mujoco_path) = env::var("MUJOCO_PATH") {
        let lib_path = PathBuf::from(&mujoco_path).join("lib");
        println!("cargo:rustc-link-search=native={}", lib_path.display());
        println!("cargo:rustc-link-lib=dylib=mujoco");
    } else {
        // Try common installation paths
        let common_paths = [
            "/usr/local/lib",
            "/opt/mujoco/lib",
            "~/.mujoco/mujoco-3.2.6/lib",
        ];

        for path in &common_paths {
            let expanded = shellexpand::tilde(path);
            let lib_path = PathBuf::from(expanded.as_ref());
            if lib_path.exists() {
                println!("cargo:rustc-link-search=native={}", lib_path.display());
                println!("cargo:rustc-link-lib=dylib=mujoco");
                break;
            }
        }
    }

    // Generate bindings
    generate_bindings(&mujoco_include);
}

fn generate_bindings(include_path: &PathBuf) {
    let bindings = bindgen::Builder::default()
        .header(include_path.join("mujoco/mujoco.h").to_str().unwrap())
        .clang_arg(format!("-I{}", include_path.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Include main types
        .allowlist_type("mjModel")
        .allowlist_type("mjData")
        .allowlist_type("mjOption")
        .allowlist_type("mjVisual")
        .allowlist_type("mjStatistic")
        .allowlist_type("mjContact")
        .allowlist_type("mjVFS")
        .allowlist_type("mjtNum")
        // Include main functions
        .allowlist_function("mj_.*")
        .allowlist_function("mju_.*")
        // Include constants
        .allowlist_var("mjVERSION_HEADER")
        .allowlist_var("mjNGEOMTYPES")
        // Derive traits
        .derive_default(true)
        .derive_debug(true)
        .generate()
        .expect("Unable to generate MuJoCo bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write MuJoCo bindings!");
}

fn generate_mock_bindings() {
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mock_bindings = r#"
// Mock MuJoCo bindings when library not available

pub type mjtNum = f64;

#[repr(C)]
#[derive(Debug, Default)]
pub struct mjModel {
    pub nq: i32,      // number of generalized coordinates
    pub nv: i32,      // number of degrees of freedom
    pub nu: i32,      // number of actuators
    pub nbody: i32,   // number of bodies
    pub njnt: i32,    // number of joints
    pub ngeom: i32,   // number of geoms
}

#[repr(C)]
#[derive(Debug, Default)]
pub struct mjData {
    pub time: mjtNum,
    pub qpos: *mut mjtNum,
    pub qvel: *mut mjtNum,
    pub qacc: *mut mjtNum,
    pub ctrl: *mut mjtNum,
}

#[repr(C)]
#[derive(Debug, Default)]
pub struct mjOption {
    pub timestep: mjtNum,
    pub gravity: [mjtNum; 3],
}

#[repr(C)]
#[derive(Debug, Default)]
pub struct mjVFS {
    pub nfile: i32,
}

extern "C" {
    pub fn mj_loadXML(filename: *const i8, vfs: *const mjVFS, error: *mut i8, error_sz: i32) -> *mut mjModel;
    pub fn mj_deleteModel(m: *mut mjModel);
    pub fn mj_makeData(m: *const mjModel) -> *mut mjData;
    pub fn mj_deleteData(d: *mut mjData);
    pub fn mj_step(m: *const mjModel, d: *mut mjData);
    pub fn mj_step1(m: *const mjModel, d: *mut mjData);
    pub fn mj_step2(m: *const mjModel, d: *mut mjData);
    pub fn mj_forward(m: *const mjModel, d: *mut mjData);
    pub fn mj_resetData(m: *const mjModel, d: *mut mjData);
    pub fn mj_version() -> i32;
}
"#;

    std::fs::write(out_path.join("bindings.rs"), mock_bindings)
        .expect("Couldn't write mock MuJoCo bindings!");
}
