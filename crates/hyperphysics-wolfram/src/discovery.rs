//! Wolfram Installation Discovery
//!
//! Uses wolfram-app-discovery to find local Wolfram installations,
//! with special handling for WolframScript.app on macOS.

use crate::types::{WolframError, WolframInstallation, WolframResult};
use std::path::PathBuf;
use tracing::{debug, info, warn};
use wolfram_app_discovery::WolframApp;

/// Discover all Wolfram installations on the system
pub fn discover_installations() -> WolframResult<Vec<WolframInstallation>> {
    let mut installations = Vec::new();

    // Try to find the default Wolfram app via wolfram-app-discovery
    match WolframApp::try_default() {
        Ok(app) => {
            if let Ok(installation) = wolfram_app_to_installation(&app) {
                installations.push(installation);
            }
        }
        Err(e) => {
            debug!("No default Wolfram app found via discovery: {}", e);
        }
    }

    // Check common paths manually (especially for WolframScript.app)
    let additional_paths = get_common_wolfram_paths();
    for path in additional_paths {
        if path.exists()
            && !installations
                .iter()
                .any(|i| i.installation_directory == path.to_string_lossy())
        {
            if let Ok(installation) = probe_installation(&path) {
                installations.push(installation);
            }
        }
    }

    // Try to find wolframscript in PATH
    if let Ok(ws_path) = which::which("wolframscript") {
        let ws_str = ws_path.to_string_lossy().to_string();
        if !installations
            .iter()
            .any(|i| i.wolfram_script_path == ws_str)
        {
            info!("Found wolframscript in PATH: {:?}", ws_path);
            installations.push(WolframInstallation {
                installation_directory: ws_path
                    .parent()
                    .and_then(|p| p.parent())
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default(),
                wolfram_script_path: ws_str,
                kernel_path: String::new(),
                product_name: "WolframScript".to_string(),
                version: get_wolframscript_version(&ws_path).unwrap_or_default(),
                is_valid: true,
            });
        }
    }

    if installations.is_empty() {
        warn!("No Wolfram installations found on system");
    } else {
        info!("Found {} Wolfram installation(s)", installations.len());
    }

    Ok(installations)
}

/// Get the default/best Wolfram installation
///
/// Preference order:
/// 1. WolframScript.app (for Pro subscription with Code Assistant)
/// 2. Wolfram Desktop
/// 3. Mathematica
/// 4. Any available installation
pub fn get_default_installation() -> WolframResult<WolframInstallation> {
    let installations = discover_installations()?;

    // Prefer WolframScript.app for Pro features
    if let Some(ws) = installations
        .iter()
        .find(|i| i.product_name.contains("WolframScript"))
    {
        return Ok(ws.clone());
    }

    // Then Wolfram Desktop
    if let Some(desktop) = installations
        .iter()
        .find(|i| i.product_name.contains("Desktop"))
    {
        return Ok(desktop.clone());
    }

    // Then Mathematica
    if let Some(math) = installations
        .iter()
        .find(|i| i.product_name.contains("Mathematica"))
    {
        return Ok(math.clone());
    }

    // Any installation
    installations.into_iter().next().ok_or(WolframError::NoInstallation)
}

/// Convert WolframApp to our installation struct
fn wolfram_app_to_installation(app: &WolframApp) -> WolframResult<WolframInstallation> {
    let install_dir = app.installation_directory();

    let wolfram_script =
        find_executable_in_installation(&install_dir, &["wolframscript", "WolframScript"]);
    let kernel =
        find_executable_in_installation(&install_dir, &["WolframKernel", "wolfram", "MathKernel"]);

    let version =
        get_version_from_installation(&install_dir).unwrap_or_else(|| "unknown".to_string());
    let product_name = detect_product_name(&install_dir);

    Ok(WolframInstallation {
        installation_directory: install_dir.to_string_lossy().to_string(),
        wolfram_script_path: wolfram_script
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default(),
        kernel_path: kernel
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default(),
        product_name,
        version,
        is_valid: true,
    })
}

/// Probe a path to see if it's a valid Wolfram installation
fn probe_installation(path: &PathBuf) -> WolframResult<WolframInstallation> {
    let wolfram_script = find_executable_in_installation(path, &["wolframscript", "WolframScript"]);
    let kernel = find_executable_in_installation(path, &["WolframKernel", "wolfram", "MathKernel"]);

    if wolfram_script.is_none() && kernel.is_none() {
        return Err(WolframError::NoInstallation);
    }

    let version = get_version_from_installation(path).unwrap_or_else(|| "unknown".to_string());
    let product_name = detect_product_name(path);

    Ok(WolframInstallation {
        installation_directory: path.to_string_lossy().to_string(),
        wolfram_script_path: wolfram_script
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default(),
        kernel_path: kernel
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default(),
        product_name,
        version,
        is_valid: true,
    })
}

/// Get common Wolfram installation paths by platform
fn get_common_wolfram_paths() -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = Vec::new();

    #[cfg(target_os = "macos")]
    {
        paths.extend([
            // WolframScript.app - preferred for Pro subscription
            PathBuf::from("/Applications/WolframScript.app/Contents"),
            PathBuf::from("/Applications/Wolfram Desktop.app/Contents"),
            PathBuf::from("/Applications/Mathematica.app/Contents"),
            PathBuf::from("/Applications/Wolfram.app/Contents"),
            // User-specific locations
            dirs::home_dir()
                .map(|h| h.join("Applications/WolframScript.app/Contents"))
                .unwrap_or_default(),
            dirs::home_dir()
                .map(|h| h.join("Applications/Mathematica.app/Contents"))
                .unwrap_or_default(),
        ]);
    }

    #[cfg(target_os = "linux")]
    {
        paths.extend([
            PathBuf::from("/usr/local/Wolfram/WolframEngine"),
            PathBuf::from("/usr/local/Wolfram/Mathematica"),
            PathBuf::from("/opt/Wolfram"),
            PathBuf::from("/opt/WolframEngine"),
        ]);
    }

    #[cfg(target_os = "windows")]
    {
        if let Some(program_files) = std::env::var_os("ProgramFiles") {
            let pf = PathBuf::from(program_files);
            paths.push(pf.join("Wolfram Research").join("WolframScript"));
            paths.push(pf.join("Wolfram Research").join("Wolfram Desktop"));
            paths.push(pf.join("Wolfram Research").join("Mathematica"));
        }
    }

    // Filter out empty paths
    paths.into_iter().filter(|p| !p.as_os_str().is_empty()).collect()
}

/// Find an executable in a Wolfram installation
fn find_executable_in_installation(
    install_dir: &std::path::Path,
    names: &[&str],
) -> Option<PathBuf> {
    let subdirs = [
        "MacOS",
        "Executables",
        "SystemFiles/Kernel/Binaries",
        "bin",
        "",
    ];

    for subdir in subdirs {
        let dir = if subdir.is_empty() {
            install_dir.to_path_buf()
        } else {
            install_dir.join(subdir)
        };

        for name in names {
            let path = dir.join(name);
            if path.exists() && path.is_file() {
                debug!("Found executable: {:?}", path);
                return Some(path);
            }

            // Check with .exe on Windows
            #[cfg(target_os = "windows")]
            {
                let path_exe = dir.join(format!("{}.exe", name));
                if path_exe.exists() && path_exe.is_file() {
                    return Some(path_exe);
                }
            }
        }
    }

    None
}

/// Get version from installation
fn get_version_from_installation(install_dir: &std::path::Path) -> Option<String> {
    // Try to read .VersionID file
    let version_file = install_dir.join(".VersionID");
    if version_file.exists() {
        if let Ok(content) = std::fs::read_to_string(&version_file) {
            return Some(content.trim().to_string());
        }
    }

    // Try Resources/VersionID
    let resources_version = install_dir.join("Resources").join(".VersionID");
    if resources_version.exists() {
        if let Ok(content) = std::fs::read_to_string(&resources_version) {
            return Some(content.trim().to_string());
        }
    }

    // Try Info.plist on macOS
    #[cfg(target_os = "macos")]
    {
        let info_plist = install_dir.join("Info.plist");
        if info_plist.exists() {
            if let Ok(content) = std::fs::read_to_string(&info_plist) {
                // Simple regex to extract version
                if let Some(caps) = regex::Regex::new(r"<key>CFBundleShortVersionString</key>\s*<string>([^<]+)</string>")
                    .ok()
                    .and_then(|re| re.captures(&content))
                {
                    return caps.get(1).map(|m| m.as_str().to_string());
                }
            }
        }
    }

    None
}

/// Get WolframScript version by executing it
fn get_wolframscript_version(path: &PathBuf) -> Option<String> {
    let output = std::process::Command::new(path)
        .arg("--version")
        .output()
        .ok()?;

    if output.status.success() {
        let version = String::from_utf8_lossy(&output.stdout);
        // Parse version from output like "WolframScript 1.13.0"
        if let Some(v) = version.split_whitespace().last() {
            return Some(v.to_string());
        }
    }

    None
}

/// Detect product name from installation path
fn detect_product_name(install_dir: &std::path::Path) -> String {
    let path_str = install_dir.to_string_lossy().to_lowercase();

    if path_str.contains("wolframscript") {
        "WolframScript.app".to_string()
    } else if path_str.contains("desktop") {
        "Wolfram Desktop".to_string()
    } else if path_str.contains("mathematica") {
        "Mathematica".to_string()
    } else if path_str.contains("engine") {
        "Wolfram Engine".to_string()
    } else {
        "Wolfram".to_string()
    }
}

/// Check if Wolfram is available on the system
pub fn is_wolfram_available() -> bool {
    get_default_installation().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discover_installations() {
        let result = discover_installations();
        assert!(result.is_ok());
    }

    #[test]
    fn test_common_paths_not_empty() {
        let paths = get_common_wolfram_paths();
        // Should have at least some paths on any platform
        assert!(!paths.is_empty() || cfg!(not(any(
            target_os = "macos",
            target_os = "linux",
            target_os = "windows"
        ))));
    }
}
