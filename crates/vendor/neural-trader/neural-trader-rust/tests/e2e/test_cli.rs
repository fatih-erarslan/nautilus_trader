//! End-to-end tests for CLI commands

use std::process::Command;

#[test]
#[ignore] // Run with: cargo test --ignored
fn test_cli_help_command() {
    let output = Command::new("cargo")
        .args(&["run", "--bin", "neural-trader-cli", "--", "--help"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success(), "CLI help should succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Neural Trader"), "Should show app name");
    assert!(stdout.contains("USAGE"), "Should show usage");
}

#[test]
#[ignore]
fn test_cli_version_command() {
    let output = Command::new("cargo")
        .args(&["run", "--bin", "neural-trader-cli", "--", "--version"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success(), "CLI version should succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("0.1"), "Should show version number");
}

#[test]
#[ignore]
fn test_cli_init_command() {
    let output = Command::new("cargo")
        .args(&["run", "--bin", "neural-trader-cli", "--", "init", "--help"])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("Init command output: {}", stdout);

    // Verify init command exists and has documentation
    assert!(
        output.status.success() || stdout.contains("init"),
        "Init command should be available"
    );
}

#[test]
#[ignore]
fn test_cli_status_command() {
    let output = Command::new("cargo")
        .args(&["run", "--bin", "neural-trader-cli", "--", "status", "--help"])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("Status command output: {}", stdout);

    assert!(
        output.status.success() || stdout.contains("status"),
        "Status command should be available"
    );
}
