//! Path Traversal Protection Module
//!
//! Provides comprehensive protection against path traversal attacks:
//! - Directory traversal detection (.., ~, absolute paths)
//! - Path canonicalization and validation
//! - Filename sanitization
//! - Safe path operations within allowed directories

use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};

/// Validate that a path is safe and within the allowed base directory
///
/// # Security Checks
/// - Rejects path traversal sequences (`..`, `~`)
/// - Rejects absolute paths (Unix `/`, Windows `C:\`)
/// - Rejects symbolic links that escape the base directory
/// - Validates the canonical path stays within base directory
///
/// # Arguments
/// * `path` - The path to validate (should be relative)
/// * `base_dir` - The base directory that paths must stay within
///
/// # Returns
/// * `Ok(PathBuf)` - Canonical path if validation passes
/// * `Err` - If path traversal or other security issue detected
///
/// # Example
/// ```
/// use std::path::Path;
/// use neural_trader_backend::security::path_validation::validate_safe_path;
///
/// let base = Path::new("/var/data");
/// assert!(validate_safe_path("files/report.txt", base).is_ok());
/// assert!(validate_safe_path("../etc/passwd", base).is_err());
/// ```
pub fn validate_safe_path(path: &str, base_dir: &Path) -> Result<PathBuf> {
    // Reject empty paths
    if path.is_empty() {
        return Err(anyhow!("Path cannot be empty"));
    }

    // Reject path traversal attempts
    if path.contains("..") {
        return Err(anyhow!(
            "Path traversal detected: path contains '..' sequence: {}",
            path
        ));
    }

    // Reject home directory expansion
    if path.contains('~') {
        return Err(anyhow!(
            "Path traversal detected: path contains '~': {}",
            path
        ));
    }

    // Reject absolute paths (Unix)
    #[cfg(unix)]
    if path.starts_with('/') {
        return Err(anyhow!(
            "Absolute paths not allowed: {}",
            path
        ));
    }

    // Reject absolute paths (Windows)
    #[cfg(windows)]
    {
        // Check for UNC paths (\\server\share)
        if path.starts_with("\\\\") {
            return Err(anyhow!(
                "UNC paths not allowed: {}",
                path
            ));
        }

        // Check for drive letters (C:\)
        if path.len() >= 2 && path.chars().nth(1) == Some(':') {
            return Err(anyhow!(
                "Absolute paths with drive letters not allowed: {}",
                path
            ));
        }
    }

    // Reject null bytes (can bypass some checks)
    if path.contains('\0') {
        return Err(anyhow!(
            "Null bytes not allowed in path: {}",
            path
        ));
    }

    // Create sanitized path
    let sanitized = PathBuf::from(path);

    // Ensure base directory exists and is valid
    if !base_dir.exists() {
        return Err(anyhow!(
            "Base directory does not exist: {}",
            base_dir.display()
        ));
    }

    let base_canonical = base_dir.canonicalize()
        .map_err(|e| anyhow!("Failed to canonicalize base directory: {}", e))?;

    // Join and canonicalize the full path
    let full_path = base_canonical.join(&sanitized);

    // For paths that don't exist yet, we need to handle canonicalization differently
    let canonical = if full_path.exists() {
        full_path.canonicalize()
            .map_err(|e| anyhow!("Invalid path '{}': {}", path, e))?
    } else {
        // For non-existent paths, canonicalize parent and append filename
        let parent = full_path.parent().ok_or_else(|| {
            anyhow!("Path has no parent: {}", full_path.display())
        })?;

        if !parent.exists() {
            return Err(anyhow!(
                "Parent directory does not exist: {}",
                parent.display()
            ));
        }

        let parent_canonical = parent.canonicalize()
            .map_err(|e| anyhow!("Failed to canonicalize parent: {}", e))?;

        let filename = full_path.file_name().ok_or_else(|| {
            anyhow!("Path has no filename: {}", full_path.display())
        })?;

        parent_canonical.join(filename)
    };

    // Verify the canonical path is still within base directory
    if !canonical.starts_with(&base_canonical) {
        return Err(anyhow!(
            "Path '{}' escapes base directory '{}' (canonical: '{}')",
            path,
            base_dir.display(),
            canonical.display()
        ));
    }

    Ok(canonical)
}

/// Validate a filename (not a full path) for security
///
/// # Security Checks
/// - Rejects path separators (/, \)
/// - Rejects special characters that could cause issues
/// - Rejects empty filenames and special names (., ..)
/// - Rejects null bytes
///
/// # Example
/// ```
/// use neural_trader_backend::security::path_validation::validate_filename;
///
/// assert!(validate_filename("report.txt").is_ok());
/// assert!(validate_filename("../etc/passwd").is_err());
/// assert!(validate_filename("file<>name").is_err());
/// ```
pub fn validate_filename(filename: &str) -> Result<()> {
    // Reject empty filenames
    if filename.is_empty() {
        return Err(anyhow!("Filename cannot be empty"));
    }

    // Reject special directory names
    if filename == "." || filename == ".." {
        return Err(anyhow!("Invalid filename: '{}'", filename));
    }

    // Define dangerous characters
    let dangerous_chars = [
        '/',   // Unix path separator
        '\\',  // Windows path separator
        ':',   // Windows drive letter separator, NTFS alternate data streams
        '*',   // Wildcard
        '?',   // Wildcard
        '"',   // Quote
        '<',   // Redirect
        '>',   // Redirect
        '|',   // Pipe
        '\0',  // Null byte
    ];

    // Check for dangerous characters
    for ch in dangerous_chars {
        if filename.contains(ch) {
            return Err(anyhow!(
                "Filename contains invalid character '{}': {}",
                ch,
                filename
            ));
        }
    }

    // Reject control characters
    for ch in filename.chars() {
        if ch.is_control() {
            return Err(anyhow!(
                "Filename contains control character: {}",
                filename
            ));
        }
    }

    // Reject excessively long filenames (common limit is 255)
    if filename.len() > 255 {
        return Err(anyhow!(
            "Filename too long ({} characters, max 255): {}",
            filename.len(),
            filename
        ));
    }

    Ok(())
}

/// Sanitize a filename by removing/replacing dangerous characters
///
/// This is a more permissive alternative to `validate_filename` that
/// attempts to make the filename safe rather than rejecting it.
///
/// # Example
/// ```
/// use neural_trader_backend::security::path_validation::sanitize_filename;
///
/// assert_eq!(sanitize_filename("my<file>name.txt"), "my_file_name.txt");
/// assert_eq!(sanitize_filename("../../etc/passwd"), "_.._.._etc_passwd");
/// ```
pub fn sanitize_filename(filename: &str) -> String {
    filename
        .chars()
        .map(|ch| {
            match ch {
                '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' | '\0' => '_',
                ch if ch.is_control() => '_',
                ch => ch,
            }
        })
        .take(255) // Limit length
        .collect()
}

/// Extract extension from a filename safely
///
/// Returns lowercase extension without the dot, or empty string if no extension
pub fn get_safe_extension(filename: &str) -> String {
    Path::new(filename)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_lowercase())
        .unwrap_or_default()
}

/// Validate file extension against allowlist
///
/// # Example
/// ```
/// use neural_trader_backend::security::path_validation::validate_extension;
///
/// let allowed = vec!["txt", "json", "csv"];
/// assert!(validate_extension("report.txt", &allowed).is_ok());
/// assert!(validate_extension("script.exe", &allowed).is_err());
/// ```
pub fn validate_extension(filename: &str, allowed_extensions: &[&str]) -> Result<()> {
    let extension = get_safe_extension(filename);

    if extension.is_empty() && !allowed_extensions.is_empty() {
        return Err(anyhow!("File has no extension: {}", filename));
    }

    if !allowed_extensions.is_empty() && !allowed_extensions.contains(&extension.as_str()) {
        return Err(anyhow!(
            "File extension '{}' not allowed. Allowed: {}",
            extension,
            allowed_extensions.join(", ")
        ));
    }

    Ok(())
}

/// Common allowlists for different file types
pub mod allowlists {
    pub const TEXT_FILES: &[&str] = &["txt", "md", "csv", "json", "xml", "yaml", "yml"];
    pub const IMAGE_FILES: &[&str] = &["jpg", "jpeg", "png", "gif", "svg", "webp"];
    pub const DOCUMENT_FILES: &[&str] = &["pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx"];
    pub const DATA_FILES: &[&str] = &["json", "csv", "xml", "yaml", "yml", "parquet"];
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_validate_safe_path_traversal_attacks() {
        let temp_dir = TempDir::new().unwrap();
        let base = temp_dir.path();

        // Test various traversal attempts
        assert!(validate_safe_path("../etc/passwd", base).is_err());
        assert!(validate_safe_path("files/../../etc/passwd", base).is_err());
        assert!(validate_safe_path("~/secret", base).is_err());
    }

    #[test]
    fn test_validate_safe_path_absolute_paths() {
        let temp_dir = TempDir::new().unwrap();
        let base = temp_dir.path();

        #[cfg(unix)]
        {
            assert!(validate_safe_path("/etc/passwd", base).is_err());
            assert!(validate_safe_path("/var/log/system.log", base).is_err());
        }

        #[cfg(windows)]
        {
            assert!(validate_safe_path("C:\\Windows\\System32", base).is_err());
            assert!(validate_safe_path("\\\\server\\share", base).is_err());
        }
    }

    #[test]
    fn test_validate_safe_path_valid_paths() {
        let temp_dir = TempDir::new().unwrap();
        let base = temp_dir.path();

        // Create a test file
        let test_file = base.join("test.txt");
        fs::write(&test_file, "test").unwrap();

        // Valid relative path to existing file
        assert!(validate_safe_path("test.txt", base).is_ok());

        // Create subdirectory
        let subdir = base.join("subdir");
        fs::create_dir(&subdir).unwrap();
        let nested_file = subdir.join("nested.txt");
        fs::write(&nested_file, "test").unwrap();

        // Valid nested path
        assert!(validate_safe_path("subdir/nested.txt", base).is_ok());
    }

    #[test]
    fn test_validate_filename() {
        // Valid filenames
        assert!(validate_filename("report.txt").is_ok());
        assert!(validate_filename("my-file_123.json").is_ok());

        // Invalid filenames
        assert!(validate_filename("").is_err());
        assert!(validate_filename(".").is_err());
        assert!(validate_filename("..").is_err());
        assert!(validate_filename("file/path.txt").is_err());
        assert!(validate_filename("file\\path.txt").is_err());
        assert!(validate_filename("file:name.txt").is_err());
        assert!(validate_filename("file*name.txt").is_err());
        assert!(validate_filename("file?name.txt").is_err());
        assert!(validate_filename("file<name>.txt").is_err());
        assert!(validate_filename("file|name.txt").is_err());
    }

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("valid-file.txt"), "valid-file.txt");
        assert_eq!(sanitize_filename("my<file>name.txt"), "my_file_name.txt");
        assert_eq!(sanitize_filename("../../etc/passwd"), "_.._.._etc_passwd");
        assert_eq!(sanitize_filename("file:name*.txt"), "file_name_.txt");
    }

    #[test]
    fn test_get_safe_extension() {
        assert_eq!(get_safe_extension("file.txt"), "txt");
        assert_eq!(get_safe_extension("file.TXT"), "txt");
        assert_eq!(get_safe_extension("archive.tar.gz"), "gz");
        assert_eq!(get_safe_extension("no_extension"), "");
    }

    #[test]
    fn test_validate_extension() {
        let allowed = vec!["txt", "json", "csv"];

        assert!(validate_extension("report.txt", &allowed).is_ok());
        assert!(validate_extension("data.json", &allowed).is_ok());
        assert!(validate_extension("values.CSV", &allowed).is_ok());
        assert!(validate_extension("script.exe", &allowed).is_err());
        assert!(validate_extension("no_ext", &allowed).is_err());
    }
}
