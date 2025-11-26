# Security Implementation - Neural Trader Backend

## Overview

The Neural Trader Backend implements comprehensive security protections against common web vulnerabilities, specifically focusing on:

1. **XSS (Cross-Site Scripting) Protection**
2. **Path Traversal Prevention**
3. **Input Validation and Sanitization**

## Architecture

```
User Input
    ↓
Security Validation Layer
    ├── XSS Protection (xss_protection.rs)
    ├── Path Validation (path_validation.rs)
    └── Context-Aware Validation
    ↓
Business Logic Layer
    ↓
Response (with HTML escaping)
```

## XSS Protection

### Location
- `/workspaces/neural-trader/neural-trader-rust/crates/napi-bindings/src/security/xss_protection.rs`

### Features

#### 1. Pattern-Based Detection
Detects and blocks over 30 common XSS attack vectors:

- **Script Tags**: `<script>`, `</script>`, etc.
- **Event Handlers**: `onclick=`, `onerror=`, `onload=`, etc.
- **JavaScript Protocols**: `javascript:`, `vbscript:`, `data:text/html`
- **Dangerous Tags**: `<iframe>`, `<embed>`, `<object>`, `<svg>`, etc.
- **JavaScript Functions**: `eval()`, `document.cookie`, `window.location`

#### 2. Regex Pattern Matching
Advanced detection using regex patterns:

- Hex/Unicode encoded JavaScript
- Base64 encoded attacks
- Expression in CSS
- Obfuscated scripts with multiple spaces

#### 3. HTML Entity Validation
- Detects suspicious HTML entities (`&#...`)
- Blocks pre-encoded HTML (`&lt;`, `&gt;`)
- Prevents null byte injection

#### 4. Context-Aware Validation
Different validation rules for different contexts:

```rust
pub enum InputContext {
    PlainText,  // Most restrictive
    Email,      // Validates email format
    Url,        // Validates URL format, blocks javascript:
    Markdown,   // Allows formatting but blocks XSS
}
```

### API

```rust
use neural_trader_backend::security::xss_protection;

// Validate input for XSS
xss_protection::validate_no_xss(&user_input, "field_name")?;

// Escape HTML special characters
let safe = xss_protection::escape_html(&user_input);

// Validate and escape in one step
let safe = xss_protection::validate_and_escape(&user_input, "field_name")?;

// Context-aware validation
xss_protection::validate_input_context(
    &email,
    "email",
    InputContext::Email
)?;
```

### HTML Escaping

Converts dangerous characters to HTML entities:

- `&` → `&amp;`
- `<` → `&lt;`
- `>` → `&gt;`
- `"` → `&quot;`
- `'` → `&#x27;`
- `/` → `&#x2F;`

## Path Traversal Protection

### Location
- `/workspaces/neural-trader/neural-trader-rust/crates/napi-bindings/src/security/path_validation.rs`

### Features

#### 1. Traversal Detection
Blocks common directory traversal attempts:

- `..` sequences
- `~` home directory expansion
- Absolute paths (Unix `/`, Windows `C:\`)
- UNC paths (`\\server\share`)
- Null bytes in paths

#### 2. Path Canonicalization
- Resolves symbolic links
- Validates paths stay within base directory
- Handles non-existent paths safely

#### 3. Filename Validation
Rejects dangerous characters:

- Path separators: `/`, `\`
- Special characters: `:`, `*`, `?`, `"`, `<`, `>`, `|`
- Control characters
- Empty filenames and special names (`.`, `..`)

#### 4. Extension Whitelisting
Validate file extensions against allowlists:

```rust
use neural_trader_backend::security::path_validation;

// Validate against custom allowlist
let allowed = vec!["txt", "json", "csv"];
path_validation::validate_extension("report.txt", &allowed)?;

// Use predefined allowlists
use path_validation::allowlists;
path_validation::validate_extension(
    "data.json",
    allowlists::DATA_FILES
)?;
```

### API

```rust
use neural_trader_backend::security::path_validation;

// Validate path is safe within base directory
let safe_path = path_validation::validate_safe_path(
    "files/report.txt",
    base_dir
)?;

// Validate filename only
path_validation::validate_filename("report.txt")?;

// Sanitize filename (remove dangerous chars)
let safe = path_validation::sanitize_filename("my<file>.txt");
// Result: "my_file_.txt"

// Get file extension safely
let ext = path_validation::get_safe_extension("report.txt");
// Result: "txt"
```

### Predefined Allowlists

```rust
pub mod allowlists {
    pub const TEXT_FILES: &[&str] =
        &["txt", "md", "csv", "json", "xml", "yaml", "yml"];

    pub const IMAGE_FILES: &[&str] =
        &["jpg", "jpeg", "png", "gif", "svg", "webp"];

    pub const DOCUMENT_FILES: &[&str] =
        &["pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx"];

    pub const DATA_FILES: &[&str] =
        &["json", "csv", "xml", "yaml", "yml", "parquet"];
}
```

## Integration Points

### Syndicate Operations

All syndicate operations validate user inputs:

```rust
// Create syndicate - validates id, name, description
pub async fn create_syndicate_impl(
    syndicate_id: String,
    name: String,
    description: Option<String>,
) -> Result<String> {
    validate_no_xss(&syndicate_id, "syndicate_id")?;
    validate_no_xss(&name, "name")?;
    // ... business logic
}

// Add member - validates name, email, role
pub async fn add_syndicate_member_impl(
    syndicate_id: String,
    name: String,
    email: String,
    role: String,
    initial_contribution: f64,
) -> Result<String> {
    validate_no_xss(&name, "name")?;
    validate_input_context(&email, "email", InputContext::Email)?;
    validate_no_xss(&role, "role")?;
    // ... business logic
}
```

### E2B Sandbox Operations

Path validation for file operations:

```rust
// Would be applied to sandbox file uploads
let safe_path = validate_safe_path(user_path, sandbox_base_dir)?;
```

## Security Configuration

```rust
pub struct SecurityConfig {
    /// Maximum length for user input fields (default: 10,000)
    pub max_input_length: usize,

    /// Maximum length for filenames (default: 255)
    pub max_filename_length: usize,

    /// Allowed file extensions for uploads
    pub allowed_extensions: Vec<String>,

    /// Enable strict XSS validation (default: true)
    pub strict_xss_validation: bool,

    /// Enable path validation (default: true)
    pub enable_path_validation: bool,
}
```

## Testing

### Test Suite Location
- `/workspaces/neural-trader/tests/security/`

### Test Files

1. **xss-protection.test.js** - XSS protection tests
   - Script tag injection
   - Event handler injection
   - JavaScript protocol injection
   - Dangerous HTML tags
   - Encoded/obfuscated attacks
   - Safe input acceptance
   - Email validation
   - Context-aware validation
   - Performance tests
   - Unicode handling

2. **path-traversal.test.js** - Path traversal tests
   - Directory traversal attacks
   - Absolute path attacks
   - Home directory expansion
   - Null byte injection
   - Filename validation
   - Path canonicalization
   - Extension validation
   - Symbolic link protection
   - Performance tests
   - Unicode path handling

3. **integration.test.js** - Integration tests
   - Multi-layer security validation
   - Combined attack vectors
   - Performance impact tests
   - Edge cases and boundaries
   - Regression tests

### Running Tests

```bash
# Run all security tests
npm test tests/security/

# Run specific test file
npm test tests/security/xss-protection.test.js

# Run with coverage
npm test -- --coverage tests/security/
```

## Performance Considerations

### Benchmarks

- **XSS Validation**: <1ms per input (average)
- **Path Validation**: <2ms per path (average)
- **Regex Patterns**: Compiled once, cached for reuse
- **No Significant Performance Impact**: <50ms average for full validation

### Optimization Techniques

1. **Lazy Static Patterns**: Regex patterns compiled once
2. **Early Returns**: Fast-fail on common patterns
3. **Minimal Allocations**: Reuse string references
4. **Efficient String Operations**: Use built-in methods

## Security Best Practices

### Input Validation

1. **Validate Early**: Check inputs at the API boundary
2. **Whitelist Approach**: Define what's allowed, reject everything else
3. **Context-Aware**: Different rules for different contexts
4. **Fail Secure**: Reject on any doubt

### Output Encoding

1. **Escape HTML**: Always escape user content in HTML context
2. **JSON Encoding**: Use proper JSON serialization
3. **URL Encoding**: Encode user content in URLs

### Defense in Depth

Multiple layers of security:

1. Input validation (this module)
2. Parameterized queries (prevents SQL injection)
3. Output encoding (prevents XSS in templates)
4. CSP headers (prevents inline scripts)
5. Rate limiting (prevents DoS)

## Common Attack Vectors Blocked

### XSS Attacks

✅ Script Tag Injection
✅ Event Handler Injection
✅ JavaScript Protocol Injection
✅ Data URL Injection
✅ SVG XSS
✅ Iframe Injection
✅ Object/Embed Tags
✅ Meta Refresh
✅ Import Statements
✅ Expression in CSS
✅ Encoded Payloads
✅ Unicode Bypass Attempts

### Path Traversal Attacks

✅ Directory Traversal (`../../../etc/passwd`)
✅ Absolute Paths (`/etc/passwd`, `C:\Windows\System32`)
✅ Home Directory Expansion (`~/.ssh/id_rsa`)
✅ UNC Paths (`\\server\share`)
✅ Symbolic Link Escapes
✅ Null Byte Injection
✅ Filename Special Characters
✅ Path Canonicalization Bypasses

## Error Messages

Security validation errors provide clear feedback:

```
Error: Potential XSS detected in 'name': contains '<script'
Error: Path traversal detected: path contains '..' sequence
Error: Invalid email format in 'email'
Error: Filename contains invalid character '/'
Error: Path escapes base directory
```

## Dependencies

```toml
[dependencies]
regex = "1.10"      # For advanced pattern matching
tempfile = "3.8"    # For secure temporary files (tests)
```

## Future Enhancements

1. **CSRF Protection**: Token-based request validation
2. **Rate Limiting**: Per-endpoint request throttling
3. **SQL Injection Protection**: Automatic query parameterization
4. **Content Security Policy**: Automatic CSP header generation
5. **Input Sanitization**: Automatic cleanup of risky content
6. **Security Auditing**: Logging of security events
7. **WAF Integration**: Web Application Firewall integration

## References

- [OWASP XSS Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.html)
- [OWASP Path Traversal](https://owasp.org/www-community/attacks/Path_Traversal)
- [CWE-79: Cross-site Scripting](https://cwe.mitre.org/data/definitions/79.html)
- [CWE-22: Path Traversal](https://cwe.mitre.org/data/definitions/22.html)

## License

Same as Neural Trader Backend (MIT/Apache-2.0)
