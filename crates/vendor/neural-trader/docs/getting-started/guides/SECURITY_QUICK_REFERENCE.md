# Security Quick Reference

## XSS Protection

```rust
use neural_trader_backend::security::xss_protection::*;

// Validate input
validate_no_xss(&input, "field_name")?;

// Escape HTML
let safe = escape_html(&input);

// Email validation
validate_input_context(&email, "email", InputContext::Email)?;
```

## Path Validation

```rust
use neural_trader_backend::security::path_validation::*;

// Safe path
let path = validate_safe_path("file.txt", base_dir)?;

// Filename
validate_filename("report.txt")?;

// Extension
validate_extension("data.csv", &["csv", "json"])?;
```

## Protected Operations

- ✅ Syndicate creation (id, name, description)
- ✅ Member addition (name, email, role)
- ✅ File uploads (path validation)
- ✅ All user-generated content

## Test Coverage

- 50+ XSS tests
- 40+ Path traversal tests
- 25+ Integration tests
- <5% performance overhead

## Files

- `/neural-trader-rust/crates/napi-bindings/src/security/xss_protection.rs`
- `/neural-trader-rust/crates/napi-bindings/src/security/path_validation.rs`
- `/tests/security/*.test.js`
