//! Production Configuration Validation
//! 
//! Validates that all security configurations meet production requirements

use std::env;
use std::process;

fn main() {
    println!("🔒 ATS-Core Production Configuration Validation");
    println!("===============================================");
    
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    
    // Check environment
    let environment = env::var("ATS_ENVIRONMENT").unwrap_or_else(|_| "development".to_string());
    println!("Environment: {}", environment);
    
    if environment == "production" {
        validate_production_config(&mut errors, &mut warnings);
    } else {
        println!("⚠️  Not a production environment, running basic validation only");
        validate_basic_config(&mut errors, &mut warnings);
    }
    
    // Report results
    println!("\n📊 Validation Results:");
    println!("=====================");
    
    if !warnings.is_empty() {
        println!("\n⚠️  WARNINGS:");
        for warning in &warnings {
            println!("   • {}", warning);
        }
    }
    
    if !errors.is_empty() {
        println!("\n❌ CRITICAL ERRORS:");
        for error in &errors {
            println!("   • {}", error);
        }
        println!("\n🚫 PRODUCTION DEPLOYMENT BLOCKED!");
        println!("   Fix all critical errors before deploying to production.");
        process::exit(1);
    } else if !warnings.is_empty() {
        println!("\n⚠️  {} warnings found - review before production deployment", warnings.len());
        process::exit(1);
    } else {
        println!("\n✅ All validation checks passed!");
        println!("   Configuration is ready for production deployment.");
    }
}

fn validate_production_config(errors: &mut Vec<String>, warnings: &mut Vec<String>) {
    // Validate JWT configuration
    validate_jwt_config(errors, warnings);
    
    // Validate encryption configuration
    validate_encryption_config(errors, warnings);
    
    // Validate database configuration
    validate_database_config(errors, warnings);
    
    // Validate security settings
    validate_security_settings(errors, warnings);
    
    // Check for development artifacts
    check_development_artifacts(errors, warnings);
}

fn validate_basic_config(errors: &mut Vec<String>, warnings: &mut Vec<String>) {
    // Basic JWT validation
    if let Ok(secret) = env::var("ATS_JWT_SECRET") {
        if secret.len() < 16 {
            errors.push("ATS_JWT_SECRET must be at least 16 characters".to_string());
        }
        
        if secret == "your-secret-key-change-this" {
            errors.push("ATS_JWT_SECRET is using default value - must be changed".to_string());
        }
    }
    
    // Basic database validation
    if env::var("ATS_DB_PASSWORD").unwrap_or_default() == "password" {
        warnings.push("ATS_DB_PASSWORD appears to be using a weak default".to_string());
    }
}

fn validate_jwt_config(errors: &mut Vec<String>, warnings: &mut Vec<String>) {
    // Check JWT secret
    match env::var("ATS_JWT_SECRET") {
        Ok(secret) => {
            if secret.len() < 64 {
                errors.push(format!("ATS_JWT_SECRET must be at least 64 characters for production (current: {})", secret.len()));
            }
            
            // Check for weak patterns
            let weak_patterns = ["password", "secret", "123456", "admin", "test", "changeme", "your-secret-key-change-this"];
            for pattern in &weak_patterns {
                if secret.to_lowercase().contains(pattern) {
                    errors.push(format!("ATS_JWT_SECRET contains weak pattern: '{}'", pattern));
                }
            }
            
            // Check character diversity
            let has_upper = secret.chars().any(|c| c.is_ascii_uppercase());
            let has_lower = secret.chars().any(|c| c.is_ascii_lowercase());
            let has_digit = secret.chars().any(|c| c.is_ascii_digit());
            let has_special = secret.chars().any(|c| !c.is_alphanumeric());
            
            if !(has_upper && has_lower && has_digit && has_special) {
                warnings.push("ATS_JWT_SECRET should contain uppercase, lowercase, digits, and special characters".to_string());
            }
        }
        Err(_) => {
            errors.push("ATS_JWT_SECRET environment variable is required for production".to_string());
        }
    }
    
    // Check JWT algorithm
    let algorithm = env::var("ATS_JWT_ALGORITHM").unwrap_or_else(|_| "HS256".to_string());
    if algorithm == "HS256" {
        warnings.push("Consider using RS256 instead of HS256 for production JWT".to_string());
    } else if !["HS256", "RS256", "ES256"].contains(&algorithm.as_str()) {
        errors.push(format!("Unsupported JWT algorithm: {}. Use HS256, RS256, or ES256", algorithm));
    }
    
    // Check JWT expiry
    if let Ok(expiry_str) = env::var("ATS_JWT_EXPIRY_SECONDS") {
        if let Ok(expiry) = expiry_str.parse::<u64>() {
            if expiry > 3600 {
                warnings.push(format!("JWT expiry time ({} seconds) is longer than recommended 1 hour for production", expiry));
            }
            if expiry < 300 {
                warnings.push(format!("JWT expiry time ({} seconds) is very short, may cause usability issues", expiry));
            }
        } else {
            errors.push("ATS_JWT_EXPIRY_SECONDS must be a valid number".to_string());
        }
    }
}

fn validate_encryption_config(errors: &mut Vec<String>, warnings: &mut Vec<String>) {
    // Check encryption at rest
    let encrypt_at_rest = env::var("ATS_ENCRYPT_AT_REST").unwrap_or_else(|_| "false".to_string());
    if encrypt_at_rest.to_lowercase() != "true" {
        errors.push("ATS_ENCRYPT_AT_REST must be enabled (true) for production".to_string());
    }
    
    // Check encryption in transit
    let encrypt_in_transit = env::var("ATS_ENCRYPT_IN_TRANSIT").unwrap_or_else(|_| "false".to_string());
    if encrypt_in_transit.to_lowercase() != "true" {
        errors.push("ATS_ENCRYPT_IN_TRANSIT must be enabled (true) for production".to_string());
    }
    
    // Check encryption key
    if let Ok(key) = env::var("ATS_ENCRYPTION_KEY") {
        if key.len() < 64 {
            errors.push(format!("ATS_ENCRYPTION_KEY must be at least 64 characters (256-bit) for production (current: {})", key.len()));
        }
        
        // Validate hex format
        if !key.chars().all(|c| c.is_ascii_hexdigit()) {
            errors.push("ATS_ENCRYPTION_KEY must be a valid hexadecimal string".to_string());
        }
    } else {
        warnings.push("ATS_ENCRYPTION_KEY is not set - encryption features may be disabled".to_string());
    }
}

fn validate_database_config(errors: &mut Vec<String>, warnings: &mut Vec<String>) {
    // Check database password
    match env::var("ATS_DB_PASSWORD") {
        Ok(password) => {
            if password.len() < 12 {
                errors.push(format!("ATS_DB_PASSWORD must be at least 12 characters for production (current: {})", password.len()));
            }
            
            let weak_passwords = ["password", "123456", "admin", "root", "postgres", "database"];
            for weak in &weak_passwords {
                if password.to_lowercase() == *weak {
                    errors.push(format!("ATS_DB_PASSWORD is using a common weak password: '{}'", weak));
                }
            }
        }
        Err(_) => {
            warnings.push("ATS_DB_PASSWORD is not set".to_string());
        }
    }
    
    // Check SSL mode
    let ssl_mode = env::var("ATS_DB_SSL_MODE").unwrap_or_else(|_| "prefer".to_string());
    if ssl_mode != "require" {
        errors.push("ATS_DB_SSL_MODE must be set to 'require' for production database connections".to_string());
    }
}

fn validate_security_settings(errors: &mut Vec<String>, warnings: &mut Vec<String>) {
    // Check rate limiting
    if let Ok(rpm_str) = env::var("ATS_RATE_LIMIT_RPM") {
        if let Ok(rpm) = rpm_str.parse::<u32>() {
            if rpm == 0 {
                errors.push("Rate limiting must be enabled (ATS_RATE_LIMIT_RPM > 0) for production".to_string());
            } else if rpm > 1000 {
                warnings.push(format!("Rate limit ({} RPM) is very high for production", rpm));
            }
        }
    } else {
        warnings.push("ATS_RATE_LIMIT_RPM is not configured - using default rate limiting".to_string());
    }
    
    // Check security audit
    let audit_enabled = env::var("ATS_SECURITY_AUDIT_ENABLED").unwrap_or_else(|_| "false".to_string());
    if audit_enabled.to_lowercase() != "true" {
        warnings.push("ATS_SECURITY_AUDIT_ENABLED should be enabled for production".to_string());
    }
    
    // Check body size limits
    if let Ok(body_size_str) = env::var("ATS_MAX_BODY_SIZE") {
        if let Ok(body_size) = body_size_str.parse::<usize>() {
            if body_size > 10 * 1024 * 1024 {  // 10MB
                warnings.push(format!("ATS_MAX_BODY_SIZE ({} bytes) is very large, may impact security", body_size));
            }
        }
    }
}

fn check_development_artifacts(errors: &mut Vec<String>, warnings: &mut Vec<String>) {
    // Check for development environment variables that shouldn't be in production
    let dev_vars = [
        "ATS_DEV_MODE",
        "ATS_DEBUG_MODE", 
        "ATS_DISABLE_AUTH",
        "ATS_MOCK_DATA",
        "ATS_TEST_MODE",
    ];
    
    for var in &dev_vars {
        if env::var(var).is_ok() {
            errors.push(format!("Development environment variable {} found in production", var));
        }
    }
    
    // Check log level
    let log_level = env::var("ATS_LOG_LEVEL").unwrap_or_else(|_| "info".to_string());
    if log_level.to_lowercase() == "debug" {
        warnings.push("Log level is set to 'debug' - consider using 'warn' or 'error' for production".to_string());
    }
    
    // Check for development secrets
    let secrets_to_check = ["ATS_JWT_SECRET", "ATS_ENCRYPTION_KEY", "ATS_DB_PASSWORD"];
    for secret_var in &secrets_to_check {
        if let Ok(value) = env::var(secret_var) {
            if value.contains("dev") || value.contains("test") || value.contains("local") {
                warnings.push(format!("{} appears to contain development-related terms", secret_var));
            }
        }
    }
}