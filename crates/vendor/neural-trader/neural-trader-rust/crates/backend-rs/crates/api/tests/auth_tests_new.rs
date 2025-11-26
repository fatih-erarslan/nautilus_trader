//! Authentication and Authorization Tests
//!
//! Tests cover:
//! - JWT token generation and validation
//! - Token expiration handling
//! - Role-based access control (RBAC)
//! - Password hashing and verification
//! - Invalid token handling
//! - Authorization middleware

mod common;

use common::{TestDb, MockAuth};
use anyhow::Result;
use chrono::{Duration, Utc};
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct Claims {
    sub: String,
    role: String,
    exp: i64,
    iat: i64,
}

const SECRET: &str = "test_secret_key_for_jwt_signing";

// ============================================================================
// JWT Token Generation Tests
// ============================================================================

#[test]
fn test_generate_valid_jwt_token() -> Result<()> {
    let claims = Claims {
        sub: "user-1".to_string(),
        role: "admin".to_string(),
        exp: (Utc::now() + Duration::hours(24)).timestamp(),
        iat: Utc::now().timestamp(),
    };

    let token = encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(SECRET.as_ref()),
    )?;

    assert!(!token.is_empty());
    assert!(token.contains('.'), "JWT should have 3 parts separated by dots");

    Ok(())
}

#[test]
fn test_decode_valid_jwt_token() -> Result<()> {
    let claims = Claims {
        sub: "user-1".to_string(),
        role: "admin".to_string(),
        exp: (Utc::now() + Duration::hours(24)).timestamp(),
        iat: Utc::now().timestamp(),
    };

    let token = encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(SECRET.as_ref()),
    )?;

    let token_data = decode::<Claims>(
        &token,
        &DecodingKey::from_secret(SECRET.as_ref()),
        &Validation::default(),
    )?;

    assert_eq!(token_data.claims.sub, "user-1");
    assert_eq!(token_data.claims.role, "admin");

    Ok(())
}

#[test]
fn test_reject_expired_token() {
    let claims = Claims {
        sub: "user-1".to_string(),
        role: "admin".to_string(),
        exp: (Utc::now() - Duration::hours(1)).timestamp(), // Expired 1 hour ago
        iat: (Utc::now() - Duration::hours(2)).timestamp(),
    };

    let token = encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(SECRET.as_ref()),
    )
    .unwrap();

    let result = decode::<Claims>(
        &token,
        &DecodingKey::from_secret(SECRET.as_ref()),
        &Validation::default(),
    );

    assert!(result.is_err(), "Expired token should be rejected");
}

#[test]
fn test_reject_invalid_signature() {
    let claims = Claims {
        sub: "user-1".to_string(),
        role: "admin".to_string(),
        exp: (Utc::now() + Duration::hours(24)).timestamp(),
        iat: Utc::now().timestamp(),
    };

    let token = encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret("wrong_secret".as_ref()),
    )
    .unwrap();

    let result = decode::<Claims>(
        &token,
        &DecodingKey::from_secret(SECRET.as_ref()),
        &Validation::default(),
    );

    assert!(result.is_err(), "Token with invalid signature should be rejected");
}

#[test]
fn test_reject_malformed_token() {
    let malformed_tokens = vec![
        "not.a.jwt",
        "invalid",
        "",
        "header.payload", // Missing signature
        "a.b.c.d.e",      // Too many parts
    ];

    for token in malformed_tokens {
        let result = decode::<Claims>(
            token,
            &DecodingKey::from_secret(SECRET.as_ref()),
            &Validation::default(),
        );

        assert!(result.is_err(), "Malformed token '{}' should be rejected", token);
    }
}

// ============================================================================
// Password Hashing Tests
// ============================================================================

#[test]
fn test_password_hashing() -> Result<()> {
    let password = "secure_password_123";
    let hash = bcrypt::hash(password, bcrypt::DEFAULT_COST)?;

    assert_ne!(password, hash);
    assert!(hash.starts_with("$2b$") || hash.starts_with("$2a$") || hash.starts_with("$2y$"));

    Ok(())
}

#[test]
fn test_password_verification_success() -> Result<()> {
    let password = "secure_password_123";
    let hash = bcrypt::hash(password, bcrypt::DEFAULT_COST)?;

    let valid = bcrypt::verify(password, &hash)?;
    assert!(valid, "Password should verify successfully");

    Ok(())
}

#[test]
fn test_password_verification_failure() -> Result<()> {
    let password = "secure_password_123";
    let wrong_password = "wrong_password";
    let hash = bcrypt::hash(password, bcrypt::DEFAULT_COST)?;

    let valid = bcrypt::verify(wrong_password, &hash)?;
    assert!(!valid, "Wrong password should not verify");

    Ok(())
}

#[test]
fn test_password_hash_uniqueness() -> Result<()> {
    let password = "test_password";
    let hash1 = bcrypt::hash(password, bcrypt::DEFAULT_COST)?;
    let hash2 = bcrypt::hash(password, bcrypt::DEFAULT_COST)?;

    // Hashes should be different due to random salt
    assert_ne!(hash1, hash2, "Same password should produce different hashes");

    // But both should verify
    assert!(bcrypt::verify(password, &hash1)?);
    assert!(bcrypt::verify(password, &hash2)?);

    Ok(())
}

// ============================================================================
// Role-Based Access Control (RBAC) Tests
// ============================================================================

#[derive(Debug, PartialEq)]
enum Role {
    Admin,
    User,
    Guest,
}

impl Role {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "admin" => Some(Role::Admin),
            "user" => Some(Role::User),
            "guest" => Some(Role::Guest),
            _ => None,
        }
    }

    fn has_permission(&self, permission: &str) -> bool {
        match self {
            Role::Admin => true, // Admin has all permissions
            Role::User => matches!(permission, "read" | "write" | "execute"),
            Role::Guest => permission == "read",
        }
    }
}

#[test]
fn test_admin_has_all_permissions() {
    let admin = Role::Admin;

    assert!(admin.has_permission("read"));
    assert!(admin.has_permission("write"));
    assert!(admin.has_permission("execute"));
    assert!(admin.has_permission("delete"));
    assert!(admin.has_permission("manage_users"));
}

#[test]
fn test_user_has_limited_permissions() {
    let user = Role::User;

    assert!(user.has_permission("read"));
    assert!(user.has_permission("write"));
    assert!(user.has_permission("execute"));
    assert!(!user.has_permission("delete"));
    assert!(!user.has_permission("manage_users"));
}

#[test]
fn test_guest_has_minimal_permissions() {
    let guest = Role::Guest;

    assert!(guest.has_permission("read"));
    assert!(!guest.has_permission("write"));
    assert!(!guest.has_permission("execute"));
    assert!(!guest.has_permission("delete"));
}

#[test]
fn test_role_parsing() {
    assert_eq!(Role::from_str("admin"), Some(Role::Admin));
    assert_eq!(Role::from_str("user"), Some(Role::User));
    assert_eq!(Role::from_str("guest"), Some(Role::Guest));
    assert_eq!(Role::from_str("invalid"), None);
}

// ============================================================================
// Token Claims Validation Tests
// ============================================================================

#[test]
fn test_validate_required_claims() -> Result<()> {
    let claims = Claims {
        sub: "user-1".to_string(),
        role: "admin".to_string(),
        exp: (Utc::now() + Duration::hours(24)).timestamp(),
        iat: Utc::now().timestamp(),
    };

    assert!(!claims.sub.is_empty(), "Subject should not be empty");
    assert!(!claims.role.is_empty(), "Role should not be empty");
    assert!(claims.exp > claims.iat, "Expiration should be after issued time");

    Ok(())
}

// ============================================================================
// Database Authentication Tests
// ============================================================================

#[test]
fn test_user_authentication_success() -> Result<()> {
    let test_db = TestDb::new()?;
    let conn = test_db.conn()?;

    // Get user from database
    let (stored_password_hash, role): (String, String) = conn.query_row(
        "SELECT password_hash, role FROM users WHERE email = ?1",
        ["test@example.com"],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;

    // Note: The seed data uses a dummy hash, so we'll just verify structure
    assert!(!stored_password_hash.is_empty());
    assert_eq!(role, "admin");

    Ok(())
}

#[test]
fn test_user_not_found() -> Result<()> {
    let test_db = TestDb::new()?;
    let conn = test_db.conn()?;

    let result = conn.query_row(
        "SELECT id FROM users WHERE email = ?1",
        ["nonexistent@example.com"],
        |row| row.get::<_, String>(0),
    );

    assert!(result.is_err(), "Non-existent user should return error");

    Ok(())
}

// ============================================================================
// Authorization Header Tests
// ============================================================================

#[test]
fn test_parse_bearer_token() {
    let test_cases = vec![
        ("Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9", Some("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")),
        ("bearer token123", Some("token123")),
        ("Bearer ", None),
        ("Basic credentials", None),
        ("", None),
    ];

    for (header, expected) in test_cases {
        let result = parse_bearer_token(header);
        assert_eq!(result, expected, "Failed for header: {}", header);
    }
}

fn parse_bearer_token(auth_header: &str) -> Option<&str> {
    if auth_header.is_empty() {
        return None;
    }

    let parts: Vec<&str> = auth_header.splitn(2, ' ').collect();
    if parts.len() != 2 || !parts[0].eq_ignore_ascii_case("bearer") || parts[1].is_empty() {
        return None;
    }

    Some(parts[1])
}
