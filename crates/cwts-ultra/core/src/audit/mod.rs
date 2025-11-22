//! Audit logging for compliance and security monitoring
//!
//! Provides comprehensive audit trail capabilities for regulatory
//! compliance, security monitoring, and operational analysis.

pub mod logger;

pub use logger::{AuditError, AuditEvent, AuditEventType, AuditLogger, AuditStats};
