//! Biometric Authentication
//!
//! This module provides biometric authentication implementations.

use crate::error::QuantumSecurityError;
use crate::types::*;
use super::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Biometric Authentication Provider
#[derive(Debug, Clone)]
pub struct BiometricProvider {
    pub id: Uuid,
    pub name: String,
    pub supported_types: Vec<BiometricType>,
    pub templates: HashMap<String, BiometricTemplate>,
    pub enabled: bool,
}

/// Biometric Template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiometricTemplate {
    pub template_id: String,
    pub biometric_type: BiometricType,
    pub template_data: Vec<u8>,
    pub quality_score: f64,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub metadata: HashMap<String, String>,
}

/// Biometric Verification Result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiometricVerificationResult {
    pub success: bool,
    pub confidence_score: f64,
    pub template_id: String,
    pub quality_score: f64,
    pub verification_time_ms: u64,
    pub error_message: Option<String>,
}

/// Biometric Enrollment Result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiometricEnrollmentResult {
    pub success: bool,
    pub template_id: String,
    pub quality_score: f64,
    pub enrollment_time_ms: u64,
    pub error_message: Option<String>,
}

impl BiometricProvider {
    /// Create new biometric provider
    pub fn new(name: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            supported_types: vec![
                BiometricType::Fingerprint,
                BiometricType::FaceRecognition,
                BiometricType::IrisRecognition,
                BiometricType::VoiceRecognition,
            ],
            templates: HashMap::new(),
            enabled: true,
        }
    }

    /// Enroll biometric template
    pub async fn enroll_template(
        &mut self,
        biometric_type: BiometricType,
        biometric_data: Vec<u8>,
        agent_id: String,
    ) -> Result<BiometricEnrollmentResult, QuantumSecurityError> {
        let start_time = std::time::Instant::now();

        // Validate biometric data
        if biometric_data.is_empty() {
            return Ok(BiometricEnrollmentResult {
                success: false,
                template_id: String::new(),
                quality_score: 0.0,
                enrollment_time_ms: start_time.elapsed().as_millis() as u64,
                error_message: Some("Empty biometric data".to_string()),
            });
        }

        // Calculate quality score (mock implementation)
        let quality_score = self.calculate_quality_score(&biometric_data);

        if quality_score < 0.7 {
            return Ok(BiometricEnrollmentResult {
                success: false,
                template_id: String::new(),
                quality_score,
                enrollment_time_ms: start_time.elapsed().as_millis() as u64,
                error_message: Some("Quality score too low".to_string()),
            });
        }

        // Create template
        let template_id = format!("{}_{}", agent_id, Uuid::new_v4());
        let template = BiometricTemplate {
            template_id: template_id.clone(),
            biometric_type,
            template_data: biometric_data,
            quality_score,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            metadata: HashMap::new(),
        };

        self.templates.insert(template_id.clone(), template);

        Ok(BiometricEnrollmentResult {
            success: true,
            template_id,
            quality_score,
            enrollment_time_ms: start_time.elapsed().as_millis() as u64,
            error_message: None,
        })
    }

    /// Verify biometric
    pub async fn verify_biometric(
        &self,
        template_id: &str,
        biometric_data: Vec<u8>,
    ) -> Result<BiometricVerificationResult, QuantumSecurityError> {
        let start_time = std::time::Instant::now();

        // Get template
        let template = self.templates.get(template_id)
            .ok_or_else(|| QuantumSecurityError::TemplateNotFound("Template not found".to_string()))?;

        // Validate biometric data
        if biometric_data.is_empty() {
            return Ok(BiometricVerificationResult {
                success: false,
                confidence_score: 0.0,
                template_id: template_id.to_string(),
                quality_score: 0.0,
                verification_time_ms: start_time.elapsed().as_millis() as u64,
                error_message: Some("Empty biometric data".to_string()),
            });
        }

        // Calculate quality score
        let quality_score = self.calculate_quality_score(&biometric_data);

        if quality_score < 0.5 {
            return Ok(BiometricVerificationResult {
                success: false,
                confidence_score: 0.0,
                template_id: template_id.to_string(),
                quality_score,
                verification_time_ms: start_time.elapsed().as_millis() as u64,
                error_message: Some("Quality score too low".to_string()),
            });
        }

        // Perform matching (mock implementation)
        let confidence_score = self.match_templates(&template.template_data, &biometric_data);
        let success = confidence_score > 0.8;

        Ok(BiometricVerificationResult {
            success,
            confidence_score,
            template_id: template_id.to_string(),
            quality_score,
            verification_time_ms: start_time.elapsed().as_millis() as u64,
            error_message: if success { None } else { Some("Biometric match failed".to_string()) },
        })
    }

    /// Get template information
    pub async fn get_template(&self, template_id: &str) -> Result<BiometricTemplate, QuantumSecurityError> {
        self.templates.get(template_id)
            .cloned()
            .ok_or_else(|| QuantumSecurityError::TemplateNotFound("Template not found".to_string()))
    }

    /// Delete template
    pub async fn delete_template(&mut self, template_id: &str) -> Result<(), QuantumSecurityError> {
        self.templates.remove(template_id)
            .ok_or_else(|| QuantumSecurityError::TemplateNotFound("Template not found".to_string()))?;
        Ok(())
    }

    /// List templates for agent
    pub async fn list_templates(&self, agent_id: &str) -> Result<Vec<BiometricTemplate>, QuantumSecurityError> {
        let templates = self.templates.values()
            .filter(|t| t.template_id.starts_with(agent_id))
            .cloned()
            .collect();
        Ok(templates)
    }

    /// Check if biometric type is supported
    pub fn supports_type(&self, biometric_type: &BiometricType) -> bool {
        self.supported_types.contains(biometric_type)
    }

    /// Calculate quality score (mock implementation)
    fn calculate_quality_score(&self, biometric_data: &[u8]) -> f64 {
        // Mock quality calculation based on data size and content
        if biometric_data.len() < 100 {
            return 0.3;
        }
        
        let non_zero_count = biometric_data.iter().filter(|&&b| b != 0).count();
        let diversity = non_zero_count as f64 / biometric_data.len() as f64;
        
        (diversity * 0.8 + 0.2).min(1.0)
    }

    /// Match templates (mock implementation)
    fn match_templates(&self, template_data: &[u8], biometric_data: &[u8]) -> f64 {
        // Mock matching algorithm
        let min_len = template_data.len().min(biometric_data.len());
        if min_len == 0 {
            return 0.0;
        }

        let matching_bytes = template_data.iter()
            .zip(biometric_data.iter())
            .take(min_len)
            .filter(|(a, b)| a == b)
            .count();

        matching_bytes as f64 / min_len as f64
    }
}

impl Default for BiometricProvider {
    fn default() -> Self {
        Self::new("default-biometric-provider".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_biometric_provider_creation() {
        let provider = BiometricProvider::new("test".to_string());
        assert_eq!(provider.name, "test");
        assert!(provider.enabled);
        assert!(!provider.supported_types.is_empty());
    }

    #[tokio::test]
    async fn test_biometric_enrollment() {
        let mut provider = BiometricProvider::new("test".to_string());
        let biometric_data = vec![1u8; 1000]; // Good quality data
        
        let result = provider.enroll_template(
            BiometricType::Fingerprint,
            biometric_data,
            "test_agent".to_string(),
        ).await;
        
        assert!(result.is_ok());
        let enrollment = result.unwrap();
        assert!(enrollment.success);
        assert!(enrollment.quality_score > 0.7);
        assert!(!enrollment.template_id.is_empty());
    }

    #[tokio::test]
    async fn test_biometric_verification() {
        let mut provider = BiometricProvider::new("test".to_string());
        let biometric_data = vec![1u8; 1000];
        
        // Enroll template
        let enrollment = provider.enroll_template(
            BiometricType::Fingerprint,
            biometric_data.clone(),
            "test_agent".to_string(),
        ).await.unwrap();
        
        // Verify with same data
        let result = provider.verify_biometric(&enrollment.template_id, biometric_data).await;
        assert!(result.is_ok());
        
        let verification = result.unwrap();
        assert!(verification.success);
        assert!(verification.confidence_score > 0.8);
    }

    #[tokio::test]
    async fn test_low_quality_rejection() {
        let mut provider = BiometricProvider::new("test".to_string());
        let low_quality_data = vec![0u8; 50]; // Low quality data
        
        let result = provider.enroll_template(
            BiometricType::Fingerprint,
            low_quality_data,
            "test_agent".to_string(),
        ).await;
        
        assert!(result.is_ok());
        let enrollment = result.unwrap();
        assert!(!enrollment.success);
        assert!(enrollment.quality_score < 0.7);
    }

    #[tokio::test]
    async fn test_template_management() {
        let mut provider = BiometricProvider::new("test".to_string());
        let biometric_data = vec![1u8; 1000];
        
        // Enroll template
        let enrollment = provider.enroll_template(
            BiometricType::Fingerprint,
            biometric_data,
            "test_agent".to_string(),
        ).await.unwrap();
        
        // Get template
        let template = provider.get_template(&enrollment.template_id).await;
        assert!(template.is_ok());
        
        // List templates
        let templates = provider.list_templates("test_agent").await.unwrap();
        assert_eq!(templates.len(), 1);
        
        // Delete template
        let result = provider.delete_template(&enrollment.template_id).await;
        assert!(result.is_ok());
        
        // Verify deletion
        let templates = provider.list_templates("test_agent").await.unwrap();
        assert_eq!(templates.len(), 0);
    }

    #[test]
    fn test_biometric_type_support() {
        let provider = BiometricProvider::new("test".to_string());
        assert!(provider.supports_type(&BiometricType::Fingerprint));
        assert!(provider.supports_type(&BiometricType::FaceRecognition));
        assert!(!provider.supports_type(&BiometricType::Keystroke));
    }
}