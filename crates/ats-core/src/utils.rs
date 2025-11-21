//! Utility functions for ATS-Core operations


/// Mathematical utility functions
pub mod math {
    
    /// Fast inverse square root approximation
    pub fn fast_inv_sqrt(x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        
        // Use Rust's built-in implementation for accuracy
        1.0 / x.sqrt()
    }
    
    /// Linear interpolation
    pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
        a + t * (b - a)
    }
    
    /// Clamps a value between min and max
    pub fn clamp(value: f64, min: f64, max: f64) -> f64 {
        if value < min {
            min
        } else if value > max {
            max
        } else {
            value
        }
    }
}

/// Memory utility functions
pub mod memory {
    
    /// Checks if a pointer is aligned to the specified boundary
    pub fn is_aligned(ptr: *const u8, alignment: usize) -> bool {
        (ptr as usize) % alignment == 0
    }
    
    /// Rounds up to the next alignment boundary
    pub fn align_up(size: usize, alignment: usize) -> usize {
        (size + alignment - 1) & !(alignment - 1)
    }
}

/// Validation utility functions
pub mod validation {
    use crate::error::AtsCoreError;
    
    /// Validates array dimensions match
    pub fn validate_dimensions(a_len: usize, b_len: usize, _operation: &str) -> crate::error::Result<()> {
        if a_len != b_len {
            return Err(AtsCoreError::dimension_mismatch(b_len, a_len));
        }
        Ok(())
    }
    
    /// Validates array is not empty
    pub fn validate_not_empty(data: &[f64], field_name: &str) -> crate::error::Result<()> {
        if data.is_empty() {
            return Err(AtsCoreError::validation(field_name, "cannot be empty"));
        }
        Ok(())
    }
    
    /// Validates all values are finite
    pub fn validate_finite(data: &[f64], field_name: &str) -> crate::error::Result<()> {
        for (i, &value) in data.iter().enumerate() {
            if !value.is_finite() {
                return Err(AtsCoreError::validation(
                    field_name,
                    &format!("non-finite value at index {}: {}", i, value),
                ));
            }
        }
        Ok(())
    }
    
    /// Validates all values are positive
    pub fn validate_positive(data: &[f64], field_name: &str) -> crate::error::Result<()> {
        for (i, &value) in data.iter().enumerate() {
            if value <= 0.0 {
                return Err(AtsCoreError::validation(
                    field_name,
                    &format!("non-positive value at index {}: {}", i, value),
                ));
            }
        }
        Ok(())
    }
}