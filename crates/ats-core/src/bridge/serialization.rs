//! High-Performance Serialization Handler
//!
//! Optimized serialization/deserialization with SIMD acceleration,
//! zero-copy operations, and multiple format support.

use crate::{AtsCoreError, Result};
use crate::bridge::BridgeConfig;
use serde::{Deserialize, Serialize};
use std::{
    mem,
    ptr,
    slice,
};

/// High-performance serialization handler
pub struct SerializationHandler {
    /// Configuration
    config: BridgeConfig,
    /// Buffer pool for reuse
    buffer_pool: Vec<Vec<u8>>,
    /// SIMD-optimized operations
    simd_enabled: bool,
}

impl SerializationHandler {
    /// Create new serialization handler
    pub fn new(config: &BridgeConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            buffer_pool: Vec::with_capacity(16),
            simd_enabled: config.simd_enabled,
        })
    }

    /// Serialize to binary format with optimal performance
    pub fn serialize_binary<T>(&self, value: &T) -> Result<Vec<u8>>
    where
        T: Serialize,
    {
        // Use efficient binary serialization
        bincode::serialize(value)
            .map_err(|e| AtsCoreError::ValidationFailed(format!("Binary serialization failed: {}", e)))
    }

    /// Deserialize from binary format
    pub fn deserialize_binary<T>(&self, data: &[u8]) -> Result<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        bincode::deserialize(data)
            .map_err(|e| AtsCoreError::ValidationFailed(format!("Binary deserialization failed: {}", e)))
    }

    /// Zero-copy serialization for compatible types
    pub fn serialize_zero_copy<T>(&self, value: &T) -> Result<&[u8]>
    where
        T: Copy + Send + Sync,
    {
        if !self.config.zero_copy_enabled {
            return Err(AtsCoreError::ComputationFailed("Zero-copy disabled".to_string()));
        }

        // Safety: T is Copy and we're returning a reference to the same data
        let bytes = unsafe {
            slice::from_raw_parts(
                value as *const T as *const u8,
                mem::size_of::<T>()
            )
        };
        Ok(bytes)
    }

    /// Zero-copy deserialization for compatible types
    pub fn deserialize_zero_copy<T>(data: &[u8]) -> Result<&T>
    where
        T: Copy + Send + Sync,
    {
        if data.len() != mem::size_of::<T>() {
            return Err(AtsCoreError::ValidationFailed("Invalid data size for zero-copy".to_string()));
        }

        // Safety: We've validated the size matches T
        let value = unsafe { &*(data.as_ptr() as *const T) };
        Ok(value)
    }

    /// SIMD-accelerated byte operations
    pub fn simd_copy(&self, src: &[u8], dst: &mut [u8]) -> Result<()> {
        if src.len() != dst.len() {
            return Err(AtsCoreError::ValidationFailed("Source and destination length mismatch".to_string()));
        }

        if self.simd_enabled && src.len() >= 32 {
            // Use SIMD for large copies
            self.simd_copy_aligned(src, dst)
        } else {
            // Fallback to standard copy
            dst.copy_from_slice(src);
            Ok(())
        }
    }

    /// SIMD-aligned memory copy
    #[cfg(target_arch = "x86_64")]
    fn simd_copy_aligned(&self, src: &[u8], dst: &mut [u8]) -> Result<()> {
        use std::arch::x86_64::*;

        if src.len() != dst.len() {
            return Err(AtsCoreError::ValidationFailed("Length mismatch".to_string()));
        }

        let len = src.len();
        let mut i = 0;

        // Process 32-byte chunks with AVX2
        if is_x86_feature_detected!("avx2") {
            unsafe {
                while i + 32 <= len {
                    let src_chunk = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
                    _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, src_chunk);
                    i += 32;
                }
            }
        }

        // Process remaining bytes
        dst[i..].copy_from_slice(&src[i..]);
        Ok(())
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn simd_copy_aligned(&self, src: &[u8], dst: &mut [u8]) -> Result<()> {
        // Fallback for non-x86_64 architectures
        dst.copy_from_slice(src);
        Ok(())
    }

    /// Compress data if beneficial
    pub fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        if !self.config.compression_enabled || data.len() < 1024 {
            // Don't compress small data
            return Ok(data.to_vec());
        }

        // Use fast compression algorithm
        use flate2::{Compression, write::GzEncoder};
        use std::io::Write;

        let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(data)
            .map_err(|e| AtsCoreError::ComputationFailed(format!("Compression failed: {}", e)))?;
        
        let compressed = encoder.finish()
            .map_err(|e| AtsCoreError::ComputationFailed(format!("Compression finalization failed: {}", e)))?;

        // Only return compressed data if it's significantly smaller
        if compressed.len() < data.len() * 9 / 10 {
            Ok(compressed)
        } else {
            Ok(data.to_vec())
        }
    }

    /// Decompress data
    pub fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        use flate2::read::GzDecoder;
        use std::io::Read;

        let mut decoder = GzDecoder::new(data);
        let mut decompressed = Vec::new();
        
        decoder.read_to_end(&mut decompressed)
            .map_err(|e| AtsCoreError::ComputationFailed(format!("Decompression failed: {}", e)))?;

        Ok(decompressed)
    }

    /// Get or create buffer from pool
    pub fn get_buffer(&mut self, min_size: usize) -> Vec<u8> {
        // Try to reuse an existing buffer
        if let Some(mut buffer) = self.buffer_pool.pop() {
            if buffer.capacity() >= min_size {
                buffer.clear();
                return buffer;
            }
        }

        // Create new buffer
        Vec::with_capacity(min_size.max(4096))
    }

    /// Return buffer to pool
    pub fn return_buffer(&mut self, buffer: Vec<u8>) {
        if self.buffer_pool.len() < 16 && buffer.capacity() > 1024 {
            self.buffer_pool.push(buffer);
        }
    }

    /// Serialize with buffer reuse
    pub fn serialize_with_buffer<T>(&mut self, value: &T) -> Result<Vec<u8>>
    where
        T: Serialize,
    {
        let mut buffer = self.get_buffer(1024);
        
        // Serialize directly into buffer
        let cursor = std::io::Cursor::new(&mut buffer);
        bincode::serialize_into(cursor, value)
            .map_err(|e| AtsCoreError::ValidationFailed(format!("Serialization failed: {}", e)))?;

        Ok(buffer)
    }

    /// Batch serialization for multiple items
    pub fn serialize_batch<T>(&mut self, items: &[T]) -> Result<Vec<u8>>
    where
        T: Serialize,
    {
        let estimated_size = items.len() * 256; // Rough estimate
        let mut buffer = self.get_buffer(estimated_size);
        
        // Serialize item count first
        let item_count = items.len() as u32;
        buffer.extend_from_slice(&item_count.to_le_bytes());

        // Serialize each item with length prefix
        for item in items {
            let item_data = bincode::serialize(item)
                .map_err(|e| AtsCoreError::ValidationFailed(format!("Item serialization failed: {}", e)))?;
            
            let item_len = item_data.len() as u32;
            buffer.extend_from_slice(&item_len.to_le_bytes());
            buffer.extend_from_slice(&item_data);
        }

        Ok(buffer)
    }

    /// Batch deserialization
    pub fn deserialize_batch<T>(&self, data: &[u8]) -> Result<Vec<T>>
    where
        T: for<'de> Deserialize<'de>,
    {
        if data.len() < 4 {
            return Err(AtsCoreError::ValidationFailed("Invalid batch data".to_string()));
        }

        let mut cursor = 0;
        
        // Read item count
        let item_count = u32::from_le_bytes([
            data[cursor], data[cursor + 1], data[cursor + 2], data[cursor + 3]
        ]) as usize;
        cursor += 4;

        let mut items = Vec::with_capacity(item_count);

        // Deserialize each item
        for _ in 0..item_count {
            if cursor + 4 > data.len() {
                return Err(AtsCoreError::ValidationFailed("Truncated batch data".to_string()));
            }

            // Read item length
            let item_len = u32::from_le_bytes([
                data[cursor], data[cursor + 1], data[cursor + 2], data[cursor + 3]
            ]) as usize;
            cursor += 4;

            if cursor + item_len > data.len() {
                return Err(AtsCoreError::ValidationFailed("Invalid item length in batch".to_string()));
            }

            // Deserialize item
            let item_data = &data[cursor..cursor + item_len];
            let item: T = bincode::deserialize(item_data)
                .map_err(|e| AtsCoreError::ValidationFailed(format!("Item deserialization failed: {}", e)))?;
            
            items.push(item);
            cursor += item_len;
        }

        Ok(items)
    }

    /// Memory-mapped serialization for large data
    pub fn serialize_memory_mapped<T>(&self, value: &T, file_path: &str) -> Result<()>
    where
        T: Serialize,
    {
        use std::fs::OpenOptions;
        use std::io::Write;

        let serialized = bincode::serialize(value)
            .map_err(|e| AtsCoreError::ValidationFailed(format!("Serialization failed: {}", e)))?;

        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(file_path)
            .map_err(|e| AtsCoreError::IntegrationError(format!("Failed to open file: {}", e)))?;

        file.write_all(&serialized)
            .map_err(|e| AtsCoreError::IntegrationError(format!("Failed to write data: {}", e)))?;

        Ok(())
    }

    /// Memory-mapped deserialization
    pub fn deserialize_memory_mapped<T>(&self, file_path: &str) -> Result<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        use memmap2::Mmap;
        use std::fs::File;

        let file = File::open(file_path)
            .map_err(|e| AtsCoreError::IntegrationError(format!("Failed to open file: {}", e)))?;

        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| AtsCoreError::IntegrationError(format!("Failed to map file: {}", e)))?;

        bincode::deserialize(&mmap[..])
            .map_err(|e| AtsCoreError::ValidationFailed(format!("Deserialization failed: {}", e)))
    }

    /// Get serialization statistics
    pub fn get_statistics(&self) -> SerializationStats {
        SerializationStats {
            buffer_pool_size: self.buffer_pool.len(),
            buffer_pool_capacity: self.buffer_pool.capacity(),
            simd_enabled: self.simd_enabled,
            compression_enabled: self.config.compression_enabled,
            zero_copy_enabled: self.config.zero_copy_enabled,
        }
    }
}

/// Serialization statistics
#[derive(Debug, Clone)]
pub struct SerializationStats {
    pub buffer_pool_size: usize,
    pub buffer_pool_capacity: usize,
    pub simd_enabled: bool,
    pub compression_enabled: bool,
    pub zero_copy_enabled: bool,
}