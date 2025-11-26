//! Model serialization utilities (SafeTensors format)

use crate::error::{MlError, MlResult};
use crate::tensor::{Tensor, TensorOps, DType};
use crate::backends::Device;
use std::collections::HashMap;
use std::path::Path;
use std::io::{Read, Write};

/// Tensor metadata for serialization
#[derive(Debug, Clone)]
pub struct TensorMeta {
    /// Tensor name/key
    pub name: String,
    /// Shape
    pub shape: Vec<usize>,
    /// Data type string
    pub dtype: String,
    /// Byte offset in data section
    pub offset: usize,
    /// Byte length
    pub length: usize,
}

/// SafeTensors-compatible serialization format
///
/// Format:
/// - 8 bytes: header size (little-endian u64)
/// - N bytes: JSON header with tensor metadata
/// - Data section: raw tensor bytes
#[derive(Debug)]
pub struct SafeTensorsFile {
    /// Tensor metadata
    pub tensors: HashMap<String, TensorMeta>,
    /// Raw data
    pub data: Vec<u8>,
}

impl SafeTensorsFile {
    /// Create a new empty SafeTensors file
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            data: Vec::new(),
        }
    }

    /// Add a tensor to the file
    pub fn add_tensor(&mut self, name: &str, tensor: &Tensor) -> MlResult<()> {
        let shape = tensor.shape().dims().to_vec();
        let dtype = match tensor.dtype() {
            DType::F32 => "F32",
            DType::F64 => "F64",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
            DType::I8 => "I8",
            DType::I32 => "I32",
            DType::I64 => "I64",
        };

        // Get tensor data
        #[cfg(feature = "cpu")]
        {
            if let Some(data) = tensor.as_slice() {
                let offset = self.data.len();
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr() as *const u8,
                        data.len() * std::mem::size_of::<f32>(),
                    )
                };
                self.data.extend_from_slice(bytes);

                let meta = TensorMeta {
                    name: name.to_string(),
                    shape,
                    dtype: dtype.to_string(),
                    offset,
                    length: bytes.len(),
                };

                self.tensors.insert(name.to_string(), meta);
                return Ok(());
            }
        }

        Err(MlError::SerializationError(
            "Cannot serialize tensor from non-CPU device".to_string()
        ))
    }

    /// Save to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> MlResult<()> {
        let mut file = std::fs::File::create(path)
            .map_err(MlError::IoError)?;

        // Build header JSON
        let header = self.build_header();
        let header_bytes = header.as_bytes();
        let header_size = header_bytes.len() as u64;

        // Write header size
        file.write_all(&header_size.to_le_bytes())
            .map_err(MlError::IoError)?;

        // Write header
        file.write_all(header_bytes)
            .map_err(MlError::IoError)?;

        // Write data
        file.write_all(&self.data)
            .map_err(MlError::IoError)?;

        Ok(())
    }

    /// Load from file
    pub fn load<P: AsRef<Path>>(path: P) -> MlResult<Self> {
        let mut file = std::fs::File::open(path)
            .map_err(MlError::IoError)?;

        // Read header size
        let mut header_size_bytes = [0u8; 8];
        file.read_exact(&mut header_size_bytes)
            .map_err(MlError::IoError)?;
        let header_size = u64::from_le_bytes(header_size_bytes) as usize;

        // Read header
        let mut header_bytes = vec![0u8; header_size];
        file.read_exact(&mut header_bytes)
            .map_err(MlError::IoError)?;
        let header = String::from_utf8(header_bytes)
            .map_err(|e| MlError::SerializationError(e.to_string()))?;

        // Parse header
        let tensors = parse_header(&header)?;

        // Read data
        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(MlError::IoError)?;

        Ok(Self { tensors, data })
    }

    /// Get tensor by name
    pub fn get_tensor(&self, name: &str, device: &Device) -> MlResult<Tensor> {
        let meta = self.tensors.get(name)
            .ok_or_else(|| MlError::SerializationError(
                format!("Tensor '{}' not found", name)
            ))?;

        // Extract bytes
        let bytes = &self.data[meta.offset..meta.offset + meta.length];

        // Convert to f32 slice (assuming F32 dtype)
        let data: &[f32] = unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const f32,
                meta.length / std::mem::size_of::<f32>(),
            )
        };

        Tensor::from_slice(data, meta.shape.clone(), device)
    }

    /// List all tensor names
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    fn build_header(&self) -> String {
        // Build simple JSON header
        let mut parts = Vec::new();

        for (name, meta) in &self.tensors {
            let tensor_json = format!(
                "\"{}\": {{\"dtype\": \"{}\", \"shape\": {:?}, \"data_offsets\": [{}, {}]}}",
                name,
                meta.dtype,
                meta.shape,
                meta.offset,
                meta.offset + meta.length
            );
            parts.push(tensor_json);
        }

        format!("{{{}}}", parts.join(", "))
    }
}

impl Default for SafeTensorsFile {
    fn default() -> Self {
        Self::new()
    }
}

fn parse_header(header: &str) -> MlResult<HashMap<String, TensorMeta>> {
    let mut tensors = HashMap::new();

    // Simple JSON parsing (would use serde_json in production)
    let trimmed = header.trim();
    if !trimmed.starts_with('{') || !trimmed.ends_with('}') {
        return Err(MlError::SerializationError("Invalid header format".to_string()));
    }

    let content = &trimmed[1..trimmed.len()-1];

    // Parse each tensor entry
    for part in content.split("},") {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }

        // Find tensor name
        let name_end = part.find(':').ok_or_else(|| {
            MlError::SerializationError("Invalid tensor entry".to_string())
        })?;
        let name = part[1..name_end-1].to_string();  // Remove quotes

        // Extract dtype
        let dtype_start = part.find("\"dtype\":")
            .map(|i| i + 10)
            .unwrap_or(0);
        let dtype_end = part[dtype_start..].find('"')
            .map(|i| dtype_start + i)
            .unwrap_or(dtype_start);
        let dtype = part[dtype_start..dtype_end].to_string();

        // Extract shape (simplified)
        let shape_start = part.find("\"shape\":")
            .map(|i| i + 9)
            .unwrap_or(0);
        let shape_end = part[shape_start..].find(']')
            .map(|i| shape_start + i + 1)
            .unwrap_or(shape_start);
        let shape_str = &part[shape_start..shape_end];
        let shape: Vec<usize> = shape_str
            .trim_matches(|c| c == '[' || c == ']')
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();

        // Extract offsets
        let offsets_start = part.find("\"data_offsets\":")
            .map(|i| i + 16)
            .unwrap_or(0);
        let offsets_str = &part[offsets_start..];
        let offset: usize = offsets_str
            .trim_matches(|c| c == '[' || c == ']' || c == '}')
            .split(',')
            .next()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);
        let end_offset: usize = offsets_str
            .trim_matches(|c| c == '[' || c == ']' || c == '}')
            .split(',')
            .nth(1)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);

        let meta = TensorMeta {
            name: name.clone(),
            shape,
            dtype,
            offset,
            length: end_offset.saturating_sub(offset),
        };

        tensors.insert(name, meta);
    }

    Ok(tensors)
}

/// Save model state dict to SafeTensors file
pub fn save_state_dict<P: AsRef<Path>>(
    state_dict: &HashMap<String, Tensor>,
    path: P,
) -> MlResult<()> {
    let mut file = SafeTensorsFile::new();

    for (name, tensor) in state_dict {
        file.add_tensor(name, tensor)?;
    }

    file.save(path)
}

/// Load model state dict from SafeTensors file
pub fn load_state_dict<P: AsRef<Path>>(
    path: P,
    device: &Device,
) -> MlResult<HashMap<String, Tensor>> {
    let file = SafeTensorsFile::load(path)?;
    let mut state_dict = HashMap::new();

    for name in file.tensor_names() {
        let tensor = file.get_tensor(name, device)?;
        state_dict.insert(name.to_string(), tensor);
    }

    Ok(state_dict)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::DType;

    #[test]
    fn test_safetensors_roundtrip() {
        let device = Device::Cpu;

        // Create test tensors
        let t1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2], &device).unwrap();
        let t2 = Tensor::from_slice(&[5.0, 6.0, 7.0], vec![3], &device).unwrap();

        // Create file
        let mut file = SafeTensorsFile::new();
        file.add_tensor("weight", &t1).unwrap();
        file.add_tensor("bias", &t2).unwrap();

        // Save
        let path = "/tmp/test_safetensors.safetensors";
        file.save(path).unwrap();

        // Load
        let loaded = SafeTensorsFile::load(path).unwrap();
        assert_eq!(loaded.tensor_names().len(), 2);

        // Verify tensor
        let w = loaded.get_tensor("weight", &device).unwrap();
        assert_eq!(w.shape().dims(), &[2, 2]);
    }
}
