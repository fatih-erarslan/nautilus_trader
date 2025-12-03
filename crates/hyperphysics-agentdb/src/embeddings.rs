//! Embedding generation for AgentDB
//!
//! Supports both local models (sentence-transformers via candle) and API-based models.

use std::sync::Arc;

/// Embedding model trait
pub trait EmbeddingModel: Send + Sync {
    /// Generate embedding for text
    fn embed(&self, text: &str) -> Vec<f32>;
    
    /// Generate embeddings for batch of texts
    fn embed_batch(&self, texts: &[&str]) -> Vec<Vec<f32>>;
    
    /// Get embedding dimension
    fn dimension(&self) -> usize;
}

/// Simple hash-based embedding for testing/fallback
/// (No external dependencies, deterministic)
#[derive(Debug, Clone)]
pub struct HashEmbedding {
    dimension: usize,
}

impl HashEmbedding {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl EmbeddingModel for HashEmbedding {
    fn embed(&self, text: &str) -> Vec<f32> {
        let mut embedding = vec![0.0f32; self.dimension];
        let bytes = text.as_bytes();
        
        // Use multiple hash functions for better distribution
        for (i, byte) in bytes.iter().enumerate() {
            // Position-dependent hashing
            let idx1 = (i * 31 + *byte as usize) % self.dimension;
            let idx2 = (i * 37 + *byte as usize * 7) % self.dimension;
            let idx3 = (i * 41 + *byte as usize * 11) % self.dimension;
            
            embedding[idx1] += (*byte as f32) / 255.0;
            embedding[idx2] += (*byte as f32) / 512.0;
            embedding[idx3] -= (*byte as f32) / 768.0;
        }
        
        // L2 normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }
        
        embedding
    }
    
    fn embed_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
    
    fn dimension(&self) -> usize {
        self.dimension
    }
}

/// TF-IDF based embedding for lightweight semantic search
#[derive(Debug)]
pub struct TfIdfEmbedding {
    dimension: usize,
    vocabulary: std::collections::HashMap<String, usize>,
    idf: Vec<f32>,
    doc_count: usize,
}

impl TfIdfEmbedding {
    /// Create new TF-IDF embedding with given dimension
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            vocabulary: std::collections::HashMap::new(),
            idf: vec![0.0; dimension],
            doc_count: 0,
        }
    }
    
    /// Update vocabulary and IDF from new document
    pub fn update(&mut self, text: &str) {
        self.doc_count += 1;
        let tokens = Self::tokenize(text);
        let unique_tokens: std::collections::HashSet<_> = tokens.iter().collect();
        
        for token in unique_tokens {
            let idx = self.get_or_create_idx(token);
            if idx < self.dimension {
                // Update document frequency for IDF
                self.idf[idx] += 1.0;
            }
        }
    }
    
    /// Finalize IDF calculation
    pub fn finalize(&mut self) {
        let n = self.doc_count as f32;
        for idf in &mut self.idf {
            if *idf > 0.0 {
                *idf = (n / *idf).ln() + 1.0;
            }
        }
    }
    
    fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| s.len() > 2)
            .map(|s| s.to_string())
            .collect()
    }
    
    fn get_or_create_idx(&mut self, token: &str) -> usize {
        if let Some(&idx) = self.vocabulary.get(token) {
            idx
        } else {
            // Hash to get deterministic index
            let hash = Self::hash_token(token);
            let idx = hash % self.dimension;
            self.vocabulary.insert(token.to_string(), idx);
            idx
        }
    }
    
    fn hash_token(token: &str) -> usize {
        let mut hash: usize = 5381;
        for byte in token.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(byte as usize);
        }
        hash
    }
}

impl EmbeddingModel for TfIdfEmbedding {
    fn embed(&self, text: &str) -> Vec<f32> {
        let mut embedding = vec![0.0f32; self.dimension];
        let tokens = Self::tokenize(text);
        
        // Calculate term frequencies
        let mut tf = std::collections::HashMap::new();
        for token in &tokens {
            *tf.entry(token.as_str()).or_insert(0) += 1;
        }
        
        // Calculate TF-IDF
        let max_tf = tf.values().copied().max().unwrap_or(1) as f32;
        for (token, count) in tf {
            if let Some(&idx) = self.vocabulary.get(token) {
                if idx < self.dimension {
                    let normalized_tf = (count as f32) / max_tf;
                    embedding[idx] = normalized_tf * self.idf[idx];
                }
            } else {
                // Unknown token - use hash
                let idx = Self::hash_token(token) % self.dimension;
                embedding[idx] += 0.1; // Small contribution for OOV
            }
        }
        
        // L2 normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }
        
        embedding
    }
    
    fn embed_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
    
    fn dimension(&self) -> usize {
        self.dimension
    }
}

// ============ Local Model Support (behind feature flag) ============

#[cfg(feature = "local-embeddings")]
pub mod local {
    use super::*;
    use candle_core::{Device, Tensor};
    use candle_nn::VarBuilder;
    use candle_transformers::models::bert::{BertModel, Config};
    use tokenizers::Tokenizer;
    use std::path::Path;
    
    /// Sentence transformer embedding model (local, no API calls)
    pub struct SentenceTransformer {
        model: BertModel,
        tokenizer: Tokenizer,
        device: Device,
        dimension: usize,
    }
    
    impl SentenceTransformer {
        /// Load from HuggingFace Hub
        pub fn from_pretrained(model_id: &str) -> anyhow::Result<Self> {
            let device = Device::Cpu; // Use GPU if available
            
            // Download model files
            let api = hf_hub::api::sync::Api::new()?;
            let repo = api.model(model_id.to_string());
            
            let config_path = repo.get("config.json")?;
            let tokenizer_path = repo.get("tokenizer.json")?;
            let weights_path = repo.get("model.safetensors")?;
            
            // Load config
            let config: Config = serde_json::from_reader(std::fs::File::open(&config_path)?)?;
            let dimension = config.hidden_size;
            
            // Load tokenizer
            let tokenizer = Tokenizer::from_file(&tokenizer_path)?;
            
            // Load model weights
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], candle_core::DType::F32, &device)?
            };
            let model = BertModel::load(vb, &config)?;
            
            Ok(Self {
                model,
                tokenizer,
                device,
                dimension,
            })
        }
        
        fn mean_pooling(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> anyhow::Result<Tensor> {
            let mask_expanded = attention_mask.unsqueeze(2)?.broadcast_as(hidden_states.shape())?;
            let sum_hidden = (hidden_states * &mask_expanded)?.sum(1)?;
            let sum_mask = mask_expanded.to_dtype(candle_core::DType::F32)?.sum(1)?.clamp(1e-9, f64::INFINITY)?;
            Ok(sum_hidden.broadcast_div(&sum_mask)?)
        }
    }
    
    impl EmbeddingModel for SentenceTransformer {
        fn embed(&self, text: &str) -> Vec<f32> {
            self.embed_batch(&[text]).pop().unwrap_or_default()
        }
        
        fn embed_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
            // Tokenize
            let encodings = self.tokenizer.encode_batch(texts.to_vec(), true).unwrap();
            
            let mut all_input_ids = Vec::new();
            let mut all_attention_masks = Vec::new();
            
            for encoding in &encodings {
                all_input_ids.push(encoding.get_ids().to_vec());
                all_attention_masks.push(encoding.get_attention_mask().to_vec());
            }
            
            // Pad to max length
            let max_len = all_input_ids.iter().map(|v| v.len()).max().unwrap_or(0);
            for ids in &mut all_input_ids {
                ids.resize(max_len, 0);
            }
            for mask in &mut all_attention_masks {
                mask.resize(max_len, 0);
            }
            
            // Convert to tensors
            let input_ids: Vec<u32> = all_input_ids.into_iter().flatten().collect();
            let attention_mask: Vec<u32> = all_attention_masks.into_iter().flatten().collect();
            
            let batch_size = texts.len();
            let input_ids = Tensor::from_vec(input_ids, (batch_size, max_len), &self.device).unwrap();
            let attention_mask = Tensor::from_vec(attention_mask, (batch_size, max_len), &self.device).unwrap();
            let token_type_ids = Tensor::zeros((batch_size, max_len), candle_core::DType::U32, &self.device).unwrap();
            
            // Forward pass
            let hidden_states = self.model.forward(&input_ids, &token_type_ids, Some(&attention_mask)).unwrap();
            
            // Mean pooling
            let embeddings = self.mean_pooling(&hidden_states, &attention_mask).unwrap();
            
            // Normalize and convert to Vec<Vec<f32>>
            let embeddings = embeddings.to_vec2::<f32>().unwrap();
            embeddings.into_iter().map(|e| {
                let norm: f32 = e.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    e.iter().map(|x| x / norm).collect()
                } else {
                    e
                }
            }).collect()
        }
        
        fn dimension(&self) -> usize {
            self.dimension
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hash_embedding() {
        let model = HashEmbedding::new(128);
        
        let emb1 = model.embed("bitcoin momentum breakout");
        let emb2 = model.embed("bitcoin momentum breakout");
        let emb3 = model.embed("ethereum mean reversion");
        
        // Same input should give same output
        assert_eq!(emb1, emb2);
        
        // Different input should give different output
        assert_ne!(emb1, emb3);
        
        // Should be normalized
        let norm: f32 = emb1.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }
    
    #[test]
    fn test_tfidf_embedding() {
        let mut model = TfIdfEmbedding::new(128);
        
        // Train on corpus
        model.update("bitcoin price momentum trading");
        model.update("ethereum defi yield farming");
        model.update("bitcoin lightning network");
        model.finalize();
        
        let emb1 = model.embed("bitcoin trading");
        let emb2 = model.embed("ethereum farming");
        
        // Should be normalized
        let norm: f32 = emb1.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
        
        // Different topics should have different embeddings
        assert_ne!(emb1, emb2);
    }
}
