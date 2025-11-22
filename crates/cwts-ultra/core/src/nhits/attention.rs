// Temporal attention mechanism for NHITS

use std::f32::consts::PI;

pub struct TemporalAttention {
    n_heads: usize,
    d_model: usize,
    d_k: usize,
    d_v: usize,
    positional_encoding: PositionalEncoding,
    
    // Weight matrices
    w_q: Vec<Vec<f32>>,
    w_k: Vec<Vec<f32>>,
    w_v: Vec<Vec<f32>>,
    w_o: Vec<Vec<f32>>,
}

pub struct PositionalEncoding {
    max_len: usize,
    d_model: usize,
    encoding_matrix: Vec<Vec<f32>>,
}

impl TemporalAttention {
    pub fn new(n_heads: usize, d_model: usize, d_k: usize, d_v: usize) -> Self {
        let scale = (1.0 / d_k as f32).sqrt();
        
        Self {
            n_heads,
            d_model,
            d_k,
            d_v,
            positional_encoding: PositionalEncoding::new(5000, d_model),
            w_q: Self::init_weights(d_model, n_heads * d_k, scale),
            w_k: Self::init_weights(d_model, n_heads * d_k, scale),
            w_v: Self::init_weights(d_model, n_heads * d_v, scale),
            w_o: Self::init_weights(n_heads * d_v, d_model, scale),
        }
    }
    
    fn init_weights(input_dim: usize, output_dim: usize, scale: f32) -> Vec<Vec<f32>> {
        let mut weights = vec![vec![0.0; input_dim]; output_dim];
        
        for row in weights.iter_mut() {
            for val in row.iter_mut() {
                *val = (rand::random::<f32>() - 0.5) * 2.0 * scale;
            }
        }
        
        weights
    }
    
    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        let seq_len = x.len() / self.d_model;
        let batch_size = 1; // Single batch for simplicity
        
        // Reshape input to [seq_len, d_model]
        let mut input_2d = vec![vec![0.0; self.d_model]; seq_len];
        for i in 0..seq_len {
            for j in 0..self.d_model {
                if i * self.d_model + j < x.len() {
                    input_2d[i][j] = x[i * self.d_model + j];
                }
            }
        }
        
        // Add positional encoding
        let encoded = self.positional_encoding.encode(&input_2d);
        
        // Compute Q, K, V
        let q = self.linear_transform(&encoded, &self.w_q);
        let k = self.linear_transform(&encoded, &self.w_k);
        let v = self.linear_transform(&encoded, &self.w_v);
        
        // Reshape for multi-head attention
        let q_heads = self.reshape_heads(&q, seq_len);
        let k_heads = self.reshape_heads(&k, seq_len);
        let v_heads = self.reshape_heads(&v, seq_len);
        
        // Compute attention for each head
        let mut attention_outputs = Vec::new();
        for h in 0..self.n_heads {
            let attention = self.scaled_dot_product_attention(
                &q_heads[h],
                &k_heads[h],
                &v_heads[h]
            );
            attention_outputs.push(attention);
        }
        
        // Concatenate heads
        let concatenated = self.concat_heads(&attention_outputs);
        
        // Final linear transformation
        let output = self.linear_transform(&concatenated, &self.w_o);
        
        // Flatten output
        output.into_iter().flatten().collect()
    }
    
    fn linear_transform(&self, x: &[Vec<f32>], weights: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let seq_len = x.len();
        let output_dim = weights.len();
        let mut output = vec![vec![0.0; output_dim]; seq_len];
        
        for t in 0..seq_len {
            for o in 0..output_dim {
                for i in 0..x[t].len().min(weights[o].len()) {
                    output[t][o] += x[t][i] * weights[o][i];
                }
            }
        }
        
        output
    }
    
    fn reshape_heads(&self, x: &[Vec<f32>], seq_len: usize) -> Vec<Vec<Vec<f32>>> {
        let mut heads = vec![vec![vec![0.0; self.d_k]; seq_len]; self.n_heads];
        
        for t in 0..seq_len {
            for h in 0..self.n_heads {
                for d in 0..self.d_k {
                    let idx = h * self.d_k + d;
                    if idx < x[t].len() {
                        heads[h][t][d] = x[t][idx];
                    }
                }
            }
        }
        
        heads
    }
    
    fn scaled_dot_product_attention(
        &self,
        q: &[Vec<f32>],
        k: &[Vec<f32>],
        v: &[Vec<f32>]
    ) -> Vec<Vec<f32>> {
        let seq_len = q.len();
        let scale = (self.d_k as f32).sqrt();
        
        // Compute attention scores
        let mut scores = vec![vec![0.0; seq_len]; seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut score = 0.0;
                for d in 0..self.d_k {
                    score += q[i][d] * k[j][d];
                }
                scores[i][j] = score / scale;
            }
        }
        
        // Apply softmax
        let attention_weights = self.softmax_2d(&scores);
        
        // Apply attention to values
        let mut output = vec![vec![0.0; self.d_v]; seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                for d in 0..self.d_v {
                    output[i][d] += attention_weights[i][j] * v[j][d];
                }
            }
        }
        
        output
    }
    
    fn softmax_2d(&self, scores: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut weights = vec![vec![0.0; scores[0].len()]; scores.len()];
        
        for i in 0..scores.len() {
            let max = scores[i].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut exp_sum = 0.0;
            
            for j in 0..scores[i].len() {
                weights[i][j] = (scores[i][j] - max).exp();
                exp_sum += weights[i][j];
            }
            
            for j in 0..scores[i].len() {
                weights[i][j] /= exp_sum;
            }
        }
        
        weights
    }
    
    fn concat_heads(&self, heads: &[Vec<Vec<f32>>]) -> Vec<Vec<f32>> {
        let seq_len = heads[0].len();
        let total_dim = self.n_heads * self.d_v;
        let mut concatenated = vec![vec![0.0; total_dim]; seq_len];
        
        for t in 0..seq_len {
            for h in 0..self.n_heads {
                for d in 0..self.d_v {
                    concatenated[t][h * self.d_v + d] = heads[h][t][d];
                }
            }
        }
        
        concatenated
    }
}

impl PositionalEncoding {
    pub fn new(max_len: usize, d_model: usize) -> Self {
        let mut encoding_matrix = vec![vec![0.0; d_model]; max_len];
        
        for pos in 0..max_len {
            for i in 0..d_model {
                let angle = if i % 2 == 0 {
                    pos as f32 / 10000_f32.powf(2.0 * (i / 2) as f32 / d_model as f32)
                } else {
                    pos as f32 / 10000_f32.powf(2.0 * ((i - 1) / 2) as f32 / d_model as f32)
                };
                
                encoding_matrix[pos][i] = if i % 2 == 0 {
                    angle.sin()
                } else {
                    angle.cos()
                };
            }
        }
        
        Self {
            max_len,
            d_model,
            encoding_matrix,
        }
    }
    
    pub fn encode(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let seq_len = x.len();
        let mut encoded = x.to_vec();
        
        for t in 0..seq_len.min(self.max_len) {
            for d in 0..self.d_model.min(x[t].len()) {
                encoded[t][d] += self.encoding_matrix[t][d];
            }
        }
        
        encoded
    }
}

// Temporal convolution for local pattern extraction
pub struct TemporalConvolution {
    kernel_size: usize,
    stride: usize,
    padding: usize,
    filters: Vec<ConvFilter>,
}

struct ConvFilter {
    weights: Vec<f32>,
    bias: f32,
}

impl TemporalConvolution {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        let mut filters = Vec::with_capacity(out_channels);
        let scale = (2.0 / (in_channels * kernel_size) as f32).sqrt();
        
        for _ in 0..out_channels {
            let mut weights = vec![0.0; in_channels * kernel_size];
            for w in weights.iter_mut() {
                *w = (rand::random::<f32>() - 0.5) * 2.0 * scale;
            }
            
            filters.push(ConvFilter {
                weights,
                bias: 0.0,
            });
        }
        
        Self {
            kernel_size,
            stride: 1,
            padding: kernel_size / 2,
            filters,
        }
    }
    
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let input_len = x.len();
        let padded = self.apply_padding(x);
        let output_len = (padded.len() - self.kernel_size) / self.stride + 1;
        let mut output = Vec::with_capacity(output_len * self.filters.len());
        
        for filter in &self.filters {
            for i in (0..padded.len() - self.kernel_size + 1).step_by(self.stride) {
                let mut conv_sum = filter.bias;
                
                for j in 0..self.kernel_size {
                    conv_sum += padded[i + j] * filter.weights[j];
                }
                
                output.push(conv_sum.max(0.0)); // ReLU activation
            }
        }
        
        output
    }
    
    fn apply_padding(&self, x: &[f32]) -> Vec<f32> {
        let mut padded = Vec::with_capacity(x.len() + 2 * self.padding);
        
        // Replicate padding
        for _ in 0..self.padding {
            padded.push(x[0]);
        }
        
        padded.extend_from_slice(x);
        
        for _ in 0..self.padding {
            padded.push(x[x.len() - 1]);
        }
        
        padded
    }
}

use rand;