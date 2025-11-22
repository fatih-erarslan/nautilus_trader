use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct MathematicalProof {
    pub theorem: String,
    pub proof_steps: Vec<String>,
    pub verified: bool,
}

pub struct ProofSystem {
    proofs: HashMap<String, MathematicalProof>,
}

impl ProofSystem {
    pub fn new() -> Self {
        Self {
            proofs: HashMap::new(),
        }
    }
    
    pub fn verify_proof(&self, theorem: &str) -> bool {
        self.proofs.get(theorem)
            .map(|p| p.verified)
            .unwrap_or(false)
    }
    
    pub fn add_proof(&mut self, theorem: String, proof: MathematicalProof) {
        self.proofs.insert(theorem, proof);
    }
    
    pub fn validate_convergence(&self, epsilon: f64) -> bool {
        epsilon > 0.0 && epsilon < 1.0
    }
    
    pub fn verify_stability(&self, eigenvalues: &[f64]) -> bool {
        eigenvalues.iter().all(|&e| e.abs() < 1.0)
    }
}

impl Default for ProofSystem {
    fn default() -> Self {
        Self::new()
    }
}