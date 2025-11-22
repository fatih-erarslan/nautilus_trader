//! Dilithium module - Post-quantum signatures via NAPI

use napi::bindgen_prelude::*;
use napi_derive::napi;
use crate::SignatureResult;

/// Sign a message synchronously
pub fn sign_message_sync(message: &Buffer, security_level: &str) -> Result<SignatureResult> {
    use hyperphysics_dilithium::{DilithiumKeypair, SecurityLevel};

    let level = match security_level.to_lowercase().as_str() {
        "standard" | "ml-dsa-44" => SecurityLevel::Standard,
        "high" | "ml-dsa-65" => SecurityLevel::High,
        "maximum" | "ml-dsa-87" => SecurityLevel::Maximum,
        _ => return Err(Error::new(Status::InvalidArg, format!("Unknown security level: {}", security_level))),
    };

    let keypair = DilithiumKeypair::generate(level)
        .map_err(|e| Error::new(Status::GenericFailure, format!("Key generation failed: {:?}", e)))?;

    let signature = keypair.sign(message.as_ref())
        .map_err(|e| Error::new(Status::GenericFailure, format!("Signing failed: {:?}", e)))?;

    let sig_bytes = signature.as_bytes();

    Ok(SignatureResult {
        signature: Buffer::from(sig_bytes.to_vec()),
        public_key: Buffer::from(keypair.public_key_bytes().to_vec()),
        security_level: security_level.to_string(),
        size: sig_bytes.len() as u32,
    })
}

/// Dilithium Keypair for post-quantum digital signatures
#[napi]
pub struct DilithiumKeypair {
    keypair: hyperphysics_dilithium::DilithiumKeypair,
}

#[napi]
impl DilithiumKeypair {
    #[napi(factory)]
    pub fn generate(security_level: Option<String>) -> Result<Self> {
        use hyperphysics_dilithium::SecurityLevel;

        let level_str = security_level.unwrap_or_else(|| "standard".to_string());
        let level = match level_str.to_lowercase().as_str() {
            "standard" | "ml-dsa-44" => SecurityLevel::Standard,
            "high" | "ml-dsa-65" => SecurityLevel::High,
            "maximum" | "ml-dsa-87" => SecurityLevel::Maximum,
            _ => return Err(Error::new(Status::InvalidArg, format!("Unknown level: {}", level_str))),
        };

        let keypair = hyperphysics_dilithium::DilithiumKeypair::generate(level)
            .map_err(|e| Error::new(Status::GenericFailure, format!("Key gen failed: {:?}", e)))?;

        Ok(Self { keypair })
    }

    #[napi]
    pub fn sign(&self, message: Buffer) -> Result<Buffer> {
        let sig = self.keypair.sign(message.as_ref())
            .map_err(|e| Error::new(Status::GenericFailure, format!("Sign failed: {:?}", e)))?;
        Ok(Buffer::from(sig.as_bytes().to_vec()))
    }

    #[napi]
    pub fn verify(&self, message: Buffer, signature: Buffer) -> Result<bool> {
        use hyperphysics_dilithium::DilithiumSignature;
        let sig = DilithiumSignature::decode(signature.as_ref(), self.keypair.security_level())
            .map_err(|e| Error::new(Status::InvalidArg, format!("Invalid signature: {:?}", e)))?;
        Ok(self.keypair.verify(message.as_ref(), &sig).unwrap_or(false))
    }

    #[napi(getter)]
    pub fn public_key(&self) -> Buffer {
        Buffer::from(self.keypair.public_key_bytes().to_vec())
    }

    #[napi(getter)]
    pub fn security_level(&self) -> String {
        format!("{:?}", self.keypair.security_level())
    }
}
