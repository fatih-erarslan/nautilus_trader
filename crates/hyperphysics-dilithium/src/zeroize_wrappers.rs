
use pqcrypto_kyber::kyber768::{PublicKey as KyberPublicKey, SecretKey as KyberSecretKey};
use pqcrypto_traits::kem::{PublicKey as _, SecretKey as _};
use zeroize::{Zeroize, ZeroizeOnDrop};

#[derive(Clone)]
pub struct PublicKey(pub KyberPublicKey);

impl Zeroize for PublicKey {
    fn zeroize(&mut self) {
        let mut bytes = self.0.as_bytes().to_vec();
        bytes.zeroize();
    }
}

impl ZeroizeOnDrop for PublicKey {}

#[derive(Clone)]
pub struct SecretKey(pub KyberSecretKey);

impl Zeroize for SecretKey {
    fn zeroize(&mut self) {
        let mut bytes = self.0.as_bytes().to_vec();
        bytes.zeroize();
    }
}

impl ZeroizeOnDrop for SecretKey {}
