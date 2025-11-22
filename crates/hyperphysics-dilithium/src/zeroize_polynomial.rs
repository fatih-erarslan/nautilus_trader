
use crate::lattice::module_lwe::Polynomial;
use zeroize::{Zeroize, ZeroizeOnDrop};
use std::ops::{Deref, DerefMut};

/// Wrapper around `Polynomial` that implements `Zeroize` and `ZeroizeOnDrop`.
/// TODO: Will be used for secure polynomial operations
#[allow(dead_code)]
#[derive(Clone, Default)]
pub struct ZeroizablePolynomial(pub Polynomial);

impl Deref for ZeroizablePolynomial {
    type Target = Polynomial;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for ZeroizablePolynomial {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Zeroize for ZeroizablePolynomial {
    fn zeroize(&mut self) {
        for coeff in self.0.iter_mut() {
            coeff.zeroize();
        }
    }
}

impl ZeroizeOnDrop for ZeroizablePolynomial {}
