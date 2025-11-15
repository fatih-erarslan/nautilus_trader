
use crate::lattice::module_lwe::PolyVec;
use zeroize::{Zeroize, ZeroizeOnDrop};
use std::ops::{Deref, DerefMut};

/// Wrapper around `PolyVec` that implements `Zeroize` and `ZeroizeOnDrop`.
#[derive(Clone, Default)]
pub struct ZeroizablePolyVec(pub PolyVec);

impl Deref for ZeroizablePolyVec {
    type Target = PolyVec;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for ZeroizablePolyVec {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Zeroize for ZeroizablePolyVec {
    fn zeroize(&mut self) {
        for poly in self.0.iter_mut() {
            poly.zeroize();
        }
    }
}

impl ZeroizeOnDrop for ZeroizablePolyVec {}
