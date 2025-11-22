//! # Autopoiesis Trading System
//! 
//! A self-organizing, biomimetic trading system inspired by autopoietic systems.

#![warn(missing_docs)]
#![allow(clippy::all)]

// Always re-export core
pub use autopoiesis_core as core;

// Conditionally re-export other crates based on features
#[cfg(feature = "ml")]
pub use autopoiesis_ml as ml;

#[cfg(feature = "finance")]
pub use autopoiesis_finance as finance;

#[cfg(feature = "consciousness")]
pub use autopoiesis_consciousness as consciousness;

#[cfg(feature = "engines")]
pub use autopoiesis_engines as engines;

#[cfg(feature = "analysis")]
pub use autopoiesis_analysis as analysis;

#[cfg(feature = "api")]
pub use autopoiesis_api as api;

/// Prelude for convenient imports
#[allow(ambiguous_glob_reexports)]
pub mod prelude {
    pub use autopoiesis_core::prelude::*;
    
    #[cfg(feature = "ml")]
    pub use autopoiesis_ml::prelude::*;
    
    #[cfg(feature = "finance")]
    pub use autopoiesis_finance::prelude::*;
    
    #[cfg(feature = "consciousness")]
    pub use autopoiesis_consciousness::prelude::*;
    
    #[cfg(feature = "engines")]
    pub use autopoiesis_engines::prelude::*;
    
    #[cfg(feature = "analysis")]
    pub use autopoiesis_analysis::prelude::*;
    
    #[cfg(feature = "api")]
    pub use autopoiesis_api::prelude::*;
}
