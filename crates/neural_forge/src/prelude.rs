//! Prelude module for convenient imports

pub use crate::{
    config::*,
    data::*,
    models::*,
    training::*,
    optimization::*,
    calibration::*,
    evaluation::*,
    error::{Result, NeuralForgeError},
    backends::*,
};

pub use candle_core::{Device, Tensor, DType, Shape};
pub use candle_nn::{Module, VarBuilder, VarMap};
pub use polars::prelude::*;
pub use serde::{Deserialize, Serialize};
pub use anyhow::anyhow;
pub use std::collections::HashMap;