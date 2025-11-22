//! Framing Effects for Prospect Theory
use crate::{ProspectTheoryError, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FramingContext {
    pub frame_type: FrameType,
    pub emphasis: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrameType { 
    Gain, 
    Loss, 
    Neutral 
}

#[derive(Debug, Clone)]
pub struct FramingProcessor {
    current_frame: FramingContext,
}

impl FramingProcessor {
    pub fn new() -> Self { 
        Self {
            current_frame: FramingContext {
                frame_type: FrameType::Neutral,
                emphasis: 0.5,
            }
        }
    }
    
    pub fn set_frame(&mut self, frame: FramingContext) {
        self.current_frame = frame;
    }
    
    pub fn get_frame(&self) -> &FramingContext {
        &self.current_frame
    }
    
    pub fn apply_framing(&self, outcomes: &[f64], frame: &FramingContext) -> Result<FramedOutcomes> {
        let effect_multiplier = match frame.frame_type {
            FrameType::Gain => 1.0 + frame.emphasis * 0.1,
            FrameType::Loss => 1.0 - frame.emphasis * 0.15,
            FrameType::Neutral => 1.0,
        };
        
        let framed_outcomes: Vec<f64> = outcomes.iter()
            .map(|&outcome| outcome * effect_multiplier)
            .collect();
        
        Ok(FramedOutcomes {
            buy_outcomes: framed_outcomes.clone(),
            sell_outcomes: framed_outcomes.clone(),
            hold_outcomes: framed_outcomes,
        })
    }
    
    pub fn calculate_framing_effect(&self, base_confidence: f64) -> f64 {
        let effect_strength = match self.current_frame.frame_type {
            FrameType::Gain => self.current_frame.emphasis * 0.1,
            FrameType::Loss => -self.current_frame.emphasis * 0.15,
            FrameType::Neutral => 0.0,
        };
        
        (base_confidence + effect_strength).clamp(0.0, 1.0)
    }
}

impl Default for FramingProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct FramedOutcomes {
    pub buy_outcomes: Vec<f64>,
    pub sell_outcomes: Vec<f64>,
    pub hold_outcomes: Vec<f64>,
}