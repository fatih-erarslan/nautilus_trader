//! Market entity to physics object mapping
//!
//! Maps market entities (orders, participants, liquidity pools) to Rapier rigid bodies
//! with physics properties that model market dynamics.

use crate::Result;
use nalgebra::Vector3;
use rapier3d::prelude::*;

/// Market state to be mapped to physics
#[derive(Debug, Clone)]
pub struct MarketState {
    /// Order book bids (price, volume)
    pub bids: Vec<(f64, f64)>,

    /// Order book asks (price, volume)
    pub asks: Vec<(f64, f64)>,

    /// Recent trades
    pub trades: Vec<Trade>,

    /// Market participants
    pub participants: Vec<MarketParticipant>,

    /// Current mid price
    pub mid_price: f64,

    /// Volatility measure
    pub volatility: f64,
}

/// A market trade
#[derive(Debug, Clone)]
pub struct Trade {
    pub price: f64,
    pub volume: f64,
    pub timestamp: i64,
    pub is_buy: bool,
}

/// Market participant (whale, retail, HFT)
#[derive(Debug, Clone)]
pub struct MarketParticipant {
    pub participant_type: ParticipantType,
    pub capital: f64,
    pub position_size: f64,
    pub aggressiveness: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParticipantType {
    Whale,
    Institutional,
    HFT,
    Retail,
}

/// Maps market state to Rapier rigid bodies
pub struct MarketMapper {
    /// Scaling factor for price to physics space
    price_scale: f32,

    /// Scaling factor for volume to mass
    volume_scale: f32,
}

impl MarketMapper {
    /// Create a new market mapper with default scaling
    pub fn new() -> Self {
        Self {
            price_scale: 0.01,
            volume_scale: 0.001,
        }
    }

    /// Create mapper with custom scaling
    pub fn with_scaling(price_scale: f32, volume_scale: f32) -> Self {
        Self {
            price_scale,
            volume_scale,
        }
    }

    /// Map market state to rigid bodies and colliders
    pub fn map_to_physics(
        &self,
        market_state: &MarketState,
        rigid_bodies: &mut RigidBodySet,
        colliders: &mut ColliderSet,
    ) -> Result<PhysicsMapping> {
        let mut mapping = PhysicsMapping::default();

        // Map order book to rigid bodies
        // Bids are positioned on the left (negative X), asks on right (positive X)
        // Y position represents price, mass represents volume

        for (i, (price, volume)) in market_state.bids.iter().enumerate() {
            let position = Vector3::new(
                -1.0 - (i as f32) * 0.1, // Negative X for bids
                (*price as f32 - market_state.mid_price as f32) * self.price_scale,
                0.0,
            );

            let mass = (*volume as f32) * self.volume_scale;

            let rb = RigidBodyBuilder::dynamic()
                .translation(position)
                .linear_damping(0.5) // Damping represents market friction
                .build();

            let rb_handle = rigid_bodies.insert(rb);

            let collider = ColliderBuilder::ball(mass.sqrt() * 0.1)
                .density(mass)
                .restitution(0.3) // Bouncy orders represent price volatility
                .build();

            colliders.insert_with_parent(collider, rb_handle, rigid_bodies);

            mapping.bid_bodies.push(rb_handle);
        }

        for (i, (price, volume)) in market_state.asks.iter().enumerate() {
            let position = Vector3::new(
                1.0 + (i as f32) * 0.1, // Positive X for asks
                (*price as f32 - market_state.mid_price as f32) * self.price_scale,
                0.0,
            );

            let mass = (*volume as f32) * self.volume_scale;

            let rb = RigidBodyBuilder::dynamic()
                .translation(position)
                .linear_damping(0.5)
                .build();

            let rb_handle = rigid_bodies.insert(rb);

            let collider = ColliderBuilder::ball(mass.sqrt() * 0.1)
                .density(mass)
                .restitution(0.3)
                .build();

            colliders.insert_with_parent(collider, rb_handle, rigid_bodies);

            mapping.ask_bodies.push(rb_handle);
        }

        // Map market participants as larger bodies with forces
        for participant in &market_state.participants {
            let position = self.participant_position(participant);
            let mass = (participant.capital as f32) * self.volume_scale * 10.0;

            let rb = RigidBodyBuilder::dynamic()
                .translation(position)
                .linear_damping(0.3)
                .build();

            let rb_handle = rigid_bodies.insert(rb);

            let collider = ColliderBuilder::ball(mass.sqrt() * 0.2)
                .density(mass)
                .restitution(0.5)
                .build();

            colliders.insert_with_parent(collider, rb_handle, rigid_bodies);

            mapping.participant_bodies.push(rb_handle);
        }

        Ok(mapping)
    }

    /// Determine position for a market participant based on their type
    fn participant_position(&self, participant: &MarketParticipant) -> Vector3<f32> {
        let base_y = (participant.position_size as f32) * self.price_scale;

        match participant.participant_type {
            ParticipantType::Whale => Vector3::new(0.0, base_y + 2.0, 0.0),
            ParticipantType::Institutional => Vector3::new(0.0, base_y + 1.0, 0.5),
            ParticipantType::HFT => Vector3::new(0.0, base_y, 1.0),
            ParticipantType::Retail => Vector3::new(0.0, base_y - 1.0, -0.5),
        }
    }
}

impl Default for MarketMapper {
    fn default() -> Self {
        Self::new()
    }
}

/// Mapping of market entities to physics bodies
#[derive(Debug, Default)]
pub struct PhysicsMapping {
    pub bid_bodies: Vec<RigidBodyHandle>,
    pub ask_bodies: Vec<RigidBodyHandle>,
    pub participant_bodies: Vec<RigidBodyHandle>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_mapper_creation() {
        let mapper = MarketMapper::new();
        assert_eq!(mapper.price_scale, 0.01);
        assert_eq!(mapper.volume_scale, 0.001);
    }

    #[test]
    fn test_map_simple_market() {
        let mapper = MarketMapper::new();
        let mut rigid_bodies = RigidBodySet::new();
        let mut colliders = ColliderSet::new();

        let market_state = MarketState {
            bids: vec![(100.0, 10.0), (99.0, 5.0)],
            asks: vec![(101.0, 8.0), (102.0, 12.0)],
            trades: vec![],
            participants: vec![],
            mid_price: 100.5,
            volatility: 0.02,
        };

        let mapping = mapper
            .map_to_physics(&market_state, &mut rigid_bodies, &mut colliders)
            .unwrap();

        assert_eq!(mapping.bid_bodies.len(), 2);
        assert_eq!(mapping.ask_bodies.len(), 2);
        assert_eq!(rigid_bodies.len(), 4);
    }

    #[test]
    fn test_participant_positioning() {
        let mapper = MarketMapper::new();

        let whale = MarketParticipant {
            participant_type: ParticipantType::Whale,
            capital: 1000000.0,
            position_size: 500.0,
            aggressiveness: 0.8,
        };

        let position = mapper.participant_position(&whale);
        assert!(position.y > 0.0); // Whales positioned above
    }
}
