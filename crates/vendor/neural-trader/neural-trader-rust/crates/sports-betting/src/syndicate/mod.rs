//! Syndicate management module

mod capital;
mod voting;
mod members;
mod collaboration;
mod manager;

pub use capital::CapitalManager;
pub use voting::VotingSystem;
pub use members::MemberManager;
pub use collaboration::CollaborationManager;
pub use manager::SyndicateManager;
