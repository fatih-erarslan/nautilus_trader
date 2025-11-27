// -------------------------------------------------------------------------------------------------
//  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
//  https://nautechsystems.io
//
//  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
//  You may not use this file except in compliance with the License.
//  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
// -------------------------------------------------------------------------------------------------

//! gRPC client implementation for the dYdX v4 protocol.
//!
//! This module provides gRPC client functionality for interacting with the dYdX v4 protocol
//! via the Cosmos SDK. It handles:
//!
//! - Transaction signing and broadcasting using `cosmrs`.
//! - gRPC communication with validator nodes.
//! - Protocol Buffer message encoding/decoding.
//! - Cosmos SDK account management.
//!
//! The client supports dYdX trading operations including:
//!
//! - Order placement, modification, and cancellation.
//! - Transfer operations between subaccounts.
//! - Subaccount management.
//! - Transaction signing with secp256k1 keys.
//!
//! # Architecture
//!
//! dYdX v4 is built on the Cosmos SDK and uses gRPC for all state-changing operations
//! (placing orders, transfers, etc.). The HTTP/REST API (Indexer) is read-only and used
//! for querying market data and historical information.

// Full gRPC implementation modules exist but are disabled due to tonic/prost version
// mismatches between dydx-proto (0.13.x) and nautilus workspace (0.14.x).
// Enable these when versions are aligned:
// pub mod builder;
// pub mod client;
// pub mod order;
pub mod types;
pub mod wallet;

// Re-exports - using stubs until version alignment
// Future: pub use builder::TxBuilder;
// Future: pub use client::{DydxGrpcClient, Height, TxHash};
// Future: pub use order::{...};
pub use types::ChainId;
pub use wallet::{Account, Subaccount, Wallet};

// Temporary stubs until tonic/prost versions are aligned
// The full implementations exist in builder.rs, client.rs, and order.rs
#[derive(Debug, Clone)]
pub struct DydxGrpcClient;

impl DydxGrpcClient {
    /// Creates a new dYdX gRPC client.
    ///
    /// NOTE: This is a stub implementation. The full gRPC client is available in
    /// `client.rs` but is disabled due to tonic/prost version conflicts.
    /// The `dydx-proto` crate uses tonic 0.13.x while nautilus uses 0.14.x.
    ///
    /// # Errors
    ///
    /// This stub currently never fails. The real implementation will return
    /// connection errors when enabled.
    pub async fn new(endpoint: String) -> anyhow::Result<Self> {
        tracing::warn!(
            "Using stub dYdX gRPC client for endpoint: {endpoint}. \
             Enable full client by aligning tonic/prost versions."
        );
        Ok(Self)
    }

    /// Creates a new dYdX gRPC client with fallback endpoints.
    ///
    /// # Errors
    ///
    /// This stub currently never fails.
    pub async fn new_with_fallback(endpoints: &[String]) -> anyhow::Result<Self> {
        if endpoints.is_empty() {
            anyhow::bail!("No dYdX gRPC endpoints provided");
        }

        for (idx, url) in endpoints.iter().enumerate() {
            tracing::info!(
                "Stub: Would connect to dYdX gRPC endpoint ({}/{}): {url}",
                idx + 1,
                endpoints.len()
            );

            if idx == 0 {
                return Self::new(url.clone()).await;
            }
        }

        Self::new(endpoints[0].clone()).await
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Height(pub u32);

pub type TxHash = String;

#[derive(Debug, Clone)]
pub struct OrderBuilder;

#[derive(Debug, Clone)]
pub struct OrderFlags;

#[derive(Debug, Clone)]
pub enum OrderGoodUntil {
    Block(u32),
    Time(chrono::DateTime<chrono::Utc>),
}

#[derive(Debug, Clone)]
pub struct OrderMarketParams {
    pub atomic_resolution: i32,
    pub clob_pair_id: u32,
    pub oracle_price: Option<u64>,
    pub quantum_conversion_exponent: i32,
    pub step_base_quantums: u64,
    pub subticks_per_tick: u32,
}

pub const SHORT_TERM_ORDER_MAXIMUM_LIFETIME: u32 = 20;
pub const DEFAULT_RUST_CLIENT_METADATA: u32 = 4;
