//! Neural Trader binary entry point.

use neural_trader_integration::api::Cli;

#[tokio::main]
async fn main() {
    if let Err(e) = Cli::run().await {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
