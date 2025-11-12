//! GPU device management and initialization

use super::GpuContext;
use std::sync::OnceLock;

/// Global GPU context singleton
static GPU_CONTEXT: OnceLock<GpuContext> = OnceLock::new();

/// Initialize global GPU context
pub async fn init_global_gpu() -> Result<&'static GpuContext, Box<dyn std::error::Error>> {
    if GPU_CONTEXT.get().is_none() {
        let ctx = GpuContext::new().await?;
        let _ = GPU_CONTEXT.set(ctx);
    }
    Ok(GPU_CONTEXT.get().unwrap())
}

/// Get global GPU context (must be initialized first)
pub fn get_global_gpu() -> Option<&'static GpuContext> {
    GPU_CONTEXT.get()
}

/// Check if GPU is available
pub async fn is_gpu_available() -> bool {
    init_global_gpu().await.is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gpu_init() {
        let result = init_global_gpu().await;
        match result {
            Ok(ctx) => {
                let info = ctx.info();
                println!("GPU: {} ({:?})", info.name, info.device_type);
                assert!(info.max_buffer_size > 0);
            }
            Err(e) => {
                println!("GPU not available: {}", e);
                // Not a failure - GPU might not be available on all systems
            }
        }
    }
}
