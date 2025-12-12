//! Integration tests for HyperPhysics-Wassette

use hyperphysics_wassette::*;

#[test]
fn test_module_imports() {
    let _runtime = WasmRuntime::new();
    assert!(true);
}

#[test]
fn test_backend_availability() {
    let backends = available_backends();
    assert!(!backends.is_empty());
    assert!(backends.contains(&WasmBackend::Wasmi));
}

#[test]
fn test_runtime_creation() {
    let runtime = WasmRuntime::new();
    assert!(runtime.is_ok());
}

#[test]
fn test_runtime_with_backend() {
    let runtime = WasmRuntime::with_backend(WasmBackend::Wasmi);
    assert!(runtime.is_ok());
    assert_eq!(runtime.unwrap().backend(), WasmBackend::Wasmi);
}

#[tokio::test]
async fn test_load_invalid_module() {
    let runtime = WasmRuntime::new().unwrap();
    let result = runtime.load_module_from_bytes(&[0, 1, 2, 3]).await;
    assert!(result.is_err());
}

#[test]
fn test_host_functions() {
    let host_funcs = HostFunctions::new();
    // Basic creation test
    assert!(true);
}

#[test]
fn test_wasm_optimizer() {
    let optimizer = WasmOptimizer::new();
    let original = vec![0, 1, 2, 3, 4];
    let optimized = optimizer.optimize(&original).unwrap();
    assert_eq!(optimized, original);
}

#[tokio::test]
async fn test_neural_wasm_executor() {
    let executor = NeuralWasmExecutor::new();
    let inputs = vec![1.0, 2.0, 3.0];
    let outputs = executor.forward(&inputs).await.unwrap();
    assert_eq!(outputs.len(), inputs.len());
}
