#[cfg(test)]
mod tests {
    use crate::WasmCWTS;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_wasm_cwts_new() {
        let cwts = WasmCWTS::new();
        assert_eq!(cwts.get_capital(), 50.0);
    }

    #[wasm_bindgen_test]
    fn test_wasm_cwts_tick() {
        let mut cwts = WasmCWTS::new();
        let empty_orderbook: &[u8] = &[];
        let decision = cwts.tick(empty_orderbook);
        
        // Should return a valid decision (0, 1, or 2)
        assert!(decision <= 2);
    }

    #[wasm_bindgen_test]
    fn test_wasm_cwts_get_capital() {
        let cwts = WasmCWTS::new();
        let capital = cwts.get_capital();
        assert!(capital > 0.0);
        assert_eq!(capital, 50.0);
    }

    #[wasm_bindgen_test]
    fn test_wasm_cwts_multiple_ticks() {
        let mut cwts = WasmCWTS::new();
        let empty_orderbook: &[u8] = &[];
        
        // Test multiple ticks to ensure consistency
        for _ in 0..10 {
            let decision = cwts.tick(empty_orderbook);
            assert!(decision <= 2);
        }
    }

    #[wasm_bindgen_test]
    fn test_neural_network_initialization() {
        let cwts = WasmCWTS::new();
        // Neural network should be initialized (Some variant)
        assert!(cwts.neural_network.is_some());
    }

    #[test]
    fn test_native_wasm_cwts() {
        // Test that can run in native Rust environment
        let cwts = WasmCWTS::new();
        assert_eq!(cwts.capital, 50.0);
    }

    #[test] 
    fn test_native_tick_logic() {
        let mut cwts = WasmCWTS::new();
        let test_data = vec![1, 2, 3, 4, 5];
        let decision = cwts.tick(&test_data);
        
        // With neural network present, should return 1 (buy decision)
        assert_eq!(decision, 1);
    }

    #[test]
    fn test_capital_bounds() {
        let cwts = WasmCWTS::new();
        let capital = cwts.get_capital();
        
        // Capital should be positive and within reasonable bounds
        assert!(capital > 0.0);
        assert!(capital < 1_000_000.0);  // Sanity check
    }
}