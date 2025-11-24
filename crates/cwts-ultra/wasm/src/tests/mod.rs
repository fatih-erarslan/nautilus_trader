use super::*;

mod quantum_trading_tests;
mod wasm_tests;

// Add your tests here
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }

    #[test]
    fn test_wasm_cwts_creation() {
        let cwts = WasmCWTS::new();
        assert!(cwts.has_neural_network());
        assert_eq!(cwts.get_capital(), 50.0);
    }
}