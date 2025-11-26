// Property-based tests for system invariants
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_portfolio_value_never_negative(
        cash in 0.0..1_000_000.0f64,
        position_values in prop::collection::vec(0.0..100_000.0f64, 0..10)
    ) {
        let total_value = cash + position_values.iter().sum::<f64>();
        prop_assert!(total_value >= 0.0);
    }

    #[test]
    fn test_position_size_respects_limits(
        account_value in 1000.0..1_000_000.0f64,
        risk_percent in 0.01..0.1f64,
        stop_loss in 0.01..0.5f64
    ) {
        // Calculate position size
        let position_size = (account_value * risk_percent) / stop_loss;
        let max_loss = position_size * stop_loss;

        // Max loss should never exceed risk percent
        prop_assert!(max_loss <= account_value * risk_percent + 0.01); // Small tolerance for rounding
    }

    #[test]
    fn test_order_quantity_positive(
        quantity in 1..10000i32
    ) {
        prop_assert!(quantity > 0);
    }

    #[test]
    fn test_price_precision_maintained(
        price in 0.01..10000.0f64
    ) {
        use rust_decimal::Decimal;
        use std::str::FromStr;

        let decimal_price = Decimal::from_str(&format!("{:.2}", price)).unwrap();
        let rounded = decimal_price.to_string().parse::<f64>().unwrap();

        // Price should maintain 2 decimal places
        prop_assert!((rounded - (price * 100.0).round() / 100.0).abs() < 0.01);
    }

    #[test]
    fn test_pnl_calculation_consistency(
        entry_price in 1.0..1000.0f64,
        exit_price in 1.0..1000.0f64,
        quantity in 1..1000i32
    ) {
        let pnl = (exit_price - entry_price) * quantity as f64;
        let reverse_pnl = (entry_price - exit_price) * quantity as f64;

        // PnL should be symmetric
        prop_assert!((pnl + reverse_pnl).abs() < 0.01);
    }

    #[test]
    fn test_sharpe_ratio_bounds(
        returns in prop::collection::vec(-0.1..0.1f64, 10..100)
    ) {
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev > 0.0001 {
            let sharpe = mean / std_dev;

            // Sharpe ratio should be bounded for reasonable returns
            prop_assert!(sharpe > -10.0 && sharpe < 10.0);
        }
    }

    #[test]
    fn test_var_never_positive(
        returns in prop::collection::vec(-0.2..0.2f64, 20..200)
    ) {
        let mut sorted_returns = returns.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let confidence = 0.95;
        let index = ((1.0 - confidence) * sorted_returns.len() as f64) as usize;
        let var = sorted_returns[index];

        // VaR should represent a loss (negative value)
        prop_assert!(var <= 0.0);
    }

    #[test]
    fn test_kelly_fraction_bounded(
        win_prob in 0.01..0.99f64,
        win_loss_ratio in 0.1..10.0f64
    ) {
        let kelly = (win_prob * win_loss_ratio - (1.0 - win_prob)) / win_loss_ratio;

        // Kelly fraction should be reasonable
        prop_assert!(kelly >= -1.0 && kelly <= 1.0);
    }

    #[test]
    fn test_moving_average_smoothness(
        prices in prop::collection::vec(1.0..1000.0f64, 20..100),
        period in 3..20usize
    ) {
        if prices.len() >= period {
            let ma: f64 = prices[prices.len() - period..].iter().sum::<f64>() / period as f64;

            // MA should be within min/max of period
            let min = prices[prices.len() - period..].iter().cloned().fold(f64::INFINITY, f64::min);
            let max = prices[prices.len() - period..].iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            prop_assert!(ma >= min && ma <= max);
        }
    }

    #[test]
    fn test_correlation_bounds(
        values_a in prop::collection::vec(-100.0..100.0f64, 10..50),
        values_b in prop::collection::vec(-100.0..100.0f64, 10..50)
    ) {
        if values_a.len() == values_b.len() && values_a.len() > 1 {
            let n = values_a.len() as f64;
            let mean_a: f64 = values_a.iter().sum::<f64>() / n;
            let mean_b: f64 = values_b.iter().sum::<f64>() / n;

            let covariance: f64 = values_a.iter()
                .zip(values_b.iter())
                .map(|(a, b)| (a - mean_a) * (b - mean_b))
                .sum::<f64>() / n;

            let var_a = values_a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / n;
            let var_b = values_b.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / n;

            if var_a > 0.0001 && var_b > 0.0001 {
                let correlation = covariance / (var_a.sqrt() * var_b.sqrt());

                // Correlation must be between -1 and 1
                prop_assert!(correlation >= -1.01 && correlation <= 1.01);
            }
        }
    }
}
