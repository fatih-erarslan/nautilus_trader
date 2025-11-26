//! Integration tests for portfolio tracking

use nt_portfolio::Portfolio;
use rust_decimal_macros::dec;

mod utils {
    include!("../utils/assertions.rs");
}

use utils::*;

#[test]
fn test_portfolio_initialization() {
    let portfolio = Portfolio::new(dec!(100000));

    assert_eq!(portfolio.cash(), dec!(100000));
    assert_eq!(portfolio.total_value(), dec!(100000));
}

#[test]
fn test_portfolio_cash_operations() {
    let mut portfolio = Portfolio::new(dec!(100000));

    // Add cash
    portfolio.add_cash(dec!(10000));
    assert_eq!(portfolio.cash(), dec!(110000));

    // Remove cash
    portfolio.remove_cash(dec!(5000)).expect("Should remove cash");
    assert_eq!(portfolio.cash(), dec!(105000));
}

#[test]
fn test_portfolio_insufficient_cash() {
    let mut portfolio = Portfolio::new(dec!(1000));

    let result = portfolio.remove_cash(dec!(2000));
    assert!(result.is_err(), "Should fail with insufficient cash");
}

#[test]
fn test_portfolio_total_value() {
    let portfolio = Portfolio::new(dec!(50000));

    // Total value should equal cash when no positions
    assert_eq!(portfolio.total_value(), dec!(50000));
}

#[test]
fn test_portfolio_value_consistency() {
    let portfolio = Portfolio::new(dec!(100000));

    // Portfolio value should always be non-negative
    assert_decimal_non_negative(portfolio.total_value());
    assert_decimal_non_negative(portfolio.cash());
}

#[test]
fn test_portfolio_zero_cash() {
    let mut portfolio = Portfolio::new(dec!(100));

    portfolio.remove_cash(dec!(100)).expect("Should remove all cash");
    assert_eq!(portfolio.cash(), dec!(0));
}
