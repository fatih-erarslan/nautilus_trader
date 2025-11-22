#!/usr/bin/env python3
"""
Financial examples demonstrating prospect theory applications
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

# Import the Rust module
try:
    import prospect_theory_rs as pt
except ImportError:
    print("ERROR: prospect_theory_rs module not found. Please compile first:")
    print("maturin develop --release")
    sys.exit(1)

def example_1_loss_aversion():
    """Demonstrate loss aversion effect"""
    print("=" * 50)
    print("EXAMPLE 1: LOSS AVERSION DEMONSTRATION")
    print("=" * 50)
    
    vf = pt.ValueFunction.default()
    
    # Equal magnitude gains and losses
    amounts = [50, 100, 200, 500, 1000]
    
    print(f"{'Amount':<10} {'Gain Value':<12} {'Loss Value':<12} {'Ratio':<10}")
    print("-" * 50)
    
    for amount in amounts:
        gain_value = vf.value(amount)
        loss_value = vf.value(-amount)
        ratio = abs(loss_value) / gain_value
        
        print(f"${amount:<9} {gain_value:<12.4f} {loss_value:<12.4f} {ratio:<10.2f}")
    
    print(f"\nLoss aversion ratio for $100: {vf.loss_aversion_ratio(100.0, -100.0):.2f}")
    print("Note: Ratio > 1 indicates loss aversion")

def example_2_probability_weighting():
    """Demonstrate probability weighting effects"""
    print("\n" + "=" * 50)
    print("EXAMPLE 2: PROBABILITY WEIGHTING")
    print("=" * 50)
    
    pw = pt.ProbabilityWeighting.default()
    
    probabilities = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    
    print(f"{'Probability':<12} {'Weight (Gains)':<15} {'Weight (Losses)':<16} {'Distortion':<12}")
    print("-" * 60)
    
    for prob in probabilities:
        weight_gains = pw.weight_gains(prob)
        weight_losses = pw.weight_losses(prob)
        distortion = weight_gains - prob
        
        print(f"{prob:<12.2f} {weight_gains:<15.4f} {weight_losses:<16.4f} {distortion:<12.4f}")
    
    print("\nNote: Positive distortion = overweighting, Negative = underweighting")

def example_3_portfolio_choice():
    """Portfolio choice under prospect theory"""
    print("\n" + "=" * 50)
    print("EXAMPLE 3: PORTFOLIO CHOICE")
    print("=" * 50)
    
    pt_calc = pt.ProspectTheory.default()
    
    # Portfolio A: Conservative (lower risk, lower return)
    portfolio_a_outcomes = [80, 60, 40, 20, 0, -10, -20]
    portfolio_a_probs = [0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05]
    
    # Portfolio B: Aggressive (higher risk, higher return)
    portfolio_b_outcomes = [200, 150, 100, 50, 0, -50, -100, -150]
    portfolio_b_probs = [0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05]
    
    value_a = pt_calc.prospect_value(portfolio_a_outcomes, portfolio_a_probs)
    value_b = pt_calc.prospect_value(portfolio_b_outcomes, portfolio_b_probs)
    
    # Expected values for comparison
    exp_value_a = sum(o * p for o, p in zip(portfolio_a_outcomes, portfolio_a_probs))
    exp_value_b = sum(o * p for o, p in zip(portfolio_b_outcomes, portfolio_b_probs))
    
    print(f"Portfolio A (Conservative):")
    print(f"  Expected Value: ${exp_value_a:.2f}")
    print(f"  Prospect Value: {value_a:.4f}")
    
    print(f"\nPortfolio B (Aggressive):")
    print(f"  Expected Value: ${exp_value_b:.2f}")
    print(f"  Prospect Value: {value_b:.4f}")
    
    comparison = pt_calc.compare_lotteries(
        portfolio_a_outcomes, portfolio_a_probs,
        portfolio_b_outcomes, portfolio_b_probs
    )
    
    print(f"\nProspect Theory Preference: {comparison['preference']}")
    print(f"Value Difference: {comparison['value_difference']:.4f}")

def example_4_insurance_decision():
    """Insurance purchase decision"""
    print("\n" + "=" * 50)
    print("EXAMPLE 4: INSURANCE DECISION")
    print("=" * 50)
    
    pt_calc = pt.ProspectTheory.default()
    
    # Scenario: Potential loss of $10,000 with 2% probability
    # Insurance costs $300
    
    # No insurance
    no_insurance_outcomes = [0, -10000]
    no_insurance_probs = [0.98, 0.02]
    
    # With insurance
    with_insurance_outcomes = [-300]
    with_insurance_probs = [1.0]
    
    value_no_insurance = pt_calc.prospect_value(no_insurance_outcomes, no_insurance_probs)
    value_with_insurance = pt_calc.prospect_value(with_insurance_outcomes, with_insurance_probs)
    
    expected_no_insurance = sum(o * p for o, p in zip(no_insurance_outcomes, no_insurance_probs))
    expected_with_insurance = -300
    
    print(f"No Insurance:")
    print(f"  Expected Value: ${expected_no_insurance:.2f}")
    print(f"  Prospect Value: {value_no_insurance:.4f}")
    
    print(f"\nWith Insurance:")
    print(f"  Expected Value: ${expected_with_insurance:.2f}")
    print(f"  Prospect Value: {value_with_insurance:.4f}")
    
    if value_with_insurance > value_no_insurance:
        print("\nRecommendation: BUY INSURANCE")
        print("Prospect theory suggests insurance is preferred despite negative expected value")
    else:
        print("\nRecommendation: SKIP INSURANCE")
        print("Expected value maximization preferred")

def example_5_option_valuation():
    """Option valuation with prospect theory"""
    print("\n" + "=" * 50)
    print("EXAMPLE 5: OPTION VALUATION")
    print("=" * 50)
    
    vf = pt.ValueFunction.default()
    
    # Call option on stock trading at $100
    # Strike prices: $90, $100, $110
    current_price = 100
    strikes = [90, 100, 110]
    
    # Simplified: assume 50% chance stock goes to $120, 50% to $80
    future_prices = [120, 80]
    probabilities = [0.5, 0.5]
    
    print(f"Current Stock Price: ${current_price}")
    print(f"Possible Future Prices: ${future_prices[0]}, ${future_prices[1]}")
    print()
    print(f"{'Strike':<8} {'Expected Payoff':<15} {'Prospect Value':<15} {'Difference':<12}")
    print("-" * 55)
    
    for strike in strikes:
        # Calculate payoffs
        payoffs = [max(price - strike, 0) for price in future_prices]
        expected_payoff = sum(p * prob for p, prob in zip(payoffs, probabilities))
        
        # Calculate prospect values
        prospect_values = [vf.value(payoff) for payoff in payoffs]
        prospect_value = sum(pv * prob for pv, prob in zip(prospect_values, probabilities))
        
        difference = prospect_value - expected_payoff
        
        print(f"${strike:<7} ${expected_payoff:<14.2f} {prospect_value:<15.4f} {difference:<12.4f}")
    
    print("\nNote: Difference shows how prospect theory valuation differs from expected value")

def example_6_risk_premium():
    """Calculate risk premiums for different lotteries"""
    print("\n" + "=" * 50)
    print("EXAMPLE 6: RISK PREMIUMS")
    print("=" * 50)
    
    vf = pt.ValueFunction.default()
    
    lotteries = [
        {
            "name": "Low Risk",
            "outcomes": [110, 90],
            "probabilities": [0.5, 0.5]
        },
        {
            "name": "Medium Risk", 
            "outcomes": [150, 50],
            "probabilities": [0.5, 0.5]
        },
        {
            "name": "High Risk",
            "outcomes": [200, 0],
            "probabilities": [0.5, 0.5]
        },
        {
            "name": "Very High Risk",
            "outcomes": [500, -100],
            "probabilities": [0.5, 0.5]
        }
    ]
    
    print(f"{'Lottery':<15} {'Expected Value':<15} {'Risk Premium':<15} {'Certainty Equiv':<15}")
    print("-" * 65)
    
    for lottery in lotteries:
        outcomes = lottery["outcomes"]
        probs = lottery["probabilities"]
        
        expected_value = sum(o * p for o, p in zip(outcomes, probs))
        risk_premium = vf.risk_premium(outcomes, probs)
        certainty_equiv = expected_value - risk_premium
        
        print(f"{lottery['name']:<15} ${expected_value:<14.2f} ${risk_premium:<14.2f} ${certainty_equiv:<14.2f}")
    
    print("\nNote: Higher risk premium indicates greater risk aversion for that lottery")

def example_7_performance_benchmark():
    """Demonstrate performance with large datasets"""
    print("\n" + "=" * 50)
    print("EXAMPLE 7: PERFORMANCE BENCHMARK")
    print("=" * 50)
    
    import time
    
    vf = pt.ValueFunction.default()
    pw = pt.ProbabilityWeighting.default()
    
    # Large dataset
    n_outcomes = 100_000
    outcomes = [i - n_outcomes//2 for i in range(n_outcomes)]
    probabilities = [1.0/n_outcomes] * n_outcomes
    
    print(f"Testing with {n_outcomes:,} data points...")
    
    # Sequential vs parallel comparison for value function
    start_time = time.time()
    values_seq = vf.values(outcomes)
    seq_time = time.time() - start_time
    
    start_time = time.time()
    values_par = vf.values_parallel(outcomes)
    par_time = time.time() - start_time
    
    # Probability weighting (exclude extremes)
    prob_subset = probabilities[1:-1]
    
    start_time = time.time()
    weights_seq = pw.weights_gains(prob_subset)
    weight_seq_time = time.time() - start_time
    
    start_time = time.time()
    weights_par = pw.weights_gains_parallel(prob_subset)
    weight_par_time = time.time() - start_time
    
    print(f"\nValue Function Results:")
    print(f"  Sequential: {seq_time:.4f} seconds")
    print(f"  Parallel:   {par_time:.4f} seconds")
    print(f"  Speedup:    {seq_time/par_time:.2f}x")
    
    print(f"\nProbability Weighting Results:")
    print(f"  Sequential: {weight_seq_time:.4f} seconds")
    print(f"  Parallel:   {weight_par_time:.4f} seconds")
    print(f"  Speedup:    {weight_seq_time/weight_par_time:.2f}x")
    
    # Verify results are identical
    max_diff_values = max(abs(v1 - v2) for v1, v2 in zip(values_seq, values_par))
    max_diff_weights = max(abs(w1 - w2) for w1, w2 in zip(weights_seq, weights_par))
    
    print(f"\nAccuracy Check:")
    print(f"  Max difference (values):  {max_diff_values:.2e}")
    print(f"  Max difference (weights): {max_diff_weights:.2e}")
    print(f"  Financial precision:      {1e-10:.2e}")

def create_visualization():
    """Create visualization of value function and probability weighting"""
    try:
        import matplotlib.pyplot as plt
        
        print("\n" + "=" * 50)
        print("CREATING VISUALIZATIONS")
        print("=" * 50)
        
        vf = pt.ValueFunction.default()
        pw = pt.ProbabilityWeighting.default()
        
        # Value function plot
        outcomes = np.linspace(-200, 200, 1000)
        values = vf.values(outcomes.tolist())
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(outcomes, values, 'b-', linewidth=2, label='Prospect Theory Value')
        plt.plot(outcomes, outcomes, 'r--', alpha=0.5, label='Linear (Expected Value)')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Outcome')
        plt.ylabel('Value')
        plt.title('Value Function')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Probability weighting plot
        probs = np.linspace(0.01, 0.99, 100)
        weights_gains = pw.weights_gains(probs.tolist())
        weights_losses = pw.weights_losses(probs.tolist())
        
        plt.subplot(1, 2, 2)
        plt.plot(probs, weights_gains, 'g-', linewidth=2, label='Gains')
        plt.plot(probs, weights_losses, 'r-', linewidth=2, label='Losses')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Linear')
        plt.xlabel('Objective Probability')
        plt.ylabel('Decision Weight')
        plt.title('Probability Weighting')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/kutlu/freqtrade/user_data/strategies/crates/prospect-theory-rs/prospect_theory_plots.png', 
                   dpi=300, bbox_inches='tight')
        print("Plots saved to: prospect_theory_plots.png")
        
    except ImportError:
        print("Matplotlib not available. Skipping visualization.")

def main():
    """Run all examples"""
    print("PROSPECT THEORY FINANCIAL EXAMPLES")
    print("Using Rust implementation with PyO3 bindings")
    
    example_1_loss_aversion()
    example_2_probability_weighting() 
    example_3_portfolio_choice()
    example_4_insurance_decision()
    example_5_option_valuation()
    example_6_risk_premium()
    example_7_performance_benchmark()
    
    create_visualization()
    
    print("\n" + "=" * 50)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 50)

if __name__ == "__main__":
    main()