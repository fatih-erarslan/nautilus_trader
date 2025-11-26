//! Sports betting example with Kelly Criterion and syndicate management

use multi_market::sports::*;
use multi_market::Result;
use rust_decimal_macros::dec;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🏈 Sports Betting Multi-Market Example\n");

    // 1. Kelly Criterion Optimization
    println!("1️⃣ Kelly Criterion Optimization");
    let kelly = KellyOptimizer::new(dec!(10000), dec!(0.25));
    println!("   Bankroll: ${}", kelly.bankroll());
    println!("   Kelly Multiplier: {}", kelly.kelly_multiplier());

    let opportunity = BettingOpportunity {
        event_id: "nba_lakers_warriors".to_string(),
        outcome: "Lakers Win".to_string(),
        odds: dec!(2.5),
        win_probability: dec!(0.45),
        max_stake: None,
    };

    if let Some(result) = kelly.calculate(&opportunity)? {
        println!("\n   📊 Betting Opportunity Analysis:");
        println!("      Event: {}", opportunity.event_id);
        println!("      Outcome: {}", opportunity.outcome);
        println!("      Odds: {}", opportunity.odds);
        println!("      Win Probability: {}%", opportunity.win_probability * dec!(100));
        println!("\n   💰 Kelly Recommendation:");
        println!("      Optimal Stake: ${:.2}", result.optimal_stake);
        println!("      Kelly Fraction: {:.4}", result.kelly_fraction);
        println!("      Expected Value: {:.4}", result.expected_value);
        println!("      Expected Growth: {:.4}", result.expected_growth);
        println!("      Risk of Ruin: {:.2}%", result.risk_of_ruin * dec!(100));
    }

    // 2. Syndicate Management
    println!("\n\n2️⃣ Betting Syndicate Management");
    let mut syndicate = Syndicate::new(
        "elite_bettors".to_string(),
        "Elite Sports Bettors".to_string(),
        "Professional sports betting syndicate".to_string(),
    );

    println!("   Created syndicate: {}", syndicate.name);

    // Add members
    syndicate.add_member(
        "john_doe".to_string(),
        "John Doe".to_string(),
        "john@example.com".to_string(),
        MemberRole::Manager,
        dec!(5000),
    )?;

    syndicate.add_member(
        "jane_smith".to_string(),
        "Jane Smith".to_string(),
        "jane@example.com".to_string(),
        MemberRole::Member,
        dec!(3000),
    )?;

    println!("\n   👥 Members:");
    for member in syndicate.active_members() {
        println!(
            "      - {} ({:?}): ${} contribution ({}% share)",
            member.name,
            member.role,
            member.capital_contributed,
            member.share_percentage * dec!(100)
        );
    }

    println!("\n   💵 Syndicate Finances:");
    println!("      Total Capital: ${}", syndicate.total_capital);
    println!("      Available Capital: ${}", syndicate.available_capital);

    // Place a bet
    println!("\n   📝 Placing Bet:");
    syndicate.place_bet(
        "bet_001".to_string(),
        "nba_lakers_warriors".to_string(),
        "Lakers vs Warriors".to_string(),
        "draftkings".to_string(),
        "Lakers Win".to_string(),
        dec!(2.5),
        dec!(500),
    )?;

    println!("      Bet placed: $500 on Lakers Win @ 2.5 odds");
    println!("      Available Capital: ${}", syndicate.available_capital);

    // Settle bet (won)
    let pnl = syndicate.settle_bet("bet_001", true)?;
    println!("\n   ✅ Bet Settled (WON):");
    println!("      Profit: ${}", pnl);
    println!("      Available Capital: ${}", syndicate.available_capital);
    println!("      Total PnL: ${}", syndicate.total_pnl);

    // Distribute profits
    println!("\n   💸 Distributing Profits:");
    let distribution = syndicate.distribute_profits()?;
    println!("      Total Distributed: ${}", distribution.total_profit);
    for (member_id, amount) in &distribution.member_distributions {
        println!("      - {}: ${}", member_id, amount);
    }

    // Performance metrics
    println!("\n   📈 Syndicate Performance:");
    println!("      Win Rate: {}%", syndicate.win_rate() * dec!(100));
    println!("      ROI: {}%", syndicate.roi());

    println!("\n✅ Example completed successfully!");
    Ok(())
}
