//! Mechanism Design Benchmark for Game Theory Engine
//!
//! Benchmarks mechanism design components based on:
//! - Myerson, R. (1981). "Optimal Auction Design". Mathematics of Operations Research
//! - Vickrey, W. (1961). "Counterspeculation, Auctions, and Competitive Sealed Tenders"
//! - Clarke, E. (1971). "Multipart Pricing of Public Goods"

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use game_theory_engine::{
    MechanismDesigner, MechanismObjectives, Mechanism, AllocationRule, PaymentRule,
    MechanismProperties, AuctionMechanism, AuctionType, Bidder, ValueDistribution,
    GameState, GameType, PayoffMatrix, MarketRegime, MarketContext,
    RegulatoryEnvironment, TransparencyLevel,
};
use std::collections::HashMap;

/// Create test bidders for auction benchmarks
fn create_bidders(count: usize) -> Vec<Bidder> {
    (0..count).map(|i| Bidder {
        id: format!("bidder_{}", i),
        valuation: 100.0 + (i as f64 * 10.0),
        budget: 1000.0 + (i as f64 * 100.0),
        risk_aversion: (0.1 + (i as f64 * 0.05)).min(0.9),
    }).collect()
}

/// Create a game state for mechanism verification
fn create_game_state() -> GameState {
    let mut payoffs = HashMap::new();
    payoffs.insert("P1_payoff_0_0".to_string(), 1.0);
    payoffs.insert("P1_payoff_0_1".to_string(), 0.0);
    payoffs.insert("P1_payoff_1_0".to_string(), 0.0);
    payoffs.insert("P1_payoff_1_1".to_string(), 1.0);
    payoffs.insert("P2_payoff_0_0".to_string(), 1.0);
    payoffs.insert("P2_payoff_0_1".to_string(), 0.0);
    payoffs.insert("P2_payoff_1_0".to_string(), 0.0);
    payoffs.insert("P2_payoff_1_1".to_string(), 1.0);

    let mut strategies = HashMap::new();
    strategies.insert("P1".to_string(), vec!["A".to_string(), "B".to_string()]);
    strategies.insert("P2".to_string(), vec!["A".to_string(), "B".to_string()]);

    let payoff_matrix = PayoffMatrix {
        players: vec!["P1".to_string(), "P2".to_string()],
        strategies,
        payoffs,
        dimension: vec![2, 2],
    };

    GameState {
        game_type: GameType::SealedBidAuction,
        players: vec![],
        market_context: MarketContext {
            regime: MarketRegime::LowVolatility,
            volatility: 0.1,
            liquidity: 1_000_000.0,
            volume: 500_000.0,
            spread: 0.01,
            market_impact: 0.001,
            information_asymmetry: 0.1,
            regulatory_environment: RegulatoryEnvironment {
                short_selling_allowed: true,
                position_limits: None,
                circuit_breakers: true,
                market_making_obligations: false,
                transparency_requirements: TransparencyLevel::Full,
            },
        },
        information_sets: HashMap::new(),
        action_history: vec![],
        current_round: 0,
        payoff_matrix: Some(payoff_matrix),
        nash_equilibria: vec![],
        nash_equilibrium_found: false,
        dominant_strategies: HashMap::new(),
        cooperation_level: 0.5,
        competition_intensity: 0.7,
    }
}

fn benchmark_mechanism_designer_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("mechanism_designer_creation");

    let configs = [
        ("ic_ir_bb", true, true, true),
        ("ic_ir", true, true, false),
        ("ic_only", true, false, false),
        ("none", false, false, false),
    ];

    for (name, ic, ir, bb) in configs {
        group.bench_with_input(
            BenchmarkId::new("new", name),
            &(ic, ir, bb),
            |b, &(ic, ir, bb)| {
                b.iter(|| black_box(MechanismDesigner::new(black_box(ic), black_box(ir), black_box(bb))))
            },
        );
    }

    group.finish();
}

fn benchmark_optimal_mechanism_design(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimal_mechanism_design");

    let designer = MechanismDesigner::new(true, true, false);

    let objectives = [
        ("revenue_max", MechanismObjectives {
            maximize_revenue: true,
            ensure_fairness: false,
            minimize_manipulation: false,
        }),
        ("fair", MechanismObjectives {
            maximize_revenue: false,
            ensure_fairness: true,
            minimize_manipulation: false,
        }),
        ("anti_manip", MechanismObjectives {
            maximize_revenue: false,
            ensure_fairness: false,
            minimize_manipulation: true,
        }),
        ("balanced", MechanismObjectives {
            maximize_revenue: true,
            ensure_fairness: true,
            minimize_manipulation: true,
        }),
    ];

    for (name, obj) in objectives {
        group.bench_with_input(
            BenchmarkId::new("design", name),
            &obj,
            |b, obj| {
                b.iter(|| designer.design_optimal_mechanism(black_box(obj)))
            },
        );
    }

    group.finish();
}

fn benchmark_mechanism_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("mechanism_verification");

    let designer = MechanismDesigner::new(true, true, false);
    let game_state = create_game_state();

    let mechanisms = [
        ("vcg", Mechanism {
            allocation_rule: AllocationRule::Efficient,
            payment_rule: PaymentRule::VCG,
            properties: MechanismProperties {
                incentive_compatible: true,
                individually_rational: true,
                budget_balanced: false,
                efficient: true,
            },
        }),
        ("first_price", Mechanism {
            allocation_rule: AllocationRule::Efficient,
            payment_rule: PaymentRule::FirstPrice,
            properties: MechanismProperties {
                incentive_compatible: false,
                individually_rational: true,
                budget_balanced: true,
                efficient: false,
            },
        }),
        ("second_price", Mechanism {
            allocation_rule: AllocationRule::Efficient,
            payment_rule: PaymentRule::SecondPrice,
            properties: MechanismProperties {
                incentive_compatible: true,
                individually_rational: true,
                budget_balanced: false,
                efficient: true,
            },
        }),
        ("proportional", Mechanism {
            allocation_rule: AllocationRule::Fair,
            payment_rule: PaymentRule::Proportional,
            properties: MechanismProperties {
                incentive_compatible: false,
                individually_rational: true,
                budget_balanced: true,
                efficient: false,
            },
        }),
    ];

    for (name, mechanism) in mechanisms {
        group.bench_with_input(
            BenchmarkId::new("verify", name),
            &mechanism,
            |b, mech| {
                b.iter(|| designer.verify_mechanism(black_box(mech), black_box(&game_state)))
            },
        );
    }

    group.finish();
}

fn benchmark_auction_mechanism_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("auction_mechanism_creation");

    let auction_types = [
        ("english", AuctionType::English),
        ("dutch", AuctionType::Dutch),
        ("first_price", AuctionType::FirstPrice),
        ("second_price", AuctionType::SecondPrice),
        ("vickrey", AuctionType::Vickrey),
        ("double", AuctionType::Double),
    ];

    for (name, auction_type) in auction_types {
        group.bench_with_input(
            BenchmarkId::new("new", name),
            &auction_type,
            |b, &atype| {
                b.iter(|| black_box(AuctionMechanism::new(black_box(atype), black_box(10.0), black_box(1.0))))
            },
        );
    }

    group.finish();
}

fn benchmark_auction_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("auction_execution");

    let auction_types = [
        ("english", AuctionType::English),
        ("dutch", AuctionType::Dutch),
        ("vickrey", AuctionType::Vickrey),
        ("first_price", AuctionType::FirstPrice),
        ("second_price", AuctionType::SecondPrice),
    ];

    for bidder_count in [5, 10, 25, 50] {
        let bidders = create_bidders(bidder_count);
        group.throughput(Throughput::Elements(bidder_count as u64));

        for (name, auction_type) in &auction_types {
            let auction = AuctionMechanism::new(*auction_type, 10.0, 1.0);
            group.bench_with_input(
                BenchmarkId::new(format!("{}_{}", name, bidder_count), bidder_count),
                &bidders,
                |b, bidders| {
                    b.iter(|| auction.run_auction(black_box(bidders)))
                },
            );
        }
    }

    group.finish();
}

fn benchmark_optimal_reserve_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimal_reserve_price");

    let auction = AuctionMechanism::new(AuctionType::Vickrey, 10.0, 1.0);

    let distributions = [
        ("uniform", ValueDistribution {
            distribution_type: "uniform".to_string(),
            parameters: {
                let mut p = HashMap::new();
                p.insert("min".to_string(), 0.0);
                p.insert("max".to_string(), 100.0);
                p
            },
        }),
        ("normal", ValueDistribution {
            distribution_type: "normal".to_string(),
            parameters: {
                let mut p = HashMap::new();
                p.insert("mean".to_string(), 50.0);
                p.insert("std".to_string(), 15.0);
                p
            },
        }),
        ("exponential", ValueDistribution {
            distribution_type: "exponential".to_string(),
            parameters: {
                let mut p = HashMap::new();
                p.insert("lambda".to_string(), 0.02);
                p
            },
        }),
    ];

    for (name, dist) in distributions {
        group.bench_with_input(
            BenchmarkId::new("calculate", name),
            &dist,
            |b, dist| {
                b.iter(|| auction.calculate_optimal_reserve(black_box(dist)))
            },
        );
    }

    group.finish();
}

fn benchmark_bidder_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("bidder_creation");

    for count in [10, 50, 100, 500] {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("create", count),
            &count,
            |b, &count| {
                b.iter(|| black_box(create_bidders(black_box(count))))
            },
        );
    }

    group.finish();
}

fn benchmark_allocation_rules(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation_rules");

    let rules = [
        ("efficient", AllocationRule::Efficient),
        ("fair", AllocationRule::Fair),
        ("random", AllocationRule::Random),
        ("priority", AllocationRule::Priority),
    ];

    for (name, rule) in rules {
        group.bench_with_input(
            BenchmarkId::new("construct_mechanism", name),
            &rule,
            |b, &rule| {
                b.iter(|| black_box(Mechanism {
                    allocation_rule: rule,
                    payment_rule: PaymentRule::VCG,
                    properties: MechanismProperties {
                        incentive_compatible: true,
                        individually_rational: true,
                        budget_balanced: false,
                        efficient: matches!(rule, AllocationRule::Efficient),
                    },
                }))
            },
        );
    }

    group.finish();
}

fn benchmark_payment_rules(c: &mut Criterion) {
    let mut group = c.benchmark_group("payment_rules");

    let rules = [
        ("vcg", PaymentRule::VCG),
        ("first_price", PaymentRule::FirstPrice),
        ("second_price", PaymentRule::SecondPrice),
        ("proportional", PaymentRule::Proportional),
    ];

    for (name, rule) in rules {
        group.bench_with_input(
            BenchmarkId::new("construct_mechanism", name),
            &rule,
            |b, &rule| {
                b.iter(|| black_box(Mechanism {
                    allocation_rule: AllocationRule::Efficient,
                    payment_rule: rule,
                    properties: MechanismProperties {
                        incentive_compatible: matches!(rule, PaymentRule::VCG | PaymentRule::SecondPrice),
                        individually_rational: true,
                        budget_balanced: matches!(rule, PaymentRule::FirstPrice),
                        efficient: true,
                    },
                }))
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_mechanism_designer_creation,
    benchmark_optimal_mechanism_design,
    benchmark_mechanism_verification,
    benchmark_auction_mechanism_creation,
    benchmark_auction_execution,
    benchmark_optimal_reserve_calculation,
    benchmark_bidder_creation,
    benchmark_allocation_rules,
    benchmark_payment_rules,
);

criterion_main!(benches);
