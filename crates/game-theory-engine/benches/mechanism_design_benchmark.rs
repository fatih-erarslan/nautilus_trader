// Mechanism Design Benchmark for Game Theory Engine
// Copyright (c) 2025 TENGRI Trading Swarm

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use game_theory_engine::{
    MechanismDesign, AuctionMechanism, VickereyAuction, EnglishAuction, DutchAuction,
    SealedBidAuction, IncentiveCompatibility, StrategyProofness, RevenueConcept,
    AllocationRule, PaymentRule, Mechanism, BidProfile, Valuation, Agent
};
use std::time::Instant;

fn create_agents(count: usize) -> Vec<Agent> {
    (0..count).map(|i| Agent {
        id: i,
        valuation: 100.0 + (i as f64 * 10.0),
        budget: 1000.0 + (i as f64 * 100.0),
        strategy_type: if i % 2 == 0 { "truthful" } else { "strategic" }.to_string(),
        risk_aversion: 0.1 + (i as f64 * 0.05),
    }).collect()
}

fn create_bid_profiles(agent_count: usize, item_count: usize) -> Vec<BidProfile> {
    (0..agent_count).map(|i| BidProfile {
        agent_id: i,
        bids: (0..item_count).map(|j| 50.0 + (i + j) as f64 * 5.0).collect(),
        timestamp: chrono::Utc::now(),
    }).collect()
}

fn benchmark_vickrey_auction(c: &mut Criterion) {
    let mut group = c.benchmark_group("vickrey_auction");
    
    for agent_count in [5, 10, 25, 50, 100].iter() {
        let agents = create_agents(*agent_count);
        let auction = VickereyAuction::new();
        
        group.bench_with_input(BenchmarkId::new("single_item", agent_count), agent_count, |b, _| {
            b.iter(|| {
                auction.run_single_item_auction(&agents)
            })
        });
        
        let bids = create_bid_profiles(*agent_count, 5);
        group.bench_with_input(BenchmarkId::new("multi_item", agent_count), agent_count, |b, _| {
            b.iter(|| {
                auction.run_multi_item_auction(&bids)
            })
        });
    }
    group.finish();
}

fn benchmark_english_auction(c: &mut Criterion) {
    let mut group = c.benchmark_group("english_auction");
    
    for agent_count in [5, 10, 25, 50].iter() {
        let agents = create_agents(*agent_count);
        let auction = EnglishAuction::new();
        
        group.bench_with_input(BenchmarkId::new("ascending_price", agent_count), agent_count, |b, _| {
            b.iter(|| {
                auction.run_ascending_auction(&agents)
            })
        });
        
        group.bench_with_input(BenchmarkId::new("bidding_rounds", agent_count), agent_count, |b, _| {
            b.iter(|| {
                auction.simulate_bidding_rounds(&agents, 10)
            })
        });
    }
    group.finish();
}

fn benchmark_dutch_auction(c: &mut Criterion) {
    let mut group = c.benchmark_group("dutch_auction");
    
    for agent_count in [5, 10, 25, 50].iter() {
        let agents = create_agents(*agent_count);
        let auction = DutchAuction::new();
        
        group.bench_with_input(BenchmarkId::new("descending_price", agent_count), agent_count, |b, _| {
            b.iter(|| {
                auction.run_descending_auction(&agents)
            })
        });
        
        group.bench_with_input(BenchmarkId::new("stopping_strategy", agent_count), agent_count, |b, _| {
            b.iter(|| {
                auction.optimize_stopping_strategy(&agents)
            })
        });
    }
    group.finish();
}

fn benchmark_sealed_bid_auction(c: &mut Criterion) {
    let mut group = c.benchmark_group("sealed_bid_auction");
    
    for agent_count in [5, 10, 25, 50, 100].iter() {
        let agents = create_agents(*agent_count);
        let auction = SealedBidAuction::new();
        
        group.bench_with_input(BenchmarkId::new("first_price", agent_count), agent_count, |b, _| {
            b.iter(|| {
                auction.first_price_sealed_bid(&agents)
            })
        });
        
        group.bench_with_input(BenchmarkId::new("second_price", agent_count), agent_count, |b, _| {
            b.iter(|| {
                auction.second_price_sealed_bid(&agents)
            })
        });
    }
    group.finish();
}

fn benchmark_incentive_compatibility(c: &mut Criterion) {
    let mut group = c.benchmark_group("incentive_compatibility");
    
    let agents = create_agents(20);
    let ic_checker = IncentiveCompatibility::new();
    
    let mechanisms = vec![
        ("Vickrey", Mechanism::Vickrey),
        ("English", Mechanism::English),
        ("Dutch", Mechanism::Dutch),
        ("FirstPrice", Mechanism::FirstPriceSealed),
    ];
    
    for (name, mechanism) in mechanisms.iter() {
        group.bench_with_input(BenchmarkId::new("check_truthfulness", name), mechanism, |b, mech| {
            b.iter(|| {
                ic_checker.check_truthfulness(&agents, mech.clone())
            })
        });
        
        group.bench_with_input(BenchmarkId::new("verify_strategy_proof", name), mechanism, |b, mech| {
            b.iter(|| {
                ic_checker.verify_strategy_proofness(&agents, mech.clone())
            })
        });
    }
    
    group.finish();
}

fn benchmark_revenue_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("revenue_analysis");
    
    let agents = create_agents(30);
    let revenue_analyzer = RevenueConcept::new();
    
    group.bench_function("expected_revenue", |b| {
        b.iter(|| {
            revenue_analyzer.calculate_expected_revenue(&agents, &Mechanism::Vickrey)
        })
    });
    
    group.bench_function("revenue_comparison", |b| {
        b.iter(|| {
            revenue_analyzer.compare_mechanism_revenues(&agents)
        })
    });
    
    group.bench_function("optimal_reserve_price", |b| {
        b.iter(|| {
            revenue_analyzer.find_optimal_reserve_price(&agents)
        })
    });
    
    group.finish();
}

fn benchmark_allocation_rules(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation_rules");
    
    for agent_count in [10, 25, 50].iter() {
        let agents = create_agents(*agent_count);
        let allocator = AllocationRule::new();
        
        group.bench_with_input(BenchmarkId::new("efficient_allocation", agent_count), agent_count, |b, _| {
            b.iter(|| {
                allocator.compute_efficient_allocation(&agents)
            })
        });
        
        group.bench_with_input(BenchmarkId::new("welfare_maximizing", agent_count), agent_count, |b, _| {
            b.iter(|| {
                allocator.welfare_maximizing_allocation(&agents)
            })
        });
    }
    group.finish();
}

fn benchmark_payment_rules(c: &mut Criterion) {
    let mut group = c.benchmark_group("payment_rules");
    
    let agents = create_agents(25);
    let payment_calculator = PaymentRule::new();
    
    group.bench_function("vickrey_payments", |b| {
        b.iter(|| {
            payment_calculator.calculate_vickrey_payments(&agents)
        })
    });
    
    group.bench_function("clarke_pivot_payments", |b| {
        b.iter(|| {
            payment_calculator.calculate_clarke_pivot_payments(&agents)
        })
    });
    
    group.bench_function("myerson_payments", |b| {
        b.iter(|| {
            payment_calculator.calculate_myerson_payments(&agents)
        })
    });
    
    group.finish();
}

fn benchmark_mechanism_design(c: &mut Criterion) {
    let mut group = c.benchmark_group("mechanism_design");
    
    let agents = create_agents(40);
    let designer = MechanismDesign::new();
    
    group.bench_function("design_optimal_mechanism", |b| {
        b.iter(|| {
            designer.design_optimal_mechanism(&agents)
        })
    });
    
    group.bench_function("myerson_mechanism", |b| {
        b.iter(|| {
            designer.construct_myerson_mechanism(&agents)
        })
    });
    
    group.bench_function("revenue_maximizing_design", |b| {
        b.iter(|| {
            designer.design_revenue_maximizing_mechanism(&agents)
        })
    });
    
    group.bench_function("welfare_maximizing_design", |b| {
        b.iter(|| {
            designer.design_welfare_maximizing_mechanism(&agents)
        })
    });
    
    group.finish();
}

fn benchmark_strategic_behavior(c: &mut Criterion) {
    let mut group = c.benchmark_group("strategic_behavior");
    
    let agents = create_agents(20);
    let strategy_analyzer = StrategyProofness::new();
    
    group.bench_function("nash_equilibrium", |b| {
        b.iter(|| {
            strategy_analyzer.find_nash_equilibrium(&agents)
        })
    });
    
    group.bench_function("dominant_strategy", |b| {
        b.iter(|| {
            strategy_analyzer.check_dominant_strategy(&agents)
        })
    });
    
    group.bench_function("bayesian_equilibrium", |b| {
        b.iter(|| {
            strategy_analyzer.compute_bayesian_equilibrium(&agents)
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_vickrey_auction,
    benchmark_english_auction,
    benchmark_dutch_auction,
    benchmark_sealed_bid_auction,
    benchmark_incentive_compatibility,
    benchmark_revenue_analysis,
    benchmark_allocation_rules,
    benchmark_payment_rules,
    benchmark_mechanism_design,
    benchmark_strategic_behavior
);
criterion_main!(benches);