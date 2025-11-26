//! Benchmark tests for Syndicate Management MCP tools
//!
//! Tests scalability and performance with varying member counts

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::time::Duration;

// Import syndicate modules
use nt_syndicate::{
    FundAllocationEngine, ProfitDistributionSystem, WithdrawalManager,
    MemberManager, VotingSystem, MemberPerformanceTracker,
    AllocationStrategy, DistributionModel, BettingOpportunity,
    MemberRole,
};

fn bench_member_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("member_operations");

    for member_count in [10, 50, 100, 500, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("add_members", member_count),
            member_count,
            |b, &count| {
                b.iter(|| {
                    let manager = MemberManager::new("bench-syndicate".to_string());
                    for i in 0..count {
                        let _ = manager.add_member(
                            format!("Member {}", i),
                            format!("member{}@test.com", i),
                            MemberRole::ContributingMember,
                            "1000.00".to_string(),
                        );
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_fund_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("fund_allocation");

    for member_count in [10, 50, 100].iter() {
        let mut engine = FundAllocationEngine::new(
            "bench-syndicate".to_string(),
            format!("{}.00", member_count * 1000),
        ).unwrap();

        let opportunity = BettingOpportunity {
            sport: "football".to_string(),
            event: "Test Event".to_string(),
            bet_type: "moneyline".to_string(),
            selection: "Team A".to_string(),
            odds: 2.0,
            probability: 0.55,
            edge: 0.10,
            confidence: 0.80,
            model_agreement: 0.90,
            time_until_event_secs: 3600,
            liquidity: 50000.0,
            is_live: false,
            is_parlay: false,
        };

        group.bench_with_input(
            BenchmarkId::new("kelly_criterion", member_count),
            member_count,
            |b, _| {
                b.iter(|| {
                    let _ = engine.allocate_funds(
                        black_box(opportunity.clone()),
                        AllocationStrategy::KellyCriterion,
                    );
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("dynamic_confidence", member_count),
            member_count,
            |b, _| {
                b.iter(|| {
                    let _ = engine.allocate_funds(
                        black_box(opportunity.clone()),
                        AllocationStrategy::DynamicConfidence,
                    );
                });
            },
        );
    }

    group.finish();
}

fn bench_profit_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("profit_distribution");

    for member_count in [10, 50, 100, 500, 1000].iter() {
        let mut system = ProfitDistributionSystem::new("bench-syndicate".to_string());
        let manager = MemberManager::new("bench-syndicate".to_string());

        // Add members
        for i in 0..*member_count {
            let _ = manager.add_member(
                format!("Member {}", i),
                format!("member{}@test.com", i),
                MemberRole::ContributingMember,
                format!("{}.00", (i + 1) * 1000),
            );
        }

        let members_json = manager.list_members(true).unwrap();

        group.bench_with_input(
            BenchmarkId::new("hybrid_distribution", member_count),
            member_count,
            |b, _| {
                b.iter(|| {
                    let _ = system.calculate_distribution(
                        "10000.00".to_string(),
                        black_box(members_json.clone()),
                        DistributionModel::Hybrid,
                    );
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("proportional_distribution", member_count),
            member_count,
            |b, _| {
                b.iter(|| {
                    let _ = system.calculate_distribution(
                        "10000.00".to_string(),
                        black_box(members_json.clone()),
                        DistributionModel::Proportional,
                    );
                });
            },
        );
    }

    group.finish();
}

fn bench_voting_system(c: &mut Criterion) {
    let mut group = c.benchmark_group("voting_system");

    for member_count in [10, 50, 100, 500].iter() {
        let voting = VotingSystem::new("bench-syndicate".to_string());

        // Create vote
        let vote_id = voting.create_vote(
            "strategy_change".to_string(),
            r#"{"description":"Change allocation strategy"}"#.to_string(),
            uuid::Uuid::new_v4().to_string(),
            Some(48),
        ).unwrap();

        group.bench_with_input(
            BenchmarkId::new("cast_votes", member_count),
            member_count,
            |b, &count| {
                b.iter(|| {
                    for i in 0..count {
                        let member_id = uuid::Uuid::new_v4().to_string();
                        let _ = voting.cast_vote(
                            vote_id.clone(),
                            member_id,
                            if i % 2 == 0 { "approve" } else { "reject" }.to_string(),
                            1.0,
                        );
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("get_vote_results", member_count),
            member_count,
            |b, _| {
                b.iter(|| {
                    let _ = voting.get_vote_results(black_box(vote_id.clone()));
                });
            },
        );
    }

    group.finish();
}

fn bench_withdrawal_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("withdrawal_processing");

    for withdrawal_count in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("process_withdrawals", withdrawal_count),
            withdrawal_count,
            |b, &count| {
                b.iter(|| {
                    let mut manager = WithdrawalManager::new("bench-syndicate".to_string());
                    for i in 0..count {
                        let member_id = uuid::Uuid::new_v4().to_string();
                        let _ = manager.request_withdrawal(
                            member_id,
                            "10000.00".to_string(),
                            format!("{}.00", (i + 1) * 100),
                            i % 5 == 0, // Every 5th is emergency
                        );
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_performance_tracking(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_tracking");

    for bet_count in [100, 500, 1000].iter() {
        let tracker = MemberPerformanceTracker::new();
        let member_id = uuid::Uuid::new_v4().to_string();

        // Add historical bets
        for i in 0..*bet_count {
            let bet_details = serde_json::json!({
                "bet_id": format!("bet_{}", i),
                "sport": "football",
                "bet_type": "moneyline",
                "odds": 2.0,
                "stake": "100",
                "outcome": if i % 2 == 0 { "won" } else { "lost" },
                "profit": if i % 2 == 0 { "100" } else { "-100" },
                "confidence": 0.8,
                "edge": 0.1,
            }).to_string();

            let _ = tracker.track_bet_outcome(member_id.clone(), bet_details);
        }

        group.bench_with_input(
            BenchmarkId::new("identify_strengths", bet_count),
            bet_count,
            |b, _| {
                b.iter(|| {
                    let _ = tracker.identify_member_strengths(black_box(member_id.clone()));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("get_performance_history", bet_count),
            bet_count,
            |b, _| {
                b.iter(|| {
                    let _ = tracker.get_performance_history(black_box(member_id.clone()));
                });
            },
        );
    }

    group.finish();
}

fn bench_concurrent_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_operations");
    group.measurement_time(Duration::from_secs(10));

    for member_count in [50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("full_syndicate_workflow", member_count),
            member_count,
            |b, &count| {
                b.iter(|| {
                    // Create syndicate
                    let manager = MemberManager::new("bench-syndicate".to_string());
                    let mut engine = FundAllocationEngine::new(
                        "bench-syndicate".to_string(),
                        format!("{}.00", count * 1000),
                    ).unwrap();
                    let mut profit_system = ProfitDistributionSystem::new("bench-syndicate".to_string());

                    // Add members
                    for i in 0..count {
                        let _ = manager.add_member(
                            format!("Member {}", i),
                            format!("member{}@test.com", i),
                            MemberRole::ContributingMember,
                            "1000.00".to_string(),
                        );
                    }

                    // Allocate funds
                    let opportunity = BettingOpportunity {
                        sport: "football".to_string(),
                        event: "Test Event".to_string(),
                        bet_type: "moneyline".to_string(),
                        selection: "Team A".to_string(),
                        odds: 2.0,
                        probability: 0.55,
                        edge: 0.10,
                        confidence: 0.80,
                        model_agreement: 0.90,
                        time_until_event_secs: 3600,
                        liquidity: 50000.0,
                        is_live: false,
                        is_parlay: false,
                    };

                    let _ = engine.allocate_funds(opportunity, AllocationStrategy::KellyCriterion);

                    // Distribute profits
                    let members_json = manager.list_members(true).unwrap();
                    let _ = profit_system.calculate_distribution(
                        "5000.00".to_string(),
                        members_json,
                        DistributionModel::Hybrid,
                    );
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_member_operations,
    bench_fund_allocation,
    bench_profit_distribution,
    bench_voting_system,
    bench_withdrawal_processing,
    bench_performance_tracking,
    bench_concurrent_operations,
);

criterion_main!(benches);
