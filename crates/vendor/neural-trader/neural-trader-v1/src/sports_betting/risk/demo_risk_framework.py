"""
Demo Script for Sports Betting Risk Management Framework

Comprehensive demonstration of:
- Kelly Criterion optimization
- Portfolio risk management
- Circuit breaker systems
- Performance monitoring
- Compliance framework
- Integrated risk system

This script demonstrates the complete risk management workflow for a sports betting syndicate.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from datetime import datetime, timedelta
import json
import time

from sports_betting.risk.integrated_risk_system import IntegratedRiskSystem
from sports_betting.risk.kelly_criterion import BettingOpportunity
from sports_betting.risk.compliance import Customer, KYCDocument, DocumentType
from sports_betting.risk.performance_monitor import BettingTransaction


def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title.upper()}")
    print("=" * 80)


def print_subsection(title: str):
    """Print formatted subsection header"""
    print(f"\n--- {title} ---")


def demo_kelly_criterion():
    """Demonstrate Kelly Criterion optimization"""
    print_section("Kelly Criterion Optimization Demo")
    
    from sports_betting.risk.kelly_criterion import KellyCriterionOptimizer
    
    # Initialize optimizer
    optimizer = KellyCriterionOptimizer(bankroll=50000, fractional_factor=0.25)
    
    # Create sample betting opportunities
    opportunities = [
        BettingOpportunity(
            bet_id="nfl_game_1",
            odds=1.91,  # -110 American odds
            probability=0.55,
            confidence=0.8,
            sport="NFL",
            event="Chiefs vs Bills",
            selection="Chiefs -3.5"
        ),
        BettingOpportunity(
            bet_id="nba_game_1",
            odds=2.10,  # +110 American odds
            probability=0.52,
            confidence=0.7,
            sport="NBA", 
            event="Lakers vs Warriors",
            selection="Lakers +2.5"
        ),
        BettingOpportunity(
            bet_id="nhl_game_1",
            odds=1.85,  # -118 American odds
            probability=0.58,
            confidence=0.9,
            sport="NHL",
            event="Bruins vs Rangers", 
            selection="Over 6.5 goals"
        )
    ]
    
    print("Betting Opportunities:")
    for opp in opportunities:
        print(f"  {opp.bet_id}: {opp.selection} @ {opp.odds} (Edge: {opp.edge:.2%})")
    
    # Calculate individual Kelly results
    print_subsection("Individual Kelly Analysis")
    for opp in opportunities:
        result = optimizer.calculate_single_kelly(opp)
        print(f"{opp.bet_id}:")
        print(f"  Kelly Fraction: {result.kelly_fraction:.3f}")
        print(f"  Fractional Kelly: {result.fractional_kelly:.3f}")
        print(f"  Recommended Stake: ${result.recommended_stake:.2f}")
        print(f"  Expected Growth: {result.expected_growth:.4f}")
        if result.warnings:
            print(f"  Warnings: {', '.join(result.warnings)}")
        print()
    
    # Portfolio optimization
    print_subsection("Portfolio Optimization")
    portfolio_results = optimizer.optimize_portfolio(opportunities)
    
    print("Optimized Portfolio Allocation:")
    for result in portfolio_results:
        print(f"  {result.bet_id}: ${result.recommended_stake:.2f} ({result.fractional_kelly:.2%} of bankroll)")
    
    # Portfolio summary
    summary = optimizer.get_portfolio_summary(portfolio_results)
    print(f"\nPortfolio Summary:")
    print(f"  Total Allocation: ${summary['total_allocation']:.2f} ({summary['total_fraction']:.2%})")
    print(f"  Number of Bets: {summary['num_bets']}")
    print(f"  Portfolio Edge: {summary['portfolio_edge']:.3f}")
    print(f"  Expected Growth: {summary['expected_growth']:.4f}")
    print(f"  Diversification Ratio: {summary['diversification_ratio']:.2f}")


def demo_portfolio_management():
    """Demonstrate portfolio risk management"""
    print_section("Portfolio Risk Management Demo")
    
    from sports_betting.risk.portfolio_manager import PortfolioRiskManager, Position
    
    # Initialize portfolio manager
    manager = PortfolioRiskManager(initial_bankroll=75000)
    
    # Create sample positions
    positions = [
        Position(
            bet_id="nfl_001",
            sport="NFL",
            event="Chiefs vs Bills", 
            selection="Chiefs -3.5",
            stake=2000,
            odds=1.91,
            probability=0.55,
            entry_time=datetime.now() - timedelta(hours=2)
        ),
        Position(
            bet_id="nba_001", 
            sport="NBA",
            event="Lakers vs Warriors",
            selection="Over 215.5",
            stake=1500,
            odds=1.85,
            probability=0.60,
            entry_time=datetime.now() - timedelta(hours=1)
        ),
        Position(
            bet_id="nfl_002",
            sport="NFL", 
            event="Cowboys vs Eagles",
            selection="Cowboys +7",
            stake=1800,
            odds=1.95,
            probability=0.53,
            entry_time=datetime.now() - timedelta(minutes=30)
        )
    ]
    
    # Add positions to portfolio
    print("Adding positions to portfolio:")
    for pos in positions:
        manager.add_position(pos)
        print(f"  Added: {pos.bet_id} - ${pos.stake:.2f}")
    
    # Analyze portfolio risk
    print_subsection("Portfolio Risk Analysis")
    risk_metrics = manager.analyze_portfolio_risk()
    
    print(f"Risk Level: {risk_metrics.risk_level.value.upper()}")
    print(f"VaR (1-day): ${risk_metrics.var_1d:.2f}")
    print(f"CVaR (1-day): ${risk_metrics.cvar_1d:.2f}")
    print(f"Max Drawdown: {risk_metrics.max_drawdown:.2%}")
    print(f"Concentration Ratio: {risk_metrics.concentration_ratio:.3f}")
    print(f"Correlation Risk: {risk_metrics.correlation_risk:.3f}")
    print(f"Diversification Ratio: {risk_metrics.diversification_ratio:.3f}")
    print(f"Leverage Ratio: {risk_metrics.leverage_ratio:.2%}")
    
    if risk_metrics.warnings:
        print("\nWarnings:")
        for warning in risk_metrics.warnings:
            print(f"  - {warning}")
    
    if risk_metrics.recommendations:
        print("\nRecommendations:")
        for rec in risk_metrics.recommendations:
            print(f"  - {rec}")
    
    # Portfolio summary
    print_subsection("Portfolio Summary")
    summary = manager.get_portfolio_summary()
    print(json.dumps(summary, indent=2, default=str))


def demo_circuit_breakers():
    """Demonstrate circuit breaker system"""
    print_section("Circuit Breaker System Demo")
    
    from sports_betting.risk.circuit_breakers import CircuitBreakerSystem
    
    # Initialize circuit breaker system
    cb_system = CircuitBreakerSystem(initial_bankroll=100000)
    
    print("Initial Circuit Breaker Configuration:")
    status = cb_system.get_system_status()
    print(f"  System Halted: {status['system_status']['halted']}")
    print(f"  Emergency Stop: {status['system_status']['emergency_stop']}")
    print(f"  Total Triggers: {status['system_status']['total_triggers']}")
    
    # Simulate trading session with losses
    print_subsection("Simulating Trading Session")
    
    scenarios = [
        ("Initial", 100000, None),
        ("Small Loss", 98000, "loss"),
        ("Another Loss", 95000, "loss"), 
        ("Bigger Loss", 90000, "loss"),
        ("Consecutive Loss", 85000, "loss"),
        ("Major Loss", 75000, "loss"),  # This should trigger drawdown
        ("Recovery", 78000, "win"),
        ("Final Loss", 70000, "loss")   # This should trigger multiple breakers
    ]
    
    for description, new_bankroll, result in scenarios:
        print(f"\n{description}: ${new_bankroll:,.2f}")
        cb_system.update_bankroll(new_bankroll, result)
        
        # Check for triggered breakers
        triggers = cb_system.check_all_breakers()
        
        if triggers:
            print(f"  ⚠️  Triggered {len(triggers)} circuit breaker(s):")
            for trigger in triggers:
                print(f"    - {trigger.breaker_id}: {trigger.message}")
                print(f"      Action: {trigger.action.value}")
        else:
            print("  ✅ No circuit breakers triggered")
        
        # Show key metrics
        current_dd = (cb_system.peak_bankroll - new_bankroll) / cb_system.peak_bankroll
        print(f"  Current Drawdown: {current_dd:.1%}")
        print(f"  Consecutive Losses: {cb_system.consecutive_losses}")
        
        if cb_system.system_halted:
            print("  🛑 SYSTEM HALTED")
            break
        if cb_system.emergency_stop:
            print("  🚨 EMERGENCY STOP ACTIVATED")
            break
    
    # Final system status
    print_subsection("Final Circuit Breaker Status")
    final_status = cb_system.get_system_status()
    print(json.dumps(final_status, indent=2, default=str))


def demo_performance_monitoring():
    """Demonstrate performance monitoring"""
    print_section("Performance Monitoring Demo")
    
    from sports_betting.risk.performance_monitor import PerformanceMonitor, BettingTransaction
    
    # Initialize performance monitor
    monitor = PerformanceMonitor(initial_bankroll=50000)
    
    # Create sample transaction history
    base_time = datetime.now() - timedelta(days=30)
    
    transactions = [
        BettingTransaction("bet_001", base_time + timedelta(days=1), "NFL", "Chiefs vs Bills", "Chiefs -3.5", "spread", 1000, 1.91, "win", 910),
        BettingTransaction("bet_002", base_time + timedelta(days=2), "NBA", "Lakers vs Warriors", "Over 215", "total", 800, 1.85, "loss", -800),
        BettingTransaction("bet_003", base_time + timedelta(days=3), "NHL", "Bruins vs Rangers", "Bruins ML", "moneyline", 1200, 2.10, "win", 1320),
        BettingTransaction("bet_004", base_time + timedelta(days=5), "NFL", "Cowboys vs Eagles", "Under 45.5", "total", 1500, 1.95, "loss", -1500),
        BettingTransaction("bet_005", base_time + timedelta(days=7), "NBA", "Celtics vs Heat", "Celtics +2", "spread", 900, 1.90, "win", 810),
        BettingTransaction("bet_006", base_time + timedelta(days=10), "NFL", "49ers vs Rams", "49ers -7", "spread", 2000, 1.91, "win", 1820),
        BettingTransaction("bet_007", base_time + timedelta(days=12), "NBA", "Nets vs Knicks", "Over 220", "total", 1100, 1.85, "loss", -1100),
        BettingTransaction("bet_008", base_time + timedelta(days=15), "NHL", "Penguins vs Caps", "Penguins ML", "moneyline", 1300, 1.95, "win", 1235),
        BettingTransaction("bet_009", base_time + timedelta(days=18), "NFL", "Packers vs Bears", "Packers -10", "spread", 1800, 1.90, "loss", -1800),
        BettingTransaction("bet_010", base_time + timedelta(days=20), "NBA", "Suns vs Mavs", "Under 230", "total", 950, 1.95, "win", 902.50),
    ]
    
    print("Recording transaction history:")
    for transaction in transactions:
        monitor.record_transaction(transaction)
        print(f"  {transaction.transaction_id}: {transaction.result} - P&L: ${transaction.pnl:.2f}")
    
    # Calculate performance metrics
    print_subsection("Performance Metrics")
    metrics = monitor.calculate_performance_metrics()
    
    print(f"Total P&L: ${metrics.total_pnl:.2f}")
    print(f"Total ROI: {metrics.total_roi:.2%}")
    print(f"Win Rate: {metrics.win_rate:.1%}")
    print(f"Average Win: ${metrics.avg_win:.2f}")
    print(f"Average Loss: ${metrics.avg_loss:.2f}")
    print(f"Profit Factor: {metrics.profit_factor:.2f}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"Current Streak: {metrics.current_streak} {metrics.current_streak_type}")
    print(f"Kelly Criterion: {metrics.kelly_criterion:.3f}")
    print(f"Expectancy: ${metrics.expectancy:.2f}")
    
    # Check for alerts
    print_subsection("Performance Alerts")
    alerts = monitor.check_performance_alerts()
    
    if alerts:
        print(f"Generated {len(alerts)} alert(s):")
        for alert in alerts:
            print(f"  - {alert.alert_type.value}: {alert.message} (Severity: {alert.severity})")
    else:
        print("No performance alerts generated")
    
    # Performance summary
    print_subsection("Performance Summary")
    summary = monitor.get_performance_summary()
    print(json.dumps(summary, indent=2, default=str))


def demo_compliance_framework():
    """Demonstrate compliance framework"""
    print_section("Compliance Framework Demo")
    
    from sports_betting.risk.compliance import ComplianceFramework, Customer, KYCDocument, DocumentType
    
    # Initialize compliance framework
    compliance = ComplianceFramework()
    
    # Register customers
    print("Registering customers:")
    
    customers = [
        Customer(
            customer_id="cust_001",
            first_name="John",
            last_name="Doe", 
            date_of_birth=datetime(1985, 3, 15),
            nationality="US",
            residence_country="US",
            residence_state="Nevada"
        ),
        Customer(
            customer_id="cust_002",
            first_name="Jane",
            last_name="Smith",
            date_of_birth=datetime(1990, 7, 22),
            nationality="UK",
            residence_country="UK",
            residence_state=None
        ),
        Customer(
            customer_id="cust_003",
            first_name="Bob",
            last_name="Johnson",
            date_of_birth=datetime(1975, 11, 8),
            nationality="US", 
            residence_country="US",
            residence_state="New Jersey"
        )
    ]
    
    for customer in customers:
        compliance.register_customer(customer)
        print(f"  Registered: {customer.first_name} {customer.last_name} ({customer.customer_id})")
    
    # Complete KYC for some customers
    print_subsection("KYC Document Verification")
    
    kyc_docs = [
        KYCDocument(
            document_id="doc_001",
            document_type=DocumentType.DRIVERS_LICENSE,
            document_number="NV123456789",
            issue_date=datetime(2020, 1, 15),
            expiry_date=datetime(2025, 1, 15),
            issuing_authority="Nevada DMV"
        ),
        KYCDocument(
            document_id="doc_002", 
            document_type=DocumentType.UTILITY_BILL,
            document_number="UTIL001",
            issue_date=datetime(2023, 10, 1),
            expiry_date=None,
            issuing_authority="Nevada Energy"
        )
    ]
    
    for doc in kyc_docs:
        compliance.verify_kyc_document("cust_001", doc)
        print(f"  Verified: {doc.document_type.value} for cust_001")
    
    # Test transaction compliance
    print_subsection("Transaction Compliance Checks")
    
    compliance_tests = [
        ("cust_001", 5000, "NFL", "spread", "Nevada", 0),  # Should pass
        ("cust_002", 1000, "NFL", "spread", "Nevada", 0),  # No KYC
        ("cust_001", 60000, "NFL", "spread", "Nevada", 0), # Exceeds limits
        ("cust_003", 2000, "college_in_state", "spread", "New Jersey", 0), # Restricted sport
    ]
    
    for customer_id, amount, sport, bet_type, jurisdiction, daily_loss in compliance_tests:
        status, violations = compliance.check_transaction_compliance(
            customer_id, amount, sport, bet_type, jurisdiction, daily_loss
        )
        
        print(f"  {customer_id} - ${amount:.0f} {sport} bet:")
        print(f"    Status: {status.value}")
        if violations:
            print(f"    Violations: {', '.join(violations)}")
        else:
            print("    ✅ Compliant")
        print()
    
    # Compliance dashboard
    print_subsection("Compliance Dashboard")
    dashboard = compliance.get_compliance_dashboard()
    print(json.dumps(dashboard, indent=2, default=str))


def demo_integrated_system():
    """Demonstrate integrated risk management system"""
    print_section("Integrated Risk Management System Demo")
    
    # Initialize integrated system
    risk_system = IntegratedRiskSystem(
        initial_bankroll=100000,
        syndicate_name="Demo Syndicate",
        risk_tolerance="moderate",
        jurisdiction="Nevada"
    )
    
    print(f"Initialized risk system: {risk_system.syndicate_name}")
    print(f"Bankroll: ${risk_system.current_bankroll:,.2f}")
    print(f"Risk Tolerance: {risk_system.risk_tolerance}")
    print(f"Jurisdiction: {risk_system.jurisdiction}")
    
    # Register a customer for compliance testing
    customer = Customer(
        customer_id="demo_cust_001",
        first_name="Demo",
        last_name="User",
        date_of_birth=datetime(1985, 5, 15),
        nationality="US",
        residence_country="US", 
        residence_state="Nevada"
    )
    risk_system.compliance.register_customer(customer)
    
    # Add KYC documents
    kyc_docs = [
        KYCDocument("demo_doc_1", DocumentType.DRIVERS_LICENSE, "NV987654321", 
                   datetime(2020, 1, 1), datetime(2025, 1, 1), "Nevada DMV"),
        KYCDocument("demo_doc_2", DocumentType.UTILITY_BILL, "BILL123",
                   datetime(2023, 11, 1), None, "NV Energy")
    ]
    
    for doc in kyc_docs:
        risk_system.compliance.verify_kyc_document("demo_cust_001", doc)
    
    # Create betting opportunities
    print_subsection("Evaluating Betting Opportunities")
    
    opportunities = [
        BettingOpportunity(
            bet_id="demo_nfl_001",
            odds=1.91,
            probability=0.58,
            confidence=0.85,
            sport="NFL",
            event="Chiefs vs Bills",
            selection="Chiefs -3.5"
        ),
        BettingOpportunity(
            bet_id="demo_nba_001", 
            odds=2.05,
            probability=0.52,
            confidence=0.75,
            sport="NBA",
            event="Lakers vs Warriors",
            selection="Lakers +2.5"
        ),
        BettingOpportunity(
            bet_id="demo_nhl_001",
            odds=1.85,
            probability=0.60,
            confidence=0.90,
            sport="NHL",
            event="Bruins vs Rangers",
            selection="Over 6.5 goals"
        )
    ]
    
    # Evaluate each opportunity
    decisions = []
    for opp in opportunities:
        print(f"\nEvaluating: {opp.bet_id} - {opp.selection}")
        
        decision = risk_system.evaluate_betting_opportunity(
            opportunity=opp,
            customer_id="demo_cust_001",
            sport=opp.sport,
            bet_type="single"
        )
        
        decisions.append(decision)
        
        print(f"  Decision: {'✅ APPROVED' if decision.approved else '❌ REJECTED'}")
        print(f"  Recommended Stake: ${decision.recommended_stake:.2f}")
        print(f"  Kelly Fraction: {decision.kelly_fraction:.3f}")
        print(f"  Risk Score: {decision.risk_score:.2f}")
        print(f"  Confidence: {decision.confidence_score:.2f}")
        print(f"  Compliance: {decision.compliance_status.value}")
        
        if decision.warnings:
            print(f"  Warnings:")
            for warning in decision.warnings[:3]:  # Show first 3 warnings
                print(f"    - {warning}")
        
        # Execute approved bets
        if decision.approved:
            success = risk_system.execute_approved_bet(decision)
            print(f"  Execution: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    # Simulate bet settlements
    print_subsection("Simulating Bet Settlements")
    
    settlement_scenarios = [
        ("demo_nfl_001", "win", 910),    # $1000 bet at 1.91 odds
        ("demo_nba_001", "loss", -800),  # $800 bet lost
        ("demo_nhl_001", "win", 765),    # $900 bet at 1.85 odds
    ]
    
    for bet_id, result, pnl in settlement_scenarios:
        print(f"\nSettling: {bet_id} - {result.upper()} (P&L: ${pnl:+.2f})")
        success = risk_system.settle_bet(bet_id, result, pnl)
        print(f"  Settlement: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    # System health check
    print_subsection("System Health Check")
    health_report = risk_system.perform_health_check()
    
    print(f"Overall Status: {health_report.overall_status.value.upper()}")
    print(f"Current Bankroll: ${health_report.bankroll_status['current']:,.2f}")
    print(f"Bankroll Change: {health_report.bankroll_status['change_percent']:+.2%}")
    print(f"Portfolio Risk Level: {health_report.portfolio_risk.get('risk_level', 'N/A')}")
    
    if health_report.active_alerts:
        print(f"\nActive Alerts ({len(health_report.active_alerts)}):")
        for alert in health_report.active_alerts[:5]:
            print(f"  - {alert}")
    
    if health_report.recommendations:
        print(f"\nRecommendations:")
        for rec in health_report.recommendations[:3]:
            print(f"  - {rec}")
    
    # System dashboard
    print_subsection("System Dashboard")
    dashboard = risk_system.get_system_dashboard()
    
    print("Dashboard Summary:")
    print(f"  Syndicate: {dashboard['system_overview']['syndicate_name']}")
    print(f"  Status: {dashboard['system_overview']['status']}")
    print(f"  Risk Tolerance: {dashboard['system_overview']['risk_tolerance']}")
    print(f"  Total Decisions: {dashboard['decisions']['total']}")
    print(f"  Approval Rate: {dashboard['decisions']['recent_approval_rate']}")
    print(f"  Avg Confidence: {dashboard['decisions']['avg_confidence']:.2f}")
    print(f"  Active Alerts: {dashboard['alerts']['count']}")


def main():
    """Run complete risk management framework demonstration"""
    print("🎯 SPORTS BETTING RISK MANAGEMENT FRAMEWORK")
    print("   Comprehensive demonstration of Agent 4 deliverables")
    print("   Author: Agent 4 - Risk Management Specialist")
    print(f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run all demonstrations
        demo_kelly_criterion()
        demo_portfolio_management() 
        demo_circuit_breakers()
        demo_performance_monitoring()
        demo_compliance_framework()
        demo_integrated_system()
        
        print_section("Demo Complete")
        print("✅ All risk management components demonstrated successfully!")
        print("\nKey Features Demonstrated:")
        print("  • Kelly Criterion optimization with portfolio-level risk management")
        print("  • Multi-sport correlation analysis and diversification optimization")
        print("  • Automated circuit breaker systems with emergency shutdown")
        print("  • Real-time performance monitoring with risk-adjusted metrics")
        print("  • Comprehensive compliance framework with KYC/AML integration")
        print("  • Integrated risk system coordinating all components")
        
        print("\nRisk Management Framework Status: 🟢 FULLY OPERATIONAL")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()