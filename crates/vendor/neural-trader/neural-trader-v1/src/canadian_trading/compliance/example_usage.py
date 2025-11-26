"""
Example usage of the Canadian Trading Compliance modules
Demonstrates integration of CIRO compliance, tax reporting, audit trail, and monitoring.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from .ciro_compliance import CIROCompliance, ClientIdentification, OrderType
from .tax_reporting import TaxReporting
from .audit_trail import AuditTrail, AuditEventType, AuditSeverity
from .monitoring import ComplianceMonitor, AlertSeverity


async def main():
    """Demonstrate compliance system usage"""
    
    # Initialize compliance components
    firm_id = "FIRM123"
    registration_number = "REG456"
    
    # 1. Initialize CIRO Compliance
    ciro_compliance = CIROCompliance(firm_id, registration_number)
    
    # 2. Initialize Tax Reporting
    tax_reporting = TaxReporting()
    
    # 3. Initialize Audit Trail
    audit_trail = AuditTrail(db_path="canadian_audit_trail.db")
    audit_trail.start()
    
    # 4. Initialize Compliance Monitor
    monitor_config = {
        'position_limits': {
            'default': {'max_value': 1000000, 'max_quantity': 10000},
            'RY.TO': {'max_value': 500000, 'max_quantity': 5000}
        },
        'concentration_limits': {
            'default': 25  # 25% max concentration
        }
    }
    compliance_monitor = ComplianceMonitor(monitor_config)
    compliance_monitor.start()
    
    print("Canadian Trading Compliance System Initialized")
    print("=" * 50)
    
    # Example 1: Client Onboarding with KYC
    print("\n1. Client Onboarding Example:")
    client_data = {
        'client_id': 'CL123456',
        'legal_name': 'John Smith',
        'account_type': 'individual',
        'sin_or_bn': '123 456 789',
        'address': {
            'street': '100 King Street West',
            'city': 'Toronto',
            'province': 'ON',
            'postal_code': 'M5H 1H1'
        },
        'phone': '416-555-1234',
        'email': 'john.smith@email.com',
        'occupation': 'Software Engineer',
        'employer': 'Tech Corp',
        'investment_knowledge': 'high',
        'risk_tolerance': 'medium',
        'net_worth_range': '500k-1M',
        'kyc_date': datetime.now()
    }
    
    # Validate client identification
    is_valid, errors = ciro_compliance.validate_client_identification(client_data)
    print(f"Client validation: {'Passed' if is_valid else 'Failed'}")
    if errors:
        print(f"Validation errors: {errors}")
    
    # Log client onboarding
    audit_trail.log_event(
        event_type=AuditEventType.CLIENT_ONBOARDING,
        event_data={'client_data': client_data, 'validation_result': is_valid},
        client_id=client_data['client_id'],
        regulatory_requirement='CIRO_KYC'
    )
    
    # Example 2: Best Execution Check
    print("\n2. Best Execution Example:")
    order = {
        'order_id': 'ORD789',
        'symbol': 'RY.TO',
        'type': 'limit',
        'side': 'buy',
        'quantity': 100,
        'limit_price': 140.50
    }
    
    market_data = {
        'TSX': {
            'RY.TO': {'bid': 140.25, 'ask': 140.35, 'bid_size': 500, 'ask_size': 300}
        },
        'NEO': {
            'RY.TO': {'bid': 140.20, 'ask': 140.30, 'bid_size': 200, 'ask_size': 400}
        }
    }
    
    best_execution = await ciro_compliance.ensure_best_execution(order, market_data)
    print(f"Best execution venue: {best_execution['primary_venue']}")
    print(f"Best price: ${best_execution['primary_price']}")
    print(f"Effective price (with fees): ${best_execution['effective_price']}")
    
    # Example 3: Trade Execution and Reporting
    print("\n3. Trade Execution Example:")
    trade = {
        'trade_id': 'TRD456',
        'order_id': order['order_id'],
        'client_id': client_data['client_id'],
        'account_id': 'ACC123',
        'symbol': 'RY.TO',
        'security_type': 'equity',
        'side': 'buy',
        'quantity': 100,
        'price': 140.30,
        'commission': 9.95,
        'execution_venue': best_execution['primary_venue'],
        'execution_timestamp': datetime.now().isoformat(),
        'trade_date': datetime.now().isoformat(),
        'status': 'executed'
    }
    
    # Process through monitoring
    monitoring_result = compliance_monitor.process_trade(trade)
    print(f"Monitoring result: {monitoring_result['monitoring_result']}")
    
    # Report trade to CIRO
    trade_report = await ciro_compliance.report_trade(trade)
    print(f"Trade reporting: {trade_report['status']}")
    print(f"Settlement date: {trade_report['settlement_date']}")
    
    # Process for tax reporting
    tax_result = tax_reporting.process_trade_for_tax({
        **trade,
        'type': 'buy',
        'account_type': 'non-registered',  # Taxable account
        'province': 'ON'
    })
    print(f"ACB updated: ${tax_result['acb_per_share']:.4f} per share")
    
    # Log trade execution
    audit_trail.log_order_event(
        order=trade,
        event_type=AuditEventType.ORDER_EXECUTED,
        user_id='USER123',
        client_id=client_data['client_id'],
        account_id=trade['account_id']
    )
    
    # Example 4: Sale and Capital Gains
    print("\n4. Sale and Capital Gains Example:")
    sale_trade = {
        'trade_id': 'TRD789',
        'client_id': client_data['client_id'],
        'account_id': 'ACC123',
        'symbol': 'RY.TO',
        'type': 'sell',
        'quantity': 50,
        'price': 145.50,
        'commission': 9.95,
        'trade_date': (datetime.now() + timedelta(days=30)).isoformat(),
        'account_type': 'non-registered',
        'province': 'ON',
        'client_name': client_data['legal_name'],
        'client_sin': client_data['sin_or_bn'],
        'client_address': client_data['address']
    }
    
    sale_result = tax_reporting.process_trade_for_tax(sale_trade)
    print(f"Capital gain: ${sale_result['capital_gain']:.2f}")
    print(f"Taxable capital gain: ${sale_result['tax_implications']['taxable_capital_gain']:.2f}")
    print(f"Estimated tax: ${sale_result['tax_implications']['estimated_total_tax']:.2f}")
    
    # Example 5: Conflict of Interest Check
    print("\n5. Conflict of Interest Check:")
    client = ClientIdentification(**client_data)
    conflict_check = ciro_compliance.check_conflicts_of_interest(trade, client)
    print(f"Conflicts found: {conflict_check['has_conflicts']}")
    if conflict_check['conflicts']:
        for conflict in conflict_check['conflicts']:
            print(f"  - {conflict['type']}: {conflict['description']}")
    
    # Example 6: Real-time Monitoring Alerts
    print("\n6. Monitoring Alerts Example:")
    
    # Simulate large position
    large_trade = {
        'trade_id': 'TRD999',
        'account_id': 'ACC123',
        'symbol': 'TD.TO',
        'side': 'buy',
        'quantity': 100000,  # Large quantity
        'price': 85.00,
        'status': 'executed'
    }
    
    monitoring_result = compliance_monitor.process_trade(large_trade)
    if monitoring_result['critical_alerts'] > 0:
        print(f"CRITICAL ALERTS: {monitoring_result['critical_alerts']}")
        
        # Get active alerts
        active_alerts = compliance_monitor.get_active_alerts()
        for alert in active_alerts[:3]:  # Show first 3 alerts
            print(f"  Alert: {alert.alert_type.value} - {alert.message}")
            print(f"  Severity: {alert.severity.value}")
            print(f"  Actions required: {', '.join(alert.actions_required)}")
    
    # Example 7: Generate Compliance Report
    print("\n7. Compliance Report Generation:")
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    
    compliance_report = ciro_compliance.generate_compliance_report(start_date, end_date)
    print(f"Total trades: {compliance_report['summary']['total_trades']}")
    print(f"Best execution decisions: {compliance_report['summary']['best_execution_decisions']}")
    print(f"Large trader reports: {compliance_report['summary']['large_trader_reports']}")
    
    # Example 8: Generate Tax Package
    print("\n8. Year-End Tax Package:")
    tax_package = tax_reporting.generate_year_end_tax_package(
        client_id=client_data['sin_or_bn'],
        tax_year=datetime.now().year
    )
    
    print(f"Capital gains summary:")
    print(f"  Total dispositions: {tax_package['capital_gains_summary']['total_dispositions']}")
    print(f"  Net capital gain/loss: ${tax_package['capital_gains_summary']['net_capital_gain_loss']:.2f}")
    print(f"  Taxable capital gain: ${tax_package['capital_gains_summary']['taxable_capital_gain']:.2f}")
    
    print(f"\nForms required:")
    for form, required in tax_package['forms_required'].items():
        if required:
            print(f"  - {form}: {'Yes' if required else 'No'}")
    
    # Example 9: Audit Report
    print("\n9. Audit Trail Report:")
    audit_report = audit_trail.generate_audit_report(start_date, end_date)
    print(f"Total audit records: {audit_report['total_records']}")
    print(f"Integrity check: {'Passed' if audit_report['integrity_check']['integrity_failures'] == 0 else 'Failed'}")
    
    print("\nAudit summary by event type:")
    for event_type, count in audit_report['summary']['by_event_type'].items():
        print(f"  {event_type}: {count}")
    
    # Example 10: Provincial Variations
    print("\n10. Provincial Regulatory Variations:")
    
    # Quebec-specific requirements
    quebec_client = client_data.copy()
    quebec_client['address']['province'] = 'QC'
    
    print("Quebec-specific requirements:")
    provincial_rules = ciro_compliance.provincial_rules['QC']
    print(f"  Regulator: {provincial_rules['regulator']}")
    print(f"  Additional requirements: {', '.join(provincial_rules['additional_requirements'])}")
    print(f"  Language requirements: {provincial_rules.get('language_requirements', 'None')}")
    
    # Clean up
    print("\n" + "=" * 50)
    print("Shutting down compliance systems...")
    
    compliance_monitor.stop()
    audit_trail.stop()
    
    print("Canadian Trading Compliance Example Complete")


if __name__ == "__main__":
    asyncio.run(main())