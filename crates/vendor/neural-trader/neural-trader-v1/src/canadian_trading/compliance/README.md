# Canadian Trading Compliance Module

This module implements comprehensive regulatory compliance for Canadian securities trading, ensuring full adherence to CIRO (Canadian Investment Regulatory Organization) regulations and CRA (Canada Revenue Agency) tax requirements.

## Components

### 1. CIRO Compliance (`ciro_compliance.py`)
Implements CIRO Universal Market Integrity Rules (UMIR) and regulatory requirements:

- **Best Execution Monitoring**: Ensures orders are executed at the best available price across all Canadian venues (TSX, TSXV, CSE, NEO, OMEGA)
- **Trade Reporting**: Automated trade reporting to CIRO with all required fields
- **Client Identification (KYC)**: Comprehensive client validation including SIN/BN verification
- **Record Keeping**: Maintains records for 6+ years as required by CIRO
- **Conflict of Interest Checks**: Detects and manages potential conflicts including proprietary trading, research coverage, and underwriting relationships
- **Provincial Variations**: Handles regulatory differences across provinces (ON, QC, BC, AB)

Key Classes:
- `CIROCompliance`: Main compliance implementation
- `ClientIdentification`: KYC data structure with validation
- `TradeReport`: CIRO-compliant trade reporting format

### 2. Tax Reporting (`tax_reporting.py`)
Comprehensive CRA tax compliance and reporting:

- **T5008 Slip Generation**: Automated generation of Statement of Securities Transactions
- **Capital Gains Tracking**: Accurate calculation of capital gains/losses
- **ACB (Adjusted Cost Base) Calculations**: Tracks ACB for each security including:
  - Purchase and sale transactions
  - Return of capital distributions
  - Stock dividends and splits
- **Foreign Income Reporting**: Tracks foreign income and calculates foreign tax credits
- **Multi-Currency Handling**: Automatic currency conversion using Bank of Canada rates
- **Provincial Tax Variations**: Handles different provincial tax rules (especially Quebec)

Key Classes:
- `TaxReporting`: Main tax reporting system
- `T5008Slip`: CRA-compliant tax slip generation
- `ACBTracker`: Sophisticated ACB tracking for accurate capital gains

### 3. Audit Trail (`audit_trail.py`)
Comprehensive audit logging system for regulatory compliance:

- **Immutable Records**: Cryptographic hashing ensures record integrity
- **7-Year Retention**: Automatic retention management per CIRO requirements
- **Event Categories**: 
  - Trading events (orders, executions, cancellations)
  - Compliance events (violations, checks)
  - System events (starts, stops, errors)
  - Client events (logins, KYC updates)
- **Compressed Storage**: Automatic compression for large records
- **Integrity Verification**: Built-in integrity checking and reporting

Key Classes:
- `AuditTrail`: Main audit system with async support
- `AuditRecord`: Immutable audit record structure
- `AuditDatabase`: SQLite-based persistent storage

### 4. Real-time Monitoring (`monitoring.py`)
Continuous compliance monitoring and pattern detection:

- **Position Monitoring**: 
  - Real-time position limit checks
  - Portfolio concentration monitoring
  - Warning thresholds at 80% of limits
- **Trading Pattern Detection**:
  - Wash trading detection
  - Layering and spoofing detection
  - Momentum ignition patterns
  - Marking the close manipulation
- **Alert Management**:
  - Severity-based alerts (Low, Medium, High, Critical)
  - Automated actions (halt trading, notify compliance)
  - Alert escalation for unacknowledged critical alerts
- **Configurable Rules**: Customizable monitoring rules with parameters

Key Classes:
- `ComplianceMonitor`: Main monitoring system
- `PositionMonitor`: Position and concentration limits
- `TradingPatternDetector`: Suspicious pattern detection
- `Alert`: Comprehensive alert structure

## Usage Example

```python
from canadian_trading.compliance import (
    CIROCompliance, 
    TaxReporting, 
    AuditTrail, 
    ComplianceMonitor
)

# Initialize compliance system
ciro = CIROCompliance("FIRM123", "REG456")
tax = TaxReporting()
audit = AuditTrail()
monitor = ComplianceMonitor(config)

# Start monitoring
audit.start()
monitor.start()

# Process a trade
trade = {
    'symbol': 'RY.TO',
    'quantity': 100,
    'price': 140.30,
    # ... other fields
}

# Ensure best execution
best_execution = await ciro.ensure_best_execution(order, market_data)

# Monitor compliance
monitoring_result = monitor.process_trade(trade)

# Report to CIRO
trade_report = await ciro.report_trade(trade)

# Process for taxes
tax_result = tax.process_trade_for_tax(trade)
```

## Provincial Regulatory Variations

The module handles provincial differences:

- **Ontario (OSC)**: Standard CIRO rules with accredited investor verification
- **Quebec (AMF)**: Bilingual documentation requirements, different dividend tax credits
- **British Columbia (BCSC)**: Venture exchange compliance
- **Alberta (ASC)**: Energy sector disclosure requirements

## Integration Points

1. **Trading System**: Integrates with order management for pre-trade compliance
2. **Market Data**: Uses real-time market data for best execution
3. **Risk Management**: Shares position data for risk calculations
4. **Reporting**: Generates regulatory reports and client tax documents

## Compliance Features

- ✅ CIRO UMIR compliance (Universal Market Integrity Rules)
- ✅ Best execution obligation (UMIR 5.1)
- ✅ Client identification and KYC
- ✅ 6+ year record retention
- ✅ Real-time position monitoring
- ✅ Manipulative trading pattern detection
- ✅ CRA tax reporting (T5008, T1135)
- ✅ Multi-currency support
- ✅ Provincial regulatory compliance
- ✅ Comprehensive audit trail
- ✅ Automated alert system
- ✅ Conflict of interest management

## Configuration

See `example_usage.py` for detailed configuration examples.

## Testing

The module includes comprehensive error handling and validation. All monetary calculations use `Decimal` for precision as required by financial regulations.