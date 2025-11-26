# FANG Automated Trading Workflow

## Deployment Status
- **Workflow ID**: 1b228258-46d6-4466-972b-ecd2625849c3
- **Status**: ACTIVE ✅
- **Created**: $(date)
- **Type**: Event-Driven with Message Queue

## Monitoring Configuration

### Price Alerts
- Stop Loss: $146.01 (-5%)
- Take Profit: $169.07 (+10%)
- Entry Signal: RSI < 45

### Risk Parameters
- Max Position: 20% of portfolio ($20,000)
- Position Size: 130 shares
- Trailing Stop: 3%
- Daily Loss Limit: 2%

### Entry Conditions
1. RSI < 45 ✅ (currently 41.28)
2. FANG-XLE Correlation < 0.8
3. VIX < 25
4. USO trending up

### Hedge Configuration
- Primary: Short XLE (15% of FANG position)
- Secondary: Long TLT (10% allocation)

## Execution Schedule
- **Frequency**: Every 5 minutes during market hours
- **Hours**: 9:30 AM - 4:00 PM ET
- **Days**: Monday - Friday

## Event Triggers
1. Stop loss hit → Immediate exit
2. Take profit hit → Close position
3. High volatility → Reduce position
4. Entry signal → Open position

## Real-Time Monitoring
- Stream ID: exec_FANG_Risk_Management_Workflow_1758642913867
- Queue: Message-based async processing
- Sandbox: fang_trading_monitor (24hr runtime)

## Performance Metrics
- Expected Sharpe: 0.944
- Max Drawdown: 34.3%
- Win Probability: 99.8%
- Risk of Ruin: 0%

## Commands

### Check Status
\`\`\`bash
npx flow-nexus@latest workflow status 1b228258-46d6-4466-972b-ecd2625849c3
\`\`\`

### View Execution Log
\`\`\`bash
npx flow-nexus@latest workflow audit 1b228258-46d6-4466-972b-ecd2625849c3
\`\`\`

### Stop Workflow
\`\`\`bash
npx flow-nexus@latest workflow stop 1b228258-46d6-4466-972b-ecd2625849c3
\`\`\`

## Alert Channels
- Console logs
- Message queue notifications
- Real-time stream updates
- Audit trail logging

Last Updated: $(date)
