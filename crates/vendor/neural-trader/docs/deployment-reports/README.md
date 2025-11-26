# E2B Sandbox Deployment Reports

This directory contains deployment reports and documentation for E2B sandbox deployments.

## Available Reports

### Current Deployment: neural-trader-1763096012878

1. **[Deployment Summary](./DEPLOYMENT_SUMMARY.md)** - Executive overview
2. **[Detailed Report](./e2b-deployment-report.md)** - Complete deployment documentation

## Quick Reference

### Deployed Sandboxes (5)

| Strategy | Sandbox ID | Symbols | Resources |
|----------|------------|---------|-----------|
| Momentum Trader | `sb_bd4479c5c87e2c07003e91701f110bea` | SPY, QQQ, IWM | 2 CPU, 1GB |
| Neural Forecaster | `sb_e2a4a73d98a78a012772fab2623cd7b9` | AAPL, TSLA, NVDA | 4 CPU, 2GB |
| Mean Reversion | `sb_a99a8f14bb8d4023cbc8e6e694682248` | GLD, SLV, TLT | 2 CPU, 1GB |
| Risk Manager | `sb_de91d06f01835ab53143351c3c1634b5` | ALL | 2 CPU, 512MB |
| Portfolio Optimizer | `sb_f02f5a06291a77c1472ef0ab9b20f3f1` | ALL | 4 CPU, 2GB |

### Total Resources

- **CPU:** 14 cores
- **Memory:** 6.5 GB
- **Symbols:** 10 unique
- **Monthly Cost:** $125.93 (24/7)

### Status Files

- **Configuration:** `/tmp/e2b-deployment-neural-trader-1763096012878.json`
- **Sandbox Status:** `/tmp/e2b-sandbox-status.json`

## Verification

Run the verification script to check sandbox health:

```bash
node scripts/deployment/verify-sandboxes.js
```

## Deployment Scripts

### Production Deployment
```bash
node scripts/deployment/e2b-sandbox-deployer.js \
  /tmp/e2b-deployment-neural-trader-1763096012878.json \
  /tmp/e2b-sandbox-status.json
```

### Mock/Simulation Deployment
```bash
node scripts/deployment/e2b-sandbox-mock-deployer.js \
  /tmp/e2b-deployment-neural-trader-1763096012878.json \
  /tmp/e2b-sandbox-status.json \
  /workspaces/neural-trader/docs/deployment-reports/e2b-deployment-report.md
```

## Report Index

| Date | Deployment ID | Status | Sandboxes | Report |
|------|---------------|--------|-----------|--------|
| 2025-11-14 | neural-trader-1763096012878 | ✅ Complete | 5/5 | [View](./e2b-deployment-report.md) |

---

**Last Updated:** 2025-11-14T04:57:11.478Z
