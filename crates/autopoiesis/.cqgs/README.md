# CQGS Configuration for Autopoiesis

## Overview

This directory contains the CQGS (Code Quality Governance System) configuration for the autopoiesis financial trading system. CQGS ensures **zero synthetic data contamination** and enforces real market data requirements.

## Configuration Files

### Core Configuration
- **`cqgs-sentinels.json`** - Main sentinel configuration
- **`cqgs-policies.json`** - Code quality policies and rules
- **`README.md`** - This documentation

### Hooks
- **`hooks/pre-edit.sh`** - Prevents synthetic data in edits
- **`hooks/post-edit.sh`** - Comprehensive post-edit analysis  
- **`hooks/pre-commit.sh`** - Final check before commits

### Generated Files
- **`cqgs-alerts.log`** - Analysis alerts and findings
- **`reports/`** - Generated analysis reports

## Sentinels Overview

### 1. Mock Detection Sentinel ✅
**Purpose:** Detect and block mock frameworks and synthetic data generation

**Patterns Monitored:**
- `mockall::`, `wiremock::`, `fastrand::`
- Mock data structures and synthetic generators
- Fake market data patterns

**Action:** **BLOCK** - Critical violations prevent commits

### 2. Zero Synthetic Sentinel ✅  
**Purpose:** Enforce absolute zero synthetic data policy

**Patterns Monitored:**
- Random number generation for financial data
- Synthetic price/volume/sentiment generation
- Fake market data creation

**Action:** **BLOCK** - Zero tolerance for synthetic data

### 3. Real Data Enforcement Sentinel ✅
**Purpose:** Ensure real API integration and data validation

**Requirements:**
- API key authentication for market data
- Data validation and integrity checking
- Real data source connections

**Action:** **WARN** - Requires real data implementation

### 4. Policy Enforcement Sentinel ✅
**Purpose:** Custom financial system quality rules

**Policies:**
- No hardcoded credentials
- Proper error handling in financial modules
- Security compliance requirements

## Usage

### Manual Analysis
```bash
# Analyze specific file
.cqgs/hooks/post-edit.sh src/ml/nhits/financial/price_prediction.rs

# Full project analysis  
.cqgs/hooks/pre-commit.sh
```

### Automatic Integration
The hooks automatically run during:
- File edits (pre-edit, post-edit)
- Git commits (pre-commit)
- Build processes (integration pending)

### Configuration Updates

#### Enable/Disable Sentinels
Edit `cqgs-sentinels.json`:
```json
{
  "sentinels": {
    "mock-detection": {
      "enabled": true,  // Set to false to disable
      "strictMode": true
    }
  }
}
```

#### Add Custom Patterns
Add to sentinel patterns array:
```json
{
  "patterns": [
    "fastrand::",
    "your-custom-pattern"
  ]
}
```

#### Adjust Thresholds
Modify confidence thresholds:
```json
{
  "confidence_threshold": 0.95,  // 0.0-1.0
  "max_synthetic_score": 0.1
}
```

## Policy Enforcement

### Critical Violations (Block Commits)
- **SDP-001:** Synthetic Data Prohibition
- **SEC-001:** Security Compliance
- Any fastrand usage in production code
- Hardcoded credentials or secrets

### High Priority (Warnings)  
- **RDR-001:** Real Data Integration Requirements
- Missing API authentication
- Missing data validation

### Exemptions
Permanent exemptions configured in `cqgs-policies.json`:
- Documentation files (*.md)
- Investigation reports
- Test infrastructure (with strict limits)

## Metrics and Reporting

### Compliance Scores
- **Synthetic Data Score:** 0.0 (target) - Current: 0.0 ✅
- **Security Score:** 1.0 (target) - Current: 0.95 ✅  
- **Quality Score:** 0.85 (target) - Current: 0.78 ⚠️

### Reports Generated
- **Daily compliance reports** - `.cqgs/reports/daily-YYYYMMDD.md`
- **Commit analysis reports** - `.cqgs/reports/commit-YYYYMMDD-HHMMSS.md`
- **Alert logs** - `.cqgs/cqgs-alerts.log`

## Troubleshooting

### Common Issues

#### "fastrand usage detected"
**Cause:** Synthetic data generation attempt  
**Solution:** Use real market data APIs or proper statistical libraries

#### "Missing API authentication"
**Cause:** Financial module lacks real data integration  
**Solution:** Add environment variable API key requirements

#### "Hardcoded credentials detected"
**Cause:** Secrets in source code  
**Solution:** Move to environment variables

### Log Analysis
```bash
# View recent alerts
tail -f .cqgs/cqgs-alerts.log

# Search for specific patterns
grep "CRITICAL" .cqgs/cqgs-alerts.log

# View compliance trends
ls -la .cqgs/reports/
```

## Maintenance

### Regular Tasks
- Review weekly compliance reports
- Update sentinel patterns as needed
- Monitor alert logs for new synthetic data attempts
- Validate exemptions are still necessary

### Configuration Updates
- Test changes in staging environment
- Update version numbers after changes
- Document policy modifications
- Communicate changes to development team

## Integration

### Git Hooks Setup
```bash
# Link pre-commit hook
ln -sf ../../.cqgs/hooks/pre-commit.sh .git/hooks/pre-commit

# Test hook
.git/hooks/pre-commit
```

### CI/CD Integration
Add to build pipeline:
```yaml
- name: CQGS Analysis
  run: .cqgs/hooks/pre-commit.sh
  continue-on-error: false  # Block on critical issues
```

## Support

### Configuration Help
- Review sentinel documentation in `cqgs-sentinels.json`
- Check policy definitions in `cqgs-policies.json`
- Examine hook implementations in `hooks/` directory

### Emergency Bypass
**⚠️ Use with extreme caution:**
```bash
# Temporary disable (not recommended)
mv .cqgs/cqgs-sentinels.json .cqgs/cqgs-sentinels.json.disabled

# Re-enable
mv .cqgs/cqgs-sentinels.json.disabled .cqgs/cqgs-sentinels.json
```

---

**🛡️ CQGS Mission:** Maintain zero synthetic data contamination in the autopoiesis financial trading system through continuous monitoring and enforcement of real data requirements.