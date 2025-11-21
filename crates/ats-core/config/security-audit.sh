#!/bin/bash
# ATS-Core Security Audit Script
# Scans for security vulnerabilities and configuration issues

set -euo pipefail

echo "🔒 ATS-Core Security Audit Starting..."
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ISSUES_FOUND=0
WARNINGS_FOUND=0

# Function to report critical issues
report_critical() {
    echo -e "${RED}[CRITICAL] $1${NC}"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
}

# Function to report warnings  
report_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
    WARNINGS_FOUND=$((WARNINGS_FOUND + 1))
}

# Function to report success
report_ok() {
    echo -e "${GREEN}[OK] $1${NC}"
}

echo
echo "🔍 Scanning for hardcoded secrets..."
echo "-----------------------------------"

# Check for hardcoded secrets in source code
SECRET_PATTERNS=(
    "password.*=.*[\"'][^\"']+[\"']"
    "secret.*=.*[\"'][^\"']+[\"']"
    "key.*=.*[\"'][^\"']{8,}[\"']"
    "token.*=.*[\"'][^\"']+[\"']"
    "your-secret-key-change-this"
    "development-secret"
    "test-secret"
    "admin.*password"
    "root.*password"
    "default.*password"
)

for pattern in "${SECRET_PATTERNS[@]}"; do
    if grep -r -i "$pattern" src/ --include="*.rs" 2>/dev/null | grep -v "template" | grep -v "example" | grep -v "comment"; then
        report_critical "Hardcoded secret pattern found: $pattern"
    fi
done

# Check for specific vulnerable configurations
if grep -r "your-secret-key-change-this" src/ --include="*.rs" 2>/dev/null; then
    report_critical "Default JWT secret found - MUST be changed for production"
fi

if grep -r "secret.*=.*\".*\"" src/ --include="*.rs" 2>/dev/null | grep -v env::var | grep -v template; then
    report_critical "Potential hardcoded secret in source code"
fi

echo
echo "🔧 Checking configuration security..."
echo "------------------------------------"

# Check if environment variables are properly configured
check_env_var() {
    local var_name=$1
    local required=$2
    
    if [ -n "${!var_name:-}" ]; then
        local value="${!var_name}"
        local value_length=${#value}
        
        if [ "$var_name" = "ATS_JWT_SECRET" ]; then
            if [ $value_length -lt 32 ]; then
                report_critical "ATS_JWT_SECRET is too short ($value_length chars, minimum 32)"
            elif [ $value_length -lt 64 ]; then
                report_warning "ATS_JWT_SECRET should be at least 64 characters for production"
            else
                report_ok "ATS_JWT_SECRET length is adequate"
            fi
            
            # Check for weak secrets
            case "$value" in
                *"password"*|*"secret"*|*"123456"*|*"admin"*)
                    report_critical "ATS_JWT_SECRET contains weak patterns"
                    ;;
                *)
                    report_ok "ATS_JWT_SECRET does not contain obvious weak patterns"
                    ;;
            esac
        fi
        
        if [ "$var_name" = "ATS_ENCRYPTION_KEY" ]; then
            if [ $value_length -lt 32 ]; then
                report_critical "ATS_ENCRYPTION_KEY is too short (need 256-bit key)"
            else
                report_ok "ATS_ENCRYPTION_KEY length is adequate"
            fi
        fi
        
    elif [ "$required" = "true" ]; then
        if [ "${ATS_ENVIRONMENT:-development}" = "production" ]; then
            report_critical "Required environment variable $var_name is not set"
        else
            report_warning "Environment variable $var_name is not set"
        fi
    else
        report_warning "Optional environment variable $var_name is not set"
    fi
}

# Check critical environment variables
check_env_var "ATS_JWT_SECRET" "true"
check_env_var "ATS_ENCRYPTION_KEY" "false"
check_env_var "ATS_DB_PASSWORD" "false"
check_env_var "ATS_ENVIRONMENT" "false"

# Check environment-specific configurations
if [ "${ATS_ENVIRONMENT:-development}" = "production" ]; then
    echo
    echo "🏭 Production environment checks..."
    echo "--------------------------------"
    
    if [ "${ATS_ENCRYPT_AT_REST:-false}" != "true" ]; then
        report_critical "Encryption at rest is not enabled in production"
    else
        report_ok "Encryption at rest is enabled"
    fi
    
    if [ "${ATS_ENCRYPT_IN_TRANSIT:-false}" != "true" ]; then
        report_critical "Encryption in transit is not enabled in production"
    else
        report_ok "Encryption in transit is enabled"
    fi
    
    if [ "${ATS_JWT_ALGORITHM:-HS256}" = "HS256" ]; then
        report_warning "Consider using RS256 instead of HS256 for production JWT"
    else
        report_ok "Using recommended JWT algorithm: ${ATS_JWT_ALGORITHM:-HS256}"
    fi
    
    if [ "${ATS_JWT_EXPIRY_SECONDS:-86400}" -gt 3600 ]; then
        report_warning "JWT expiry time is longer than recommended 1 hour for production"
    else
        report_ok "JWT expiry time is reasonable for production"
    fi
fi

echo
echo "📁 Checking file permissions..."
echo "------------------------------"

# Check for sensitive files with wrong permissions
if [ -f ".env" ]; then
    PERM=$(stat -f "%p" .env 2>/dev/null || stat -c "%a" .env 2>/dev/null || echo "000")
    if [ "${PERM: -3}" != "600" ]; then
        report_warning ".env file should have 600 permissions (currently ${PERM: -3})"
    else
        report_ok ".env file has correct permissions"
    fi
fi

# Check for config files
for file in config/.env.production config/.env.staging; do
    if [ -f "$file" ]; then
        PERM=$(stat -f "%p" "$file" 2>/dev/null || stat -c "%a" "$file" 2>/dev/null || echo "000")
        if [ "${PERM: -3}" != "600" ]; then
            report_warning "$file should have 600 permissions (currently ${PERM: -3})"
        else
            report_ok "$file has correct permissions"
        fi
    fi
done

echo
echo "🗂️ Checking for sensitive files..."
echo "---------------------------------"

# Check for files that shouldn't be committed
SENSITIVE_FILES=(
    ".env"
    ".env.production"
    ".env.local"
    "config/.env.production"
    "config/.env.local"
    "*.key"
    "*.pem"
    "*.p12"
    "*.pfx"
    "*password*"
    "*secret*"
)

for pattern in "${SENSITIVE_FILES[@]}"; do
    if find . -name "$pattern" -type f 2>/dev/null | grep -v template | grep -v example | head -5; then
        report_warning "Found potentially sensitive files matching: $pattern"
    fi
done

echo
echo "🔐 Checking dependencies for known vulnerabilities..."
echo "---------------------------------------------------"

# Run cargo audit if available
if command -v cargo-audit &> /dev/null; then
    if cargo audit --quiet 2>/dev/null; then
        report_ok "No known vulnerabilities found in dependencies"
    else
        report_warning "Potential vulnerabilities found in dependencies - run 'cargo audit' for details"
    fi
else
    report_warning "cargo-audit not installed - install with 'cargo install cargo-audit'"
fi

echo
echo "🧪 Running security tests..."
echo "---------------------------"

# Run security-related tests
if cargo test security 2>/dev/null; then
    report_ok "Security tests passed"
else
    report_warning "Some security tests failed or no security tests found"
fi

echo
echo "📊 Security Audit Summary"
echo "========================="

if [ $ISSUES_FOUND -eq 0 ] && [ $WARNINGS_FOUND -eq 0 ]; then
    echo -e "${GREEN}✅ Security audit completed successfully!${NC}"
    echo -e "${GREEN}   No critical issues or warnings found.${NC}"
    exit 0
elif [ $ISSUES_FOUND -eq 0 ]; then
    echo -e "${YELLOW}⚠️  Security audit completed with warnings.${NC}"
    echo -e "${YELLOW}   Found $WARNINGS_FOUND warnings.${NC}"
    echo -e "${YELLOW}   Please review and address warnings before production deployment.${NC}"
    exit 1
else
    echo -e "${RED}❌ Security audit failed!${NC}"
    echo -e "${RED}   Found $ISSUES_FOUND critical issues and $WARNINGS_FOUND warnings.${NC}"
    echo -e "${RED}   CRITICAL ISSUES MUST BE FIXED BEFORE PRODUCTION DEPLOYMENT.${NC}"
    exit 2
fi