# Neuro-Divergent Quick Start Guide

## Build All Platforms

\`\`\`bash
cd neural-trader-rust/crates/neuro-divergent
npm install
npm run build:all
\`\`\`

## Test Local Build

\`\`\`bash
npm run build -- --target x86_64-unknown-linux-gnu --release --strip
npm run test:node
\`\`\`

## Trigger CI/CD Build

\`\`\`bash
git tag neuro-divergent-v0.1.0
git push origin neuro-divergent-v0.1.0
\`\`\`

## Verify Artifacts

\`\`\`bash
ls -lh artifacts/*/native/*.node
\`\`\`

## Documentation

- **BUILD_GUIDE.md** - Comprehensive build instructions
- **BINARY_VERIFICATION_REPORT.md** - Verification procedures
- **PLATFORM_BUILD_COMPLETION.md** - Complete status report

## Support

GitHub Issue: #76 - Neuro-Divergent Integration
Workflow: \`.github/workflows/build-neuro-divergent.yml\`
