# Part 4: Flow Nexus Setup
**Duration**: 8 minutes | **Difficulty**: Beginner

## 🌐 What is Flow Nexus?

Flow Nexus is a cloud-based platform that extends Neural Trader with:
- **E2B Sandboxes**: Isolated execution environments
- **Neural Network Templates**: Pre-trained models marketplace
- **Workflow Automation**: Visual workflow builder
- **Credit System**: Pay-per-use GPU resources

## 📝 Step 1: Create Flow Nexus Account (2 min)

### Option A: Web Signup
1. Visit [flow-nexus.com](https://flow-nexus.com)
2. Click "Sign Up"
3. Use code `NEURAL101` for 1000 free credits
4. Verify email

### Option B: CLI Signup
```bash
# Initialize Flow Nexus
npx flow-nexus@latest auth init --mode user

# Register new account
npx flow-nexus@latest user register \
  --email your-email@example.com \
  --password your-secure-password

# Verify email (check inbox)
npx flow-nexus@latest user verify-email --token YOUR_TOKEN
```

## 🔐 Step 2: Configure Authentication (1 min)

```bash
# Login via CLI
npx flow-nexus@latest user login \
  --email your-email@example.com \
  --password your-password

# Check authentication status
npx flow-nexus@latest auth status

# Your API key will be displayed
# Save it to your .env file:
echo "FLOW_NEXUS_API_KEY=your_api_key_here" >> .env
```

## 🔌 Step 3: Add Flow Nexus MCP Server (2 min)

```bash
# Add to Claude Code
claude mcp add flow-nexus "npx -y flow-nexus@latest mcp start"

# Verify connection
claude mcp list

# Should show:
# flow-nexus: npx -y flow-nexus@latest mcp start - ✓ Connected
```

## 💳 Step 4: Credit System Overview (1 min)

### Credit Pricing
| Action | Credits | USD Equivalent |
|--------|---------|----------------|
| Sandbox Hour | 10 | $0.10 |
| Neural Training | 100 | $1.00 |
| GPU Inference | 1 | $0.01 |
| Workflow Run | 5 | $0.05 |

### Free Tier Includes
- 1000 credits/month
- 5 sandboxes
- 10 neural templates
- Basic support

### Check Balance
```bash
# Via CLI
npx flow-nexus@latest check-balance

# Via Claude Code
claude "Check my Flow Nexus balance"
```

## 🚀 Step 5: Test Flow Nexus Features (2 min)

### 1. Create Your First Sandbox
```bash
# Create a Node.js sandbox
claude "Create a Flow Nexus sandbox for testing"

# Or directly:
npx flow-nexus@latest sandbox create \
  --template node \
  --name my-first-sandbox
```

### 2. List Available Templates
```bash
# View neural network templates
claude "Show Flow Nexus neural templates"

# Or directly:
npx flow-nexus@latest neural list-templates
```

### 3. Test Workflow Creation
```bash
# Create a simple workflow
claude "Create a Flow Nexus workflow for daily market analysis"
```

## 🎯 Key Features

### 1. E2B Sandboxes
Isolated environments for safe execution:
```javascript
// Sandbox configuration
{
  template: "node",        // or python, react, etc.
  memory_mb: 512,
  cpu_count: 1,
  timeout: 3600,          // 1 hour
  env_vars: {
    API_KEY: "your_key"
  }
}
```

### 2. Neural Templates Marketplace
```bash
# Browse templates
npx flow-nexus@latest neural list-templates \
  --category timeseries

# Deploy a template
npx flow-nexus@latest neural deploy-template \
  --template-id lstm-predictor \
  --custom-config '{"horizon": 5}'
```

### 3. Workflow Automation
```javascript
// Example workflow
{
  name: "Morning Trading Setup",
  steps: [
    { type: "data_fetch", symbols: ["SPY", "QQQ"] },
    { type: "sentiment_analysis", sources: ["news", "twitter"] },
    { type: "signal_generation", strategy: "momentum" },
    { type: "risk_check", max_position: 0.1 },
    { type: "order_placement", mode: "paper" }
  ],
  triggers: [
    { type: "schedule", cron: "0 9 * * 1-5" }  // 9 AM weekdays
  ]
}
```

## 🔧 Advanced Configuration

### Environment Variables
```bash
# Add to .env
FLOW_NEXUS_API_KEY=your_api_key
FLOW_NEXUS_DEFAULT_TEMPLATE=node
FLOW_NEXUS_AUTO_REFILL=true
FLOW_NEXUS_REFILL_THRESHOLD=100
FLOW_NEXUS_REFILL_AMOUNT=1000
```

### Auto-refill Setup
```bash
# Configure automatic credit refill
npx flow-nexus@latest configure-auto-refill \
  --enabled true \
  --threshold 100 \
  --amount 1000
```

### Payment Methods
```bash
# Add payment method
npx flow-nexus@latest create-payment-link --amount 10

# Returns a secure Stripe link
# Minimum: $10 (1000 credits)
```

## 🎨 Flow Nexus Dashboard

Access web dashboard for:
- Visual workflow builder
- Sandbox management
- Credit usage analytics
- Neural model performance
- API documentation

Visit: [dashboard.flow-nexus.com](https://dashboard.flow-nexus.com)

## 🧪 Quick Exercises

### Exercise 1: Sandbox Test
```bash
# Create and execute code in sandbox
claude "Create Flow Nexus sandbox and run console.log('Hello Trading!')"
```

### Exercise 2: Neural Template
```bash
# Deploy a pre-trained model
claude "Deploy LSTM price predictor template from Flow Nexus"
```

### Exercise 3: Workflow Creation
```bash
# Build an automated workflow
claude "Create workflow to analyze top movers every hour"
```

## 🛠 Troubleshooting

### Common Issues

1. **Authentication Failed**
```bash
# Reset authentication
npx flow-nexus@latest user logout
npx flow-nexus@latest user login --email your@email.com
```

2. **Insufficient Credits**
```bash
# Check balance
npx flow-nexus@latest check-balance

# Purchase credits
npx flow-nexus@latest create-payment-link --amount 10
```

3. **Sandbox Timeout**
```bash
# List sandboxes
npx flow-nexus@latest sandbox list

# Terminate stuck sandbox
npx flow-nexus@latest sandbox delete --sandbox-id ID
```

## ✅ Setup Checklist

- [ ] Flow Nexus account created
- [ ] Email verified
- [ ] API key saved to .env
- [ ] MCP server connected
- [ ] Credits available (1000 free)
- [ ] First sandbox created
- [ ] Templates accessible

## 📊 Resource Limits

### Free Tier
- Sandboxes: 5 concurrent
- API calls: 10,000/month
- Storage: 1GB
- Neural models: 3 active

### Pro Tier ($29/month)
- Sandboxes: 20 concurrent
- API calls: 100,000/month
- Storage: 10GB
- Neural models: Unlimited

## ⏭ Next Steps

Continue to [Claude Code as Trading UI](05-claude-code-ui.md) to learn how to manage complex strategies through simple commands.

---

**Progress**: 30 min / 2 hours | [← Previous: Claude Flow](03-claude-flow-basics.md) | [Back to Contents](README.md) | [Next: Claude Code UI →](05-claude-code-ui.md)