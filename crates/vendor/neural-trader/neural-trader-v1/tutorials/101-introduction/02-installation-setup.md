# Part 2: Installation & Setup
**Duration**: 10 minutes | **Difficulty**: Beginner

## 🚀 Step 1: Launch GitHub Codespace (2 min)

The easiest way to get started is using GitHub Codespaces - a cloud development environment with everything pre-configured.

### Create Your Codespace

1. **Open the Neural Trader Repository**
   - Go to: [https://github.com/ruvnet/neural-trader](https://github.com/ruvnet/neural-trader)

2. **Click the Green "Code" Button**
   - Select the "Codespaces" tab
   - Click "Create codespace on main"
   
   ![Codespace Creation](https://docs.github.com/assets/cb-13837/images/help/codespaces/new-codespace-button.png)

3. **Wait for Setup** (1-2 minutes)
   - GitHub will create your cloud environment
   - VS Code will open in your browser
   - All dependencies are pre-installed

### Why Codespaces?
- ✅ No local setup required
- ✅ Works on any device with a browser
- ✅ Pre-configured environment
- ✅ Free tier includes 60 hours/month
- ✅ Your work is saved automatically

## 🤖 Step 2: Install Claude Code (3 min)

Once your Codespace is running, open the terminal (Terminal → New Terminal in VS Code) and run:

```bash
# Install Claude Code CLI globally
npm install -g @anthropic-ai/claude-code

# Verify installation
claude --version
```

### Run Claude in Safe Mode

For Codespaces, we'll use a special flag to bypass permission prompts for faster setup:

```bash
# (Optional) Skip permissions check for faster setup
# Only use if you understand the security implications
claude --dangerously-skip-permissions
```

**Note**: This flag is safe in Codespaces since it's an isolated environment. It allows Claude to work without constantly asking for file access permissions.

## 🔧 Step 3: Initialize Claude Flow & Flow Nexus (3 min)

Claude Flow provides the AI orchestration layer with enhanced MCP setup:

```bash
# Initialize Claude Flow with enhanced MCP setup (auto-configures permissions!)
npx claude-flow@alpha init --force

# Explore all revolutionary capabilities
npx claude-flow@alpha --help
```

The `init --force` command:
- Auto-configures MCP permissions
- Sets up swarm orchestration
- Configures agent coordination
- Prepares neural network integration
- Creates necessary configuration files
- Optimizes for your environment

### 🚀 Quick Start with Flow Nexus

For access to neural networks, cloud orchestration, and advanced features, register with Flow Nexus:

```bash
# Option 1: Command line registration
npx flow-nexus@latest auth register -e pilot@ruv.io -p yourpassword

# Option 2: Initialize Flow Nexus only (minimal setup)
npx claude-flow init --flow-nexus
```

**Or register via Claude Code MCP tools** (recommended):

```bash
# Ask Claude to help you register
claude "Register me for Flow Nexus using the MCP tools:
- Email: your@email.com
- Create a secure password
- Login and verify access
- Show me my account status"
```

Claude will use the MCP tools:
- `mcp__flow-nexus__user_register()` - Create your account
- `mcp__flow-nexus__user_login()` - Sign you in
- `mcp__flow-nexus__user_profile()` - Show account details
- `mcp__flow-nexus__check_balance()` - Check your credits

### Flow Nexus Benefits

With Flow Nexus registration you get:
- ✅ 27+ neural trading models
- ✅ Cloud sandbox environments
- ✅ Advanced swarm orchestration  
- ✅ Real-time market streaming
- ✅ GPU-accelerated backtesting
- ✅ Cross-session memory
- ✅ Template marketplace access

## 🎯 Step 4: Let Claude Handle the Rest! (3 min)

Now for the magic - Claude will complete the setup for you:

```bash
# In your terminal, tell Claude to finish the installation
claude "Complete the Neural Trader setup:
1. Create .env file with placeholder API keys
2. Install Python dependencies
3. Add all MCP servers
4. Configure trading environment
5. Verify everything is working"
```

Claude will:
- ✅ Create necessary configuration files
- ✅ Install all required packages
- ✅ Set up MCP servers for trading
- ✅ Configure the environment
- ✅ Run verification tests

### What Claude Is Doing

While Claude works, it will:
1. **Environment Setup**: Create `.env` with safe defaults
2. **Dependencies**: Install Python and Node packages
3. **MCP Servers**: Add ai-news-trader, claude-flow, flow-nexus
4. **Verification**: Test all components are working

## 🔑 Step 5: Add Your API Keys (Optional)

For full functionality, you'll want to add real API keys. Claude can help:

```bash
# Ask Claude to guide you through API setup
claude "Help me get and configure API keys for:
- Free market data (Yahoo Finance)
- Paper trading (Alpaca)
- News sentiment (NewsAPI free tier)
Show me exactly where to sign up and what to add"
```

### Free Tier Options

Start with these free services:
- **Market Data**: Yahoo Finance (no key needed)
- **Paper Trading**: Alpaca (free simulator)
- **News**: NewsAPI (100 requests/day free)

## ✅ Step 6: Verify Everything Works (1 min)

Let's make sure everything is set up correctly:

```bash
# Ask Claude to run a complete system check
claude "Run a complete system check:
- Test market data connection
- Verify trading strategies load
- Check neural models are accessible
- Confirm MCP servers are running
Show me a status report"
```

You should see something like:
```
System Status: ✅ READY
- Market Data: Connected
- Trading Engine: Operational
- Neural Models: Available
- MCP Servers: All running
- Paper Trading: Ready

You can now start trading!
```

## 🚀 Your First Trade

Let's do something exciting right away:

```bash
# Start with a simple paper trade
claude "Show me Apple stock price and tell me if it's a good time to buy"

# Set up monitoring
claude "Monitor AAPL and alert me if it moves more than 2%"

# Try a backtest
claude "Backtest a simple buy-and-hold strategy on AAPL for the last month"
```

## 🛠 Troubleshooting

### If Claude Doesn't Respond

```bash
# Make sure you're using the skip permissions flag in Codespaces
claude --dangerously-skip-permissions

# Or restart Claude
pkill -f claude
claude --dangerously-skip-permissions
```

### If Installation Fails

```bash
# Ask Claude to diagnose
claude "Diagnose and fix any installation issues"

# Or manually check
npm list -g @anthropic/claude-code
which claude
```

### Codespace Tips

- **Auto-save**: Your work saves automatically
- **Stop Codespace**: When done, stop it to save hours
- **Resume Later**: Your Codespace preserves everything
- **Share**: You can share your Codespace URL with others

## 💡 Pro Tips for Codespaces

### Save Your Free Hours
GitHub gives you 60 free hours/month. To maximize them:

1. **Stop when not using**: 
   ```bash
   # In VS Code: F1 → "Codespaces: Stop Current Codespace"
   ```

2. **Set auto-stop**: Codespaces auto-stop after 30 min of inactivity

3. **Check usage**: Visit github.com/settings/billing

### Performance Optimization
- **Use 4-core machine**: For better performance (still free tier)
- **Prebuilds**: The repository has prebuilds for faster startup
- **Browser**: Chrome/Edge work best with Codespaces

## ✅ Setup Complete Checklist

- [ ] Codespace is running
- [ ] Claude Code installed
- [ ] Claude Flow initialized
- [ ] Claude completed auto-setup
- [ ] System check passed
- [ ] First command worked

## 🎉 Congratulations!

You now have a fully functional Neural Trader environment running in the cloud! No local setup required, no configuration headaches - just pure trading power at your fingertips.

## ⏭ Next Steps

Continue to [Claude Flow Basics](03-claude-flow-basics.md) to learn about intelligent orchestration.

---

**Progress**: 15 min / 2 hours | [← Previous: What is Neural Trader](01-what-is-neural-trader.md) | [Back to Contents](README.md) | [Next: Claude Flow →](03-claude-flow-basics.md)