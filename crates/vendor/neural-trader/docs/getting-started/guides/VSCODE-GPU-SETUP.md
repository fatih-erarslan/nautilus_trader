# VS Code GPU Environment on Fly.io

## Quick Deploy

```bash
./deploy-vscode-gpu.sh
```

## What You Get

🖥️ **Full VS Code in Browser** with GPU acceleration
- NVIDIA A100 40GB GPU
- 32GB RAM, 8 performance CPUs  
- 50GB persistent workspace storage

🐍 **Pre-installed Python Stack**
- PyTorch with CUDA 12.1
- NeuralForecast with GPU support
- TA-Lib for financial indicators
- FastAPI, Jupyter, Pandas, NumPy
- All trading platform dependencies

🔐 **Secure Access**
- Username: `trader`
- Password: `TradingDev2024!` 
- HTTPS connection

📁 **Persistent Workspace**
- `/home/trader/workspace` - your code lives here
- Survives container restarts
- Git, vim, nano, htop included

## Connection Info

After deployment, connect at: `https://ai-trader-vscode-gpu.fly.dev`

## Manual Commands

```bash
# Create app
fly apps create ai-trader-vscode-gpu

# Create volume  
fly volumes create vscode_data --app ai-trader-vscode-gpu --region ord --size 50 --yes

# Set secrets
fly secrets set VSCODE_PASSWORD="TradingDev2024!" VSCODE_USER="trader" --app ai-trader-vscode-gpu

# Deploy
fly deploy --config fly_deployment/fly-vscode.toml --app ai-trader-vscode-gpu

# Get status
fly status --app ai-trader-vscode-gpu
```

## GPU Testing

Once connected, test GPU access:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## Cost Optimization

- Auto-stops when not in use
- Single instance (no auto-scaling)
- Can be manually stopped: `fly machine stop --app ai-trader-vscode-gpu`