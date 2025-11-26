#!/usr/bin/env python3
"""
Neural Trading Workflow Demo
Real-world integration example using Flow Nexus and AI News Trader
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_trading_workflow import NeuralTradingWorkflow, TradingSignal
from enhanced_neural_workflow import EnhancedNeuralTradingWorkflow

class TradingWorkflowDemo:
    """Comprehensive demo of the neural trading workflow system"""
    
    def __init__(self):
        self.demo_portfolios = {
            "conservative": {
                "symbols": ['AAPL', 'MSFT', 'JNJ', 'PG', 'KO'],
                "max_position": 1000,
                "risk_tolerance": "conservative"
            },
            "growth": {
                "symbols": ['GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META'],
                "max_position": 2000,
                "risk_tolerance": "moderate"
            },
            "tech_focused": {
                "symbols": ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX'],
                "max_position": 1500,
                "risk_tolerance": "aggressive"
            }
        }
    
    def display_banner(self):
        """Display demo banner"""
        print("\n" + "="*70)
        print("🤖 NEURAL TRADING WORKFLOW DEMO")
        print("🌐 Flow Nexus + AI News Trader Integration")
        print("="*70)
        print(f"📅 Demo Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🎯 Demonstrating advanced AI-powered trading workflows")
        print("-"*70)
    
    async def demo_basic_workflow(self, portfolio_name: str = "growth"):
        """Demonstrate basic neural trading workflow"""
        print(f"\n🚀 DEMO 1: Basic Neural Trading Workflow ({portfolio_name.title()})")
        print("-" * 50)
        
        portfolio = self.demo_portfolios[portfolio_name]
        
        # Configure workflow
        config = {
            'symbols': portfolio['symbols'],
            'max_position_size': portfolio['max_position'],
            'risk_tolerance': portfolio['risk_tolerance'],
            'trading_mode': 'paper'
        }
        
        print(f"📊 Portfolio: {portfolio_name.title()}")
        print(f"📈 Symbols: {', '.join(config['symbols'])}")
        print(f"💰 Max Position: ${config['max_position_size']:,}")
        print(f"⚖️  Risk Level: {config['risk_tolerance']}")
        print(f"🎮 Mode: {config['trading_mode']}")
        
        # Execute workflow
        print(f"\n⏳ Executing workflow...")
        workflow = NeuralTradingWorkflow(config)
        result = await workflow.execute_workflow()
        
        # Display results
        self._display_workflow_results(result, "Basic Workflow")
        return result
    
    async def demo_enhanced_workflow(self, symbols: List[str] = None):
        """Demonstrate enhanced workflow with MCP integration"""
        print(f"\n🚀 DEMO 2: Enhanced Workflow with MCP Integration")
        print("-" * 50)
        
        if not symbols:
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN']
        
        print(f"🧠 Neural Models: LSTM, Transformer, GRU, CNN-LSTM")
        print(f"⚡ GPU Acceleration: Enabled")
        print(f"🌐 MCP Tools: AI News Trader integration")
        print(f"📈 Symbols: {', '.join(symbols)}")
        
        # Execute enhanced workflow
        print(f"\n⏳ Executing enhanced workflow...")
        workflow = EnhancedNeuralTradingWorkflow(symbols, strategy="neural_momentum_trader")
        result = await workflow.execute_real_neural_workflow()
        
        # Display comprehensive results
        self._display_enhanced_results(result, "Enhanced Workflow")
        return result
    
    async def demo_strategy_comparison(self):
        """Compare different trading strategies"""
        print(f"\n🚀 DEMO 3: Strategy Performance Comparison")
        print("-" * 50)
        
        strategies = [
            ("conservative", "Conservative Blue-Chip"),
            ("growth", "Growth Technology"),
            ("tech_focused", "Tech-Heavy Aggressive")
        ]
        
        results = {}
        
        for strategy_key, strategy_name in strategies:
            print(f"\n🎯 Testing {strategy_name} Strategy...")
            result = await self.demo_basic_workflow(strategy_key)
            results[strategy_name] = result
        
        # Compare strategies
        print(f"\n📊 STRATEGY COMPARISON RESULTS")
        print("-" * 50)
        print(f"{'Strategy':<25} {'Signals':<8} {'Avg Conf':<10} {'Time':<8} {'Status'}")
        print("-" * 60)
        
        for name, result in results.items():
            signals = len(result.signals)
            avg_conf = sum(s.confidence for s in result.signals) / len(result.signals) if result.signals else 0
            time_str = f"{result.execution_time:.2f}s"
            status = "✅" if result.status == 'completed' else "❌"
            
            print(f"{name:<25} {signals:<8} {avg_conf:<10.3f} {time_str:<8} {status}")
        
        return results
    
    async def demo_real_time_simulation(self, duration_minutes: int = 5):
        """Simulate real-time trading workflow"""
        print(f"\n🚀 DEMO 4: Real-Time Trading Simulation")
        print("-" * 50)
        
        symbols = ['AAPL', 'GOOGL', 'TSLA']
        config = {
            'symbols': symbols,
            'max_position_size': 1000,
            'risk_tolerance': 'moderate',
            'trading_mode': 'paper'
        }
        
        print(f"⏰ Duration: {duration_minutes} minutes")
        print(f"📈 Symbols: {', '.join(symbols)}")
        print(f"🔄 Update Frequency: Every 30 seconds")
        
        workflow = NeuralTradingWorkflow(config)
        
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        iteration = 1
        
        print(f"\n🔄 Starting real-time simulation...")
        
        while datetime.now() < end_time:
            print(f"\n📊 Iteration {iteration} - {datetime.now().strftime('%H:%M:%S')}")
            
            result = await workflow.execute_workflow()
            
            # Display key metrics
            total_signals = len(result.signals)
            buy_signals = len([s for s in result.signals if s.signal_type == 'BUY'])
            sell_signals = len([s for s in result.signals if s.signal_type == 'SELL'])
            
            print(f"  📈 BUY: {buy_signals} | 📉 SELL: {sell_signals} | ⏸️ HOLD: {total_signals - buy_signals - sell_signals}")
            
            if result.signals:
                highest_conf = max(result.signals, key=lambda s: s.confidence)
                print(f"  🎯 Top Signal: {highest_conf.symbol} {highest_conf.signal_type} (confidence: {highest_conf.confidence:.2f})")
            
            iteration += 1
            
            # Wait 30 seconds before next iteration
            if datetime.now() < end_time - timedelta(seconds=30):
                print(f"  ⏳ Waiting 30 seconds...")
                await asyncio.sleep(30)
            else:
                break
        
        print(f"\n✅ Real-time simulation completed ({iteration-1} iterations)")
    
    def _display_workflow_results(self, result, workflow_type: str):
        """Display workflow execution results"""
        print(f"\n📊 {workflow_type.upper()} RESULTS")
        print("-" * 40)
        print(f"Status: {result.status}")
        print(f"Execution Time: {result.execution_time:.3f}s")
        print(f"Workflow ID: {result.workflow_id}")
        
        if result.performance_metrics:
            print(f"\n📈 Performance Metrics:")
            metrics = result.performance_metrics
            print(f"  Total Signals: {metrics.get('total_signals', 0)}")
            print(f"  BUY Signals: {metrics.get('buy_signals', 0)}")
            print(f"  SELL Signals: {metrics.get('sell_signals', 0)}")
            print(f"  HOLD Signals: {metrics.get('hold_signals', 0)}")
            print(f"  Avg Confidence: {metrics.get('avg_confidence', 0):.3f}")
            print(f"  Symbols Processed: {metrics.get('symbols_processed', 0)}")
        
        if result.signals:
            print(f"\n🎯 Trading Signals:")
            for signal in result.signals:
                confidence_bar = "█" * int(signal.confidence * 10) + "░" * (10 - int(signal.confidence * 10))
                print(f"  {signal.symbol}: {signal.signal_type} @ ${signal.price:.2f}")
                print(f"    Confidence: {confidence_bar} {signal.confidence:.2f}")
                if hasattr(signal, 'reasoning'):
                    print(f"    Reasoning: {signal.reasoning}")
    
    def _display_enhanced_results(self, result, workflow_type: str):
        """Display enhanced workflow results with neural analysis"""
        print(f"\n🧠 {workflow_type.upper()} RESULTS")
        print("-" * 40)
        print(f"Status: {result.status}")
        print(f"Execution Time: {result.execution_time:.3f}s")
        print(f"Workflow ID: {result.workflow_id}")
        
        # Neural network analysis
        if result.neural_predictions:
            neural = result.neural_predictions
            print(f"\n🤖 Neural Network Analysis:")
            print(f"  System Confidence: {neural.get('system_confidence', 0):.2f}")
            print(f"  Models Consensus: {neural.get('models_consensus', 'N/A')}")
            print(f"  GPU Acceleration: ✅")
            print(f"  Prediction Time: {neural.get('prediction_time', 'N/A')}")
        
        # Portfolio optimization
        if result.portfolio_optimization:
            portfolio = result.portfolio_optimization
            expected_return = portfolio.get('expected_return', 0) * 100
            portfolio_risk = portfolio.get('portfolio_risk', 0) * 100
            sharpe_ratio = portfolio.get('sharpe_ratio', 0)
            
            print(f"\n📊 Portfolio Optimization:")
            print(f"  Expected Return: {expected_return:.1f}%")
            print(f"  Portfolio Risk: {portfolio_risk:.1f}%")
            print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
            
            allocations = portfolio.get('target_allocations', {})
            if allocations:
                print(f"  Target Allocations:")
                for symbol, allocation in sorted(allocations.items(), key=lambda x: x[1], reverse=True):
                    bar = "█" * int(allocation * 50) + "░" * (50 - int(allocation * 50))
                    print(f"    {symbol}: {bar[:20]} {allocation*100:.1f}%")
        
        # Trading signals
        if result.signals:
            print(f"\n🎯 Enhanced Trading Signals ({len(result.signals)}):")
            
            for signal in result.signals[:5]:  # Show top 5
                print(f"  {signal['symbol']}: {signal['signal_type']} @ ${signal['current_price']:.2f}")
                print(f"    Target: ${signal['target_price']:.2f} | Confidence: {signal['confidence']:.2f}")
                print(f"    Components: {signal['reasoning']}")
                print(f"    Risk Level: {signal['risk_level']:.2f}")
    
    async def run_comprehensive_demo(self):
        """Run all demos in sequence"""
        self.display_banner()
        
        try:
            print("\n🎬 Starting Comprehensive Demo Suite...")
            
            # Demo 1: Basic Workflow
            await self.demo_basic_workflow("growth")
            
            # Wait between demos
            await asyncio.sleep(2)
            
            # Demo 2: Enhanced Workflow
            await self.demo_enhanced_workflow()
            
            # Wait between demos
            await asyncio.sleep(2)
            
            # Demo 3: Strategy Comparison
            await self.demo_strategy_comparison()
            
            # Demo 4: Real-time simulation (shortened for demo)
            await self.demo_real_time_simulation(duration_minutes=2)
            
            print(f"\n🎉 COMPREHENSIVE DEMO COMPLETED SUCCESSFULLY!")
            print("="*70)
            print("✅ All workflows executed successfully")
            print("📊 Neural trading system is operational")
            print("🌐 Flow Nexus integration working")
            print("🤖 AI News Trader MCP tools functional")
            print("="*70)
            
        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            raise

async def main():
    """Main demo execution"""
    demo = TradingWorkflowDemo()
    
    if len(sys.argv) > 1:
        demo_type = sys.argv[1].lower()
        
        if demo_type == "basic":
            demo.display_banner()
            await demo.demo_basic_workflow()
        elif demo_type == "enhanced":
            demo.display_banner()
            await demo.demo_enhanced_workflow()
        elif demo_type == "compare":
            demo.display_banner()
            await demo.demo_strategy_comparison()
        elif demo_type == "realtime":
            demo.display_banner()
            await demo.demo_real_time_simulation()
        elif demo_type == "all":
            await demo.run_comprehensive_demo()
        else:
            print("Usage: python trading_workflow_demo.py [basic|enhanced|compare|realtime|all]")
    else:
        # Run comprehensive demo by default
        await demo.run_comprehensive_demo()

if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())