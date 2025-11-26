"""
Example usage of the market simulation engine.
"""
import asyncio
import json
from src.simulation.market_simulator import MarketSimulator, SimulationConfig
from src.simulation.event_simulator import EventSimulator
from src.simulation.scenarios import (
    BullMarketScenario, BearMarketScenario,
    HighVolatilityScenario, FlashCrashScenario
)


async def run_bull_market_simulation():
    """Run a bull market simulation."""
    print("Running Bull Market Simulation...")
    
    # Create scenario
    scenario = BullMarketScenario(
        symbols=["AAPL", "MSFT", "GOOGL"],
        duration=60  # 1 minute for demo
    )
    
    # Create simulator
    config = scenario.create_config()
    simulator = MarketSimulator(config)
    
    # Configure for bull market
    scenario.configure_simulator(simulator)
    
    # Create and configure event simulator
    event_sim = EventSimulator(config.symbols)
    scenario.configure_events(event_sim)
    
    # Run simulation
    result = await simulator.run()
    
    # Print results
    print(f"\nBull Market Results:")
    print(f"Duration: {result.duration:.2f} seconds")
    print(f"Total Ticks: {result.total_ticks:,}")
    print(f"Total Trades: {result.total_trades:,}")
    
    for symbol, stats in result.market_stats.items():
        print(f"\n{symbol}:")
        print(f"  Volume: {stats.total_volume:,}")
        print(f"  Trades: {stats.total_trades:,}")
        print(f"  Avg Spread: ${stats.average_spread:.4f}")
        print(f"  Volatility: {stats.price_volatility:.2%}")
        print(f"  VWAP: ${stats.vwap:.2f}")


async def run_flash_crash_simulation():
    """Run a flash crash simulation."""
    print("\nRunning Flash Crash Simulation...")
    
    # Create scenario
    scenario = FlashCrashScenario(
        symbols=["SPY", "QQQ"],
        duration=30  # 30 seconds for demo
    )
    
    # Create simulator
    config = scenario.create_config()
    simulator = MarketSimulator(config)
    
    # Configure for flash crash
    scenario.configure_simulator(simulator)
    
    # Create and configure event simulator
    event_sim = EventSimulator(config.symbols)
    scenario.configure_events(event_sim)
    
    # Schedule the actual flash crash
    crash_time = config.duration * 0.3
    simulator.schedule_event(
        time=crash_time,
        event_type="flash_crash",
        data={
            "trigger": scenario.trigger_flash_crash,
            "simulator": simulator,
            "event_sim": event_sim
        }
    )
    
    # Run simulation
    result = await simulator.run()
    
    # Analyze crash metrics
    print(f"\nFlash Crash Results:")
    print(f"Duration: {result.duration:.2f} seconds")
    print(f"Total Trades: {result.total_trades:,}")
    
    for symbol, stats in result.market_stats.items():
        prices = result.price_series.get(symbol, [])
        if prices:
            max_price = max(prices[:int(len(prices)*0.3)])  # Before crash
            min_price = min(prices)  # During crash
            final_price = prices[-1]  # After recovery
            
            crash_magnitude = (max_price - min_price) / max_price
            recovery = (final_price - min_price) / (max_price - min_price)
            
            print(f"\n{symbol}:")
            print(f"  Pre-crash price: ${max_price:.2f}")
            print(f"  Crash low: ${min_price:.2f}")
            print(f"  Final price: ${final_price:.2f}")
            print(f"  Crash magnitude: {crash_magnitude:.2%}")
            print(f"  Recovery: {recovery:.2%}")


async def run_all_scenarios():
    """Run all market scenarios."""
    scenarios = [
        ("Bull Market", BullMarketScenario),
        ("Bear Market", BearMarketScenario),
        ("High Volatility", HighVolatilityScenario),
        ("Flash Crash", FlashCrashScenario)
    ]
    
    results = {}
    
    for name, scenario_class in scenarios:
        print(f"\n{'='*50}")
        print(f"Running {name} Scenario")
        print('='*50)
        
        # Create scenario with short duration for demo
        scenario = scenario_class(
            symbols=["TEST1", "TEST2"],
            duration=10  # 10 seconds each
        )
        
        # Run simulation
        config = scenario.create_config()
        simulator = MarketSimulator(config)
        scenario.configure_simulator(simulator)
        
        result = await simulator.run()
        
        # Calculate key metrics
        total_volume = sum(stats.total_volume for stats in result.market_stats.values())
        avg_volatility = sum(stats.price_volatility for stats in result.market_stats.values()) / len(result.market_stats)
        
        results[name] = {
            "duration": result.duration,
            "ticks": result.total_ticks,
            "trades": result.total_trades,
            "volume": total_volume,
            "volatility": avg_volatility,
            "tick_rate": result.total_ticks / result.duration
        }
        
        print(f"Completed: {result.total_ticks:,} ticks, {result.total_trades:,} trades")
    
    # Summary
    print(f"\n{'='*50}")
    print("Simulation Summary")
    print('='*50)
    
    for scenario, metrics in results.items():
        print(f"\n{scenario}:")
        print(f"  Tick Rate: {metrics['tick_rate']:.0f} ticks/sec")
        print(f"  Trade Rate: {metrics['trades']/metrics['duration']:.0f} trades/sec")
        print(f"  Volatility: {metrics['volatility']:.2%}")
    
    # Save results
    with open('benchmark/simulation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to benchmark/simulation_results.json")


if __name__ == "__main__":
    # Run examples
    asyncio.run(run_bull_market_simulation())
    asyncio.run(run_flash_crash_simulation())
    
    # Uncomment to run all scenarios
    # asyncio.run(run_all_scenarios())