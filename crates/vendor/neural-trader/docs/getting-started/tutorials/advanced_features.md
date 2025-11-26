# Advanced Features Tutorial

Master advanced neural forecasting techniques for sophisticated trading strategies and production-scale deployments.

## Overview

This tutorial covers advanced neural forecasting capabilities:

- Multi-symbol portfolio forecasting
- Exogenous variables integration
- Custom loss functions and metrics
- Hyperparameter optimization
- Probabilistic forecasting
- Real-time streaming forecasts
- Model interpretation and explainability

**Prerequisites**: Complete [Basic Forecasting Tutorial](basic_forecasting.md)

**Time**: 60-90 minutes

## Setup and Data Preparation

### Advanced Imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Core neural forecasting
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, NBEATS, NBEATSx, AutoNBEATS
from neuralforecast.losses.pytorch import MAE, MSE, MAPE, SMAPE, MASE

# Advanced features
from neuralforecast.auto import AutoNHITS
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.utils import AirPassengersDF

# Hyperparameter optimization
import optuna
from optuna.integration import PyTorchLightningPruningCallback

# Model interpretation
import shap
from captum.attr import IntegratedGradients

# Portfolio analysis
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print("✓ Advanced libraries imported successfully")
```

### Multi-Symbol Dataset

```python
def create_portfolio_data(symbols=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'], 
                         start_date='2020-01-01', end_date='2024-06-01'):
    """Create multi-symbol dataset for portfolio forecasting"""
    
    np.random.seed(42)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    portfolio_data = []
    
    for symbol in symbols:
        # Create realistic price series for each symbol
        if symbol == 'AAPL':
            initial_price, volatility, drift = 150.0, 0.02, 0.0008
        elif symbol == 'GOOGL':
            initial_price, volatility, drift = 2500.0, 0.025, 0.0012
        elif symbol == 'MSFT':
            initial_price, volatility, drift = 300.0, 0.022, 0.0010
        elif symbol == 'TSLA':
            initial_price, volatility, drift = 800.0, 0.045, 0.0015
        elif symbol == 'NVDA':
            initial_price, volatility, drift = 400.0, 0.035, 0.0020
        
        # Generate returns with correlation
        returns = np.random.normal(drift, volatility, len(dates))
        
        # Add market correlation (simplified)
        market_factor = np.random.normal(0, 0.01, len(dates))
        returns += 0.6 * market_factor  # Beta = 0.6
        
        # Convert to prices
        prices = [initial_price]
        for r in returns[1:]:
            prices.append(prices[-1] * (1 + r))
        
        # Add trend and seasonality
        trend = np.linspace(0, initial_price * 0.5, len(dates))
        seasonal = initial_price * 0.05 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
        prices = np.array(prices) + trend + seasonal
        
        # Create symbol dataframe
        symbol_data = pd.DataFrame({
            'ds': dates,
            'unique_id': symbol,
            'y': prices
        })
        
        portfolio_data.append(symbol_data)
    
    # Combine all symbols
    full_data = pd.concat(portfolio_data, ignore_index=True)
    
    print(f"Portfolio data created:")
    print(f"Symbols: {symbols}")
    print(f"Total records: {len(full_data)}")
    print(f"Date range: {full_data['ds'].min()} to {full_data['ds'].max()}")
    
    return full_data

# Create portfolio dataset
portfolio_data = create_portfolio_data()
```

### Exogenous Variables

```python
def add_exogenous_variables(data):
    """Add external factors that might influence stock prices"""
    
    # Economic indicators (simulated)
    dates = data['ds'].unique()
    
    # Market volatility index (VIX-like)
    np.random.seed(42)
    vix = 20 + 15 * np.random.beta(2, 5, len(dates))
    
    # Interest rates (Fed funds rate simulation)
    interest_rate = 2.0 + np.cumsum(np.random.normal(0, 0.01, len(dates)))
    interest_rate = np.clip(interest_rate, 0, 8)  # Realistic bounds
    
    # Dollar index
    dollar_index = 100 + np.cumsum(np.random.normal(0, 0.5, len(dates)))
    
    # Oil prices
    oil_price = 70 + np.cumsum(np.random.normal(0.01, 2, len(dates)))
    oil_price = np.clip(oil_price, 20, 150)
    
    # Technology sector sentiment (for tech stocks)
    tech_sentiment = 50 + 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25) + np.random.normal(0, 5, len(dates))
    
    # Create exogenous dataframe
    exog_data = pd.DataFrame({
        'ds': dates,
        'vix': vix,
        'interest_rate': interest_rate,
        'dollar_index': dollar_index,
        'oil_price': oil_price,
        'tech_sentiment': tech_sentiment
    })
    
    # Add technical indicators for each symbol
    enhanced_data = []
    
    for symbol in data['unique_id'].unique():
        symbol_data = data[data['unique_id'] == symbol].copy()
        
        # Merge with exogenous variables
        symbol_data = symbol_data.merge(exog_data, on='ds', how='left')
        
        # Add technical indicators
        symbol_data['price_sma_20'] = symbol_data['y'].rolling(20).mean()
        symbol_data['price_sma_50'] = symbol_data['y'].rolling(50).mean()
        symbol_data['rsi'] = calculate_rsi(symbol_data['y'])
        symbol_data['volatility_20'] = symbol_data['y'].pct_change().rolling(20).std()
        
        # Add sector-specific variables
        if symbol in ['AAPL', 'GOOGL', 'MSFT', 'NVDA']:
            symbol_data['sector_sentiment'] = symbol_data['tech_sentiment']
        else:
            symbol_data['sector_sentiment'] = 50  # Neutral for non-tech
        
        enhanced_data.append(symbol_data)
    
    result = pd.concat(enhanced_data, ignore_index=True)
    
    # Fill NaN values
    result = result.fillna(method='bfill').fillna(method='ffill')
    
    print(f"Added exogenous variables:")
    exog_cols = [col for col in result.columns if col not in ['ds', 'unique_id', 'y']]
    print(f"Exogenous columns: {exog_cols}")
    
    return result

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Add exogenous variables
enhanced_portfolio_data = add_exogenous_variables(portfolio_data)
```

## Multi-Symbol Forecasting

### Portfolio-Level Forecasting

```python
def portfolio_neural_forecast(data, horizon=30, models=None):
    """Generate forecasts for entire portfolio simultaneously"""
    
    if models is None:
        models = [
            NHITS(
                input_size=84,  # 12 weeks
                h=horizon,
                max_epochs=100,
                batch_size=64,
                n_freq_downsample=[168, 24, 1],
                stack_types=['trend', 'seasonality'],
                alias='NHITS_portfolio'
            )
        ]
    
    # Create forecaster
    nf = NeuralForecast(models=models, freq='D')
    
    print(f"Training portfolio forecaster on {data['unique_id'].nunique()} symbols...")
    
    # Fit models
    nf.fit(data)
    
    # Generate forecasts
    forecasts = nf.predict(h=horizon, level=[80, 95])
    
    print(f"✓ Portfolio forecasts generated")
    print(f"Forecast shape: {forecasts.shape}")
    
    return nf, forecasts

# Train portfolio forecaster
portfolio_nf, portfolio_forecasts = portfolio_neural_forecast(
    enhanced_portfolio_data.dropna(), horizon=30
)
```

### Cross-Symbol Analysis

```python
def analyze_cross_symbol_correlations(forecasts):
    """Analyze correlations between symbol forecasts"""
    
    # Pivot forecasts to wide format
    forecast_pivot = forecasts.pivot(index='ds', columns='unique_id', values='NHITS_portfolio')
    
    # Calculate correlation matrix
    correlation_matrix = forecast_pivot.corr()
    
    # Visualize correlations
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                fmt='.2f', square=True, linewidths=0.5)
    plt.title('Cross-Symbol Forecast Correlations', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Calculate portfolio metrics
    portfolio_returns = forecast_pivot.pct_change().dropna()
    portfolio_volatility = portfolio_returns.std()
    average_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
    
    print(f"=== Portfolio Analysis ===")
    print(f"Average cross-correlation: {average_correlation:.3f}")
    print(f"Individual volatilities:")
    for symbol in portfolio_volatility.index:
        print(f"  {symbol}: {portfolio_volatility[symbol]:.4f}")
    
    return correlation_matrix, portfolio_returns

# Analyze cross-symbol relationships
correlation_matrix, portfolio_returns = analyze_cross_symbol_correlations(portfolio_forecasts)
```

### Portfolio Optimization with Forecasts

```python
def optimize_portfolio_weights(forecasts, risk_aversion=1.0):
    """Optimize portfolio weights using neural forecasts"""
    
    # Prepare data
    forecast_pivot = forecasts.pivot(index='ds', columns='unique_id', values='NHITS_portfolio')
    expected_returns = forecast_pivot.pct_change().mean()
    cov_matrix = forecast_pivot.pct_change().cov()
    
    n_assets = len(expected_returns)
    
    # Objective function (maximize utility = return - risk_aversion * variance)
    def objective(weights):
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        return -(portfolio_return - risk_aversion * portfolio_variance)
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
    ]
    
    # Bounds (no short selling)
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Initial guess (equal weights)
    initial_guess = np.array([1.0/n_assets] * n_assets)
    
    # Optimize
    result = minimize(objective, initial_guess, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    optimal_weights = result.x
    
    # Calculate portfolio metrics
    portfolio_return = np.sum(optimal_weights * expected_returns)
    portfolio_variance = np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))
    portfolio_sharpe = portfolio_return / np.sqrt(portfolio_variance)
    
    print(f"=== Optimal Portfolio Weights ===")
    for i, symbol in enumerate(expected_returns.index):
        print(f"{symbol}: {optimal_weights[i]:.1%}")
    
    print(f"\nPortfolio Metrics:")
    print(f"Expected Return: {portfolio_return:.2%}")
    print(f"Volatility: {np.sqrt(portfolio_variance):.2%}")
    print(f"Sharpe Ratio: {portfolio_sharpe:.3f}")
    
    return optimal_weights, {
        'return': portfolio_return,
        'volatility': np.sqrt(portfolio_variance),
        'sharpe': portfolio_sharpe
    }

# Optimize portfolio
optimal_weights, portfolio_metrics = optimize_portfolio_weights(portfolio_forecasts)
```

## Advanced Model Architectures

### Exogenous Variables with NBEATSx

```python
def advanced_exogenous_modeling(data, exog_cols, horizon=30):
    """Use NBEATSx for forecasting with exogenous variables"""
    
    # Prepare exogenous variables for future periods
    # In production, you would forecast these or use known future values
    
    models = [
        NBEATSx(
            input_size=84,
            h=horizon,
            max_epochs=100,
            batch_size=32,
            stack_types=['trend', 'seasonality'],
            scaler_type='robust',
            futr_exog_list=exog_cols,  # Future exogenous variables
            hist_exog_list=exog_cols,  # Historical exogenous variables
            alias='NBEATSx_exog'
        )
    ]
    
    # Create forecaster
    nf = NeuralForecast(models=models, freq='D')
    
    print(f"Training NBEATSx with exogenous variables: {exog_cols}")
    
    # Fit model
    nf.fit(data)
    
    # For forecasting, we need future exogenous variables
    # Here we'll simulate them (in production, use actual forecasts)
    future_exog = simulate_future_exogenous(data, horizon, exog_cols)
    
    # Generate forecasts
    forecasts = nf.predict(df=future_exog, h=horizon, level=[80, 95])
    
    return nf, forecasts

def simulate_future_exogenous(data, horizon, exog_cols):
    """Simulate future exogenous variables for forecasting"""
    
    future_data = []
    
    for symbol in data['unique_id'].unique():
        symbol_data = data[data['unique_id'] == symbol].tail(1).copy()
        last_date = symbol_data['ds'].iloc[0]
        
        # Create future dates
        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                   periods=horizon, freq='D')
        
        # Create future exogenous values (simplified approach)
        future_symbol_data = []
        
        for date in future_dates:
            row = {
                'ds': date,
                'unique_id': symbol
            }
            
            # Simple persistence model for exogenous variables
            for col in exog_cols:
                if col in symbol_data.columns:
                    # Add small random walk
                    last_value = symbol_data[col].iloc[0]
                    row[col] = last_value + np.random.normal(0, abs(last_value) * 0.01)
                else:
                    row[col] = 0
            
            future_symbol_data.append(row)
        
        future_data.extend(future_symbol_data)
    
    return pd.DataFrame(future_data)

# Example with selected exogenous variables
exog_cols = ['vix', 'interest_rate', 'tech_sentiment', 'rsi']
exog_nf, exog_forecasts = advanced_exogenous_modeling(
    enhanced_portfolio_data.dropna(), exog_cols, horizon=30
)
```

### Custom Loss Functions

```python
class FinancialLoss:
    """Custom loss functions for financial forecasting"""
    
    @staticmethod
    def directional_loss(y_true, y_pred):
        """Loss that penalizes wrong directional predictions more heavily"""
        import torch
        
        # Calculate actual and predicted directions
        y_true_diff = torch.diff(y_true, dim=1)
        y_pred_diff = torch.diff(y_pred, dim=1)
        
        # Directional accuracy
        direction_correct = torch.sign(y_true_diff) == torch.sign(y_pred_diff)
        
        # Standard MSE loss
        mse_loss = torch.mean((y_true - y_pred) ** 2)
        
        # Directional penalty
        direction_penalty = 10 * torch.mean((~direction_correct).float())
        
        return mse_loss + direction_penalty
    
    @staticmethod
    def asymmetric_loss(y_true, y_pred, alpha=0.7):
        """Asymmetric loss that penalizes overestimation more than underestimation"""
        import torch
        
        error = y_true - y_pred
        loss = torch.where(error >= 0, 
                          alpha * error**2, 
                          (1-alpha) * error**2)
        
        return torch.mean(loss)
    
    @staticmethod
    def volatility_adjusted_loss(y_true, y_pred, volatility):
        """Loss function adjusted by volatility"""
        import torch
        
        # Higher penalty for errors during high volatility periods
        error = y_true - y_pred
        volatility_weights = 1 + volatility
        weighted_error = error * volatility_weights
        
        return torch.mean(weighted_error ** 2)

# Example usage with custom loss
def train_with_custom_loss(data, loss_fn='directional'):
    """Train model with custom loss function"""
    
    if loss_fn == 'directional':
        loss = FinancialLoss.directional_loss
    elif loss_fn == 'asymmetric':
        loss = FinancialLoss.asymmetric_loss
    else:
        loss = MSE()
    
    model = NHITS(
        input_size=84,
        h=30,
        max_epochs=50,
        loss=loss,
        alias=f'NHITS_{loss_fn}'
    )
    
    nf = NeuralForecast(models=[model], freq='D')
    nf.fit(data)
    
    return nf

# Train with custom loss
# custom_loss_nf = train_with_custom_loss(enhanced_portfolio_data.dropna(), 'directional')
```

## Hyperparameter Optimization

### Automated Hyperparameter Tuning

```python
def optimize_hyperparameters(data, n_trials=50, symbol='AAPL'):
    """Optimize neural forecasting hyperparameters using Optuna"""
    
    # Filter data for single symbol to speed up optimization
    symbol_data = data[data['unique_id'] == symbol].copy()
    
    def objective(trial):
        # Suggest hyperparameters
        input_size = trial.suggest_categorical('input_size', [28, 56, 84, 112])
        max_epochs = trial.suggest_int('max_epochs', 20, 100)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        
        # NHITS specific parameters
        n_blocks = [trial.suggest_int('n_blocks_1', 1, 3), 
                   trial.suggest_int('n_blocks_2', 1, 3)]
        mlp_units_size = trial.suggest_categorical('mlp_units_size', [256, 512, 1024])
        
        try:
            # Create model with suggested hyperparameters
            model = NHITS(
                input_size=input_size,
                h=30,
                max_epochs=max_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                n_blocks=n_blocks,
                mlp_units=[[mlp_units_size, mlp_units_size], 
                          [mlp_units_size, mlp_units_size]],
                early_stop_patience_steps=10,
                alias='optuna_trial'
            )
            
            nf = NeuralForecast(models=[model], freq='D')
            
            # Split data for validation
            train_size = int(len(symbol_data) * 0.8)
            train_data = symbol_data.iloc[:train_size]
            val_data = symbol_data.iloc[train_size:]
            
            # Train model
            nf.fit(train_data)
            
            # Generate forecasts
            forecasts = nf.predict(h=len(val_data))
            
            # Calculate validation MAPE
            merged = forecasts.merge(val_data[['ds', 'y']], on='ds', how='inner')
            if len(merged) > 0:
                mape = np.mean(np.abs((merged['y'] - merged['optuna_trial']) / merged['y'])) * 100
                return mape
            else:
                return float('inf')
                
        except Exception as e:
            print(f"Trial failed: {e}")
            return float('inf')
    
    # Run optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=1800)  # 30 min timeout
    
    print(f"=== Hyperparameter Optimization Results ===")
    print(f"Best MAPE: {study.best_value:.2f}%")
    print(f"Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study.best_params

# Run hyperparameter optimization (warning: can take 30+ minutes)
# best_params = optimize_hyperparameters(enhanced_portfolio_data.dropna(), n_trials=20)
```

### AutoML for Neural Forecasting

```python
def auto_neural_forecasting(data, symbol='AAPL', horizon=30):
    """Use AutoNHITS for automated model selection"""
    
    symbol_data = data[data['unique_id'] == symbol].copy()
    
    # AutoNHITS automatically optimizes hyperparameters
    auto_model = AutoNHITS(
        h=horizon,
        loss=MAE(),
        config={
            'max_steps': 500,
            'val_check_steps': 50,
            'early_stop_patience_steps': 20
        },
        num_samples=10,  # Number of hyperparameter combinations to try
        alias='AutoNHITS'
    )
    
    nf = NeuralForecast(models=[auto_model], freq='D')
    
    print("Running AutoNHITS optimization...")
    nf.fit(symbol_data)
    
    forecasts = nf.predict(h=horizon, level=[80, 95])
    
    print("✓ AutoNHITS optimization completed")
    
    return nf, forecasts

# Run AutoML forecasting
# auto_nf, auto_forecasts = auto_neural_forecasting(enhanced_portfolio_data.dropna())
```

## Probabilistic Forecasting

### Monte Carlo Prediction Intervals

```python
def probabilistic_forecasting(nf, n_samples=1000, horizon=30):
    """Generate probabilistic forecasts using Monte Carlo sampling"""
    
    # Generate multiple forecast samples
    forecast_samples = []
    
    for i in range(n_samples):
        # Add noise to input data for Monte Carlo sampling
        # This simulates uncertainty in the input
        sample_forecast = nf.predict(h=horizon)
        forecast_samples.append(sample_forecast['NHITS_portfolio'].values)
    
    # Calculate percentiles
    forecast_array = np.array(forecast_samples)
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    
    # Create probabilistic forecast dataframe
    base_forecast = nf.predict(h=horizon)
    prob_forecast = base_forecast[['ds', 'unique_id']].copy()
    
    for p in percentiles:
        prob_forecast[f'p{p}'] = np.percentile(forecast_array, p, axis=0)
    
    # Add mean and std
    prob_forecast['mean'] = np.mean(forecast_array, axis=0)
    prob_forecast['std'] = np.std(forecast_array, axis=0)
    
    return prob_forecast

def plot_probabilistic_forecast(prob_forecast, historical_data, symbol='AAPL'):
    """Plot probabilistic forecast with uncertainty bands"""
    
    symbol_hist = historical_data[historical_data['unique_id'] == symbol].tail(100)
    symbol_prob = prob_forecast[prob_forecast['unique_id'] == symbol]
    
    plt.figure(figsize=(16, 8))
    
    # Plot historical data
    plt.plot(symbol_hist['ds'], symbol_hist['y'], 
             label='Historical', linewidth=2, color='blue')
    
    # Plot forecast mean
    plt.plot(symbol_prob['ds'], symbol_prob['mean'], 
             label='Forecast Mean', linewidth=2, color='red', linestyle='--')
    
    # Plot uncertainty bands
    colors = ['red', 'orange', 'yellow']
    alphas = [0.1, 0.15, 0.2]
    
    for i, (lower, upper) in enumerate([(5, 95), (10, 90), (25, 75)]):
        plt.fill_between(symbol_prob['ds'], 
                        symbol_prob[f'p{lower}'], 
                        symbol_prob[f'p{upper}'],
                        alpha=alphas[i], color=colors[i], 
                        label=f'{100-2*lower}% Confidence')
    
    plt.title(f'{symbol} - Probabilistic Forecast', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Print forecast statistics
    print(f"=== Probabilistic Forecast Summary for {symbol} ===")
    print(f"Mean forecast: ${symbol_prob['mean'].iloc[-1]:.2f}")
    print(f"Std deviation: ${symbol_prob['std'].iloc[-1]:.2f}")
    print(f"90% Confidence Interval: ${symbol_prob['p5'].iloc[-1]:.2f} - ${symbol_prob['p95'].iloc[-1]:.2f}")
    
    return symbol_prob

# Generate probabilistic forecasts
# prob_forecasts = probabilistic_forecasting(portfolio_nf, n_samples=500)
# plot_probabilistic_forecast(prob_forecasts, enhanced_portfolio_data)
```

### Value at Risk (VaR) Calculation

```python
def calculate_portfolio_var(prob_forecasts, weights, confidence_level=0.05):
    """Calculate Portfolio Value at Risk using probabilistic forecasts"""
    
    # Get forecast distributions for all symbols
    symbols = prob_forecasts['unique_id'].unique()
    
    # Calculate portfolio value distribution
    portfolio_values = []
    
    for percentile in range(1, 100):  # Use all percentiles
        portfolio_value = 0
        for i, symbol in enumerate(symbols):
            symbol_forecast = prob_forecasts[prob_forecasts['unique_id'] == symbol]
            if len(symbol_forecast) > 0:
                # Interpolate percentile value
                p_col = f'p{percentile}' if f'p{percentile}' in symbol_forecast.columns else 'mean'
                if p_col in symbol_forecast.columns:
                    portfolio_value += weights[i] * symbol_forecast[p_col].iloc[-1]
        
        portfolio_values.append(portfolio_value)
    
    # Calculate VaR
    portfolio_values = np.array(portfolio_values)
    var_value = np.percentile(portfolio_values, confidence_level * 100)
    
    # Calculate Conditional VaR (Expected Shortfall)
    cvar_value = np.mean(portfolio_values[portfolio_values <= var_value])
    
    print(f"=== Portfolio Risk Metrics ===")
    print(f"Portfolio weights: {dict(zip(symbols, weights))}")
    print(f"VaR ({confidence_level:.1%}): ${var_value:.2f}")
    print(f"CVaR ({confidence_level:.1%}): ${cvar_value:.2f}")
    print(f"Expected portfolio value: ${np.mean(portfolio_values):.2f}")
    
    return {
        'var': var_value,
        'cvar': cvar_value,
        'expected_value': np.mean(portfolio_values),
        'portfolio_distribution': portfolio_values
    }

# Calculate portfolio VaR
# var_metrics = calculate_portfolio_var(prob_forecasts, optimal_weights)
```

## Real-time Streaming Forecasts

### Streaming Forecast System

```python
class StreamingForecaster:
    """Real-time streaming forecasting system"""
    
    def __init__(self, model, update_frequency='1H', max_history=1000):
        self.model = model
        self.update_frequency = update_frequency
        self.max_history = max_history
        self.data_buffer = {}
        self.last_update = {}
        
    def add_data_point(self, symbol, timestamp, price, exog_data=None):
        """Add new data point to streaming buffer"""
        
        if symbol not in self.data_buffer:
            self.data_buffer[symbol] = []
        
        data_point = {
            'ds': timestamp,
            'unique_id': symbol,
            'y': price
        }
        
        if exog_data:
            data_point.update(exog_data)
        
        self.data_buffer[symbol].append(data_point)
        
        # Maintain buffer size
        if len(self.data_buffer[symbol]) > self.max_history:
            self.data_buffer[symbol] = self.data_buffer[symbol][-self.max_history:]
    
    def should_update_forecast(self, symbol):
        """Check if forecast should be updated based on frequency"""
        
        if symbol not in self.last_update:
            return True
        
        now = datetime.now()
        if self.update_frequency == '1H':
            return (now - self.last_update[symbol]).total_seconds() > 3600
        elif self.update_frequency == '15min':
            return (now - self.last_update[symbol]).total_seconds() > 900
        
        return False
    
    def generate_streaming_forecast(self, symbol, horizon=30):
        """Generate forecast for symbol using latest data"""
        
        if symbol not in self.data_buffer or len(self.data_buffer[symbol]) < 50:
            return None
        
        # Convert buffer to DataFrame
        data_df = pd.DataFrame(self.data_buffer[symbol])
        
        # Retrain model with latest data (in production, use incremental learning)
        self.model.fit(data_df)
        
        # Generate forecast
        forecast = self.model.predict(h=horizon, level=[80, 95])
        
        # Update last forecast time
        self.last_update[symbol] = datetime.now()
        
        return forecast
    
    def get_forecast_signal(self, symbol):
        """Get trading signal from latest forecast"""
        
        forecast = self.generate_streaming_forecast(symbol, horizon=5)
        
        if forecast is None:
            return None
        
        # Calculate signal strength
        current_price = forecast.iloc[0]['NHITS_portfolio']
        future_price = forecast.iloc[-1]['NHITS_portfolio']
        expected_return = (future_price - current_price) / current_price
        
        # Determine signal
        if expected_return > 0.02:  # 2% threshold
            signal = 'BUY'
        elif expected_return < -0.02:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        return {
            'symbol': symbol,
            'signal': signal,
            'expected_return': expected_return,
            'confidence': abs(expected_return) * 10,  # Simple confidence measure
            'timestamp': datetime.now()
        }

# Example usage
streaming_forecaster = StreamingForecaster(portfolio_nf, update_frequency='15min')

# Simulate streaming data
def simulate_streaming_data(forecaster, symbols, duration_minutes=60):
    """Simulate real-time data streaming"""
    
    import time
    from datetime import datetime, timedelta
    
    start_time = datetime.now()
    
    for minute in range(duration_minutes):
        current_time = start_time + timedelta(minutes=minute)
        
        for symbol in symbols:
            # Simulate price update (in production, get from market data feed)
            base_price = 150 if symbol == 'AAPL' else 300
            price = base_price + np.random.normal(0, 5)
            
            # Add data point
            forecaster.add_data_point(symbol, current_time, price)
            
            # Check for forecast update
            if forecaster.should_update_forecast(symbol):
                signal = forecaster.get_forecast_signal(symbol)
                if signal:
                    print(f"{current_time}: {symbol} - {signal['signal']} "
                          f"(Expected return: {signal['expected_return']:.2%})")
        
        # Sleep to simulate real-time (remove in production)
        time.sleep(0.1)

# Run simulation (optional)
# simulate_streaming_data(streaming_forecaster, ['AAPL', 'GOOGL'], duration_minutes=10)
```

## Model Interpretation and Explainability

### Feature Importance Analysis

```python
def analyze_feature_importance(model, data, symbol='AAPL'):
    """Analyze which features are most important for forecasting"""
    
    symbol_data = data[data['unique_id'] == symbol].copy()
    
    # Get feature columns (excluding target and identifiers)
    feature_cols = [col for col in symbol_data.columns 
                   if col not in ['ds', 'unique_id', 'y']]
    
    if len(feature_cols) == 0:
        print("No exogenous features found for importance analysis")
        return
    
    # Prepare data for analysis
    X = symbol_data[feature_cols].fillna(0).values
    y = symbol_data['y'].values
    
    # Use permutation importance
    from sklearn.metrics import mean_absolute_error
    
    # Get baseline score
    forecast_result = model.predict(h=30)
    baseline_forecast = forecast_result[forecast_result['unique_id'] == symbol]
    
    if len(baseline_forecast) == 0:
        print("No forecast available for importance analysis")
        return
    
    # Calculate feature importance using correlation with target
    importances = {}
    
    for i, feature in enumerate(feature_cols):
        correlation = np.corrcoef(X[:, i], y[1:])[0, 1]  # Lag-1 correlation
        importances[feature] = abs(correlation)
    
    # Sort by importance
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    
    # Visualize feature importance
    features, importance_scores = zip(*sorted_features)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(features)), importance_scores)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Absolute Correlation with Target')
    plt.title(f'Feature Importance Analysis - {symbol}')
    plt.tight_layout()
    plt.show()
    
    print(f"=== Feature Importance for {symbol} ===")
    for feature, importance in sorted_features:
        print(f"{feature}: {importance:.3f}")
    
    return sorted_features

# Analyze feature importance
# feature_importance = analyze_feature_importance(portfolio_nf, enhanced_portfolio_data)
```

### Forecast Decomposition

```python
def decompose_forecast_components(forecasts, symbol='AAPL'):
    """Decompose forecast into trend and seasonal components"""
    
    symbol_forecasts = forecasts[forecasts['unique_id'] == symbol].copy()
    
    if len(symbol_forecasts) == 0:
        print(f"No forecasts found for {symbol}")
        return
    
    # Extract forecast values
    forecast_values = symbol_forecasts['NHITS_portfolio'].values
    
    # Simple trend decomposition
    from scipy import signal
    
    # Detrend to get residuals
    detrended = signal.detrend(forecast_values)
    trend = forecast_values - detrended
    
    # Extract seasonal component (simplified)
    if len(forecast_values) >= 7:  # Weekly seasonality
        seasonal_period = 7
        seasonal = np.tile(np.mean(detrended[:seasonal_period]), 
                          len(forecast_values) // seasonal_period + 1)[:len(forecast_values)]
        residual = detrended - seasonal[:len(detrended)]
    else:
        seasonal = np.zeros_like(detrended)
        residual = detrended
    
    # Plot decomposition
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Original forecast
    axes[0].plot(symbol_forecasts['ds'], forecast_values, 'b-', linewidth=2)
    axes[0].set_title(f'{symbol} - Original Forecast')
    axes[0].set_ylabel('Price ($)')
    axes[0].grid(True, alpha=0.3)
    
    # Trend component
    axes[1].plot(symbol_forecasts['ds'], trend, 'g-', linewidth=2)
    axes[1].set_title('Trend Component')
    axes[1].set_ylabel('Trend')
    axes[1].grid(True, alpha=0.3)
    
    # Seasonal component
    axes[2].plot(symbol_forecasts['ds'], seasonal[:len(symbol_forecasts)], 'r-', linewidth=2)
    axes[2].set_title('Seasonal Component')
    axes[2].set_ylabel('Seasonal')
    axes[2].grid(True, alpha=0.3)
    
    # Residual component
    axes[3].plot(symbol_forecasts['ds'], residual, 'm-', linewidth=2)
    axes[3].set_title('Residual Component')
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Date')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print component statistics
    print(f"=== Forecast Decomposition for {symbol} ===")
    print(f"Trend variance: {np.var(trend):.2f}")
    print(f"Seasonal variance: {np.var(seasonal[:len(symbol_forecasts)]):.2f}")
    print(f"Residual variance: {np.var(residual):.2f}")
    print(f"Signal-to-noise ratio: {np.var(trend) / np.var(residual):.2f}")
    
    return {
        'original': forecast_values,
        'trend': trend,
        'seasonal': seasonal[:len(symbol_forecasts)],
        'residual': residual
    }

# Decompose forecasts
# decomposition = decompose_forecast_components(portfolio_forecasts)
```

## Production Integration

### Model Versioning and A/B Testing

```python
class ModelVersionManager:
    """Manage multiple model versions for A/B testing"""
    
    def __init__(self):
        self.models = {}
        self.performance_metrics = {}
        
    def register_model(self, version_name, model, description=""):
        """Register a new model version"""
        self.models[version_name] = {
            'model': model,
            'description': description,
            'created_at': datetime.now(),
            'prediction_count': 0
        }
        self.performance_metrics[version_name] = []
        
    def predict_with_version(self, version_name, data, **kwargs):
        """Generate prediction with specific model version"""
        if version_name not in self.models:
            raise ValueError(f"Model version {version_name} not found")
        
        model = self.models[version_name]['model']
        forecast = model.predict(data, **kwargs)
        
        # Update prediction count
        self.models[version_name]['prediction_count'] += 1
        
        return forecast
    
    def record_performance(self, version_name, actual_value, predicted_value):
        """Record performance metric for a model version"""
        if version_name in self.performance_metrics:
            error = abs(actual_value - predicted_value) / actual_value
            self.performance_metrics[version_name].append(error)
    
    def get_version_performance(self, version_name):
        """Get performance statistics for a model version"""
        if version_name not in self.performance_metrics:
            return None
        
        errors = self.performance_metrics[version_name]
        if not errors:
            return None
        
        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'prediction_count': len(errors),
            'last_updated': datetime.now()
        }
    
    def compare_versions(self):
        """Compare performance across all model versions"""
        comparison = {}
        
        for version in self.models.keys():
            perf = self.get_version_performance(version)
            if perf:
                comparison[version] = perf
        
        print("=== Model Version Comparison ===")
        for version, metrics in comparison.items():
            print(f"{version}:")
            print(f"  Mean Error: {metrics['mean_error']:.3f}")
            print(f"  Std Error: {metrics['std_error']:.3f}")
            print(f"  Predictions: {metrics['prediction_count']}")
            print()
        
        return comparison

# Example usage
version_manager = ModelVersionManager()

# Register different model versions
version_manager.register_model('v1_baseline', portfolio_nf, "Baseline NHITS model")
# version_manager.register_model('v2_exogenous', exog_nf, "NHITS with exogenous variables")
# version_manager.register_model('v3_auto', auto_nf, "AutoNHITS optimized model")
```

### Automated Model Retraining

```python
class AutoRetrainingSystem:
    """Automated model retraining system"""
    
    def __init__(self, model, retrain_threshold=0.1, min_new_data=100):
        self.model = model
        self.retrain_threshold = retrain_threshold  # Performance degradation threshold
        self.min_new_data = min_new_data
        self.baseline_performance = None
        self.recent_errors = []
        self.last_retrain = datetime.now()
        
    def evaluate_current_performance(self, actual_values, predicted_values):
        """Evaluate current model performance"""
        if len(actual_values) != len(predicted_values):
            return None
        
        errors = [abs(a - p) / a for a, p in zip(actual_values, predicted_values)]
        current_mape = np.mean(errors)
        
        return current_mape
    
    def should_retrain(self, current_performance):
        """Determine if model should be retrained"""
        if self.baseline_performance is None:
            self.baseline_performance = current_performance
            return False
        
        # Check performance degradation
        degradation = (current_performance - self.baseline_performance) / self.baseline_performance
        
        if degradation > self.retrain_threshold:
            return True
        
        # Check time since last retrain
        days_since_retrain = (datetime.now() - self.last_retrain).days
        if days_since_retrain > 30:  # Monthly retraining
            return True
        
        return False
    
    def retrain_model(self, new_data):
        """Retrain model with new data"""
        print(f"Retraining model with {len(new_data)} new data points...")
        
        try:
            # Retrain model
            self.model.fit(new_data)
            
            # Update retraining timestamp
            self.last_retrain = datetime.now()
            
            print("✓ Model retraining completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Model retraining failed: {e}")
            return False
    
    def monitor_and_retrain(self, new_data, recent_actual, recent_predicted):
        """Monitor performance and retrain if necessary"""
        
        # Evaluate current performance
        current_perf = self.evaluate_current_performance(recent_actual, recent_predicted)
        
        if current_perf is None:
            return False
        
        print(f"Current MAPE: {current_perf:.3f}")
        if self.baseline_performance:
            print(f"Baseline MAPE: {self.baseline_performance:.3f}")
        
        # Check if retraining is needed
        if self.should_retrain(current_perf) and len(new_data) >= self.min_new_data:
            success = self.retrain_model(new_data)
            if success:
                # Update baseline performance
                self.baseline_performance = current_perf
            return success
        
        return False

# Example usage
auto_retrain = AutoRetrainingSystem(portfolio_nf, retrain_threshold=0.15)

# Simulate monitoring
# new_data = enhanced_portfolio_data.tail(200)  # New data for retraining
# recent_actual = [150, 152, 148, 151]  # Recent actual prices
# recent_predicted = [149, 153, 147, 152]  # Recent predictions

# retrained = auto_retrain.monitor_and_retrain(new_data, recent_actual, recent_predicted)
```

## Performance Optimization for Production

### Batch Forecasting for Multiple Symbols

```python
def optimized_batch_forecasting(data, symbols, horizon=30, batch_size=32):
    """Optimized batch forecasting for multiple symbols"""
    
    # Group symbols into batches for efficient processing
    symbol_batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
    
    all_forecasts = []
    
    for batch_idx, symbol_batch in enumerate(symbol_batches):
        print(f"Processing batch {batch_idx + 1}/{len(symbol_batches)} "
              f"({len(symbol_batch)} symbols)")
        
        # Filter data for current batch
        batch_data = data[data['unique_id'].isin(symbol_batch)]
        
        if len(batch_data) == 0:
            continue
        
        # Create optimized model for batch processing
        batch_model = NHITS(
            input_size=56,
            h=horizon,
            max_epochs=50,
            batch_size=64,  # Larger batch size for efficiency
            accelerator='auto',
            enable_progress_bar=False,  # Disable for batch processing
            alias=f'batch_{batch_idx}'
        )
        
        # Train and predict
        batch_nf = NeuralForecast(models=[batch_model], freq='D')
        batch_nf.fit(batch_data)
        batch_forecasts = batch_nf.predict(h=horizon, level=[80, 95])
        
        all_forecasts.append(batch_forecasts)
    
    # Combine all forecasts
    combined_forecasts = pd.concat(all_forecasts, ignore_index=True)
    
    print(f"✓ Batch forecasting completed for {len(symbols)} symbols")
    print(f"Total forecasts generated: {len(combined_forecasts)}")
    
    return combined_forecasts

# Example batch forecasting
symbols = enhanced_portfolio_data['unique_id'].unique()
# batch_forecasts = optimized_batch_forecasting(enhanced_portfolio_data.dropna(), symbols)
```

### Caching and Memoization

```python
from functools import lru_cache
import hashlib
import pickle
import os

class ForecastCache:
    """Caching system for neural forecasts"""
    
    def __init__(self, cache_dir='forecast_cache', max_cache_size=1000):
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        os.makedirs(cache_dir, exist_ok=True)
        
    def _generate_cache_key(self, data, model_params, horizon):
        """Generate unique cache key for forecast request"""
        # Create hash from data and parameters
        data_hash = hashlib.md5(pd.util.hash_pandas_object(data).values).hexdigest()
        param_hash = hashlib.md5(str(model_params).encode()).hexdigest()
        return f"{data_hash}_{param_hash}_{horizon}"
    
    def get_cached_forecast(self, cache_key):
        """Retrieve cached forecast if exists"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Check if cache is still valid (not older than 1 hour)
                if (datetime.now() - cached_data['timestamp']).total_seconds() < 3600:
                    print("✓ Using cached forecast")
                    return cached_data['forecast']
            except:
                pass
        
        return None
    
    def cache_forecast(self, cache_key, forecast):
        """Cache forecast result"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        cache_data = {
            'forecast': forecast,
            'timestamp': datetime.now()
        }
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            print(f"Warning: Could not cache forecast: {e}")
    
    def cached_forecast(self, model, data, horizon=30, **kwargs):
        """Generate forecast with caching"""
        
        # Generate cache key
        model_params = getattr(model, 'alias', 'unknown')
        cache_key = self._generate_cache_key(data, model_params, horizon)
        
        # Try to get cached result
        cached_result = self.get_cached_forecast(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Generate new forecast
        print("Generating new forecast...")
        forecast = model.predict(h=horizon, **kwargs)
        
        # Cache the result
        self.cache_forecast(cache_key, forecast)
        
        return forecast

# Example usage
forecast_cache = ForecastCache()
# cached_forecast = forecast_cache.cached_forecast(portfolio_nf, enhanced_portfolio_data.dropna())
```

## Summary and Next Steps

### Advanced Techniques Mastered

In this tutorial, you've learned:

✅ **Multi-symbol portfolio forecasting** with correlation analysis  
✅ **Exogenous variables integration** for enhanced accuracy  
✅ **Custom loss functions** for financial-specific optimization  
✅ **Hyperparameter optimization** using Optuna and AutoML  
✅ **Probabilistic forecasting** with uncertainty quantification  
✅ **Real-time streaming forecasts** for live trading  
✅ **Model interpretation** and explainability techniques  
✅ **Production systems** with versioning and auto-retraining  
✅ **Performance optimization** for large-scale deployment  

### Key Advanced Concepts

1. **Portfolio-Level Modeling**: Forecast multiple assets simultaneously
2. **Exogenous Enhancement**: Incorporate external market factors
3. **Custom Objectives**: Design loss functions for specific trading goals
4. **Automated Optimization**: Use AutoML for model selection
5. **Uncertainty Quantification**: Generate probabilistic forecasts
6. **Real-time Adaptation**: Stream processing and online learning
7. **Model Governance**: Version control and performance monitoring
8. **Production Scale**: Batch processing and caching strategies

### Production Deployment Checklist

Before deploying advanced neural forecasting in production:

- [ ] **Data Pipeline**: Automated data collection and validation
- [ ] **Model Monitoring**: Performance tracking and alerting
- [ ] **Version Control**: Model versioning and rollback capabilities
- [ ] **A/B Testing**: Compare model versions safely
- [ ] **Scalability**: Handle multiple symbols and high frequency updates
- [ ] **Error Handling**: Robust exception handling and fallbacks
- [ ] **Security**: Secure API endpoints and data access
- [ ] **Documentation**: Comprehensive operational documentation

### Next Learning Paths

Continue your neural forecasting journey:

1. **GPU Optimization**: [GPU Optimization Tutorial](gpu_optimization.md)
   - CUDA acceleration
   - Memory optimization
   - Distributed training

2. **Integration Examples**: [Python API Examples](../examples/python_api.py)
   - Trading strategy integration
   - Risk management systems
   - Portfolio optimization

3. **Production Deployment**: [Deployment Guide](../guides/deployment.md)
   - Kubernetes deployment
   - Monitoring and alerting
   - High availability setup

4. **Research and Development**:
   - Custom model architectures
   - Novel loss functions
   - Advanced ensemble methods

### Best Practices for Advanced Usage

- **Start Simple**: Begin with basic models and add complexity gradually
- **Validate Thoroughly**: Use proper cross-validation for all techniques
- **Monitor Continuously**: Track model performance in production
- **Document Everything**: Maintain detailed records of experiments
- **Test Incrementally**: Use A/B testing for production changes
- **Plan for Scale**: Design systems to handle growth
- **Stay Updated**: Follow latest research and best practices

### Common Advanced Pitfalls

- **Overfitting to Recent Data**: Ensure models generalize well
- **Ignoring Transaction Costs**: Include realistic trading costs
- **Overcomplicating Models**: More complex isn't always better
- **Poor Feature Engineering**: Ensure exogenous variables are meaningful
- **Inadequate Testing**: Test all edge cases and failure modes
- **Neglecting Interpretability**: Understand what models are learning

**Congratulations!** You now have advanced neural forecasting capabilities that can power sophisticated trading systems. Use these techniques responsibly and always validate results thoroughly before making trading decisions.

---

*This tutorial represents the cutting-edge of neural forecasting for financial markets. Continue experimenting and learning to stay at the forefront of AI-powered trading technology.*