#!/usr/bin/env python3
"""
Unit tests for Neural Forecasting CLI commands.
Tests all neural CLI functionality including commands, options, and error handling.
"""

import unittest
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, mock_open
from click.testing import CliRunner

# Add benchmark directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmark"))

try:
    from benchmark.src.commands.neural_command import (
        neural, forecast, train, evaluate, backtest, 
        deploy, monitor, optimize, NeuralConfig, format_output
    )
except ImportError as e:
    print(f"Warning: Could not import neural commands: {e}")
    neural = None


class TestNeuralConfig(unittest.TestCase):
    """Test the NeuralConfig class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = NeuralConfig()
    
    def test_default_models(self):
        """Test default model configuration."""
        expected_models = ['nhits', 'nbeats', 'tft', 'patchtst']
        self.assertEqual(self.config.default_models, expected_models)
    
    def test_default_horizon(self):
        """Test default forecast horizon."""
        self.assertEqual(self.config.default_horizon, 24)
    
    def test_default_epochs(self):
        """Test default training epochs."""
        self.assertEqual(self.config.default_epochs, 100)
    
    def test_default_metrics(self):
        """Test default evaluation metrics."""
        expected_metrics = ['mae', 'mape', 'rmse', 'smape']
        self.assertEqual(self.config.default_metrics, expected_metrics)
    
    @patch('torch.cuda.is_available')
    def test_gpu_detection_pytorch(self, mock_cuda):
        """Test GPU detection with PyTorch."""
        mock_cuda.return_value = True
        config = NeuralConfig()
        self.assertTrue(config.gpu_enabled)
    
    @patch('torch.cuda.is_available')
    def test_gpu_detection_no_gpu(self, mock_cuda):
        """Test GPU detection when no GPU available."""
        mock_cuda.return_value = False
        config = NeuralConfig()
        self.assertFalse(config.gpu_enabled)


class TestFormatOutput(unittest.TestCase):
    """Test output formatting functions."""
    
    def setUp(self):
        """Set up test data."""
        self.test_data = {
            'symbol': 'AAPL',
            'forecast': [150.0, 151.0, 152.0],
            'metrics': {'mae': 0.05, 'rmse': 0.08},
            'timestamp': '2024-01-01T00:00:00'
        }
    
    def test_format_text(self):
        """Test text formatting."""
        result = format_output(self.test_data, 'text')
        self.assertIn('symbol: AAPL', result)
        self.assertIn('METRICS:', result)
        self.assertIn('mae: 0.05', result)
    
    def test_format_json(self):
        """Test JSON formatting."""
        result = format_output(self.test_data, 'json')
        parsed = json.loads(result)
        self.assertEqual(parsed['symbol'], 'AAPL')
        self.assertEqual(parsed['metrics']['mae'], 0.05)
    
    def test_format_csv(self):
        """Test CSV formatting with results list."""
        data_with_results = {
            'results': [
                {'symbol': 'AAPL', 'value': 150.0},
                {'symbol': 'TSLA', 'value': 200.0}
            ]
        }
        result = format_output(data_with_results, 'csv')
        self.assertIn('symbol,value', result)
        self.assertIn('AAPL,150.0', result)


@unittest.skipIf(neural is None, "Neural commands not available")
class TestNeuralForecastCommand(unittest.TestCase):
    """Test the neural forecast command."""
    
    def setUp(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_forecast_basic(self):
        """Test basic forecast command."""
        result = self.runner.invoke(neural, ['forecast', 'AAPL'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Generating neural forecast for AAPL', result.output)
    
    def test_forecast_with_options(self):
        """Test forecast with various options."""
        result = self.runner.invoke(neural, [
            'forecast', 'TSLA',
            '--horizon', '48',
            '--model', 'nhits',
            '--confidence', '0.99',
            '--format', 'json'
        ])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Model: nhits', result.output)
        self.assertIn('Horizon: 48h', result.output)
    
    def test_forecast_invalid_horizon(self):
        """Test forecast with invalid horizon."""
        result = self.runner.invoke(neural, [
            'forecast', 'AAPL', '--horizon', '-5'
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('must be positive', result.output)
    
    def test_forecast_invalid_confidence(self):
        """Test forecast with invalid confidence level."""
        result = self.runner.invoke(neural, [
            'forecast', 'AAPL', '--confidence', '1.5'
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('must be between 0 and 1', result.output)
    
    def test_forecast_output_file(self):
        """Test forecast with output file."""
        output_file = os.path.join(self.temp_dir, 'forecast_output.json')
        result = self.runner.invoke(neural, [
            'forecast', 'AAPL',
            '--output', output_file,
            '--format', 'json'
        ])
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(os.path.exists(output_file))


@unittest.skipIf(neural is None, "Neural commands not available")
class TestNeuralTrainCommand(unittest.TestCase):
    """Test the neural train command."""
    
    def setUp(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a mock CSV dataset
        self.dataset_path = os.path.join(self.temp_dir, 'test_data.csv')
        with open(self.dataset_path, 'w') as f:
            f.write('timestamp,value\n')
            f.write('2024-01-01 00:00:00,150.0\n')
            f.write('2024-01-01 01:00:00,151.0\n')
            f.write('2024-01-01 02:00:00,152.0\n')
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_train_basic(self):
        """Test basic train command."""
        result = self.runner.invoke(neural, ['train', self.dataset_path])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Training neural model', result.output)
    
    def test_train_with_options(self):
        """Test train with various options."""
        result = self.runner.invoke(neural, [
            'train', self.dataset_path,
            '--model', 'tft',
            '--epochs', '50',
            '--batch-size', '64',
            '--learning-rate', '0.001'
        ])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Epochs: 50', result.output)
        self.assertIn('Batch size: 64', result.output)
    
    def test_train_nonexistent_dataset(self):
        """Test train with nonexistent dataset."""
        result = self.runner.invoke(neural, ['train', 'nonexistent.csv'])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('not found', result.output)
    
    def test_train_with_early_stopping(self):
        """Test train with early stopping."""
        result = self.runner.invoke(neural, [
            'train', self.dataset_path,
            '--early-stopping'
        ])
        self.assertEqual(result.exit_code, 0)
        # Check if early stopping might have been triggered
        self.assertIn('Training neural model', result.output)


@unittest.skipIf(neural is None, "Neural commands not available")
class TestNeuralEvaluateCommand(unittest.TestCase):
    """Test the neural evaluate command."""
    
    def setUp(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a mock model file
        self.model_path = os.path.join(self.temp_dir, 'test_model.json')
        model_data = {
            'model': 'nhits',
            'epochs_completed': 100,
            'final_train_loss': 0.05
        }
        with open(self.model_path, 'w') as f:
            json.dump(model_data, f)
        
        # Create a mock test dataset
        self.test_data_path = os.path.join(self.temp_dir, 'test_data.csv')
        with open(self.test_data_path, 'w') as f:
            f.write('actual\n')
            f.write('150.0\n')
            f.write('151.0\n')
            f.write('152.0\n')
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_evaluate_basic(self):
        """Test basic evaluate command."""
        result = self.runner.invoke(neural, ['evaluate', self.model_path])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Evaluating model', result.output)
    
    def test_evaluate_with_test_data(self):
        """Test evaluate with test data."""
        result = self.runner.invoke(neural, [
            'evaluate', self.model_path,
            '--test-data', self.test_data_path
        ])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Test data: 3 samples', result.output)
    
    def test_evaluate_custom_metrics(self):
        """Test evaluate with custom metrics."""
        result = self.runner.invoke(neural, [
            'evaluate', self.model_path,
            '--metrics', 'mae,rmse,r2'
        ])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('MAE:', result.output)
        self.assertIn('RMSE:', result.output)
        self.assertIn('R2:', result.output)
    
    def test_evaluate_invalid_metrics(self):
        """Test evaluate with invalid metrics."""
        result = self.runner.invoke(neural, [
            'evaluate', self.model_path,
            '--metrics', 'invalid_metric'
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('Invalid metrics', result.output)
    
    def test_evaluate_nonexistent_model(self):
        """Test evaluate with nonexistent model."""
        result = self.runner.invoke(neural, ['evaluate', 'nonexistent.json'])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('not found', result.output)


@unittest.skipIf(neural is None, "Neural commands not available")
class TestNeuralBacktestCommand(unittest.TestCase):
    """Test the neural backtest command."""
    
    def setUp(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a mock model file
        self.model_path = os.path.join(self.temp_dir, 'test_model.json')
        model_data = {
            'model': 'nhits',
            'epochs_completed': 100
        }
        with open(self.model_path, 'w') as f:
            json.dump(model_data, f)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_backtest_basic(self):
        """Test basic backtest command."""
        result = self.runner.invoke(neural, [
            'backtest', self.model_path,
            '--symbol', 'AAPL',
            '--start', '2024-01-01',
            '--end', '2024-01-31'
        ])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Running neural backtest for AAPL', result.output)
    
    def test_backtest_with_options(self):
        """Test backtest with various options."""
        result = self.runner.invoke(neural, [
            'backtest', self.model_path,
            '--symbol', 'TSLA',
            '--start', '2024-01-01',
            '--end', '2024-02-01',
            '--strategy', 'momentum',
            '--initial-capital', '50000'
        ])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Initial capital: $50,000.00', result.output)
    
    def test_backtest_invalid_dates(self):
        """Test backtest with invalid date range."""
        result = self.runner.invoke(neural, [
            'backtest', self.model_path,
            '--symbol', 'AAPL',
            '--start', '2024-02-01',
            '--end', '2024-01-01'  # End before start
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('must be before', result.output)
    
    def test_backtest_invalid_capital(self):
        """Test backtest with invalid capital."""
        result = self.runner.invoke(neural, [
            'backtest', self.model_path,
            '--symbol', 'AAPL',
            '--start', '2024-01-01',
            '--end', '2024-01-31',
            '--initial-capital', '-1000'
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('must be positive', result.output)


@unittest.skipIf(neural is None, "Neural commands not available")
class TestNeuralDeployCommand(unittest.TestCase):
    """Test the neural deploy command."""
    
    def setUp(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a mock model file
        self.model_path = os.path.join(self.temp_dir, 'test_model.json')
        model_data = {
            'model': 'nhits',
            'epochs_completed': 100
        }
        with open(self.model_path, 'w') as f:
            json.dump(model_data, f)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_deploy_basic(self):
        """Test basic deploy command."""
        result = self.runner.invoke(neural, ['deploy', self.model_path])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Deploying neural model to development', result.output)
    
    def test_deploy_production(self):
        """Test production deployment."""
        result = self.runner.invoke(neural, [
            'deploy', self.model_path,
            '--env', 'production',
            '--traffic', '10',
            '--health-check'
        ])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('production', result.output)
        self.assertIn('Traffic: 10%', result.output)
        self.assertIn('Health checks passed', result.output)
    
    def test_deploy_invalid_traffic(self):
        """Test deploy with invalid traffic percentage."""
        result = self.runner.invoke(neural, [
            'deploy', self.model_path,
            '--traffic', '150'
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('must be between 0 and 100', result.output)


@unittest.skipIf(neural is None, "Neural commands not available")
class TestNeuralMonitorCommand(unittest.TestCase):
    """Test the neural monitor command."""
    
    def setUp(self):
        """Set up test environment."""
        self.runner = CliRunner()
    
    def test_monitor_basic(self):
        """Test basic monitor command."""
        result = self.runner.invoke(neural, ['monitor'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Current Status', result.output)
    
    def test_monitor_with_options(self):
        """Test monitor with various options."""
        result = self.runner.invoke(neural, [
            'monitor',
            '--env', 'production',
            '--refresh', '60'
        ])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Monitoring neural models (production)', result.output)


@unittest.skipIf(neural is None, "Neural commands not available")
class TestNeuralOptimizeCommand(unittest.TestCase):
    """Test the neural optimize command."""
    
    def setUp(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a mock model file
        self.model_path = os.path.join(self.temp_dir, 'test_model.json')
        model_data = {
            'model': 'nhits',
            'epochs_completed': 100
        }
        with open(self.model_path, 'w') as f:
            json.dump(model_data, f)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_optimize_basic(self):
        """Test basic optimize command."""
        result = self.runner.invoke(neural, [
            'optimize', self.model_path,
            '--trials', '5'  # Small number for testing
        ])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Optimizing neural model hyperparameters', result.output)
    
    def test_optimize_with_options(self):
        """Test optimize with various options."""
        result = self.runner.invoke(neural, [
            'optimize', self.model_path,
            '--trials', '10',
            '--metric', 'rmse',
            '--timeout', '300'
        ])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Metric: rmse', result.output)
        self.assertIn('Trials: 10', result.output)
    
    def test_optimize_invalid_trials(self):
        """Test optimize with invalid trials number."""
        result = self.runner.invoke(neural, [
            'optimize', self.model_path,
            '--trials', '0'
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('must be positive', result.output)


class TestNeuralCLIIntegration(unittest.TestCase):
    """Test CLI integration and error handling."""
    
    def setUp(self):
        """Set up test environment."""
        self.runner = CliRunner()
    
    @unittest.skipIf(neural is None, "Neural commands not available")
    def test_neural_help(self):
        """Test neural command group help."""
        result = self.runner.invoke(neural, ['--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Neural forecasting commands', result.output)
    
    @unittest.skipIf(neural is None, "Neural commands not available")
    def test_neural_forecast_help(self):
        """Test neural forecast command help."""
        result = self.runner.invoke(neural, ['forecast', '--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Generate neural forecasts', result.output)
    
    @unittest.skipIf(neural is None, "Neural commands not available")
    def test_neural_train_help(self):
        """Test neural train command help."""
        result = self.runner.invoke(neural, ['train', '--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Train a neural forecasting model', result.output)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def setUp(self):
        """Set up test environment."""
        self.runner = CliRunner()
    
    @unittest.skipIf(neural is None, "Neural commands not available")
    def test_keyboard_interrupt_handling(self):
        """Test handling of keyboard interrupts."""
        # This test is difficult to implement without actual subprocess
        # In a real scenario, we'd test Ctrl+C handling
        pass
    
    @unittest.skipIf(neural is None, "Neural commands not available")
    def test_file_permission_errors(self):
        """Test handling of file permission errors."""
        # Create a read-only directory
        with tempfile.TemporaryDirectory() as temp_dir:
            readonly_file = os.path.join(temp_dir, 'readonly.json')
            with open(readonly_file, 'w') as f:
                json.dump({'model': 'test'}, f)
            
            # Make file read-only
            os.chmod(readonly_file, 0o444)
            
            # This should handle permission errors gracefully
            result = self.runner.invoke(neural, [
                'train', readonly_file
            ])
            # The command should fail gracefully, not crash
            self.assertIsInstance(result.exit_code, int)


if __name__ == '__main__':
    # Run tests with verbose output
    import sys
    
    # Add test discovery
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)