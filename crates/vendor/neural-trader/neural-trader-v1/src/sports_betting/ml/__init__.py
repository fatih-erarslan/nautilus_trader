"""
Sports Betting Machine Learning Module.

This module contains advanced neural network models for sports betting predictions:
- Outcome prediction models (LSTM/GRU for win/loss/draw)
- Score prediction systems (Transformer models)
- Value betting detection
- Model training pipelines
- GPU acceleration support
"""

from .outcome_predictor import OutcomePredictor
from .score_predictor import ScorePredictor
from .value_detector import ValueDetector
from .training_pipeline import TrainingPipeline

__all__ = [
    "OutcomePredictor",
    "ScorePredictor", 
    "ValueDetector",
    "TrainingPipeline"
]