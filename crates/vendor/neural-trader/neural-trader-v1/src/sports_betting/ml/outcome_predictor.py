"""
Outcome Prediction Models for Sports Betting.

This module implements advanced neural network models for predicting game outcomes
including win/loss/draw probabilities using LSTM/GRU architectures with team
performance analysis and feature engineering.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class OutcomePrediction:
    """Outcome prediction result."""
    home_win_prob: float
    draw_prob: float  # For sports that allow draws
    away_win_prob: float
    confidence: float
    model_version: str
    prediction_date: str
    features_used: List[str]
    

@dataclass
class TeamStats:
    """Team performance statistics."""
    team_name: str
    recent_form: List[int]  # Recent match results (1=win, 0=draw, -1=loss)
    goals_scored_avg: float
    goals_conceded_avg: float
    home_performance: float
    away_performance: float
    head_to_head_record: Dict[str, int]
    injury_list: List[str]
    player_ratings: Dict[str, float]


class LSTMOutcomeModel(nn.Module):
    """LSTM-based outcome prediction model."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,  # Win/Draw/Loss
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super(LSTMOutcomeModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate final hidden size
        final_hidden_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Classification layers
        self.fc_layers = nn.Sequential(
            nn.Linear(final_hidden_size, final_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_hidden_size // 2, final_hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_hidden_size // 4, num_classes)
        )
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            final_hidden = hidden[-1]
        
        # Classification
        output = self.fc_layers(final_hidden)
        probabilities = self.softmax(output)
        
        return probabilities


class GRUOutcomeModel(nn.Module):
    """GRU-based outcome prediction model."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super(GRUOutcomeModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate final hidden size
        final_hidden_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Classification layers with attention
        self.attention = nn.Sequential(
            nn.Linear(final_hidden_size, final_hidden_size // 2),
            nn.Tanh(),
            nn.Linear(final_hidden_size // 2, 1)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(final_hidden_size, final_hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(final_hidden_size // 2),
            nn.Dropout(dropout),
            nn.Linear(final_hidden_size // 2, final_hidden_size // 4),
            nn.ReLU(),
            nn.BatchNorm1d(final_hidden_size // 4),
            nn.Dropout(dropout),
            nn.Linear(final_hidden_size // 4, num_classes)
        )
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # GRU forward pass
        gru_out, hidden = self.gru(x)
        
        # Apply attention mechanism
        attention_weights = self.attention(gru_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Weighted sum of GRU outputs
        context_vector = torch.sum(gru_out * attention_weights, dim=1)
        
        # Classification
        output = self.fc_layers(context_vector)
        probabilities = self.softmax(output)
        
        return probabilities, attention_weights


class OutcomePredictor:
    """
    Advanced outcome prediction system for sports betting.
    
    Features:
    - LSTM/GRU models for sequence modeling
    - Team performance analysis
    - Feature engineering pipeline
    - GPU acceleration support
    - Model ensemble for improved accuracy
    """
    
    def __init__(
        self,
        model_type: str = "gru",  # "lstm" or "gru"
        use_gpu: bool = True,
        hidden_size: int = 128,
        num_layers: int = 2,
        sequence_length: int = 10,
        model_save_path: str = "models/sports_betting/outcome"
    ):
        """
        Initialize Outcome Predictor.
        
        Args:
            model_type: Type of model ("lstm" or "gru")
            use_gpu: Enable GPU acceleration
            hidden_size: Hidden layer size
            num_layers: Number of LSTM/GRU layers
            sequence_length: Input sequence length
            model_save_path: Path to save trained models
        """
        self.model_type = model_type
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        # Model paths
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        # Feature engineering parameters
        self.feature_columns = [
            'home_goals_scored_avg', 'home_goals_conceded_avg',
            'away_goals_scored_avg', 'away_goals_conceded_avg',
            'home_form_points', 'away_form_points',
            'home_win_streak', 'away_win_streak',
            'head_to_head_home_wins', 'head_to_head_away_wins',
            'home_injury_impact', 'away_injury_impact',
            'venue_advantage', 'weather_impact',
            'referee_home_bias', 'days_since_last_match_home',
            'days_since_last_match_away', 'league_position_home',
            'league_position_away', 'market_pressure'
        ]
        
        self.input_size = len(self.feature_columns)
        self.model = None
        self.scaler = None
        self.is_trained = False
        
        logger.info(f"Initialized OutcomePredictor with {model_type} model on {self.device}")
    
    def create_model(self, num_classes: int = 3) -> nn.Module:
        """Create neural network model."""
        if self.model_type == "lstm":
            model = LSTMOutcomeModel(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                num_classes=num_classes
            )
        elif self.model_type == "gru":
            model = GRUOutcomeModel(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                num_classes=num_classes
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return model.to(self.device)
    
    def engineer_features(
        self,
        home_team_stats: TeamStats,
        away_team_stats: TeamStats,
        match_context: Dict[str, Any]
    ) -> np.ndarray:
        """
        Engineer features for model input.
        
        Args:
            home_team_stats: Home team statistics
            away_team_stats: Away team statistics
            match_context: Match context (venue, weather, etc.)
            
        Returns:
            Engineered feature array
        """
        features = []
        
        # Team performance features
        features.extend([
            home_team_stats.goals_scored_avg,
            home_team_stats.goals_conceded_avg,
            away_team_stats.goals_scored_avg,
            away_team_stats.goals_conceded_avg
        ])
        
        # Form features
        home_form_points = sum([max(0, result) for result in home_team_stats.recent_form[-5:]])
        away_form_points = sum([max(0, result) for result in away_team_stats.recent_form[-5:]])
        
        features.extend([home_form_points, away_form_points])
        
        # Streak features
        home_win_streak = self._calculate_win_streak(home_team_stats.recent_form)
        away_win_streak = self._calculate_win_streak(away_team_stats.recent_form)
        
        features.extend([home_win_streak, away_win_streak])
        
        # Head-to-head features
        h2h_home_wins = home_team_stats.head_to_head_record.get('wins', 0)
        h2h_away_wins = away_team_stats.head_to_head_record.get('wins', 0)
        
        features.extend([h2h_home_wins, h2h_away_wins])
        
        # Injury impact
        home_injury_impact = len(home_team_stats.injury_list) * 0.1
        away_injury_impact = len(away_team_stats.injury_list) * 0.1
        
        features.extend([home_injury_impact, away_injury_impact])
        
        # Context features
        features.extend([
            match_context.get('venue_advantage', 0.1),
            match_context.get('weather_impact', 0.0),
            match_context.get('referee_home_bias', 0.0),
            match_context.get('days_since_last_match_home', 7),
            match_context.get('days_since_last_match_away', 7),
            match_context.get('league_position_home', 10),
            match_context.get('league_position_away', 10),
            match_context.get('market_pressure', 0.0)
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_win_streak(self, recent_form: List[int]) -> int:
        """Calculate current win streak."""
        streak = 0
        for result in reversed(recent_form):
            if result == 1:  # Win
                streak += 1
            else:
                break
        return streak
    
    def prepare_sequence_data(
        self,
        team_histories: List[Tuple[TeamStats, TeamStats, Dict[str, Any]]],
        outcomes: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare sequence data for training/prediction.
        
        Args:
            team_histories: List of (home_team, away_team, context) tuples
            outcomes: List of match outcomes (0=away_win, 1=draw, 2=home_win)
            
        Returns:
            Feature tensor and outcome tensor
        """
        sequences = []
        labels = []
        
        for i in range(len(team_histories) - self.sequence_length + 1):
            sequence_features = []
            
            for j in range(i, i + self.sequence_length):
                home_stats, away_stats, context = team_histories[j]
                features = self.engineer_features(home_stats, away_stats, context)
                sequence_features.append(features)
            
            sequences.append(sequence_features)
            labels.append(outcomes[i + self.sequence_length - 1])
        
        # Convert to tensors
        sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return sequences_tensor.to(self.device), labels_tensor.to(self.device)
    
    def train(
        self,
        training_data: List[Tuple[TeamStats, TeamStats, Dict[str, Any]]],
        outcomes: List[int],
        validation_data: Optional[Tuple] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10
    ) -> Dict[str, Any]:
        """
        Train the outcome prediction model.
        
        Args:
            training_data: List of (home_team, away_team, context) tuples
            outcomes: List of match outcomes
            validation_data: Optional validation data
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            early_stopping_patience: Early stopping patience
            
        Returns:
            Training metrics
        """
        logger.info("Starting outcome prediction model training...")
        
        # Prepare data
        X_train, y_train = self.prepare_sequence_data(training_data, outcomes)
        
        if validation_data:
            X_val, y_val = self.prepare_sequence_data(
                validation_data[0], validation_data[1]
            )
        
        # Create model
        num_classes = len(set(outcomes))
        self.model = self.create_model(num_classes)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Training history
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            epoch_train_loss = 0.0
            
            # Create batches
            num_batches = len(X_train) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                batch_X = X_train[start_idx:end_idx]
                batch_y = y_train[start_idx:end_idx]
                
                optimizer.zero_grad()
                
                # Forward pass
                if self.model_type == "gru":
                    outputs, _ = self.model(batch_X)
                else:
                    outputs = self.model(batch_X)
                
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Validation
            if validation_data:
                self.model.eval()
                with torch.no_grad():
                    if self.model_type == "gru":
                        val_outputs, _ = self.model(X_val)
                    else:
                        val_outputs = self.model(X_val)
                    
                    val_loss = criterion(val_outputs, y_val).item()
                    val_losses.append(val_loss)
                    
                    scheduler.step(val_loss)
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        self.save_model()
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}")
                if validation_data:
                    logger.info(f"Validation Loss: {val_loss:.4f}")
        
        self.is_trained = True
        
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "epochs_trained": epoch + 1,
            "model_type": self.model_type,
            "training_completed": datetime.now().isoformat()
        }
    
    def predict(
        self,
        home_team_stats: TeamStats,
        away_team_stats: TeamStats,
        match_context: Dict[str, Any],
        team_history: Optional[List[Tuple[TeamStats, TeamStats, Dict[str, Any]]]] = None
    ) -> OutcomePrediction:
        """
        Predict match outcome.
        
        Args:
            home_team_stats: Home team statistics
            away_team_stats: Away team statistics
            match_context: Match context
            team_history: Historical data for sequence prediction
            
        Returns:
            Outcome prediction
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        
        # Prepare features
        if team_history and len(team_history) >= self.sequence_length:
            # Use sequence prediction
            recent_history = team_history[-self.sequence_length:]
            recent_history.append((home_team_stats, away_team_stats, match_context))
            
            X, _ = self.prepare_sequence_data(recent_history, [0])
            input_tensor = X[-1:]  # Take last sequence
        else:
            # Use single prediction
            features = self.engineer_features(
                home_team_stats, away_team_stats, match_context
            )
            # Reshape for sequence input
            input_tensor = torch.tensor(
                features.reshape(1, 1, -1), dtype=torch.float32
            ).to(self.device)
        
        with torch.no_grad():
            if self.model_type == "gru":
                probabilities, attention_weights = self.model(input_tensor)
            else:
                probabilities = self.model(input_tensor)
            
            probs = probabilities.cpu().numpy()[0]
        
        # Calculate confidence as max probability minus entropy
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        max_entropy = np.log(len(probs))
        confidence = 1.0 - (entropy / max_entropy)
        
        return OutcomePrediction(
            home_win_prob=float(probs[2]) if len(probs) == 3 else float(probs[1]),
            draw_prob=float(probs[1]) if len(probs) == 3 else 0.0,
            away_win_prob=float(probs[0]),
            confidence=float(confidence),
            model_version=f"{self.model_type}_v1.0",
            prediction_date=datetime.now().isoformat(),
            features_used=self.feature_columns
        )
    
    def save_model(self, model_name: Optional[str] = None) -> str:
        """Save trained model."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        if model_name is None:
            model_name = f"outcome_predictor_{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model_path = self.model_save_path / f"{model_name}.pth"
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'sequence_length': self.sequence_length,
            'feature_columns': self.feature_columns,
            'training_date': datetime.now().isoformat()
        }, model_path)
        
        logger.info(f"Model saved to {model_path}")
        return str(model_path)
    
    def load_model(self, model_path: str) -> None:
        """Load trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Update parameters
        self.model_type = checkpoint['model_type']
        self.input_size = checkpoint['input_size']
        self.hidden_size = checkpoint['hidden_size']
        self.num_layers = checkpoint['num_layers']
        self.sequence_length = checkpoint['sequence_length']
        self.feature_columns = checkpoint['feature_columns']
        
        # Create and load model
        self.model = self.create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = True
        
        logger.info(f"Model loaded from {model_path}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance using gradient-based method.
        
        Returns:
            Feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        # This is a simplified implementation
        # In practice, you might use more sophisticated methods like SHAP
        importance_scores = {}
        
        # Create dummy input
        dummy_input = torch.randn(1, 1, self.input_size).to(self.device)
        dummy_input.requires_grad_(True)
        
        self.model.eval()
        
        if self.model_type == "gru":
            output, _ = self.model(dummy_input)
        else:
            output = self.model(dummy_input)
        
        # Calculate gradients
        for i, class_output in enumerate(output[0]):
            self.model.zero_grad()
            class_output.backward(retain_graph=True)
            
            gradients = dummy_input.grad.abs().mean().item()
            
            for j, feature_name in enumerate(self.feature_columns):
                if feature_name not in importance_scores:
                    importance_scores[feature_name] = 0.0
                importance_scores[feature_name] += gradients
        
        # Normalize
        total_importance = sum(importance_scores.values())
        if total_importance > 0:
            importance_scores = {
                k: v / total_importance for k, v in importance_scores.items()
            }
        
        return importance_scores