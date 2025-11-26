"""
Score Prediction System for Sports Betting.

This module implements Transformer-based models with attention mechanisms for
predicting exact scores, integrating player performance data, weather conditions,
and venue factors for comprehensive score forecasting.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
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
class ScorePrediction:
    """Score prediction result."""
    home_score_expected: float
    away_score_expected: float
    score_probabilities: Dict[str, float]  # e.g., {"0-0": 0.05, "1-0": 0.12, ...}
    total_goals_expected: float
    over_under_probabilities: Dict[str, float]  # e.g., {"over_2.5": 0.67, "under_2.5": 0.33}
    confidence: float
    model_version: str
    prediction_date: str
    factors_considered: List[str]


@dataclass
class PlayerPerformance:
    """Individual player performance data."""
    player_id: str
    player_name: str
    position: str
    goals_per_game: float
    assists_per_game: float
    minutes_played_avg: float
    form_rating: float  # 0-10 scale
    injury_status: str  # "fit", "doubt", "injured"
    suspension_status: bool
    historical_vs_opponent: Dict[str, Any]


@dataclass
class WeatherConditions:
    """Weather and environmental conditions."""
    temperature_celsius: float
    humidity_percentage: float
    wind_speed_kmh: float
    precipitation_chance: float
    visibility_km: float
    surface_condition: str  # "dry", "wet", "icy"


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        return context, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        context, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.W_o(context)
        return output, attention_weights


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and feed-forward."""
    
    def __init__(self, d_model: int, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Multi-head attention with residual connection
        attn_output, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights


class ScoreTransformerModel(nn.Module):
    """Transformer-based score prediction model."""
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 1024,
        max_score: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_score = max_score
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.home_score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, max_score + 1)  # 0 to max_score
        )
        
        self.away_score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, max_score + 1)  # 0 to max_score
        )
        
        # Total goals prediction
        self.total_goals_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # Continuous output
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Input projection and positional encoding
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through transformer layers
        attention_weights = []
        for layer in self.transformer_layers:
            x, attn_weights = layer(x)
            attention_weights.append(attn_weights)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Score predictions
        home_score_logits = self.home_score_head(x)
        away_score_logits = self.away_score_head(x)
        total_goals = self.total_goals_head(x).squeeze(-1)
        
        # Apply softmax for score probabilities
        home_score_probs = F.softmax(home_score_logits, dim=-1)
        away_score_probs = F.softmax(away_score_logits, dim=-1)
        
        return {
            'home_score_probs': home_score_probs,
            'away_score_probs': away_score_probs,
            'total_goals': total_goals,
            'attention_weights': attention_weights
        }


class ScorePredictor:
    """
    Advanced score prediction system using Transformer models.
    
    Features:
    - Transformer architecture with multi-head attention
    - Player performance integration
    - Weather and venue factor analysis
    - Exact score probability distributions
    - Over/under goal predictions
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        max_score: int = 10,
        sequence_length: int = 15,
        model_save_path: str = "models/sports_betting/score"
    ):
        """
        Initialize Score Predictor.
        
        Args:
            use_gpu: Enable GPU acceleration
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            max_score: Maximum score to predict
            sequence_length: Input sequence length
            model_save_path: Path to save trained models
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_score = max_score
        self.sequence_length = sequence_length
        
        # Model paths
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        # Feature engineering parameters
        self.team_features = [
            'attack_rating', 'defense_rating', 'midfield_rating',
            'goals_scored_last_5', 'goals_conceded_last_5',
            'shots_per_game', 'shots_on_target_per_game',
            'possession_percentage', 'pass_accuracy',
            'corners_per_game', 'fouls_per_game'
        ]
        
        self.player_features = [
            'key_player_1_rating', 'key_player_2_rating', 'key_player_3_rating',
            'striker_form', 'midfielder_form', 'defender_form', 'goalkeeper_form',
            'injury_impact_attack', 'injury_impact_defense',
            'suspension_impact'
        ]
        
        self.context_features = [
            'venue_advantage', 'weather_temperature', 'weather_wind',
            'weather_precipitation', 'surface_condition_rating',
            'referee_cards_per_game', 'referee_penalties_per_game',
            'rivalry_factor', 'importance_factor', 'fatigue_factor'
        ]
        
        self.feature_columns = (
            [f'home_{f}' for f in self.team_features] +
            [f'away_{f}' for f in self.team_features] +
            [f'home_{f}' for f in self.player_features] +
            [f'away_{f}' for f in self.player_features] +
            self.context_features
        )
        
        self.input_size = len(self.feature_columns)
        self.model = None
        self.is_trained = False
        
        logger.info(f"Initialized ScorePredictor with Transformer model on {self.device}")
    
    def create_model(self) -> nn.Module:
        """Create Transformer model."""
        model = ScoreTransformerModel(
            input_size=self.input_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            max_score=self.max_score
        )
        return model.to(self.device)
    
    def engineer_features(
        self,
        home_team_stats: Dict[str, Any],
        away_team_stats: Dict[str, Any],
        home_players: List[PlayerPerformance],
        away_players: List[PlayerPerformance],
        weather: WeatherConditions,
        match_context: Dict[str, Any]
    ) -> np.ndarray:
        """
        Engineer features for score prediction.
        
        Args:
            home_team_stats: Home team statistics
            away_team_stats: Away team statistics
            home_players: Home team player performances
            away_players: Away team player performances
            weather: Weather conditions
            match_context: Additional match context
            
        Returns:
            Engineered feature array
        """
        features = []
        
        # Team features (home)
        for feature in self.team_features:
            features.append(home_team_stats.get(feature, 0.0))
        
        # Team features (away)
        for feature in self.team_features:
            features.append(away_team_stats.get(feature, 0.0))
        
        # Player features (home)
        home_player_features = self._extract_player_features(home_players)
        features.extend(home_player_features)
        
        # Player features (away)
        away_player_features = self._extract_player_features(away_players)
        features.extend(away_player_features)
        
        # Context features
        context_features = [
            match_context.get('venue_advantage', 0.1),
            weather.temperature_celsius / 30.0,  # Normalize
            weather.wind_speed_kmh / 50.0,  # Normalize
            weather.precipitation_chance / 100.0,
            self._get_surface_rating(weather.surface_condition),
            match_context.get('referee_cards_per_game', 3.0),
            match_context.get('referee_penalties_per_game', 0.1),
            match_context.get('rivalry_factor', 0.0),
            match_context.get('importance_factor', 0.5),
            match_context.get('fatigue_factor', 0.0)
        ]
        
        features.extend(context_features)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_player_features(self, players: List[PlayerPerformance]) -> List[float]:
        """Extract aggregated player features."""
        # Sort players by importance (goals + assists)
        key_players = sorted(
            players,
            key=lambda p: p.goals_per_game + p.assists_per_game,
            reverse=True
        )[:3]
        
        # Get top 3 player ratings
        player_ratings = [p.form_rating for p in key_players[:3]]
        while len(player_ratings) < 3:
            player_ratings.append(5.0)  # Default rating
        
        # Position-specific form
        position_form = {'striker': 5.0, 'midfielder': 5.0, 'defender': 5.0, 'goalkeeper': 5.0}
        for player in players:
            if player.position.lower() in position_form:
                position_form[player.position.lower()] = max(
                    position_form[player.position.lower()], player.form_rating
                )
        
        # Injury impact
        injury_impact_attack = sum(
            1 for p in players 
            if p.injury_status != "fit" and p.position.lower() in ['striker', 'midfielder']
        ) * 0.2
        
        injury_impact_defense = sum(
            1 for p in players 
            if p.injury_status != "fit" and p.position.lower() in ['defender', 'goalkeeper']
        ) * 0.2
        
        # Suspension impact
        suspension_impact = sum(1 for p in players if p.suspension_status) * 0.15
        
        return (
            player_ratings +
            list(position_form.values()) +
            [injury_impact_attack, injury_impact_defense, suspension_impact]
        )
    
    def _get_surface_rating(self, surface_condition: str) -> float:
        """Get surface condition rating."""
        ratings = {"dry": 1.0, "wet": 0.7, "icy": 0.3}
        return ratings.get(surface_condition.lower(), 0.8)
    
    def prepare_sequence_data(
        self,
        match_histories: List[Tuple],
        home_scores: List[int],
        away_scores: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare sequence data for training/prediction.
        
        Args:
            match_histories: List of match feature tuples
            home_scores: List of home team scores
            away_scores: List of away team scores
            
        Returns:
            Feature tensor, home score tensor, away score tensor
        """
        sequences = []
        home_targets = []
        away_targets = []
        
        for i in range(len(match_histories) - self.sequence_length + 1):
            sequence_features = []
            
            for j in range(i, i + self.sequence_length):
                match_data = match_histories[j]
                features = self.engineer_features(*match_data)
                sequence_features.append(features)
            
            sequences.append(sequence_features)
            home_targets.append(home_scores[i + self.sequence_length - 1])
            away_targets.append(away_scores[i + self.sequence_length - 1])
        
        # Convert to tensors
        sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
        home_targets_tensor = torch.tensor(home_targets, dtype=torch.long)
        away_targets_tensor = torch.tensor(away_targets, dtype=torch.long)
        
        return (
            sequences_tensor.to(self.device),
            home_targets_tensor.to(self.device),
            away_targets_tensor.to(self.device)
        )
    
    def train(
        self,
        training_data: List[Tuple],
        home_scores: List[int],
        away_scores: List[int],
        validation_data: Optional[Tuple] = None,
        epochs: int = 100,
        batch_size: int = 16,
        learning_rate: float = 0.0001,
        early_stopping_patience: int = 15
    ) -> Dict[str, Any]:
        """
        Train the score prediction model.
        
        Args:
            training_data: List of match feature tuples
            home_scores: List of home team scores
            away_scores: List of away team scores
            validation_data: Optional validation data
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            early_stopping_patience: Early stopping patience
            
        Returns:
            Training metrics
        """
        logger.info("Starting score prediction model training...")
        
        # Prepare data
        X_train, y_home_train, y_away_train = self.prepare_sequence_data(
            training_data, home_scores, away_scores
        )
        
        if validation_data:
            X_val, y_home_val, y_away_val = self.prepare_sequence_data(
                validation_data[0], validation_data[1], validation_data[2]
            )
        
        # Create model
        self.model = self.create_model()
        
        # Setup training
        criterion_classification = nn.CrossEntropyLoss()
        criterion_regression = nn.MSELoss()
        
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
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
                batch_y_home = y_home_train[start_idx:end_idx]
                batch_y_away = y_away_train[start_idx:end_idx]
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X)
                
                # Calculate losses
                home_loss = criterion_classification(
                    outputs['home_score_probs'], batch_y_home
                )
                away_loss = criterion_classification(
                    outputs['away_score_probs'], batch_y_away
                )
                
                # Total goals targets
                total_goals_target = (batch_y_home + batch_y_away).float()
                total_goals_loss = criterion_regression(
                    outputs['total_goals'], total_goals_target
                )
                
                # Combined loss
                total_loss = home_loss + away_loss + 0.5 * total_goals_loss
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                epoch_train_loss += total_loss.item()
            
            avg_train_loss = epoch_train_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Validation
            if validation_data:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    
                    val_home_loss = criterion_classification(
                        val_outputs['home_score_probs'], y_home_val
                    )
                    val_away_loss = criterion_classification(
                        val_outputs['away_score_probs'], y_away_val
                    )
                    
                    val_total_goals_target = (y_home_val + y_away_val).float()
                    val_total_goals_loss = criterion_regression(
                        val_outputs['total_goals'], val_total_goals_target
                    )
                    
                    val_loss = (
                        val_home_loss + val_away_loss + 0.5 * val_total_goals_loss
                    ).item()
                    
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
            "model_type": "transformer",
            "training_completed": datetime.now().isoformat()
        }
    
    def predict(
        self,
        home_team_stats: Dict[str, Any],
        away_team_stats: Dict[str, Any],
        home_players: List[PlayerPerformance],
        away_players: List[PlayerPerformance],
        weather: WeatherConditions,
        match_context: Dict[str, Any],
        match_history: Optional[List[Tuple]] = None
    ) -> ScorePrediction:
        """
        Predict match score.
        
        Args:
            home_team_stats: Home team statistics
            away_team_stats: Away team statistics
            home_players: Home team player performances
            away_players: Away team player performances
            weather: Weather conditions
            match_context: Match context
            match_history: Historical data for sequence prediction
            
        Returns:
            Score prediction
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        
        # Prepare features
        match_data = (
            home_team_stats, away_team_stats,
            home_players, away_players,
            weather, match_context
        )
        
        if match_history and len(match_history) >= self.sequence_length:
            # Use sequence prediction
            recent_history = match_history[-self.sequence_length:]
            recent_history.append(match_data)
            
            X, _, _ = self.prepare_sequence_data(recent_history, [0], [0])
            input_tensor = X[-1:]  # Take last sequence
        else:
            # Use single prediction
            features = self.engineer_features(*match_data)
            # Reshape for sequence input
            input_tensor = torch.tensor(
                features.reshape(1, 1, -1), dtype=torch.float32
            ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            home_probs = outputs['home_score_probs'].cpu().numpy()[0]
            away_probs = outputs['away_score_probs'].cpu().numpy()[0]
            total_goals = outputs['total_goals'].cpu().numpy()[0]
        
        # Calculate expected scores
        home_expected = sum(i * prob for i, prob in enumerate(home_probs))
        away_expected = sum(i * prob for i, prob in enumerate(away_probs))
        
        # Generate score probabilities
        score_probabilities = {}
        for home_score in range(self.max_score + 1):
            for away_score in range(self.max_score + 1):
                prob = home_probs[home_score] * away_probs[away_score]
                if prob > 0.001:  # Only include probabilities > 0.1%
                    score_probabilities[f"{home_score}-{away_score}"] = float(prob)
        
        # Sort by probability
        score_probabilities = dict(
            sorted(score_probabilities.items(), key=lambda x: x[1], reverse=True)
        )
        
        # Over/under probabilities
        over_under_probs = {}
        for threshold in [0.5, 1.5, 2.5, 3.5, 4.5]:
            over_prob = 0.0
            for home_score in range(self.max_score + 1):
                for away_score in range(self.max_score + 1):
                    if home_score + away_score > threshold:
                        over_prob += home_probs[home_score] * away_probs[away_score]
            
            over_under_probs[f"over_{threshold}"] = float(over_prob)
            over_under_probs[f"under_{threshold}"] = float(1.0 - over_prob)
        
        # Calculate confidence
        max_score_prob = max(score_probabilities.values())
        entropy = -sum(p * np.log(p + 1e-8) for p in score_probabilities.values())
        max_entropy = np.log(len(score_probabilities))
        confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.5
        
        return ScorePrediction(
            home_score_expected=float(home_expected),
            away_score_expected=float(away_expected),
            score_probabilities=score_probabilities,
            total_goals_expected=float(total_goals),
            over_under_probabilities=over_under_probs,
            confidence=float(confidence),
            model_version="transformer_v1.0",
            prediction_date=datetime.now().isoformat(),
            factors_considered=[
                "team_performance", "player_form", "weather_conditions",
                "venue_factors", "historical_data", "injury_impact"
            ]
        )
    
    def save_model(self, model_name: Optional[str] = None) -> str:
        """Save trained model."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        if model_name is None:
            model_name = f"score_predictor_transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model_path = self.model_save_path / f"{model_name}.pth"
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': self.input_size,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'max_score': self.max_score,
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
        self.input_size = checkpoint['input_size']
        self.d_model = checkpoint['d_model']
        self.num_heads = checkpoint['num_heads']
        self.num_layers = checkpoint['num_layers']
        self.max_score = checkpoint['max_score']
        self.sequence_length = checkpoint['sequence_length']
        self.feature_columns = checkpoint['feature_columns']
        
        # Create and load model
        self.model = self.create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = True
        
        logger.info(f"Model loaded from {model_path}")
    
    def get_attention_weights(
        self,
        home_team_stats: Dict[str, Any],
        away_team_stats: Dict[str, Any],
        home_players: List[PlayerPerformance],
        away_players: List[PlayerPerformance],
        weather: WeatherConditions,
        match_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get attention weights for interpretability.
        
        Returns:
            Attention weights for each layer and head
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get attention weights")
        
        # Prepare input
        match_data = (
            home_team_stats, away_team_stats,
            home_players, away_players,
            weather, match_context
        )
        
        features = self.engineer_features(*match_data)
        input_tensor = torch.tensor(
            features.reshape(1, 1, -1), dtype=torch.float32
        ).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            attention_weights = outputs['attention_weights']
        
        # Process attention weights
        processed_weights = {}
        for layer_idx, layer_weights in enumerate(attention_weights):
            layer_weights_np = layer_weights.cpu().numpy()[0]  # Remove batch dimension
            processed_weights[f'layer_{layer_idx}'] = {
                f'head_{head_idx}': head_weights.tolist()
                for head_idx, head_weights in enumerate(layer_weights_np)
            }
        
        return processed_weights