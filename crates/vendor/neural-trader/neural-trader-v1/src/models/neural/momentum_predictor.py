"""
Neural Momentum Predictor
Advanced neural network for momentum prediction with multi-modal inputs
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class MomentumNet(nn.Module):
    """
    Neural network for momentum prediction combining technical, sentiment, and market microstructure signals
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32]):
        super(MomentumNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Output layers
        layers.extend([
            nn.Linear(prev_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # [momentum_direction, strength, confidence]
        ])
        
        self.network = nn.Sequential(*layers)
        
        # Attention mechanism for feature importance (ensure input_dim is divisible by num_heads)
        self.num_heads = 4 if input_dim % 4 == 0 else (2 if input_dim % 2 == 0 else 1)
        self.attention = nn.MultiheadAttention(input_dim, num_heads=self.num_heads, batch_first=True)
        
    def forward(self, x):
        # Apply attention to input features
        attended_x, attention_weights = self.attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        attended_x = attended_x.squeeze(1)
        
        # Combine original and attended features
        combined_x = 0.7 * x + 0.3 * attended_x
        
        # Forward pass through main network
        output = self.network(combined_x)
        
        # Split output
        momentum_direction = torch.tanh(output[:, 0])  # -1 to 1
        strength = torch.sigmoid(output[:, 1])  # 0 to 1
        confidence = torch.sigmoid(output[:, 2])  # 0 to 1
        
        return momentum_direction, strength, confidence, attention_weights

class MomentumPredictor:
    """
    Complete momentum prediction system with training and inference capabilities
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model parameters
        self.input_dim = config.get('input_dim', 50)
        self.hidden_dims = config.get('hidden_dims', [128, 64, 32])
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 64)
        
        # Initialize model
        self.model = MomentumNet(self.input_dim, self.hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.scaler = StandardScaler()
        
        # Training state
        self.is_trained = False
        self.training_history = []
        
        # Feature extractors
        self.feature_extractors = self._initialize_feature_extractors()
        
        logger.info(f"MomentumPredictor initialized with device: {self.device}")
    
    def _initialize_feature_extractors(self) -> Dict[str, Any]:
        """Initialize feature extraction components"""
        return {
            'technical': TechnicalFeatureExtractor(),
            'sentiment': SentimentFeatureExtractor(),
            'microstructure': MicrostructureFeatureExtractor(),
            'cross_asset': CrossAssetFeatureExtractor()
        }
    
    async def predict(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Generate momentum prediction for a symbol
        """
        try:
            if not self.is_trained:
                await self._load_pretrained_model()
            
            # Extract features
            features = await self._extract_features(symbol, market_data)
            if features is None:
                return {'prediction': 0.0, 'confidence': 0.0}
            
            # Prepare input tensor
            feature_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                direction, strength, confidence, attention_weights = self.model(feature_tensor)
                
                prediction = direction.item() * strength.item()  # Scale direction by strength
                confidence_score = confidence.item()
            
            # Log prediction details
            logger.debug(f"Prediction for {symbol}: {prediction:.3f}, confidence: {confidence_score:.3f}")
            
            return {
                'prediction': prediction,
                'confidence': confidence_score,
                'direction': direction.item(),
                'strength': strength.item(),
                'attention_weights': attention_weights.cpu().numpy().tolist()
            }
            
        except Exception as e:
            logger.error(f"Error making prediction for {symbol}: {e}")
            return {'prediction': 0.0, 'confidence': 0.0}
    
    async def _extract_features(self, symbol: str, market_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract comprehensive feature vector"""
        try:
            feature_components = []
            
            # Technical features (20 dimensions)
            technical_features = self.feature_extractors['technical'].extract(market_data)
            feature_components.extend(technical_features)
            
            # Sentiment features (10 dimensions)
            sentiment_features = await self.feature_extractors['sentiment'].extract(symbol)
            feature_components.extend(sentiment_features)
            
            # Market microstructure features (10 dimensions)
            microstructure_features = self.feature_extractors['microstructure'].extract(market_data)
            feature_components.extend(microstructure_features)
            
            # Cross-asset features (10 dimensions)
            cross_asset_features = await self.feature_extractors['cross_asset'].extract(symbol)
            feature_components.extend(cross_asset_features)
            
            # Ensure we have the expected number of features
            if len(feature_components) != self.input_dim:
                # Pad or truncate to match expected input dimension
                if len(feature_components) < self.input_dim:
                    feature_components.extend([0.0] * (self.input_dim - len(feature_components)))
                else:
                    feature_components = feature_components[:self.input_dim]
            
            features = np.array(feature_components, dtype=np.float32)
            
            # Apply scaling if trained
            if self.is_trained:
                features = self.scaler.transform(features.reshape(1, -1)).flatten()
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features for {symbol}: {e}")
            return None
    
    async def train(self, training_data: List[Dict[str, Any]], epochs: int = 100) -> Dict[str, Any]:
        """
        Train the momentum prediction model
        """
        try:
            logger.info(f"Starting training with {len(training_data)} samples for {epochs} epochs")
            
            # Prepare training data
            X, y = await self._prepare_training_data(training_data)
            if X is None or y is None:
                raise ValueError("Failed to prepare training data")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            
            # Training loop
            self.model.train()
            training_losses = []
            
            for epoch in range(epochs):
                epoch_losses = []
                
                # Batch training
                for i in range(0, len(X_tensor), self.batch_size):
                    batch_X = X_tensor[i:i+self.batch_size]
                    batch_y = y_tensor[i:i+self.batch_size]
                    
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    direction_pred, strength_pred, confidence_pred, _ = self.model(batch_X)
                    
                    # Calculate losses for each output
                    direction_loss = self.criterion(direction_pred, batch_y[:, 0])
                    strength_loss = self.criterion(strength_pred, batch_y[:, 1])
                    confidence_loss = self.criterion(confidence_pred, batch_y[:, 2])
                    
                    total_loss = direction_loss + strength_loss + confidence_loss
                    
                    # Backward pass
                    total_loss.backward()
                    self.optimizer.step()
                    
                    epoch_losses.append(total_loss.item())
                
                avg_loss = np.mean(epoch_losses)
                training_losses.append(avg_loss)
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")
            
            self.is_trained = True
            self.training_history.append({
                'timestamp': datetime.now(),
                'epochs': epochs,
                'final_loss': training_losses[-1],
                'training_samples': len(training_data)
            })
            
            logger.info("Training completed successfully")
            
            return {
                'status': 'success',
                'final_loss': training_losses[-1],
                'training_losses': training_losses,
                'epochs_trained': epochs
            }
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def _prepare_training_data(self, training_data: List[Dict[str, Any]]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare features and labels for training"""
        try:
            X_list = []
            y_list = []
            
            for sample in training_data:
                # Extract features
                features = await self._extract_features(sample['symbol'], sample['market_data'])
                if features is None:
                    continue
                
                # Extract labels
                labels = [
                    sample.get('momentum_direction', 0.0),
                    sample.get('momentum_strength', 0.0),
                    sample.get('prediction_confidence', 0.5)
                ]
                
                X_list.append(features)
                y_list.append(labels)
            
            if not X_list:
                return None, None
            
            X = np.array(X_list)
            y = np.array(y_list)
            
            logger.info(f"Prepared {len(X)} training samples with {X.shape[1]} features")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None, None
    
    async def _load_pretrained_model(self):
        """Load pretrained model if available"""
        try:
            # Mock pretrained model loading - in practice, load from file
            logger.info("Loading pretrained momentum prediction model")
            self.is_trained = True
            
            # Initialize scaler with dummy data for now
            dummy_data = np.random.randn(100, self.input_dim)
            self.scaler.fit(dummy_data)
            
        except Exception as e:
            logger.warning(f"Could not load pretrained model: {e}")
            self.is_trained = False
    
    def save_model(self, filepath: str):
        """Save trained model"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scaler': self.scaler,
                'config': self.config,
                'training_history': self.training_history
            }, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scaler = checkpoint['scaler']
            self.training_history = checkpoint['training_history']
            self.is_trained = True
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")

class TechnicalFeatureExtractor:
    """Extract technical analysis features"""
    
    def extract(self, market_data: Dict[str, Any]) -> List[float]:
        """Extract technical features (20 dimensions)"""
        features = []
        
        # Price-based features
        price = market_data.get('price', 100)
        features.extend([
            market_data.get('price_change_pct', 0) / 100,
            market_data.get('price_velocity', 0),
            market_data.get('price_acceleration', 0),
            market_data.get('bollinger_position', 0.5)
        ])
        
        # Momentum indicators
        features.extend([
            (market_data.get('rsi', 50) - 50) / 50,  # Normalized RSI
            market_data.get('macd', 0),
            market_data.get('macd_signal', 0),
            market_data.get('momentum', 0)
        ])
        
        # Volume features
        volume = market_data.get('volume', 1000000)
        avg_volume = market_data.get('avg_volume', volume)
        features.extend([
            (volume - avg_volume) / avg_volume if avg_volume > 0 else 0,
            market_data.get('volume_trend', 0),
            market_data.get('vwap_position', 0),
            market_data.get('on_balance_volume', 0)
        ])
        
        # Volatility features
        features.extend([
            market_data.get('volatility', 0.2),
            market_data.get('atr', 0),
            market_data.get('volatility_trend', 0),
            market_data.get('garch_vol', 0.2)
        ])
        
        # Pattern recognition
        features.extend([
            market_data.get('breakout_signal', 0),
            market_data.get('support_resistance', 0),
            market_data.get('trend_strength', 0),
            market_data.get('pattern_score', 0)
        ])
        
        # Ensure we have exactly 20 features
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]

class SentimentFeatureExtractor:
    """Extract sentiment analysis features"""
    
    async def extract(self, symbol: str) -> List[float]:
        """Extract sentiment features (10 dimensions)"""
        # Mock sentiment extraction - in practice, use real sentiment analysis
        features = [
            0.3,   # overall_sentiment
            0.8,   # sentiment_confidence
            0.2,   # sentiment_trend
            0.1,   # news_volume
            0.0,   # social_sentiment
            0.4,   # analyst_sentiment
            0.0,   # earnings_sentiment
            0.1,   # sector_sentiment
            0.2,   # market_sentiment
            0.0    # event_sentiment
        ]
        
        return features

class MicrostructureFeatureExtractor:
    """Extract market microstructure features"""
    
    def extract(self, market_data: Dict[str, Any]) -> List[float]:
        """Extract microstructure features (10 dimensions)"""
        features = [
            market_data.get('bid_ask_spread', 0.01),
            market_data.get('order_imbalance', 0),
            market_data.get('trade_size_imbalance', 0),
            market_data.get('price_impact', 0),
            market_data.get('liquidity_score', 0.5),
            market_data.get('market_depth', 0.5),
            market_data.get('tick_direction', 0),
            market_data.get('volume_weighted_price', 0),
            market_data.get('realized_spread', 0.005),
            market_data.get('effective_spread', 0.008)
        ]
        
        return features

class CrossAssetFeatureExtractor:
    """Extract cross-asset correlation features"""
    
    async def extract(self, symbol: str) -> List[float]:
        """Extract cross-asset features (10 dimensions)"""
        # Mock cross-asset features - in practice, calculate real correlations
        features = [
            0.3,   # spy_correlation
            0.1,   # sector_correlation
            0.0,   # commodity_correlation
            0.2,   # currency_correlation
            0.1,   # bond_correlation
            0.0,   # crypto_correlation
            0.4,   # market_beta
            0.2,   # sector_beta
            0.1,   # momentum_factor
            0.3    # mean_reversion_factor
        ]
        
        return features