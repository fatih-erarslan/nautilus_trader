#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 11:23:16 2025

@author: ashina
"""

"""
Enhanced Q* Learning Implementation

This module implements an advanced implementation of the Q* learning algorithm,
expanding on the concepts allegedly leaked from OpenAI. It features dynamic exploration,
self-refinement capabilities, adaptive state-action spaces, quantum-inspired enhancements,
and comprehensive environment interactions.

Key improvements:
- Environment integration with custom and gym environments
- Quantum-inspired state representation
- Adaptive learning rates
- Experience replay buffer
- Prioritized sampling
- Advanced convergence metrics
- Model saving/loading capabilities
- Detailed performance visualization
- Automated hyperparameter optimization
"""
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import random
import matplotlib.pyplot as plt
import pickle
import datetime
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from collections import deque
import logging

# --- Path Setup (Assuming this script remains in user_data/strategies/QStar) ---
SCRIPT_DIR = Path(__file__).resolve().parent
STRATEGY_DIR = SCRIPT_DIR
USER_DATA_DIR = SCRIPT_DIR.parent.parent
STRATEGIES_PARENT_DIR = STRATEGY_DIR.parent
if str(STRATEGIES_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGIES_PARENT_DIR))
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))
# --- End Path Setup ---

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger("Q*LearningDemo")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Q*Learning")

try:
    import gym
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    logger.warning("Gym not available, some environment features will be limited")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logger.warning("TQDM not available, progress bars will be disabled")

try:
    from qstar_river import TradingEnvironment, TradingAction, MarketState
    from river_ml import RiverOnlineML, RIVER_AVAILABLE
except ImportError as e:
    logger.error(f"Failed to import Q* River components: {e}", exc_info=True)
    sys.exit(1)
# --- End Imports ---

class ExperienceBuffer:
    """Memory buffer for experience replay with prioritized sampling."""

    def __init__(self, max_size: int = 10000, alpha: float = 0.6):
        """
        Initialize experience buffer.

        Args:
            max_size: Maximum buffer size
            alpha: Priority exponent (0 = uniform sampling)
        """
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.alpha = alpha
        self.epsilon = 1e-6  # Small constant to avoid zero priority

    def add(self, experience: Tuple, error: float = None):
        """
        Add experience to buffer.

        Args:
            experience: (state, action, reward, next_state) tuple
            error: TD error for prioritization (if None, max priority is used)
        """
        self.buffer.append(experience)

        # Use max priority for new experiences or provided error
        if error is None:
            priority = max(self.priorities) if self.priorities else 1.0
        else:
            priority = abs(error) + self.epsilon

        self.priorities.append(priority)

    def sample(self, batch_size: int) -> Tuple[List[Tuple], List[int], np.ndarray]:
        """
        Sample batch from buffer with prioritization.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple containing experiences, indices, and importance sampling weights
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        # Convert priorities to probabilities
        priorities = np.array(self.priorities)**self.alpha
        probabilities = priorities / np.sum(priorities)

        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)

        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices])**(-0.4)
        weights /= np.max(weights)  # Normalize weights

        # Get experiences
        experiences = [self.buffer[i] for i in indices]

        return experiences, indices, weights

    def update_priorities(self, indices: List[int], errors: List[float]):
        """
        Update priorities for experiences.

        Args:
            indices: Indices of experiences to update
            errors: TD errors for prioritization
        """
        for i, error in zip(indices, errors):
            self.priorities[i] = abs(error) + self.epsilon

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)


class QLearningMetrics:
    """Tracks and analyzes Q* learning performance metrics."""

    def __init__(self):
        """Initialize metrics tracking."""
        self.episode_rewards = []
        self.q_value_changes = []
        self.exploration_rates = []
        self.learning_rates = []
        self.convergence_measures = []
        self.episode_lengths = []
        self.refinement_points = []

    def add_episode_data(self, reward: float, length: int, q_change: float,
                         exploration_rate: float, learning_rate: float,
                         convergence: float):
        """Add metrics from a completed episode."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.q_value_changes.append(q_change)
        self.exploration_rates.append(exploration_rate)
        self.learning_rates.append(learning_rate)
        self.convergence_measures.append(convergence)

    def add_refinement_point(self, episode: int):
        """Mark an episode where refinement occurred."""
        self.refinement_points.append(episode)

    def plot_metrics(self, save_path: Optional[str] = None):
        """
        Plot performance metrics.

        Args:
            save_path: Path to save the figure or None to display
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))

        # Rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')

        # Episode lengths
        axes[0, 1].plot(self.episode_lengths)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')

        # Q-value changes
        axes[1, 0].plot(self.q_value_changes)
        axes[1, 0].set_title('Q-Value Changes')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Average Change')

        # Exploration rate
        axes[1, 1].plot(self.exploration_rates)
        axes[1, 1].set_title('Exploration Rate')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Rate')

        # Learning rate
        axes[2, 0].plot(self.learning_rates)
        axes[2, 0].set_title('Learning Rate')
        axes[2, 0].set_xlabel('Episode')
        axes[2, 0].set_ylabel('Rate')

        # Convergence
        axes[2, 1].plot(self.convergence_measures)
        axes[2, 1].set_title('Convergence Measure')
        axes[2, 1].set_xlabel('Episode')
        axes[2, 1].set_ylabel('Value')

        # Mark refinement points on all plots
        for i in range(3):
            for j in range(2):
                for point in self.refinement_points:
                    if point < len(self.episode_rewards):
                        axes[i, j].axvline(x=point, color='r', linestyle='--', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics of metrics."""
        if not self.episode_rewards:
            return {}

        return {
            "avg_reward": np.mean(self.episode_rewards[-100:]),
            "max_reward": np.max(self.episode_rewards),
            "min_reward": np.min(self.episode_rewards),
            "avg_episode_length": np.mean(self.episode_lengths[-100:]),
            "final_convergence": self.convergence_measures[-1] if self.convergence_measures else None,
            "final_exploration_rate": self.exploration_rates[-1] if self.exploration_rates else None,
            "total_episodes": len(self.episode_rewards)
        }


class EnvironmentWrapper:
    """Wrapper for environments to standardize interaction with Q* agent."""

    def __init__(self, environment=None, gym_env_name: Optional[str] = None,
                 state_mapper: Callable = None, reward_shaper: Callable = None):
        """
        Initialize environment wrapper.

        Args:
            environment: Custom environment object or None if using gym
            gym_env_name: Name of gym environment if using gym
            state_mapper: Function to map environment state to agent state index
            reward_shaper: Function to reshape rewards
        """
        self.env = None
        self.gym_env = None
        self.state_mapper = state_mapper
        self.reward_shaper = reward_shaper
        self.num_states = 0
        self.num_actions = 0

        if environment is not None:
            # Custom environment
            self.env = environment
            self.num_states = getattr(environment, 'num_states', 30)
            self.num_actions = getattr(environment, 'num_actions', 4)
        elif gym_env_name and GYM_AVAILABLE:
            # Gym environment
            self.gym_env = gym.make(gym_env_name)
            self.num_states = self.gym_env.observation_space.n if hasattr(self.gym_env.observation_space, 'n') else 100
            self.num_actions = self.gym_env.action_space.n
        else:
            # Default dummy environment
            self.num_states = 30
            self.num_actions = 4

    def reset(self) -> int:
        """
        Reset environment and return initial state.

        Returns:
            Initial state as an integer index
        """
        if self.gym_env:
            obs = self.gym_env.reset()
            return self._process_state(obs)
        elif self.env and hasattr(self.env, 'reset'):
            return self.env.reset()
        else:
            return random.randint(0, self.num_states - 1)

    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        Take action in environment. Handles modern Gym API (5 return values).

        Args:
            action: Action to take

        Returns:
            Tuple of (next_state, reward, done)
        """
        if self.gym_env:
            # --- FIX: Unpack 5 values from modern Gym/Gymnasium step ---
            try:
                # Standard new signature
                observation, reward, terminated, truncated, info = self.gym_env.step(action)
                # Combine terminated and truncated to get the 'done' status
                done = terminated or truncated
            except ValueError:
                # Fallback for potentially older gym versions (less likely given the error)
                logger.warning("Gym step returned unexpected number of values, trying older signature.")
                observation, reward, done, info = self.gym_env.step(action) # Old signature

            next_state = self._process_state(observation)

            # Apply reward shaping if provided
            if self.reward_shaper:
                # Pass necessary info if the shaper needs it
                reward = self.reward_shaper(reward=reward, next_state=next_state, done=done)

            return next_state, reward, done

        elif self.env and hasattr(self.env, 'step'):
            # Handle custom environment step
            return self.env.step(action)

        else:
            # Simple dummy environment - reach the highest state
            next_state = random.randint(0, self.num_states - 1)
            reward = 1.0 if next_state == self.num_states - 1 else 0.0
            done = next_state == self.num_states - 1
            return next_state, reward, done

    def _process_state(self, observation) -> int:
        """
        Process observation into state index.

        Args:
            observation: Raw environment observation

        Returns:
            State index
        """
        if self.state_mapper:
            return self.state_mapper(observation)

        # Default behavior depends on observation type
        if isinstance(observation, (int, np.integer)):
            return observation
        elif hasattr(observation, 'shape') and len(observation.shape) > 0:
            # Convert continuous observation to discrete state
            return hash(str(observation.flatten().round(1))) % self.num_states
        else:
            return hash(str(observation)) % self.num_states

    def render(self):
        """Render environment if supported."""
        if self.gym_env:
            self.gym_env.render()
        elif self.env and hasattr(self.env, 'render'):
            self.env.render()

    def close(self):
        """Close environment."""
        if self.gym_env:
            self.gym_env.close()
        elif self.env and hasattr(self.env, 'close'):
            self.env.close()


class SophisticatedQLearningAgent:
    """
    Advanced Q* Learning Agent with enhanced capabilities.

    Features:
    - Dynamic exploration strategy with decay
    - Experience replay with prioritized sampling
    - Adaptive learning rates
    - Quantum-inspired state representations
    - Convergence detection and monitoring
    - Automatic refinement of the state-action space
    - Comprehensive metrics tracking
    """

    def __init__(self, states: int, actions: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 exploration_rate: float = 1.0,
                 min_exploration_rate: float = 0.01,
                 exploration_decay_rate: float = 0.995,
                 use_adaptive_learning_rate: bool = True,
                 use_experience_replay: bool = True,
                 experience_buffer_size: int = 10000,
                 batch_size: int = 32,
                 max_episodes: int = 10000,
                 max_steps_per_episode: int = 200,
                 use_quantum_representation: bool = False):
        """
        Initialize the Q* learning agent.

        Args:
            states: Number of states in the environment
            actions: Number of possible actions
            learning_rate: Initial learning rate
            discount_factor: Discount factor for future rewards
            exploration_rate: Initial exploration rate
            min_exploration_rate: Minimum exploration rate
            exploration_decay_rate: Decay rate for exploration
            use_adaptive_learning_rate: Whether to use adaptive learning rate
            use_experience_replay: Whether to use experience replay
            experience_buffer_size: Size of experience replay buffer
            batch_size: Batch size for experience replay
            max_episodes: Maximum number of training episodes
            max_steps_per_episode: Maximum steps per episode
            use_quantum_representation: Whether to use quantum-inspired representation
        """
        self.states = states
        self.actions = actions
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate

        # Advanced features
        self.use_adaptive_learning_rate = use_adaptive_learning_rate
        self.use_experience_replay = use_experience_replay
        self.batch_size = batch_size
        self.use_quantum_representation = use_quantum_representation

        # Training parameters
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode

        # Initialize Q-table
        self.q_table = np.zeros((states, actions))

        # Initialize experience replay buffer if enabled
        self.experience_buffer = ExperienceBuffer(max_size=experience_buffer_size) if use_experience_replay else None

        # Initialize metrics tracking
        self.metrics = QLearningMetrics()

        # State for adaptive learning
        self.avg_q_change = 0.0
        self.episode_rewards = []
        self.convergence_history = []

        # Quantum representation (if enabled)
        self.quantum_phases = np.random.uniform(0, 2*np.pi, (states, actions)) if use_quantum_representation else None

        logger.info(f"Initialized Q* Learning Agent with {states} states and {actions} actions")
        logger.info(f"Advanced features: adaptive_lr={use_adaptive_learning_rate}, "
                   f"experience_replay={use_experience_replay}, "
                   f"quantum_representation={use_quantum_representation}")

    def choose_action(self, state: int) -> int:
        """
        Choose an action based on the current state with exploration-exploitation balance.

        Args:
            state: Current state index

        Returns:
            Selected action index
        """
        # Ensure state is within bounds
        state = self._validate_state(state)

        # Exploration-exploitation tradeoff
        if random.uniform(0, 1) < self.exploration_rate:
            # Exploration: choose random action
            return random.randint(0, self.actions - 1)
        else:
            # Exploitation: choose best known action
            if self.use_quantum_representation:
                # Quantum-inspired action selection
                return self._quantum_action_selection(state)
            else:
                # Classical action selection: highest Q-value
                return np.argmax(self.q_table[state, :])

    def _quantum_action_selection(self, state: int) -> int:
        """
        Select action using quantum-inspired probabilistic approach.

        Args:
            state: Current state

        Returns:
            Selected action
        """
        # Calculate probabilities using quantum phases
        q_values = self.q_table[state, :]

        # Normalize q-values to avoid numerical issues
        q_values = q_values - np.min(q_values) if np.min(q_values) < 0 else q_values

        if np.sum(q_values) == 0:
            # If all Q-values are zero, use uniform distribution
            probs = np.ones(self.actions) / self.actions
        else:
            # Apply quantum phase interference
            amplitudes = np.sqrt(q_values / np.sum(q_values))
            phases = self.quantum_phases[state, :]

            # Calculate complex amplitudes
            complex_amplitudes = amplitudes * np.exp(1j * phases)

            # Calculate probabilities from interference
            probs = np.abs(complex_amplitudes)**2
            probs = probs / np.sum(probs)  # Normalize

        # Sample action based on probabilities
        return np.random.choice(self.actions, p=probs)

    def learn(self, state: int, action: int, reward: float, next_state: int) -> float:
        """
        Update the Q-table based on experience.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: New state

        Returns:
            Q-value change magnitude
        """
        # Ensure states are within bounds
        state = self._validate_state(state)
        next_state = self._validate_state(next_state)

        # Store experience if using replay buffer
        if self.use_experience_replay:
            self.experience_buffer.add((state, action, reward, next_state))

        # Calculate current prediction and target Q-value
        predict = self.q_table[state, action]
        target = reward + self.discount_factor * np.max(self.q_table[next_state, :])

        # Update Q-value with current learning rate
        self.q_table[state, action] += self.learning_rate * (target - predict)

        # Return magnitude of update for monitoring
        return abs(target - predict)

    def replay_experiences(self) -> float:
        """
        Learn from stored experiences using prioritized replay.

        Returns:
            Average Q-value change magnitude
        """
        if not self.use_experience_replay or len(self.experience_buffer) < self.batch_size:
            return 0.0

        # Sample batch of experiences with priorities
        experiences, indices, weights = self.experience_buffer.sample(self.batch_size)

        total_change = 0.0
        td_errors = []

        # Learn from each experience in batch
        for i, (state, action, reward, next_state) in enumerate(experiences):
            # Calculate current prediction and target
            predict = self.q_table[state, action]
            target = reward + self.discount_factor * np.max(self.q_table[next_state, :])

            # Get TD error for prioritization
            td_error = target - predict
            td_errors.append(td_error)

            # Apply importance sampling weight to update
            weighted_update = self.learning_rate * td_error * weights[i]
            self.q_table[state, action] += weighted_update

            total_change += abs(td_error)

        # Update priorities based on new TD errors
        self.experience_buffer.update_priorities(indices, td_errors)

        return total_change / self.batch_size

    def update_exploration_rate(self) -> None:
        """Decrease exploration rate according to decay schedule."""
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay_rate
        )

    def update_learning_rate(self, episode: int, avg_q_change: float) -> None:
        """
        Update learning rate adaptively based on progress.

        Args:
            episode: Current episode number
            avg_q_change: Average Q-value change magnitude
        """
        if not self.use_adaptive_learning_rate:
            return

        # Adaptive learning rate strategies
        if avg_q_change < 0.01:
            # Small Q-value changes might indicate convergence or getting stuck
            # Increase learning rate slightly to escape local minima
            self.learning_rate = min(0.5, self.learning_rate * 1.05)
        elif avg_q_change > 0.5:
            # Large Q-value changes might indicate instability
            # Decrease learning rate to stabilize learning
            self.learning_rate = max(0.01, self.learning_rate * 0.95)
        else:
            # Schedule-based decay
            self.learning_rate = self.initial_learning_rate * (1.0 - episode / self.max_episodes)

    def has_converged(self, threshold: float = 0.005, window_size: int = 100) -> Tuple[bool, float]:
        """
        Check if the Q-values have converged.

        Args:
            threshold: Convergence threshold for Q-value changes
            window_size: Window size for convergence check

        Returns:
            Tuple of (has_converged, convergence_measure)
        """
        # Check if we have enough history
        if len(self.convergence_history) < window_size:
            return False, 1.0

        # Calculate average change over recent episodes
        recent_changes = np.mean(self.convergence_history[-window_size:])

        # Compare to threshold
        converged = recent_changes < threshold

        return converged, recent_changes

    def _validate_state(self, state: int) -> int:
        """
        Ensure state is within bounds of Q-table.

        Args:
            state: State index

        Returns:
            Valid state index
        """
        return max(0, min(state, self.states - 1))

    def resize_q_table(self, new_states: int, new_actions: int) -> None:
            """
            Resize the Q-table and quantum phases array for expanded state/action spaces.

            Args:
                new_states: New number of states
                new_actions: New number of actions
            """
            # --- FIX: Store old dimensions BEFORE updating self attributes ---
            old_states = self.states
            old_actions = self.actions

            if new_states <= old_states and new_actions <= old_actions:
                logger.warning(f"Skipping resize: New dimensions ({new_states}, {new_actions}) not larger than current ({old_states}, {old_actions})")
                return

            logger.info(f"Resizing Q-table from ({old_states}, {old_actions}) to ({new_states}, {new_actions})")

            # --- Resize Q-table ---
            new_q_table = np.zeros((new_states, new_actions))
            # Use OLD dimensions for slicing during copy
            new_q_table[:old_states, :old_actions] = self.q_table
            self.q_table = new_q_table # Assign the resized table

            # --- Resize Quantum Phases (if used) ---
            if self.use_quantum_representation:
                new_phases = np.random.uniform(0, 2*np.pi, (new_states, new_actions))
                # Use OLD dimensions for slicing during copy
                if self.quantum_phases is not None: # Check if it exists (e.g., if loaded model didn't have it)
                     new_phases[:old_states, :old_actions] = self.quantum_phases
                self.quantum_phases = new_phases # Assign the resized phases

            # --- FIX: Update state/action counts AFTER copying is done ---
            self.states = new_states
            self.actions = new_actions

            logger.info(f"Q-table resized to {self.q_table.shape}")
            if self.use_quantum_representation:
                logger.info(f"Quantum phases resized to {self.quantum_phases.shape}")

    def train(self, environment: EnvironmentWrapper = None) -> Tuple[bool, int]:
        """
        Train the agent in the provided environment.

        Args:
            environment: Environment wrapper or None for random environment

        Returns:
            Tuple of (converged, episodes_used)
        """
        if environment is None:
            environment = EnvironmentWrapper()

        logger.info(f"Starting training for up to {self.max_episodes} episodes")

        # Store initial settings for reference
        initial_states = self.states
        initial_actions = self.actions

        # Initialize metrics
        episodes_completed = 0
        total_q_changes = []

        # Initialize progress bar if available
        episodes_range = tqdm(range(self.max_episodes)) if TQDM_AVAILABLE else range(self.max_episodes)

        for episode in episodes_range:
            # Reset environment for new episode
            state = environment.reset()
            episode_reward = 0
            episode_steps = 0
            episode_q_changes = []

            # Reset quantum phases slightly for this episode (if using quantum representation)
            if self.use_quantum_representation and episode % 10 == 0:
                self.quantum_phases += np.random.uniform(-0.1, 0.1, (self.states, self.actions))

            # Episode loop
            for step in range(self.max_steps_per_episode):
                # Choose and take action
                action = self.choose_action(state)
                next_state, reward, done = environment.step(action)

                # Learn from this experience
                q_change = self.learn(state, action, reward, next_state)
                episode_q_changes.append(q_change)

                # Accumulate reward
                episode_reward += reward
                episode_steps += 1

                # Move to next state
                state = next_state

                # Check if episode is done
                if done:
                    break

            # After episode actions

            # Learn from experience replay
            if self.use_experience_replay and len(self.experience_buffer) >= self.batch_size:
                replay_q_change = self.replay_experiences()
                episode_q_changes.append(replay_q_change)

            # Update rates
            self.update_exploration_rate()

            # Calculate average Q-change for this episode
            avg_q_change = np.mean(episode_q_changes) if episode_q_changes else 0
            self.convergence_history.append(avg_q_change)

            # Update learning rate adaptively
            self.update_learning_rate(episode, avg_q_change)

            # Store metrics
            self.episode_rewards.append(episode_reward)
            total_q_changes.append(avg_q_change)

            # Record metrics
            converged, convergence_value = self.has_converged()
            self.metrics.add_episode_data(
                reward=episode_reward,
                length=episode_steps,
                q_change=avg_q_change,
                exploration_rate=self.exploration_rate,
                learning_rate=self.learning_rate,
                convergence=convergence_value
            )

            # Check for convergence
            if converged:
                logger.info(f"Converged after {episode+1} episodes")
                episodes_completed = episode + 1
                break

            # Update progress bar description if available
            if TQDM_AVAILABLE:
                episodes_range.set_description(
                    f"Reward: {episode_reward:.2f}, Conv: {convergence_value:.4f}, Expl: {self.exploration_rate:.4f}"
                )

        if episodes_completed == 0:
            episodes_completed = self.max_episodes
            logger.info(f"Maximum episodes ({self.max_episodes}) reached without convergence")

        # Log training summary
        logger.info(f"Training completed: {episodes_completed}/{self.max_episodes} episodes")
        logger.info(f"Final metrics: ")
        for key, value in self.metrics.get_summary().items():
            logger.info(f"  {key}: {value}")
        logger.info(f"Agent size: {self.states} states, {self.actions} actions")

        # Return convergence status and episodes used
        return converged, episodes_completed

    def save(self, filepath: str) -> None:
        """
        Save the agent to a file.

        Args:
            filepath: Path to save the agent
        """
        agent_data = {
            'q_table': self.q_table,
            'states': self.states,
            'actions': self.actions,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'exploration_rate': self.exploration_rate,
            'min_exploration_rate': self.min_exploration_rate,
            'exploration_decay_rate': self.exploration_decay_rate,
            'use_adaptive_learning_rate': self.use_adaptive_learning_rate,
            'use_experience_replay': self.use_experience_replay,
            'use_quantum_representation': self.use_quantum_representation,
            'quantum_phases': self.quantum_phases,
            'metrics': self.metrics.__dict__,
            'timestamp': datetime.datetime.now().isoformat()
        }

        with open(filepath, 'wb') as f:
            pickle.dump(agent_data, f)

        logger.info(f"Agent saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'SophisticatedQLearningAgent':
        """
        Load an agent from a file.

        Args:
            filepath: Path to load the agent from

        Returns:
            Loaded agent
        """
        with open(filepath, 'rb') as f:
            agent_data = pickle.load(f)

        agent = cls(
            states=agent_data['states'],
            actions=agent_data['actions'],
            learning_rate=agent_data['learning_rate'],
            discount_factor=agent_data['discount_factor'],
            exploration_rate=agent_data['exploration_rate'],
            min_exploration_rate=agent_data['min_exploration_rate'],
            exploration_decay_rate=agent_data['exploration_decay_rate'],
            use_adaptive_learning_rate=agent_data['use_adaptive_learning_rate'],
            use_experience_replay=agent_data['use_experience_replay'],
            use_quantum_representation=agent_data['use_quantum_representation']
        )

        # Restore Q-table
        agent.q_table = agent_data['q_table']

        # Restore quantum phases if using quantum representation
        if agent.use_quantum_representation and 'quantum_phases' in agent_data:
            agent.quantum_phases = agent_data['quantum_phases']

        # Restore metrics if available
        if 'metrics' in agent_data:
            for key, value in agent_data['metrics'].items():
                setattr(agent.metrics, key, value)

        logger.info(f"Agent loaded from {filepath}")
        return agent

    def evaluate(self, environment: EnvironmentWrapper, num_episodes: int = 100,
                render: bool = False) -> Dict[str, float]:
        """
        Evaluate the agent in the environment without learning.

        Args:
            environment: Environment to evaluate in
            num_episodes: Number of evaluation episodes
            render: Whether to render the environment

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating agent for {num_episodes} episodes")

        rewards = []
        episode_lengths = []
        success_count = 0

        # No exploration during evaluation
        original_exploration_rate = self.exploration_rate
        self.exploration_rate = 0.0

        for episode in range(num_episodes):
            state = environment.reset()
            episode_reward = 0
            episode_steps = 0
            episode_success = False

            for step in range(self.max_steps_per_episode):
                # Render if requested
                if render:
                    environment.render()

                # Choose and take action (no learning)
                action = self.choose_action(state)
                next_state, reward, done = environment.step(action)

                # Accumulate reward
                episode_reward += reward
                episode_steps += 1

                # Move to next state
                state = next_state

                # Check if episode is done
                if done:
                    if reward > 0:  # Assuming positive reward means success
                        episode_success = True
                    break

            # Record episode results
            rewards.append(episode_reward)
            episode_lengths.append(episode_steps)
            if episode_success:
                success_count += 1

        # Restore exploration rate
        self.exploration_rate = original_exploration_rate

        # Calculate evaluation metrics
        eval_metrics = {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_episode_length": np.mean(episode_lengths),
            "success_rate": success_count / num_episodes
        }

        logger.info(f"Evaluation results: ")
        for key, value in eval_metrics.items():
            logger.info(f"  {key}: {value}")

        return eval_metrics


# --- Configuration for Market Data ---
PAIR = "BTC/USDT"
TIMEFRAME = "6h"
EXCHANGE_NAME = "binance"
pair_file_base = PAIR.replace('/', '_')
HISTORICAL_DATA_FILE = USER_DATA_DIR / "data" / EXCHANGE_NAME / f"{pair_file_base}-{TIMEFRAME}.feather"
MODELS_DIR = USER_DATA_DIR / "models"
# --- Save the AGENT state using a specific name ---
AGENT_SAVE_FILENAME = f"qstar_agent_{pair_file_base}_{TIMEFRAME}.pkl"
AGENT_SAVE_PATH = MODELS_DIR / AGENT_SAVE_FILENAME
# --- End Configuration ---

# --- Reusable Data Loader (from train script) ---
def load_freqtrade_data(filepath: Path) -> pd.DataFrame:
    """Loads data downloaded via `freqtrade download-data` (Feather format)."""
    logger.info(f"Attempting to load data from: {filepath}")
    if not filepath.is_file():
        logger.error(f"Data file not found: {filepath}")
        logger.error(f"Please download data using: freqtrade download-data --exchange {EXCHANGE_NAME} --pairs {PAIR} --timeframes {TIMEFRAME}")
        return pd.DataFrame() # Return empty instead of exiting
    try:
        df = pd.read_feather(filepath)
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
             missing = [c for c in required_cols if c not in df.columns]
             logger.error(f"Feather file {filepath} missing required columns: {missing}")
             return pd.DataFrame()
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
             df['date'] = pd.to_datetime(df['date'], utc=True)
        df.set_index('date', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['close'], inplace=True)
        df.sort_index(inplace=True)
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Error loading feather data file {filepath}: {e}", exc_info=True)
        return pd.DataFrame()

# --- Function to calculate indicators (Needs QStar instance) ---
# We need a way to run populate_indicators outside the main script
# Option 1: Keep it simple, assume data has indicators (e.g., from offline save)
# Option 2: Instantiate QStar here just for indicators (like training script)

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder: Calculates indicators needed by the TradingEnvironment.
    Ideally, this uses the same logic as QStar.populate_indicators.
    For demonstration, we'll use a minimal set or the full QStar logic.
    """
    logger.info("Calculating indicators for training/demonstration...")
    try:
        # --- Use QStar instance to calculate indicators ---
        strategy_config = {
            'strategy': "QStar", 'stake_currency': 'USDT', 'stake_amount': 1000,
            'exchange': {'name': 'debug', 'pair_whitelist': [PAIR]},
            'user_data_dir': str(USER_DATA_DIR), 'runmode': 'other',
            '_train_script_mode': True, # Prevent state loading
            'api_key': None, 'llm_provider': 'none',
        }
        from QStar import QStar # Import locally to avoid top-level issues if run differently
        strategy_indic = QStar(config=strategy_config)
        metadata = {'pair': PAIR, 'timeframe': TIMEFRAME}
        processed_df = strategy_indic.populate_indicators(df, metadata)
        logger.info("Indicators calculated using QStar strategy logic.")
        return processed_df
    except Exception as e:
        logger.error(f"Failed to calculate indicators using QStar: {e}", exc_info=True)
        # --- Fallback: Add MINIMAL indicators needed by TradingEnvironment/MarketState ---
        logger.warning("Falling back to minimal indicator calculation.")
        df['close'] = pd.to_numeric(df['close'], errors='coerce') # Ensure numeric
        df['returns'] = df['close'].pct_change().fillna(0)
        df['volatility_regime'] = df['returns'].rolling(20).std().fillna(0.5) # Example
        df['qerc_trend'] = df['returns'].rolling(10).mean().fillna(0.5) * 10 # Example
        df['qerc_momentum'] = df['close'].pct_change(5).fillna(0.5) # Example
        df['iqad_score'] = 0.1 # Example default
        df['performance_metric'] = df['returns'].rolling(20).mean().fillna(0.5) # Example
        return df

# --- Refine Agent (Keep as is, now operates on TradingEnvironment) ---
# (Keep the refine_agent function definition exactly as you provided)
def refine_agent(agent: SophisticatedQLearningAgent, environment: TradingEnvironment, # Changed type hint
                max_refinements: int = 5, evaluation_episodes: int = 50) -> SophisticatedQLearningAgent:
    # ... (Function content remains the same, it just uses the TradingEnvironment now) ...
    logger.info(f"Beginning refinement process with up to {max_refinements} steps")
    best_performance = float('-inf')
    best_agent_path = None # Changed variable name for clarity
    refinement_count = 0
    # Convergence check might need adjustment based on reward scale
    convergence_threshold = 0.0001 # Example threshold for TradingEnv rewards
    window_size = 50

    # Ensure environment is compatible with evaluation if needed
    if not hasattr(environment, 'reset') or not hasattr(environment, 'step'):
        logger.error("Environment passed to refine_agent lacks reset/step methods.")
        return agent

    for ref_step in range(max_refinements):
        logger.info(f"--- Refinement Step {ref_step + 1}/{max_refinements} ---")
        # Increase agent complexity (optional, might not be needed if state space is fixed)
        # new_states = agent.states + max(10, int(agent.states * 0.1)) # Slower expansion
        # new_actions = agent.actions
        # agent.resize_q_table(new_states, new_actions) # Consider if state mapping changes

        # Adjust training parameters for refinement phase
        agent.max_episodes += 2000 # Add more episodes per refinement
        agent.max_steps_per_episode += 50
        agent.learning_rate = max(0.01, agent.learning_rate * 0.9) # Slightly decrease LR
        agent.exploration_rate = max(agent.min_exploration_rate, agent.exploration_rate * 0.9) # Decrease exploration faster

        # Train the agent further
        logger.info(f"Refinement Training (Max Ep: {agent.max_episodes})...")
        converged, episodes = agent.train(environment) # Pass the TradingEnvironment

        # Evaluate the agent
        logger.info(f"Refinement Evaluation ({evaluation_episodes} episodes)...")
        # Use the agent's own evaluate method, passing the TradingEnvironment
        eval_results = agent.evaluate(environment, num_episodes=evaluation_episodes)
        current_performance = eval_results['mean_reward'] # Assuming evaluate returns this key

        logger.info(f"Refinement Step {ref_step + 1}: Performance={current_performance:.4f}, Best={best_performance:.4f}, Converged={converged}, Episodes={episodes}")

        # Check if this is the best performing agent so far
        if current_performance > best_performance:
            best_performance = current_performance
            # Save the current best agent state
            try:
                 if best_agent_path: # Remove previous best temp file
                      if os.path.exists(best_agent_path): os.remove(best_agent_path)
                 best_agent_path = f"temp_refined_agent_{ref_step+1}.pkl"
                 agent.save(best_agent_path) # Save the agent's current state
                 logger.info(f"Saved new best agent state to {best_agent_path}")
            except Exception as e_save:
                 logger.error(f"Error saving temporary refined agent state: {e_save}")
                 best_agent_path = None # Reset if save failed

        # Optional: Check for early stopping based on convergence or performance stagnation
        # _, convergence_value = agent.has_converged(threshold=convergence_threshold, window_size=window_size)
        # if converged:
        #     logger.info("Refinement converged early.")
        #     break

    # Load the best agent state found during refinement
    if best_agent_path and os.path.exists(best_agent_path):
        logger.info(f"Loading best agent found during refinement from: {best_agent_path}")
        try:
            # Use the agent's load method - creates a new instance!
            best_agent_instance = SophisticatedQLearningAgent.load(best_agent_path)
            # Clean up ALL temp files after loading the final best one
            for i in range(max_refinements):
                 temp_path = f"temp_refined_agent_{i+1}.pkl"
                 if os.path.exists(temp_path): os.remove(temp_path)
            return best_agent_instance # Return the loaded best agent
        except Exception as e_load:
             logger.error(f"Error loading best refined agent state: {e_load}. Returning last agent state.")
             # Fallback: clean up any temp file we created but couldn't load
             if os.path.exists(best_agent_path): os.remove(best_agent_path)
             return agent # Return the agent as it was at the end of the loop
    else:
         logger.warning("No better agent state saved during refinement. Returning last agent state.")
         # Clean up any temp files that might exist if best_agent_path got reset
         for i in range(max_refinements):
             temp_path = f"temp_refined_agent_{i+1}.pkl"
             if os.path.exists(temp_path): os.remove(temp_path)
         return agent # Return the agent as it was at the end of the loop


# --- REVISED run_demonstration ---
def run_training():
    """Run demonstration using TradingEnvironment and historical market data."""
    logger.info("Starting Q* Learning demonstration with Market Data")

    # 1. Load Historical Data
    market_df = load_freqtrade_data(HISTORICAL_DATA_FILE)
    if market_df.empty:
        logger.error("Cannot run demonstration without market data.")
        return

    # 2. Calculate Indicators (Crucial Step)
    # This step ensures the DataFrame has all columns the environment expects
    processed_df = calculate_indicators(market_df)
    if processed_df.empty:
        logger.error("Indicator calculation failed. Cannot run demonstration.")
        return

    # 3. Initialize RiverML (Needed by TradingEnvironment)
    river_ml_instance = None
    if RIVER_AVAILABLE:
        try:
            river_ml_instance = RiverOnlineML( # Use default config or define here
                 drift_detector_type='adwin', anomaly_detector_type='hst',
                 feature_window=50, drift_sensitivity=0.05, anomaly_threshold=0.95
            )
            logger.info("RiverOnlineML initialized for TradingEnvironment.")
        except Exception as e:
            logger.error(f"Failed to initialize RiverML for demo: {e}")
            # Decide if you want to continue without RiverML features in env
            river_ml_instance = None # Ensure it's None
            logger.warning("Continuing demonstration without RiverML features in environment.")
    else:
        logger.warning("RiverML not available. Environment will lack drift/anomaly features.")


    # 4. Create TradingEnvironment
    try:
        # Pass the DataFrame WITH indicators calculated
        env = TradingEnvironment(
            river_ml=river_ml_instance, # Pass the instance (can be None)
            price_data=processed_df,
            window_size=50, # Must match MarketState's expectation
            initial_balance=10000.0,
            transaction_fee=0.001
        )
        logger.info("TradingEnvironment created successfully.")
    except Exception as e:
        logger.error(f"Failed to create TradingEnvironment: {e}", exc_info=True)
        return

    # 5. Initialize Q* Agent
    # The state/action counts come from the TradingEnvironment instance
    agent = SophisticatedQLearningAgent(
        states=env.num_states,
        actions=env.num_actions,
        learning_rate=0.05, # Potentially tune these
        discount_factor=0.97,
        exploration_rate=1.0,
        min_exploration_rate=0.05,
        exploration_decay_rate=0.997, # Slower decay for more complex env
        use_adaptive_learning_rate=True,
        use_experience_replay=True,
        experience_buffer_size=50000, # Larger buffer
        batch_size=64,
        use_quantum_representation=True, # Example
        max_episodes=50 # Reduce episodes for faster demo run initially
    )
    logger.info(f"Agent initialized: States={agent.states}, Actions={agent.actions}")


    # 6. Train the Agent using the TradingEnvironment
    logger.info("Starting agent training...")
    converged, episodes = agent.train(env) # Pass the TradingEnvironment instance
    logger.info(f"Training finished. Converged: {converged}, Episodes: {episodes}")

    # 7. Plot metrics from training
    agent.metrics.plot_metrics()

    # 8. Evaluate the Agent (Optional, uses the same env)
    logger.info("Evaluating agent post-training...")
    agent.evaluate(env, num_episodes=5) # Evaluate on a few episodes

    # 9. Refine the Agent (Optional)
    logger.info("Refining agent...")
    refined_agent = refine_agent(agent, env, max_refinements=1, evaluation_episodes=10) # Refine lightly for demo

    # 10. Save the FINAL Trained/Refined Agent State
    logger.info(f"Saving trained agent state to: {AGENT_SAVE_PATH}")
    try:
        # Use the potentially refined agent if refinement was run
        agent_to_save = agent # Use original agent if refinement skipped
        agent_to_save = refined_agent # Use this line if refinement was run
        agent_to_save.save(str(AGENT_SAVE_PATH))
        logger.info("Agent state saved successfully.")
    except Exception as e:
        logger.error(f"Error saving final agent state: {e}", exc_info=True)


    logger.info("Training completed")

    
if __name__ == "__main__":
    # Ensure model directory exists before running
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    run_training()