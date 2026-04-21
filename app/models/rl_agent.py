"""
Reinforcement Learning Agent for Trading
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple, Optional

# Try to import torch for deep RL
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    DEEP_RL_AVAILABLE = True
except ImportError:
    DEEP_RL_AVAILABLE = False
    print("Warning: PyTorch not available. Using basic RL.")

class TradingEnvironment(gym.Env):
    """Custom trading environment for RL"""
    def __init__(self, data=None, initial_balance=10000, max_position=0.3):
        super().__init__()
        
        self.data = data if data is not None else []
        self.initial_balance = initial_balance
        self.max_position = max_position
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0
        self.trades = []
        self.returns = []
        
        # Action space: [position_size, risk_tolerance, exit_threshold]
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([max_position, 1, 1]),
            dtype=np.float32
        )
        
        # Observation space: [price, volume, rsi, macd, sentiment, ...]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(20,),
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        self.trades = []
        self.returns = []
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute trading action"""
        position_size, risk_tolerance, exit_threshold = action
        
        # Get current price (simplified)
        current_price = 50000  # Default BTC price
        
        if self.data and len(self.data) > 0:
            if hasattr(self.data, 'iloc') and self.current_step < len(self.data):
                try:
                    current_price = self.data['Close'].iloc[self.current_step]
                except:
                    current_price = 50000
        
        prev_balance = self.balance
        
        # Execute trade based on RL action
        if position_size > self.position:
            # Buy
            buy_amount = (position_size - self.position) * current_price
            if buy_amount <= self.balance:
                self.balance -= buy_amount
                self.position = position_size
        
        elif position_size < self.position:
            # Sell
            sell_amount = (self.position - position_size) * current_price
            self.balance += sell_amount
            self.position = position_size
        
        # Update step
        self.current_step += 1
        terminated = self.current_step >= 100  # Max steps
        truncated = False
        
        # Calculate reward
        reward = (self.balance - prev_balance) / self.initial_balance
        self.returns.append(reward)
        
        # Add risk penalty
        if reward < -0.05:  # Large loss penalty
            reward -= 0.1
        
        # Get next observation
        next_obs = self._get_observation()
        
        info = {
            'balance': self.balance,
            'position': self.position,
            'step': self.current_step,
            'returns': self.returns
        }
        
        return next_obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Get current market observation"""
        if self.current_step >= 100:
            return np.zeros(20)
        
        # Create synthetic observations
        obs = np.array([
            self.position,
            self.balance / self.initial_balance,
            np.mean(self.returns[-10:]) if self.returns else 0,
            np.std(self.returns[-10:]) if len(self.returns) >= 10 else 0,
            len(self.trades),
        ])
        
        # Pad to 20 dimensions
        if len(obs) < 20:
            obs = np.pad(obs, (0, 20 - len(obs)), 'constant')
        
        return obs.astype(np.float32)


class RLTrader:
    """Reinforcement Learning Trading Agent"""
    def __init__(self, state_dim=20, action_dim=3, hidden_dim=256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        if DEEP_RL_AVAILABLE:
            self._init_deep_rl()
        else:
            self._init_basic_rl()
    
    def _init_deep_rl(self):
        """Initialize deep RL networks"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor Network (Policy)
        self.actor = self._build_network(self.state_dim, self.action_dim, self.hidden_dim)
        self.actor_target = self._build_network(self.state_dim, self.action_dim, self.hidden_dim)
        
        # Critic Network (Value)
        self.critic = self._build_critic(self.state_dim, self.action_dim, self.hidden_dim)
        self.critic_target = self._build_critic(self.state_dim, self.action_dim, self.hidden_dim)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        
        self.tau = 0.005
        self.use_deep_rl = True
    
    def _init_basic_rl(self):
        """Initialize basic Q-learning"""
        self.q_table = {}
        self.use_deep_rl = False
    
    def _build_network(self, state_dim, action_dim, hidden_dim):
        """Build actor network"""
        return nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()
        ).to(self.device)
    
    def _build_critic(self, state_dim, action_dim, hidden_dim):
        """Build critic network"""
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)
    
    def act(self, state, explore=True):
        """Select action using policy"""
        if not self.use_deep_rl:
            return self._act_basic(state, explore)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        if explore and np.random.random() < self.epsilon:
            # Add exploration noise
            action += np.random.normal(0, 0.1, size=action.shape)
            action = np.clip(action, 0, 1)
        
        return action
    
    def _act_basic(self, state, explore=True):
        """Basic Q-learning action selection"""
        if explore and np.random.random() < self.epsilon:
            return np.random.uniform(0, 1, self.action_dim)
        
        state_key = self._get_state_key(state)
        if state_key in self.q_table:
            return self.q_table[state_key]
        else:
            return np.array([0.5, 0.5, 0.5])
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience"""
        self.memory.append((state, action, reward, next_state, done))
        
        if not self.use_deep_rl:
            # Update Q-table for basic RL
            state_key = self._get_state_key(state)
            if state_key not in self.q_table:
                self.q_table[state_key] = action
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
    
    def train(self, batch_size=64):
        """Train the RL agent"""
        if len(self.memory) < batch_size:
            return
        
        if self.use_deep_rl:
            self._train_deep(batch_size)
    
    def _train_deep(self, batch_size):
        """Deep RL training"""
        # Sample batch
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        
        # Update Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(torch.cat([next_states, next_actions], dim=1))
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q = self.critic(torch.cat([states, actions], dim=1))
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor
        new_actions = self.actor(states)
        actor_loss = -self.critic(torch.cat([states, new_actions], dim=1)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update targets
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _soft_update(self, source, target):
        """Soft update target networks"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)
    
    def _get_state_key(self, state):
        """Convert state to hashable key for Q-table"""
        return tuple(np.round(state[:5], 2))
    
    def save(self, path):
        """Save model"""
        import joblib
        if self.use_deep_rl:
            torch.save({
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'epsilon': self.epsilon
            }, path)
        else:
            joblib.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon
            }, path)
    
    def load(self, path):
        """Load model"""
        import os
        if not os.path.exists(path):
            return
        
        if self.use_deep_rl:
            checkpoint = torch.load(path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.epsilon = checkpoint.get('epsilon', 1.0)
        else:
            import joblib
            data = joblib.load(path)
            self.q_table = data.get('q_table', {})
            self.epsilon = data.get('epsilon', 1.0)
