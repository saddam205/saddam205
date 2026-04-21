"""
bayesian_nn.py
Part of the app/models module.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class BayesianLinear(nn.Module):
    """Bayesian Linear Layer with uncertainty"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameters (mean and variance)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.01))
        self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features).normal_(-5, 0.01))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.01))
        self.bias_logvar = nn.Parameter(torch.Tensor(out_features).normal_(-5, 0.01))
        
    def forward(self, x):
        # Sample weights from distribution
        weight = self.weight_mu + torch.exp(0.5 * self.weight_logvar) * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + torch.exp(0.5 * self.bias_logvar) * torch.randn_like(self.bias_mu)
        
        return F.linear(x, weight, bias)

class BayesianTradingNetwork(nn.Module):
    """Complete BNN for trading with uncertainty estimation"""
    def __init__(self, input_dim, hidden_dim=256, output_dim=3):
        super().__init__()
        
        self.fc1 = BayesianLinear(input_dim, hidden_dim)
        self.fc2 = BayesianLinear(hidden_dim, hidden_dim)
        self.fc3 = BayesianLinear(hidden_dim, hidden_dim // 2)
        self.output_layer = BayesianLinear(hidden_dim // 2, output_dim)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, num_samples=10):
        """Forward pass with Monte Carlo dropout for uncertainty"""
        predictions = []
        
        for _ in range(num_samples):
            h = F.relu(self.fc1(x))
            h = self.dropout(h)
            h = F.relu(self.fc2(h))
            h = self.dropout(h)
            h = F.relu(self.fc3(h))
            output = self.output_layer(h)
            predictions.append(output)
        
        # Stack predictions
        predictions = torch.stack(predictions)
        
        # Mean and variance
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        uncertainty = torch.sqrt(variance)  # Standard deviation
        
        return mean, uncertainty
    
    def predict_with_confidence(self, x):
        """Get prediction with confidence intervals"""
        mean, uncertainty = self.forward(x)
        
        # 95% confidence interval
        lower_bound = mean - 1.96 * uncertainty
        upper_bound = mean + 1.96 * uncertainty
        
        # Trading signal based on confidence
        signal_probs = torch.softmax(mean, dim=-1)
        max_prob, signal_idx = torch.max(signal_probs, dim=-1)
        
        # Adjust confidence based on uncertainty
        adjusted_confidence = max_prob * (1 / (1 + uncertainty.mean()))
        
        return {
            'signal': ['SELL', 'HOLD', 'BUY'][signal_idx.item()],
            'confidence': adjusted_confidence.item(),
            'uncertainty': uncertainty.mean().item(),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'probabilities': signal_probs.detach().numpy()
        }

class BayesianTradingBot:
    """Complete Bayesian trading system"""
    def __init__(self, input_dim=50):
        self.model = BayesianTradingNetwork(input_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.trade_history = []
        
    def train(self, X, y, epochs=100):
        """Train with Bayesian inference"""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Forward pass with multiple samples
            mean, uncertainty = self.model.forward(X_tensor)
            
            # Loss: Cross-entropy + KL divergence (for Bayesian)
            ce_loss = F.cross_entropy(mean, y_tensor)
            kl_loss = self._kl_divergence()
            
            total_loss = ce_loss + 0.01 * kl_loss
            
            total_loss.backward()
            self.optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss={total_loss.item():.4f}")
    
    def _kl_divergence(self):
        """Calculate KL divergence for Bayesian layers"""
        kl = 0
        for module in self.model.modules():
            if isinstance(module, BayesianLinear):
                weight_var = torch.exp(module.weight_logvar)
                bias_var = torch.exp(module.bias_logvar)
                
                kl += 0.5 * torch.sum(weight_var + module.weight_mu**2 - 1 - module.weight_logvar)
                kl += 0.5 * torch.sum(bias_var + module.bias_mu**2 - 1 - module.bias_logvar)
        
        return kl
    
    def predict(self, features):
        """Get prediction with full uncertainty quantification"""
        X_tensor = torch.FloatTensor(features).unsqueeze(0)
        return self.model.predict_with_confidence(X_tensor)