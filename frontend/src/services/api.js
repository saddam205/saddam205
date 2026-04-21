const API_URL = 'http://localhost:8000';

class TradingAPI {
  async getPortfolio() {
    const response = await fetch(`${API_URL}/api/v1/balance`);
    return response.json();
  }

  async getPerformance() {
    const response = await fetch(`${API_URL}/api/v1/performance`);
    return response.json();
  }

  async getMarketData(symbol) {
    const response = await fetch(`${API_URL}/api/v1/market/prices`);
    return response.json();
  }

  async getStrategies() {
    // Return mock strategies data
    return {
      strategies: [
        { name: 'xgboost', display_name: 'XGBoost', enabled: true, description: 'Gradient boosting for price prediction', performance: { win_rate: 62, profit: 1250 } },
        { name: 'bayesian', display_name: 'Bayesian NN', enabled: true, description: 'Bayesian neural network with uncertainty', performance: { win_rate: 58, profit: 980 } },
        { name: 'rl', display_name: 'Reinforcement Learning', enabled: true, description: 'Deep Q-learning for optimal trading', performance: { win_rate: 65, profit: 2100 } }
      ]
    };
  }

  async executeTrade(symbol, quantity, side) {
    const response = await fetch(`${API_URL}/api/v1/trade/execute?symbol=${symbol}&side=${side}&quantity=${quantity}`, {
      method: 'POST'
    });
    return response.json();
  }
}

export default new TradingAPI();
