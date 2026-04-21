import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [balance, setBalance] = useState(10000);
  const [totalPnl, setTotalPnl] = useState(0);
  const [performance, setPerformance] = useState({ win_rate: 0, total_trades: 0, sharpe_ratio: 0 });
  const [prices, setPrices] = useState({});
  const [symbol, setSymbol] = useState('BTC-USD');
  const [quantity, setQuantity] = useState(0.001);
  const [tradeMessage, setTradeMessage] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch initial data
    fetchData();
    
    // Refresh every 3 seconds
    const interval = setInterval(fetchData, 3000);
    
    return () => clearInterval(interval);
  }, []);

  const fetchData = async () => {
    try {
      // Fetch balance
      const balanceRes = await fetch('http://localhost:8000/api/v1/balance');
      const balanceData = await balanceRes.json();
      setBalance(balanceData.balance);
      setTotalPnl(balanceData.total_pnl);
      
      // Fetch performance
      const perfRes = await fetch('http://localhost:8000/api/v1/performance');
      const perfData = await perfRes.json();
      setPerformance(perfData);
      
      // Fetch prices
      const pricesRes = await fetch('http://localhost:8000/api/v1/market/prices');
      const pricesData = await pricesRes.json();
      setPrices(pricesData);
      
      setLoading(false);
    } catch (error) {
      console.error('Error fetching data:', error);
      setLoading(false);
    }
  };

  const executeTrade = async (side) => {
    try {
      const response = await fetch(
        `http://localhost:8000/api/v1/trade/execute?symbol=${symbol}&side=${side}&quantity=${quantity}`,
        { method: 'POST' }
      );
      const result = await response.json();
      setTradeMessage(result.message);
      setTimeout(() => setTradeMessage(''), 3000);
      fetchData(); // Refresh after trade
    } catch (error) {
      setTradeMessage('Error executing trade: ' + error.message);
      setTimeout(() => setTradeMessage(''), 3000);
    }
  };

  if (loading) {
    return (
      <div className="loading">
        <div className="spinner"></div>
        <h2>Loading AI Trading Bot...</h2>
        <p>Connecting to backend at http://localhost:8000</p>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="header">
        <div className="logo">
          <span className="logo-icon">🤖</span>
          <h1>AI Trading Bot</h1>
        </div>
        <div className="status-badge">● VIRTUAL MODE</div>
      </header>

      <div className="dashboard">
        {/* Balance Card */}
        <div className="card">
          <h3>💰 Account Balance</h3>
          <div className="metric-large">${balance.toLocaleString()}</div>
          <div className={`metric-change ${totalPnl >= 0 ? 'positive' : 'negative'}`}>
            Total P&L: ${totalPnl.toLocaleString()}
          </div>
        </div>

        {/* Performance Card */}
        <div className="card">
          <h3>📊 Performance</h3>
          <div className="metrics-grid">
            <div>
              <div className="metric-label">Win Rate</div>
              <div className="metric-value positive">{performance.win_rate}%</div>
            </div>
            <div>
              <div className="metric-label">Total Trades</div>
              <div className="metric-value">{performance.total_trades}</div>
            </div>
            <div>
              <div className="metric-label">Sharpe Ratio</div>
              <div className="metric-value">{performance.sharpe_ratio}</div>
            </div>
          </div>
        </div>

        {/* Market Prices Card */}
        <div className="card">
          <h3>💹 Market Prices</h3>
          <div className="price-list">
            {Object.entries(prices).slice(0, 5).map(([sym, price]) => (
              <div key={sym} className="price-item">
                <span className="symbol">{sym}</span>
                <span className="price">${typeof price === 'number' ? price.toLocaleString() : price}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Trading Card */}
        <div className="card full-width">
          <h3>🔄 Quick Trade</h3>
          <div className="trade-controls">
            <select value={symbol} onChange={(e) => setSymbol(e.target.value)}>
              <option value="BTC-USD">BTC-USD</option>
              <option value="ETH-USD">ETH-USD</option>
              <option value="BNB-USD">BNB-USD</option>
              <option value="SOL-USD">SOL-USD</option>
              <option value="ADA-USD">ADA-USD</option>
            </select>
            <input
              type="number"
              value={quantity}
              onChange={(e) => setQuantity(parseFloat(e.target.value))}
              step="0.001"
              placeholder="Quantity"
            />
            <button className="btn-buy" onClick={() => executeTrade('BUY')}>
              BUY
            </button>
            <button className="btn-sell" onClick={() => executeTrade('SELL')}>
              SELL
            </button>
          </div>
          {tradeMessage && <div className="trade-message">{tradeMessage}</div>}
        </div>

        {/* AI Strategies Card */}
        <div className="card full-width">
          <h3>🧠 Active AI Strategies</h3>
          <div className="strategies-grid">
            <div className="strategy-card">
              <h4>XGBoost</h4>
              <p>Gradient boosting for price prediction</p>
              <div className="strategy-stats">Win Rate: 62% | Profit: +$1,250</div>
            </div>
            <div className="strategy-card">
              <h4>Bayesian NN</h4>
              <p>Neural network with uncertainty estimation</p>
              <div className="strategy-stats">Win Rate: 58% | Profit: +$980</div>
            </div>
            <div className="strategy-card">
              <h4>Reinforcement Learning</h4>
              <p>Deep Q-learning for optimal trading</p>
              <div className="strategy-stats">Win Rate: 65% | Profit: +$2,100</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
