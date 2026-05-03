# 🤖 Advanced AI-Powered Cryptocurrency Trading Bot

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red.svg)](https://xgboost.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced AI-powered cryptocurrency trading bot featuring multiple machine learning models, Reinforcement Learning, RAG-based market intelligence, and comprehensive backtesting capabilities.

**Language Composition:**
- 🐍 Python: 78.9%
- 🌐 HTML: 10.7%
- 📱 JavaScript: 9.4%
- Other: 1%

## 🚀 Key Features

- **Multiple AI Models**: XGBoost, Bayesian Neural Networks, and Reinforcement Learning ensemble
- **RAG System**: Market intelligence with MiniLM embeddings and vector search (ChromaDB/FAISS)
- **Multi-Timeframe Analysis**: Technical signals across 1m, 5m, 15m, 1h, 4h, 1d timeframes
- **Real-time Dashboard**: HTML/JavaScript dashboards with trading metrics and visualizations
- **Risk Management**: Position sizing, stop-loss, take-profit, and drawdown limits
- **Backtesting Engine**: Walk-forward validation, Monte Carlo simulation, stress testing
- **Paper Trading**: Virtual trading mode with realistic costs and slippage simulation
- **Telegram Integration**: Real-time alerts and trading command execution
- **Performance Monitoring**: Prometheus metrics and comprehensive logging

## 📊 Expected Performance Targets

| Metric | Target |
|--------|--------|
| Accuracy | 58-65% |
| Sharpe Ratio | 1.8-2.2 |
| Max Drawdown | 3-6% |
| Win Rate | 60-68% |

## 🏗️ Project Architecture

```
ai_trading_bot/
├── app/
│   ├── api/              # FastAPI endpoints & WebSocket
│   ├── core/             # Trading engine & risk management
│   ├── models/           # XGBoost, BNN, RL, Ensemble models
│   ├── strategies/       # Trading strategy implementations
│   ├── rag/              # RAG knowledge base system
│   ├── analysis/         # Technical & sentiment analysis
│   ├── validation/       # Backtesting & walk-forward validation
│   ├── services/         # Exchange & notification services
│   ├── utils/            # Logger, helpers, utilities
│   └── config.py         # Configuration management
├── frontend/             # React/HTML dashboards
│   ├── dashboard.html    # Main trading dashboard
│   ├── dashboard_dark_mobile.html  # Mobile-optimized dark mode
│   ├── trading_dashboard.html      # Trading metrics dashboard
│   ├── enhanced_dashboard.html     # Advanced analytics dashboard
│   └── simple_dashboard.html       # Minimal dashboard
├── data/                 # Market data, models, and knowledge base
│   ├── market/           # Historical OHLCV data
│   ├── models/           # Trained model files
│   ├── trades/           # Trade logs
│   ├── reports/          # Backtesting reports
│   ├── knowledge/        # RAG knowledge base & embeddings
│   └── logs/             # Application logs
├── scripts/              # Training and monitoring scripts
│   ├── train_advanced_model.py
│   ├── train_ensemble.py
│   ├── run_backtest.py
│   ├── update_knowledge.py
│   ├── init_knowledge_base.py
│   └── monitor.py
├── tests/                # Unit and integration tests
├── configs/              # YAML configuration files
├── requirements.txt      # Python dependencies
├── requirements-dev.txt  # Development dependencies
├── run.py               # Main entry point
├── fixall.py            # Automated bug fixing script
├── Dockerfile           # Docker containerization
└── docker-compose.yml   # Multi-container orchestration
```

## 📋 Prerequisites

- **Python**: 3.9 or higher
- **Node.js**: 16+ (for frontend dashboard)
- **System**: 8GB RAM minimum (16GB recommended), 10GB free disk space
- **Optional**: Docker & Docker Compose for containerized deployment

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Advanced-AI-powered-cryptocurrency-trading-bot.git
cd Advanced-AI-powered-cryptocurrency-trading-bot
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Development Dependencies (Optional)

```bash
pip install -r requirements-dev.txt
```

### 5. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your:
# - Binance API keys
# - Telegram bot token
# - Trading parameters
# - Database credentials
```

### 6. Create Required Data Directories

```bash
mkdir -p data/{market,models,trades,logs,reports,knowledge}
```

### 7. Initialize Knowledge Base

```bash
python scripts/init_knowledge_base.py
```

## 🚀 Running the Application

### Start Backend API Server

```bash
# Development mode with auto-reload
python run.py --mode api --reload

# Production mode
python run.py --mode api --port 8000
```

### Start Trading Engine

```bash
# Virtual trading mode (recommended for testing)
python run.py --mode trading --trading-mode VIRTUAL

# Paper trading mode (simulated with real data)
python run.py --mode trading --trading-mode PAPER

# Real trading mode (USE WITH EXTREME CAUTION!)
python run.py --mode trading --trading-mode REAL
```

### Run Both API and Trading Engine

```bash
python run.py --mode both --trading-mode VIRTUAL
```

### Access Dashboard

After starting the API server, open your browser:

- **Dashboard**: http://localhost:8000/docs (API documentation)
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **HTML Dashboards**: 
  - Main: `dashboard.html`
  - Mobile: `dashboard_dark_mobile.html`
  - Trading: `trading_dashboard.html`

## 📊 API Endpoints

Once the backend is running, access the interactive documentation at `http://localhost:8000/docs`

### Main Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/trade/execute` | Execute a trade |
| GET | `/api/v1/balance/{mode}` | Get account balance |
| GET | `/api/v1/positions` | Get open positions |
| GET | `/api/v1/performance` | Get performance metrics |
| POST | `/api/v1/backtest/run` | Run backtest |
| GET | `/api/v1/analysis/sentiment/{symbol}` | Get sentiment analysis |
| GET | `/api/v1/health` | Health check |
| WS | `/ws` | WebSocket for real-time updates |

## 🤖 Training Models

### Train XGBoost Model

```bash
python scripts/train_advanced_model.py --symbol BTCUSDT --optimize
python scripts/train_advanced_model.py --symbol ETHUSDT --epochs 100
```

### Train Ensemble Model

```bash
python scripts/train_ensemble.py --symbol BTCUSDT --epochs 100
python scripts/train_ensemble.py --all-symbols --concurrent
```

### Update RAG Knowledge Base

```bash
# Initialize with sample documents
python scripts/update_knowledge.py --init

# Add new documents
python scripts/update_knowledge.py --add-sample

# Search knowledge base
python scripts/update_knowledge.py --search "How to manage risk?"

# Export knowledge base
python scripts/update_knowledge.py --export data/knowledge/export.json
```

## 📈 Backtesting

### Run Single Strategy Backtest

```bash
python scripts/run_backtest.py --symbol BTCUSDT --strategy trend_following --days 180
```

### Run All Strategies with Validation

```bash
python scripts/run_backtest.py --symbol BTCUSDT --strategy all --walk-forward
```

### Monte Carlo Simulation

```bash
python scripts/run_backtest.py --symbol BTCUSDT --monte-carlo --iterations 10000
```

### Stress Testing

```bash
python scripts/run_backtest.py --symbol BTCUSDT --stress-test --scenarios market_crash flash_crash
```

### Save Results

```bash
python scripts/run_backtest.py --symbol BTCUSDT --output data/reports/my_backtest.json
```

## 📊 Monitoring & Logging

### System Monitor

```bash
# Run continuous monitoring
python scripts/monitor.py

# One-shot health check
python scripts/monitor.py --one-shot

# Generate report only
python scripts/monitor.py --report
```

### Health Check

```bash
curl http://localhost:8000/health
```

### View Logs

```bash
# Application logs
tail -f data/logs/trading.log

# Error logs
tail -f data/logs/errors.log

# Performance logs
tail -f data/logs/performance.log
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v

# Run with coverage report
pytest tests/ --cov=app --cov-report=html

# Run specific test case
pytest tests/test_models.py::TestXGBoostModel::test_train_and_predict -v
```

## 🐳 Docker Deployment

### Build and Run with Docker Compose

```bash
docker-compose up -d
```

### View Logs

```bash
docker-compose logs -f trading-bot
docker-compose logs -f api
```

### Stop Containers

```bash
docker-compose down
```

## 📝 Configuration

Edit `app/config.py` or use environment variables:

| Variable | Description | Default | Type |
|----------|-------------|---------|------|
| `TRADING_MODE` | VIRTUAL, PAPER, or REAL | VIRTUAL | str |
| `BINANCE_API_KEY` | Binance API key | "" | str |
| `BINANCE_API_SECRET` | Binance API secret | "" | str |
| `MIN_CONFIDENCE_THRESHOLD` | Minimum signal confidence | 0.65 | float |
| `STOP_LOSS_PERCENT` | Stop loss percentage | 2 | float |
| `TAKE_PROFIT_PERCENT` | Take profit percentage | 4 | float |
| `MAX_POSITION_SIZE` | Max position % of capital | 0.1 | float |
| `DEFAULT_INVESTMENT` | Initial capital for trading | 10000.0 | float |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token | "" | str |
| `TELEGRAM_CHAT_ID` | Telegram chat ID | "" | str |
| `API_HOST` | API host address | 0.0.0.0 | str |
| `API_PORT` | API port number | 8000 | int |

## 🔐 Security Best Practices

1. **Never commit `.env` file** with real API keys to version control
2. **Use testnet** for initial testing (`BINANCE_TESTNET=True`)
3. **Start with virtual trading mode** before paper or real trading
4. **Set conservative risk limits** in production
5. **Enable kill switch** for automatic trading halt on errors
6. **Use API key restrictions**:
   - IP whitelist
   - Disable withdrawal permissions
   - Limit to trading pairs only
7. **Rotate API keys** regularly
8. **Monitor logs** for unusual activity

## ⚠️ Risk Disclaimer

**IMPORTANT**: Trading cryptocurrencies involves substantial risk of loss and is not suitable for all investors. This software is for **educational purposes only**.

**The authors and contributors assume no responsibility for:**
- Financial losses resulting from the use of this software
- Bugs, errors, or unexpected behavior
- API changes or platform disruptions
- Incorrect configuration or usage

### Before Trading

- Always start with **virtual/paper trading mode**
- Test thoroughly on historical data (backtesting)
- Never invest more than you can afford to lose
- Set appropriate **stop-losses** on all positions
- Monitor the bot regularly and check logs
- Understand the trading strategies being used
- Have a clear risk management plan

## 🐛 Troubleshooting

### Module Not Found Errors

```bash
pip install -r requirements.txt --upgrade
pip install --no-cache-dir -r requirements.txt
```

### WebSocket Connection Failed

- Ensure port 8000 is not blocked by firewall
- Check that backend API is running: `curl http://localhost:8000/health`
- Verify WebSocket endpoint: `ws://localhost:8000/ws`

### Binance API Errors

- Verify API key and secret are correct
- Ensure IP is whitelisted in Binance settings
- Test on testnet first: `BINANCE_TESTNET=True`
- Check API key permissions (Trading enabled)

### Memory Issues

- Reduce batch sizes in `config.py`
- Use `--limit` parameter for data fetching
- Increase system swap space
- Run on a machine with more RAM (16GB+ recommended)

### Database Errors

- Check database connection in `.env`
- Ensure data directories exist: `mkdir -p data/{market,models,trades,logs}`
- Clear cache if needed: `rm -rf data/models/*.cache`

### Model Training Issues

- Ensure sufficient historical data (minimum 200 candles)
- Check if required columns exist: close, high, low, open, volume
- Verify training data quality in logs

## 📚 Documentation

- [API Documentation](http://localhost:8000/docs) - Interactive API docs
- [Architecture Overview](docs/ARCHITECTURE.md) - System design
- [Strategy Guide](docs/STRATEGIES.md) - Trading strategies
- [Risk Management](docs/RISK_MANAGEMENT.md) - Risk controls
- [Deployment Guide](docs/DEPLOYMENT.md) - Production setup

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Standards

- Follow PEP 8 style guide
- Add docstrings to all functions
- Write unit tests for new features
- Update README and docs
- Run tests before submitting PR: `pytest tests/ -v`

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **XGBoost** - Gradient boosting framework
- **FastAPI** - Modern Python web framework
- **PyTorch** - Deep learning library
- **Sentence Transformers** - Embedding models
- **ChromaDB/FAISS** - Vector databases
- **Binance** - Cryptocurrency exchange API
- **scikit-learn** - Machine learning library
- **Pandas** - Data manipulation library

## 📞 Support & Community

- **GitHub Issues**: [Report bugs or request features](https://github.com/yourusername/Advanced-AI-powered-cryptocurrency-trading-bot/issues)
- **Documentation**: Check the docs folder for detailed guides
- **Email**: support@aitradingbot.com (if available)

---

**Built with ❤️ for the trading and AI community**

*Disclaimer: This is a personal project for educational purposes. Use at your own risk. Always conduct thorough testing before deploying in production.*
