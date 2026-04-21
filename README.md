# 🤖 AI Trading Bot

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red.svg)](https://xgboost.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced AI-powered cryptocurrency trading bot featuring Bayesian Neural Networks, Reinforcement Learning, and RAG-based market intelligence.

## 🚀 Features

- **Multiple AI Models**: XGBoost, Bayesian Neural Networks, and Reinforcement Learning
- **RAG System**: Market intelligence with MiniLM embeddings and vector search
- **Multi-Timeframe Analysis**: Signals across 1m, 5m, 15m, 1h, 4h, 1d timeframes
- **Real-time Dashboard**: React-based dashboard with WebSocket updates
- **Risk Management**: Position sizing, stop-loss, take-profit, and drawdown limits
- **Backtesting Engine**: Walk-forward validation, Monte Carlo simulation, stress testing
- **Paper Trading**: Virtual trading with realistic costs and slippage
- **Telegram Integration**: Real-time alerts and trading commands

## 📊 Performance Targets

| Metric | Target |
|--------|--------|
| Accuracy | 58-65% |
| Sharpe Ratio | 1.8-2.2 |
| Max Drawdown | 3-6% |
| Win Rate | 60-68% |

## 🏗️ Architecture
ai_trading_bot/
├── app/
│ ├── api/ # FastAPI endpoints & WebSocket
│ ├── core/ # Trading engine & risk management
│ ├── models/ # XGBoost, BNN, RL, Ensemble
│ ├── strategies/ # Trading strategies
│ ├── rag/ # RAG knowledge base
│ ├── validation/ # Backtesting & validation
│ └── services/ # Exchange & notification services
├── frontend/ # React dashboard
├── data/ # Market data & models
├── scripts/ # Training & monitoring scripts
└── tests/ # Unit tests

## 📋 Prerequisites

- Python 3.9 or higher
- Node.js 16+ (for frontend)
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-trading-bot.git
cd ai-trading-bot
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
I'll create the comprehensive requirements.txt and README.md files for the AI Trading Bot.
requirements.txt
txt

# ============================================
# AI Trading Bot - Core Dependencies
# ============================================

# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
websockets==12.0
python-socketio==5.10.0
python-engineio==4.7.1

# Data Processing
pandas==2.1.3
numpy==1.24.3
scipy==1.11.4

# Machine Learning & AI
xgboost==2.0.2
scikit-learn==1.3.2
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0

# Reinforcement Learning
gymnasium==0.29.1

# RAG & Embeddings
sentence-transformers==2.2.2
faiss-cpu==1.7.4
chromadb==0.4.18

# Market Data & Exchanges
yfinance==0.2.33
ccxt==4.2.22
python-binance==1.0.19
ta==0.10.2

# Data Visualization
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.18.0

# API & Web
requests==2.31.0
httpx==0.25.1
aiohttp==3.9.1
websocket-client==1.6.4

# Configuration & Environment
python-dotenv==1.0.0
pyyaml==6.0.1
pydantic==2.5.0
pydantic-settings==2.1.0

# Database & Storage
sqlalchemy==2.0.23
redis==5.0.1
joblib==1.3.2

# Monitoring & Logging
prometheus-client==0.19.0
psutil==5.9.6
loguru==0.7.2

# Notifications
python-telegram-bot==20.6
aiofiles==23.2.1

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
pytest-mock==3.12.0
httpx==0.25.1
asyncio==3.4.3

# Utilities
tqdm==4.66.1
click==8.1.7
schedule==1.2.0
python-dateutil==2.8.2
pytz==2023.3

requirements-dev.txt
txt

# Development Dependencies
-r requirements.txt

# Code Quality
black==23.11.0
isort==5.12.0
flake8==6.1.0
pylint==3.0.2
mypy==1.7.0

# Documentation
mkdocs==1.5.3
mkdocs-material==9.5.1
mkdocstrings==0.24.0

# Debugging
ipython==8.17.2
ipdb==0.13.13
jupyter==1.0.0
notebook==7.0.6

# Profiling
py-spy==0.3.14
memory-profiler==0.61.0

# Pre-commit Hooks
pre-commit==3.5.0

README.md
markdown

# 🤖 AI Trading Bot

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red.svg)](https://xgboost.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced AI-powered cryptocurrency trading bot featuring Bayesian Neural Networks, Reinforcement Learning, and RAG-based market intelligence.

## 🚀 Features

- **Multiple AI Models**: XGBoost, Bayesian Neural Networks, and Reinforcement Learning
- **RAG System**: Market intelligence with MiniLM embeddings and vector search
- **Multi-Timeframe Analysis**: Signals across 1m, 5m, 15m, 1h, 4h, 1d timeframes
- **Real-time Dashboard**: React-based dashboard with WebSocket updates
- **Risk Management**: Position sizing, stop-loss, take-profit, and drawdown limits
- **Backtesting Engine**: Walk-forward validation, Monte Carlo simulation, stress testing
- **Paper Trading**: Virtual trading with realistic costs and slippage
- **Telegram Integration**: Real-time alerts and trading commands

## 📊 Performance Targets

| Metric | Target |
|--------|--------|
| Accuracy | 58-65% |
| Sharpe Ratio | 1.8-2.2 |
| Max Drawdown | 3-6% |
| Win Rate | 60-68% |

## 🏗️ Architecture

ai_trading_bot/
├── app/
│ ├── api/ # FastAPI endpoints & WebSocket
│ ├── core/ # Trading engine & risk management
│ ├── models/ # XGBoost, BNN, RL, Ensemble
│ ├── strategies/ # Trading strategies
│ ├── rag/ # RAG knowledge base
│ ├── validation/ # Backtesting & validation
│ └── services/ # Exchange & notification services
├── frontend/ # React dashboard
├── data/ # Market data & models
├── scripts/ # Training & monitoring scripts
└── tests/ # Unit tests
text


## 📋 Prerequisites

- Python 3.9 or higher
- Node.js 16+ (for frontend)
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-trading-bot.git
cd ai-trading-bot

2. Create Virtual Environment
bash

# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate

3. Install Python Dependencies
bash

pip install -r requirements.txt

4. Install Frontend Dependencies
bash

cd frontend
npm install
cd ..

5. Configure Environment Variables
bash

cp .env.example .env
# Edit .env with your API keys and settings

6. Create Required Directories
bash

mkdir -p data/{market,models,trades,logs,reports,knowledge}
mkdir -p data/knowledge

7. Initialize Knowledge Base
bash

# Generate embeddings and FAISS index
python scripts/init_knowledge_base.py

🚀 Running the Application
Start Backend API Server
bash

# Development mode with auto-reload
python run.py --mode api --reload

# Or using uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Start Trading Engine
bash

# Virtual trading mode (recommended for testing)
python run.py --mode trading --trading-mode VIRTUAL

# Paper trading mode
python run.py --mode trading --trading-mode PAPER

# Real trading mode (USE WITH CAUTION!)
python run.py --mode trading --trading-mode REAL

Start Both API and Trading Engine
bash

python run.py --mode both --trading-mode VIRTUAL

Start Frontend Dashboard
bash

cd frontend
npm start

The dashboard will be available at http://localhost:3000
📊 API Endpoints

Once running, access the interactive API documentation:

    Swagger UI: http://localhost:8000/docs

    ReDoc: http://localhost:8000/redoc

Main Endpoints
Method	Endpoint	Description
POST	/api/v1/trade/execute	Execute a trade
GET	/api/v1/balance/{mode}	Get account balance
GET	/api/v1/positions	Get open positions
GET	/api/v1/performance	Get performance metrics
POST	/api/v1/backtest/run	Run backtest
GET	/api/v1/analysis/sentiment/{symbol}	Get sentiment analysis
WS	/ws	WebSocket for real-time updates
🤖 Training Models
Train XGBoost Model
bash

python scripts/train_advanced_model.py --symbol BTC-USD --optimize

Train Ensemble Model
bash

python scripts/train_ensemble.py --symbol BTC-USD --epochs 100

Update RAG Knowledge Base
bash

# Add sample documents
python scripts/update_knowledge.py --add-sample

# Search knowledge base
python scripts/update_knowledge.py --search "How to manage risk?"

# Export knowledge base
python scripts/update_knowledge.py --export data/knowledge/export.json

📈 Running Backtests
bash

# Run single strategy backtest
python scripts/run_backtest.py --symbol BTC-USD --strategy trend_following --days 180

# Run all strategies with walk-forward validation
python scripts/run_backtest.py --symbol BTC-USD --strategy all --walk-forward

# Run with Monte Carlo simulation
python scripts/run_backtest.py --symbol BTC-USD --strategy all --monte-carlo

# Save results to file
python scripts/run_backtest.py --symbol BTC-USD --output data/reports/my_backtest.json

📊 Monitoring
System Monitor
bash

# Run continuous monitoring
python scripts/monitor.py

# One-shot health check
python scripts/monitor.py --one-shot

# Generate report only
python scripts/monitor.py --report

Health Check Endpoint
bash

curl http://localhost:8000/health

🧪 Running Tests
bash

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run specific test
pytest tests/test_models.py::TestXGBoostModel::test_train_and_predict -v

📝 Configuration

Edit app/config.py or use environment variables:
Variable	Description	Default
TRADING_MODE	VIRTUAL, PAPER, REAL	VIRTUAL
BINANCE_API_KEY	Binance API key	""
BINANCE_API_SECRET	Binance API secret	""
MIN_CONFIDENCE_THRESHOLD	Minimum signal confidence	0.65
STOP_LOSS_PERCENT	Stop loss percentage	2
TAKE_PROFIT_PERCENT	Take profit percentage	4
MAX_POSITION_SIZE	Max position size (% of capital)	0.1
TELEGRAM_BOT_TOKEN	Telegram bot token	""
TELEGRAM_CHAT_ID	Telegram chat ID	""
🔐 Security Best Practices

    Never commit .env file with real API keys

    Use testnet for initial testing (BINANCE_TESTNET=True)

    Start with virtual trading mode before real trading

    Set conservative risk limits in production

    Enable kill switch for automatic trading halt

    Use API key restrictions (IP whitelist, withdraw disabled)

🛡️ Risk Disclaimer

⚠️ IMPORTANT: Trading cryptocurrencies involves substantial risk of loss and is not suitable for all investors. This software is for educational purposes only. The authors and contributors are not responsible for any financial losses incurred while using this bot.

    Always start with virtual/paper trading

    Test thoroughly before using real funds

    Never invest more than you can afford to lose

    Set appropriate stop-losses

    Monitor the bot regularly

🐛 Troubleshooting
Common Issues

Module not found errors
bash

pip install -r requirements.txt --upgrade

WebSocket connection failed

    Ensure port 8000 is not blocked

    Check firewall settings

    Verify backend is running

Binance API errors

    Check API key permissions

    Ensure testnet is enabled for testing

    Verify IP is whitelisted

Memory issues

    Reduce batch sizes in config

    Use --limit parameter for data fetching

    Increase system swap space

Logs
bash

# Check application logs
tail -f data/logs/trading.log

# Check error logs
tail -f data/logs/errors.log

# Check performance logs
tail -f data/logs/performance.log

📚 Documentation

    API Documentation

    Architecture Overview

    Strategy Guide

    Risk Management

    Deployment Guide

🤝 Contributing

    Fork the repository

    Create a feature branch

    Commit your changes

    Push to the branch

    Create a Pull Request

📄 License

MIT License - see LICENSE file for details
🙏 Acknowledgments

    XGBoost for gradient boosting

    FastAPI for the API framework

    Binance for exchange integration

    Sentence Transformers for embeddings

📞 Support

    Issues: GitHub Issues

    Discord: Join our Discord

    Email: support@aitradingbot.com

Built with ❤️ for the trading community
text


## `scripts/init_knowledge_base.py`

```python
#!/usr/bin/env python3
"""
init_knowledge_base.py
Initialize RAG knowledge base with embeddings and FAISS index.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app.rag.knowledge_base import KnowledgeBase
from app.utils.logger import setup_logger

logger = setup_logger()


def init_knowledge_base():
    """Initialize knowledge base with sample documents"""
    print("=" * 60)
    print("📚 Initializing RAG Knowledge Base")
    print("=" * 60)
    
    # Create knowledge base
    kb = KnowledgeBase("data/knowledge/")
    
    # Build trading knowledge base
    print("\n📝 Building trading knowledge base...")
    kb.build_trading_knowledge_base()
    
    # Save knowledge base
    print("\n💾 Saving knowledge base...")
    kb.save()
    
    print("\n✅ Knowledge base initialized successfully!")
    print(f"   Location: data/knowledge/")
    print(f"   Documents: {len(kb.vector_store.documents)}")
    print(f"   Categories: {list(kb.categories.keys())}")
    
    # Test search
    print("\n🔍 Testing search...")
    results = kb.search("How to manage risk in trading?", top_k=3)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['content'][:100]}...")


if __name__ == "__main__":
    init_knowledge_base()

    # 1. Clone and setup
git clone <repository>
cd ai-trading-bot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
make install

# 4. Initialize data directories
make init-data

# 5. Initialize knowledge base
make init-kb

# 6. Run the bot (virtual trading mode)
make run-both

# 7. In another terminal, start the dashboard
make run-frontend

# 8. Open browser to http://localhost:3000