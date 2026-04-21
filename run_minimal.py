#!/usr/bin/env python
"""
Simple runner for the AI Trading Bot
"""
from app.minimal_main import app
import uvicorn
try:
    import gymnasium as gym
except ImportError:
    import gym
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 AI TRADING BOT - STARTING...")
    print("=" * 60)
    print("📍 API URL: http://localhost:8000")
    print("📍 Documentation: http://localhost:8000/docs")
    print("📍 Health Check: http://localhost:8000/health")
    print("📍 Dashboard: Open dashboard.html in browser")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
