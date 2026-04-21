"""
news_api.py
Part of the app/services module.
News API integration for market sentiment and event detection.
"""

import aiohttp
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """News article data structure"""
    title: str
    description: str
    content: str
    source: str
    url: str
    published_at: datetime
    sentiment_score: Optional[float] = None
    relevance_score: Optional[float] = None
    symbols: List[str] = None


class NewsAPIService:
    """
    News API service for fetching financial news and sentiment analysis
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize News API service
        
        Args:
            api_key: News API key (optional, uses config if not provided)
        """
        from app.config import config
        self.api_key = api_key or getattr(config, 'NEWS_API_KEY', None)
        self.base_url = "https://newsapi.org/v2"
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def get_market_news(self, symbols: List[str] = None, 
                              days_back: int = 7,
                              page_size: int = 50) -> List[NewsArticle]:
        """
        Get market news for specified symbols
        
        Args:
            symbols: List of trading symbols
            days_back: Number of days to look back
            page_size: Number of articles to fetch
        
        Returns:
            List of news articles
        """
        if not self.api_key:
            logger.warning("No News API key provided")
            return []
        
        query = " OR ".join(symbols) if symbols else "crypto OR stocks OR forex"
        
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        params = {
            'q': query,
            'from': from_date,
            'sortBy': 'relevancy',
            'pageSize': page_size,
            'apiKey': self.api_key,
            'language': 'en'
        }
        
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/everything", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = self._parse_articles(data.get('articles', []), symbols)
                    logger.info(f"Fetched {len(articles)} news articles")
                    return articles
                else:
                    logger.error(f"News API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Failed to fetch news: {e}")
            return []
    
    async def get_top_headlines(self, category: str = "business",
                                page_size: int = 20) -> List[NewsArticle]:
        """
        Get top headlines by category
        
        Args:
            category: News category (business, technology, etc.)
            page_size: Number of headlines to fetch
        
        Returns:
            List of news articles
        """
        if not self.api_key:
            return []
        
        params = {
            'category': category,
            'pageSize': page_size,
            'apiKey': self.api_key,
            'country': 'us'
        }
        
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/top-headlines", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = self._parse_articles(data.get('articles', []))
                    return articles
                return []
        except Exception as e:
            logger.error(f"Failed to fetch headlines: {e}")
            return []
    
    def _parse_articles(self, articles: List[Dict], 
                        target_symbols: List[str] = None) -> List[NewsArticle]:
        """Parse API response into NewsArticle objects"""
        parsed = []
        
        for article in articles:
            # Extract symbols from content (simplified)
            symbols = []
            if target_symbols:
                content = f"{article.get('title', '')} {article.get('description', '')}".lower()
                for symbol in target_symbols:
                    if symbol.lower() in content:
                        symbols.append(symbol)
            
            parsed.append(NewsArticle(
                title=article.get('title', ''),
                description=article.get('description', ''),
                content=article.get('content', ''),
                source=article.get('source', {}).get('name', 'unknown'),
                url=article.get('url', ''),
                published_at=datetime.fromisoformat(article.get('publishedAt', '').replace('Z', '+00:00')),
                symbols=symbols
            ))
        
        return parsed
    
    async def analyze_sentiment(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """
        Analyze sentiment of news articles
        
        Args:
            articles: List of news articles
        
        Returns:
            Articles with sentiment scores added
        """
        for article in articles:
            # Simple sentiment analysis (in production, use NLP model)
            sentiment = self._simple_sentiment(article.title + " " + article.description)
            article.sentiment_score = sentiment
            article.relevance_score = 0.7  # Default relevance
        
        return articles
    
    def _simple_sentiment(self, text: str) -> float:
        """Simple sentiment scoring (placeholder for actual NLP)"""
        bullish_words = ['bullish', 'surge', 'rally', 'gain', 'rise', 'up', 'positive', 'growth']
        bearish_words = ['bearish', 'crash', 'drop', 'fall', 'down', 'negative', 'decline', 'loss']
        
        text_lower = text.lower()
        bullish_count = sum(1 for word in bullish_words if word in text_lower)
        bearish_count = sum(1 for word in bearish_words if word in text_lower)
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0
        
        return (bullish_count - bearish_count) / total
    
    async def close(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()