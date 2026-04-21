"""
twitter_api.py
Part of the app/services module.
Twitter/X API integration for social sentiment analysis.
"""

import aiohttp
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Tweet:
    """Tweet data structure"""
    id: str
    text: str
    author: str
    created_at: datetime
    like_count: int = 0
    retweet_count: int = 0
    reply_count: int = 0
    sentiment_score: Optional[float] = None


class TwitterService:
    """
    Twitter/X API service for fetching tweets and sentiment analysis
    """
    
    def __init__(self, bearer_token: str = None):
        """
        Initialize Twitter service
        
        Args:
            bearer_token: Twitter API bearer token
        """
        from app.config import config
        self.bearer_token = bearer_token or getattr(config, 'TWITTER_BEARER_TOKEN', None)
        self.base_url = "https://api.twitter.com/2"
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def search_tweets(self, query: str, max_results: int = 50,
                           start_time: datetime = None) -> List[Tweet]:
        """
        Search for tweets matching query
        
        Args:
            query: Search query
            max_results: Maximum number of results
            start_time: Start time for search
        
        Returns:
            List of tweets
        """
        if not self.bearer_token:
            logger.warning("Twitter API not configured")
            return []
        
        headers = {
            'Authorization': f'Bearer {self.bearer_token}'
        }
        
        params = {
            'query': query,
            'max_results': min(max_results, 100),
            'tweet.fields': 'created_at,public_metrics,author_id',
            'expansions': 'author_id'
        }
        
        if start_time:
            params['start_time'] = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/tweets/search/recent", 
                                   headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    tweets = self._parse_tweets(data)
                    logger.info(f"Fetched {len(tweets)} tweets for query: {query}")
                    return tweets
                else:
                    logger.error(f"Twitter API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Failed to search tweets: {e}")
            return []
    
    async def get_tweets_by_symbol(self, symbol: str, 
                                   max_results: int = 100) -> List[Tweet]:
        """
        Get tweets mentioning a trading symbol
        
        Args:
            symbol: Trading symbol (e.g., BTC, ETH)
            max_results: Maximum number of tweets
        
        Returns:
            List of tweets
        """
        # Create search query for symbol with cashtag or hashtag
        query = f"${symbol} OR #{symbol} OR {symbol} crypto"
        
        start_time = datetime.now() - timedelta(hours=24)
        return await self.search_tweets(query, max_results, start_time)
    
    def _parse_tweets(self, data: Dict) -> List[Tweet]:
        """Parse Twitter API response into Tweet objects"""
        tweets = []
        
        # Extract user info
        users = {}
        if 'includes' in data and 'users' in data['includes']:
            for user in data['includes']['users']:
                users[user['id']] = user.get('username', 'unknown')
        
        for tweet_data in data.get('data', []):
            metrics = tweet_data.get('public_metrics', {})
            
            tweet = Tweet(
                id=tweet_data.get('id', ''),
                text=tweet_data.get('text', ''),
                author=users.get(tweet_data.get('author_id', ''), 'unknown'),
                created_at=datetime.fromisoformat(tweet_data.get('created_at', '').replace('Z', '+00:00')),
                like_count=metrics.get('like_count', 0),
                retweet_count=metrics.get('retweet_count', 0),
                reply_count=metrics.get('reply_count', 0)
            )
            
            tweets.append(tweet)
        
        return tweets
    
    async def analyze_sentiment(self, tweets: List[Tweet]) -> List[Tweet]:
        """
        Analyze sentiment of tweets
        
        Args:
            tweets: List of tweets
        
        Returns:
            Tweets with sentiment scores
        """
        for tweet in tweets:
            sentiment = self._analyze_text_sentiment(tweet.text)
            tweet.sentiment_score = sentiment
        
        return tweets
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of tweet text
        
        Args:
            text: Tweet text
        
        Returns:
            Sentiment score (-1 to 1)
        """
        # Bullish keywords
        bullish = {
            'bullish', 'moon', 'pump', 'surge', 'rally', 'breakout', 'buy',
            'long', 'up', 'green', 'profit', 'gain', 'high', 'support'
        }
        
        # Bearish keywords
        bearish = {
            'bearish', 'dump', 'crash', 'drop', 'fall', 'breakdown', 'sell',
            'short', 'down', 'red', 'loss', 'low', 'resistance', 'fear'
        }
        
        text_lower = text.lower()
        bullish_count = sum(1 for word in bullish if word in text_lower)
        bearish_count = sum(1 for word in bearish if word in text_lower)
        
        # Weight by engagement (likes + retweets)
        total = bullish_count + bearish_count
        if total == 0:
            return 0
        
        raw_score = (bullish_count - bearish_count) / total
        
        # Normalize to [-1, 1]
        return max(-1, min(1, raw_score))
    
    async def get_market_sentiment(self, symbol: str) -> Dict:
        """
        Get overall market sentiment for a symbol
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Sentiment analysis results
        """
        tweets = await self.get_tweets_by_symbol(symbol, max_results=200)
        
        if not tweets:
            return {
                'symbol': symbol,
                'sentiment_score': 0,
                'confidence': 0,
                'tweet_count': 0,
                'sentiment': 'NEUTRAL'
            }
        
        # Analyze sentiment
        tweets = await self.analyze_sentiment(tweets)
        
        # Calculate weighted sentiment (by engagement)
        total_weight = 0
        weighted_sentiment = 0
        
        for tweet in tweets:
            weight = 1 + tweet.like_count * 0.5 + tweet.retweet_count * 0.3
            weighted_sentiment += tweet.sentiment_score * weight
            total_weight += weight
        
        avg_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0
        
        # Calculate confidence based on tweet volume
        confidence = min(0.95, len(tweets) / 500)
        
        # Determine sentiment classification
        if avg_sentiment > 0.3:
            sentiment = 'BULLISH'
        elif avg_sentiment < -0.3:
            sentiment = 'BEARISH'
        else:
            sentiment = 'NEUTRAL'
        
        return {
            'symbol': symbol,
            'sentiment_score': avg_sentiment,
            'confidence': confidence,
            'tweet_count': len(tweets),
            'sentiment': sentiment,
            'timestamp': datetime.now().isoformat()
        }
    
    async def stream_tweets(self, query: str, callback: callable):
        """
        Stream tweets matching query (filtered stream)
        
        Args:
            query: Filter query
            callback: Async callback for each tweet
        """
        if not self.bearer_token:
            logger.warning("Twitter API not configured")
            return
        
        headers = {
            'Authorization': f'Bearer {self.bearer_token}'
        }
        
        params = {
            'tweet.fields': 'created_at,public_metrics'
        }
        
        # This would use Twitter's filtered stream API
        # Simplified for now
        logger.info(f"Starting tweet stream for query: {query}")
        
        try:
            session = await self._get_session()
            # In production, use filtered stream endpoint
            # async with session.get(f"{self.base_url}/tweets/search/stream", 
            #                       headers=headers, params=params) as response:
            #     async for line in response.content:
            #         if line:
            #             tweet = self._parse_tweet_stream(line)
            #             await callback(tweet)
            pass
        except Exception as e:
            logger.error(f"Tweet stream error: {e}")
    
    async def close(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()