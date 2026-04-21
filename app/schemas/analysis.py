"""
analysis.py
Part of the app/schemas module.
Pydantic schemas for technical and sentiment analysis.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class IndicatorType(str, Enum):
    """Technical indicator types"""
    SMA = "sma"
    EMA = "ema"
    RSI = "rsi"
    MACD = "macd"
    BB = "bollinger_bands"
    ATR = "atr"
    STOCH = "stochastic"
    ADX = "adx"
    ICHIMOKU = "ichimoku"


class Timeframe(str, Enum):
    """Timeframe for analysis"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


class TechnicalAnalysisRequest(BaseModel):
    """Technical analysis request"""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: Timeframe = Field(Timeframe.H1, description="Timeframe")
    indicators: List[IndicatorType] = Field(default_factory=list, description="Indicators to calculate")
    lookback_bars: int = Field(200, ge=50, le=1000, description="Number of bars to analyze")
    include_all: bool = Field(False, description="Calculate all indicators")


class IndicatorValue(BaseModel):
    """Individual indicator value"""
    name: str
    value: float
    timestamp: datetime
    parameters: Optional[Dict[str, Any]] = None


class TechnicalAnalysisResponse(BaseModel):
    """Technical analysis response"""
    symbol: str
    timeframe: str
    current_price: float
    timestamp: datetime
    indicators: Dict[str, Any] = Field(default_factory=dict)
    signals: Dict[str, str] = Field(default_factory=dict)
    summary: str = Field(..., description="Human-readable summary")


class SentimentSource(str, Enum):
    """Sentiment data sources"""
    NEWS = "news"
    SOCIAL = "social"
    ONCHAIN = "onchain"
    TECHNICAL = "technical"
    FUNDING = "funding"


class SentimentAnalysisRequest(BaseModel):
    """Sentiment analysis request"""
    symbol: str = Field(..., description="Trading symbol")
    sources: List[SentimentSource] = Field(default_factory=list, description="Sources to analyze")
    lookback_hours: int = Field(24, ge=1, le=168, description="Lookback period")
    include_text: bool = Field(False, description="Include text samples")


class SentimentScore(BaseModel):
    """Sentiment score for a source"""
    source: str
    score: float = Field(..., ge=-1, le=1)
    confidence: float = Field(..., ge=0, le=1)
    classification: str = Field(..., description="BULLISH, BEARISH, NEUTRAL")
    text_sample: Optional[str] = None


class SentimentShift(BaseModel):
    """Sentiment shift detection"""
    shift_detected: bool
    shift_magnitude: float
    shift_direction: str
    current_sentiment: float
    previous_sentiment: float


class SentimentAnalysisResponse(BaseModel):
    """Sentiment analysis response"""
    symbol: str
    overall_score: float = Field(..., ge=-1, le=1)
    overall_sentiment: str = Field(..., description="VERY_BULLISH, BULLISH, NEUTRAL, BEARISH, VERY_BEARISH")
    confidence: float = Field(..., ge=0, le=1)
    sources_analyzed: int
    source_scores: List[SentimentScore]
    sentiment_shift: Optional[SentimentShift] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class CorrelationRequest(BaseModel):
    """Correlation analysis request"""
    symbols: List[str] = Field(..., min_items=2, max_items=20, description="Symbols to analyze")
    lookback_days: int = Field(60, ge=30, le=365, description="Lookback period")
    method: str = Field("pearson", description="pearson, spearman, kendall")


class CorrelationPair(BaseModel):
    """Correlation between two assets"""
    asset1: str
    asset2: str
    correlation: float = Field(..., ge=-1, le=1)
    p_value: float
    is_significant: bool
    relationship: str = Field(..., description="positive, negative, uncorrelated")


class CorrelationResponse(BaseModel):
    """Correlation analysis response"""
    correlation_matrix: List[List[float]]
    symbols: List[str]
    top_correlations: List[CorrelationPair]
    hedge_opportunities: List[Dict[str, Any]]
    average_correlation: float
    timestamp: datetime = Field(default_factory=datetime.now)


class MarketRegime(str, Enum):
    """Market regime types"""
    STRONG_TREND_UP = "strong_trend_up"
    WEAK_TREND_UP = "weak_trend_up"
    STRONG_TREND_DOWN = "strong_trend_down"
    WEAK_TREND_DOWN = "weak_trend_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    MEAN_REVERTING = "mean_reverting"


class RegimeDetectionResponse(BaseModel):
    """Market regime detection response"""
    symbol: str
    current_regime: MarketRegime
    confidence: float = Field(..., ge=0, le=1)
    features: Dict[str, float] = Field(..., description="Regime detection features")
    is_trending: bool
    direction: str = Field(..., description="up, down, neutral")
    transitions: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class MultiTimeframeRequest(BaseModel):
    """Multi-timeframe analysis request"""
    symbol: str = Field(..., description="Trading symbol")
    timeframes: List[Timeframe] = Field(..., min_items=2, description="Timeframes to analyze")
    lookback_bars: int = Field(200, ge=50, le=500)


class TimeframeSignal(BaseModel):
    """Signal from a single timeframe"""
    timeframe: str
    signal: str
    confidence: float
    trend: str
    momentum: str


class CombinedSignal(BaseModel):
    """Combined signal from all timeframes"""
    signal: str
    confidence: float
    aggregate_score: float
    bullish_timeframes: int
    bearish_timeframes: int


class Divergence(BaseModel):
    """Timeframe divergence"""
    type: str
    lower_tf: str
    higher_tf: str
    lower_signal: str
    higher_signal: str
    severity: str


class MultiTimeframeResponse(BaseModel):
    """Multi-timeframe analysis response"""
    symbol: str
    timeframe_signals: Dict[str, TimeframeSignal]
    combined_signal: CombinedSignal
    divergences: List[Divergence]
    trend_alignment: Dict[str, Any]
    recommendation: str
    timestamp: datetime = Field(default_factory=datetime.now)