"""
market_intelligence.py
Part of the app/rag module.
Market intelligence generation using RAG.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from .knowledge_base import KnowledgeBase
from .retriever import DocumentRetriever, HybridRetriever
from .context_processor import ContextProcessor, PromptBuilder

logger = logging.getLogger(__name__)


@dataclass
class MarketInsight:
    """Market insight data structure"""
    title: str
    description: str
    confidence: float
    category: str
    timestamp: datetime
    supporting_evidence: List[str]
    actionable: bool
    risk_level: str  # 'low', 'medium', 'high'


class MarketIntelligence:
    """
    Generates market intelligence using RAG system
    """
    
    def __init__(self, knowledge_base: KnowledgeBase):
        """
        Initialize market intelligence
        
        Args:
            knowledge_base: Knowledge base instance
        """
        self.kb = knowledge_base
        self.retriever = HybridRetriever(knowledge_base)
        self.context_processor = ContextProcessor()
        self.prompt_builder = PromptBuilder()
        
        self.insights_history: List[MarketInsight] = []
        
    async def analyze_market_conditions(self, market_data: Dict) -> Dict:
        """
        Analyze current market conditions
        
        Args:
            market_data: Current market data
        
        Returns:
            Market analysis results
        """
        query = f"Market analysis for {market_data.get('symbol', 'crypto')} with price ${market_data.get('price', 0)}"
        
        # Retrieve relevant knowledge
        docs = self.retriever.retrieve(query, filters={'category': 'technical'}, top_k=5)
        context = self.context_processor.process_context(docs)
        
        # Extract insights
        insights = self.context_processor.extract_key_insights(docs)
        
        # Build analysis
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'symbol': market_data.get('symbol'),
            'price': market_data.get('price'),
            'context_used': len(docs),
            'key_insights': insights,
            'relevance_score': self.context_processor.calculate_context_relevance(query, context),
            'sources': [{'id': d['id'], 'relevance': d['relevance_score']} for d in docs[:3]]
        }
        
        return analysis
    
    async def generate_trading_insight(self, signal: Dict, market_data: Dict) -> MarketInsight:
        """
        Generate trading insight based on signal and market data
        
        Args:
            signal: Trading signal
            market_data: Current market data
        
        Returns:
            MarketInsight object
        """
        # Build query based on signal
        if signal.get('signal') == 'BUY':
            query = f"Buy signals and entry strategies for {market_data.get('symbol')}"
        elif signal.get('signal') == 'SELL':
            query = f"Sell signals and exit strategies for {market_data.get('symbol')}"
        else:
            query = f"Market conditions analysis for {market_data.get('symbol')}"
        
        # Retrieve relevant knowledge
        docs = self.retriever.retrieve(query, top_k=5)
        context = self.context_processor.process_context(docs)
        
        # Generate insight
        insight_text = self._generate_insight_text(signal, market_data, docs)
        
        # Determine confidence based on signal and context
        confidence = signal.get('confidence', 0.5)
        if len(docs) >= 3:
            confidence = min(0.9, confidence + 0.1)
        
        insight = MarketInsight(
            title=f"{signal.get('signal', 'HOLD')} Opportunity for {market_data.get('symbol')}",
            description=insight_text,
            confidence=confidence,
            category='trading',
            timestamp=datetime.now(),
            supporting_evidence=[d['content'][:200] for d in docs[:3]],
            actionable=signal.get('confidence', 0) > 0.6,
            risk_level=self._assess_risk_level(signal, market_data)
        )
        
        self.insights_history.append(insight)
        
        # Keep only last 100 insights
        if len(self.insights_history) > 100:
            self.insights_history.pop(0)
        
        return insight
    
    def _generate_insight_text(self, signal: Dict, market_data: Dict, 
                               docs: List[Dict]) -> str:
        """Generate insight text from signal and context"""
        signal_type = signal.get('signal', 'HOLD')
        confidence = signal.get('confidence', 0.5)
        symbol = market_data.get('symbol', 'Asset')
        price = market_data.get('price', 0)
        
        if signal_type == 'BUY':
            return f"Buy signal detected for {symbol} at ${price:.2f} with {confidence:.1%} confidence. " \
                   f"Supported by {len(docs)} relevant market intelligence documents. " \
                   f"Consider entering with appropriate position sizing and stop-loss."
        elif signal_type == 'SELL':
            return f"Sell signal detected for {symbol} at ${price:.2f} with {confidence:.1%} confidence. " \
                   f"Based on {len(docs)} supporting sources. Consider taking profits or reducing exposure."
        else:
            return f"No clear signal for {symbol} at ${price:.2f}. " \
                   f"Market conditions suggest waiting for better entry points. " \
                   f"Confidence in hold: {confidence:.1%}"
    
    def _assess_risk_level(self, signal: Dict, market_data: Dict) -> str:
        """Assess risk level for the insight"""
        confidence = signal.get('confidence', 0.5)
        volatility = market_data.get('volatility', 0.02)
        
        if confidence < 0.5 or volatility > 0.05:
            return 'high'
        elif confidence < 0.7 or volatility > 0.03:
            return 'medium'
        else:
            return 'low'
    
    async def get_market_summary(self, symbol: str) -> Dict:
        """
        Get comprehensive market summary
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Market summary dictionary
        """
        # Retrieve from multiple categories
        technical = self.retriever.retrieve(f"Technical analysis {symbol}", 
                                           filters={'category': 'technical'}, top_k=3)
        risk = self.retriever.retrieve(f"Risk management {symbol}", 
                                      filters={'category': 'risk'}, top_k=3)
        strategy = self.retriever.retrieve(f"Trading strategy {symbol}", 
                                          filters={'category': 'strategy'}, top_k=3)
        
        summary = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'technical_context': self.context_processor.process_context(technical),
            'risk_context': self.context_processor.process_context(risk),
            'strategy_context': self.context_processor.process_context(strategy),
            'key_insights': self.context_processor.extract_key_insights(technical + risk + strategy),
            'sources_analyzed': len(technical) + len(risk) + len(strategy)
        }
        
        return summary
    
    def get_recent_insights(self, limit: int = 10) -> List[Dict]:
        """
        Get recent trading insights
        
        Args:
            limit: Number of insights to return
        
        Returns:
            List of recent insights
        """
        return [
            {
                'title': i.title,
                'description': i.description,
                'confidence': i.confidence,
                'category': i.category,
                'timestamp': i.timestamp.isoformat(),
                'actionable': i.actionable,
                'risk_level': i.risk_level
            }
            for i in self.insights_history[-limit:]
        ]
    
    def get_insight_statistics(self) -> Dict:
        """
        Get insight generation statistics
        
        Returns:
            Statistics dictionary
        """
        if not self.insights_history:
            return {'total_insights': 0}
        
        actionable = [i for i in self.insights_history if i.actionable]
        
        return {
            'total_insights': len(self.insights_history),
            'actionable_insights': len(actionable),
            'actionable_rate': len(actionable) / len(self.insights_history) if self.insights_history else 0,
            'avg_confidence': sum(i.confidence for i in self.insights_history) / len(self.insights_history),
            'risk_distribution': {
                'low': len([i for i in self.insights_history if i.risk_level == 'low']),
                'medium': len([i for i in self.insights_history if i.risk_level == 'medium']),
                'high': len([i for i in self.insights_history if i.risk_level == 'high'])
            },
            'last_insight': self.insights_history[-1].timestamp.isoformat() if self.insights_history else None
        }


class AdvancedTradingRAG:
    """
    Complete RAG system for trading with knowledge base and retrieval
    """
    
    def __init__(self, storage_path: str = "data/knowledge/"):
        """
        Initialize advanced trading RAG
        
        Args:
            storage_path: Path for knowledge base storage
        """
        self.knowledge_base = KnowledgeBase(storage_path)
        self.retriever = HybridRetriever(self.knowledge_base)
        self.context_processor = ContextProcessor()
        self.prompt_builder = PromptBuilder()
        self.market_intel = MarketIntelligence(self.knowledge_base)
        
        # Try to load existing knowledge base
        try:
            self.knowledge_base.load()
            logger.info("Knowledge base loaded successfully")
        except:
            logger.info("No existing knowledge base found. Will build new one.")
    
    def build_trading_knowledge_base(self):
        """Build the trading knowledge base"""
        self.knowledge_base.build_trading_knowledge_base()
        self.knowledge_base.save()
        logger.info("Trading knowledge base built and saved")
    
    def get_market_context(self, symbol: str) -> Dict:
        """
        Get market context for trading decisions
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Market context dictionary
        """
        # Retrieve relevant documents
        docs = self.retriever.retrieve(f"Trading context for {symbol}", top_k=5)
        context = self.context_processor.process_context(docs)
        insights = self.context_processor.extract_key_insights(docs)
        
        # Determine sentiment from context
        sentiment = self._determine_sentiment(insights)
        confidence = self.context_processor.calculate_context_relevance(symbol, context)
        
        return {
            'context': context,
            'insights': insights,
            'sentiment': sentiment,
            'confidence': confidence,
            'sources': len(docs),
            'timestamp': datetime.now().isoformat()
        }
    
    def _determine_sentiment(self, insights: List[str]) -> str:
        """Determine sentiment from insights"""
        bullish_indicators = ['bullish', 'buy', 'uptrend', 'support', 'breakout', 'momentum']
        bearish_indicators = ['bearish', 'sell', 'downtrend', 'resistance', 'breakdown', 'reversal']
        
        bullish_count = sum(1 for i in insights for w in bullish_indicators if w in i.lower())
        bearish_count = sum(1 for i in insights for w in bearish_indicators if w in i.lower())
        
        if bullish_count > bearish_count:
            return 'BULLISH'
        elif bearish_count > bullish_count:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def query(self, question: str, use_context: bool = True) -> str:
        """
        Query the RAG system
        
        Args:
            question: User question
            use_context: Whether to use retrieved context
        
        Returns:
            Response string
        """
        context = ""
        if use_context:
            docs = self.retriever.retrieve(question, top_k=3)
            context = self.context_processor.process_context(docs)
        
        prompt = self.prompt_builder.build_trading_prompt(question, context)
        
        # In production, this would call an LLM
        # For now, return a formatted response
        response = f"Based on the trading knowledge base:\n\n"
        if context:
            response += f"Context: {context[:500]}...\n\n"
        response += f"Question: {question}\n\n"
        response += "Note: In production, this would be processed by an LLM with the provided context."
        
        return response