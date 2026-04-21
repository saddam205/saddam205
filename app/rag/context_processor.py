"""
context_processor.py
Part of the app/rag module.
Context processing and prompt building for RAG.
"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ContextProcessor:
    """
    Processes retrieved context for optimal RAG performance
    """
    
    def __init__(self, max_context_length: int = 2000):
        """
        Initialize context processor
        
        Args:
            max_context_length: Maximum context length in characters
        """
        self.max_context_length = max_context_length
        
    def process_context(self, retrieved_docs: List[Dict]) -> str:
        """
        Process and combine retrieved documents into context
        
        Args:
            retrieved_docs: List of retrieved documents
        
        Returns:
            Processed context string
        """
        if not retrieved_docs:
            return ""
        
        context_parts = []
        total_length = 0
        
        for i, doc in enumerate(retrieved_docs):
            # Format document with metadata
            doc_text = f"[Source: {doc['metadata'].get('category', 'general')}]\n{doc['content']}"
            
            # Check length limit
            if total_length + len(doc_text) > self.max_context_length:
                # Truncate if needed
                remaining = self.max_context_length - total_length
                if remaining > 100:
                    doc_text = doc_text[:remaining] + "..."
                else:
                    break
            
            context_parts.append(doc_text)
            total_length += len(doc_text)
        
        return "\n\n---\n\n".join(context_parts)
    
    def extract_key_insights(self, retrieved_docs: List[Dict]) -> List[str]:
        """
        Extract key insights from retrieved documents
        
        Args:
            retrieved_docs: List of retrieved documents
        
        Returns:
            List of key insights
        """
        insights = []
        
        # Keywords that indicate important insights
        insight_indicators = [
            'key point', 'important', 'critical', 'warning', 'signal',
            'strategy', 'rule', 'principle', 'always', 'never',
            'significant', 'notable', 'remember'
        ]
        
        for doc in retrieved_docs:
            content = doc['content'].lower()
            sentences = content.split('.')
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:
                    continue
                
                # Check if sentence contains insight indicators
                if any(indicator in sentence for indicator in insight_indicators):
                    # Also check relevance score
                    if doc.get('relevance_score', 0) > 0.6:
                        insights.append(sentence.capitalize())
        
        # Remove duplicates and limit
        unique_insights = list(dict.fromkeys(insights))
        return unique_insights[:10]
    
    def calculate_context_relevance(self, query: str, context: str) -> float:
        """
        Calculate relevance of context to query
        
        Args:
            query: Original query
            context: Retrieved context
        
        Returns:
            Relevance score (0-1)
        """
        if not context:
            return 0.0
        
        # Simple keyword overlap
        query_words = set(query.lower().split())
        context_words = set(context.lower().split())
        
        overlap = len(query_words & context_words)
        relevance = overlap / len(query_words) if query_words else 0
        
        return min(1.0, relevance)


class PromptBuilder:
    """
    Builds prompts for different RAG scenarios
    """
    
    def __init__(self, system_prompt: str = None):
        """
        Initialize prompt builder
        
        Args:
            system_prompt: Custom system prompt
        """
        self.system_prompt = system_prompt or self._default_system_prompt()
        
    def _default_system_prompt(self) -> str:
        """Default system prompt for trading assistant"""
        return """You are an expert trading AI assistant with deep knowledge of financial markets, 
technical analysis, risk management, and trading strategies. Use the provided context to answer 
queries accurately and provide actionable trading insights. Always consider risk management 
principles and avoid giving absolute predictions. Focus on probabilities and risk-reward analysis."""
    
    def build_trading_prompt(self, query: str, context: str, 
                            market_data: Dict = None) -> str:
        """
        Build prompt for trading queries
        
        Args:
            query: User query
            context: Retrieved context
            market_data: Current market data (optional)
        
        Returns:
            Complete prompt
        """
        prompt_parts = [f"System: {self.system_prompt}"]
        
        # Add context
        if context:
            prompt_parts.append(f"\nRelevant Knowledge:\n{context}")
        
        # Add market data if available
        if market_data:
            market_summary = self._format_market_data(market_data)
            prompt_parts.append(f"\nCurrent Market Conditions:\n{market_summary}")
        
        # Add query
        prompt_parts.append(f"\nUser Query: {query}")
        
        # Add instruction
        prompt_parts.append("\nProvide a clear, actionable response based on the knowledge provided.")
        
        return "\n".join(prompt_parts)
    
    def build_signal_prompt(self, signals: Dict, context: str) -> str:
        """
        Build prompt for signal interpretation
        
        Args:
            signals: Trading signals with confidence
            context: Retrieved context
        
        Returns:
            Signal interpretation prompt
        """
        prompt = f"""Based on the following trading signals and market knowledge, provide an interpretation:

Trading Signals:
- Signal: {signals.get('signal', 'HOLD')}
- Confidence: {signals.get('confidence', 0):.1%}
- Uncertainty: {signals.get('uncertainty', 0):.3f}

Relevant Knowledge:
{context}

Provide:
1. Signal strength assessment
2. Key factors supporting this signal
3. Potential risks to consider
4. Suggested position sizing adjustment"""
        
        return prompt
    
    def build_risk_prompt(self, portfolio: Dict, context: str) -> str:
        """
        Build prompt for risk assessment
        
        Args:
            portfolio: Portfolio data
            context: Retrieved context
        
        Returns:
            Risk assessment prompt
        """
        prompt = f"""Analyze the risk profile of this portfolio:

Portfolio:
- Total Value: ${portfolio.get('total_value', 0):,.2f}
- Open Positions: {portfolio.get('open_positions', 0)}
- Current Drawdown: {portfolio.get('current_drawdown', 0):.1f}%
- Win Rate: {portfolio.get('win_rate', 0):.1f}%

Risk Management Knowledge:
{context}

Provide:
1. Risk assessment (Low/Medium/High)
2. Identified risk factors
3. Suggested risk mitigation actions
4. Position sizing recommendations"""
        
        return prompt
    
    def build_market_analysis_prompt(self, symbol: str, market_data: Dict, 
                                     context: str) -> str:
        """
        Build prompt for market analysis
        
        Args:
            symbol: Trading symbol
            market_data: Market data
            context: Retrieved context
        
        Returns:
            Market analysis prompt
        """
        prompt = f"""Provide a comprehensive market analysis for {symbol}:

Current Market Data:
- Price: ${market_data.get('price', 0):.2f}
- 24h Change: {market_data.get('change_24h', 0):.2f}%
- Volume: ${market_data.get('volume', 0):,.0f}
- RSI: {market_data.get('rsi', 50):.1f}
- Volatility: {market_data.get('volatility', 0):.2%}

Technical Analysis Knowledge:
{context}

Provide:
1. Current trend analysis
2. Key support and resistance levels
3. Technical indicators interpretation
4. Short-term outlook (1-7 days)
5. Key levels to watch"""
        
        return prompt
    
    def _format_market_data(self, market_data: Dict) -> str:
        """Format market data for prompt"""
        lines = []
        for key, value in market_data.items():
            if isinstance(value, float):
                if 'pct' in key or 'change' in key:
                    lines.append(f"- {key}: {value:.2f}%")
                elif 'price' in key:
                    lines.append(f"- {key}: ${value:.2f}")
                else:
                    lines.append(f"- {key}: {value:.4f}")
            else:
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)
    
    def build_chat_prompt(self, message: str, history: List[Dict], 
                          context: str) -> str:
        """
        Build prompt for chat conversation
        
        Args:
            message: Current message
            history: Conversation history
            context: Retrieved context
        
        Returns:
            Chat prompt
        """
        prompt_parts = [f"System: {self.system_prompt}"]
        
        if context:
            prompt_parts.append(f"\nRelevant Knowledge:\n{context}")
        
        # Add conversation history
        if history:
            history_text = "\n".join([
                f"{'User' if h['role'] == 'user' else 'Assistant'}: {h['content']}"
                for h in history[-10:]  # Last 10 messages
            ])
            prompt_parts.append(f"\nConversation History:\n{history_text}")
        
        # Add current message
        prompt_parts.append(f"\nUser: {message}")
        prompt_parts.append("\nAssistant:")
        
        return "\n".join(prompt_parts)
    
    def get_system_prompt(self) -> str:
        """Get current system prompt"""
        return self.system_prompt
    
    def update_system_prompt(self, new_prompt: str):
        """Update system prompt"""
        self.system_prompt = new_prompt
        logger.info("System prompt updated")