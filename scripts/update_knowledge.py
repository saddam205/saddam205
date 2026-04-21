#!/usr/bin/env python3
"""
update_knowledge.py
Update RAG knowledge base with new documents and embeddings.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app.rag.knowledge_base import KnowledgeBase
from app.rag.embeddings import EmbeddingGenerator
from app.utils.logger import setup_logger

logger = setup_logger()


class KnowledgeUpdater:
    """Update and maintain knowledge base"""
    
    def __init__(self, knowledge_path: str = "data/knowledge/"):
        """
        Initialize knowledge updater
        
        Args:
            knowledge_path: Path to knowledge base directory
        """
        self.knowledge_path = Path(knowledge_path)
        self.kb = KnowledgeBase(str(self.knowledge_path))
        self.embedder = EmbeddingGenerator()
        
    def add_trading_document(self, title: str, content: str, category: str, tags: list = None):
        """Add a new trading document to knowledge base"""
        doc = {
            'content': content,
            'metadata': {
                'title': title,
                'category': category,
                'tags': tags or [],
                'source': 'manual',
                'created_at': datetime.now().isoformat()
            }
        }
        
        self.kb.add_document(doc['content'], doc['metadata'])
        logger.info(f"Added document: {title}")
        
    def add_batch_documents(self, documents: list):
        """Add multiple documents at once"""
        self.kb.add_documents_batch(documents)
        logger.info(f"Added {len(documents)} documents")
        
    def search_knowledge(self, query: str, top_k: int = 5):
        """Search knowledge base"""
        results = self.kb.search(query, top_k=top_k)
        
        print(f"\n🔍 Search Results for: '{query}'")
        print("-" * 50)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result['relevance_score']:.4f}")
            print(f"   Category: {result['metadata'].get('category', 'general')}")
            print(f"   Content: {result['content'][:150]}...")
            print()
        
        return results
    
    def get_statistics(self) -> dict:
        """Get knowledge base statistics"""
        return {
            'total_documents': self.kb.vector_store.get_document_count() if hasattr(self.kb.vector_store, 'get_document_count') else 0,
            'categories': self.kb.categories,
            'storage_path': str(self.knowledge_path)
        }
    
    def rebuild_index(self):
        """Rebuild FAISS index"""
        logger.info("Rebuilding FAISS index...")
        
        # Reload knowledge base
        self.kb.load()
        
        # Rebuild embeddings
        for doc in self.kb.vector_store.documents:
            doc.embedding = self.embedder.get_embedding(doc.content)
        
        self.kb.vector_store._rebuild_index()
        self.kb.save()
        
        logger.info("FAISS index rebuilt")
    
    def export_knowledge(self, output_file: str):
        """Export knowledge base to JSON"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get all documents
        documents = []
        for doc in self.kb.vector_store.documents:
            documents.append({
                'id': doc.id,
                'content': doc.content,
                'metadata': doc.metadata,
                'timestamp': doc.timestamp.isoformat()
            })
        
        export_data = {
            'export_date': datetime.now().isoformat(),
            'total_documents': len(documents),
            'documents': documents
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Knowledge base exported to {output_file}")


def add_sample_documents(updater: KnowledgeUpdater):
    """Add sample trading documents to knowledge base"""
    
    sample_docs = [
        {
            'content': """Technical Analysis: Moving averages are lagging indicators that smooth price data. 
            The 50-day and 200-day moving averages are widely followed. Golden cross (50 above 200) signals bullish trend.""",
            'metadata': {
                'title': 'Moving Averages Guide',
                'category': 'technical',
                'tags': ['moving_averages', 'trend', 'technical_analysis']
            }
        },
        {
            'content': """Risk Management: Never risk more than 1-2% of your trading capital on a single trade. 
            Use stop-loss orders to limit losses. Position sizing should be based on the distance to your stop loss.""",
            'metadata': {
                'title': 'Risk Management Principles',
                'category': 'risk',
                'tags': ['risk_management', 'position_sizing', 'stop_loss']
            }
        },
        {
            'content': """Market Regimes: Trending markets favor momentum/trend following strategies. 
            Ranging markets favor mean reversion strategies. High volatility requires smaller position sizes.""",
            'metadata': {
                'title': 'Market Regime Guide',
                'category': 'strategy',
                'tags': ['market_regime', 'strategy_selection', 'volatility']
            }
        },
        {
            'content': """Crypto Market Cycles: Typically follow 4-year cycles aligned with Bitcoin halving events. 
            Bull markets last 12-18 months, bear markets last 12-24 months.""",
            'metadata': {
                'title': 'Crypto Market Cycles',
                'category': 'market',
                'tags': ['crypto', 'market_cycles', 'bitcoin']
            }
        },
        {
            'content': """Sentiment Indicators: Fear and Greed Index measures market sentiment. 
            Extreme fear suggests capitulation and potential buying opportunity. Extreme greed suggests market top.""",
            'metadata': {
                'title': 'Sentiment Analysis Guide',
                'category': 'sentiment',
                'tags': ['sentiment', 'fear_greed', 'contrarian']
            }
        }
    ]
    
    updater.add_batch_documents(sample_docs)
    logger.info(f"Added {len(sample_docs)} sample documents")


def main():
    parser = argparse.ArgumentParser(description='Update RAG knowledge base')
    parser.add_argument('--add-doc', nargs=3, metavar=('TITLE', 'CONTENT', 'CATEGORY'), 
                        help='Add a single document')
    parser.add_argument('--add-sample', action='store_true', help='Add sample documents')
    parser.add_argument('--search', type=str, help='Search knowledge base')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild FAISS index')
    parser.add_argument('--export', type=str, help='Export knowledge base to JSON file')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--path', type=str, default='data/knowledge/', help='Knowledge base path')
    
    args = parser.parse_args()
    
    updater = KnowledgeUpdater(knowledge_path=args.path)
    
    if args.add_doc:
        title, content, category = args.add_doc
        updater.add_trading_document(title, content, category)
        updater.kb.save()
        
    elif args.add_sample:
        add_sample_documents(updater)
        updater.kb.save()
        
    elif args.search:
        updater.search_knowledge(args.search)
        
    elif args.rebuild:
        updater.rebuild_index()
        
    elif args.export:
        updater.export_knowledge(args.export)
        
    elif args.stats:
        stats = updater.get_statistics()
        print("\n📚 Knowledge Base Statistics")
        print("=" * 40)
        for key, value in stats.items():
            print(f"{key}: {value}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()