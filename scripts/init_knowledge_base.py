import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app.rag.knowledge_base import KnowledgeBase
from app.utils.logger import logger

def init_kb():
    print("📚 Initializing RAG Knowledge Base...")
    kb = KnowledgeBase("data/knowledge/")
    # This triggers the initial build/load logic
    kb.build_trading_knowledge_base() 
    kb.save()
    print("✅ Knowledge base saved to data/knowledge/")

if __name__ == "__main__":
    init_kb()
