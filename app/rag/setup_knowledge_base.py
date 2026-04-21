# setup_knowledge_base.py
# One-click setup for knowledge base

import subprocess
import sys
import os
from pathlib import Path

def setup_knowledge_base():
    """Complete setup for knowledge base"""
    
    print("=" * 60)
    print("📚 Setting up Knowledge Base for RAG System")
    print("=" * 60)
    
    # Step 1: Install dependencies
    print("\n📦 Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers", "faiss-cpu", "numpy"])
    
    # Step 2: Generate embeddings
    print("\n🔧 Generating embeddings...")
    from data.knowledge.generate_embeddings import generate_embeddings
    generate_embeddings()
    
    # Step 3: Create FAISS index
    print("\n🔧 Creating FAISS index...")
    from data.knowledge.generate_faiss_index import create_faiss_index
    create_faiss_index()
    
    # Step 4: Verify setup
    print("\n✅ Verifying setup...")
    from data.knowledge.generate_faiss_index import test_faiss_search
    test_faiss_search()
    
    print("\n" + "=" * 60)
    print("🎉 Knowledge Base Setup Complete!")
    print("=" * 60)
    print("\nFiles created:")
    print("  - data/knowledge/documents.json (25 documents)")
    print("  - data/knowledge/embeddings.npy (embeddings array)")
    print("  - data/knowledge/vectors.faiss (FAISS index)")
    print("\nYou can now use the RAG system!")

if __name__ == "__main__":
    setup_knowledge_base()