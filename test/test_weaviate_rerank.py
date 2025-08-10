"""
Test Real Document Ingestion and Reranking Pipeline

This test script demonstrates:
1. Ingesting a real document from URL using the full pipeline
2. Querying with hybrid search (25 chunks)
3. Reranking with BAAI/bge-reranker-base (top 15)
4. Comparing original vs reranked results
5. Testing with actual questions about the document
"""

import os
import sys
import time
import uuid
import asyncio
from typing import List, Tuple
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Add parent directory to path to import components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.ingest_engine import ingest_from_url
from components.chunking import semantic_chunk_texts
from components.search import hybrid_search
from components.embeddings import create_embeddings, create_query_embedding
from components.weaviate_db import connect_to_weaviate, ingest_to_weaviate


# Load environment variables
load_dotenv()

# Configuration
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "weaviate")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", 8080))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", 50051))
EMBEDDING_MODEL_NAME = 'BAAI/bge-base-en-v1.5'
RERANKER_MODEL_NAME = 'BAAI/bge-reranker-base'
CACHE_DIR = "../huggingface_cache"

# Test configuration - MODIFY THESE FOR YOUR TEST
TEST_DOCUMENT_URL = "https://example.com/document.pdf"  # Replace with your document URL
TEST_QUESTIONS = [
    "What is the main topic of this document?",
    "What are the key points mentioned?",
    "What conclusions are drawn?"
]

# You can also use these example URLs for testing:
EXAMPLE_CONFIGS = {
    "research_paper": {
        "url": "https://superman-rx.s3.eu-north-1.amazonaws.com/policy.pdf", 
        "questions": [
   "What is the waiting period for cataract surgery?",
        ]
    },

}


class WeaviateRerankerTester:
    """Test class for Weaviate ingestion and reranking pipeline with real documents."""
    
    def __init__(self, document_url: str = None, questions: List[str] = None):
        """Initialize the tester with models and Weaviate client."""
        self.collection_name = f"TestCollection_{str(uuid.uuid4()).replace('-', '')[:12]}"
        self.embedding_model = None
        self.reranker_model = None
        self.weaviate_client = None
        self.document_url = document_url or TEST_DOCUMENT_URL
        self.questions = questions or TEST_QUESTIONS
        self.chunks = []
        
    async def setup(self):
        """Set up models and Weaviate connection."""
        print("🚀 Setting up test environment...")

        # Load embedding model
        print(f"📦 Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device='cpu',
            cache_folder=CACHE_DIR
        )
        print("✅ Embedding model loaded")

        # Load reranker model
        print(f"📦 Loading reranker model: {RERANKER_MODEL_NAME}")
        from sentence_transformers import CrossEncoder
        self.reranker_model = CrossEncoder(
            RERANKER_MODEL_NAME,
            device='cpu'
        )
        print("✅ Reranker model loaded")

        # Connect to Weaviate
        print(f"🔗 Connecting to Weaviate at {WEAVIATE_HOST}:{WEAVIATE_PORT}")
        self.weaviate_client = await connect_to_weaviate(
            WEAVIATE_HOST,
            WEAVIATE_PORT,
            WEAVIATE_GRPC_PORT
        )
        print("✅ Connected to Weaviate")

    async def ingest_document_from_url(self):
        """Ingest document from URL using the full pipeline."""
        print(f"\n� Processing document from URL: {self.document_url}")
        
        # 1. Ingest document using the full pipeline
        print("📥 Downloading and parsing document...")
        start_time = time.time()
        raw_chunks = ingest_from_url(self.document_url)
        parsing_time = time.time() - start_time
        print(f"✅ Parsed document into {len(raw_chunks)} raw chunks in {parsing_time:.2f}s")
        
        if not raw_chunks:
            raise ValueError("No content extracted from document")
        
        # 2. Apply semantic chunking with overlap
        print("🧩 Applying semantic chunking with 10% overlap...")
        start_time = time.time()
        self.chunks = semantic_chunk_texts(
            raw_chunks,
            embedding_model=self.embedding_model,
            model_name=EMBEDDING_MODEL_NAME,
            similarity_threshold=0.8,
            min_chunk_size=3,
            max_chunk_size=12
        )
        chunking_time = time.time() - start_time
        print(f"✅ Created {len(self.chunks)} semantic chunks in {chunking_time:.2f}s")
        
        # 3. Create embeddings
        print("🔢 Creating embeddings...")
        start_time = time.time()
        embeddings = create_embeddings(
            self.chunks, 
            self.embedding_model, 
            EMBEDDING_MODEL_NAME
        )
        embedding_time = time.time() - start_time
        print(f"✅ Created {len(embeddings)} embeddings in {embedding_time:.2f}s")
        
        # 4. Ingest to Weaviate
        print(f"📤 Pushing data to collection: {self.collection_name}")
        self.weaviate_client = await ingest_to_weaviate(
            self.weaviate_client,
            self.collection_name,
            self.chunks,
            embeddings,
            host=WEAVIATE_HOST,
            port=WEAVIATE_PORT,
            grpc_port=WEAVIATE_GRPC_PORT
        )
        print("✅ Data ingested successfully")
        
        # Show sample chunks with more detailed analysis
        print(f"\n📋 Sample chunks preview:")
        for i, chunk in enumerate(self.chunks[:3]):
            print(f"  Chunk {i+1} ({len(chunk)} chars):")
            
            # Check for table structure (from Excel files)
            if "TABLE:" in chunk[:20]:
                print("  📊 Structured Table Data Detected")
                # Show table header and first few rows
                lines = chunk.split("\n")
                header_lines = [line for line in lines[:6] if line.strip()]
                print(f"  Table Info: {' | '.join(header_lines[:2])}")
                print("  Table Preview:")
                for line in lines[6:12]:  # Show a few rows
                    if line.strip():
                        print(f"    {line[:80]}" + ("..." if len(line) > 80 else ""))
                print(f"    ... (and {len(lines) - 12} more lines)")
                
            # Check for slide structure (from PowerPoint files)
            elif chunk.startswith("Slide "):
                print("  🖼️ PowerPoint Slide Content Detected")
                sections = chunk.split("\n\n")
                print(f"  {sections[0]}")  # Slide title
                
                # Show text and image content sections
                text_section = next((s for s in sections if s.startswith("Text Content:")), None)
                image_section = next((s for s in sections if s.startswith("Image Content:")), None)
                
                if text_section:
                    text_preview = text_section.split("\n", 1)[1] if "\n" in text_section else text_section
                    print(f"  Text Content Preview: {text_preview[:100]}...")
                    
                if image_section:
                    print(f"  🖼️ Contains extracted image text ({len(image_section)} chars)")
            
            # Standard text content
            else:
                print(f"  {chunk[:200]}...")
                print(f"  ... (total {len(chunk)} characters)")

    def rerank_chunks(self, query: str, chunks: List[str]) -> Tuple[List[str], List[float]]:
        """Rerank chunks using the reranker model."""
        if not chunks:
            return [], []

        print(f"🔄 Reranking {len(chunks)} chunks...")
        start_time = time.time()

        # Create query-chunk pairs for reranking
        query_chunk_pairs = [[query, chunk] for chunk in chunks]

        # Get reranking scores
        rerank_scores = self.reranker_model.predict(query_chunk_pairs)

        # Sort by reranking scores
        ranked_indices = sorted(
            range(len(rerank_scores)),
            key=lambda i: rerank_scores[i],
            reverse=True
        )

        # Reorder chunks and scores
        reranked_chunks = [chunks[i] for i in ranked_indices]
        reranked_scores = [float(rerank_scores[i]) for i in ranked_indices]

        rerank_time = time.time() - start_time
        print(f"✅ Reranking completed in {rerank_time:.2f}s")

        return reranked_chunks, reranked_scores

    async def test_query_and_rerank(self, query: str, top_k: int = 15):
        """Test querying and reranking for a single query."""
        print(f"\n🔍 Testing query: '{query}'")

        # Create query embedding
        query_vector = create_query_embedding(
            query,
            self.embedding_model,
            EMBEDDING_MODEL_NAME
        )

        # Perform hybrid search to get 25 chunks
        print("🔎 Performing hybrid search (fetching 25 chunks)...")
        start_time = time.time()
        context_chunks, chunk_ids, chunk_scores = hybrid_search(
            question=query,
            query_vector=query_vector,
            collection_name=self.collection_name,
            weaviate_client=self.weaviate_client,
            limit=25,  # Get 25 chunks for reranking
            alpha=0.5
        )
        search_time = time.time() - start_time
        print(
            f"✅ Retrieved {len(context_chunks)} chunks in {search_time:.2f}s")

        if not context_chunks:
            print("❌ No chunks retrieved from search")
            return

        # Display original top results with better formatting for structured content
        print("\n📊 Original Top 5 Results from Hybrid Search:")
        for i, (chunk, score) in enumerate(zip(context_chunks[:5], chunk_scores[:5])):
            score_str = f"{score:.4f}" if score is not None else "N/A"
            
            # Check for different content types and display appropriately
            if chunk.startswith("TABLE:"):
                table_name = chunk.split('\n')[0].replace("TABLE:", "").strip()
                print(f"  {i+1}. Score: {score_str} | 📊 Excel Table: {table_name} ({len(chunk)} chars)")
            elif chunk.startswith("Slide "):
                slide_title = chunk.split('\n')[0]
                print(f"  {i+1}. Score: {score_str} | 🖼️ {slide_title} ({len(chunk)} chars)")
            else:
                print(f"  {i+1}. Score: {score_str} | {chunk[:100]}...")

        # Rerank all retrieved chunks
        reranked_chunks, reranked_scores = self.rerank_chunks(
            query, context_chunks)

        # Take top K after reranking
        final_chunks = reranked_chunks[:top_k]
        final_scores = reranked_scores[:top_k]

        # Display reranked top results with better formatting for structured content
        print(f"\n🏆 Top 5 Results After Reranking:")
        for i, (chunk, score) in enumerate(zip(final_chunks[:5], final_scores[:5])):
            score_str = f"{score:.4f}" if score is not None else "N/A"
            
            # Check for different content types and display appropriately
            if chunk.startswith("TABLE:"):
                table_name = chunk.split('\n')[0].replace("TABLE:", "").strip()
                print(f"  {i+1}. Score: {score_str} | 📊 Excel Table: {table_name} ({len(chunk)} chars)")
            elif chunk.startswith("Slide "):
                slide_title = chunk.split('\n')[0]
                print(f"  {i+1}. Score: {score_str} | 🖼️ {slide_title} ({len(chunk)} chars)")
            else:
                print(f"  {i+1}. Score: {score_str} | {chunk[:100]}...")

        # Show improvement metrics
        print(f"\n📈 Metrics:")
        print(f"  • Original chunks: {len(context_chunks)}")
        print(f"  • Final chunks: {len(final_chunks)}")
        
        # Handle None scores safely
        valid_original_scores = [s for s in chunk_scores if s is not None]
        valid_reranked_scores = [s for s in final_scores if s is not None]
        
        if valid_original_scores:
            print(f"  • Top original score: {max(valid_original_scores):.4f}")
        else:
            print(f"  • Top original score: N/A (no scores available)")
            
        if valid_reranked_scores:
            print(f"  • Top reranked score: {max(valid_reranked_scores):.4f}")
            print(f"  • Avg reranked score: {sum(valid_reranked_scores)/len(valid_reranked_scores):.4f}")
        else:
            print(f"  • Top reranked score: N/A")
            print(f"  • Avg reranked score: N/A")

        return {
            "query": query,
            "original_chunks": context_chunks,
            "original_scores": chunk_scores,
            "reranked_chunks": final_chunks,
            "reranked_scores": final_scores,
            "search_time": search_time
        }

    async def run_test(self):
        """Run the complete document ingestion and query test pipeline."""
        print("=" * 60)
        print("🧪 WEAVIATE RERANKING PIPELINE TEST")
        print("=" * 60)

        try:
            # Setup
            await self.setup()

            # Ingest document from URL
            await self.ingest_document_from_url()

            # Test the user's queries
            print(f"\n{'='*60}")
            print("🔍 TESTING QUERIES AGAINST DOCUMENT")
            print(f"{'='*60}")
            
            queries = input("Enter your questions (separate multiple questions with '||'): ").strip()
            if not queries:
                print("No queries provided. Using default question.")
                queries = "What is this document about?"
            
            query_list = [q.strip() for q in queries.split('||') if q.strip()]
            results = []
            
            for i, query in enumerate(query_list, 1):
                print(f"\n{'='*20} QUERY {i}/{len(query_list)} {'='*20}")
                result = await self.test_query_and_rerank(query)
                if result:
                    results.append(result)

            # Summary
            print(f"\n{'='*60}")
            print("📋 TEST SUMMARY")
            print(f"{'='*60}")
            print(f"✅ Processed document: {self.document_url}")
            print(f"📦 Collection used: {self.collection_name}")
            print(f"📚 Total chunks indexed: {len(self.chunks)}")
            print(f"❓ Queries tested: {len(results)}")

            return results

        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

        finally:
            # Cleanup
            if self.weaviate_client:
                try:
                    # Delete test collection
                    print(f"\n🧹 Cleaning up collection: {self.collection_name}")
                    self.weaviate_client.collections.delete(self.collection_name)
                    self.weaviate_client.close()
                    print("✅ Cleanup completed")
                except Exception as e:
                    print(f"⚠️ Cleanup warning: {str(e)}")

    async def run_full_test(self):
        """Run the complete test pipeline (alias for run_test)."""
        return await self.run_test()


async def main():
    """Main function to run the document ingestion and query test."""
    print("🚀 Document Ingestion and Reranking Test")
    print("=" * 50)
    
    # Get document URL from user
    document_url = input("Enter document URL (PDF, ZIP, or image): ").strip()
    if not document_url:
        print("❌ No URL provided. Exiting.")
        return
    
    # Create and run test
    tester = WeaviateRerankerTester(document_url)
    try:
        results = await tester.run_test()
        print(f"\n🎉 Test completed successfully!")
        print(f"📊 Total results: {len(results) if results else 0}")
    except Exception as e:
        print(f"💥 Test failed: {str(e)}")
        return


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
