"""
Main RAG Pipeline Entry Point

This module implements the complete RAG (Retrieval-Augmented Generation) pipeline
for insurance policy evaluation, following Clean Code and SOLID principles.

Key Features:
- User query input and parsing
- Vector database retrieval (top 3 chunks)
- Gemini API integration with detailed token usage tracking
- Clean separation of concerns with single responsibility components
"""

import os
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from google import genai
from dotenv import load_dotenv

# Import our components
from src.components.vector_storage import VectorStorage, VectorStorageConfig
from src.schemas.models import DocumentChunk, RetrievedContext


# ===== CONFIGURATION =====
class RAGConfig:
    """
    Configuration for the RAG pipeline.
    
    Follows Single Responsibility Principle by containing only
    RAG pipeline related configuration.
    """
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        load_dotenv()
        
        # Get API key and validate
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        # Gemini API Configuration
        self.gemini_api_key: str = api_key
        self.gemini_model = "gemini-2.5-flash-lite"
        
        # Vector Database Configuration
        self.chroma_db_path = "./chroma_db"
        self.collection_name = "policy_documents"
        
        # Retrieval Configuration
        self.top_k_chunks = 7


# ===== GEMINI CLIENT WRAPPER =====
class GeminiClient:
    """
    Wrapper for Google Gemini API client with enhanced error handling.
    
    Follows Single Responsibility Principle by focusing solely on
    Gemini API interactions and token tracking.
    """
    
    def __init__(self, api_key: str, model: str):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Gemini API key
            model: Model name to use
        """
        self.api_key = api_key
        self.model = model
        self.client = genai.Client()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
    
    def generate_response(self, prompt: str) -> Dict[str, Any]:
        """
        Generate response using Gemini API with detailed token tracking.
        
        Args:
            prompt: The complete prompt to send to the model
            
        Returns:
            Dictionary containing response text and detailed usage metadata
            
        Raises:
            Exception: If API call fails
        """
        try:
            start_time = datetime.now()
            
            # Make API call
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Extract usage metadata
            usage_metadata = response.usage_metadata if hasattr(response, 'usage_metadata') else None
            
            # Prepare detailed response
            result = {
                "response_text": response.text,
                "processing_time": processing_time,
                "model_used": self.model,
                "timestamp": end_time.isoformat()
            }
            
            # Add detailed token usage if available
            if usage_metadata:
                result["token_usage"] = {
                    "prompt_token_count": getattr(usage_metadata, 'prompt_token_count', 0),
                    "candidates_token_count": getattr(usage_metadata, 'candidates_token_count', 0),
                    "total_token_count": getattr(usage_metadata, 'total_token_count', 0)
                }
                
                # Log token usage
                self.logger.info(f"Token Usage - Input: {result['token_usage']['prompt_token_count']}, "
                               f"Output: {result['token_usage']['candidates_token_count']}, "
                               f"Total: {result['token_usage']['total_token_count']}")
            else:
                result["token_usage"] = {
                    "prompt_token_count": 0,
                    "candidates_token_count": 0,
                    "total_token_count": 0,
                    "note": "Usage metadata not available"
                }
                self.logger.warning("Token usage metadata not available in response")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Gemini API call failed: {e}")
            raise Exception(f"Failed to generate response: {e}")


# ===== RETRIEVAL COMPONENT =====
class DocumentRetriever:
    """
    Handles document retrieval from vector database.
    
    Follows Single Responsibility Principle by focusing solely on
    document retrieval operations.
    """
    
    def __init__(self, vector_storage: VectorStorage):
        """
        Initialize retriever with vector storage.
        
        Args:
            vector_storage: Configured VectorStorage instance
        """
        self.vector_storage = vector_storage
        self.logger = logging.getLogger(__name__)
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> RetrievedContext:
        """
        Retrieve most relevant document chunks for a query.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            
        Returns:
            RetrievedContext with retrieved chunks and scores
        """
        try:
            self.logger.info(f"Retrieving top {top_k} chunks for query: '{query[:50]}...'")
            
            # Search for similar chunks
            search_results = self.vector_storage.search_similar_chunks(
                query=query,
                n_results=top_k
            )
            
            # Convert to DocumentChunk objects
            chunks = []
            relevance_scores = []
            
            for result in search_results:
                # Create DocumentChunk from search result
                from src.schemas.models import ChunkMetadata
                
                metadata = ChunkMetadata(
                    source_document=result["metadata"].get("source_document", "unknown"),
                    clause_id=result["metadata"].get("clause_id", "unknown"),
                    chunk_index=result["metadata"].get("chunk_index", 0),
                    original_text=result["content"],
                    char_count=result["metadata"].get("char_count", len(result["content"])),
                    page_number=result["metadata"].get("page_number")
                )
                
                chunk = DocumentChunk(
                    content=result["content"],
                    metadata=metadata,
                    embedding=None  # Embedding not needed for retrieval results
                )
                
                chunks.append(chunk)
                relevance_scores.append(result["similarity_score"])
            
            retrieved_context = RetrievedContext(
                chunks=chunks,
                relevance_scores=relevance_scores,
                total_retrieved=len(chunks)
            )
            
            self.logger.info(f"Successfully retrieved {len(chunks)} relevant chunks")
            return retrieved_context
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve chunks: {e}")
            # Return empty context on failure
            return RetrievedContext(
                chunks=[],
                relevance_scores=[],
                total_retrieved=0
            )


# ===== PROMPT TEMPLATE MANAGER =====
class PromptTemplate:
    """
    Manages prompt templates for the RAG pipeline.
    
    Follows Single Responsibility Principle by handling only
    prompt construction and formatting.
    """
    
    @staticmethod
    def create_policy_evaluation_prompt(query: str, context: RetrievedContext) -> str:
        """
        Create a structured prompt for policy evaluation.
        
        Args:
            query: User query
            context: Retrieved context from vector database
            
        Returns:
            Formatted prompt string
        """
        # System instruction
        system_prompt = """You are an expert insurance policy evaluator. Your role is to analyze insurance claims based on policy documents and provide clear, accurate decisions.

INSTRUCTIONS:
1. Analyze the user's query against the provided policy context
2. Determine if the claim should be approved, rejected, or partially approved
3. Provide specific justification referencing exact policy clauses
4. Be precise and cite sources accurately
5. Keep your responses to the point and short.
6. Answer what is asked, and nothing more.

OUTPUT FORMAT:
Provide your response as a structured analysis with:
- Decision: [approved/rejected/partial]
- Amount: [if applicable]
- Reasoning: Clear explanation with specific policy references"""

        # Format context
        context_section = "\n=== POLICY CONTEXT ===\n"
        if context.chunks:
            for i, chunk in enumerate(context.chunks, 1):
                context_section += f"\n--- Context {i} (Relevance: {context.relevance_scores[i-1]:.3f}) ---\n"
                context_section += f"Source: {chunk.metadata.source_document}\n"
                context_section += f"Clause ID: {chunk.metadata.clause_id}\n"
                context_section += f"Content: {chunk.content}\n"
        else:
            context_section += "\nNo relevant policy context found.\n"
        
        # User query section
        query_section = f"\n=== USER QUERY ===\n{query}\n"
        
        # Instructions section
        instructions_section = "\n=== EVALUATION REQUEST ===\nPlease evaluate this query against the provided policy context and provide your decision with detailed justification."
        
        # Combine all sections
        full_prompt = system_prompt + context_section + query_section + instructions_section
        
        return full_prompt


# ===== MAIN RAG PIPELINE =====
class RAGPipeline:
    """
    Main RAG pipeline orchestrator.
    
    Follows Single Responsibility Principle by coordinating the
    different components without implementing their core logic.
    """
    
    def __init__(self, config: RAGConfig):
        """
        Initialize RAG pipeline with configuration.
        
        Args:
            config: RAG configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        try:
            # Initialize vector storage
            vector_config = VectorStorageConfig(
                chroma_db_path=self.config.chroma_db_path,
                collection_name=self.config.collection_name
            )
            self.vector_storage = VectorStorage(vector_config)
            
            # Initialize retriever
            self.retriever = DocumentRetriever(self.vector_storage)
            
            # Initialize Gemini client
            self.gemini_client = GeminiClient(
                api_key=self.config.gemini_api_key,
                model=self.config.gemini_model
            )
            
            self.logger.info("RAG pipeline components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Process a user query through the complete RAG pipeline.
        
        Args:
            user_query: Natural language query from user
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            start_time = datetime.now()
            
            self.logger.info(f"Processing query: '{user_query[:100]}...'")
            
            # Step 1: Retrieve relevant context
            self.logger.info("Step 1: Retrieving relevant document chunks...")
            retrieved_context = self.retriever.retrieve_relevant_chunks(
                query=user_query,
                top_k=self.config.top_k_chunks
            )
            
            # Step 2: Create prompt
            self.logger.info("Step 2: Creating evaluation prompt...")
            prompt = PromptTemplate.create_policy_evaluation_prompt(
                query=user_query,
                context=retrieved_context
            )
            
            # Step 3: Generate response using Gemini
            self.logger.info("Step 3: Generating response with Gemini API...")
            gemini_response = self.gemini_client.generate_response(prompt)
            
            # Calculate total processing time
            end_time = datetime.now()
            total_processing_time = (end_time - start_time).total_seconds()
            
            # Prepare final response
            result = {
                "query": user_query,
                "response": gemini_response["response_text"],
                "retrieved_chunks_count": retrieved_context.total_retrieved,
                "retrieved_chunks": [
                    {
                        "source": chunk.metadata.source_document,
                        "clause_id": chunk.metadata.clause_id,
                        "relevance_score": retrieved_context.relevance_scores[i],
                        "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
                    }
                    for i, chunk in enumerate(retrieved_context.chunks)
                ],
                "token_usage": gemini_response["token_usage"],
                "processing_times": {
                    "total_seconds": total_processing_time,
                    "gemini_api_seconds": gemini_response["processing_time"]
                },
                "model_used": gemini_response["model_used"],
                "timestamp": end_time.isoformat()
            }
            
            self.logger.info(f"Query processed successfully in {total_processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process query: {e}")
            return {
                "query": user_query,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG pipeline.
        
        Returns:
            Dictionary with pipeline statistics
        """
        try:
            vector_stats = self.vector_storage.get_collection_stats()
            
            return {
                "vector_database": vector_stats,
                "configuration": {
                    "top_k_chunks": self.config.top_k_chunks,
                    "gemini_model": self.config.gemini_model,
                    "collection_name": self.config.collection_name
                }
            }
        except Exception as e:
            self.logger.error(f"Failed to get pipeline stats: {e}")
            return {"error": str(e)}


# ===== USER INTERFACE FUNCTIONS =====
def display_welcome_message():
    """Display welcome message and instructions."""
    print("\n" + "="*60)
    print("🤖 INSURANCE POLICY RAG EVALUATION SYSTEM")
    print("="*60)
    print("\nWelcome! This system helps evaluate insurance claims against policy documents.")
    print("\nInstructions:")
    print("• Ask questions about insurance coverage, claims, procedures, etc.")
    print("• The system will find relevant policy sections and provide evaluations")
    print("• Type 'quit', 'exit', or 'q' to end the session")
    print("• Type 'stats' to see system statistics")
    print("\nExample queries:")
    print("• 'I am 35 years old and need knee surgery. Is this covered?'")
    print("• 'What is the coverage limit for cardiac procedures?'")
    print("• 'Are maternity expenses covered under this policy?'")
    print("\n" + "="*60 + "\n")


def display_response(result: Dict[str, Any]):
    """
    Display the RAG pipeline response in a user-friendly format.
    
    Args:
        result: Response from RAG pipeline
    """
    if "error" in result:
        print(f"\n❌ Error: {result['error']}\n")
        return
    
    print("\n" + "="*60)
    print("📋 POLICY EVALUATION RESULT")
    print("="*60)
    
    # Display main response
    print(f"\n💬 Response:")
    print("-" * 40)
    print(result["response"])
    
    # Display retrieval information
    print(f"\n🔍 Retrieved Information:")
    print("-" * 40)
    print(f"Chunks found: {result['retrieved_chunks_count']}")
    
    if result.get("retrieved_chunks"):
        for i, chunk in enumerate(result["retrieved_chunks"], 1):
            print(f"\n  📄 Source {i}: {chunk['source']}")
            print(f"     Clause: {chunk['clause_id']}")
            print(f"     Relevance: {chunk['relevance_score']:.3f}")
            print(f"     Preview: {chunk['content_preview']}")
    
    # Display token usage
    print(f"\n📊 Token Usage:")
    print("-" * 40)
    token_usage = result.get("token_usage", {})
    print(f"Input tokens: {token_usage.get('prompt_token_count', 'N/A')}")
    print(f"Output tokens: {token_usage.get('candidates_token_count', 'N/A')}")
    print(f"Total tokens: {token_usage.get('total_token_count', 'N/A')}")
    
    # Display performance metrics
    print(f"\n⏱️ Performance:")
    print("-" * 40)
    processing_times = result.get("processing_times", {})
    print(f"Total time: {processing_times.get('total_seconds', 'N/A'):.2f}s")
    print(f"API call time: {processing_times.get('gemini_api_seconds', 'N/A'):.2f}s")
    print(f"Model used: {result.get('model_used', 'N/A')}")
    
    print("\n" + "="*60 + "\n")


def display_stats(pipeline: RAGPipeline):
    """
    Display pipeline statistics.
    
    Args:
        pipeline: RAG pipeline instance
    """
    print("\n" + "="*60)
    print("📈 SYSTEM STATISTICS")
    print("="*60)
    
    stats = pipeline.get_pipeline_stats()
    
    if "error" in stats:
        print(f"\n❌ Error getting stats: {stats['error']}\n")
        return
    
    # Vector database stats
    vector_stats = stats.get("vector_database", {})
    print(f"\n🗄️ Vector Database:")
    print("-" * 40)
    print(f"Collection: {vector_stats.get('collection_name', 'N/A')}")
    print(f"Total chunks: {vector_stats.get('total_chunks', 'N/A')}")
    print(f"Embedding model: {vector_stats.get('embedding_model', 'N/A')}")
    print(f"Distance metric: {vector_stats.get('distance_metric', 'N/A')}")
    
    # Configuration
    config = stats.get("configuration", {})
    print(f"\n⚙️ Configuration:")
    print("-" * 40)
    print(f"Retrieval chunks: {config.get('top_k_chunks', 'N/A')}")
    print(f"Gemini model: {config.get('gemini_model', 'N/A')}")
    
    print("\n" + "="*60 + "\n")


# ===== MAIN FUNCTION =====
def main():
    """
    Main function to run the interactive RAG pipeline.
    
    Provides a clean command-line interface for users to interact
    with the insurance policy evaluation system.
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize configuration
        config = RAGConfig()
        
        # Initialize RAG pipeline
        print("🚀 Initializing RAG Pipeline...")
        pipeline = RAGPipeline(config)
        print("✅ Pipeline initialized successfully!")
        
        # Display welcome message
        display_welcome_message()
        
        # Main interaction loop
        while True:
            try:
                # Get user input
                user_query = input("💬 Enter your insurance query: ").strip()
                
                # Handle special commands
                if user_query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thank you for using the Insurance Policy RAG System!")
                    break
                elif user_query.lower() == 'stats':
                    display_stats(pipeline)
                    continue
                elif not user_query:
                    print("⚠️ Please enter a valid query.")
                    continue
                
                # Process the query
                print("🔄 Processing your query...")
                result = pipeline.process_query(user_query)
                
                # Display the response
                display_response(result)
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ An error occurred: {e}")
                print("Please try again with a different query.\n")
    
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        print("Please check your configuration and try again.")


if __name__ == "__main__":
    main()
