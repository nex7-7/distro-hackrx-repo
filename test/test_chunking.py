"""
Test Chunking Component

This module tests the chunking functionality of the RAG pipeline, which is
responsible for splitting text documents into semantically meaningful chunks
for better retrieval and context preservation.

Usage:
    python test_chunking.py
"""
import os
import sys
import unittest
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

# Add parent directory to path to import components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the component to test
from components.chunking import semantic_chunk_texts, simple_chunk_text


class ChunkingTests(unittest.TestCase):
    """
    Test class for the chunking component of the RAG pipeline.
    
    This class contains test methods to verify the functionality of both simple and
    semantic chunking methods.
    """
    
    def setUp(self) -> None:
        """
        Set up the test environment before each test method.
        
        This method initializes test data and models needed for testing.
        
        Returns:
            None
        """
        # Sample texts for testing
        self.short_text = "This is a short sample text for testing chunking functionality."
        
        self.medium_text = """
        Retrieval-Augmented Generation (RAG) is a technique that enhances large language models
        by allowing them to access external knowledge. This approach combines the strengths of
        retrieval-based and generation-based methods. The retrieval component fetches relevant
        information from a knowledge base, while the generation component uses this information
        to produce more accurate and contextually appropriate responses. RAG is particularly
        useful when dealing with domain-specific queries or when up-to-date information is required.
        """
        
        self.long_text = """
        Retrieval-Augmented Generation (RAG) is a technique that enhances large language models
        by allowing them to access external knowledge. This approach combines the strengths of
        retrieval-based and generation-based methods. The retrieval component fetches relevant
        information from a knowledge base, while the generation component uses this information
        to produce more accurate and contextually appropriate responses.
        
        RAG is particularly useful when dealing with domain-specific queries or when up-to-date
        information is required. By incorporating an external knowledge base, RAG helps address
        some limitations of traditional language models, such as hallucinations or outdated information.
        
        The process typically involves several steps. First, the user query is encoded into a vector
        representation. Then, similar vectors are retrieved from the knowledge base. The retrieved
        information, along with the original query, is passed to the language model to generate the final response.
        
        Implementations of RAG can vary in complexity. Some use simple keyword matching for retrieval,
        while others employ sophisticated vector search techniques. Similarly, the generation component
        might use different types of language models, from simple seq2seq models to large transformer-based architectures.
        
        One of the key advantages of RAG is its ability to cite sources. Since the generated content
        is based on retrieved information, it's possible to track and reference the original sources,
        enhancing transparency and trustworthiness.
        """
        
        # Load a small embedding model for testing
        self.model_name = 'BAAI/bge-base-en-v1.5'
        self.cache_dir = "../huggingface_cache"
        try:
            self.embedding_model = SentenceTransformer(
                self.model_name,
                device='cpu',
                cache_folder=self.cache_dir
            )
        except Exception as e:
            self.skipTest(f"Embedding model could not be loaded: {str(e)}")
    
    def test_simple_chunk_text(self) -> None:
        """
        Test the simple_chunk_text function.
        
        This test verifies that the simple chunking function correctly splits text
        into chunks based on the specified chunk size and overlap.
        
        Returns:
            None
            
        Raises:
            AssertionError: If the chunking function doesn't behave as expected.
        """
        # Test with short text and no overlap
        chunks = simple_chunk_text(self.short_text, chunk_size=10, overlap=0)
        self.assertIsInstance(chunks, list, "Result should be a list")
        self.assertTrue(all(isinstance(chunk, str) for chunk in chunks), "All chunks should be strings")
        
        # Test with medium text and 10% overlap
        chunks = simple_chunk_text(self.medium_text, chunk_size=100, overlap=10)
        self.assertGreater(len(chunks), 1, "Medium text should be split into multiple chunks")
        
        # Test with exact chunk size that fits the text
        text = "This is a test text"
        chunks = simple_chunk_text(text, chunk_size=len(text), overlap=0)
        self.assertEqual(len(chunks), 1, "Text exactly fitting chunk size should result in 1 chunk")
        self.assertEqual(chunks[0], text, "Chunk content should match original text")
        
        # Test with very large chunk size
        chunks = simple_chunk_text(self.long_text, chunk_size=10000, overlap=0)
        self.assertEqual(len(chunks), 1, "Very large chunk size should result in 1 chunk")
        
        # Test with empty text
        chunks = simple_chunk_text("", chunk_size=100, overlap=0)
        self.assertEqual(len(chunks), 0, "Empty text should result in empty list")
    
    def test_semantic_chunk_texts(self) -> None:
        """
        Test the semantic_chunk_texts function.
        
        This test verifies that the semantic chunking function correctly splits text
        based on semantic similarity and the specified parameters.
        
        Returns:
            None
            
        Raises:
            AssertionError: If the semantic chunking function doesn't behave as expected.
        """
        # Skip if model is not available
        if not hasattr(self, 'embedding_model'):
            self.skipTest("Embedding model not available")
        
        # Test with list of texts
        texts = [self.short_text, self.medium_text, self.long_text]
        chunks = semantic_chunk_texts(
            texts,
            embedding_model=self.embedding_model,
            model_name=self.model_name,
            similarity_threshold=0.8,
            min_chunk_size=3,
            max_chunk_size=12
        )
        
        self.assertIsInstance(chunks, list, "Result should be a list")
        self.assertTrue(all(isinstance(chunk, str) for chunk in chunks), "All chunks should be strings")
        self.assertGreater(len(chunks), len(texts), "Should generate more chunks than input texts")
        
        # Test with extreme similarity threshold (high)
        high_sim_chunks = semantic_chunk_texts(
            texts,
            embedding_model=self.embedding_model,
            model_name=self.model_name,
            similarity_threshold=0.99,  # Very high threshold
            min_chunk_size=3,
            max_chunk_size=12
        )
        
        # Test with extreme similarity threshold (low)
        low_sim_chunks = semantic_chunk_texts(
            texts,
            embedding_model=self.embedding_model,
            model_name=self.model_name,
            similarity_threshold=0.1,  # Very low threshold
            min_chunk_size=3,
            max_chunk_size=12
        )
        
        # High similarity should lead to more chunks than low similarity
        self.assertGreaterEqual(
            len(high_sim_chunks), 
            len(low_sim_chunks),
            "Higher similarity threshold should result in more chunks"
        )
    
    def test_semantic_chunk_edge_cases(self) -> None:
        """
        Test edge cases for the semantic_chunk_texts function.
        
        This test verifies that the semantic chunking function correctly handles
        edge cases such as empty texts, single words, and very short texts.
        
        Returns:
            None
            
        Raises:
            AssertionError: If the semantic chunking function doesn't handle edge cases properly.
        """
        # Skip if model is not available
        if not hasattr(self, 'embedding_model'):
            self.skipTest("Embedding model not available")
        
        # Test with empty list
        empty_chunks = semantic_chunk_texts(
            [],
            embedding_model=self.embedding_model,
            model_name=self.model_name,
            similarity_threshold=0.8
        )
        self.assertEqual(len(empty_chunks), 0, "Empty input should result in empty output")
        
        # Test with list of empty strings
        empty_str_chunks = semantic_chunk_texts(
            ["", "", ""],
            embedding_model=self.embedding_model,
            model_name=self.model_name,
            similarity_threshold=0.8
        )
        self.assertEqual(len(empty_str_chunks), 0, "Empty strings should be filtered out")
        
        # Test with very short texts
        short_texts = ["Hello", "World", "Test"]
        short_chunks = semantic_chunk_texts(
            short_texts,
            embedding_model=self.embedding_model,
            model_name=self.model_name,
            similarity_threshold=0.8
        )
        # Each short text should be preserved as a separate chunk
        self.assertEqual(len(short_chunks), len(short_texts), 
                         "Short texts should be preserved as separate chunks")


def main() -> None:
    """
    Main function to run the chunking component tests.
    
    Returns:
        None
    """
    unittest.main()


if __name__ == "__main__":
    main()
