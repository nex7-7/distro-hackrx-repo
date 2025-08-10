"""
Test Embeddings Component

This module tests the embeddings functionality of the RAG pipeline, which is responsible
for converting text chunks into vector representations for semantic search.

Usage:
    python test_embeddings.py
"""
import os
import sys
import unittest
from typing import List, Dict, Any, Tuple
import math

# Add parent directory to path to import components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the component to test
from components.embeddings import create_embeddings, create_query_embedding


class MockEmbeddingModel:
    """
    Mock embedding model for testing without requiring actual ML model loading.
    
    This mock class simulates the behavior of a SentenceTransformer model for testing.
    """
    
    def __init__(self, embedding_dim: int = 384):
        """
        Initialize the mock embedding model.
        
        Parameters:
            embedding_dim (int): The dimension of the embeddings to generate.
            
        Returns:
            None
        """
        self.embedding_dim = embedding_dim
    
    def encode(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Mock encode method that generates deterministic pseudo-embeddings.
        
        This method creates unique but deterministic embeddings for each text
        based on the text's length and character values.
        
        Parameters:
            texts (List[str]): List of text strings to encode.
            **kwargs: Additional arguments (ignored in the mock).
            
        Returns:
            List[List[float]]: List of mock embedding vectors.
        """
        embeddings = []
        for text in texts:
            if not text:
                # Empty text gets zero embedding
                embeddings.append([0.0] * self.embedding_dim)
                continue
                
            # Create a deterministic but unique embedding based on text content
            seed = sum(ord(c) for c in text)
            embedding = []
            for i in range(self.embedding_dim):
                # Generate a value between -1 and 1 based on text and position
                value = math.sin(seed * (i + 1) / 100)
                embedding.append(value)
                
            # Normalize the embedding to unit length
            magnitude = math.sqrt(sum(v * v for v in embedding))
            if magnitude > 0:
                embedding = [v / magnitude for v in embedding]
            
            embeddings.append(embedding)
        
        return embeddings


class EmbeddingsTests(unittest.TestCase):
    """
    Test class for the embeddings component of the RAG pipeline.
    
    This class contains test methods to verify the functionality of embedding creation
    for both documents and queries.
    """
    
    def setUp(self) -> None:
        """
        Set up the test environment before each test method.
        
        This method initializes test data and models needed for testing.
        
        Returns:
            None
        """
        # Sample texts for testing
        self.test_texts = [
            "This is the first test text for embedding creation.",
            "Here is another test text with different content.",
            "A third example with some overlapping terms from the previous texts."
        ]
        
        # Create a mock embedding model
        self.mock_model = MockEmbeddingModel(embedding_dim=384)
        self.model_name = "mock-embedding-model"
    
    def test_create_embeddings(self) -> None:
        """
        Test the create_embeddings function.
        
        This test verifies that the function correctly creates embeddings for a list of texts.
        
        Returns:
            None
            
        Raises:
            AssertionError: If the embedding creation doesn't behave as expected.
        """
        # Create embeddings using the mock model
        embeddings = create_embeddings(
            self.test_texts,
            self.mock_model,
            self.model_name
        )
        
        # Verify the basic properties of the embeddings
        self.assertIsInstance(embeddings, list, "Result should be a list")
        self.assertEqual(len(embeddings), len(self.test_texts), 
                        f"Should create {len(self.test_texts)} embeddings")
        
        # Verify each embedding is a list of floats with the expected dimension
        for embedding in embeddings:
            self.assertIsInstance(embedding, list, "Each embedding should be a list")
            self.assertEqual(len(embedding), self.mock_model.embedding_dim, 
                            f"Each embedding should have {self.mock_model.embedding_dim} dimensions")
            self.assertTrue(all(isinstance(value, float) for value in embedding), 
                           "All embedding values should be floats")
        
        # Verify embeddings are normalized (have unit length)
        for embedding in embeddings:
            magnitude = math.sqrt(sum(v * v for v in embedding))
            self.assertAlmostEqual(magnitude, 1.0, places=5, 
                                  msg="Embeddings should be normalized to unit length")
    
    def test_create_query_embedding(self) -> None:
        """
        Test the create_query_embedding function.
        
        This test verifies that the function correctly creates an embedding for a query text.
        
        Returns:
            None
            
        Raises:
            AssertionError: If the query embedding creation doesn't behave as expected.
        """
        # Test query
        query = "What is the purpose of embeddings in RAG?"
        
        # Create a query embedding
        query_embedding = create_query_embedding(
            query,
            self.mock_model,
            self.model_name
        )
        
        # Verify the basic properties of the query embedding
        self.assertIsInstance(query_embedding, list, "Result should be a list")
        self.assertEqual(len(query_embedding), self.mock_model.embedding_dim, 
                        f"Query embedding should have {self.mock_model.embedding_dim} dimensions")
        self.assertTrue(all(isinstance(value, float) for value in query_embedding), 
                       "All embedding values should be floats")
        
        # Verify the query embedding is normalized
        magnitude = math.sqrt(sum(v * v for v in query_embedding))
        self.assertAlmostEqual(magnitude, 1.0, places=5, 
                              msg="Query embedding should be normalized to unit length")
    
    def test_empty_inputs(self) -> None:
        """
        Test embedding creation with empty inputs.
        
        This test verifies that the embedding functions correctly handle empty inputs.
        
        Returns:
            None
            
        Raises:
            AssertionError: If empty inputs aren't handled properly.
        """
        # Test with empty list
        empty_embeddings = create_embeddings(
            [],
            self.mock_model,
            self.model_name
        )
        self.assertEqual(len(empty_embeddings), 0, "Empty input should result in empty output")
        
        # Test with empty string for query
        empty_query_embedding = create_query_embedding(
            "",
            self.mock_model,
            self.model_name
        )
        self.assertEqual(len(empty_query_embedding), self.mock_model.embedding_dim, 
                        "Empty query should still produce an embedding of correct dimension")
        
        # Verify all values in empty query embedding are zeros
        self.assertTrue(all(v == 0.0 for v in empty_query_embedding), 
                       "Empty query should result in all-zero embedding")
    
    def test_consistency(self) -> None:
        """
        Test consistency of embedding creation.
        
        This test verifies that the same input text consistently produces the same embedding.
        
        Returns:
            None
            
        Raises:
            AssertionError: If embedding creation isn't consistent.
        """
        # Create embeddings twice for the same text
        text = "This is a test for embedding consistency."
        embedding1 = create_query_embedding(
            text,
            self.mock_model,
            self.model_name
        )
        
        embedding2 = create_query_embedding(
            text,
            self.mock_model,
            self.model_name
        )
        
        # Verify the embeddings are identical
        self.assertEqual(embedding1, embedding2, "Same input should produce same embedding")


def main() -> None:
    """
    Main function to run the embeddings component tests.
    
    Returns:
        None
    """
    unittest.main()


if __name__ == "__main__":
    main()
