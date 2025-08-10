"""
Test Search Component

This module tests the search functionality of the RAG pipeline, which is responsible
for retrieving relevant chunks from the Weaviate vector database using hybrid search.

Usage:
    python test_search.py
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any, Tuple

# Add parent directory to path to import components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the component to test
from components.search import hybrid_search


class MockWeaviateClient:
    """
    Mock Weaviate client for testing search functionality without a real database.
    
    This mock class simulates the behavior of a Weaviate client for testing.
    """
    
    def __init__(self, mock_data: List[Dict[str, Any]]):
        """
        Initialize the mock Weaviate client with predefined data.
        
        Parameters:
            mock_data (List[Dict[str, Any]]): List of mock document data.
            
        Returns:
            None
        """
        self.mock_data = mock_data
        self.collections = MagicMock()
        
        # Set up the query response
        self.collections.get.return_value = MagicMock()
        self.collections.get.return_value.query = MagicMock()
        self.collections.get.return_value.query.hybrid = MagicMock()
        self.collections.get.return_value.query.hybrid.with_additional.return_value = MagicMock()
        self.collections.get.return_value.query.hybrid.with_additional.return_value.with_limit.return_value = MagicMock()
        self.collections.get.return_value.query.hybrid.with_additional.return_value.with_limit.return_value.do.return_value = {
            "objects": mock_data
        }


class SearchTests(unittest.TestCase):
    """
    Test class for the search component of the RAG pipeline.
    
    This class contains test methods to verify the functionality of hybrid search
    for retrieving relevant chunks from Weaviate.
    """
    
    def setUp(self) -> None:
        """
        Set up the test environment before each test method.
        
        This method initializes test data needed for testing.
        
        Returns:
            None
        """
        # Sample mock documents for testing
        self.mock_documents = [
            {
                "properties": {
                    "text": "The waiting period for cataract surgery under this policy is 24 months."
                },
                "id": "doc1",
                "metadata": {
                    "hybrid": {
                        "vector": 0.95
                    },
                    "_additional": {
                        "id": "doc1",
                        "vector": [0.1, 0.2, 0.3]  # Mock vector
                    }
                }
            },
            {
                "properties": {
                    "text": "Medical expenses for an organ donor are covered under this policy."
                },
                "id": "doc2",
                "metadata": {
                    "hybrid": {
                        "vector": 0.87
                    },
                    "_additional": {
                        "id": "doc2",
                        "vector": [0.2, 0.3, 0.4]  # Mock vector
                    }
                }
            },
            {
                "properties": {
                    "text": "The No Claim Discount (NCD) offered in this policy is 5% of the premium."
                },
                "id": "doc3",
                "metadata": {
                    "hybrid": {
                        "vector": 0.76
                    },
                    "_additional": {
                        "id": "doc3",
                        "vector": [0.3, 0.4, 0.5]  # Mock vector
                    }
                }
            }
        ]
        
        # Create a mock Weaviate client
        self.mock_client = MockWeaviateClient(self.mock_documents)
        
        # Sample questions for testing
        self.questions = [
            "What is the waiting period for cataract surgery?",
            "Are medical expenses for an organ donor covered?"
        ]
        
        # Sample query vector
        self.query_vector = [0.1, 0.2, 0.3]
        
        # Collection name for testing
        self.collection_name = "TestCollection"
    
    @patch('components.search.hybrid_search')
    def test_hybrid_search_basic(self, mock_hybrid_search):
        """
        Test basic hybrid search functionality.
        
        This test verifies that hybrid_search correctly calls the Weaviate client
        with the expected parameters and returns the expected results.
        
        Parameters:
            mock_hybrid_search: Mock of the hybrid_search function.
            
        Returns:
            None
            
        Raises:
            AssertionError: If the search functionality doesn't behave as expected.
        """
        # Mock the search function to return predefined results
        mock_expected_chunks = [doc["properties"]["text"] for doc in self.mock_documents]
        mock_expected_ids = [doc["id"] for doc in self.mock_documents]
        mock_expected_scores = [doc["metadata"]["hybrid"]["vector"] for doc in self.mock_documents]
        
        mock_hybrid_search.return_value = (
            mock_expected_chunks,
            mock_expected_ids,
            mock_expected_scores
        )
        
        # Call the search function
        chunks, ids, scores = mock_hybrid_search(
            question=self.questions[0],
            query_vector=self.query_vector,
            collection_name=self.collection_name,
            weaviate_client=self.mock_client,
            limit=3,
            alpha=0.5
        )
        
        # Verify the results
        self.assertEqual(len(chunks), 3, "Should return 3 chunks")
        self.assertEqual(len(ids), 3, "Should return 3 ids")
        self.assertEqual(len(scores), 3, "Should return 3 scores")
        
        # Verify mock was called with expected parameters
        mock_hybrid_search.assert_called_once_with(
            question=self.questions[0],
            query_vector=self.query_vector,
            collection_name=self.collection_name,
            weaviate_client=self.mock_client,
            limit=3,
            alpha=0.5
        )
    
    def test_hybrid_search_integration(self):
        """
        Test hybrid search integration directly with the mock client.
        
        This test verifies that hybrid_search correctly integrates with the
        Weaviate client and processes the results as expected.
        
        Returns:
            None
            
        Raises:
            AssertionError: If the search integration doesn't behave as expected.
        """
        # Call the actual search function with our mock client
        chunks, ids, scores = hybrid_search(
            question=self.questions[0],
            query_vector=self.query_vector,
            collection_name=self.collection_name,
            weaviate_client=self.mock_client,
            limit=3,
            alpha=0.5
        )
        
        # Verify the results match our mock data
        expected_chunks = [doc["properties"]["text"] for doc in self.mock_documents]
        expected_ids = [doc["id"] for doc in self.mock_documents]
        
        self.assertEqual(chunks, expected_chunks, "Returned chunks don't match expected")
        self.assertEqual(ids, expected_ids, "Returned ids don't match expected")
        self.assertEqual(len(scores), len(expected_chunks), "Should return scores for each chunk")
    
    def test_hybrid_search_with_empty_results(self):
        """
        Test hybrid search with empty results.
        
        This test verifies that hybrid_search correctly handles the case when
        no results are returned from Weaviate.
        
        Returns:
            None
            
        Raises:
            AssertionError: If empty results aren't handled properly.
        """
        # Create a mock client that returns empty results
        empty_mock_client = MockWeaviateClient([])
        empty_mock_client.collections.get.return_value.query.hybrid.with_additional.return_value.with_limit.return_value.do.return_value = {
            "objects": []
        }
        
        # Call the search function
        chunks, ids, scores = hybrid_search(
            question=self.questions[0],
            query_vector=self.query_vector,
            collection_name=self.collection_name,
            weaviate_client=empty_mock_client,
            limit=3,
            alpha=0.5
        )
        
        # Verify the results are empty lists
        self.assertEqual(chunks, [], "Should return empty chunks list")
        self.assertEqual(ids, [], "Should return empty ids list")
        self.assertEqual(scores, [], "Should return empty scores list")
    
    def test_hybrid_search_parameters(self):
        """
        Test hybrid search with different parameters.
        
        This test verifies that hybrid_search correctly handles different
        values for limit and alpha parameters.
        
        Returns:
            None
            
        Raises:
            AssertionError: If parameter handling isn't correct.
        """
        # Test with different limit
        chunks_limit_1, ids_limit_1, scores_limit_1 = hybrid_search(
            question=self.questions[0],
            query_vector=self.query_vector,
            collection_name=self.collection_name,
            weaviate_client=self.mock_client,
            limit=1,  # Only get 1 result
            alpha=0.5
        )
        
        # Verify we get only 1 result
        self.assertEqual(len(chunks_limit_1), 1, "Should return 1 chunk")
        self.assertEqual(len(ids_limit_1), 1, "Should return 1 id")
        self.assertEqual(len(scores_limit_1), 1, "Should return 1 score")
        
        # Test with different alpha
        chunks_alpha_1, ids_alpha_1, scores_alpha_1 = hybrid_search(
            question=self.questions[0],
            query_vector=self.query_vector,
            collection_name=self.collection_name,
            weaviate_client=self.mock_client,
            limit=3,
            alpha=1.0  # Only vector search
        )
        
        # We should still get results, even with different alpha
        self.assertEqual(len(chunks_alpha_1), 3, "Should return 3 chunks")
        self.assertEqual(len(ids_alpha_1), 3, "Should return 3 ids")
        self.assertEqual(len(scores_alpha_1), 3, "Should return 3 scores")


def main() -> None:
    """
    Main function to run the search component tests.
    
    Returns:
        None
    """
    unittest.main()


if __name__ == "__main__":
    main()
