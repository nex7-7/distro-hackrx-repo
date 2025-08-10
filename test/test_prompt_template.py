"""
Test Prompt Template Component

This module tests the prompt template functionality of the RAG pipeline, which is responsible
for formatting retrieved context and user questions into effective prompts for the LLM.

Usage:
    python test_prompt_template.py
"""
import os
import sys
import unittest
from typing import List, Dict, Any, Optional

# Add parent directory to path to import components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the component to test
from components.prompt_template import create_prompt


class PromptTemplateTests(unittest.TestCase):
    """
    Test class for the prompt template component of the RAG pipeline.
    
    This class contains test methods to verify the functionality of prompt formatting
    for various types of questions and contexts.
    """
    
    def setUp(self) -> None:
        """
        Set up the test environment before each test method.
        
        This method initializes test data needed for testing.
        
        Returns:
            None
        """
        # Sample questions for testing
        self.questions = [
            "What is the waiting period for cataract surgery?",
            "Are medical expenses for an organ donor covered?",
            "What is the No Claim Discount offered in this policy?"
        ]
        
        # Sample context chunks with varying relevance scores
        self.context_chunks = [
            "The waiting period for cataract surgery under this policy is 24 months from the date of policy issuance. This applies to all insured members irrespective of their age or pre-existing conditions.",
            "Medical expenses for an organ donor are covered under this policy. This includes pre-hospitalization expenses, surgery costs, and post-hospitalization expenses for the donor.",
            "The No Claim Discount (NCD) offered in this policy is 5% of the premium for each claim-free year, up to a maximum of 20%. This discount is applied on the premium payable for the subsequent policy year.",
            "This policy covers hospitalization expenses, daycare procedures, and domiciliary treatment as per the terms and conditions mentioned."
        ]
        
        # Sample relevance scores (higher is better)
        self.relevance_scores = [0.95, 0.87, 0.76, 0.55]
    
    def test_prompt_creation_basic(self) -> None:
        """
        Test basic prompt creation.
        
        This test verifies that the create_prompt function correctly formats a basic prompt
        with a question and context chunks.
        
        Returns:
            None
            
        Raises:
            AssertionError: If the prompt creation doesn't behave as expected.
        """
        # Create a prompt with the first question and all context chunks
        prompt = create_prompt(
            self.questions[0],
            self.context_chunks,
            self.relevance_scores
        )
        
        # Verify the prompt is a non-empty string
        self.assertIsInstance(prompt, str, "Prompt should be a string")
        self.assertTrue(len(prompt) > 0, "Prompt should not be empty")
        
        # Verify the prompt contains the question
        self.assertIn(self.questions[0], prompt, "Prompt should contain the question")
        
        # Verify the prompt contains all context chunks
        for chunk in self.context_chunks:
            self.assertIn(chunk, prompt, f"Prompt should contain context: {chunk[:50]}...")
    
    def test_prompt_with_empty_context(self) -> None:
        """
        Test prompt creation with empty context.
        
        This test verifies that the create_prompt function correctly handles empty context.
        
        Returns:
            None
            
        Raises:
            AssertionError: If empty context isn't handled properly.
        """
        # Create a prompt with empty context
        prompt = create_prompt(
            self.questions[1],
            [],
            []
        )
        
        # Verify the prompt is a non-empty string
        self.assertIsInstance(prompt, str, "Prompt should be a string")
        self.assertTrue(len(prompt) > 0, "Prompt should not be empty")
        
        # Verify the prompt contains the question
        self.assertIn(self.questions[1], prompt, "Prompt should contain the question")
        
        # Verify the prompt contains a message about no context
        self.assertIn("no relevant", prompt.lower(), 
                     "Prompt should indicate that no context is available")
    
    def test_prompt_with_relevance_sorting(self) -> None:
        """
        Test prompt creation with relevance-based sorting.
        
        This test verifies that the create_prompt function correctly sorts context chunks
        based on relevance scores when specified.
        
        Returns:
            None
            
        Raises:
            AssertionError: If relevance-based sorting isn't handled properly.
        """
        # Create prompts with different sorting approaches
        prompt_with_sorting = create_prompt(
            self.questions[2],
            self.context_chunks,
            self.relevance_scores,
            sort_by_relevance=True
        )
        
        prompt_without_sorting = create_prompt(
            self.questions[2],
            self.context_chunks,
            self.relevance_scores,
            sort_by_relevance=False
        )
        
        # Verify both prompts are non-empty strings
        self.assertIsInstance(prompt_with_sorting, str, "Sorted prompt should be a string")
        self.assertIsInstance(prompt_without_sorting, str, "Unsorted prompt should be a string")
        
        # Verify both prompts contain the question
        self.assertIn(self.questions[2], prompt_with_sorting, 
                     "Sorted prompt should contain the question")
        self.assertIn(self.questions[2], prompt_without_sorting, 
                     "Unsorted prompt should contain the question")
        
        # The sorted prompt should have a different order of context than the unsorted prompt
        # if sorting has an effect (which it should with different relevance scores)
        self.assertNotEqual(prompt_with_sorting, prompt_without_sorting, 
                           "Sorting should affect the prompt structure")
    
    def test_prompt_structure(self) -> None:
        """
        Test the structure of the created prompt.
        
        This test verifies that the prompt has the expected structure with instructions,
        context, and question.
        
        Returns:
            None
            
        Raises:
            AssertionError: If the prompt structure isn't as expected.
        """
        prompt = create_prompt(
            self.questions[0],
            self.context_chunks,
            self.relevance_scores
        )
        
        # Check for common structural elements in RAG prompts
        structural_elements = [
            "question", "answer", "context", "information", "document"
        ]
        
        found_elements = sum(1 for element in structural_elements if element in prompt.lower())
        self.assertGreaterEqual(found_elements, 2, 
                               "Prompt should contain at least 2 structural elements")
        
        # Check that the prompt has multiple sections (system prompt, context, question)
        # by looking for multiple line breaks
        self.assertGreaterEqual(prompt.count("\n\n"), 1, 
                               "Prompt should have multiple sections separated by blank lines")


def main() -> None:
    """
    Main function to run the prompt template component tests.
    
    Returns:
        None
    """
    unittest.main()


if __name__ == "__main__":
    main()
