"""
LLM service for the RAG Application API.

This module provides Google Gemini integration for Stage 3 (query classification)
and Stage 5 (response generation) of the RAG pipeline. It supports parallel LLM
calls and structured prompt templates as specified in the requirements.
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from config.settings import settings
from app.models.schemas import QueryAnalysis, QueryClassification, GeneratedResponse
from app.utils.logger import get_logger
from app.utils.exceptions import (
    LLMGenerationError,
    QueryClassificationError,
    ResponseGenerationError,
    create_error
)

logger = get_logger(__name__)


class LLMService:
    """
    Google Gemini LLM integration for query analysis and response generation.
    
    This service handles Stage 3 (query classification) and Stage 5 (response
    generation) with support for parallel processing and structured prompts.
    """
    
    def __init__(self) -> None:
        """Initialize the LLM service."""
        self.model_name = settings.llm_model
        self.api_key = settings.google_api_key
        self.executor = ThreadPoolExecutor(max_workers=settings.max_workers)
        
        # Configure Google Generative AI
        genai.configure(api_key=self.api_key)
        
        # Initialize model with safety settings
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        logger.info("LLM service initialized", model=self.model_name)
    
    async def classify_queries(
        self, 
        queries: List[str], 
        document_name: str, 
        document_context: str
    ) -> List[QueryAnalysis]:
        """
        Classify queries and extract keywords/prompt templates (Stage 3).
        
        This is the first LLM call that processes all questions at once.
        
        Args:
            queries: List of user queries
            document_name: Name of the document
            document_context: First 5 pages of document text
            
        Returns:
            List[QueryAnalysis]: Classification results for each query
            
        Raises:
            QueryClassificationError: If classification fails
        """
        start_time = time.time()
        
        logger.log_stage("Query Classification", "Starting", 
                        query_count=len(queries),
                        document=document_name)
        
        try:
            # Construct the classification prompt
            prompt = self._build_classification_prompt(
                queries, document_name, document_context
            )
            
            # Generate classification response
            response_text = await self._generate_response(prompt)
            
            # Parse JSON response
            classification_data = self._parse_classification_response(response_text)
            
            # Convert to QueryAnalysis objects
            analyses = []
            for i, query in enumerate(queries):
                query_key = f"Query {i + 1}"
                query_data = classification_data.get(query_key, {})
                
                analysis = QueryAnalysis(
                    query_id=f"q_{i}",
                    original_query=query,
                    classification=QueryClassification(query_data.get("Class", "Not From Document")),
                    keywords=query_data.get("Keywords"),
                    prompt_template=query_data.get("Prompt Template")
                )
                analyses.append(analysis)
            
            duration = time.time() - start_time
            logger.log_performance(
                "Query Classification",
                duration,
                query_count=len(queries),
                from_document_count=sum(1 for a in analyses if a.classification == QueryClassification.FROM_DOCUMENT)
            )
            
            return analyses
            
        except Exception as e:
            logger.error(f"Query classification failed: {str(e)}", 
                        query_count=len(queries))
            
            if isinstance(e, QueryClassificationError):
                raise
            
            raise create_error(
                QueryClassificationError,
                f"Failed to classify queries: {str(e)}",
                "CLASSIFICATION_FAILED",
                query_count=len(queries),
                error_type=type(e).__name__
            )
    
    def _build_classification_prompt(
        self, 
        queries: List[str], 
        document_name: str, 
        document_context: str
    ) -> str:
        """Build the classification prompt template."""
        queries_formatted = [
            f"Query {i + 1}: {query}" 
            for i, query in enumerate(queries)
        ]
        
        prompt = f"""You are an expert at analyzing user queries and extracting key phrases for document retrieval for a vector database.

TASK: For each query below, classify it and extract the most important key phrases that would be useful for searching the provided document context.

DOCUMENT CONTEXT:
- Document Name: {document_name}
- First 5 Pages: {document_context}

INSTRUCTIONS:
1. For each query, determine if it can likely be answered from the document ("From Document") or if it requires external knowledge ("Not From Document").
2. If "From Document", extract 3-5 concise but meaningful key phrases (2-5 words each) for vector search. Focus on specific terms.
3. If "Not From Document", create a new, self-contained prompt template that can be used to answer the original query using general knowledge.
4. Return a single JSON object. The keys should be "Query 1", "Query 2", etc. The value for each key will be another object containing "Class", and either "Keywords" or "Prompt Template".

QUERIES:
[
    {','.join(f'"{q}"' for q in queries_formatted)}
]

Return only the JSON object, no additional text."""
        
        return prompt
    
    def _parse_classification_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the JSON classification response."""
        try:
            # Clean the response text
            response_text = response_text.strip()
            
            # Remove any markdown code blocks
            if response_text.startswith("```"):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1])
            
            # Parse JSON
            return json.loads(response_text)
            
        except json.JSONDecodeError as e:
            logger.error("Failed to parse classification JSON", 
                        response=response_text[:200])
            
            raise create_error(
                QueryClassificationError,
                f"Invalid JSON response from LLM: {str(e)}",
                "INVALID_JSON_RESPONSE"
            )
    
    async def generate_responses(
        self, 
        analyses: List[QueryAnalysis], 
        contexts: Dict[str, str]
    ) -> List[GeneratedResponse]:
        """
        Generate responses for all queries in parallel (Stage 5).
        
        Args:
            analyses: Query analysis results from Stage 3
            contexts: Retrieved contexts keyed by query_id
            
        Returns:
            List[GeneratedResponse]: Generated responses for each query
            
        Raises:
            ResponseGenerationError: If response generation fails
        """
        start_time = time.time()
        
        logger.log_stage("Response Generation", "Starting", 
                        query_count=len(analyses))
        
        try:
            # Create tasks for parallel execution
            tasks = []
            for analysis in analyses:
                context = contexts.get(analysis.query_id, "")
                
                if analysis.classification == QueryClassification.FROM_DOCUMENT:
                    task = self._generate_document_response(analysis, context)
                else:
                    task = self._generate_general_response(analysis)
                
                tasks.append(task)
            
            # Execute all tasks in parallel
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            generated_responses = []
            for i, (analysis, response) in enumerate(zip(analyses, responses)):
                if isinstance(response, Exception):
                    logger.error(f"Response generation failed for query {i}: {str(response)}")
                    # Create fallback response
                    response = GeneratedResponse(
                        query_id=analysis.query_id,
                        original_query=analysis.original_query,
                        answer="I apologize, but I encountered an error while processing your question.",
                        context_used=False,
                        generation_method="error_fallback"
                    )
                
                generated_responses.append(response)
            
            duration = time.time() - start_time
            logger.log_performance(
                "Response Generation",
                duration,
                query_count=len(analyses),
                responses_per_second=len(analyses) / duration
            )
            
            return generated_responses
            
        except Exception as e:
            logger.error(f"Parallel response generation failed: {str(e)}")
            
            raise create_error(
                ResponseGenerationError,
                f"Failed to generate responses: {str(e)}",
                "PARALLEL_GENERATION_FAILED",
                query_count=len(analyses)
            )
    
    async def _generate_document_response(
        self, 
        analysis: QueryAnalysis, 
        context: str
    ) -> GeneratedResponse:
        """Generate response for document-based query."""
        prompt = f"""You are an expert response agent. Your role is to analyze the context provided and provide clear, accurate responses.

INSTRUCTIONS:
1. Analyze the user's query against the provided context.
2. Do not mention chunk ids or references.
3. The answer MUST be in the context provided. If there's ambiguity, assume the context is correct and answer.
4. If the Policy Name in the query does not match, assume it is the same policy as mentioned in the context.
5. You MUST provide a concise, single-paragraph answer. Your entire response MUST NOT exceed 75 words.
6. Answer ONLY what is asked.
7. If you absolutely cannot find the answer in the context, state that the information is not available in the provided document.

CONTEXT:
---
{context}
---
QUERY: {analysis.original_query}"""
        
        try:
            answer = await self._generate_response(prompt)
            
            return GeneratedResponse(
                query_id=analysis.query_id,
                original_query=analysis.original_query,
                answer=answer.strip(),
                context_used=True,
                generation_method="from_document"
            )
            
        except Exception as e:
            raise create_error(
                ResponseGenerationError,
                f"Document response generation failed: {str(e)}",
                "DOCUMENT_RESPONSE_FAILED",
                query_id=analysis.query_id
            )
    
    async def _generate_general_response(
        self, 
        analysis: QueryAnalysis
    ) -> GeneratedResponse:
        """Generate response for general knowledge query."""
        try:
            # Use the dynamic prompt template from Stage 3
            prompt = analysis.prompt_template or f"As a knowledgeable AI, {analysis.original_query}"
            
            answer = await self._generate_response(prompt)
            
            return GeneratedResponse(
                query_id=analysis.query_id,
                original_query=analysis.original_query,
                answer=answer.strip(),
                context_used=False,
                generation_method="general_knowledge"
            )
            
        except Exception as e:
            raise create_error(
                ResponseGenerationError,
                f"General response generation failed: {str(e)}",
                "GENERAL_RESPONSE_FAILED",
                query_id=analysis.query_id
            )
    
    async def _generate_response(self, prompt: str) -> str:
        """
        Generate response from LLM asynchronously.
        
        Args:
            prompt: Input prompt
            
        Returns:
            str: Generated response text
            
        Raises:
            LLMGenerationError: If generation fails
        """
        try:
            # Run in thread pool to avoid blocking
            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._generate_response_sync,
                prompt
            )
            
            return response
            
        except Exception as e:
            logger.error(f"LLM response generation failed: {str(e)}")
            
            if isinstance(e, LLMGenerationError):
                raise
            
            raise create_error(
                LLMGenerationError,
                f"LLM generation failed: {str(e)}",
                "LLM_GENERATION_FAILED",
                error_type=type(e).__name__
            )
    
    def _generate_response_sync(self, prompt: str) -> str:
        """
        Synchronous LLM response generation.
        
        Args:
            prompt: Input prompt
            
        Returns:
            str: Generated response
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for consistent responses
                    max_output_tokens=200,  # Limit response length
                    top_p=0.8,
                    top_k=40
                )
            )
            
            if not response.text:
                raise create_error(
                    LLMGenerationError,
                    "Empty response from LLM",
                    "EMPTY_RESPONSE"
                )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Synchronous LLM generation failed: {str(e)}")
            raise
    
    async def generate_single_response(
        self, 
        prompt: str, 
        max_tokens: int = 200
    ) -> str:
        """
        Generate a single response (utility method).
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum response tokens
            
        Returns:
            str: Generated response
        """
        return await self._generate_response(prompt)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the LLM model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            "model_name": self.model_name,
            "provider": "Google Generative AI",
            "max_workers": settings.max_workers,
            "api_configured": bool(self.api_key)
        }
    
    def __del__(self) -> None:
        """Cleanup resources when service is destroyed."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Global LLM service instance
llm_service = LLMService()


async def get_llm_service() -> LLMService:
    """
    Get the global LLM service instance.
    
    Returns:
        LLMService: The LLM service
    """
    return llm_service


# Convenience functions
async def classify_queries(
    queries: List[str], 
    document_name: str, 
    document_context: str
) -> List[QueryAnalysis]:
    """
    Classify queries using the global LLM service.
    
    Args:
        queries: List of user queries
        document_name: Document name
        document_context: Document context (first 5 pages)
        
    Returns:
        List[QueryAnalysis]: Query classifications
    """
    service = await get_llm_service()
    return await service.classify_queries(queries, document_name, document_context)


async def generate_responses(
    analyses: List[QueryAnalysis], 
    contexts: Dict[str, str]
) -> List[GeneratedResponse]:
    """
    Generate responses using the global LLM service.
    
    Args:
        analyses: Query analyses
        contexts: Retrieved contexts
        
    Returns:
        List[GeneratedResponse]: Generated responses
    """
    service = await get_llm_service()
    return await service.generate_responses(analyses, contexts)
