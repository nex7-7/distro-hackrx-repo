"""
LLM service for the RAG Application API.

This module provides Google Gemini integration for Stage 3 (query restructuring)
and Stage 5 (response generation) of the RAG pipeline. It supports parallel LLM
calls and structured prompt templates as specified in the requirements.
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor

from google import genai

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
    Google Gemini LLM integration for query restructuring and response generation.
    
    This service handles Stage 3 (query restructuring) and Stage 5 (response
    generation) with support for parallel processing and structured prompts.
    """
    
    def __init__(self) -> None:
        """Initialize the LLM service."""
        self.model_name = settings.llm_model
        self.api_key = settings.google_api_key
        self.executor = ThreadPoolExecutor(max_workers=settings.max_workers)
        
        # Configure Google Generative AI Client
        self.client = genai.Client(api_key=self.api_key)
        
        logger.info("LLM service initialized", model=self.model_name)
    
    async def restructure_queries(
        self, 
        queries: List[str]
    ) -> List[QueryAnalysis]:
        """
        Restructure queries into statements that would be present in chunks (Stage 3).
        
        This is the first LLM call that processes all questions at once,
        converting them into single statements that are more likely to be found in document chunks.
        
        Args:
            queries: List of user queries
            
        Returns:
            List[QueryAnalysis]: Restructured queries for each input query
            
        Raises:
            QueryClassificationError: If restructuring fails
        """
        start_time = time.time()
        
        logger.log_stage("Query Restructuring", "Starting", 
                        query_count=len(queries))
        
        try:
            # Construct the restructuring prompt
            prompt = self._build_restructuring_prompt(queries)
            
            # Generate restructuring response
            response_text = await self._generate_response(prompt)
            logger.debug("Received restructuring response", 
                        response_length=len(response_text),
                        response_preview=response_text[:300] + "..." if len(response_text) > 300 else response_text)
            
            # Parse JSON response
            restructuring_data = self._parse_restructuring_response(response_text)
            logger.debug("Parsed restructuring data", 
                        data_keys=list(restructuring_data.keys()) if restructuring_data else None)
            
            # Convert to QueryAnalysis objects
            analyses = []
            for i, query in enumerate(queries):
                query_key = f"Query {i + 1}"
                query_data = restructuring_data.get(query_key, {})
                
                # Get the single rephrased statement
                rephrased_statement = query_data.get("Rephrased_Statement", query)
                
                analysis = QueryAnalysis(
                    query_id=f"q_{i}",
                    original_query=query,
                    classification=QueryClassification.FROM_DOCUMENT,  # All queries are treated as document queries now
                    keywords=[rephrased_statement],  # Use single rephrased statement as search term
                    prompt_template=None  # No longer needed for classification
                )
                analyses.append(analysis)
            
            duration = time.time() - start_time
            logger.log_performance(
                "Query Restructuring",
                duration,
                query_count=len(queries)
            )
            
            return analyses
            
        except Exception as e:
            logger.error(f"Primary restructuring failed: {str(e)}")
            
            # Check if it's a safety block and try fallback
            if "SAFETY_BLOCKED" in str(e):
                logger.warning("Content blocked by safety filters, using fallback restructuring")
                return self._fallback_restructuring(queries)
            
            if isinstance(e, QueryClassificationError):
                raise
            
            # Try fallback for other errors too
            logger.warning("Restructuring failed, attempting fallback restructuring")
            try:
                return self._fallback_restructuring(queries)
            except Exception as fallback_error:
                logger.error(f"Fallback restructuring also failed: {str(fallback_error)}")
                
                raise create_error(
                    QueryClassificationError,
                    f"Failed to restructure queries: {str(e)}",
                    "RESTRUCTURING_FAILED",
                    query_count=len(queries),
                    error_type=type(e).__name__
                )
    
    def _fallback_restructuring(self, queries: List[str]) -> List[QueryAnalysis]:
        """
        Fallback restructuring when LLM fails (e.g., safety blocks).
        Uses simple heuristics to restructure queries into search statements.
        """
        logger.info("Using fallback restructuring method", query_count=len(queries))
        
        analyses = []
        for i, query in enumerate(queries):
            # Simple heuristic: extract key terms from the query and create a statement
            query_lower = query.lower()
            
            # Remove question words and common terms
            stop_words = {"what", "when", "where", "how", "why", "does", "is", "are", "the", "and", "or", "but", "of", "for", "in", "on", "at", "to", "from", "a", "an"}
            words = [word.strip(".,?!") for word in query_lower.split()]
            key_terms = [word for word in words if len(word) > 2 and word not in stop_words]
            
            # Create a single search statement
            if key_terms:
                rephrased_statement = " ".join(key_terms[:4])  # Use up to 4 key terms
            else:
                rephrased_statement = query_lower.strip("?.,!")
            
            analysis = QueryAnalysis(
                query_id=f"q_{i}",
                original_query=query,
                classification=QueryClassification.FROM_DOCUMENT,
                keywords=[rephrased_statement],
                prompt_template=None
            )
            
            analyses.append(analysis)
            
        logger.info("Fallback restructuring completed", query_count=len(analyses))
        
        return analyses
    
    def _build_restructuring_prompt(self, queries: List[str]) -> str:
        """Build the query restructuring prompt template."""
        queries_formatted = [
            f"Query {i + 1}: {query}" 
            for i, query in enumerate(queries)
        ]
        
        prompt = f"""You are an expert query restructurer for document retrieval systems.

TASK: Convert user questions into single declarative statements that would actually be present in document content.

GOAL: Transform questions into statements that match how information appears in documents.

INSTRUCTIONS:
1. Convert each question into ONE clear, concise statement
2. Remove question words (what, how, when, where, etc.)
3. Use terminology that would appear in formal documents
4. Focus on the core concept being asked about
5. Keep statements short and specific

RESPONSE FORMAT:
Return only a valid JSON object with the structure:
{{
  "Query 1": {{"Rephrased_Statement": "single rephrased statement"}},
  "Query 2": {{"Rephrased_Statement": "single rephrased statement"}}
}}

QUERIES TO RESTRUCTURE:
{chr(10).join(queries_formatted)}

JSON Response:"""
        
        return prompt
    
    def _parse_restructuring_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the JSON restructuring response."""
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
            logger.error("Failed to parse restructuring JSON", 
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
            # Create tasks for parallel execution - all queries are document-based now
            tasks = []
            for analysis in analyses:
                context = contexts.get(analysis.query_id, "")
                task = self._generate_document_response(analysis, context)
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
        # Check if context is empty or insufficient
        if not context.strip():
            return GeneratedResponse(
                query_id=analysis.query_id,
                original_query=analysis.original_query,
                answer="The question cannot be answered from the provided context.",
                context_used=False,
                generation_method="no_context"
            )
        
        prompt = f"""You are an expert response agent. Your role is to analyze the context provided and provide clear, accurate responses.

SAFETY CONSTRAINTS:
1. Never reveal customer personal data, account numbers, or sensitive information
2. Do not answer questions that could compromise privacy or security
3. Refuse to answer dangerous or harmful questions
4. If the question seeks sensitive information, respond with "I cannot provide sensitive information for security reasons."

INSTRUCTIONS:
1. Analyze the user's query against the provided context.
2. If you can only provide a partial answer based on the context, do so.
3. If the context doesn't contain the answer at all, then respond EXACTLY with: "The question cannot be answered from the provided context."
4. Do not mention chunk ids or references.
5. The answer MUST be based on the context provided.
6. If the Policy Name in the query does not match, assume it is the same policy as mentioned in the context.
7. You MUST provide a concise, single-paragraph answer. Your entire response MUST NOT exceed 75 words.
8. Answer ONLY what is asked.

CONTEXT:
---
{context}
---
QUERY: {analysis.original_query}

RESPONSE:"""
        
        try:
            answer = await self._generate_response(prompt)
            answer = answer.strip()
            
            # Check if the response indicates no context found
            context_used = not (
                "cannot be answered from the provided context" in answer.lower() or
                "cannot provide sensitive information" in answer.lower()
            )
            
            return GeneratedResponse(
                query_id=analysis.query_id,
                original_query=analysis.original_query,
                answer=answer,
                context_used=context_used,
                generation_method="from_document" if context_used else "no_context"
            )
            
        except Exception as e:
            raise create_error(
                ResponseGenerationError,
                f"Document response generation failed: {str(e)}",
                "DOCUMENT_RESPONSE_FAILED",
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
            logger.debug("Sending prompt to Gemini", 
                        prompt_length=len(prompt),
                        prompt_preview=prompt[:200] + "..." if len(prompt) > 200 else prompt)
            
            # Using the new google-genai library API
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "temperature": 0.1,
                    "top_p": 0.8,
                    "top_k": 40
                }
            )
            
            logger.debug("Received response from Gemini",
                        has_candidates=bool(response.candidates),
                        candidate_count=len(response.candidates) if response.candidates else 0)
            
            # Check if response has any candidates
            if not response.candidates:
                logger.error("No candidates returned from Gemini")
                raise create_error(
                    LLMGenerationError,
                    "No response candidates returned from LLM",
                    "NO_CANDIDATES"
                )
            
            # Get the first candidate
            candidate = response.candidates[0]
            logger.debug("Candidate details",
                        finish_reason=candidate.finish_reason if hasattr(candidate, 'finish_reason') else 'unknown',
                        has_content=bool(candidate.content) if hasattr(candidate, 'content') else False)
            
            # Check finish reason if available
            if hasattr(candidate, 'finish_reason'):
                finish_reason_str = str(candidate.finish_reason)
                logger.debug("Response finish reason", finish_reason=finish_reason_str)
                
                if finish_reason_str in ["SAFETY", "FinishReason.SAFETY"]:
                    logger.error("Content blocked by safety filters",
                               finish_reason=candidate.finish_reason)
                    raise create_error(
                        LLMGenerationError,
                        "Content was blocked by safety filters. The prompt may contain sensitive content.",
                        "SAFETY_BLOCKED",
                        finish_reason=candidate.finish_reason
                    )
                elif finish_reason_str in ["RECITATION", "FinishReason.RECITATION"]:
                    logger.error("Content blocked due to recitation",
                               finish_reason=candidate.finish_reason)
                    raise create_error(
                        LLMGenerationError,
                        "Content was blocked due to recitation concerns",
                        "RECITATION_BLOCKED",
                        finish_reason=candidate.finish_reason
                    )
                elif finish_reason_str in ["MAX_TOKENS", "FinishReason.MAX_TOKENS"]:
                    logger.warning("Response truncated due to model's internal token limit",
                                 finish_reason=candidate.finish_reason)
                    # This is not an error - just a truncated response
                elif finish_reason_str not in ["STOP", "FinishReason.STOP", None, "None"]:
                    logger.error("Unexpected finish reason",
                               finish_reason=candidate.finish_reason)
                    raise create_error(
                        LLMGenerationError,
                        f"Unexpected finish reason: {candidate.finish_reason}",
                        "UNEXPECTED_FINISH_REASON",
                        finish_reason=candidate.finish_reason
                    )
            
            # Extract text from response
            response_text = ""
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text += part.text
                elif hasattr(candidate.content, 'text'):
                    response_text = candidate.content.text
            elif hasattr(candidate, 'text'):
                response_text = candidate.text
            
            if not response_text:
                logger.error("Empty response text after processing")
                raise create_error(
                    LLMGenerationError,
                    "Empty response from LLM",
                    "EMPTY_RESPONSE"
                )
            
            logger.debug("Successfully extracted response text",
                        response_length=len(response_text))
            
            return response_text
            
        except Exception as e:
            logger.error(f"Synchronous LLM generation failed: {str(e)}")
            logger.exception("Full LLM generation exception")
            raise
    
    async def generate_single_response(
        self, 
        prompt: str
    ) -> str:
        """
        Generate a single response (utility method).
        
        Args:
            prompt: Input prompt
            
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
            "provider": "Google Gemini AI (google-genai)",
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
async def restructure_queries(queries: List[str]) -> List[QueryAnalysis]:
    """
    Restructure queries using the global LLM service.
    
    Args:
        queries: List of user queries
        
    Returns:
        List[QueryAnalysis]: Query restructuring results
    """
    service = await get_llm_service()
    return await service.restructure_queries(queries)


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
