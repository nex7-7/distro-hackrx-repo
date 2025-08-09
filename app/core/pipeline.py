"""
Main RAG pipeline orchestrator for the RAG Application API.

This module implements the complete 6-stage RAG pipeline:
1. Document Ingestion & Preprocessing
2. Data Chunking & Vectorization  
3. Query Analysis & Classification
4. Context Retrieval
5. Response Generation
6. Final Response Formatting
"""

import asyncio
import time
from typing import List, Dict, Tuple

from app.models.schemas import (
    HackRXRequest, 
    HackRXResponse, 
    DocumentInfo,
    QueryAnalysis,
    RetrievalResult,
    GeneratedResponse,
    QueryClassification
)
from app.services.document_processor import DocumentProcessor
from app.core.chunking import document_chunker
from app.services.llm_service import llm_service
from app.services.vector_store import vector_store
from app.services.embedding_service import embedding_service
from app.services.reranking_service import reranking_service
from app.utils.logger import get_logger
from app.utils.exceptions import (
    RAGApplicationError,
    DocumentProcessingError,
    ChunkingError,
    RetrievalError,
    LLMGenerationError,
    create_error
)

logger = get_logger(__name__)


class RAGPipeline:
    """
    Main RAG pipeline orchestrator that coordinates all 6 stages of processing.
    
    This class implements the complete RAG workflow from document processing
    to final response generation with proper error handling and performance monitoring.
    """
    
    def __init__(self) -> None:
        """Initialize the RAG pipeline with all required services."""
        self.document_processor = DocumentProcessor()
        
        logger.info("RAG Pipeline initialized")
    
    async def process_request(self, request: HackRXRequest) -> HackRXResponse:
        """
        Main entry point for processing HackRX requests.
        
        Executes the complete 6-stage RAG pipeline:
        1. Document Processing
        2. Chunking & Vectorization 
        3. Query Restructuring (simplified, no document context needed)
        4. Context Retrieval & Reranking
        5. Response Generation
        6. Response Formatting
        
        Args:
            request: HackRX request with document URL and questions
            
        Returns:
            HackRXResponse: Formatted response with answers
            
        Raises:
            RAGApplicationError: If any stage fails
        """
        start_time = time.time()
        
        logger.log_stage("RAG Pipeline", "Starting",
                        document_url=str(request.documents),
                        question_count=len(request.questions))
        
        try:
            # Stage 1: Document Ingestion & Preprocessing
            document_info = await self._stage_1_document_processing(str(request.documents))
            
            # Stage 2: Data Chunking & Vectorization
            was_processed = await self._stage_2_chunking_vectorization(document_info)
            
            # Stage 3: Query Restructuring (was Query Classification)
            query_analyses = await self._stage_3_query_restructuring(
                request.questions, document_info
            )
            
            # Stage 4: Context Retrieval & Reranking
            retrieval_results = await self._stage_4_context_retrieval_reranking(
                query_analyses, document_info.content_hash
            )
            
            # Stage 5: Response Generation
            generated_responses = await self._stage_5_response_generation(
                query_analyses, retrieval_results
            )
            
            # Stage 6: Final Response Formatting
            final_response = self._stage_6_response_formatting(generated_responses)
            
            duration = time.time() - start_time
            logger.log_performance(
                "Complete RAG Pipeline",
                duration,
                question_count=len(request.questions),
                document_processed=was_processed,
                responses_per_second=len(request.questions) / duration
            )
            
            return final_response
            
        except Exception as e:
            logger.error(f"RAG Pipeline failed: {str(e)}",
                        document_url=str(request.documents),
                        question_count=len(request.questions))
            
            if isinstance(e, RAGApplicationError):
                raise
            
            raise create_error(
                RAGApplicationError,
                f"RAG Pipeline execution failed: {str(e)}",
                "PIPELINE_FAILED",
                error_type=type(e).__name__
            )
    
    async def _stage_1_document_processing(self, document_url: str) -> DocumentInfo:
        """
        Stage 1: Document Ingestion & Preprocessing.
        
        Downloads and extracts text from the document using appropriate
        extraction methods based on file type.
        """
        logger.log_stage("Stage 1", "Document Processing", url=document_url)
        
        try:
            document_info = await self.document_processor.process_document(document_url)
            
            logger.log_stage("Stage 1", "Completed",
                            file_type=document_info.file_type,
                            text_length=len(document_info.extracted_text),
                            extraction_method=document_info.extraction_method)
            
            return document_info
            
        except Exception as e:
            logger.error(f"Stage 1 failed: {str(e)}")
            raise
    
    async def _stage_2_chunking_vectorization(self, document_info: DocumentInfo) -> bool:
        """
        Stage 2: Data Chunking & Vectorization.
        
        Checks for existing document, chunks text if needed, generates embeddings,
        and stores in vector database using multiprocessing.
        """
        logger.log_stage("Stage 2", "Chunking & Vectorization",
                        document_hash=document_info.content_hash)
        
        try:
            was_processed = await document_chunker.chunk_and_store_document(document_info)
            
            status = "Processed" if was_processed else "Already Existed"
            logger.log_stage("Stage 2", f"Completed - {status}")
            
            return was_processed
            
        except Exception as e:
            logger.error(f"Stage 2 failed: {str(e)}")
            raise
    
    async def _stage_3_query_restructuring(
        self, 
        questions: List[str], 
        document_info: DocumentInfo
    ) -> List[QueryAnalysis]:
        """
        Stage 3: Query Restructuring.
        
        Uses LLM to restructure queries into statements that would be present in chunks.
        This is the first LLM call that processes all questions at once.
        No document context is needed for this step.
        """
        logger.log_stage("Stage 3", "Query Restructuring", 
                        question_count=len(questions))
        
        try:
            analyses = await llm_service.restructure_queries(questions)
            
            logger.log_stage("Stage 3", "Completed",
                            total_queries=len(analyses))
            
            return analyses
            
        except Exception as e:
            logger.error(f"Stage 3 failed: {str(e)}")
            raise
    
    async def _stage_4_context_retrieval_reranking(
        self, 
        query_analyses: List[QueryAnalysis],
        document_hash: str
    ) -> Dict[str, str]:
        """
        Stage 4: Context Retrieval & Reranking.
        
        Retrieves top 30 chunks for each query using parallel vector search,
        then reranks them using cross-encoder to get top 7 chunks.
        """
        logger.log_stage("Stage 4", "Context Retrieval & Reranking", document_hash=document_hash)
        
        try:
            # All queries are now treated as document queries
            if not query_analyses:
                logger.log_stage("Stage 4", "Completed - No queries")
                return {}
            
            # Create retrieval tasks for parallel execution
            retrieval_tasks = []
            for analysis in query_analyses:
                task = self._retrieve_and_rerank_context_for_query(analysis, document_hash)
                retrieval_tasks.append(task)
            
            # Execute retrievals in parallel
            retrieval_results = await asyncio.gather(*retrieval_tasks)
            
            # Build context dictionary
            contexts = {}
            for analysis, context in zip(query_analyses, retrieval_results):
                contexts[analysis.query_id] = context
            
            logger.log_stage("Stage 4", "Completed",
                            retrieval_count=len(query_analyses),
                            contexts_found=len([c for c in contexts.values() if c]))
            
            return contexts
            
        except Exception as e:
            logger.error(f"Stage 4 failed: {str(e)}")
            raise
    
    async def _retrieve_and_rerank_context_for_query(self, analysis: QueryAnalysis, document_hash: str) -> str:
        """
        Retrieve top 30 chunks and rerank to get top 7 for a single query.
        
        Args:
            analysis: Query analysis with restructured phrases
            document_hash: Hash of the document to search within
            
        Returns:
            str: Combined context from top 7 reranked chunks
        """
        try:
            if not analysis.keywords:
                logger.warning("No restructured statement for query", 
                             query_id=analysis.query_id)
                return ""
            
            # Step 1: Use vector search to get top 30 chunks
            # keywords[0] contains the single rephrased statement
            search_query = analysis.keywords[0] if analysis.keywords else analysis.original_query
            chunks_and_scores = await vector_store.search_by_keywords(
                search_query, document_hash, limit=30
            )
            
            if not chunks_and_scores:
                logger.warning("No chunks found for query",
                             query_id=analysis.query_id,
                             keywords=analysis.keywords,
                             document_hash=document_hash)
                return ""
            
            # Extract just the chunks for reranking
            chunks = [chunk for chunk, _ in chunks_and_scores]
            
            logger.debug("Retrieved chunks for reranking",
                        query_id=analysis.query_id,
                        chunk_count=len(chunks))
            
            # Step 2: Rerank chunks using cross-encoder to get top 7
            reranked_chunks = await reranking_service.rerank_chunks(
                analysis.original_query,
                chunks,
                top_k=7
            )
            
            logger.debug("Reranked chunks",
                        query_id=analysis.query_id,
                        original_count=len(chunks),
                        reranked_count=len(reranked_chunks))
            
            # Step 3: Combine chunk contents (de-duplicate if needed)
            seen_chunks = set()
            context_parts = []
            
            for chunk in reranked_chunks:
                if chunk.chunk_id not in seen_chunks:
                    context_parts.append(chunk.content)
                    seen_chunks.add(chunk.chunk_id)
            
            context = "\n\n".join(context_parts)
            
            logger.debug("Context assembled for query",
                        query_id=analysis.query_id,
                        unique_chunks=len(seen_chunks),
                        context_length=len(context))
            
            return context
            
        except Exception as e:
            logger.error(f"Context retrieval and reranking failed for query {analysis.query_id}: {str(e)}")
            return ""
    
    async def _stage_5_response_generation(
        self, 
        query_analyses: List[QueryAnalysis],
        contexts: Dict[str, str]
    ) -> List[GeneratedResponse]:
        """
        Stage 5: Response Generation.
        
        Generates responses for all queries in parallel using multiple
        simultaneous LLM calls.
        """
        logger.log_stage("Stage 5", "Response Generation",
                        query_count=len(query_analyses))
        
        try:
            generated_responses = await llm_service.generate_responses(
                query_analyses, contexts
            )
            
            logger.log_stage("Stage 5", "Completed",
                            response_count=len(generated_responses))
            
            return generated_responses
            
        except Exception as e:
            logger.error(f"Stage 5 failed: {str(e)}")
            raise
    
    def _stage_6_response_formatting(
        self, 
        generated_responses: List[GeneratedResponse]
    ) -> HackRXResponse:
        """
        Stage 6: Final Response Formatting.
        
        Formats the generated responses into the final API response format.
        """
        logger.log_stage("Stage 6", "Response Formatting")
        
        try:
            # Extract answers in original question order
            answers = []
            for response in generated_responses:
                # Clean the answer text
                cleaned_answer = response.answer.strip()
                answers.append(cleaned_answer)
            
            final_response = HackRXResponse(answers=answers)
            
            logger.log_stage("Stage 6", "Completed",
                            answer_count=len(answers),
                            avg_answer_length=sum(len(a) for a in answers) // len(answers))
            
            return final_response
            
        except Exception as e:
            logger.error(f"Stage 6 failed: {str(e)}")
            raise create_error(
                RAGApplicationError,
                f"Response formatting failed: {str(e)}",
                "FORMATTING_FAILED"
            )


# Global pipeline instance
rag_pipeline = RAGPipeline()


async def process_hackrx_request(request: HackRXRequest) -> HackRXResponse:
    """
    Process a HackRX request using the global RAG pipeline.
    
    Args:
        request: HackRX request
        
    Returns:
        HackRXResponse: Processed response
    """
    return await rag_pipeline.process_request(request)
