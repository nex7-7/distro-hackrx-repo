"""Enhanced Riddle Solver Module with Agentic RAG

Implements an intelligent agentic RAG system with chain-of-thought reasoning:
1. **PDF Parsing Phase**: Parse the special PDF to extract all textual content
2. **Agentic Query Engine**: Build a query engine that uses chain-of-thought reasoning
3. **Goal-Oriented Processing**: The agent iteratively works towards finding the objective
4. **Chain of Thought**: Uses step-by-step reasoning to reach the final answer
5. **Result Extraction**: Outputs the final number or value found

The system combines document understanding with intelligent reasoning to solve 
complex riddles that may require multiple steps of logical deduction.

Design Philosophy:
- Parse first, then reason: Extract all available information before attempting to solve
- Agentic approach: The system autonomously decides on reasoning steps
- Goal-oriented: Continuously works towards finding the final objective
- Chain-of-thought: Maintains logical reasoning chains for transparency
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any
import httpx
import asyncio
import uuid
import time
from components.utils.logger import log_service_event, log_error
from components.http_tools import HTTPRequestTool, create_http_tool_description
import re

SPECIAL_PDF_URL = (
    "https://hackrx.blob.core.windows.net/hackrx/rounds/FinalRound4SubmissionPDF.pdf?sv=2023-01-03&spr=https&st=2025-08-07T14%3A23%3A48Z&se=2027-08-08T14%3A23%3A00Z&sr=b&sp=r&sig=nMtZ2x9aBvz%2FPjRWboEOZIGB%2FaGfNf5TfBOrhGqSv4M%3D"
)


class AgenticQueryEngine:
    """
    An agentic query engine that uses chain-of-thought reasoning to solve riddles.

    The engine works by:
    1. Analyzing the parsed document content
    2. Identifying the main objective/goal
    3. Breaking down the problem into logical steps
    4. Using chain-of-thought reasoning to work towards the solution
    5. Iteratively refining its approach until the goal is reached
    """

    def __init__(self, document_chunks: List[str], http_client: httpx.AsyncClient):
        """
        Initialize the agentic query engine.

        Parameters:
            document_chunks (List[str]): Parsed content from the PDF
            http_client (httpx.AsyncClient): HTTP client for making API calls
        """
        self.document_chunks = document_chunks
        self.http_client = http_client
        self.reasoning_chain: List[Dict[str, Any]] = []
        self.session_id = str(uuid.uuid4())
        # Initialize HTTP tools for making external requests
        self.http_tool = HTTPRequestTool(http_client, self.session_id)

    async def _generate_reasoning_step(self, query: str, context: str, step_number: int) -> Dict[str, Any]:
        """
        Generate a single reasoning step using the LLM.

        Parameters:
            query (str): The current query/question to reason about
            context (str): Relevant context from the document
            step_number (int): Current step number in the reasoning chain

        Returns:
            Dict[str, Any]: The reasoning step with analysis and next action
        """
        from components.gemini_api import generate_gemini_response_httpx

        reasoning_prompt = f"""
You are an intelligent agent solving a riddle step by step. Use chain-of-thought reasoning.

DOCUMENT CONTEXT:
{context}

CURRENT OBJECTIVE: {query}

{create_http_tool_description()}

STEP {step_number} - Chain of Thought Reasoning:

Analyze the information available and reason through this step:
1. What information do I have?
2. What is my current objective?
3. What logical step should I take next?
4. Do I need to make any HTTP requests to get additional data?
5. What specific action or conclusion can I draw?

PREVIOUS REASONING CHAIN:
{self._format_reasoning_chain()}

Provide your reasoning in this format:
ANALYSIS: [Your analysis of the current situation]
REASONING: [Your logical reasoning for this step]
ACTION: [What specific action to take or conclusion to draw. If you need to make HTTP requests, specify them here using the HTTP tool format]
CONFIDENCE: [Low/Medium/High - how confident you are in this step]
NEXT_OBJECTIVE: [What should be the next focus/question, or "COMPLETE" if goal is reached]
EXTRACTED_VALUE: [If you found a specific number/value, state it here, otherwise "NONE"]
"""

        try:
            response = await generate_gemini_response_httpx(self.http_client, reasoning_prompt)

            # Parse the structured response
            reasoning_step = {
                "step_number": step_number,
                "query": query,
                "response": response,
                "timestamp": time.time(),
                "analysis": self._extract_field(response, "ANALYSIS"),
                "reasoning": self._extract_field(response, "REASONING"),
                "action": self._extract_field(response, "ACTION"),
                "confidence": self._extract_field(response, "CONFIDENCE"),
                "next_objective": self._extract_field(response, "NEXT_OBJECTIVE"),
                "extracted_value": self._extract_field(response, "EXTRACTED_VALUE")
            }

            # Check if the action contains HTTP requests and execute them
            action_text = reasoning_step["action"]
            http_results = await self._execute_http_actions(action_text)

            if http_results:
                reasoning_step["http_requests"] = http_results
                # Update the action text with the results
                reasoning_step[
                    "action"] = f"{action_text}\n\nHTTP Request Results:\n{self._format_http_results(http_results)}"

            log_service_event("reasoning_step_generated", "Generated reasoning step", {
                "session_id": self.session_id,
                "step_number": step_number,
                "confidence": reasoning_step["confidence"],
                "has_extracted_value": reasoning_step["extracted_value"] != "NONE",
                "has_http_requests": len(http_results) > 0 if http_results else False
            })

            return reasoning_step

        except Exception as e:
            log_error("reasoning_step_failed", {
                "session_id": self.session_id,
                "step_number": step_number,
                "error": str(e)
            })

            # Fallback reasoning step
            return {
                "step_number": step_number,
                "query": query,
                "response": f"Error in reasoning: {str(e)}",
                "analysis": "Unable to analyze due to error",
                "reasoning": "Error occurred during reasoning",
                "action": "Continue with fallback approach",
                "confidence": "Low",
                "next_objective": "FALLBACK",
                "extracted_value": "NONE",
                "timestamp": time.time()
            }

    def _extract_field(self, response: str, field_name: str) -> str:
        """Extract a specific field from the structured LLM response."""
        try:
            lines = response.split('\n')
            for line in lines:
                if line.strip().startswith(f"{field_name}:"):
                    return line.split(':', 1)[1].strip()
            return "Not specified"
        except Exception:
            return "Error extracting field"

    async def _execute_http_actions(self, action_text: str) -> List[Dict[str, Any]]:
        """
        Parse and execute HTTP requests found in the action text.

        Parameters:
            action_text (str): The action text that may contain HTTP requests

        Returns:
            List[Dict[str, Any]]: Results from executed HTTP requests
        """
        http_results = []

        # Parse different HTTP request patterns
        patterns = {
            'GET': r'HTTP_GET:\s*([^\s\|]+)',
            'POST': r'HTTP_POST:\s*([^\s\|]+)\s*\|\s*(.+)',
            'PUT': r'HTTP_PUT:\s*([^\s\|]+)\s*\|\s*(.+)',
            'DELETE': r'HTTP_DELETE:\s*([^\s\|]+)'
        }

        for method, pattern in patterns.items():
            matches = re.finditer(pattern, action_text,
                                  re.IGNORECASE | re.MULTILINE)

            for match in matches:
                try:
                    url = match.group(1).strip()

                    if method in ['POST', 'PUT'] and len(match.groups()) > 1:
                        # Try to parse JSON data
                        data_str = match.group(2).strip()
                        try:
                            import json
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            data = data_str  # Use as string if not valid JSON
                    else:
                        data = None

                    log_service_event("http_action_detected", f"Executing {method} request", {
                        "session_id": self.session_id,
                        "method": method,
                        "url": url,
                        "has_data": data is not None
                    })

                    # Execute the request
                    if method == 'GET':
                        result = await self.http_tool.get(url)
                    elif method == 'POST':
                        result = await self.http_tool.post(url, data=data)
                    elif method == 'PUT':
                        result = await self.http_tool.put(url, data=data)
                    elif method == 'DELETE':
                        result = await self.http_tool.delete(url)

                    result['method'] = method
                    result['url'] = url
                    http_results.append(result)

                except Exception as e:
                    log_error("http_action_execution_failed", {
                        "session_id": self.session_id,
                        "method": method,
                        "url": url if 'url' in locals() else "unknown",
                        "error": str(e)
                    })

                    http_results.append({
                        "method": method,
                        "url": url if 'url' in locals() else "unknown",
                        "success": False,
                        "error": f"Failed to execute {method} request: {str(e)}",
                        "content": None
                    })

        return http_results

    def _format_http_results(self, http_results: List[Dict[str, Any]]) -> str:
        """
        Format HTTP request results for inclusion in reasoning context.

        Parameters:
            http_results (List[Dict[str, Any]]): Results from HTTP requests

        Returns:
            str: Formatted results string
        """
        if not http_results:
            return "No HTTP requests were made."

        formatted = []
        for i, result in enumerate(http_results, 1):
            method = result.get('method', 'UNKNOWN')
            url = result.get('url', 'unknown')
            success = result.get('success', False)
            status_code = result.get('status_code')
            content = result.get('content')
            error = result.get('error')

            result_text = f"{i}. {method} {url}"

            if success:
                result_text += f" → SUCCESS (HTTP {status_code})"
                if content:
                    # Truncate content if too long
                    content_str = str(content)
                    if len(content_str) > 500:
                        content_str = content_str[:500] + "... [truncated]"
                    result_text += f"\nResponse: {content_str}"
            else:
                result_text += f" → FAILED"
                if status_code:
                    result_text += f" (HTTP {status_code})"
                if error:
                    result_text += f"\nError: {error}"

            formatted.append(result_text)

        return "\n\n".join(formatted)

    def _format_reasoning_chain(self) -> str:
        """Format the current reasoning chain for context."""
        if not self.reasoning_chain:
            return "No previous reasoning steps."

        formatted = []
        for step in self.reasoning_chain:
            formatted.append(f"Step {step['step_number']}: {step['analysis']}")
            formatted.append(f"Action taken: {step['action']}")
            # Include HTTP request results if any
            if 'http_requests' in step and step['http_requests']:
                formatted.append(
                    f"HTTP Results: {len(step['http_requests'])} request(s) made")

        return "\n".join(formatted)

    def _find_relevant_context(self, query: str) -> str:
        """Find the most relevant document chunks for the current query."""
        # Simple relevance scoring based on keyword overlap
        # In a more sophisticated system, this would use embeddings

        query_words = set(query.lower().split())
        scored_chunks = []

        for chunk in self.document_chunks:
            chunk_words = set(chunk.lower().split())
            overlap = len(query_words.intersection(chunk_words))
            scored_chunks.append((overlap, chunk))

        # Sort by relevance and take top chunks
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [chunk for score, chunk in scored_chunks[:3] if score > 0]

        if not top_chunks:
            # If no relevant chunks found, use first few chunks
            top_chunks = self.document_chunks[:3]

        return "\n\n".join(top_chunks)

    async def solve_with_chain_of_thought(self, initial_objective: str, max_steps: int = 5) -> str:
        """
        Solve the riddle using chain-of-thought reasoning.

        Parameters:
            initial_objective (str): The initial goal/question to solve
            max_steps (int): Maximum number of reasoning steps

        Returns:
            str: The final answer or result
        """
        log_service_event("agentic_solving_start", "Starting agentic riddle solving", {
            "session_id": self.session_id,
            "initial_objective": initial_objective,
            "max_steps": max_steps,
            "document_chunks": len(self.document_chunks)
        })

        current_objective = initial_objective
        final_answer = None

        for step_num in range(1, max_steps + 1):
            # Find relevant context for current objective
            relevant_context = self._find_relevant_context(current_objective)

            # Generate reasoning step
            reasoning_step = await self._generate_reasoning_step(
                current_objective, relevant_context, step_num
            )

            # Add to reasoning chain
            self.reasoning_chain.append(reasoning_step)

            # Check if we found a value
            if reasoning_step["extracted_value"] != "NONE":
                final_answer = reasoning_step["extracted_value"]
                log_service_event("value_extracted", "Value extracted from reasoning", {
                    "session_id": self.session_id,
                    "step_number": step_num,
                    "extracted_value": final_answer
                })

            # Check if we're done
            next_objective = reasoning_step["next_objective"]
            if next_objective in ["COMPLETE", "DONE", "FINISHED"] or reasoning_step["confidence"] == "High" and final_answer:
                break
            elif next_objective == "FALLBACK":
                log_service_event("fallback_triggered", "Agentic approach triggered fallback", {
                    "session_id": self.session_id,
                    "step_number": step_num
                })
                break
            else:
                current_objective = next_objective

        # Log completion
        log_service_event("agentic_solving_complete", "Completed agentic riddle solving", {
            "session_id": self.session_id,
            "steps_taken": len(self.reasoning_chain),
            "final_answer_found": final_answer is not None,
            "final_answer": final_answer
        })

        return final_answer


async def parse_special_pdf(http_client: httpx.AsyncClient) -> List[str]:
    """
    Parse the special PDF and extract textual content for the agentic engine.

    Parameters:
        http_client (httpx.AsyncClient): HTTP client for downloading the PDF

    Returns:
        List[str]: List of text chunks extracted from the PDF
    """
    from components.ingest_engine import ingest_from_url

    try:
        log_service_event("pdf_parsing_start", "Starting special PDF parsing", {
            "pdf_url": SPECIAL_PDF_URL
        })

        # Use the existing ingest engine to parse the PDF
        chunks = ingest_from_url(SPECIAL_PDF_URL)

        log_service_event("pdf_parsing_complete", "Completed special PDF parsing", {
            "chunks_extracted": len(chunks),
            "total_chars": sum(len(chunk) for chunk in chunks)
        })

        return chunks

    except Exception as e:
        log_error("pdf_parsing_failed", {
            "pdf_url": SPECIAL_PDF_URL,
            "error": str(e)
        })
        return []


async def solve_riddle_with_query(http_client: httpx.AsyncClient, user_query: str) -> str:
    """
    Enhanced riddle solving with agentic RAG using user's query as the objective.

    This function implements a sophisticated approach:
    1. **PDF Parsing Phase**: First parse the special PDF to extract content
    2. **User Query as Objective**: Use the user's query as the goal for reasoning
    3. **Agentic RAG**: Use intelligent query engine with chain-of-thought
    4. **Goal-Oriented Reasoning**: Work iteratively towards answering the user's query

    Parameters:
        http_client (httpx.AsyncClient): HTTP client for network operations
        user_query (str): The user's question/query that needs to be answered

    Returns:
        str: The answer to the user's query found through reasoning
    """
    session_id = str(uuid.uuid4())
    start_time = time.time()

    log_service_event("enhanced_riddle_solving_start", "Starting enhanced riddle solving with user query", {
        "session_id": session_id,
        "user_query": user_query
    })

    try:
        # Phase 1: Parse the special PDF first
        log_service_event("phase_1_start", "Phase 1: Parsing special PDF", {
            "session_id": session_id
        })

        document_chunks = await parse_special_pdf(http_client)

        if not document_chunks:
            log_service_event("pdf_parsing_failed", "PDF parsing failed, no content to analyze", {
                "session_id": session_id
            })
            return "Unable to parse document content for analysis."

        # Phase 2: Initialize the agentic query engine
        log_service_event("phase_2_start", "Phase 2: Initializing agentic query engine", {
            "session_id": session_id,
            "document_chunks": len(document_chunks)
        })

        query_engine = AgenticQueryEngine(document_chunks, http_client)

        # Phase 3: Use user's query as the objective
        user_objective = f"""
        Based on the document content, please answer this question: {user_query}
        
        Use the information from the document to provide a comprehensive and accurate answer.
        If the document contains specific numbers, codes, values, or data that answers this question, extract them.
        Think step by step through the document content to find the most relevant information.
        """

        # Phase 4: Execute agentic reasoning with user's query
        log_service_event("phase_3_start", "Phase 3: Executing agentic chain-of-thought reasoning with user query", {
            "session_id": session_id,
            "user_objective": user_objective
        })

        agentic_result = await query_engine.solve_with_chain_of_thought(
            user_objective,
            max_steps=7  # Allow more steps for complex reasoning
        )

        # Phase 5: Process the result
        if agentic_result and agentic_result != "NONE":
            processing_time = time.time() - start_time
            log_service_event("agentic_success", "Agentic approach found result for user query", {
                "session_id": session_id,
                "user_query": user_query,
                "result": agentic_result,
                "processing_time": processing_time,
                "reasoning_steps": len(query_engine.reasoning_chain)
            })

            # Return the result directly
            return agentic_result

        else:
            log_service_event("agentic_no_result", "Agentic approach found no result", {
                "session_id": session_id,
                "user_query": user_query,
                "reasoning_steps": len(query_engine.reasoning_chain)
            })
            return "Based on the provided documents, I cannot find relevant information to answer this question."

    except Exception as e:
        log_error("enhanced_riddle_solving_error", {
            "session_id": session_id,
            "user_query": user_query,
            "error": str(e),
            "error_type": type(e).__name__
        })

        return f"An error occurred while processing your query: {str(e)}"


async def solve_riddle(http_client: httpx.AsyncClient) -> str:
    """
    Enhanced riddle solving with agentic RAG and chain-of-thought reasoning.

    This function implements a sophisticated approach:
    1. **PDF Parsing Phase**: First parse the special PDF to extract content
    2. **Agentic Analysis**: Use an intelligent query engine with chain-of-thought
    3. **Goal-Oriented Reasoning**: Work iteratively towards finding the objective

    Parameters:
        http_client (httpx.AsyncClient): HTTP client for network operations

    Returns:
        str: The final answer found through reasoning
    """
    session_id = str(uuid.uuid4())
    start_time = time.time()

    log_service_event("enhanced_riddle_solving_start", "Starting enhanced riddle solving", {
        "session_id": session_id
    })

    try:
        # Phase 1: Parse the special PDF first
        log_service_event("phase_1_start", "Phase 1: Parsing special PDF", {
            "session_id": session_id
        })

        document_chunks = await parse_special_pdf(http_client)

        if not document_chunks:
            log_service_event("pdf_parsing_failed", "PDF parsing failed, no content to analyze", {
                "session_id": session_id
            })
            return "Unable to parse document content for analysis."

        # Phase 2: Initialize the agentic query engine
        log_service_event("phase_2_start", "Phase 2: Initializing agentic query engine", {
            "session_id": session_id,
            "document_chunks": len(document_chunks)
        })

        query_engine = AgenticQueryEngine(document_chunks, http_client)

        # Phase 3: Define the initial objective
        initial_objective = """
        Analyze this document to find the main objective or goal. 
        Look for any numbers, flight numbers, codes, or values that might be the final answer.
        Pay special attention to any instructions, puzzles, or riddles that need to be solved.
        What is the final number or value that this document is asking me to find?
        """

        # Phase 4: Execute agentic reasoning
        log_service_event("phase_3_start", "Phase 3: Executing agentic chain-of-thought reasoning", {
            "session_id": session_id,
            "initial_objective": initial_objective
        })

        agentic_result = await query_engine.solve_with_chain_of_thought(
            initial_objective,
            max_steps=7  # Allow more steps for complex reasoning
        )

        # Phase 5: Process the result
        if agentic_result and agentic_result != "NONE":
            processing_time = time.time() - start_time
            log_service_event("agentic_success", "Agentic approach found result", {
                "session_id": session_id,
                "result": agentic_result,
                "processing_time": processing_time,
                "reasoning_steps": len(query_engine.reasoning_chain)
            })

            # Wait for 4 seconds before sending the response (per original requirement)
            await asyncio.sleep(4)
            return f"The flight number is {agentic_result}"

        else:
            log_service_event("agentic_no_result", "Agentic approach found no result", {
                "session_id": session_id,
                "reasoning_steps": len(query_engine.reasoning_chain)
            })
            return "Unable to determine the answer from the document analysis."

    except Exception as e:
        log_error("enhanced_riddle_solving_error", {
            "session_id": session_id,
            "error": str(e),
            "error_type": type(e).__name__
        })

        return f"An error occurred during riddle solving: {str(e)}"


__all__ = ["solve_riddle", "solve_riddle_with_query",
           "AgenticQueryEngine", "parse_special_pdf"]
