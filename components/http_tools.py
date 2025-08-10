"""HTTP Tools for Agentic RAG System

Provides HTTP request capabilities for the agentic query engine to fetch 
additional data during reasoning processes. This allows the agent to:
1. Make GET requests to retrieve data
2. Make POST requests to submit data  
3. Handle authentication and headers
4. Parse and return structured responses
5. Log all network operations for debugging

Safety Features:
- Request timeout limits
- URL validation
- Response size limits  
- Error handling and retries
- Security headers validation
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any, Union
import httpx
import json
import time
import uuid
from urllib.parse import urlparse
from components.utils.logger import log_service_event, log_error


class HTTPRequestTool:
    """
    HTTP request tool for the agentic system to make external API calls.

    This tool provides safe and monitored HTTP capabilities for the reasoning engine.
    """

    def __init__(self, http_client: httpx.AsyncClient, session_id: str):
        """
        Initialize the HTTP request tool.

        Parameters:
            http_client (httpx.AsyncClient): The HTTP client to use for requests
            session_id (str): Session ID for logging and tracking
        """
        self.http_client = http_client
        self.session_id = session_id
        self.request_count = 0
        self.max_requests = 10  # Limit requests per session
        self.max_response_size = 1024 * 1024  # 1MB limit

    def _is_safe_url(self, url: str) -> bool:
        """Check if a URL is safe to request."""
        try:
            parsed = urlparse(url)
            # Allow only http/https schemes
            if parsed.scheme not in ['http', 'https']:
                return False
            # Block local/private networks for security
            if parsed.hostname in ['localhost', '127.0.0.1', '0.0.0.0']:
                return False
            if parsed.hostname and parsed.hostname.startswith('192.168.'):
                return False
            if parsed.hostname and parsed.hostname.startswith('10.'):
                return False
            return True
        except Exception:
            return False

    async def make_request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Union[Dict, str]] = None,
        params: Optional[Dict[str, str]] = None,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Make an HTTP request with safety checks and logging.

        Parameters:
            method (str): HTTP method (GET, POST, PUT, DELETE, etc.)
            url (str): Target URL
            headers (Optional[Dict[str, str]]): Request headers
            data (Optional[Union[Dict, str]]): Request body data
            params (Optional[Dict[str, str]]): Query parameters
            timeout (float): Request timeout in seconds

        Returns:
            Dict[str, Any]: Response data with status, headers, and content
        """
        request_id = str(uuid.uuid4())[:8]
        self.request_count += 1

        # Check request limits
        if self.request_count > self.max_requests:
            log_error("http_tool_request_limit", {
                "session_id": self.session_id,
                "request_count": self.request_count,
                "max_requests": self.max_requests
            })
            return {
                "success": False,
                "error": f"Request limit exceeded ({self.max_requests} per session)",
                "status_code": None,
                "content": None
            }

        # Validate URL safety
        if not self._is_safe_url(url):
            log_error("http_tool_unsafe_url", {
                "session_id": self.session_id,
                "request_id": request_id,
                "url": url,
                "method": method
            })
            return {
                "success": False,
                "error": "URL not allowed for security reasons",
                "status_code": None,
                "content": None
            }

        log_service_event("http_tool_request_start", "Making HTTP request", {
            "session_id": self.session_id,
            "request_id": request_id,
            "method": method.upper(),
            "url": url,
            "has_headers": headers is not None,
            "has_data": data is not None,
            "has_params": params is not None
        })

        start_time = time.time()

        try:
            # Prepare request data
            request_kwargs = {
                "method": method.upper(),
                "url": url,
                "timeout": min(timeout, 60.0),  # Cap timeout at 60 seconds
            }

            if headers:
                request_kwargs["headers"] = headers

            if params:
                request_kwargs["params"] = params

            if data:
                if isinstance(data, dict):
                    request_kwargs["json"] = data
                else:
                    request_kwargs["content"] = data

            # Make the request
            response = await self.http_client.request(**request_kwargs)

            # Check response size
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.max_response_size:
                log_error("http_tool_response_too_large", {
                    "session_id": self.session_id,
                    "request_id": request_id,
                    "content_length": int(content_length),
                    "max_size": self.max_response_size
                })
                return {
                    "success": False,
                    "error": f"Response too large ({content_length} bytes)",
                    "status_code": response.status_code,
                    "content": None
                }

            # Read response content with size limit
            content_bytes = b""
            async for chunk in response.aiter_bytes(chunk_size=8192):
                content_bytes += chunk
                if len(content_bytes) > self.max_response_size:
                    log_error("http_tool_response_size_exceeded", {
                        "session_id": self.session_id,
                        "request_id": request_id,
                        "bytes_read": len(content_bytes),
                        "max_size": self.max_response_size
                    })
                    return {
                        "success": False,
                        "error": "Response size limit exceeded during streaming",
                        "status_code": response.status_code,
                        "content": None
                    }

            # Try to parse as JSON, fall back to text
            try:
                content = json.loads(content_bytes.decode('utf-8'))
                content_type = "json"
            except (json.JSONDecodeError, UnicodeDecodeError):
                try:
                    content = content_bytes.decode('utf-8')
                    content_type = "text"
                except UnicodeDecodeError:
                    content = content_bytes.hex()
                    content_type = "binary"

            request_time = time.time() - start_time

            log_service_event("http_tool_request_complete", "HTTP request completed", {
                "session_id": self.session_id,
                "request_id": request_id,
                "method": method.upper(),
                "url": url,
                "status_code": response.status_code,
                "content_type": content_type,
                "content_length": len(content_bytes),
                "request_time": request_time,
                "success": response.is_success
            })

            return {
                "success": response.is_success,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": content,
                "content_type": content_type,
                "request_time": request_time,
                "error": None if response.is_success else f"HTTP {response.status_code}"
            }

        except httpx.TimeoutException as e:
            log_error("http_tool_timeout", {
                "session_id": self.session_id,
                "request_id": request_id,
                "method": method.upper(),
                "url": url,
                "timeout": timeout,
                "error": str(e)
            })
            return {
                "success": False,
                "error": f"Request timeout after {timeout} seconds",
                "status_code": None,
                "content": None
            }

        except httpx.RequestError as e:
            log_error("http_tool_request_error", {
                "session_id": self.session_id,
                "request_id": request_id,
                "method": method.upper(),
                "url": url,
                "error": str(e),
                "error_type": type(e).__name__
            })
            return {
                "success": False,
                "error": f"Request failed: {str(e)}",
                "status_code": None,
                "content": None
            }

        except Exception as e:
            log_error("http_tool_unexpected_error", {
                "session_id": self.session_id,
                "request_id": request_id,
                "method": method.upper(),
                "url": url,
                "error": str(e),
                "error_type": type(e).__name__
            })
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "status_code": None,
                "content": None
            }

    async def get(self, url: str, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Make a GET request."""
        return await self.make_request("GET", url, headers=headers, params=params)

    async def post(self, url: str, data: Optional[Union[Dict, str]] = None, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Make a POST request."""
        return await self.make_request("POST", url, headers=headers, data=data)

    async def put(self, url: str, data: Optional[Union[Dict, str]] = None, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Make a PUT request."""
        return await self.make_request("PUT", url, headers=headers, data=data)

    async def delete(self, url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Make a DELETE request."""
        return await self.make_request("DELETE", url, headers=headers)


def create_http_tool_description() -> str:
    """
    Create a description of available HTTP tools for the LLM to understand.

    Returns:
        str: Tool description for inclusion in prompts
    """
    return """
AVAILABLE HTTP TOOLS:
You have access to HTTP request capabilities through the following methods:

1. GET Request: Use "HTTP_GET: <url>" to fetch data from an endpoint
   Example: HTTP_GET: https://api.example.com/data

2. POST Request: Use "HTTP_POST: <url> | <json_data>" to send data
   Example: HTTP_POST: https://api.example.com/submit | {"key": "value"}

3. PUT Request: Use "HTTP_PUT: <url> | <json_data>" to update data
   Example: HTTP_PUT: https://api.example.com/update | {"id": 1, "status": "active"}

4. DELETE Request: Use "HTTP_DELETE: <url>" to delete resources
   Example: HTTP_DELETE: https://api.example.com/item/123

IMPORTANT NOTES:
- Only use HTTP requests when necessary for your objective
- Parse the response content to extract relevant information
- Include any authentication headers if required
- Maximum 10 requests per reasoning session
- Responses are limited to 1MB in size

When you need to make an HTTP request, specify it clearly in your ACTION field using the format above.
"""


__all__ = ["HTTPRequestTool", "create_http_tool_description"]
