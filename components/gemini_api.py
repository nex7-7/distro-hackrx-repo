"""
Gemini API Module

This module contains functions for interacting with the Gemini API.
"""
import json
import time
import random
import httpx
from typing import List, Dict, Any
import os

# Import local modules
from logger import log_service_event, log_error

# Get API keys from environment
GEMINI_MODEL_NAME = os.getenv('GEMINI_MODEL_NAME', 'gemini-2.5-flash')
GEMINI_API_KEYS_STR = os.getenv("GEMINI_API_KEYS", "")
GEMINI_API_KEY_LIST = [key.strip()
                       for key in GEMINI_API_KEYS_STR.split(',') if key.strip()]
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"


async def generate_gemini_response_httpx(session: httpx.AsyncClient, prompt: str) -> str:
    """
    Generate a response using the Gemini API.

    Parameters:
        session (httpx.AsyncClient): HTTP client session for making requests.
        prompt (str): The prompt to send to the Gemini API.

    Returns:
        str: The generated text response from Gemini.
    """
    start_time = time.time()
    api_key = random.choice(GEMINI_API_KEY_LIST)
    headers = {'Content-Type': 'application/json'}
    payload = {'contents': [{'parts': [{'text': prompt}]}]}

    # Log Gemini request (excluding the full prompt for brevity)
    log_service_event("gemini_request", "Sending request to Gemini API", {
        "prompt_length": len(prompt),
        "model": GEMINI_MODEL_NAME
    })

    # --- DEBUG PRINT 2: PAYLOAD GIVEN ---
    print("\n" + "="*50)
    print("2. PAYLOAD GIVEN TO GEMINI")
    print("="*50)
    # Use json.dumps for pretty printing the dictionary
    print(json.dumps(payload, indent=2))
    print("="*50 + "\n")
    # ------------------------------------

    try:
        response = await session.post(f"{GEMINI_API_URL}?key={api_key}", json=payload, headers=headers, timeout=90)
        duration = time.time() - start_time

        # Log response timing
        log_service_event("gemini_response_received", "Received response from Gemini API", {
            "status_code": response.status_code,
            "duration_seconds": duration
        })

        # --- DEBUG PRINT 3: OUTPUT RECEIVED ---
        print("\n" + "="*50)
        print("3. OUTPUT RECEIVED FROM GEMINI")
        print("="*50)
        print(f"Status Code: {response.status_code}")
        # Try to pretty print if it's JSON, otherwise print as text
        try:
            json_response = response.json()
            print(json.dumps(json_response, indent=2))
        except json.JSONDecodeError:
            print(response.text)
        print("="*50 + "\n")
        # --------------------------------------

        response.raise_for_status()
        data = response.json()
        response_text = data['candidates'][0]['content']['parts'][0]['text']

        # Log successful response (only first 100 chars of text for brevity)
        log_service_event("gemini_response_success", "Successfully processed Gemini response", {
            "response_length": len(response_text),
            "response_preview": response_text[:100] + ("..." if len(response_text) > 100 else ""),
            "duration_seconds": duration
        })

        return response_text
    except httpx.HTTPStatusError as e:
        error_msg = f"Gemini API error: {e.response.status_code} - {e.response.text}"
        print(f"❌ {error_msg}")
        log_error("gemini_api_error", {
            "status_code": e.response.status_code,
            "response_text": e.response.text,
            "duration_seconds": time.time() - start_time
        })
        return error_msg
    except (KeyError, IndexError) as e:
        error_msg = f"Error parsing Gemini response: {e}. Response: {response.text}"
        print(f"❌ {error_msg}")
        log_error("gemini_response_parse_error", {
            "error": str(e),
            "response_text": response.text if hasattr(response, 'text') else None,
            "duration_seconds": time.time() - start_time
        })
        return error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred during Gemini call: {e}"
        print(f"❌ {error_msg}")
        log_error("gemini_unexpected_error", {
            "error": str(e),
            "duration_seconds": time.time() - start_time
        })
        return error_msg
