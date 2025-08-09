# GitHub Models API config
from components.utils.logger import log_service_event, log_error
import os
from typing import List, Dict, Any
import httpx
import random
import time
import json
GITHUB_MODELS_API_URL = os.getenv(
    "GITHUB_MODELS_API_URL", "https://models.github.ai/inference/chat/completions")
GITHUB_MODELS_TOKEN = os.getenv("GITHUB_MODELS_TOKEN", "")
GITHUB_MODELS_MODEL_ID = os.getenv("GITHUB_MODELS_MODEL_ID", "openai/gpt-4.1")


async def generate_github_models_response_httpx(session: httpx.AsyncClient, prompt: str) -> str:
    """
    Generate a response using GitHub Models API (final fallback).

    Parameters:
        session (httpx.AsyncClient): HTTP client session for making requests.
        prompt (str): The prompt to send to the GitHub Models API.

    Returns:
        str: The generated text response from GitHub Models.
    """
    start_time = time.time()
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_MODELS_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GITHUB_MODELS_MODEL_ID,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    log_service_event("github_models_request", "Sending request to GitHub Models API (final fallback)", {
        "prompt_length": len(prompt),
        "model": GITHUB_MODELS_MODEL_ID
    })
    try:
        response = await session.post(GITHUB_MODELS_API_URL, json=payload, headers=headers, timeout=90)
        duration = time.time() - start_time
        log_service_event("github_models_response_received", "Received response from GitHub Models API", {
            "status_code": response.status_code,
            "duration_seconds": duration
        })
        response.raise_for_status()
        data = response.json()
        response_text = data["choices"][0]["message"]["content"]
        log_service_event("github_models_response_success", "Successfully processed GitHub Models response", {
            "response_length": len(response_text),
            "response_preview": response_text[:100] + ("..." if len(response_text) > 100 else ""),
            "duration_seconds": duration
        })
        return response_text
    except Exception as e:
        error_msg = f"GitHub Models API error: {e}"
        print(f"❌ {error_msg}")
        log_error("github_models_api_error", {
            "error": str(e),
            "duration_seconds": time.time() - start_time
        })
        return error_msg
"""
Gemini API Module

This module contains functions for interacting with the Gemini API.
"""

# Import local modules


# Gemini API config
GEMINI_MODEL_NAME = os.getenv('GEMINI_MODEL_NAME', 'gemini-2.5-flash')
GEMINI_API_KEYS_STR = os.getenv("GEMINI_API_KEYS", "")
GEMINI_API_KEY_LIST = [key.strip()
                       for key in GEMINI_API_KEYS_STR.split(',') if key.strip()]
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"

# Mistral API config
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_API_URL = os.getenv(
    "MISTRAL_API_URL", "https://api.mistral.ai/v1/chat/completions")
MISTRAL_MODEL_NAME = os.getenv("MISTRAL_MODEL_NAME", "mistral-large-latest")


async def generate_mistral_response_httpx(session: httpx.AsyncClient, prompt: str) -> str:
    """
    Generate a response using the Mistral API (fallback).

    Parameters:
        session (httpx.AsyncClient): HTTP client session for making requests.
        prompt (str): The prompt to send to the Mistral API.

    Returns:
        str: The generated text response from Mistral.
    """
    start_time = time.time()
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {MISTRAL_API_KEY}'
    }
    payload = {
        "model": MISTRAL_MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    log_service_event("mistral_request", "Sending request to Mistral API (fallback)", {
        "prompt_length": len(prompt),
        "model": MISTRAL_MODEL_NAME
    })
    try:
        response = await session.post(MISTRAL_API_URL, json=payload, headers=headers, timeout=90)
        duration = time.time() - start_time
        log_service_event("mistral_response_received", "Received response from Mistral API", {
            "status_code": response.status_code,
            "duration_seconds": duration
        })
        response.raise_for_status()
        data = response.json()
        response_text = data["choices"][0]["message"]["content"]
        log_service_event("mistral_response_success", "Successfully processed Mistral response", {
            "response_length": len(response_text),
            "response_preview": response_text[:100] + ("..." if len(response_text) > 100 else ""),
            "duration_seconds": duration
        })
        return response_text
    except Exception as e:
        error_msg = f"Mistral API error: {e}"
        print(f"❌ {error_msg}")
        log_error("mistral_api_fallback", {
            "error": str(e),
            "duration_seconds": time.time() - start_time
        })
        # Fallback to GitHub Models
        log_service_event("mistral_fallback_to_github_models", "Falling back to GitHub Models API due to Mistral error", {
            "error": str(e),
            "prompt_length": len(prompt)
        })
        github_response = await generate_github_models_response_httpx(session, prompt)
        return github_response


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
    except Exception as e:
        error_msg = f"Gemini API error or unexpected failure: {e}"
        print(f"❌ {error_msg}")
        log_error("gemini_api_fallback", {
            "error": str(e),
            "duration_seconds": time.time() - start_time
        })
        # Fallback to Mistral
        log_service_event("gemini_fallback_to_mistral", "Falling back to Mistral API due to Gemini error", {
            "error": str(e),
            "prompt_length": len(prompt)
        })
        mistral_response = await generate_mistral_response_httpx(session, prompt)
        return mistral_response
