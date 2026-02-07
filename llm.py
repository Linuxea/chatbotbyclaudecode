"""
OpenAI API wrapper for chat completions with streaming support.
"""
import os
from typing import List, Dict, Iterator, Optional
from openai import OpenAI


def get_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    """Get OpenAI client with API key and optional custom base URL."""
    kwargs = {}

    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAI(**kwargs)


class StreamChunk:
    """Represents a chunk of streamed response."""
    def __init__(self, content: str = "", reasoning: str = ""):
        self.content = content
        self.reasoning = reasoning


def stream_chat_response(
    messages: List[Dict[str, str]],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 2000
) -> Iterator[StreamChunk]:
    """
    Stream chat response from OpenAI API.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        api_key: OpenAI API key (optional if set in environment)
        base_url: Custom API base URL for OpenAI-compatible services (e.g., Ollama, vLLM)
        model: Model name to use
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens to generate

    Yields:
        StreamChunk objects containing content and/or reasoning
    """
    client = get_client(api_key, base_url)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )

        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta
                content = getattr(delta, 'content', None) or ""
                # DeepSeek R1 returns reasoning_content in delta
                reasoning = getattr(delta, 'reasoning_content', None) or ""
                if content or reasoning:
                    yield StreamChunk(content=content, reasoning=reasoning)

    except Exception as e:
        yield StreamChunk(content=f"\n\nError: {str(e)}")


def chat_response(
    messages: List[Dict[str, str]],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 2000
) -> str:
    """
    Get complete chat response from OpenAI API (non-streaming).

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        api_key: OpenAI API key (optional if set in environment)
        base_url: Custom API base URL for OpenAI-compatible services (e.g., Ollama, vLLM)
        model: Model name to use
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens to generate

    Returns:
        Complete response text
    """
    client = get_client(api_key, base_url)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content or ""

    except Exception as e:
        return f"Error: {str(e)}"


def format_messages_for_api(
    system_prompt: Optional[str],
    history: List[Dict[str, str]],
    user_message: str,
    file_content: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Format messages for OpenAI API.

    Args:
        system_prompt: Optional system prompt
        history: Previous conversation messages
        user_message: Current user message
        file_content: Optional file content to include as context

    Returns:
        Formatted messages list for API
    """
    messages = []

    # Add system prompt if provided
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Add conversation history
    messages.extend(history)

    # Combine user message with file content if present
    if file_content:
        combined_content = f"{user_message}\n\n[File Content]:\n{file_content}"
        messages.append({"role": "user", "content": combined_content})
    else:
        messages.append({"role": "user", "content": user_message})

    return messages


# Available models (DeepSeek)
AVAILABLE_MODELS = [
    "deepseek-chat",
    "deepseek-reasoner",
]
