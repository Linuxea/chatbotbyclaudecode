"""
OpenAI API wrapper for chat completions with streaming support.
"""
import os
import re
import json
from typing import List, Dict, Iterator, Optional, Tuple
from openai import OpenAI

import logger


def get_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    """Get OpenAI client with API key and optional custom base URL."""
    kwargs = {}

    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAI(**kwargs)


def parse_dsml_tool_calls(content: str) -> Tuple[List[Dict], str]:
    """
    Parse DeepSeek DSML format tool calls from content.

    DeepSeek uses a proprietary DSML (DeepSeek Markup Language) format:
    <｜DSML｜function_calls>
    <function_cinvoke name="function_name">
      <parameter name="param_name" string="true">value</parameter>
    </function_cinvoke>
    </｜DSML｜function_calls>

    Args:
        content: The response content that may contain DSML tags

    Returns:
        Tuple of (list of parsed tool calls, cleaned content without DSML tags)
    """
    tool_calls = []

    # Pattern to match the entire DSML function_calls block (supports full-width and ASCII pipes)
    dsml_pattern = r'<[|｜]DSML[|｜]function_calls[^>]*>(.*?)</[|｜]DSML[|｜]function_calls>'

    matches = re.findall(dsml_pattern, content, re.DOTALL | re.IGNORECASE)

    for match in matches:
        # Parse function invocations within the block
        func_pattern = r'<function_cinvoke\s+name="([^"]+)"[^>]*>(.*?)</function_cinvoke>'
        func_matches = re.findall(func_pattern, match, re.DOTALL | re.IGNORECASE)

        for func_name, params_block in func_matches:
            # Parse parameters
            params = {}
            param_pattern = r'<parameter\s+name="([^"]+)"(?:\s+string="[^"]*")?\s*>(.*?)</parameter>'
            param_matches = re.findall(param_pattern, params_block, re.DOTALL | re.IGNORECASE)

            for param_name, param_value in param_matches:
                # Try to parse as JSON if it looks like JSON
                param_value = param_value.strip()
                if param_value.startswith('{') or param_value.startswith('['):
                    try:
                        params[param_name] = json.loads(param_value)
                    except json.JSONDecodeError:
                        params[param_name] = param_value
                else:
                    params[param_name] = param_value

            tool_call = {
                "id": f"call_{len(tool_calls)}",
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(params) if params else "{}"
                }
            }
            tool_calls.append(tool_call)

    # Clean the content by removing DSML blocks
    cleaned_content = re.sub(dsml_pattern, '', content, flags=re.DOTALL | re.IGNORECASE)

    return tool_calls, cleaned_content


def clean_dsml_content(content: str) -> str:
    """
    Remove DSML (DeepSeek Markup Language) tags from content.

    DSML format used by DeepSeek for tool calls:
    <｜DSML｜function_calls>
      <function_cinvoke name="...">...</function_cinvoke>
    </｜DSML｜function_calls>

    Args:
        content: Content that may contain DSML tags

    Returns:
        Cleaned content without DSML markup
    """
    if not content:
        return ""

    # Single regex to remove entire DSML blocks (handles both full-width and ASCII pipes)
    dsml_pattern = r'<[|｜]DSML[|｜][^>]*>.*?</[|｜]DSML[|｜][^>]*>'
    cleaned = re.sub(dsml_pattern, '', content, flags=re.DOTALL | re.IGNORECASE)

    # Clean up extra blank lines (preserve single newlines, collapse multiple)
    cleaned = re.sub(r'\n\s*\n+', '\n\n', cleaned)

    return cleaned.strip()


class StreamChunk:
    """Represents a chunk of streamed response."""
    def __init__(self, content: str = "", reasoning: str = "", tool_calls: Optional[List[Dict]] = None, finish_reason: Optional[str] = None):
        self.content = content
        self.reasoning = reasoning
        self.tool_calls = tool_calls
        self.finish_reason = finish_reason


class LLMProvider:
    """Provider wrapper to isolate model-specific behaviors."""
    def __init__(
        self,
        name: str,
        api_key: Optional[str],
        base_url: Optional[str],
        model: str
    ):
        self.name = name
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def supports_system_role(self) -> bool:
        return True

    def supports_tools(self) -> bool:
        return True

    def stream_chat(
        self,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = "auto"
    ) -> Iterator[StreamChunk]:
        return stream_chat_response(
            messages=messages,
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice
        )

    def chat_with_tools(
        self,
        messages: List[Dict],
        tools: List[Dict],
        temperature: float,
        max_tokens: int,
        tool_choice: str = "auto"
    ) -> Dict:
        return chat_response_with_tools(
            messages=messages,
            tools=tools,
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            tool_choice=tool_choice
        )


class DeepSeekProvider(LLMProvider):
    """DeepSeek-specific provider behavior."""
    def supports_system_role(self) -> bool:
        return "reasoner" not in self.model.lower()


def get_provider(
    provider_name: str,
    api_key: Optional[str],
    base_url: Optional[str],
    model: str
) -> LLMProvider:
    if provider_name.lower() == "deepseek":
        return DeepSeekProvider(provider_name, api_key, base_url, model)
    return LLMProvider(provider_name, api_key, base_url, model)


def stream_chat_response(
    messages: List[Dict],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 2000,
    tools: Optional[List[Dict]] = None,
    tool_choice: Optional[str] = "auto"
) -> Iterator[StreamChunk]:
    """
    Stream chat response from OpenAI API.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
                 Content can be a string or a list for multimodal (image) support.
        api_key: OpenAI API key (optional if set in environment)
        base_url: Custom API base URL for OpenAI-compatible services (e.g., Ollama, vLLM)
        model: Model name to use
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens to generate
        tools: Optional list of tool definitions for function calling
        tool_choice: Tool choice strategy ('auto', 'none', or specific tool)

    Yields:
        StreamChunk objects containing content and/or reasoning
    """
    client = get_client(api_key, base_url)

    # Check if this is a DeepSeek model (which uses DSML format)
    is_deepseek = "deepseek" in model.lower() or (base_url and "deepseek" in base_url.lower())

    # Log the request
    logger.log_llm_request(messages, model, temperature, max_tokens, tools, stream=True)

    try:
        # Build request parameters
        request_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }

        # Add tools if provided
        if tools:
            request_params["tools"] = tools
            if tool_choice:
                request_params["tool_choice"] = tool_choice

        response = client.chat.completions.create(**request_params)

        # Track accumulated content for DSML parsing
        accumulated_content = ""
        accumulated_reasoning = ""
        accumulated_tool_calls = {}
        finish_reason = None
        chunk_index = 0

        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta
                content = getattr(delta, 'content', None) or ""
                # DeepSeek R1 returns reasoning_content in delta
                reasoning = getattr(delta, 'reasoning_content', None) or ""

                # Log the chunk for debugging
                chunk_tool_calls = getattr(delta, 'tool_calls', None)
                logger.log_stream_chunk(
                    chunk_index=chunk_index,
                    content=content,
                    reasoning=reasoning,
                    tool_calls=chunk_tool_calls,
                    finish_reason=chunk.choices[0].finish_reason
                )
                chunk_index += 1

                # Accumulate for DSML parsing (DeepSeek format)
                if content:
                    accumulated_content += content
                if reasoning:
                    accumulated_reasoning += reasoning

                # Handle standard OpenAI tool calls in streaming mode
                tool_calls_chunk = getattr(delta, 'tool_calls', None)
                if tool_calls_chunk:
                    for tc in tool_calls_chunk:
                        index = tc.index if hasattr(tc, 'index') else 0
                        if index not in accumulated_tool_calls:
                            accumulated_tool_calls[index] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}

                        if hasattr(tc, 'id') and tc.id:
                            accumulated_tool_calls[index]['id'] = tc.id
                        if hasattr(tc, 'type') and tc.type:
                            accumulated_tool_calls[index]['type'] = tc.type
                        if hasattr(tc, 'function'):
                            func = tc.function
                            if hasattr(func, 'name') and func.name:
                                accumulated_tool_calls[index]['function']['name'] += func.name
                            if hasattr(func, 'arguments') and func.arguments:
                                accumulated_tool_calls[index]['function']['arguments'] += func.arguments

                # Check for finish reason
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

                # Stream in real-time for better UX
                # For DeepSeek with tools, skip real-time streaming to avoid duplication
                # (we'll yield the cleaned content after DSML parsing)
                if not (is_deepseek and tools):
                    if content or reasoning:
                        yield StreamChunk(content=content, reasoning=reasoning)

        # After streaming completes:
        # 1. Parse DSML for DeepSeek models
        # 2. Yield any accumulated tool calls

        # Log stream summary before any DSML processing
        logger.log_stream_summary(
            total_chunks=chunk_index,
            accumulated_content=accumulated_content,
            accumulated_reasoning=accumulated_reasoning,
            parsed_tool_calls=None  # Will be updated below
        )

        if is_deepseek and tools:
            # Parse DSML format tool calls from accumulated content
            dsml_tool_calls, cleaned_content = parse_dsml_tool_calls(accumulated_content)

            # Log the parsed result
            logger.log_stream_summary(
                total_chunks=chunk_index,
                accumulated_content=cleaned_content,
                accumulated_reasoning=accumulated_reasoning,
                parsed_tool_calls=dsml_tool_calls
            )

            # Also clean up any remaining DSML tags from reasoning
            cleaned_reasoning = clean_dsml_content(accumulated_reasoning) if accumulated_reasoning else ""

            # Yield the cleaned content as a single chunk
            if cleaned_content or cleaned_reasoning:
                yield StreamChunk(content=cleaned_content, reasoning=cleaned_reasoning)

            # Yield DSML tool calls if found
            if dsml_tool_calls:
                yield StreamChunk(tool_calls=dsml_tool_calls, finish_reason="tool_calls")
            # Otherwise yield standard tool calls if found
            elif accumulated_tool_calls and finish_reason == "tool_calls":
                tool_calls_list = [accumulated_tool_calls[i] for i in sorted(accumulated_tool_calls.keys())]
                yield StreamChunk(tool_calls=tool_calls_list, finish_reason="tool_calls")
        else:
            # Yield any accumulated tool calls
            if accumulated_tool_calls and finish_reason == "tool_calls":
                tool_calls_list = [accumulated_tool_calls[i] for i in sorted(accumulated_tool_calls.keys())]
                yield StreamChunk(tool_calls=tool_calls_list, finish_reason="tool_calls")


    except Exception as e:
        logger.log_error(e, context="stream_chat_response")
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


# Provider configurations
PROVIDERS = {
    "DeepSeek": {
        "base_url": "https://api.deepseek.com",
        "env_key": "DEEPSEEK_API_KEY",
        "default_models": ["deepseek-chat", "deepseek-reasoner"],
    },
    "Moonshot (Kimi)": {
        "base_url": "https://api.moonshot.cn/v1",
        "env_key": "MOONSHOT_API_KEY",
        "default_models": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "default_models": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4-turbo"],
    },
    "自定义": {
        "base_url": "",
        "env_key": "",
        "default_models": [],
    },
}


def chat_response_with_tools(
    messages: List[Dict],
    tools: List[Dict],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 2000,
    tool_choice: str = "auto"
) -> Dict:
    """
    Get chat response with function calling support (non-streaming).

    Args:
        messages: List of message dicts
        tools: List of tool/function definitions
        api_key: API key
        base_url: Custom base URL
        model: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        tool_choice: Tool choice strategy

    Returns:
        Dict with 'content', 'tool_calls', 'finish_reason', etc.
    """
    client = get_client(api_key, base_url)

    # Log the request
    logger.log_llm_request(messages, model, temperature, max_tokens, tools, stream=False)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens
        )

        message = response.choices[0].message
        result = {
            "content": message.content or "",
            "tool_calls": [],
            "finish_reason": response.choices[0].finish_reason
        }

        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]

        # Log the response
        logger.log_non_streaming_response(
            content=result["content"],
            tool_calls=result["tool_calls"],
            finish_reason=result["finish_reason"]
        )

        return result

    except Exception as e:
        logger.log_error(e, context="chat_response_with_tools")
        return {"content": f"Error: {str(e)}", "tool_calls": [], "finish_reason": "error"}


def get_available_models(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> List[str]:
    """
    Fetch available models from the API.

    Args:
        api_key: API key
        base_url: Custom base URL

    Returns:
        List of model IDs
    """
    try:
        client = get_client(api_key, base_url)
        models = client.models.list()
        return sorted([m.id for m in models.data])
    except Exception as e:
        # Return empty list if API call fails
        return []
