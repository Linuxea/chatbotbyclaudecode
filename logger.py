"""
Debug logging module for LLM interactions.
Logs all requests and responses to a file for troubleshooting.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

# Configure logging
LOG_FILE = os.path.join(os.path.dirname(__file__), "llm_debug.log")

# Create logger
logger = logging.getLogger("llm_debug")
logger.setLevel(logging.DEBUG)

# File handler with append mode
file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)

# Formatter
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s\n%(message)s\n{'='*80}\n"
)
file_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(file_handler)


def _serialize_messages(messages: List[Dict]) -> str:
    """Serialize messages for logging, truncating long content."""
    serialized = []
    for msg in messages:
        copied = msg.copy()
        if "content" in copied and isinstance(copied["content"], str):
            content = copied["content"]
            if len(content) > 500:
                copied["content"] = content[:250] + "... [truncated] ..." + content[-250:]
        serialized.append(copied)
    return json.dumps(serialized, ensure_ascii=False, indent=2)


def log_llm_request(
    messages: List[Dict],
    model: str,
    temperature: float,
    max_tokens: int,
    tools: Optional[List[Dict]] = None,
    stream: bool = False
) -> None:
    """Log LLM API request."""
    log_data = {
        "type": "REQUEST",
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
        "messages": json.loads(_serialize_messages(messages))
    }
    if tools:
        log_data["tools"] = tools

    logger.debug(json.dumps(log_data, ensure_ascii=False, indent=2))


def log_stream_chunk(
    chunk_index: int,
    content: str,
    reasoning: str,
    tool_calls: Optional[Any] = None,
    finish_reason: Optional[str] = None
) -> None:
    """Log a single chunk from streaming response."""
    # Convert tool_calls to serializable format
    serializable_tool_calls = None
    if tool_calls:
        serializable_tool_calls = []
        for tc in tool_calls:
            try:
                tc_dict = {
                    "id": getattr(tc, 'id', None),
                    "type": getattr(tc, 'type', None),
                    "function": {
                        "name": getattr(getattr(tc, 'function', None), 'name', None),
                        "arguments": getattr(getattr(tc, 'function', None), 'arguments', None)
                    } if hasattr(tc, 'function') else None
                }
                serializable_tool_calls.append(tc_dict)
            except Exception:
                serializable_tool_calls.append(str(tc))

    log_data = {
        "type": "STREAM_CHUNK",
        "chunk_index": chunk_index,
        "content": content if content else None,
        "reasoning": reasoning if reasoning else None,
        "tool_calls": serializable_tool_calls,
        "finish_reason": finish_reason
    }
    logger.debug(json.dumps(log_data, ensure_ascii=False, indent=2))


def log_stream_summary(
    total_chunks: int,
    accumulated_content: str,
    accumulated_reasoning: str,
    parsed_tool_calls: Optional[List[Dict]] = None
) -> None:
    """Log summary of streaming response."""
    log_data = {
        "type": "STREAM_SUMMARY",
        "total_chunks": total_chunks,
        "accumulated_content_length": len(accumulated_content),
        "accumulated_content_preview": accumulated_content[:500] if accumulated_content else None,
        "accumulated_reasoning_length": len(accumulated_reasoning),
        "accumulated_reasoning_preview": accumulated_reasoning[:500] if accumulated_reasoning else None,
        "parsed_tool_calls": parsed_tool_calls
    }
    logger.debug(json.dumps(log_data, ensure_ascii=False, indent=2))


def log_tool_call_execution(
    tool_calls: List[Dict],
    results: List[Dict]
) -> None:
    """Log tool call execution."""
    log_data = {
        "type": "TOOL_CALL_EXECUTION",
        "tool_calls": tool_calls,
        "results": results
    }
    logger.debug(json.dumps(log_data, ensure_ascii=False, indent=2))


def log_non_streaming_response(
    content: str,
    tool_calls: Optional[List[Dict]] = None,
    finish_reason: Optional[str] = None
) -> None:
    """Log non-streaming response."""
    log_data = {
        "type": "NON_STREAMING_RESPONSE",
        "content_length": len(content),
        "content_preview": content[:1000] if content else None,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason
    }
    logger.debug(json.dumps(log_data, ensure_ascii=False, indent=2))


def log_error(error: Exception, context: str = "") -> None:
    """Log an error with context."""
    log_data = {
        "type": "ERROR",
        "context": context,
        "error_type": type(error).__name__,
        "error_message": str(error)
    }
    logger.error(json.dumps(log_data, ensure_ascii=False, indent=2))
