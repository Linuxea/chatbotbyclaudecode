"""
Tool registry and execution utilities.
"""
import json
from typing import Any, Callable, Dict, List, Optional, Tuple


ToolHandler = Callable[..., Dict[str, Any]]


class ToolRegistry:
    """Registry for tool definitions and their handlers."""
    def __init__(self) -> None:
        self._definitions: List[Dict[str, Any]] = []
        self._handlers: Dict[str, ToolHandler] = {}

    def register_toolset(
        self,
        tool_definitions: List[Dict[str, Any]],
        handlers: Dict[str, ToolHandler]
    ) -> None:
        for definition in tool_definitions:
            name = definition.get("function", {}).get("name")
            if not name:
                continue
            self._definitions.append(definition)
            if name in handlers:
                self._handlers[name] = handlers[name]

    def definitions(self) -> List[Dict[str, Any]]:
        return list(self._definitions)

    def get_handler(self, name: str) -> Optional[ToolHandler]:
        return self._handlers.get(name)


class ToolExecutor:
    """Executes tool calls and builds tool response messages."""
    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def execute_call(self, tool_call: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        function = tool_call.get("function", {})
        function_name = function.get("name", "")
        arguments = function.get("arguments", "{}")

        try:
            parsed_args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            parsed_args = {}

        handler = self._registry.get_handler(function_name)
        if not handler:
            result = {"success": False, "error": f"Unknown tool: {function_name}"}
        else:
            try:
                result = handler(**parsed_args)
            except Exception as exc:
                result = {"success": False, "error": str(exc)}

        return tool_call, result

    def execute_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        results: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for tool_call in tool_calls:
            results.append(self.execute_call(tool_call))
        return results

    def build_tool_messages(self, executions: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        for tool_call, result in executions:
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call]
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.get("id", ""),
                "content": json.dumps(result)
            })
        return messages
