"""
Streamlit Chatbot Application
Supports OpenAI API, multi-turn conversations, chat history persistence, and file upload.
"""
import streamlit as st
import os
from typing import Optional, Tuple

import database
import llm
import file_processor
import chart_tools
import tool_executor
import logger

# Page configuration
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
database.init_database()

# Initialize session state
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "base_url" not in st.session_state:
    st.session_state.base_url = "https://api.deepseek.com"
if "provider" not in st.session_state:
    st.session_state.provider = "DeepSeek"
if "available_models" not in st.session_state:
    st.session_state.available_models = llm.PROVIDERS["DeepSeek"]["default_models"]
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "models_fetched_for" not in st.session_state:
    st.session_state.models_fetched_for = None
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 2000
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = ""


def get_model_default_max_tokens(model: str) -> int:
    """Get recommended max_tokens for a specific model."""
    model_lower = model.lower()

    # DeepSeek models
    if "deepseek-reasoner" in model_lower or "reasoner" in model_lower:
        return 32000  # Reasoner needs more tokens for CoT
    if "deepseek-chat" in model_lower:
        return 4096

    # OpenAI models
    if "gpt-4o" in model_lower or "gpt-4-turbo" in model_lower:
        return 4096
    if "gpt-3.5-turbo" in model_lower:
        return 2048

    # Kimi models
    if "moonshot" in model_lower:
        if "128k" in model_lower:
            return 8192
        if "32k" in model_lower:
            return 4096
        return 2048  # 8k model

    # Claude models (if supported)
    if "claude" in model_lower:
        return 4096

    # Default for unknown models
    return 2000


def parse_message_content(content: str) -> dict:
    """Parse message content to extract reasoning if present."""
    import re
    reasoning_pattern = r'\[REASONING\](.*?)\[/REASONING\]'
    match = re.search(reasoning_pattern, content, re.DOTALL)

    if match:
        reasoning = match.group(1).strip()
        # Remove the reasoning tag from content
        clean_content = re.sub(reasoning_pattern, '', content, flags=re.DOTALL).strip()
        return {"content": clean_content, "reasoning": reasoning}
    return {"content": content}


def load_conversation(conversation_id: int):
    """Load a conversation into session state."""
    st.session_state.current_conversation_id = conversation_id
    raw_messages = database.get_conversation_messages(conversation_id)

    # Parse messages to extract reasoning
    parsed_messages = []
    for msg in raw_messages:
        parsed_msg = {"role": msg["role"], "content": msg["content"]}
        if msg["role"] == "assistant":
            parsed = parse_message_content(msg["content"])
            parsed_msg["content"] = parsed["content"]
            if "reasoning" in parsed:
                parsed_msg["reasoning"] = parsed["reasoning"]
        parsed_messages.append(parsed_msg)

    st.session_state.messages = parsed_messages


def start_new_conversation():
    """Start a new conversation."""
    st.session_state.current_conversation_id = None
    st.session_state.messages = []


def save_current_conversation():
    """Save the current conversation to database."""
    if st.session_state.messages and st.session_state.current_conversation_id:
        # Update conversation title if needed (based on first user message)
        user_messages = [m for m in st.session_state.messages if m["role"] == "user"]
        if user_messages:
            title = database.generate_conversation_title(user_messages[0]["content"])
            database.update_conversation_title(st.session_state.current_conversation_id, title)


def render_sidebar() -> Tuple[str, float, int, Optional[str]]:
    """Render the sidebar with settings and history."""
    with st.sidebar:
        st.title("⚙️ Settings")

        # Provider selection
        provider = st.selectbox(
            "服务商",
            options=list(llm.PROVIDERS.keys()),
            index=list(llm.PROVIDERS.keys()).index(st.session_state.provider),
            help="选择 AI 服务商"
        )

        # Update provider and related settings when changed
        if provider != st.session_state.provider:
            st.session_state.provider = provider
            provider_config = llm.PROVIDERS[provider]
            st.session_state.base_url = provider_config["base_url"]
            st.session_state.api_key = os.getenv(provider_config["env_key"], "")
            st.session_state.available_models = provider_config["default_models"]
            st.session_state.selected_model = None  # Reset selected model
            st.session_state.models_fetched_for = None  # Reset fetch flag
            st.rerun()

        provider_config = llm.PROVIDERS[provider]

        # API Key input
        env_key = provider_config["env_key"]
        api_key = st.text_input(
            "API Key",
            type="password",
            value=st.session_state.api_key,
            placeholder=f"从 {env_key} 环境变量读取" if env_key else "输入 API Key",
            help=f"Leave empty to use {env_key} environment variable." if env_key else "输入 API Key"
        )
        st.session_state.api_key = api_key

        # Base URL display/input
        if provider == "自定义":
            base_url = st.text_input(
                "Base URL",
                value=st.session_state.base_url,
                placeholder="https://api.example.com/v1",
                help="自定义 API Base URL"
            )
            st.session_state.base_url = base_url
        else:
            base_url = provider_config["base_url"]
            st.info(f"Base URL: {base_url}")

        # Auto-fetch models when provider changes or on initial load
        actual_api_key = api_key or os.getenv(env_key, "")
        if (st.session_state.models_fetched_for != provider and
            actual_api_key and base_url and provider != "自定义"):
            with st.spinner("获取模型列表..."):
                models = llm.get_available_models(actual_api_key, base_url)
                if models:
                    st.session_state.available_models = models
                    st.session_state.models_fetched_for = provider
                    # Try to keep previous selection if available
                    if st.session_state.selected_model in models:
                        st.rerun()

        # Manual refresh button
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("**模型列表**")
        with col2:
            if st.button("🔄", help="刷新模型列表"):
                if actual_api_key and base_url:
                    with st.spinner("获取中..."):
                        models = llm.get_available_models(actual_api_key, base_url)
                        if models:
                            st.session_state.available_models = models
                            st.session_state.models_fetched_for = provider
                            st.success(f"获取 {len(models)} 个模型")
                            st.rerun()
                        else:
                            st.error("获取失败，使用默认列表")
                else:
                    st.error("请先配置 API Key")

        # Model selection
        available_models = st.session_state.available_models or provider_config["default_models"]

        # Determine index for model selectbox
        if st.session_state.selected_model and st.session_state.selected_model in available_models:
            model_index = available_models.index(st.session_state.selected_model)
        else:
            model_index = 0

        current_model = st.selectbox(
            "选择模型",
            options=available_models,
            index=model_index,
            help="选择要使用的模型"
        )

        # Update max_tokens when model changes
        if current_model != st.session_state.selected_model:
            st.session_state.selected_model = current_model
            recommended_tokens = get_model_default_max_tokens(current_model)
            st.session_state.max_tokens = recommended_tokens
            st.rerun()

        st.session_state.selected_model = current_model

        # Temperature slider
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Higher values make output more random, lower values more deterministic"
        )

        # Determine max tokens range based on model
        max_tokens_limit = 64000 if "reasoner" in current_model.lower() else 16000
        default_tokens = get_model_default_max_tokens(current_model)

        # Max tokens slider
        max_tokens = st.slider(
            "Max Tokens",
            min_value=100,
            max_value=max_tokens_limit,
            value=st.session_state.max_tokens if st.session_state.max_tokens <= max_tokens_limit else default_tokens,
            step=100,
            help=f"Maximum number of tokens to generate. Recommended: {default_tokens} for {current_model}"
        )
        st.session_state.max_tokens = max_tokens

        # System Prompt (collapsible)
        # Show indicator if system prompt is set
        system_prompt_set = bool(st.session_state.system_prompt.strip())
        expander_label = "📝 System Prompt (已设置)" if system_prompt_set else "📝 System Prompt (可选)"

        with st.expander(expander_label, expanded=False):
            system_prompt = st.text_area(
                "自定义系统提示词",
                value=st.session_state.system_prompt,
                placeholder="你是一个有帮助的助手...",
                help="设置 AI 的角色和行为方式。留空则使用默认设置。",
                height=100,
                key="system_prompt_input"
            )
            st.session_state.system_prompt = system_prompt

            # Show status
            if system_prompt_set:
                st.success(f"✅ 已启用: {st.session_state.system_prompt[:30]}...")
            else:
                st.info("使用默认系统提示词")

            # Quick presets
            preset_col1, preset_col2 = st.columns(2)
            with preset_col1:
                if st.button("🎯 代码助手", use_container_width=True):
                    st.session_state.system_prompt = "You are an expert programmer. Provide clean, well-documented code with explanations."
                    st.rerun()
            with preset_col2:
                if st.button("📊 图表助手", use_container_width=True):
                    st.session_state.system_prompt = chart_tools.CHART_ASSISTANT_PROMPT
                    st.rerun()

            preset_col3, preset_col4 = st.columns(2)
            with preset_col3:
                if st.button("📝 翻译助手", use_container_width=True):
                    st.session_state.system_prompt = "You are a professional translator. Translate accurately while maintaining the original meaning and tone."
                    st.rerun()
            with preset_col4:
                if st.button("🧹 清除", use_container_width=True):
                    st.session_state.system_prompt = ""
                    st.rerun()

        st.divider()

        # New conversation button
        if st.button("➕ New Conversation", use_container_width=True):
            save_current_conversation()
            start_new_conversation()
            st.rerun()

        st.divider()

        # Conversation history
        st.subheader("📜 History")

        conversations = database.list_conversations()

        if not conversations:
            st.info("No conversations yet")
        else:
            for conv in conversations:
                col1, col2 = st.columns([4, 1])

                with col1:
                    # Highlight current conversation
                    is_current = conv["id"] == st.session_state.current_conversation_id
                    button_type = "primary" if is_current else "secondary"

                    if st.button(
                        conv["title"],
                        key=f"conv_{conv['id']}",
                        use_container_width=True,
                        type=button_type
                    ):
                        save_current_conversation()
                        load_conversation(conv["id"])
                        st.rerun()

                with col2:
                    if st.button("🗑️", key=f"del_{conv['id']}", help="Delete conversation"):
                        database.delete_conversation(conv["id"])
                        if st.session_state.current_conversation_id == conv["id"]:
                            start_new_conversation()
                        st.rerun()

        return current_model, temperature, max_tokens, base_url


def render_chat_interface(model: str, temperature: float, max_tokens: int, base_url: Optional[str]):
    """Render the main chat interface."""
    # Header
    if st.session_state.current_conversation_id:
        conv = database.get_conversation(st.session_state.current_conversation_id)
        title = conv["title"] if conv else "Chat"
    else:
        title = "New Conversation"

    st.title(f"💬 {title}")

    # File uploader (supports both text and images)
    uploaded_file = st.file_uploader(
        "📎 Upload a file (optional)",
        type=list(file_processor.SUPPORTED_EXTENSIONS),
        help="Upload a text file or image to include in the conversation"
    )

    # Display file info if uploaded
    file_content = None
    image_data = None
    is_image = False

    if uploaded_file is not None:
        filename = uploaded_file.name

        # Check if it's an image
        if file_processor.is_image_file(filename):
            is_image = True
            image_data, error = file_processor.read_image_as_base64(uploaded_file, filename)
            if error:
                st.error(error)
                image_data = None
            else:
                # Display image preview
                with st.expander(f"🖼️ {filename} (Image)", expanded=True):
                    st.image(uploaded_file, use_container_width=True)
        else:
            # Text file
            content, error = file_processor.extract_text_from_file(uploaded_file, filename)
            if error:
                st.error(error)
            else:
                file_content = content
                with st.expander(f"📄 {filename} ({len(content)} characters)"):
                    st.code(content[:2000] + ("..." if len(content) > 2000 else ""))

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # If assistant message has reasoning, display it with collapsible component
            if message["role"] == "assistant" and message.get("reasoning"):
                st.html(
                    f"""
                    <style>
                        .thinking-container {{
                            margin-bottom: 12px;
                            border: 1px solid #e0e3e7;
                            border-radius: 8px;
                            overflow: hidden;
                        }}
                        .thinking-header {{
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white;
                            padding: 10px 16px;
                            cursor: pointer;
                            display: flex;
                            align-items: center;
                            gap: 8px;
                            font-weight: 500;
                            user-select: none;
                        }}
                        .thinking-header:hover {{
                            background: linear-gradient(135deg, #5a6fd6 0%, #6a4190 100%);
                        }}
                        .thinking-header::before {{
                            content: "🧠";
                        }}
                        .thinking-content {{
                            background-color: #f8f9fa;
                            padding: 16px;
                            color: #555;
                            line-height: 1.6;
                            white-space: pre-wrap;
                            font-size: 0.95em;
                            border-top: 1px solid #e0e3e7;
                        }}
                        details > summary {{
                            list-style: none;
                        }}
                        details > summary::-webkit-details-marker {{
                            display: none;
                        }}
                        details > summary::after {{
                            content: "▲";
                            margin-left: auto;
                            font-size: 0.8em;
                            transition: transform 0.2s;
                        }}
                        details[open] > summary::after {{
                            content: "▼";
                        }}
                    </style>
                    <details class="thinking-container" open>
                        <summary class="thinking-header">Thinking Process</summary>
                        <div class="thinking-content">{message['reasoning']}</div>
                    </details>
                    """
                )
            # Display message content
            st.markdown(message["content"])
            # Display attached image if present
            if message.get("image"):
                st.image(message["image"], use_container_width=True)

    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Create conversation if needed
        if st.session_state.current_conversation_id is None:
            title = database.generate_conversation_title(prompt)
            conv_id = database.create_conversation(title)
            st.session_state.current_conversation_id = conv_id

        # Prepare user message with file content
        display_message = prompt
        if file_content:
            display_message = f"{prompt}\n\n*[File attached: {uploaded_file.name}]*"
        elif image_data:
            display_message = f"{prompt}\n\n*[Image attached: {uploaded_file.name}]*"

        # Add user message to UI
        with st.chat_message("user"):
            st.markdown(display_message)
            if image_data:
                st.image(uploaded_file, use_container_width=True)

        # Save user message to database (text only, image reference as text)
        database.save_message(
            st.session_state.current_conversation_id,
            "user",
            display_message
        )

        # Update session state
        message_data = {"role": "user", "content": display_message}
        if image_data:
            message_data["image"] = image_data  # Store base64 for display
        st.session_state.messages.append(message_data)

        # Get API key
        provider_config = llm.PROVIDERS.get(st.session_state.provider, {})
        env_key = provider_config.get("env_key", "")
        api_key = st.session_state.api_key or (os.getenv(env_key) if env_key else "")

        if not api_key:
            with st.chat_message("assistant"):
                st.error("Please provide a DeepSeek API key in the sidebar or set DEEPSEEK_API_KEY environment variable.")
        else:
            # Prepare provider and messages for API
            provider = llm.get_provider(
                st.session_state.provider,
                api_key,
                base_url if base_url else None,
                model
            )

            # We need to reconstruct the history with the full content (including file/image)
            api_messages = []

            # Add system prompt if set
            system_prompt = st.session_state.system_prompt.strip()
            system_prefix = ""
            system_prefix_pending = False
            if system_prompt:
                if provider.supports_system_role():
                    api_messages.append({"role": "system", "content": system_prompt})
                    st.info(f"📝 使用 System Prompt: {system_prompt[:50]}...")
                else:
                    system_prefix = f"[System Instruction: {system_prompt}]\n\n"
                    system_prefix_pending = True
                    st.info(f"📝 System Prompt 已合并到消息: {system_prompt[:50]}...")

            def apply_system_prefix(content: str) -> str:
                nonlocal system_prefix_pending
                if system_prefix_pending:
                    system_prefix_pending = False
                    return system_prefix + content
                return content

            # Add historical messages
            for msg in st.session_state.messages[:-1]:  # All except current
                if msg.get("image"):
                    text_content = msg["content"]
                    if msg["role"] == "user":
                        text_content = apply_system_prefix(text_content)
                    # Multimodal format for messages with images
                    api_messages.append({
                        "role": msg["role"],
                        "content": [
                            {"type": "text", "text": text_content},
                            {"type": "image_url", "image_url": {"url": msg["image"]}}
                        ]
                    })
                else:
                    content = msg["content"]
                    if msg["role"] == "user":
                        content = apply_system_prefix(content)
                    api_messages.append({"role": msg["role"], "content": content})

            # Add current user message with file/image content
            if image_data:
                text_prompt = apply_system_prefix(prompt)
                # Multimodal format for image
                api_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {"type": "image_url", "image_url": {"url": image_data}}
                    ]
                })
            elif file_content:
                # Text file content appended to message
                api_message = f"{prompt}\n\n[File Content from {uploaded_file.name}]:\n{file_content}"
                api_message = apply_system_prefix(api_message)
                api_messages.append({"role": "user", "content": api_message})
            else:
                # Plain text message
                api_messages.append({"role": "user", "content": apply_system_prefix(prompt)})

            # Get streaming response
            with st.chat_message("assistant"):
                import time

                full_response = ""
                full_reasoning = ""
                has_reasoning = False
                tool_calls = None

                # Use a single container for all streaming content to reduce rerenders
                stream_container = st.empty()
                last_update_time = time.time()
                update_interval = 0.05  # Update at most 20 times per second
                pending_update = False

                def render_stream_content():
                    """Render current streaming state."""
                    content_parts = []

                    # Add reasoning section if present
                    if full_reasoning:
                        content_parts.append(
                            f'<div style="margin-bottom: 12px; border: 1px solid #e0e3e7; border-radius: 8px; overflow: hidden;">'
                            f'<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 10px 16px; font-weight: 500;">🧠 Thinking Process</div>'
                            f'<div style="background-color: #f8f9fa; padding: 16px; color: #555; line-height: 1.6; white-space: pre-wrap; font-size: 0.95em; border-top: 1px solid #e0e3e7;">{full_reasoning}</div>'
                            f'</div>'
                        )

                    # Add main content
                    if full_response:
                        content_parts.append(f'<div style="line-height: 1.6;">{full_response}</div>')

                    return '\n'.join(content_parts)

                # Check if user wants charts
                needs_chart = chart_tools.should_use_chart_tools(prompt)

                tool_registry = tool_executor.ToolRegistry()
                tool_registry.register_toolset(chart_tools.CHART_TOOLS, chart_tools.CHART_TOOL_HANDLERS)
                executor = tool_executor.ToolExecutor(tool_registry)

                # Stream the response with potential tool calls
                for chunk in provider.stream_chat(
                    messages=api_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tool_registry.definitions() if needs_chart else None,
                    tool_choice="auto" if needs_chart else None
                ):
                    current_time = time.time()

                    # Accumulate content
                    if chunk.reasoning:
                        has_reasoning = True
                        full_reasoning += chunk.reasoning
                        pending_update = True

                    if chunk.content:
                        full_response += chunk.content
                        pending_update = True

                    # Capture tool calls
                    if chunk.tool_calls:
                        tool_calls = chunk.tool_calls

                    if pending_update and (current_time - last_update_time >= update_interval):
                        stream_container.markdown(render_stream_content(), unsafe_allow_html=True)
                        last_update_time = current_time
                        pending_update = False

                # Final update to ensure all content is displayed
                if pending_update:
                    stream_container.markdown(render_stream_content(), unsafe_allow_html=True)

                # Handle tool calls if present
                if tool_calls:
                    st.markdown("🛠️ **执行图表生成...**")

                    # Log tool calls before execution
                    logger.log_tool_call_execution(tool_calls, [])

                    executions = executor.execute_calls(tool_calls)
                    api_messages.extend(executor.build_tool_messages(executions))

                    # Log tool execution results
                    logger.log_tool_call_execution(tool_calls, executions)

                    # Get final response after tool execution
                    st.markdown("✅ **图表已生成，正在生成解释...**")
                    final_response = provider.chat_with_tools(
                        messages=api_messages,
                        tools=tool_registry.definitions(),
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tool_choice="none"  # Don't call more tools
                    )

                    if final_response["content"]:
                        # Clean any DSML tags from final response too
                        clean_final = llm.clean_dsml_content(final_response["content"])
                        st.markdown(clean_final)
                        full_response += "\n\n" + clean_final

                # Update session state with both reasoning and content
                message_data = {
                    "role": "assistant",
                    "content": full_response,
                }
                if full_reasoning.strip():
                    message_data["reasoning"] = full_reasoning
                st.session_state.messages.append(message_data)

                # Save assistant response to database (store reasoning + content)
                content_to_save = full_response
                if full_reasoning.strip():
                    content_to_save = f"[REASONING]{full_reasoning}[/REASONING]\n\n{full_response}"
                database.save_message(
                    st.session_state.current_conversation_id,
                    "assistant",
                    content_to_save
                )

        # Rerun to update UI (clear file uploader, etc.)
        st.rerun()


def main():
    """Main application entry point."""
    # Render sidebar and get settings
    model, temperature, max_tokens, base_url = render_sidebar()

    # Render chat interface
    render_chat_interface(model, temperature, max_tokens, base_url)


if __name__ == "__main__":
    main()
