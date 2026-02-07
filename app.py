"""
Streamlit Chatbot Application
Supports OpenAI API, multi-turn conversations, chat history persistence, and file upload.
"""
import streamlit as st
import os
from typing import Optional

import database
import llm
import file_processor

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
    st.session_state.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")


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


def render_sidebar():
    """Render the sidebar with settings and history."""
    with st.sidebar:
        st.title("⚙️ Settings")

        # API Key input
        api_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            value=st.session_state.api_key,
            placeholder="sk-...",
            help="Your DeepSeek API key. Leave empty to use DEEPSEEK_API_KEY environment variable."
        )
        st.session_state.api_key = api_key

        # Base URL input for DeepSeek API
        base_url = st.text_input(
            "Base URL",
            value=st.session_state.base_url,
            placeholder="https://api.deepseek.com",
            help="DeepSeek API base URL (default: https://api.deepseek.com)"
        )
        st.session_state.base_url = base_url

        # Model selection
        model = st.selectbox(
            "Model",
            options=llm.AVAILABLE_MODELS,
            index=0,
            help="Select the OpenAI model to use"
        )

        # Temperature slider
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Higher values make output more random, lower values more deterministic"
        )

        # Max tokens slider
        max_tokens = st.slider(
            "Max Tokens",
            min_value=100,
            max_value=4000,
            value=2000,
            step=100,
            help="Maximum number of tokens to generate"
        )

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

        return model, temperature, max_tokens, base_url


def render_chat_interface(model: str, temperature: float, max_tokens: int, base_url: str):
    """Render the main chat interface."""
    # Header
    if st.session_state.current_conversation_id:
        conv = database.get_conversation(st.session_state.current_conversation_id)
        title = conv["title"] if conv else "Chat"
    else:
        title = "New Conversation"

    st.title(f"💬 {title}")

    # File uploader (only show when there's a message or at start)
    uploaded_file = st.file_uploader(
        "📎 Upload a file (optional)",
        type=list(file_processor.SUPPORTED_EXTENSIONS),
        help="Upload a text file to include its content in the conversation"
    )

    # Display file info if uploaded
    file_content = None
    if uploaded_file is not None:
        content, error = file_processor.extract_text_from_file(
            uploaded_file, uploaded_file.name
        )
        if error:
            st.error(error)
        else:
            file_content = content
            with st.expander(f"📄 {uploaded_file.name} ({len(content)} characters)"):
                st.code(content[:2000] + ("..." if len(content) > 2000 else ""))

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # If assistant message has reasoning, display it
            if message["role"] == "assistant" and message.get("reasoning"):
                st.markdown("**🤔 Thinking:**")
                st.markdown(
                    f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; color: #666;'>"
                    f"{message['reasoning']}"
                    f"</div>",
                    unsafe_allow_html=True
                )
                st.markdown("**💬 Response:**")
            st.markdown(message["content"])

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

        # Add user message to UI
        with st.chat_message("user"):
            st.markdown(display_message)

        # Save user message to database
        # If there's file content, we include it in what we send to the API but save the original
        api_message = prompt
        if file_content:
            api_message = f"{prompt}\n\n[File Content from {uploaded_file.name}]:\n{file_content}"

        database.save_message(
            st.session_state.current_conversation_id,
            "user",
            display_message
        )

        # Update session state
        st.session_state.messages.append({"role": "user", "content": display_message})

        # Get API key
        api_key = st.session_state.api_key or os.getenv("DEEPSEEK_API_KEY")

        if not api_key:
            with st.chat_message("assistant"):
                st.error("Please provide a DeepSeek API key in the sidebar or set DEEPSEEK_API_KEY environment variable.")
        else:
            # Prepare messages for API
            # We need to reconstruct the history with the full content (including file)
            api_messages = []
            for msg in st.session_state.messages[:-1]:  # All except current
                api_messages.append({"role": msg["role"], "content": msg["content"]})
            api_messages.append({"role": "user", "content": api_message})

            # Get streaming response
            with st.chat_message("assistant"):
                full_response = ""
                full_reasoning = ""
                has_reasoning = False

                # Create containers for reasoning and content
                reasoning_header = st.empty()
                reasoning_container = st.empty()
                content_header = st.empty()
                content_container = st.empty()

                # Stream the response
                for chunk in llm.stream_chat_response(
                    messages=api_messages,
                    api_key=api_key,
                    base_url=base_url if base_url else None,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                ):
                    # Accumulate reasoning content
                    if chunk.reasoning:
                        has_reasoning = True
                        full_reasoning += chunk.reasoning
                        reasoning_header.markdown("**🤔 Thinking:**")
                        reasoning_container.markdown(
                            f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; color: #666;'>"
                            f"{full_reasoning}"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                    # Accumulate main content
                    if chunk.content:
                        full_response += chunk.content
                        content_header.markdown("**💬 Response:**")
                        content_container.markdown(full_response)

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
