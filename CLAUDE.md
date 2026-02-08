# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Streamlit-based AI chatbot application with multi-provider LLM support (DeepSeek, OpenAI, Moonshot/Kimi, and custom OpenAI-compatible APIs). It features persistent chat history via SQLite, file upload support (text and images), streaming responses, and reasoning display for models like DeepSeek-R1.

## Common Commands

### Run the Application
```bash
streamlit run app.py
```
The app starts at `http://localhost:8501` by default.

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Environment Setup
```bash
cp .env.example .env
# Edit .env with your API keys (DEEPSEEK_API_KEY, MOONSHOT_API_KEY, OPENAI_API_KEY)
```

## Architecture

### Module Structure

- **app.py**: Main Streamlit application with UI rendering, session state management, and chat interface
- **llm.py**: OpenAI API wrapper with streaming support, provider configurations, and model fetching
- **database.py**: SQLite operations for conversation and message persistence
- **file_processor.py**: Text and image file handling for multimodal conversations

### Key Architectural Patterns

**Session State Management**: The app relies heavily on Streamlit's session state (st.session_state) to track:
- `current_conversation_id`: Active conversation ID
- `messages`: Current conversation messages
- `api_key`, `base_url`, `provider`: API configuration
- `selected_model`, `available_models`: Model selection
- `system_prompt`: Custom system instructions

**Provider Configuration**: Providers are defined in `llm.py:PROVIDERS` dict with base URLs and environment variable keys. The "自定义" (Custom) provider allows arbitrary base URLs.

**Message Formatting**: The `llm.format_messages_for_api()` function handles combining system prompts, conversation history, user messages, and file content into the API-compatible format.

**Reasoning Extraction**: DeepSeek-R1 and similar reasoning models return `reasoning_content` in the delta. The app stores reasoning wrapped in `[REASONING]...[/REASONING]` tags in the database and parses it for display.

**Multimodal Support**: Images are converted to base64 data URIs via `file_processor.read_image_as_base64()` and included in message content as OpenAI-compatible image objects.

### Database Schema

SQLite database with two tables:
- `conversations`: id, title, created_at, updated_at
- `messages`: id, conversation_id, role, content, created_at

The database is initialized automatically on app startup via `database.init_database()`.

### Model-Specific Defaults

Default max_tokens vary by model (see `get_model_default_max_tokens()` in app.py):
- DeepSeek-Reasoner: 32000 (for chain-of-thought)
- DeepSeek-Chat: 4096
- GPT-4o/GPT-4-Turbo: 4096
- GPT-3.5-Turbo: 2048
- Moonshot 128k: 8192
- Moonshot 32k: 4096
- Moonshot 8k: 2048

## Adding Features

**New Provider**: Add entry to `PROVIDERS` dict in `llm.py` with base_url, env_key, and default_models.

**New File Type**: Extend `SUPPORTED_EXTENSIONS` in `file_processor.py` and add handling logic.

**UI Changes**: Modify `render_sidebar()` or `render_chat_interface()` in `app.py`.

**Database Changes**: Update schema in `database.init_database()` and migration logic if needed.
