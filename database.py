"""
SQLite database operations for chat history persistence.
"""
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
import json

DB_PATH = "chat_history.db"


def get_connection() -> sqlite3.Connection:
    """Get a database connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initialize the database with required tables."""
    conn = get_connection()
    cursor = conn.cursor()

    # Create conversations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create messages table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()


def create_conversation(title: str = "New Conversation") -> int:
    """Create a new conversation and return its ID."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO conversations (title, updated_at) VALUES (?, ?)",
        (title, datetime.now())
    )
    conversation_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return conversation_id


def update_conversation_title(conversation_id: int, title: str):
    """Update the title of a conversation."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
        (title, datetime.now(), conversation_id)
    )

    conn.commit()
    conn.close()


def update_conversation_timestamp(conversation_id: int):
    """Update the updated_at timestamp of a conversation."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?",
        (datetime.now(), conversation_id)
    )

    conn.commit()
    conn.close()


def save_message(conversation_id: int, role: str, content: str) -> int:
    """Save a message to a conversation."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
        (conversation_id, role, content)
    )
    message_id = cursor.lastrowid

    # Update conversation timestamp
    cursor.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?",
        (datetime.now(), conversation_id)
    )

    conn.commit()
    conn.close()

    return message_id


def get_conversation_messages(conversation_id: int) -> List[Dict]:
    """Get all messages for a conversation."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
        (conversation_id,)
    )

    messages = [{"role": row["role"], "content": row["content"]} for row in cursor.fetchall()]

    conn.close()

    return messages


def list_conversations() -> List[Dict]:
    """List all conversations ordered by most recent."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
    )

    conversations = [
        {
            "id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"]
        }
        for row in cursor.fetchall()
    ]

    conn.close()

    return conversations


def get_conversation(conversation_id: int) -> Optional[Dict]:
    """Get a single conversation by ID."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
        (conversation_id,)
    )

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return {
        "id": row["id"],
        "title": row["title"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"]
    }


def delete_conversation(conversation_id: int):
    """Delete a conversation and all its messages."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    conn.commit()
    conn.close()


def generate_conversation_title(first_message: str, max_length: int = 30) -> str:
    """Generate a title from the first message."""
    # Take first line or first max_length characters
    title = first_message.strip().split('\n')[0]

    if len(title) > max_length:
        title = title[:max_length].rstrip() + "..."

    return title if title else "New Conversation"
