"""
File content extraction for text-based and image files.
"""
import os
import base64
from typing import Optional, Tuple, Union
from io import BytesIO

# Maximum file size in bytes (5MB)
MAX_FILE_SIZE = 5 * 1024 * 1024

# Supported text file extensions
TEXT_EXTENSIONS = {
    '.txt', '.md', '.markdown',
    '.py', '.js', '.ts', '.jsx', '.tsx',
    '.json', '.yaml', '.yml', '.toml',
    '.csv', '.tsv',
    '.html', '.htm', '.css', '.scss', '.sass',
    '.xml', '.sql',
    '.c', '.cpp', '.h', '.hpp', '.java', '.go', '.rs', '.rb', '.php',
    '.sh', '.bash', '.zsh', '.ps1',
    '.log', '.ini', '.cfg', '.conf',
}

# Supported image file extensions
IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff', '.tif'
}

# All supported extensions
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | IMAGE_EXTENSIONS


def get_file_extension(filename: str) -> str:
    """Get the lowercase file extension."""
    return os.path.splitext(filename)[1].lower()


def is_supported_file(filename: str) -> bool:
    """Check if file type is supported."""
    return get_file_extension(filename) in SUPPORTED_EXTENSIONS


def is_image_file(filename: str) -> bool:
    """Check if file is an image."""
    return get_file_extension(filename) in IMAGE_EXTENSIONS


def is_text_file(filename: str) -> bool:
    """Check if file is a text file."""
    return get_file_extension(filename) in TEXT_EXTENSIONS


def check_file_size(file_size: int) -> Tuple[bool, str]:
    """
    Check if file size is within limits.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if file_size > MAX_FILE_SIZE:
        return False, f"File too large ({file_size / 1024 / 1024:.1f}MB). Maximum size is {MAX_FILE_SIZE / 1024 / 1024}MB."
    return True, ""


def read_file_content(file_obj) -> Tuple[Optional[str], str]:
    """
    Read and return text content from a file object.

    Args:
        file_obj: A file-like object (e.g., from st.file_uploader)

    Returns:
        Tuple of (content, error_message)
        content is None if there was an error
    """
    try:
        # Check file size
        file_obj.seek(0, os.SEEK_END)
        file_size = file_obj.tell()
        file_obj.seek(0)

        is_valid, error_msg = check_file_size(file_size)
        if not is_valid:
            return None, error_msg

        # Read content
        content = file_obj.read()

        # Handle bytes
        if isinstance(content, bytes):
            # Try UTF-8 first
            try:
                content = content.decode('utf-8')
            except UnicodeDecodeError:
                # Try other encodings
                for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        content = content.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    return None, "Could not decode file content. Please ensure it's a valid text file."

        return content, ""

    except Exception as e:
        return None, f"Error reading file: {str(e)}"


def extract_text_from_file(file_obj, filename: str) -> Tuple[Optional[str], str]:
    """
    Extract text content from an uploaded file.

    Args:
        file_obj: A file-like object
        filename: Name of the file

    Returns:
        Tuple of (content, error_message)
    """
    # Check if file type is supported
    if not is_supported_file(filename):
        ext = get_file_extension(filename)
        supported = ', '.join(sorted(SUPPORTED_EXTENSIONS))
        return None, f"Unsupported file type '{ext}'. Supported types: {supported}"

    return read_file_content(file_obj)


def format_file_for_context(filename: str, content: str, max_length: int = 10000) -> str:
    """
    Format file content for inclusion in the conversation context.

    Args:
        filename: Name of the file
        content: File content
        max_length: Maximum length of content to include

    Returns:
        Formatted string
    """
    # Truncate if too long
    if len(content) > max_length:
        truncated = content[:max_length]
        return f"File: {filename}\n```\n{truncated}\n... (truncated, {len(content) - max_length} more characters)\n```"

    return f"File: {filename}\n```\n{content}\n```"


def read_image_as_base64(file_obj, filename: str) -> Tuple[Optional[str], str]:
    """
    Read an image file and return as base64 encoded string.

    Args:
        file_obj: A file-like object
        filename: Name of the file

    Returns:
        Tuple of (base64_string, error_message)
        base64_string includes data URI prefix (e.g., data:image/jpeg;base64,...)
    """
    try:
        # Check file size
        file_obj.seek(0, os.SEEK_END)
        file_size = file_obj.tell()
        file_obj.seek(0)

        is_valid, error_msg = check_file_size(file_size)
        if not is_valid:
            return None, error_msg

        # Check if it's an image file
        if not is_image_file(filename):
            return None, f"File '{filename}' is not a supported image format"

        # Read image bytes
        image_bytes = file_obj.read()

        # Determine MIME type
        ext = get_file_extension(filename)
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff',
        }
        mime_type = mime_types.get(ext, 'image/jpeg')

        # Encode to base64
        base64_data = base64.b64encode(image_bytes).decode('utf-8')
        data_uri = f"data:{mime_type};base64,{base64_data}"

        return data_uri, ""

    except Exception as e:
        return None, f"Error reading image: {str(e)}"
