"""
Utility helpers for loading external HTML/CSS assets and background images.
"""

from __future__ import annotations

import base64
import io
import os
from typing import Dict, Iterable, Optional

from PIL import Image


FRONTEND_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(FRONTEND_DIR)


def find_file(
    filename: str,
    search_dirs: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """
    Try to locate a file by name within known project directories.

    Args:
        filename: Name of the file to find (e.g. "zoom_ahu.html").
        search_dirs: Optional iterable of absolute directories to search.

    Returns:
        Absolute path to the first matching file, or None if not found.
    """
    candidate_dirs = list(search_dirs or [])
    candidate_dirs.extend([FRONTEND_DIR, PROJECT_ROOT])

    for directory in candidate_dirs:
        if not directory:
            continue
        candidate_path = os.path.join(directory, filename)
        if os.path.isfile(candidate_path):
            return candidate_path
    return None


def load_html_template(
    template_name: str,
    placeholders: Optional[Dict[str, str]] = None,
) -> str:
    """
    Load an HTML template file and optionally substitute placeholders.

    This reads the file each time it is called, so editing the HTML does not
    require restarting the Streamlit server—just trigger a rerun.

    Args:
        template_name: File name relative to the frontend directory.
        placeholders: Mapping of placeholder -> value. Each key is replaced
            as ``{{KEY}}`` within the template.

    Returns:
        Render-ready HTML string or empty string if file is missing.
    """
    template_path = find_file(template_name)
    if not template_path:
        return ""

    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()

    if placeholders:
        for key, value in placeholders.items():
            html = html.replace(f"{{{{{key}}}}}", value)
    return html


def get_html_section(template_html: str, section_id: str) -> str:
    """
    Extract a named section surrounded by HTML comments.

    Sections use the following format inside the template:
        <!-- section:hero:start --> ... <!-- section:hero:end -->

    Args:
        template_html: Raw HTML content.
        section_id: Section identifier used in the template comments.

    Returns:
        The inner HTML of the section, or empty string if not found.
    """
    start_marker = f"<!-- section:{section_id}:start -->"
    end_marker = f"<!-- section:{section_id}:end -->"
    start_idx = template_html.find(start_marker)
    end_idx = template_html.find(end_marker)

    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return ""
    start_idx += len(start_marker)
    return template_html[start_idx:end_idx].strip()


def set_background(image_path: str) -> str:
    """
    Build CSS to set the Streamlit background using the provided image.

    Args:
        image_path: Absolute path to the image file.

    Returns:
        CSS string ready to embed via st.markdown, or empty string if missing.
    """
    if not image_path or not os.path.isfile(image_path):
        return ""

    with Image.open(image_path) as img:
        buffer = io.BytesIO()
        fmt = "PNG" if img.format is None else img.format
        img.save(buffer, format=fmt)
        b64 = base64.b64encode(buffer.getvalue()).decode()

    css = f"""
    <style>
        .stApp {{
            background-image: url("data:image/{fmt.lower()};base64,{b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
    </style>
    """
    return css

