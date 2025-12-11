"""
HVAC Frontend Styles - Light Industrial Dashboard
=================================================
"""

import os

_BASE_DIR = os.path.dirname(__file__)


def _load_main_css() -> str:
    css_path = os.path.join(_BASE_DIR, "main.css")
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
    except FileNotFoundError:
        return ""
    return f"<style>{css}</style>"


MAIN_CSS = _load_main_css()
