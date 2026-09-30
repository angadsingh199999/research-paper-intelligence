"""
Frontend entry point redirecting to the primary Streamlit application.
Usage:
    streamlit run frontend/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.ui.streamlit_app import *  # noqa: F401, F403
