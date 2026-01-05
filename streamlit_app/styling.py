"""Streamlit app utilities for IMSocio.

This module contains Streamlit-specific helper functions for the web interface.
"""
import streamlit as st
from pathlib import Path
import hashlib


def load_custom_css(css_file: str = "streamlit_app/static/styles.css"):
    """Load custom CSS styling for the Streamlit app.
    
    This function adds a cache-busting hash to force browser reload when CSS changes.
    
    Args:
        css_file: Relative path to CSS file from project root
    """
    try:
        # Navigate up to project root from this file
        app_root = Path(__file__).resolve().parents[1]
        css_path = (app_root / css_file).resolve()
        
        if not css_path.exists():
            st.warning(f"CSS file not found at {css_path}. Using default minimal styles.")
            st.markdown("<style>:root{--dummy:0}</style>", unsafe_allow_html=True)
            return

        css_text = css_path.read_text(encoding="utf-8")
        
        # Add cache-busting comment with file hash to force browser reload
        css_hash = hashlib.md5(css_text.encode()).hexdigest()[:8]
        css_with_bust = f"/* Cache-bust: {css_hash} */\n{css_text}"
        
        st.markdown(f"<style>{css_with_bust}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Failed to load CSS: {e}")
        st.markdown("<style>:root{--dummy:0}</style>", unsafe_allow_html=True)
