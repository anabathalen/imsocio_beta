"""Import and file handling utilities for Streamlit app.

Functions for handling file uploads, reading data, and managing temporary files.
"""
import os
import zipfile
import shutil
from pathlib import Path

import pandas as pd
import streamlit as st


def handle_zip_upload(uploaded_file):
    """Handle uploaded ZIP files and extract contents.
    
    Extracts uploaded ZIP file to a temporary directory and returns
    the subfolder names found within.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Tuple of (folders, temp_dir) where folders is a list of subfolder names
        and temp_dir is the path to the extraction directory
    """
    temp_dir = '/tmp/extracted_zip/'

    # Clean up existing temp directory
    if os.path.exists(temp_dir):
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    os.makedirs(temp_dir, exist_ok=True)

    # Extract ZIP file
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    # Find subfolders
    folders = [f for f in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, f))]
    if not folders:
        st.error("No folders found in the ZIP file :(")
    
    return folders, temp_dir


def read_bush():
    """Read and return the Bush calibrant database.
    
    Returns:
        DataFrame containing Bush calibrant data, or empty DataFrame on error
    """
    file_path = os.path.join(os.path.dirname(__file__), '../data/bush.csv')
    
    if os.path.exists(file_path):
        bush_df = pd.read_csv(file_path)
    else:
        st.error(f"'{file_path}' not found. Make sure 'bush.csv' is in the 'data' folder.")
        bush_df = pd.DataFrame()  # Empty dataframe if file not found
    
    return bush_df


def clear_cache():
    """Remove temporary data and cached files.
    
    Note: Does NOT call st.rerun() - caller should handle rerun if needed.
    """
    temp_dir = Path('/tmp/extracted_zip/')
    
    if temp_dir.exists() and temp_dir.is_dir():
        shutil.rmtree(temp_dir, ignore_errors=True)
