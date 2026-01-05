"""
Extraction module for input file generation.
"""

from .input_generator import (
    InputParams,
    InputProcessingResult,
    InputProcessor,
)
from .output_processor import (
    ProteinOutput,
    OutputProcessingResult,
    OutputFileProcessor,
)

__all__ = [
    'InputParams',
    'InputProcessingResult',
    'InputProcessor',
    'ProteinOutput',
    'OutputProcessingResult',
    'OutputFileProcessor',
]
