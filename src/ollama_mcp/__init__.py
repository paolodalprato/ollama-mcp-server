"""
Ollama MCP Server - Comprehensive Ollama Management via MCP

A specialized Model Context Protocol server for complete Ollama integration,
including model management, server control, and intelligent recommendations.

Version: 0.9.0
Author: Paolo Dalprato
License: MIT
"""

__version__ = "0.9.1"
__author__ = "Paolo Dalprato"
__email__ = "paolo@paolodalprato.com"
__license__ = "MIT"

from .client import OllamaClient
from .model_manager import ModelManager
from .hardware_checker import HardwareChecker

__all__ = [
    "OllamaClient",
    "ModelManager",
    "HardwareChecker"
]
