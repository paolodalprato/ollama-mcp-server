"""
Ollama MCP Server - Comprehensive Ollama Management via MCP

A specialized Model Context Protocol server for complete Ollama integration,
including model management, server control, and intelligent recommendations.

Version: 1.0.0
Author: Paolo Dalprato
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Paolo Dalprato"
__email__ = "paolo@dalprato.dev"
__license__ = "MIT"

from .client import OllamaClient
from .server_manager import OllamaServerManager
from .model_manager import ModelManager
from .job_manager import JobManager
from .hardware_checker import HardwareChecker

__all__ = [
    "OllamaClient",
    "OllamaServerManager", 
    "ModelManager",
    "JobManager",
    "HardwareChecker"
]
