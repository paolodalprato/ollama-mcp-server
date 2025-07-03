"""
Configuration - MCP Server v1.1 Simplified
Minimal configuration with environment variable support

Reduced from 400+ lines to 50 lines - only essentials
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class OllamaConfig:
    """Simple Ollama configuration"""
    host: str = "localhost"
    port: int = 11434
    timeout: int = 30
    
    @property
    def url(self) -> str:
        """Full server URL"""
        return f"http://{self.host}:{self.port}"

def get_config() -> OllamaConfig:
    """
    Load configuration from environment variables
    
    Supported env vars:
    - OLLAMA_HOST (default: localhost)
    - OLLAMA_PORT (default: 11434)  
    - OLLAMA_TIMEOUT (default: 30)
    """
    return OllamaConfig(
        host=os.getenv("OLLAMA_HOST", "localhost"),
        port=int(os.getenv("OLLAMA_PORT", "11434")),
        timeout=int(os.getenv("OLLAMA_TIMEOUT", "30"))
    )
