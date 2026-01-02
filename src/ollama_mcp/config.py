"""
Configuration - MCP Server v2.0 Refactored
Configuration with environment variable support and hardware settings.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class HardwareConfig:
    """Configuration for hardware checking."""
    enable_gpu_detection: bool = True
    gpu_memory_fraction: float = 0.9  # Fraction of GPU memory considered usable
    enable_cpu_fallback: bool = True
    memory_threshold_gb: float = 4.0 # Warn if system RAM is below this

@dataclass
class OllamaConfig:
    """Main configuration for the Ollama MCP server."""
    host: str = "localhost"
    port: int = 11434
    timeout: int = 60
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    
    @property
    def url(self) -> str:
        """Full server URL"""
        return f"http://{self.host}:{self.port}"

def get_config() -> OllamaConfig:
    """
    Load configuration from environment variables.
    
    Supported env vars:
    - OLLAMA_HOST
    - OLLAMA_PORT
    - OLLAMA_TIMEOUT
    - HARDWARE_ENABLE_GPU_DETECTION
    - HARDWARE_GPU_MEMORY_FRACTION
    - HARDWARE_ENABLE_CPU_FALLBACK
    - HARDWARE_MEMORY_THRESHOLD_GB
    """
    hardware_config = HardwareConfig(
        enable_gpu_detection=os.getenv("HARDWARE_ENABLE_GPU_DETECTION", "True").lower() in ("true", "1"),
        gpu_memory_fraction=float(os.getenv("HARDWARE_GPU_MEMORY_FRACTION", "0.9")),
        enable_cpu_fallback=os.getenv("HARDWARE_ENABLE_CPU_FALLBACK", "True").lower() in ("true", "1"),
        memory_threshold_gb=float(os.getenv("HARDWARE_MEMORY_THRESHOLD_GB", "4.0"))
    )

    return OllamaConfig(
        host=os.getenv("OLLAMA_HOST", "localhost"),
        port=int(os.getenv("OLLAMA_PORT", "11434")),
        timeout=int(os.getenv("OLLAMA_TIMEOUT", "60")),
        hardware=hardware_config
    )
