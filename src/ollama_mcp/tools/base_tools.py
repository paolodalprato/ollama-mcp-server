"""
Base Tools - MCP Server v0.9.3 Enhanced
Essential 4 tools with enhanced error handling and robustness

Tools:
1. list_local_models - List available models
2. local_llm_chat - Chat with local models  
3. ollama_health_check - Diagnose Ollama status
4. system_resource_check - Enhanced system info with robust GPU detection

v0.9.3 improvements:
- Robust GPU detection parsing (no more crashes on non-numeric VRAM)
- Enhanced error handling with specific exception types
- Increased timeouts for better reliability (5s -> 10s)
- Graceful degradation when GPU detection fails
- Better cross-platform compatibility
"""

import json
import psutil
from typing import Dict, Any, List
from mcp.types import Tool, TextContent

import sys
from pathlib import Path

# Fix import path
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from client import OllamaClient


def get_base_tools() -> List[Tool]:
    """Return list of base tools for MCP registration"""
    return [
        Tool(
            name="list_local_models",
            description="List all locally installed Ollama models with details",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="local_llm_chat", 
            description="Chat with a local Ollama model",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message to send to the model"
                    },
                    "model": {
                        "type": "string", 
                        "description": "Model name (optional, uses first available if not specified)"
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Generation temperature 0.0-1.0 (default: 0.7)"
                    }
                },
                "required": ["message"]
            }
        ),
        Tool(
            name="ollama_health_check",
            description="Check Ollama server health and provide diagnostics", 
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="system_resource_check",
            description="Check system resources and compatibility",
            inputSchema={
                "type": "object", 
                "properties": {},
                "required": []
            }
        )
    ]


async def handle_base_tool(name: str, arguments: Dict[str, Any], client: OllamaClient) -> List[TextContent]:
    """Handle base tool calls with resilient error handling"""
    
    if name == "list_local_models":
        return await _handle_list_models(client)
    elif name == "local_llm_chat":
        return await _handle_chat(arguments, client)
    elif name == "ollama_health_check":
        return await _handle_health_check(client)
    elif name == "system_resource_check":
        return await _handle_system_check()
    else:
        return [TextContent(
            type="text",
            text=f"Unknown base tool: {name}"
        )]


async def _handle_list_models(client: OllamaClient) -> List[TextContent]:
    """Handle list models with helpful diagnostics"""
    result = await client.list_models()
    
    if result["success"]:
        if result["models"]:
            # Format model list nicely
            model_info = {
                "success": True,
                "models": [
                    {
                        "name": model.name,
                        "size": model.size_human,
                        "modified": model.modified
                    }
                    for model in result["models"]
                ],
                "total_count": result["count"],
                "usage_tip": "Use 'local_llm_chat' tool to chat with any of these models"
            }
        else:
            model_info = {
                "success": True,
                "models": [],
                "total_count": 0,
                "message": "No models found locally",
                "next_steps": {
                    "download_model": "Download a model first: 'ollama pull llama3.2'",
                    "popular_models": ["llama3.2", "qwen2.5", "phi3.5", "mistral"]
                }
            }
    else:
        # Ollama not accessible - provide helpful guidance
        model_info = {
            "success": False,
            "error": result["error"],
            "models": [],
            "troubleshooting": {
                "check_server": "Use 'ollama_health_check' for detailed diagnosis",
                "start_ollama": "Try running: 'ollama serve' in terminal",
                "install_ollama": "Download from: https://ollama.com"
            }
        }
    
    return [TextContent(type="text", text=json.dumps(model_info, indent=2))]


async def _handle_chat(arguments: Dict[str, Any], client: OllamaClient) -> List[TextContent]:
    """Handle chat with automatic model selection"""
    message = arguments.get("message", "")
    model = arguments.get("model")
    temperature = arguments.get("temperature", 0.7)
    
    if not message:
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": "Message is required",
                "example": "Use: local_llm_chat with message='Hello, how are you?'"
            }, indent=2)
        )]
    
    # Auto-select model if not specified
    if not model:
        models_result = await client.list_models()
        if models_result["success"] and models_result["models"]:
            model = models_result["models"][0].name
        else:
            return [TextContent(
                type="text", 
                text=json.dumps({
                    "success": False,
                    "error": "No models available",
                    "user_message": message,
                    "next_steps": {
                        "download_model": "Download a model: 'ollama pull llama3.2'",
                        "check_server": "Verify Ollama is running: 'ollama_health_check'"
                    }
                }, indent=2)
            )]
    
    # Generate response
    result = await client.chat(model, message, temperature)
    
    if result["success"]:
        chat_result = {
            "success": True,
            "response": result["response"],
            "model_used": model,
            "user_message": message,
            "metadata": result.get("metadata", {}),
            "privacy_note": "All processing done locally - no data sent to cloud"
        }
    else:
        chat_result = {
            "success": False,
            "error": result["error"],
            "user_message": message,
            "model_requested": model,
            "troubleshooting": {
                "check_model": f"Verify '{model}' is available with 'list_local_models'",
                "check_server": "Check Ollama server with 'ollama_health_check'",
                "download_model": f"If model missing: 'ollama pull {model}'"
            }
        }
    
    return [TextContent(type="text", text=json.dumps(chat_result, indent=2))]


async def _handle_health_check(client: OllamaClient) -> List[TextContent]:
    """Comprehensive health check with actionable guidance"""
    health = await client.health_check()
    
    # Enhance health check with actionable guidance
    if health["healthy"]:
        health_result = {
            "status": "HEALTHY",
            "server_url": health["host"],
            "models_available": health.get("models_count", 0),
            "message": "Ollama server is running and responsive",
            "next_steps": {
                "list_models": "See available models: 'list_local_models'", 
                "start_chat": "Start chatting: 'local_llm_chat' with your message"
            }
        }
    else:
        health_result = {
            "status": "UNHEALTHY",
            "server_url": health["host"],
            "error": health["error"],
            "troubleshooting": {
                "step_1": "Check if Ollama is installed: 'ollama --version'",
                "step_2": "Start Ollama server: 'ollama serve'",
                "step_3": "Install if missing: https://ollama.com",
                "step_4": "Check firewall/antivirus blocking port 11434"
            },
            "quick_fixes": {
                "terminal_command": "ollama serve",
                "installation_url": "https://ollama.com"
            }
        }
    
    return [TextContent(type="text", text=json.dumps(health_result, indent=2))]


async def _handle_system_check() -> List[TextContent]:
    """Complete system resource check with GPU detection and cross-platform support"""
    try:
        # Get basic system info
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        
        # Cross-platform disk usage with specific error handling
        import platform
        try:
            if platform.system() == "Windows":
                disk = psutil.disk_usage('C:\\')
            else:
                disk = psutil.disk_usage('/')
        except OSError as e:
            # Handle disk access errors gracefully
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error": f"Disk access error: {e}",
                "error_type": "disk_access_error",
                "suggestion": "Check disk permissions or try different drive"
            }, indent=2))]
        
        # GPU detection with error isolation
        try:
            gpu_info = await _get_gpu_info()
        except Exception as e:
            # Don't fail entire system check if GPU detection fails
            gpu_info = {
                "gpu_count": 0,
                "gpus": [],
                "detection_method": "failed",
                "error": f"GPU detection failed: {e}"
            }
        
        system_info = {
            "success": True,
            "system_resources": {
                "cpu_cores": cpu_count,
                "total_memory_gb": round(memory.total / (1024**3), 1),
                "available_memory_gb": round(memory.available / (1024**3), 1),
                "memory_usage_percent": memory.percent,
                "disk_free_gb": round(disk.free / (1024**3), 1),
                "disk_total_gb": round(disk.total / (1024**3), 1)
            },
            "gpu_resources": gpu_info,
            "ai_readiness": {
                "memory_sufficient": memory.available > 4 * (1024**3),  # >4GB free
                "disk_sufficient": disk.free > 10 * (1024**3),  # >10GB free
                "cpu_adequate": cpu_count >= 4,
                "gpu_available": gpu_info.get("gpu_count", 0) > 0,
                "gpu_acceleration_ready": gpu_info.get("gpu_count", 0) > 0 and gpu_info.get("detection_method") not in ["none", "failed"]
            },
            "recommendations": {
                "minimum_ram": "4GB available for small models",
                "recommended_ram": "8GB+ for larger models", 
                "disk_space": "10GB+ free for model storage",
                "gpu_acceleration": "GPU detected - can accelerate model inference" if gpu_info.get("gpu_count", 0) > 0 else "No GPU detected - CPU-only processing"
            }
        }
        
    except psutil.AccessDenied as e:
        system_info = {
            "success": False,
            "error": f"Access denied to system resources: {e}",
            "error_type": "access_denied",
            "suggestion": "Try running with administrator/root privileges"
        }
    except psutil.NoSuchProcess as e:
        system_info = {
            "success": False,
            "error": f"Process monitoring error: {e}",
            "error_type": "process_error"
        }
    except psutil.Error as e:
        system_info = {
            "success": False,
            "error": f"System monitoring error: {e}",
            "error_type": "psutil_error"
        }
    except ImportError as e:
        system_info = {
            "success": False,
            "error": f"Missing system dependencies: {e}",
            "error_type": "import_error",
            "suggestion": "Install required system monitoring dependencies"
        }
    except Exception as e:
        system_info = {
            "success": False,
            "error": f"Unexpected system check error: {e}",
            "error_type": "general_error"
        }
    
    return [TextContent(type="text", text=json.dumps(system_info, indent=2))]


async def _get_gpu_info() -> dict:
    """Get GPU information using cross-platform commands with robust error handling"""
    import subprocess
    import platform
    
    gpu_info = {
        "gpu_count": 0,
        "gpus": [],
        "detection_method": "none"
    }
    
    try:
        if platform.system() == "Windows":
            # Windows: Use multiple detection methods for better accuracy
            gpu_info = await _windows_gpu_detection()
                
        else:
            # Linux/macOS: Try nvidia-smi first, then lspci
            gpu_info = await _unix_gpu_detection()
                    
    except subprocess.SubprocessError as e:
        gpu_info["error"] = f"Subprocess error during GPU detection: {e}"
    except OSError as e:
        gpu_info["error"] = f"OS error during GPU detection: {e}"
    except Exception as e:
        gpu_info["error"] = f"Unexpected error during GPU detection: {e}"
    
    return gpu_info


async def _windows_gpu_detection() -> dict:
    """Windows-specific GPU detection with multiple fallback methods"""
    import subprocess
    
    gpu_info = {
        "gpu_count": 0,
        "gpus": [],
        "detection_method": "none"
    }
    
    # Method 1: Try nvidia-smi first (most accurate for NVIDIA GPUs)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 2:
                        try:
                            gpu_name = parts[0].strip()
                            vram_str = parts[1].strip()
                            
                            import re
                            vram_match = re.match(r'^(\d+)', vram_str)
                            if vram_match:
                                vram_mb = int(vram_match.group(1))
                                vram_gb = round(vram_mb / 1024, 1)
                                
                                gpu_info["gpus"].append({
                                    "name": gpu_name
                                })
                                gpu_info["gpu_count"] += 1
                                
                        except (ValueError, IndexError) as e:
                            gpu_info.setdefault("parsing_errors", []).append(f"nvidia-smi parsing error: {e}")
                            continue
            
            if gpu_info["gpu_count"] > 0:
                gpu_info["detection_method"] = "nvidia-smi"
                return gpu_info
                
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
        # nvidia-smi not available, continue to fallback methods
        pass
    
    # Method 2: Fallback to wmic (less accurate VRAM but detects GPU names)
    try:
        result = subprocess.run(
            ["wmic", "path", "win32_VideoController", "get", "name,AdapterRAM"],
            capture_output=True, text=True, timeout=10
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines[1:]:  # Skip header
                if line.strip():
                    parts = line.strip().split(None, 1)  # Split on first whitespace
                    if len(parts) >= 2:
                        try:
                            import re
                            vram_str = parts[0].strip()
                            gpu_name = parts[1].strip()
                            
                            # Extract numeric value from VRAM string
                            vram_match = re.match(r'^(\d+)', vram_str)
                            if vram_match:
                                vram_bytes = int(vram_match.group(1))
                                
                                # Don't report VRAM values - they're unreliable across platforms
                                # GPU name is sufficient for capability assessment
                                gpu_info["gpus"].append({
                                    "name": gpu_name
                                })
                                gpu_info["gpu_count"] += 1
                            else:
                                # Handle completely non-numeric VRAM values
                                gpu_info["gpus"].append({
                                    "name": gpu_name,
                                    "vram_status": f"detection_failed_invalid_format: {vram_str}"
                                })
                                gpu_info["gpu_count"] += 1
                                
                        except (ValueError, IndexError) as e:
                            gpu_info.setdefault("parsing_errors", []).append(f"wmic parsing error: {e}")
                            continue
            
            gpu_info["detection_method"] = "wmic"
            return gpu_info
            
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError) as e:
        gpu_info["wmic_error"] = f"wmic detection failed: {e}"
    
    # Method 3: Try DirectX diagnostic (basic detection)
    try:
        result = subprocess.run(
            ["dxdiag", "/t", "temp_dxdiag.txt"],
            capture_output=True, text=True, timeout=15
        )
        
        # This would require parsing the temp file, but it's complex
        # For now, just mark that we tried
        gpu_info["dxdiag_attempted"] = True
        
    except Exception:
        pass
    
    return gpu_info


async def _unix_gpu_detection() -> dict:
    """Unix-specific GPU detection (Linux/macOS)"""
    import subprocess
    
    gpu_info = {
        "gpu_count": 0,
        "gpus": [],
        "detection_method": "none"
    }
    
    # Try nvidia-smi first
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 2:
                        try:
                            gpu_name = parts[0].strip()
                            vram_str = parts[1].strip()
                            
                            import re
                            vram_match = re.match(r'^(\d+)', vram_str)
                            if vram_match:
                                vram_mb = int(vram_match.group(1))
                                vram_gb = round(vram_mb / 1024, 1)
                                
                                gpu_info["gpus"].append({
                                    "name": gpu_name
                                })
                                gpu_info["gpu_count"] += 1
                            else:
                                gpu_info["gpus"].append({
                                    "name": gpu_name,
                                    "vram_status": f"detection_failed_invalid_format: {vram_str}"
                                })
                                gpu_info["gpu_count"] += 1
                                
                        except (ValueError, IndexError) as e:
                            gpu_info.setdefault("parsing_errors", []).append(f"nvidia-smi parsing error: {e}")
                            continue
            
            gpu_info["detection_method"] = "nvidia-smi"
            return gpu_info
            
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError) as e:
        # Fallback to lspci for basic detection
        try:
            result = subprocess.run(
                ["lspci", "-v"],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                gpu_lines = [line for line in result.stdout.split('\n') if 'VGA' in line or 'GPU' in line]
                gpu_info["gpu_count"] = len(gpu_lines)
                gpu_info["gpus"] = [{"name": "GPU detected via lspci"} for _ in gpu_lines]
                gpu_info["detection_method"] = "lspci"
                
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError) as e:
            gpu_info["fallback_error"] = f"lspci fallback failed: {e}"
    
    return gpu_info
