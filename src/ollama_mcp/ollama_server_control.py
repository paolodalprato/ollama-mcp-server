"""
Ollama Server Control Module
Manages starting, stopping, and monitoring the local Ollama server for MCP v0.9.
Based on the working Bridge v1.0 implementation.
"""

import asyncio
import logging
import os
import platform
import psutil
import subprocess
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

@dataclass
class OllamaProcessInfo:
    """Information about the Ollama process."""
    pid: Optional[int]
    status: str  # "running", "stopped", "unknown"
    port: int
    uptime_seconds: Optional[int]
    memory_mb: Optional[float]
    cpu_percent: Optional[float]


class OllamaServerController:
    """Controller to manage the Ollama server."""
    
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host
        self.port = 11434
        self.logger = logging.getLogger(__name__)
        
    async def get_server_status(self) -> OllamaProcessInfo:
        """Get detailed status of the Ollama server."""
        try:
            # Find the Ollama process
            ollama_pid = None
            ollama_process = None
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
                try:
                    if proc.info['name'] and 'ollama' in proc.info['name'].lower():
                        # Verify it's the server (not client commands)
                        cmdline = proc.info.get('cmdline', [])
                        if any('serve' in str(cmd).lower() for cmd in cmdline) or \
                           any('11434' in str(cmd) for cmd in cmdline):
                            ollama_pid = proc.info['pid']
                            ollama_process = proc
                            break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if ollama_process:
                try:
                    # Get detailed process info
                    memory_info = ollama_process.memory_info()
                    memory_mb = memory_info.rss / 1024 / 1024
                    cpu_percent = ollama_process.cpu_percent()
                    create_time = ollama_process.create_time()
                    uptime = int(time.time() - create_time)
                    
                    return OllamaProcessInfo(
                        pid=ollama_pid,
                        status="running",
                        port=self.port,
                        uptime_seconds=uptime,
                        memory_mb=memory_mb,
                        cpu_percent=cpu_percent
                    )
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    self.logger.warning(f"Could not get process details: {e}")
                    return OllamaProcessInfo(
                        pid=ollama_pid,
                        status="running",
                        port=self.port,
                        uptime_seconds=None,
                        memory_mb=None,
                        cpu_percent=None
                    )
            else:
                return OllamaProcessInfo(
                    pid=None,
                    status="stopped",
                    port=self.port,
                    uptime_seconds=None,
                    memory_mb=None,
                    cpu_percent=None
                )
                
        except Exception as e:
            self.logger.error(f"Error getting Ollama status: {e}")
            return OllamaProcessInfo(
                pid=None,
                status="unknown",
                port=self.port,
                uptime_seconds=None,
                memory_mb=None,
                cpu_percent=None
            )
    
    async def start_server(self) -> Dict[str, Any]:
        """Start the Ollama server."""
        try:
            # Check if already running
            status = await self.get_server_status()
            if status.status == "running":
                return {
                    "success": True,
                    "message": "Ollama server is already running.",
                    "pid": status.pid,
                    "already_running": True
                }
            
            # Determine start command based on OS
            if platform.system() == "Windows":
                # On Windows, 'ollama serve' starts in the background
                process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:
                # On Linux/macOS
                process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            
            # Wait for the server to start (max 10 seconds)
            for i in range(10):
                await asyncio.sleep(1)
                status = await self.get_server_status()
                if status.status == "running":
                    return {
                        "success": True,
                        "message": "Ollama server started successfully.",
                        "pid": status.pid,
                        "startup_time_seconds": i + 1
                    }
            
            # If it hasn't started within 10 seconds
            return {
                "success": False,
                "message": "Timeout: Ollama server did not start within 10 seconds.",
                "error": "startup_timeout"
            }
            
        except FileNotFoundError:
            return {
                "success": False,
                "message": "Ollama not found. Please ensure it is installed and in your system's PATH.",
                "error": "ollama_not_found",
                "install_guide": self._get_install_guide()
            }
        except Exception as e:
            self.logger.error(f"Error starting Ollama server: {e}")
            return {
                "success": False,
                "message": f"Error starting server: {str(e)}",
                "error": str(e)
            }
    
    async def stop_server(self) -> Dict[str, Any]:
        """Stop the Ollama server."""
        try:
            status = await self.get_server_status()
            
            if status.status != "running":
                return {
                    "success": True,
                    "message": "Ollama server is not running.",
                    "already_stopped": True
                }
            
            if status.pid is None:
                return {
                    "success": False,
                    "message": "Could not find the PID of the Ollama process.",
                    "error": "pid_not_found"
                }
            
            # Terminate the process
            try:
                process = psutil.Process(status.pid)
                process.terminate()
                
                # Wait for the process to terminate (max 5 seconds)
                for i in range(5):
                    await asyncio.sleep(1)
                    if not process.is_running():
                        return {
                            "success": True,
                            "message": "Ollama server stopped successfully.",
                            "shutdown_time_seconds": i + 1
                        }
                
                # If it didn't stop, force kill it
                process.kill()
                await asyncio.sleep(1)
                
                if not process.is_running():
                    return {
                        "success": True,
                        "message": "Ollama server stopped forcefully.",
                        "forced_shutdown": True
                    }
                else:
                    return {
                        "success": False,
                        "message": "Failed to stop the Ollama server.",
                        "error": "shutdown_failed"
                    }
                    
            except psutil.NoSuchProcess:
                return {
                    "success": True,
                    "message": "Ollama server was already stopped.",
                    "already_stopped": True
                }
                
        except Exception as e:
            self.logger.error(f"Error stopping Ollama server: {e}")
            return {
                "success": False,
                "message": f"Error stopping server: {str(e)}",
                "error": str(e)
            }
    
    async def restart_server(self) -> Dict[str, Any]:
        """Restart the Ollama server."""
        try:
            # First, stop the server
            stop_result = await self.stop_server()
            
            if not stop_result["success"] and not stop_result.get("already_stopped", False):
                return {
                    "success": False,
                    "message": "Could not stop the server for restart.",
                    "error": stop_result.get("error", "stop_failed")
                }
            
            # Wait a moment before restarting
            await asyncio.sleep(2)
            
            # Then, start it again
            start_result = await self.start_server()
            
            if start_result["success"]:
                return {
                    "success": True,
                    "message": "Ollama server restarted successfully.",
                    "restart_completed": True,
                    "new_pid": start_result.get("pid")
                }
            else:
                return {
                    "success": False,
                    "message": "Restart failed: could not start the server.",
                    "error": start_result.get("error", "restart_failed")
                }
                
        except Exception as e:
            self.logger.error(f"Error restarting Ollama server: {e}")
            return {
                "success": False,
                "message": f"Error restarting server: {str(e)}",
                "error": str(e)
            }
    
    def _get_install_guide(self) -> Dict[str, str]:
        """Returns an installation guide for the current OS."""
        os_name = platform.system()
        
        if os_name == "Windows":
            return {
                "download_url": "https://ollama.com/download/windows",
                "instructions": "1. Download Ollama for Windows from the link above.\n2. Run the installer.\n3. Restart your terminal.\n4. Type: ollama serve"
            }
        elif os_name == "Darwin":  # macOS
            return {
                "download_url": "https://ollama.com/download/mac",
                "instructions": "1. Download Ollama for macOS from the link above.\n2. Drag Ollama to your Applications folder.\n3. Open Terminal and type: ollama serve"
            }
        else:  # Linux
            return {
                "download_url": "https://ollama.com/download/linux",
                "instructions": "1. Run: curl -fsSL https://ollama.com/install.sh | sh\n2. Or download manually from the link above.\n3. Start with: ollama serve"
            }
    
    def format_uptime(self, seconds: Optional[int]) -> str:
        """Formats uptime in a human-readable way."""
        if seconds is None:
            return "N/A"
        
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes}m {secs}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    
    async def get_diagnostic_info(self) -> Dict[str, Any]:
        """Get comprehensive diagnostic information."""
        diagnostic = {
            "ollama_installed": False,
            "ollama_in_path": False,
            "server_running": False,
            "port_accessible": False,
            "system_resources": {},
            "recommendations": []
        }
        
        try:
            # Check if Ollama is installed
            result = subprocess.run(["ollama", "--version"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                diagnostic["ollama_installed"] = True
                diagnostic["ollama_in_path"] = True
                diagnostic["ollama_version"] = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            diagnostic["recommendations"].append("Install Ollama from the official website.")
        
        # Check server status
        status = await self.get_server_status()
        diagnostic["server_running"] = status.status == "running"
        
        if status.status == "running":
            diagnostic["server_info"] = {
                "pid": status.pid,
                "uptime": self.format_uptime(status.uptime_seconds),
                "memory_mb": status.memory_mb,
                "cpu_percent": status.cpu_percent
            }
        else:
            diagnostic["recommendations"].append("Start the Ollama server with: ollama serve")
        
        # System information
        try:
            diagnostic["system_resources"] = {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "memory_available_gb": psutil.virtual_memory().available / (1024**3),
                "disk_free_gb": psutil.disk_usage('.').free / (1024**3)
            }
        except Exception:
            pass
        
        return diagnostic
