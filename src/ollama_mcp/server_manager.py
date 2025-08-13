"""
Server Manager - Claude Ollama Bridge v0.9
Cross-platform Ollama server lifecycle management

Design Principles:
- Type safety with full annotations
- Cross-platform compatibility
- Comprehensive error handling
- Clean separation of concerns
"""

import asyncio
import logging
import os
import platform
import psutil
import subprocess
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

from .config import OllamaConfig as ServerConfig

logger = logging.getLogger(__name__)

@dataclass
class OllamaProcessInfo:
    """
    Information about Ollama server process
    
    Contains all relevant details about the running Ollama server
    including performance metrics and status information.
    """
    pid: Optional[int]
    status: str  # "running", "stopped", "unknown"
    port: int
    uptime_seconds: Optional[int]
    memory_mb: Optional[float]
    cpu_percent: Optional[float]
    command_line: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate process info after creation"""
        self._validate_status()
    
    def _validate_status(self):
        """Validate status values"""
        valid_statuses = {"running", "stopped", "unknown"}
        if self.status not in valid_statuses:
            self.status = "unknown"
    
    @property
    def uptime_human(self) -> str:
        """Human readable uptime"""
        if self.uptime_seconds is None:
            return "N/A"
        
        seconds = self.uptime_seconds
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




class CrossPlatformProcessManager:
    """
    Cross-platform process management utilities
    
    Handles OS-specific process creation and management for Ollama server.
    """
    
    @staticmethod
    def get_platform_info() -> Dict[str, str]:
        """Get platform information"""
        return {
            "system": platform.system(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "processor": platform.processor()
        }
    
    @staticmethod
    def create_subprocess_args(command: List[str]) -> Dict[str, Any]:
        """
        Create platform-specific subprocess arguments
        
        Args:
            command: Command to execute
            
        Returns:
            Dict with subprocess.Popen arguments
        """
        base_args = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "start_new_session": True
        }
        
        system = platform.system()
        
        if system == "Windows":
            # Windows-specific: hide console window
            base_args["creationflags"] = subprocess.CREATE_NO_WINDOW
        elif system in ["Linux", "Darwin"]:
            # Unix-like: detach from terminal
            base_args["preexec_fn"] = os.setsid
        
        return base_args
    
    @staticmethod
    def find_ollama_executable() -> Optional[Path]:
        """
        Find Ollama executable across platforms
        
        Returns:
            Path to Ollama executable or None if not found
        """
        system = platform.system()
        
        # Common executable names
        executable_names = ["ollama"]
        if system == "Windows":
            executable_names.append("ollama.exe")
        
        # Check PATH first
        for name in executable_names:
            if path := Path.resolve_path(name):
                return path
        
        # Platform-specific default locations
        default_paths = []
        
        if system == "Windows":
            default_paths = [
                Path.home() / "AppData" / "Local" / "Programs" / "Ollama",
                Path("C:") / "Program Files" / "Ollama",
                Path("C:") / "Program Files (x86)" / "Ollama"
            ]
        elif system == "Darwin":  # macOS
            default_paths = [
                Path("/usr/local/bin"),
                Path("/opt/homebrew/bin"),
                Path("/Applications/Ollama.app/Contents/MacOS")
            ]
        elif system == "Linux":
            default_paths = [
                Path("/usr/local/bin"),
                Path("/usr/bin"),
                Path.home() / ".local" / "bin",
                Path("/opt/ollama/bin")
            ]
        
        # Check default paths
        for base_path in default_paths:
            for name in executable_names:
                full_path = base_path / name
                if full_path.exists() and full_path.is_file():
                    return full_path
        
        return None
    
    @staticmethod
    def resolve_path(command: str) -> Optional[Path]:
        """
        Resolve command path using system PATH
        
        Args:
            command: Command name to resolve
            
        Returns:
            Full path to command or None if not found
        """
        try:
            import shutil
            path = shutil.which(command)
            return Path(path) if path else None
        except Exception:
            return None


class OllamaServerManager:
    """
    Cross-platform Ollama server lifecycle manager
    
    Provides complete control over Ollama server including
    start, stop, restart, and detailed status monitoring
    across Windows, Linux, and macOS platforms.
    """
    
    def __init__(self, config: Optional[ServerConfig] = None):
        """
        Initialize server manager
        
        Args:
            config: Server configuration, uses defaults if None
        """
        self.config = config or ServerConfig()
        self.process_manager = CrossPlatformProcessManager()
        self._cached_executable_path: Optional[Path] = None
        
        logger.debug(f"Initialized OllamaServerManager for {self.config.full_url}")
        logger.debug(f"Platform: {self.process_manager.get_platform_info()}")
    
    def _get_ollama_executable(self) -> Optional[Path]:
        """
        Get Ollama executable path with caching
        
        Returns:
            Path to Ollama executable or None if not found
        """
        if self._cached_executable_path is None:
            if self.config.custom_executable_path:
                self._cached_executable_path = Path(self.config.custom_executable_path)
            else:
                self._cached_executable_path = self.process_manager.find_ollama_executable()
        
        return self._cached_executable_path
    
    async def get_server_status(self) -> OllamaProcessInfo:
        """
        Get detailed Ollama server status
        
        Returns:
            OllamaProcessInfo with current server state
        """
        try:
            ollama_pid = None
            ollama_process = None
            
            # Search for Ollama server process
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
                try:
                    proc_info = proc.info
                    if not proc_info['name']:
                        continue
                    
                    # Check if this is an Ollama process
                    if 'ollama' in proc_info['name'].lower():
                        cmdline = proc_info.get('cmdline', [])
                        
                        # Look for server process (not client commands)
                        is_server_process = (
                            any('serve' in str(cmd).lower() for cmd in cmdline) or
                            any(str(self.config.port) in str(cmd) for cmd in cmdline) or
                            len(cmdline) == 1  # Just "ollama" typically means server
                        )
                        
                        if is_server_process:
                            ollama_pid = proc_info['pid']
                            ollama_process = proc
                            break
                
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if ollama_process:
                try:
                    # Get detailed process information
                    memory_info = ollama_process.memory_info()
                    memory_mb = memory_info.rss / 1024 / 1024
                    cpu_percent = ollama_process.cpu_percent()
                    create_time = ollama_process.create_time()
                    uptime = int(time.time() - create_time)
                    cmdline = ollama_process.cmdline()
                    
                    return OllamaProcessInfo(
                        pid=ollama_pid,
                        status="running",
                        port=self.config.port,
                        uptime_seconds=uptime,
                        memory_mb=memory_mb,
                        cpu_percent=cpu_percent,
                        command_line=cmdline
                    )
                
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    logger.warning(f"Could not get detailed process info: {e}")
                    return OllamaProcessInfo(
                        pid=ollama_pid,
                        status="running",
                        port=self.config.port,
                        uptime_seconds=None,
                        memory_mb=None,
                        cpu_percent=None
                    )
            else:
                return OllamaProcessInfo(
                    pid=None,
                    status="stopped",
                    port=self.config.port,
                    uptime_seconds=None,
                    memory_mb=None,
                    cpu_percent=None
                )
        
        except Exception as e:
            logger.error(f"Error getting Ollama server status: {e}")
            return OllamaProcessInfo(
                pid=None,
                status="unknown",
                port=self.config.port,
                uptime_seconds=None,
                memory_mb=None,
                cpu_percent=None
            )
    
    async def start_server(self) -> Dict[str, Any]:
        """
        Start Ollama server if not running
        
        Returns:
            Dict with start operation results and troubleshooting info
        """
        try:
            # Check if already running
            status = await self.get_server_status()
            if status.status == "running":
                return {
                    "success": True,
                    "message": "Ollama server is already running",
                    "pid": status.pid,
                    "already_running": True,
                    "server_url": self.config.full_url
                }
            
            # Find Ollama executable
            executable_path = self._get_ollama_executable()
            if not executable_path:
                return {
                    "success": False,
                    "message": "Ollama executable not found",
                    "error": "ollama_not_found",
                    "troubleshooting": {
                        "install_ollama": "Download from https://ollama.ai",
                        "check_path": "Ensure Ollama is in system PATH",
                        "manual_path": "Set custom_path in configuration"
                    }
                }
            
            # Prepare command and environment
            command = [str(executable_path), "serve"]
            subprocess_args = self.process_manager.create_subprocess_args(command)
            
            # Add custom environment variables
            env = os.environ.copy()
            env.update(self.config.environment_vars)
            if env != os.environ:
                subprocess_args["env"] = env
            
            # Start server process
            logger.info(f"Starting Ollama server: {' '.join(command)}")
            process = subprocess.Popen(command, **subprocess_args)
            
            # Wait for server to start
            for i in range(self.config.startup_timeout):
                await asyncio.sleep(1)
                status = await self.get_server_status()
                if status.status == "running":
                    return {
                        "success": True,
                        "message": "Ollama server started successfully",
                        "pid": status.pid,
                        "startup_time_seconds": i + 1,
                        "server_url": self.config.full_url,
                        "executable_path": str(executable_path)
                    }
            
            # Timeout case
            return {
                "success": False,
                "message": f"Server start timeout ({self.config.startup_timeout} seconds exceeded)",
                "error": "startup_timeout",
                "troubleshooting": {
                    "check_port": f"Ensure port {self.config.port} is available",
                    "check_logs": "Check system logs for Ollama errors",
                    "manual_start": f"Try running '{executable_path} serve' manually"
                }
            }
        
        except PermissionError:
            return {
                "success": False,
                "message": "Permission denied starting Ollama server",
                "error": "permission_denied",
                "troubleshooting": {
                    "run_as_admin": "Try running with administrator privileges",
                    "check_permissions": "Ensure Ollama executable has proper permissions"
                }
            }
        except Exception as e:
            logger.error(f"Error starting Ollama server: {e}")
            return {
                "success": False,
                "message": f"Failed to start server: {str(e)}",
                "error": str(e),
                "troubleshooting": {
                    "check_installation": "Verify Ollama installation",
                    "check_system_resources": "Ensure sufficient memory and disk space"
                }
            }
    
    async def stop_server(self) -> Dict[str, Any]:
        """
        Stop Ollama server if running
        
        Returns:
            Dict with stop operation results
        """
        try:
            status = await self.get_server_status()
            
            if status.status != "running":
                return {
                    "success": True,
                    "message": "Ollama server is not running",
                    "already_stopped": True
                }
            
            if status.pid is None:
                return {
                    "success": False,
                    "message": "Cannot find Ollama server process ID",
                    "error": "pid_not_found"
                }
            
            # Terminate the process gracefully
            try:
                process = psutil.Process(status.pid)
                process.terminate()
                
                # Wait for graceful termination
                for i in range(self.config.shutdown_timeout):
                    await asyncio.sleep(1)
                    if not process.is_running():
                        return {
                            "success": True,
                            "message": "Ollama server stopped successfully",
                            "shutdown_time_seconds": i + 1
                        }
                
                # Force kill if graceful termination failed
                logger.warning("Graceful shutdown failed, forcing termination")
                process.kill()
                await asyncio.sleep(1)
                
                if not process.is_running():
                    return {
                        "success": True,
                        "message": "Ollama server stopped (forced termination)",
                        "forced_shutdown": True
                    }
                else:
                    return {
                        "success": False,
                        "message": "Failed to stop Ollama server",
                        "error": "shutdown_failed",
                        "troubleshooting": {
                            "manual_kill": f"Try manually killing process {status.pid}",
                            "system_restart": "Consider system restart if process is stuck"
                        }
                    }
            
            except psutil.NoSuchProcess:
                return {
                    "success": True,
                    "message": "Ollama server was already stopped",
                    "already_stopped": True
                }
        
        except Exception as e:
            logger.error(f"Error stopping Ollama server: {e}")
            return {
                "success": False,
                "message": f"Failed to stop server: {str(e)}",
                "error": str(e)
            }
    
    async def restart_server(self) -> Dict[str, Any]:
        """
        Restart Ollama server (stop + start)
        
        Returns:
            Dict with restart operation results
        """
        try:
            # Stop the server first
            stop_result = await self.stop_server()
            
            if not stop_result["success"] and not stop_result.get("already_stopped", False):
                return {
                    "success": False,
                    "message": "Failed to stop server for restart",
                    "error": stop_result.get("error", "stop_failed"),
                    "stop_details": stop_result
                }
            
            # Wait before restart
            await asyncio.sleep(2)
            
            # Start the server
            start_result = await self.start_server()
            
            if start_result["success"]:
                return {
                    "success": True,
                    "message": "Ollama server restarted successfully",
                    "restart_completed": True,
                    "new_pid": start_result.get("pid"),
                    "server_url": self.config.full_url
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to start server after stop",
                    "error": start_result.get("error", "restart_failed"),
                    "start_details": start_result
                }
        
        except Exception as e:
            logger.error(f"Error restarting Ollama server: {e}")
            return {
                "success": False,
                "message": f"Restart failed: {str(e)}",
                "error": str(e)
            }
    
    async def ensure_server_running(self) -> Dict[str, Any]:
        """
        Ensure Ollama server is running, start if necessary
        
        Returns:
            Dict with operation results
        """
        try:
            status = await self.get_server_status()
            
            if status.status == "running":
                return {
                    "success": True,
                    "message": "Ollama server is running",
                    "pid": status.pid,
                    "uptime": status.uptime_human,
                    "server_url": self.config.full_url,
                    "already_running": True
                }
            else:
                # Server not running, try to start it
                start_result = await self.start_server()
                return start_result
        
        except Exception as e:
            logger.error(f"Error ensuring server running: {e}")
            return {
                "success": False,
                "message": f"Failed to ensure server running: {str(e)}",
                "error": str(e)
            }
    
    def get_configuration_info(self) -> Dict[str, Any]:
        """
        Get current configuration information
        
        Returns:
            Dict with configuration details
        """
        executable_path = self._get_ollama_executable()
        platform_info = self.process_manager.get_platform_info()
        
        return {
            "server_config": {
                "host": self.config.host,
                "port": self.config.port,
                "full_url": self.config.full_url,
                "startup_timeout": self.config.startup_timeout,
                "shutdown_timeout": self.config.shutdown_timeout
            },
            "platform_info": platform_info,
            "executable_path": str(executable_path) if executable_path else None,
            "environment_vars": self.config.environment_vars
        }


# Export main classes
__all__ = [
    "OllamaServerManager",
    "OllamaProcessInfo",
    "OllamaServerConfig",
    "CrossPlatformProcessManager"
]
