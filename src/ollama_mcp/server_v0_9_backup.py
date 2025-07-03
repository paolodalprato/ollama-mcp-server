"""
Ollama MCP Server - Main Server Implementation v0.9
Exposes Ollama management capabilities via MCP protocol with enhanced cross-platform support

Design Principles:
- Type safety with full annotations
- Cross-platform compatibility
- Comprehensive error handling
- Clean separation of concerns
"""

import asyncio
import logging
import logging.handlers
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Fix per import quando eseguito da Claude Desktop - Bridge compatibility
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource, Tool, TextContent, ImageContent, EmbeddedResource
)

# Import moduli locali con approccio Bridge
from client import OllamaClient
from server_manager import OllamaServerManager
from model_manager import ModelManager
from job_manager import get_job_manager
from hardware_checker import HardwareChecker
from config import get_config, get_config_manager, OllamaMCPConfig
from ollama_server_control import OllamaServerController

logger = logging.getLogger(__name__)

class OllamaMCPServer:
    """
    Enhanced Ollama MCP Server implementation v2.0
    
    Provides comprehensive Ollama management via MCP protocol with
    cross-platform support and advanced configuration management.
    """
    
    def __init__(self, config: Optional[OllamaMCPConfig] = None):
        """
        Initialize the MCP server
        
        Args:
            config: Optional configuration, uses global config if None
        """
        self.config = config or get_config()
        self.server = Server("ollama-mcp-server")
        
        # Configure logging from config
        self._setup_logging()
        
        # Initialize Ollama components with configuration - don't fail if Ollama is offline
        try:
            self.client = OllamaClient(host=self.config.server.full_url)
            logger.info(f"Ollama client initialized for {self.config.server.full_url}")
        except Exception as e:
            logger.warning(f"Could not initialize Ollama client: {e}")
            # Create a minimal client that will handle errors gracefully
            self.client = OllamaClient(host=self.config.server.full_url)
        
        try:
            self.server_manager = OllamaServerManager(self.config.server)
            logger.debug("Server manager initialized")
        except Exception as e:
            logger.warning(f"Could not initialize server manager: {e}")
            self.server_manager = None
        
        try:
            self.model_manager = ModelManager(self.client)
            logger.debug("Model manager initialized")
        except Exception as e:
            logger.warning(f"Could not initialize model manager: {e}")
            self.model_manager = None
        
        try:
            self.job_manager = get_job_manager()
            logger.debug("Job manager initialized")
        except Exception as e:
            logger.warning(f"Could not initialize job manager: {e}")
            self.job_manager = None
        
        try:
            self.hardware_checker = HardwareChecker(self.config.hardware)
            logger.debug("Hardware checker initialized")
        except Exception as e:
            logger.warning(f"Could not initialize hardware checker: {e}")
            self.hardware_checker = None
        
        # Initialize Ollama server controller for auto-start and diagnostics
        try:
            self.ollama_controller = OllamaServerController(host=self.config.server.full_url)
            logger.debug("Ollama controller initialized")
        except Exception as e:
            logger.warning(f"Could not initialize Ollama controller: {e}")
            self.ollama_controller = None
        
        logger.info(f"Initialized Ollama MCP Server v0.9 for {self.config.server.full_url}")
        logger.info(f"Configuration: GPU detection={self.config.hardware.enable_gpu_detection}, "
                   f"CPU fallback={self.config.hardware.enable_cpu_fallback}")
        
        # Register tool handlers
        self._register_tools()
    
    def _setup_logging(self):
        """Setup logging based on configuration"""
        log_level = getattr(logging, self.config.logging.level.upper(), logging.INFO)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console logging
        if self.config.logging.enable_console_logging:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        # File logging
        if self.config.logging.enable_file_logging:
            log_file = self.config.logging.log_file_path or "ollama-mcp-server.log"
            try:
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=self.config.logging.max_log_size_mb * 1024 * 1024,
                    backupCount=self.config.logging.backup_count
                )
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                file_handler.setFormatter(file_formatter)
                root_logger.addHandler(file_handler)
            except Exception as e:
                logger.warning(f"Could not setup file logging: {e}")
    
    def _register_tools(self):
        """Register all MCP tools with enhanced functionality"""
        
        # Model Management Tools
        @self.server.call_tool()
        async def list_local_models() -> List[TextContent]:
            """List all locally installed Ollama models with detailed information"""
            try:
                # Check if model manager is available
                if self.model_manager is None:
                    return [TextContent(type="text", text=json.dumps({
                        "success": False,
                        "error": "Model manager not available",
                        "status": "service_unavailable",
                        "troubleshooting": {
                            "check_server": "Use ollama_health_check tool for diagnosis",
                            "restart_mcp": "Restart MCP server if issues persist"
                        }
                    }, indent=2))]
                
                # First check if Ollama is running
                health = await self.client.health_check()
                if not health["healthy"]:
                    return [TextContent(type="text", text=json.dumps({
                        "success": False,
                        "error": "Ollama server not accessible",
                        "status": "server_offline",
                        "troubleshooting": {
                            "check_server": "Use ollama_health_check tool for diagnosis",
                            "start_server": "Use ollama_server_control tool with action='start'",
                            "manual_start": "Or manually run: ollama serve"
                        },
                        "quick_fix": {
                            "tool": "ollama_server_control",
                            "action": "start",
                            "description": "Start Ollama server automatically"
                        }
                    }, indent=2))]
                
                result = await self.model_manager.list_models()
                
                # Enhance response with usage guidance
                if result["success"] and result["models"]:
                    result["usage_guide"] = {
                        "chat_tool": "local_llm_chat",
                        "example": "Use local_llm_chat with message parameter to chat with models"
                    }
                else:
                    result["next_steps"] = {
                        "download_model": "Use download_model_async tool to get a model",
                        "popular_models": ["llama3.2", "qwen2.5", "phi3.5", "mistral"]
                    }
                
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                logger.error(f"Error in list_local_models: {e}")
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "troubleshooting": {
                        "check_server": "Use ollama_health_check tool to check Ollama server",
                        "start_server": "Use ollama_server_control tool if Ollama is not running"
                    }
                }, indent=2))]
        
        @self.server.call_tool()
        async def local_llm_chat(message: str, model: str = None) -> List[TextContent]:
            """
            Chat directly with a local Ollama model
            
            Args:
                message: Message to send to the model
                model: Optional model name (uses default if not specified)
            """
            try:
                # Check if model manager is available
                if self.model_manager is None:
                    return [TextContent(type="text", text=json.dumps({
                        "success": False,
                        "error": "Model manager not available",
                        "status": "service_unavailable",
                        "user_message": message,
                        "troubleshooting": {
                            "check_server": "Use ollama_health_check tool for diagnosis",
                            "restart_mcp": "Restart MCP server if issues persist"
                        }
                    }, indent=2))]
                
                # First check if Ollama is running
                health = await self.client.health_check()
                if not health["healthy"]:
                    return [TextContent(type="text", text=json.dumps({
                        "success": False,
                        "error": "Ollama server not accessible",
                        "status": "server_offline",
                        "user_message": message,
                        "troubleshooting": {
                            "step_1": "Check Ollama server status with ollama_health_check",
                            "step_2": "Start server with ollama_server_control action='start'",
                            "step_3": "Retry your message once Ollama is running"
                        },
                        "quick_fix": {
                            "auto_start": "Use ollama_server_control with action='start' to start Ollama",
                            "then_retry": "Then run this chat command again"
                        }
                    }, indent=2))]
                
                # Use default model if none specified
                if not model:
                    models = await self.model_manager.list_models()
                    if models["success"] and models["models"]:
                        model = models["models"][0]["name"]
                    else:
                        return [TextContent(type="text", text=json.dumps({
                            "success": False,
                            "error": "No models available",
                            "status": "no_models",
                            "user_message": message,
                            "troubleshooting": {
                                "download_model": "Use download_model_async tool to get a model",
                                "popular_choices": ["llama3.2", "qwen2.5", "phi3.5"],
                                "check_server": "Verify Ollama server is running"
                            },
                            "next_steps": {
                                "step_1": "Download a model: download_model_async with model_name parameter",
                                "step_2": "Retry chat once model is available"
                            }
                        }, indent=2))]
                
                result = await self.client.chat(model=model, prompt=message)
                
                # Enhance successful response
                if result.get("success"):
                    result["model_used"] = model
                    result["privacy_note"] = "All processing done locally - no data sent to cloud"
                
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                logger.error(f"Error in local_llm_chat: {e}")
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "user_message": message,
                    "model_requested": model,
                    "troubleshooting": {
                        "check_model": "Verify model is downloaded and available",
                        "check_server": "Ensure Ollama server is running",
                        "list_models": "Use list_local_models to see available models"
                    }
                }, indent=2))]
        
        # Server Management Tools  
        @self.server.call_tool()
        async def server_status() -> List[TextContent]:
            """Get detailed Ollama server status with troubleshooting info"""
            try:
                # Check if server manager is available
                if self.server_manager is None:
                    return [TextContent(type="text", text=json.dumps({
                        "success": False,
                        "error": "Server manager not available",
                        "status": "service_unavailable",
                        "troubleshooting": {
                            "use_alternative": "Use ollama_health_check tool for basic status",
                            "restart_mcp": "Restart MCP server if issues persist"
                        }
                    }, indent=2))]
                
                result = await self.server_manager.get_server_status()
                status_dict = {
                    "success": True,
                    "server_status": {
                        "pid": result.pid,
                        "status": result.status,
                        "port": result.port,
                        "uptime_seconds": result.uptime_seconds,
                        "uptime_human": result.uptime_human,
                        "memory_mb": result.memory_mb,
                        "cpu_percent": result.cpu_percent
                    },
                    "server_url": self.config.server.full_url,
                    "configuration": self.server_manager.get_configuration_info()
                }
                
                if result.status != "running":
                    status_dict["troubleshooting"] = {
                        "start_server": "Use start_server tool to start Ollama",
                        "check_installation": "Verify Ollama is installed and in PATH",
                        "check_port": f"Ensure port {self.config.server.port} is available"
                    }
                
                return [TextContent(type="text", text=json.dumps(status_dict, indent=2))]
            except Exception as e:
                logger.error(f"Error in server_status: {e}")
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "troubleshooting": {
                        "check_ollama": "Verify Ollama installation",
                        "check_permissions": "Ensure sufficient system permissions",
                        "use_alternative": "Try ollama_health_check tool for basic diagnosis"
                    }
                }, indent=2))]
        
        @self.server.call_tool()
        async def start_server() -> List[TextContent]:
            """Start Ollama server if not running"""
            try:
                # Check if server manager is available
                if self.server_manager is None:
                    return [TextContent(type="text", text=json.dumps({
                        "success": False,
                        "error": "Server manager not available",
                        "status": "service_unavailable",
                        "troubleshooting": {
                            "use_alternative": "Use ollama_server_control tool with action='start'",
                            "manual_start": "Or manually run: ollama serve"
                        }
                    }, indent=2))]
                
                result = await self.server_manager.start_server()
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                logger.error(f"Error in start_server: {e}")
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "troubleshooting": {
                        "check_installation": "Verify Ollama is installed",
                        "check_permissions": "Run with appropriate permissions",
                        "check_port": f"Ensure port {self.config.server.port} is available",
                        "use_alternative": "Try ollama_server_control tool as alternative"
                    }
                }, indent=2))]
        
        @self.server.call_tool()
        async def stop_server() -> List[TextContent]:
            """Stop Ollama server if running"""
            try:
                # Check if server manager is available
                if self.server_manager is None:
                    return [TextContent(type="text", text=json.dumps({
                        "success": False,
                        "error": "Server manager not available",
                        "status": "service_unavailable",
                        "troubleshooting": {
                            "use_alternative": "Use ollama_server_control tool with action='stop'",
                            "manual_stop": "Or manually stop Ollama process"
                        }
                    }, indent=2))]
                
                result = await self.server_manager.stop_server()
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                logger.error(f"Error in stop_server: {e}")
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": str(e)
                }, indent=2))]
        
        # Hardware and System Tools
        @self.server.call_tool()
        async def system_resource_check(model_name: str = None) -> List[TextContent]:
            """
            Check system resources and model compatibility with multi-GPU support
            
            Args:
                model_name: Optional model name to check compatibility for
            """
            try:
                # Check if hardware checker is available
                if self.hardware_checker is None:
                    return [TextContent(type="text", text=json.dumps({
                        "success": False,
                        "error": "Hardware checker not available",
                        "status": "service_unavailable",
                        "troubleshooting": {
                            "restart_mcp": "Restart MCP server to reinitialize hardware checker",
                            "check_permissions": "Ensure sufficient permissions for system monitoring"
                        }
                    }, indent=2))]
                
                # Get comprehensive system info
                system_result = await self.hardware_checker.get_system_info()
                
                result = system_result.copy()
                
                # If model specified, check compatibility
                if model_name:
                    compatibility = await self.hardware_checker.check_model_compatibility(model_name)
                    result["model_compatibility"] = compatibility
                
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                logger.error(f"Error in system_resource_check: {e}")
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "troubleshooting": {
                        "check_permissions": "Ensure sufficient permissions for system monitoring",
                        "check_gpu_drivers": "Verify GPU drivers are installed",
                        "check_tools": "Install nvidia-smi, rocm-smi, or intel-gpu-top as needed"
                    }
                }, indent=2))]
        
        # Diagnostic and Control Tools
        @self.server.call_tool()
        async def ollama_health_check() -> List[TextContent]:
            """Complete health check of Ollama installation and server status"""
            try:
                # Check if ollama controller is available
                if self.ollama_controller is None:
                    # Fallback to basic health check using client
                    health = await self.client.health_check()
                    return [TextContent(type="text", text=json.dumps({
                        "success": True,
                        "diagnostic": {
                            "ollama_installed": "unknown",
                            "server_running": health["healthy"],
                            "basic_check": True,
                            "controller_available": False
                        },
                        "status_summary": "BASIC_CHECK" if health["healthy"] else "NEEDS_ATTENTION",
                        "quick_actions": [{
                            "action": "Limited Diagnosis",
                            "description": "Ollama controller not available - basic check only"
                        }],
                        "health_check": health
                    }, indent=2))]
                
                diagnostic = await self.ollama_controller.get_diagnostic_info()
                
                # Enhanced diagnostic response
                result = {
                    "success": True,
                    "diagnostic": diagnostic,
                    "status_summary": "HEALTHY" if diagnostic["server_running"] else "NEEDS_ATTENTION",
                    "quick_actions": []
                }
                
                # Add quick action recommendations
                if not diagnostic["ollama_installed"]:
                    result["quick_actions"].append({
                        "action": "Install Ollama",
                        "description": "Download and install from https://ollama.com"
                    })
                elif not diagnostic["server_running"]:
                    result["quick_actions"].append({
                        "action": "Start Server",
                        "description": "Use ollama_server_control tool with action='start'"
                    })
                else:
                    result["quick_actions"].append({
                        "action": "All Good",
                        "description": "System ready for AI model operations"
                    })
                
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                logger.error(f"Error in ollama_health_check: {e}")
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "status_summary": "ERROR",
                    "quick_actions": [{
                        "action": "Check Installation", 
                        "description": "Verify Ollama is properly installed"
                    }]
                }, indent=2))]
        
        @self.server.call_tool()
        async def ollama_server_control(action: str = "status") -> List[TextContent]:
            """
            Control Ollama server (start, stop, restart, status)
            
            Args:
                action: Action to perform (status, start, stop, restart)
            """
            try:
                # Check if ollama controller is available
                if self.ollama_controller is None:
                    return [TextContent(type="text", text=json.dumps({
                        "success": False,
                        "error": "Ollama controller not available",
                        "action": action,
                        "status": "service_unavailable",
                        "troubleshooting": {
                            "basic_check": "Use ollama_health_check for basic status",
                            "manual_control": "Use manual commands: 'ollama serve' to start",
                            "restart_mcp": "Restart MCP server to reinitialize controller"
                        }
                    }, indent=2))]
                
                if action == "status":
                    status = await self.ollama_controller.get_server_status()
                    result = {
                        "success": True,
                        "action": "status",
                        "server_status": {
                            "status": status.status,
                            "pid": status.pid,
                            "port": status.port,
                            "uptime": self.ollama_controller.format_uptime(status.uptime_seconds),
                            "memory_mb": status.memory_mb,
                            "cpu_percent": status.cpu_percent
                        }
                    }
                    
                    if status.status == "running":
                        result["message"] = "Ollama server is running and healthy"
                    else:
                        result["message"] = "Ollama server is not running"
                        result["next_steps"] = {
                            "start_server": "Use action='start' to start the server",
                            "check_installation": "Verify Ollama installation if start fails"
                        }
                
                elif action == "start":
                    result = await self.ollama_controller.start_server()
                    result["action"] = "start"
                
                elif action == "stop":
                    result = await self.ollama_controller.stop_server()
                    result["action"] = "stop"
                    
                elif action == "restart":
                    result = await self.ollama_controller.restart_server()
                    result["action"] = "restart"
                    
                else:
                    result = {
                        "success": False,
                        "error": f"Unknown action: {action}",
                        "valid_actions": ["status", "start", "stop", "restart"]
                    }
                
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                logger.error(f"Error in ollama_server_control: {e}")
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "action": action,
                    "error": str(e),
                    "troubleshooting": {
                        "check_permissions": "Ensure sufficient permissions for process control",
                        "check_installation": "Verify Ollama is properly installed",
                        "manual_start": "Try manually: ollama serve"
                    }
                }, indent=2))]
        
        # Configuration Management Tools
        @self.server.call_tool()
        async def get_configuration() -> List[TextContent]:
            """Get current server configuration"""
            try:
                config_dict = self.config.to_dict()
                
                # Add runtime information
                result = {
                    "success": True,
                    "configuration": config_dict,
                    "config_sources": {
                        "config_directory": str(get_config_manager().config_dir),
                        "environment_variables": "OLLAMA_MCP_* variables applied"
                    }
                }
                
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                logger.error(f"Error in get_configuration: {e}")
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": str(e)
                }, indent=2))]
        
        @self.server.call_tool()
        async def create_default_config() -> List[TextContent]:
            """Create default configuration file for user customization"""
            try:
                config_manager = get_config_manager()
                config_path = config_manager.create_default_config_file()
                
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "message": "Default configuration file created",
                    "config_path": str(config_path),
                    "instructions": {
                        "edit_config": f"Edit {config_path} to customize settings",
                        "restart_server": "Restart MCP server to apply changes",
                        "environment_vars": "Use OLLAMA_MCP_* environment variables for overrides"
                    }
                }, indent=2))]
            except Exception as e:
                logger.error(f"Error in create_default_config: {e}")
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": str(e)
                }, indent=2))]

    async def run(self):
        """Run the MCP server with enhanced error handling"""
        try:
            logger.info("Starting Ollama MCP Server v0.9...")
            
            # Test Ollama connection but don't fail if it's not available
            try:
                health = await self.client.health_check()
                if health["healthy"]:
                    logger.info(f"Connected to Ollama: {health['message']}")
                else:
                    logger.warning(f"Ollama not accessible: {health.get('error', 'Unknown error')}")
                    logger.info("Server will start but some tools may not work until Ollama is available")
                    logger.info("Use 'ollama_health_check' tool for diagnosis and 'ollama_server_control' to start Ollama")
            except Exception as e:
                logger.warning(f"Could not check Ollama status: {e}")
                logger.info("Server will start but Ollama functionality may be limited")
            
            # Log configuration summary
            logger.info(f"Configuration loaded from: {get_config_manager().config_dir}")
            logger.info(f"GPU detection: {self.config.hardware.enable_gpu_detection}")
            logger.info(f"Log level: {self.config.logging.level}")
            logger.info("MCP server ready - all tools available including diagnostics")
            
            # Start stdio server
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options()
                )
                
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise


def main():
    """Main entry point with enhanced error handling"""
    try:
        # Load configuration first
        config = get_config()
        
        # Create and run server
        server = OllamaMCPServer(config)
        asyncio.run(server.run())
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error("Check configuration and Ollama installation")
        raise


if __name__ == "__main__":
    main()
