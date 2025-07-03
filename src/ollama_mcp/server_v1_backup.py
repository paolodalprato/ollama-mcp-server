"""
Ollama MCP Server - Main Server Implementation
Exposes Ollama management capabilities via MCP protocol

Version: 1.0.0
"""

import asyncio
import logging
import os
from typing import Dict, Any, List

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource, Tool, TextContent, ImageContent, EmbeddedResource
)

from .client import OllamaClient
from .server_manager import OllamaServerManager
from .model_manager import ModelManager
from .job_manager import get_job_manager
from .hardware_checker import HardwareChecker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OllamaMCPServer:
    """Main Ollama MCP Server implementation"""
    
    def __init__(self):
        """Initialize the MCP server"""
        self.server = Server("ollama-mcp-server")
        
        # Initialize Ollama components
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.client = OllamaClient(host=self.ollama_host)
        self.server_manager = OllamaServerManager(host=self.ollama_host)
        self.model_manager = ModelManager(self.client)
        self.job_manager = get_job_manager()
        self.hardware_checker = HardwareChecker()
        
        logger.info(f"Initialized Ollama MCP Server for {self.ollama_host}")
        
        # Register tool handlers
        self._register_tools()
    
    def _register_tools(self):
        """Register all MCP tools"""
        
        # Model Management Tools
        @self.server.call_tool()
        async def list_local_models() -> List[TextContent]:
            """List all locally installed Ollama models with detailed information"""
            try:
                result = await self.model_manager.list_models()
                return [TextContent(type="text", text=str(result))]
            except Exception as e:
                logger.error(f"Error in list_local_models: {e}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]
        
        @self.server.call_tool()
        async def download_model_async(model_name: str) -> List[TextContent]:
            """
            Start asynchronous model download
            
            Args:
                model_name: Name of model to download (e.g., 'llama3.1:8b')
            """
            try:
                result = await self.model_manager.download_model_async(model_name)
                return [TextContent(type="text", text=str(result))]
            except Exception as e:
                logger.error(f"Error in download_model_async: {e}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]
        
        @self.server.call_tool()
        async def remove_model(model_name: str, force: bool = False) -> List[TextContent]:
            """
            Remove a model from local storage
            
            Args:
                model_name: Name of model to remove
                force: Force removal even if model is default
            """
            try:
                result = await self.model_manager.remove_model(model_name, force)
                return [TextContent(type="text", text=str(result))]
            except Exception as e:
                logger.error(f"Error in remove_model: {e}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]
        
        @self.server.call_tool()
        async def local_llm_chat(message: str, model: str = None, max_tokens: int = None) -> List[TextContent]:
            """
            Chat directly with a local Ollama model
            
            Args:
                message: Message to send to the model
                model: Optional model name (uses default if not specified)
                max_tokens: Optional maximum tokens to generate
            """
            try:
                # Use default model if none specified
                if not model:
                    model = self.model_manager.default_model
                    if not model:
                        # Try to get first available model
                        models = await self.model_manager.list_models()
                        if models["success"] and models["models"]:
                            model = models["models"][0]["name"]
                        else:
                            return [TextContent(type="text", text="Error: No models available")]
                
                result = await self.client.chat(
                    model=model,
                    prompt=message,
                    max_tokens=max_tokens
                )
                
                return [TextContent(type="text", text=str(result))]
            except Exception as e:
                logger.error(f"Error in local_llm_chat: {e}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]
        
        @self.server.call_tool()
        async def server_status() -> List[TextContent]:
            """Get detailed Ollama server status"""
            try:
                result = await self.server_manager.get_server_status()
                status_dict = {
                    "pid": result.pid,
                    "status": result.status,
                    "port": result.port,
                    "uptime_seconds": result.uptime_seconds,
                    "uptime_human": result.uptime_human,
                    "memory_mb": result.memory_mb,
                    "cpu_percent": result.cpu_percent
                }
                return [TextContent(type="text", text=str({"success": True, "server_status": status_dict}))]
            except Exception as e:
                logger.error(f"Error in server_status: {e}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]
        
        @self.server.call_tool()
        async def system_resource_check(model_name: str = None) -> List[TextContent]:
            """
            Check system resources and model compatibility
            
            Args:
                model_name: Optional model name to check compatibility for
            """
            try:
                # Get system info
                system_result = await self.hardware_checker.get_system_info()
                
                result = system_result.copy()
                
                # If model specified, check compatibility
                if model_name:
                    compatibility = await self.hardware_checker.check_model_compatibility(model_name)
                    result["model_compatibility"] = compatibility
                
                return [TextContent(type="text", text=str(result))]
            except Exception as e:
                logger.error(f"Error in system_resource_check: {e}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]

    async def run(self):
        """Run the MCP server"""
        try:
            logger.info("Starting Ollama MCP Server...")
            
            # Test Ollama connection
            health = await self.client.health_check()
            if health["healthy"]:
                logger.info(f"Connected to Ollama: {health['message']}")
            else:
                logger.warning(f"Ollama not accessible: {health.get('error', 'Unknown error')}")
            
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
    """Main entry point"""
    try:
        server = OllamaMCPServer()
        asyncio.run(server.run())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
