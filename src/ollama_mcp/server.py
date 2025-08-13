"""
Ollama MCP Server v0.9 - Complete & Advanced
Extended from v1.1 with comprehensive model management capabilities

Key Features:
- All base tools: list, chat, health_check, system_check
- Advanced tools: suggest_models, download_model, progress tracking, search, etc.
- Asynchronous downloads with progress monitoring  
- Intelligent model recommendations based on user needs
- Automatic Ollama server management
- Resilient design: starts even if Ollama offline
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from mcp.server import Server  
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from ollama_mcp.client import OllamaClient
from ollama_mcp.config import get_config
from ollama_mcp.tools.base_tools import get_base_tools, handle_base_tool
from ollama_mcp.tools.advanced_tools import get_advanced_tools, handle_advanced_tool

logger = logging.getLogger(__name__)


class OllamaMCPServer:
    """
    Simplified MCP server for Ollama management
    
    Design principles:
    - Start successfully even if Ollama is offline
    - Provide helpful diagnostics when things don't work
    - Minimal code, maximum functionality
    - Based on proven v1.0 patterns
    """
    
    def __init__(self):
        """Initialize server with resilient setup"""
        try:
            # Load config
            self.config = get_config()
            
            # Initialize resilient client (won't fail if Ollama offline)
            self.client = OllamaClient(
                host=self.config.url,
                timeout=self.config.timeout
            )
            
            # Create MCP server
            self.server = Server("ollama-mcp-server")
            
            # Register handlers using v1.0 working pattern
            self._register_handlers()
            
            logger.info(f"Ollama MCP Server v0.9 initialized for {self.config.url}")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            # Create minimal fallback
            self.server = Server("ollama-mcp-server-fallback")
            self.client = None
            
    def _register_handlers(self):
        """Register MCP handlers using CORRECT v1.0 pattern"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """Return available tools (always works)"""
            try:
                # Return both base and advanced tools
                base_tools = get_base_tools()
                advanced_tools = get_advanced_tools()
                return base_tools + advanced_tools
            except Exception as e:
                logger.error(f"Error listing tools: {e}")
                return []
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Route tool calls to appropriate handlers (v1.0 pattern)"""
            try:
                # Get the names of the advanced tools to route the call correctly.
                advanced_tool_names = {t.name for t in get_advanced_tools()}
                
                # Route to appropriate handler
                if name in advanced_tool_names:
                    return await handle_advanced_tool(name, arguments, self.client)
                else:
                    return await handle_base_tool(name, arguments, self.client)
                
            except Exception as e:
                logger.error(f"Error in tool {name}: {e}")
                return [TextContent(
                    type="text",
                    text=f"Error executing {name}: {str(e)}"
                )]
    
    async def run(self):
        """Run the MCP server with enhanced startup"""
        try:
            logger.info("Starting Ollama MCP Server v0.9...")
            
            # Test Ollama connection but don't fail startup
            if self.client:
                health = await self.client.health_check()
                if health["healthy"]:
                    logger.info("Ollama connected: %d models available", health['models_count'])
                else:
                    logger.warning("Ollama not accessible: %s", health['error'])
                    logger.info("Server will start anyway - tools provide diagnostics")
            
            num_base_tools = len(get_base_tools())
            num_advanced_tools = len(get_advanced_tools())
            logger.info(f"MCP server ready - {num_base_tools + num_advanced_tools} tools available ({num_base_tools} base + {num_advanced_tools} advanced)")
            
            # Start MCP stdio server
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
    """Main entry point for MCP server"""
    try:
        # Setup basic logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create and run server
        server = OllamaMCPServer()
        asyncio.run(server.run())
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
