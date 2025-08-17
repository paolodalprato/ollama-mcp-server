#!/usr/bin/env python3
"""
Minimal MCP server for debugging datetime issue
"""

import asyncio
import logging
from typing import Dict, List, Any
from mcp.server import Server  
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

logger = logging.getLogger(__name__)

def create_minimal_server():
    """Create minimal MCP server for testing"""
    server = Server("minimal-test-server")
    
    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """Return simple test tool"""
        return [
            Tool(
                name="test_tool",
                description="Simple test tool",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            )
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Return simple text response"""
        if name == "test_tool":
            return [TextContent(type="text", text="TEST: Simple response from minimal server")]
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    return server

async def main():
    """Run minimal server"""
    try:
        logging.basicConfig(level=logging.INFO)
        logger.info("Starting minimal MCP test server...")
        
        server = create_minimal_server()
        
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
            
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
