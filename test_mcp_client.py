#!/usr/bin/env python3
"""
Test client MCP per verificare che list_local_models funzioni via MCP protocol
"""

import asyncio
import json
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_mcp_list_models():
    """Test list_local_models via MCP protocol"""
    
    print("TEST: list_local_models via MCP protocol")
    print("=" * 50)
    
    try:
        # Configurazione per avviare il server MCP
        server_params = StdioServerParameters(
            command="python",
            args=["src/ollama_mcp/server.py"],
            env=None
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                
                print("1. Inizializzazione MCP client...")
                await session.initialize()
                
                print("2. Lista tools disponibili...")
                tools_result = await session.list_tools()
                print(f"Tools disponibili: {len(tools_result.tools)}")
                
                # Cerca il tool list_local_models
                list_models_tool = None
                for tool in tools_result.tools:
                    if tool.name == "list_local_models":
                        list_models_tool = tool
                        break
                
                if not list_models_tool:
                    print("ERRORE: Tool list_local_models non trovato!")
                    return
                
                print("3. Chiamata list_local_models...")
                result = await session.call_tool(list_models_tool.name, {})
                
                print("SUCCESSO! Risultato ricevuto:")
                print(f"Tipo: {type(result)}")
                print(f"IsError: {result.isError}")
                
                if hasattr(result, 'content'):
                    print(f"Content tipo: {type(result.content)}")
                    print(f"Content length: {len(result.content) if result.content else 0}")
                    
                    if result.content and len(result.content) > 0:
                        first_content = result.content[0]
                        if hasattr(first_content, 'text'):
                            content_text = first_content.text[:300]
                            print(f"Primi 300 caratteri: {content_text}")
                
                print("Test completato con successo!")
                
    except Exception as e:
        print(f"ERRORE durante test MCP: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_mcp_list_models())