#!/usr/bin/env python3
"""
Test MCP client con forzatura reload moduli
"""

import asyncio
import json
import sys
import subprocess
import os

async def test_mcp_fixed():
    """Test via nuovo processo MCP per evitare cache"""
    
    print("TEST: list_local_models via nuovo processo MCP")
    print("=" * 60)
    
    try:
        # Avvia il server MCP in un nuovo processo
        print("1. Avvio MCP server in nuovo processo...")
        
        # Comando per avviare il server
        cmd = [sys.executable, "src/ollama_mcp/server.py"]
        
        # Avvia il processo server
        server_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.getcwd()
        )
        
        print("2. Invio richiesta list_tools...")
        
        # Invia una richiesta MCP di base per list_tools
        list_tools_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list"
        }
        
        # Invia init prima
        init_request = {
            "jsonrpc": "2.0", 
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"}
            }
        }
        
        server_process.stdin.write(json.dumps(init_request) + "\n")
        server_process.stdin.flush()
        
        # Leggi risposta init
        init_response = server_process.stdout.readline()
        print(f"Init response: {init_response.strip()}")
        
        # Lista tools
        server_process.stdin.write(json.dumps(list_tools_request) + "\n")
        server_process.stdin.flush()
        
        # Leggi risposta tools
        tools_response = server_process.stdout.readline()
        print(f"Tools response: {tools_response.strip()}")
        
        print("3. Invio richiesta list_local_models...")
        
        # Richiesta call tool
        call_tool_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "list_local_models",
                "arguments": {}
            }
        }
        
        server_process.stdin.write(json.dumps(call_tool_request) + "\n")
        server_process.stdin.flush()
        
        # Leggi risposta
        tool_response = server_process.stdout.readline()
        print(f"Tool response: {tool_response.strip()}")
        
        # Parsing della risposta
        try:
            response_data = json.loads(tool_response)
            if "result" in response_data:
                print("SUCCESSO! list_local_models ha restituito risultato")
                if "content" in response_data["result"] and response_data["result"]["content"]:
                    content = response_data["result"]["content"][0]
                    if "text" in content:
                        text_content = content["text"]
                        print(f"Contenuto (primi 200 char): {text_content[:200]}...")
                        
                        # Verifica che non contenga errori
                        if "Error executing" in text_content:
                            print("ERRORE: Trovato messaggio di errore nel contenuto!")
                        elif "[MODELLI]" in text_content:
                            print("SUCCESSO: Contenuto valido trovato!")
            else:
                print(f"Errore nella risposta: {response_data}")
                
        except Exception as parse_error:
            print(f"Errore parsing risposta: {parse_error}")
        
        # Chiudi il server
        server_process.terminate()
        
        # Leggi stderr per errori
        stderr_output = server_process.stderr.read()
        if stderr_output and "Object of type datetime is not JSON serializable" in stderr_output:
            print("ERRORE: Trovato errore datetime in stderr!")
            print(f"Stderr: {stderr_output}")
        elif stderr_output:
            print(f"Server stderr: {stderr_output}")
        
    except Exception as e:
        print(f"ERRORE test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_mcp_fixed())