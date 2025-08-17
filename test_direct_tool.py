#!/usr/bin/env python3
"""Test diretto del tool nel server per verifica finale"""

import asyncio
import sys
sys.path.insert(0, './src')

# Force reload
import importlib
modules_to_reload = [
    'ollama_mcp.client',
    'ollama_mcp.tools.base_tools', 
    'ollama_mcp.server'
]

for module_name in modules_to_reload:
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])

from ollama_mcp.server import OllamaMCPServer

async def test_server_direct():
    print("=== TEST DIRETTO SERVER ===")
    
    try:
        # Crea server
        server = OllamaMCPServer()
        
        # Test diretto del call_tool handler
        print("1. Test list_local_models via server...")
        
        # Non possiamo chiamare direttamente il decorator, ma possiamo testare il client
        if server.client:
            print("2. Test client list_models...")
            result = await server.client.list_models()
            
            print(f"Success: {result['success']}")
            if result['success'] and result['models']:
                print(f"Models count: {result['count']}")
                first_model = result['models'][0]
                print(f"First model type: {type(first_model)}")
                print(f"First model keys: {list(first_model.keys()) if isinstance(first_model, dict) else 'Not a dict'}")
                
                # Test JSON serialization
                import json
                try:
                    json_str = json.dumps(result)
                    print("JSON serialization: SUCCESS")
                except Exception as e:
                    print(f"JSON serialization: FAILED - {e}")
            
            print("3. Test health_check...")
            health = await server.client.health_check()
            print(f"Health check: {health}")
            
            # Test JSON del health check
            try:
                json_str = json.dumps(health)
                print("Health check JSON: SUCCESS")
            except Exception as e:
                print(f"Health check JSON: FAILED - {e}")
                
        else:
            print("Server client not available")
            
    except Exception as e:
        print(f"ERRORE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_server_direct())