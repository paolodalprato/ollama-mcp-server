#!/usr/bin/env python3
import sys
import importlib

# Forza ricarica del modulo
if 'ollama_mcp.client' in sys.modules:
    del sys.modules['ollama_mcp.client']

sys.path.insert(0, './src')

import asyncio
import json
from ollama_mcp.client import OllamaClient

async def test_fresh():
    print("=== TEST FRESH CLIENT ===")
    
    client = OllamaClient()
    result = await client.list_models()
    
    print(f"Success: {result['success']}")
    if result['success'] and result['models']:
        first_model = result['models'][0]
        print(f"First model type: {type(first_model)}")
        print(f"First model: {first_model}")
        
        # Test JSON
        try:
            json_str = json.dumps(result)
            print("JSON serialization: SUCCESS!")
        except Exception as e:
            print(f"JSON serialization: FAILED - {e}")

if __name__ == "__main__":
    asyncio.run(test_fresh())