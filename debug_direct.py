#!/usr/bin/env python3
"""
Test diretto per identificare il datetime object che causa problemi
"""

import asyncio
import json
from datetime import datetime

# Importiamo il nostro client
import sys
sys.path.append(r'D:\MCP_SERVER\INSTALLED\ollama-mcp-server\src')

from ollama_mcp.client import OllamaClient

async def test_datetime_issue():
    """Test per trovare il datetime object problematico"""
    print("=== TESTING CLIENT RESPONSE ===")
    
    client = OllamaClient()
    result = await client.list_models()
    
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Models count: {result['count']}")
        
        if result['models']:
            first_model = result['models'][0]
            print(f"First model type: {type(first_model)}")
            print(f"First model: {first_model}")
            
            # Test JSON serialization of the result
            try:
                json_str = json.dumps(result)
                print("JSON serialization: SUCCESS")
            except Exception as e:
                print(f"JSON serialization FAILED: {e}")
                
                # Find the problematic object
                print("\nAnalyzing each model:")
                for i, model in enumerate(result['models']):
                    print(f"\nModel {i}:")
                    for key, value in model.items():
                        print(f"  {key}: {type(value)} = {repr(value)}")
                        try:
                            json.dumps(value)
                        except Exception as field_error:
                            print(f"    ^^^ PROBLEM FIELD: {field_error}")
    else:
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    asyncio.run(test_datetime_issue())
