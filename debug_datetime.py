#!/usr/bin/env python3
"""
Debug script per testare il problema datetime con ollama
"""

import json
import asyncio
from datetime import datetime

try:
    import ollama
    print(f"Ollama import: OK")
    
    # Test sincrono
    print("\n=== TEST SINCRONO ===")
    client = ollama.Client()
    response = client.list()
    
    if response.models:
        model = response.models[0]
        print(f"Model name: {model.model}")
        print(f"Modified type: {type(model.modified_at)}")
        print(f"Modified value: {model.modified_at}")
        
        # Test model_dump
        model_dict = model.model_dump()
        print(f"Model dict keys: {list(model_dict.keys())}")
        
        for key, value in model_dict.items():
            print(f"  {key}: {type(value)} = {value}")
        
        # Test JSON serialization
        try:
            json_str = json.dumps(model_dict)
            print("JSON serialization: OK")
        except Exception as e:
            print(f"JSON serialization FAILED: {e}")
            
            # Try converting datetime fields
            clean_dict = {}
            for key, value in model_dict.items():
                if isinstance(value, datetime):
                    clean_dict[key] = str(value)
                    print(f"Converted {key} from datetime to string")
                else:
                    clean_dict[key] = value
            
            try:
                json_str = json.dumps(clean_dict)
                print("JSON serialization after datetime conversion: OK")
            except Exception as e2:
                print(f"JSON serialization still FAILED: {e2}")
    
    # Test asincrono
    print("\n=== TEST ASINCRONO ===")
    async def test_async():
        async_client = ollama.AsyncClient()
        try:
            response = await async_client.list()
            if response.models:
                model = response.models[0]
                print(f"Async model name: {model.model}")
                print(f"Async modified type: {type(model.modified_at)}")
                
                model_dict = model.model_dump()
                try:
                    json_str = json.dumps(model_dict)
                    print("Async JSON serialization: OK")
                except Exception as e:
                    print(f"Async JSON serialization FAILED: {e}")
        except Exception as e:
            print(f"Async test failed: {e}")
    
    asyncio.run(test_async())
    
except ImportError as e:
    print(f"Ollama import failed: {e}")
except Exception as e:
    print(f"Test failed: {e}")
