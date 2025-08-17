#!/usr/bin/env python3
import asyncio
import json
import sys
sys.path.append('./src')

from ollama_mcp.client import OllamaClient

async def test_client_json():
    print("=== TEST CLIENT JSON SERIALIZATION ===")
    
    try:
        client = OllamaClient()
        result = await client.list_models()
        
        print(f"Result success: {result['success']}")
        print(f"Result keys: {list(result.keys())}")
        
        # Test JSON serialization of the complete result
        try:
            json_str = json.dumps(result, indent=2)
            print("JSON serialization: SUCCESS")
            print(f"JSON length: {len(json_str)}")
            
            # Check first model
            if result['success'] and result['models']:
                first_model = result['models'][0]
                print(f"First model type: {type(first_model)}")
                print(f"First model keys: {list(first_model.keys())}")
                
                for key, value in first_model.items():
                    print(f"  {key}: {type(value)} = {repr(value)[:50]}...")
                    
        except Exception as e:
            print(f"JSON serialization FAILED: {e}")
            print(f"Error type: {type(e)}")
            
            # Debug each part
            for key, value in result.items():
                print(f"Testing result['{key}']...")
                try:
                    json.dumps(value)
                    print(f"  {key}: OK")
                except Exception as field_error:
                    print(f"  {key}: FAILED - {field_error}")
                    
                    if key == "models" and isinstance(value, list):
                        for i, model in enumerate(value):
                            print(f"    Testing model {i}...")
                            try:
                                json.dumps(model)
                                print(f"      model {i}: OK")
                            except Exception as model_error:
                                print(f"      model {i}: FAILED - {model_error}")
                                
                                if isinstance(model, dict):
                                    for model_key, model_value in model.items():
                                        try:
                                            json.dumps(model_value)
                                            print(f"        {model_key}: OK")
                                        except Exception as attr_error:
                                            print(f"        {model_key}: FAILED - {attr_error}")
    
    except Exception as e:
        print(f"GENERAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_client_json())