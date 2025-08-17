#!/usr/bin/env python3
import asyncio
import json
from datetime import datetime
import ollama
import sys
sys.path.append(r'D:\MCP_SERVER\INSTALLED\ollama-mcp-server\src')

from ollama_mcp.client import ModelInfo, _format_datetime_safe

async def debug_modelinfo_creation():
    print("=== DEBUG ModelInfo CREATION ===")
    
    try:
        # Get raw data
        client = ollama.AsyncClient(host="http://localhost:11434", timeout=30)
        response = await client.list()
        
        if response.models:
            model_obj = response.models[0]
            model_data = model_obj.model_dump()
            
            print("Raw model_data keys:", list(model_data.keys()))
            print("model field exists:", "model" in model_data)
            print("model value:", model_data.get("model"))
            print("modified_at type:", type(model_data.get("modified_at")))
            
            # Test _format_datetime_safe
            modified_raw = model_data.get("modified_at")
            modified_safe = _format_datetime_safe(modified_raw)
            print(f"modified_at raw: {type(modified_raw)} -> safe: {type(modified_safe)}")
            
            # Create ModelInfo manually step by step
            print("\nCreating ModelInfo:")
            print(f"name: {model_data.get('model', 'unknown')}")
            print(f"size: {model_data.get('size', 0)}")
            print(f"digest: {model_data.get('digest', '')}")
            print(f"modified_at: {modified_safe}")
            
            model = ModelInfo(
                name=model_data.get("model", "unknown"),
                size=model_data.get("size", 0),
                digest=model_data.get("digest", ""),
                modified_at=modified_safe,
                details=model_data.get("details")
            )
            
            print(f"\nModelInfo created: {model}")
            print(f"ModelInfo name: {model.name}")
            print(f"ModelInfo modified_at: {model.modified_at} (type: {type(model.modified_at)})")
            print(f"ModelInfo modified property: {model.modified} (type: {type(model.modified)})")
            
            # Test JSON serialization of the ModelInfo object
            try:
                # Convert to dict for JSON
                model_dict = {
                    'name': model.name,
                    'size': model.size,
                    'digest': model.digest,
                    'modified_at': model.modified_at,
                    'details': model.details
                }
                json_str = json.dumps(model_dict)
                print("ModelInfo JSON serialization: SUCCESS")
            except Exception as e:
                print(f"ModelInfo JSON serialization: FAILED - {e}")
                
                # Check each field
                for field, value in model_dict.items():
                    try:
                        json.dumps(value)
                        print(f"  {field}: OK")
                    except Exception as field_error:
                        print(f"  {field}: FAILED - {field_error}")
        
    except Exception as e:
        print(f"ERRORE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_modelinfo_creation())