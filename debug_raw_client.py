#!/usr/bin/env python3
import asyncio
import json
from datetime import datetime
import ollama

async def debug_raw_data():
    print("=== DEBUG RAW OLLAMA DATA ===")
    
    try:
        # Test diretto con ollama library
        client = ollama.AsyncClient(host="http://localhost:11434", timeout=30)
        response = await client.list()
        
        print(f"Response type: {type(response)}")
        print(f"Models count: {len(response.models)}")
        
        if response.models:
            first_model = response.models[0]
            print(f"\nFirst model object type: {type(first_model)}")
            
            # Dump del primo modello
            model_data = first_model.model_dump()
            print(f"Model dump type: {type(model_data)}")
            
            print("\nTutti i campi del model_dump:")
            for key, value in model_data.items():
                print(f"  {key}: {type(value)} = {repr(value)}")
                
                # Test serializzazione del singolo campo
                try:
                    json.dumps(value)
                    print(f"    -> JSON OK")
                except Exception as e:
                    print(f"    -> JSON ERRORE: {e}")
        
        print("\n=== TEST _format_datetime_safe ===")
        
        # Test della funzione _format_datetime_safe
        import sys
        sys.path.append(r'D:\MCP_SERVER\INSTALLED\ollama-mcp-server\src')
        from ollama_mcp.client import _format_datetime_safe
        
        if response.models:
            model_data = response.models[0].model_dump()
            modified_at_raw = model_data.get("modified_at")
            print(f"modified_at raw: {type(modified_at_raw)} = {repr(modified_at_raw)}")
            
            converted = _format_datetime_safe(modified_at_raw)
            print(f"converted: {type(converted)} = {repr(converted)}")
            
            # Test JSON del converted
            try:
                json.dumps(converted)
                print("converted JSON: OK")
            except Exception as e:
                print(f"converted JSON: ERRORE {e}")
        
    except Exception as e:
        print(f"ERRORE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_raw_data())