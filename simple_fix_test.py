#!/usr/bin/env python3
import asyncio
import json
from datetime import datetime
import ollama

async def test_simple_fix():
    print("=== SIMPLE FIX TEST ===")
    
    # Get raw data
    client = ollama.AsyncClient(host="http://localhost:11434", timeout=30)
    response = await client.list()
    
    if response.models:
        model_obj = response.models[0]
        model_data = model_obj.model_dump()
        
        print("Raw fields:")
        for key, value in model_data.items():
            print(f"  {key}: {type(value)} = {repr(value)}")
        
        # Manual conversion following claude-ollama-bridge approach
        converted_data = {
            'name': model_data.get("model", "unknown"),
            'size': model_data.get("size", 0),
            'digest': model_data.get("digest", ""),
            'modified_at': model_data.get("modified_at").isoformat() if isinstance(model_data.get("modified_at"), datetime) else str(model_data.get("modified_at", "")),
            'details': model_data.get("details")
        }
        
        print(f"\nConverted data:")
        for key, value in converted_data.items():
            print(f"  {key}: {type(value)} = {repr(value)}")
        
        # Test JSON serialization
        try:
            json_str = json.dumps(converted_data)
            print("\nJSON serialization: SUCCESS")
            print(f"JSON length: {len(json_str)}")
        except Exception as e:
            print(f"\nJSON serialization: FAILED - {e}")

if __name__ == "__main__":
    asyncio.run(test_simple_fix())