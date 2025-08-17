#!/usr/bin/env python3
import asyncio
import traceback
from src.ollama_mcp.client import OllamaClient
from src.ollama_mcp.tools.base_tools import handle_base_tool

async def test_list_models():
    print("TEST: list_local_models")
    
    try:
        client = OllamaClient()
        result = await handle_base_tool("list_local_models", {}, client)
        
        print("SUCCESSO! Risultato tipo:", type(result))
        print("Numero di risultati:", len(result))
        
        if result and len(result) > 0:
            print("Primo risultato tipo:", type(result[0]))
            content = str(result[0])[:200]
            print("Contenuto (primi 200 char):", content)
        
        print("Test serializzazione JSON...")
        import json
        json_str = json.dumps([str(r) for r in result])
        print("Serializzazione JSON: RIUSCITA")
        
    except Exception as e:
        print("ERRORE:", str(e))
        print("Tipo errore:", type(e))
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_list_models())