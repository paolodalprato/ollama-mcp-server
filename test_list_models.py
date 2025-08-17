#!/usr/bin/env python3
"""
Test script per verificare che list_local_models funzioni correttamente
"""

import asyncio
import sys
import traceback
from src.ollama_mcp.client import OllamaClient
from src.ollama_mcp.tools.base_tools import handle_base_tool

async def test_list_models():
    """Test della funzione list_local_models"""
    
    print("=" * 60)
    print("TEST: list_local_models dopo correzione DateTimeEncoder")
    print("=" * 60)
    
    try:
        # Inizializza client
        client = OllamaClient()
        
        print("1. Test chiamata handle_base_tool con list_local_models...")
        result = await handle_base_tool("list_local_models", {}, client)
        
        print(f"✅ SUCCESSO! Risultato tipo: {type(result)}")
        print(f"Numero di risultati: {len(result)}")
        
        if result and len(result) > 0:
            print(f"Primo risultato tipo: {type(result[0])}")
            print(f"Contenuto (primi 200 char): {str(result[0])[:200]}...")
        
        print("\n2. Test serializzazione JSON diretta del risultato...")
        import json
        json_str = json.dumps([r.__dict__ if hasattr(r, '__dict__') else str(r) for r in result])
        print("✅ Serializzazione JSON: RIUSCITA")
        
    except Exception as e:
        print(f"❌ ERRORE: {e}")
        print(f"Tipo errore: {type(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_list_models())