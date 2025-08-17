#!/usr/bin/env python3
"""
Mostra la lista dei modelli usando il tool list_local_models corretto
"""

import asyncio
import sys
sys.path.insert(0, './src')

from ollama_mcp.client import OllamaClient
from ollama_mcp.tools.base_tools import handle_base_tool

async def show_models():
    print("=" * 60)
    print("LISTA MODELLI OLLAMA - Tool list_local_models")
    print("=" * 60)
    
    try:
        # Inizializza client
        client = OllamaClient()
        
        # Chiama il tool list_local_models
        result = await handle_base_tool("list_local_models", {}, client)
        
        if result and len(result) > 0:
            # Estrai il contenuto testuale
            content = result[0]
            if hasattr(content, 'text'):
                print(content.text)
            else:
                print(str(content))
        else:
            print("Nessun risultato ricevuto")
            
    except Exception as e:
        print(f"ERRORE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(show_models())