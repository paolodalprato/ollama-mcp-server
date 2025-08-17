#!/usr/bin/env python3
"""Test finale di conferma che list_local_models funziona"""

import asyncio
import sys
sys.path.insert(0, './src')

from ollama_mcp.tools.base_tools import handle_base_tool
from ollama_mcp.client import OllamaClient

async def final_test():
    print("=== TEST FINALE DI CONFERMA ===")
    print("Percorso server:", sys.path[0])
    
    try:
        client = OllamaClient()
        print("1. Client inizializzato")
        
        # Test diretto del tool
        result = await handle_base_tool("list_local_models", {}, client)
        print(f"2. Tool eseguito - Risultati: {len(result)}")
        
        if result and len(result) > 0:
            content = result[0].text if hasattr(result[0], 'text') else str(result[0])
            
            # Verifica che non ci siano errori
            if "Error executing" in content:
                print("❌ ERRORE: Trovato messaggio di errore!")
                print(content)
            elif "Object of type datetime is not JSON serializable" in content:
                print("❌ ERRORE: Trovato errore datetime!")
            elif "[MODELLI]" in content and "gpt-oss:20b" in content:
                print("✅ SUCCESSO: Lista modelli formattata correttamente")
                lines = content.split('\n')
                model_count = len([line for line in lines if line.strip().startswith('- ')])
                print(f"   Modelli trovati: {model_count}")
                print(f"   Prima riga: {lines[0]}")
                print(f"   Ultimo modello: {[line for line in lines if line.strip().startswith('- ')][-1] if model_count > 0 else 'N/A'}")
            else:
                print("⚠️ ANOMALIA: Contenuto inaspettato")
                print(f"Prime 100 char: {content[:100]}")
        else:
            print("❌ ERRORE: Nessun risultato")
            
    except Exception as e:
        print(f"❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(final_test())