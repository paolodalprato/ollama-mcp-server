#!/usr/bin/env python3
"""
Debug script per analizzare i dati grezzi dei modelli di Ollama
Questo script mostrerà come si presentano i dati prima della serializzazione JSON
"""

import asyncio
import sys
import json
from pprint import pprint
import ollama
from datetime import datetime

async def debug_raw_model_data():
    """Analizza i dati grezzi restituiti da ollama.list()"""
    
    print("=" * 60)
    print("DEBUG: Analisi dati grezzi modelli Ollama")
    print("=" * 60)
    
    try:
        # Inizializza client ollama
        client = ollama.AsyncClient(host="http://localhost:11434", timeout=30)
        
        print("\n1. Chiamata diretta a ollama.list()...")
        response = await client.list()
        print(f"Tipo response: {type(response)}")
        print(f"Response: {response}")
        
        print("\n2. Conversione response.models con model_dump()...")
        raw_models = []
        for i, model_obj in enumerate(response.models):
            print(f"\n--- MODELLO {i+1} ---")
            print(f"Tipo model_obj: {type(model_obj)}")
            
            # Conversione a dict
            model_data = model_obj.model_dump()
            print(f"Dati grezzi model_dump(): {type(model_data)}")
            
            # Stampa ogni campo con il suo tipo
            print("Campi del modello:")
            for key, value in model_data.items():
                print(f"  {key}: {type(value)} = {repr(value)}")
            
            raw_models.append(model_data)
            
        print("\n3. Test serializzazione JSON...")
        try:
            json_str = json.dumps(raw_models, indent=2)
            print("✅ Serializzazione JSON riuscita")
        except Exception as e:
            print(f"❌ ERRORE serializzazione JSON: {e}")
            print(f"Tipo errore: {type(e)}")
            
            # Cerca oggetti datetime
            print("\n4. Ricerca oggetti datetime...")
            for i, model in enumerate(raw_models):
                for key, value in model.items():
                    if isinstance(value, datetime):
                        print(f"  🔍 TROVATO datetime in modello {i}: {key} = {value} (tipo: {type(value)})")
                    elif hasattr(value, '__dict__'):
                        print(f"  🔍 Oggetto complesso in {key}: {type(value)}")
                        
        print("\n5. Test con conversione datetime->string...")
        clean_models = []
        for model in raw_models:
            clean_model = {}
            for key, value in model.items():
                if isinstance(value, datetime):
                    clean_model[key] = value.isoformat()
                    print(f"  Convertito {key}: {value} -> {clean_model[key]}")
                else:
                    clean_model[key] = value
            clean_models.append(clean_model)
        
        try:
            json_str = json.dumps(clean_models, indent=2)
            print("✅ Serializzazione JSON con datetime convertiti: RIUSCITA")
        except Exception as e:
            print(f"❌ Ancora errore: {e}")
        
    except Exception as e:
        print(f"❌ ERRORE generale: {e}")
        print(f"Tipo errore: {type(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_raw_model_data())