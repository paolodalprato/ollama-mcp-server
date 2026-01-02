"""
Base Tools - MCP Server v0.9.3 Enhanced
Essential 4 tools with enhanced error handling and robustness

Tools:
1. list_local_models - List available models
2. local_llm_chat - Chat with local models  
3. ollama_health_check - Diagnose Ollama status
4. system_resource_check - Enhanced system info with robust GPU detection

v0.9.3 improvements:
- Robust GPU detection parsing (no more crashes on non-numeric VRAM)
- Enhanced error handling with specific exception types
- Increased timeouts for better reliability (5s -> 10s)
- Graceful degradation when GPU detection fails
- Better cross-platform compatibility
"""

import json
import psutil
from typing import Dict, Any, List
from mcp.types import Tool, TextContent

from ollama_mcp.client import OllamaClient, DateTimeEncoder


# === TOOL DEFINITIONS ===
# Each tool is defined with name, description, and JSON Schema for input validation.
# The MCP client uses these definitions to present available tools to the user.

def get_base_tools() -> List[Tool]:
    """Return list of base tools for MCP registration"""
    return [
        Tool(
            name="list_local_models",
            description="List all locally installed Ollama models with details",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="local_llm_chat", 
            description="Chat with a local Ollama model",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message to send to the model"
                    },
                    "model": {
                        "type": "string", 
                        "description": "Model name (optional, uses first available if not specified)"
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Generation temperature 0.0-1.0 (default: 0.7)"
                    }
                },
                "required": ["message"]
            }
        ),
        Tool(
            name="ollama_health_check",
            description="Check Ollama server health and provide diagnostics", 
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="system_resource_check",
            description="Check system resources and compatibility",
            inputSchema={
                "type": "object", 
                "properties": {},
                "required": []
            }
        )
    ]


# === TOOL HANDLERS ===
# Each handler processes arguments, calls the client, and formats the response.
# All handlers return List[TextContent] for MCP protocol compliance.

async def handle_base_tool(name: str, arguments: Dict[str, Any], client: OllamaClient) -> List[TextContent]:
    """Route tool calls to appropriate handler functions."""
    
    if name == "list_local_models":
        return await _handle_list_models(client)
    elif name == "local_llm_chat":
        return await _handle_chat(arguments, client)
    elif name == "ollama_health_check":
        return await _handle_health_check(client)
    elif name == "system_resource_check":
        return await _handle_system_check()
    else:
        return [TextContent(
            type="text",
            text=f"Unknown base tool: {name}"
        )]


async def _handle_list_models(client: OllamaClient) -> List[TextContent]:
    """Handle list models - Bridge compatible text formatting approach"""
    result = await client.list_models()
    
    if result["success"]:
        if result["models"]:
            # Use bridge approach: format as text, not JSON
            response = "[MODELLI] LLM Locali Disponibili\n\n"
            
            for model in result["models"]:
                size_gb = model["size"] / (1024**3) if model["size"] > 0 else 0
                response += f"- {model['name']}\n"
                response += f"  Dimensione: {size_gb:.1f} GB\n"
                response += f"  Aggiornato: {model['modified_display']}\n\n"
            
            response += f"[TOTALE] Modelli disponibili: {result['count']}\n"
            response += "[PRIVACY] Tutti i modelli vengono eseguiti localmente, nessun dato inviato al cloud"
            
        else:
            response = "[ERROR] Nessun modello Ollama trovato.\n\n"
            response += "Possibili soluzioni:\n"
            response += "1. Scarica un modello: ollama pull llama3.2\n"
            response += "2. Verifica Ollama: ollama list\n"
            response += "3. Modelli popolari: llama3.2, qwen2.5, phi3.5, mistral"
    else:
        # Ollama not accessible - provide helpful guidance
        response = "[ERROR] Ollama Server non accessibile\n\n"
        response += f"Errore: {result['error']}\n\n"
        response += "Soluzioni:\n"
        response += "1. Verifica installazione: https://ollama.com\n"
        response += "2. Avvia server: ollama serve\n"
        response += "3. Controlla porta 11434\n"
        response += "4. Usa 'ollama_health_check' per diagnostica"
    
    # Return plain text like the bridge, NO JSON serialization
    return [TextContent(type="text", text=response)]


async def _handle_chat(arguments: Dict[str, Any], client: OllamaClient) -> List[TextContent]:
    """Handle chat with automatic model selection"""
    message = arguments.get("message", "")
    model = arguments.get("model")
    temperature = arguments.get("temperature", 0.7)
    
    if not message:
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": "Message is required",
                "example": "Use: local_llm_chat with message='Hello, how are you?'"
            }, cls=DateTimeEncoder, indent=2)
        )]
    
    # Auto-select model if not specified
    if not model:
        models_result = await client.list_models()
        if models_result["success"] and models_result["models"]:
            model = str(models_result["models"][0]["name"])  # Access dict key, not object attr
        else:
            return [TextContent(
                type="text", 
                text=json.dumps({
                    "success": False,
                    "error": "No models available",
                    "user_message": message,
                    "next_steps": {
                        "download_model": "Download a model: 'ollama pull llama3.2'",
                        "check_server": "Verify Ollama is running: 'ollama_health_check'"
                    }
                }, cls=DateTimeEncoder, indent=2)
            )]
    
    # Generate response
    result = await client.chat(model, message, temperature)
    
    if result["success"]:
        chat_result = {
            "success": True,
            "response": result["response"],
            "model_used": model,
            "user_message": message,
            "metadata": result.get("metadata", {}),
            "privacy_note": "All processing done locally - no data sent to cloud"
        }
    else:
        chat_result = {
            "success": False,
            "error": result["error"],
            "user_message": message,
            "model_requested": model,
            "troubleshooting": {
                "check_model": f"Verify '{model}' is available with 'list_local_models'",
                "check_server": "Check Ollama server with 'ollama_health_check'",
                "download_model": f"If model missing: 'ollama pull {model}'"
            }
        }
    
    return [TextContent(type="text", text=json.dumps(chat_result, cls=DateTimeEncoder, indent=2))]


async def _handle_health_check(client: OllamaClient) -> List[TextContent]:
    """Comprehensive health check with actionable guidance"""
    health = await client.health_check()
    
    # Enhance health check with actionable guidance
    if health["healthy"]:
        health_result = {
            "status": "HEALTHY",
            "server_url": health["host"],
            "models_available": health.get("models_count", 0),
            "message": "Ollama server is running and responsive",
            "next_steps": {
                "list_models": "See available models: 'list_local_models'", 
                "start_chat": "Start chatting: 'local_llm_chat' with your message"
            }
        }
    else:
        health_result = {
            "status": "UNHEALTHY",
            "server_url": health["host"],
            "error": health["error"],
            "troubleshooting": {
                "step_1": "Check if Ollama is installed: 'ollama --version'",
                "step_2": "Start Ollama server: 'ollama serve'",
                "step_3": "Install if missing: https://ollama.com",
                "step_4": "Check firewall/antivirus blocking port 11434"
            },
            "quick_fixes": {
                "terminal_command": "ollama serve",
                "installation_url": "https://ollama.com"
            }
        }
    
    return [TextContent(type="text", text=json.dumps(health_result, cls=DateTimeEncoder, indent=2))]


async def _handle_system_check() -> List[TextContent]:
    """Complete system resource check using the dedicated HardwareChecker."""
    from ollama_mcp.hardware_checker import HardwareChecker
    
    checker = HardwareChecker()
    system_info = await checker.get_system_info()
    
    return [TextContent(type="text", text=json.dumps(system_info, cls=DateTimeEncoder, indent=2))]
