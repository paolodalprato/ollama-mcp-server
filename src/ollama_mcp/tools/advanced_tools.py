"""
Advanced Tools - MCP Server v0.9.2
Extended functionality for comprehensive Ollama management

Tools:
1. suggest_models - Intelligent model recommendations based on user needs
2. download_model - Start asynchronous model download
3. check_download_progress - Monitor download progress
4. remove_model - Remove models from local storage
5. search_available_models - Search Ollama Hub for models
6. start_ollama_server - Attempt to start Ollama if offline
7. select_chat_model - Interactive model selection for chat
"""

import json
import subprocess
import asyncio
import time
from typing import Dict, Any, List
from mcp.types import Tool, TextContent

from ollama_mcp.client import OllamaClient
from ollama_mcp.model_manager import ModelManager


def get_advanced_tools() -> List[Tool]:
    """Return list of advanced tools for MCP registration"""
    return [
        Tool(
            name="suggest_models",
            description="Suggests the best **locally installed** model for a specific task based on user needs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_needs": {
                        "type": "string",
                        "description": "Description of what the user wants to do with the model (e.g., 'I want to write code', 'I need help with creative writing', 'I want to analyze documents')"
                    },
                    "priority": {
                        "type": "string",
                        "description": "Priority: 'speed' for fast responses, 'quality' for best results, 'balanced' for compromise",
                        "enum": ["speed", "quality", "balanced"]
                    }
                },
                "required": ["user_needs"]
            }
        ),
        Tool(
            name="remove_model",
            description="Remove a model from local storage",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Name of the model to remove"
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Force removal even if it's the default model",
                        "default": False
                    }
                },
                "required": ["model_name"]
            }
        ),
        Tool(
            name="start_ollama_server",
            description="Attempt to start Ollama server if it's not running",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="select_chat_model",
            description="Present available models and help user select one for chat",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message the user wants to send after selecting a model"
                    }
                },
                "required": ["message"]
            }
        ),
        Tool(
            name="test_model_responsiveness",
            description="Test the responsiveness of a specific model by sending a simple prompt.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "The name of the model to test."
                    }
                },
                "required": ["model_name"]
            }
        )
    ]


async def handle_advanced_tool(name: str, arguments: Dict[str, Any], client: OllamaClient) -> List[TextContent]:
    """Handle advanced tool calls"""
    
    if name == "suggest_models":
        return await _handle_suggest_models(arguments, client)
    elif name == "remove_model":
        return await _handle_remove_model(arguments, client)
    elif name == "start_ollama_server":
        return await _handle_start_server(client)
    elif name == "select_chat_model":
        return await _handle_select_chat_model(arguments, client)
    elif name == "test_model_responsiveness":
        return await _handle_test_model_responsiveness(arguments, client)
    else:
        return [TextContent(
            type="text",
            text=f"Unknown advanced tool: {name}"
        )]


async def _handle_suggest_models(arguments: Dict[str, Any], client: OllamaClient) -> List[TextContent]:
    """Suggests the best locally installed model for a given task."""
    user_needs = arguments.get("user_needs", "")
    
    # 1. Get locally installed models
    local_models_result = await client.list_models()
    if not local_models_result["success"] or not local_models_result["models"]:
        return [TextContent(type="text", text=json.dumps({
            "success": False,
            "error": "No local models found.",
            "suggestion": "Download a model first using the 'download_model' tool."
        }, indent=2))]
    
    # 2. Get details for each local model
    local_models = local_models_result["models"]
    model_details = []
    for model in local_models:
        details = await client.show(model.name)
        if details.get("success"):
            model_details.append(details)

    if not model_details:
        return [TextContent(type="text", text=json.dumps({
            "success": False,
            "error": "Could not retrieve details for any local models."
        }, indent=2))]

    # 3. Analyze and rank models
    recommendations = _analyze_local_models(user_needs, model_details)
    
    result = {
        "success": True,
        "user_request": user_needs,
        "recommendations": recommendations,
        "next_steps": {
            "chat": "Use 'local_llm_chat' with the recommended model name.",
            "example": f"local_llm_chat with model='{recommendations[0]['name'] if recommendations else '...'}' and message='Your question...'"
        }
    }
    
    return [TextContent(type="text", text=json.dumps(result, indent=2))]




async def _handle_remove_model(arguments: Dict[str, Any], client: OllamaClient) -> List[TextContent]:
    """Remove a model from local storage"""
    model_name = arguments.get("model_name", "")
    force = arguments.get("force", False)
    
    if not model_name:
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": "Model name is required"
            }, indent=2)
        )]
    
    model_manager = ModelManager(client)
    result = await model_manager.remove_model(model_name, force=force)
    
    return [TextContent(type="text", text=json.dumps(result, indent=2))]




async def _handle_start_server(client: OllamaClient) -> List[TextContent]:
    """Attempt to start Ollama server using the dedicated OllamaServerController."""
    from ollama_mcp.ollama_server_control import OllamaServerController

    controller = OllamaServerController(host=client.host)
    result = await controller.start_server()
    
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _handle_select_chat_model(arguments: Dict[str, Any], client: OllamaClient) -> List[TextContent]:
    """Present model selection for chat"""
    message = arguments.get("message", "")
    
    if not message:
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": "Message is required"
            }, indent=2)
        )]
    
    # Get available models
    models_result = await client.list_models()
    
    if not models_result["success"] or not models_result["models"]:
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "message": "No models available for chat",
                "user_message": message,
                "next_steps": {
                    "download_model": "Download a model first using 'download_model'",
                    "suggestions": "Try 'suggest_models' to see recommended models"
                }
            }, indent=2)
        )]
    
    # Present model selection with details
    model_options = []
    for i, model in enumerate(models_result["models"], 1):
        model_options.append({
            "index": i,
            "name": model.name,
            "size": model.size_human,
            "modified": model.modified
        })
    
    result = {
        "success": True,
        "message": "Please select a model for your chat",
        "user_message": message,
        "available_models": model_options,
        "instructions": "Respond with the model name you want to use, then I'll send your message to that model"
    }
    
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _handle_test_model_responsiveness(arguments: Dict[str, Any], client: OllamaClient) -> List[TextContent]:
    """Handle testing a model's responsiveness."""
    model_name = arguments.get("model_name")
    if not model_name:
        return [TextContent(type="text", text=json.dumps({"success": False, "error": "model_name is a required argument."}, indent=2))]

    model_manager = ModelManager(client)
    # The responsiveness test is a private method, but we can call it here for the tool.
    # A cleaner refactor could make it a public method if desired.
    result = await model_manager._test_model_responsiveness(model_name)
    
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


CONCEPT_KEYWORDS = {
    "coding": {"code", "coding", "develop", "development", "sviluppo", "programmazione", "script", "coder", "codellama", "devstral"},
    "writing": {"write", "writing", "scrivere", "creative", "storia", "racconto", "article", "articolo", "creativewriting"},
    "chat": {"chat", "conversation", "conversazione", "general", "generale", "assistant", "assistente", "llama3"},
    "vision": {"vision", "image", "multimodal", "immagine", "vedere", "bakllava", "llava"},
    "medical": {"medical", "medicine", "medicina", "medico", "medgemma", "meditron", "medllama"},
    "reasoning": {"reasoning", "logic", "ragionamento", "logica", "deepseek-r1"},
}

def _analyze_local_models(user_needs: str, model_details: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Analyze local models based on user needs and return ranked recommendations using concept matching."""
    user_keywords = set(user_needs.lower().split())
    
    # Identify concepts from user's request
    requested_concepts = set()
    for concept, keywords in CONCEPT_KEYWORDS.items():
        if not user_keywords.isdisjoint(keywords):
            requested_concepts.add(concept)

    # If no specific concept is found, assume general chat
    if not requested_concepts:
        requested_concepts.add("chat")

    scored_models = []
    for details in model_details:
        score = 0
        reasons = []

        model_name = details.get("name", "")

        # Build searchable text from model info
        searchable_text = model_name.lower()
        searchable_text += details.get("modelfile", "").lower()
        searchable_text += json.dumps(details.get("details", {})).lower()

        # Score model based on how well it matches the requested concepts
        for concept in requested_concepts:
            for keyword in CONCEPT_KEYWORDS[concept]:
                if keyword in searchable_text:
                    score += 10
                    reasons.append(f"Suitable for '{concept}' (matched keyword: '{keyword}')")
                    break # Only score once per concept

        if score > 0:
            scored_models.append({
                "name": model_name,
                "score": score,
                "reasons": sorted(list(set(reasons))), # Unique reasons
                "family": details.get("details", {}).get("family"),
                "parameter_size": details.get("details", {}).get("parameter_size"),
            })
    
    # Sort by score
    scored_models.sort(key=lambda x: x["score"], reverse=True)
    
    return scored_models
