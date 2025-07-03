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

import sys
from pathlib import Path

# Fix import path
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from client import OllamaClient
from job_manager import get_job_manager
from model_manager import ModelManager


def get_advanced_tools() -> List[Tool]:
    """Return list of advanced tools for MCP registration"""
    return [
        Tool(
            name="suggest_models",
            description="Get intelligent model recommendations based on user requirements and system resources",
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
            name="download_model",
            description="Start asynchronous download of a model from Ollama Hub",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Name of the model to download (e.g., 'llama3.2', 'qwen2.5:7b')"
                    }
                },
                "required": ["model_name"]
            }
        ),
        Tool(
            name="check_download_progress",
            description="Check progress of a running model download",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Job ID returned from download_model command"
                    }
                },
                "required": ["job_id"]
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
            name="search_available_models",
            description="Search for available models in Ollama Hub",
            inputSchema={
                "type": "object",
                "properties": {
                    "search_term": {
                        "type": "string",
                        "description": "Search term for finding models (e.g., 'code', 'chat', 'llama')"
                    },
                    "category": {
                        "type": "string",
                        "description": "Model category filter",
                        "enum": ["all", "code", "chat", "reasoning", "multimodal", "small", "large"]
                    }
                },
                "required": []
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
        )
    ]


async def handle_advanced_tool(name: str, arguments: Dict[str, Any], client: OllamaClient) -> List[TextContent]:
    """Handle advanced tool calls"""
    
    if name == "suggest_models":
        return await _handle_suggest_models(arguments, client)
    elif name == "download_model":
        return await _handle_download_model(arguments, client)
    elif name == "check_download_progress":
        return await _handle_check_progress(arguments)
    elif name == "remove_model":
        return await _handle_remove_model(arguments, client)
    elif name == "search_available_models":
        return await _handle_search_models(arguments)
    elif name == "start_ollama_server":
        return await _handle_start_server(client)
    elif name == "select_chat_model":
        return await _handle_select_chat_model(arguments, client)
    else:
        return [TextContent(
            type="text",
            text=f"Unknown advanced tool: {name}"
        )]


async def _handle_suggest_models(arguments: Dict[str, Any], client: OllamaClient) -> List[TextContent]:
    """Provide intelligent model recommendations based on user needs"""
    user_needs = arguments.get("user_needs", "")
    priority = arguments.get("priority", "balanced")
    
    # Analyze user needs and generate recommendations
    recommendations = _analyze_user_needs(user_needs, priority)
    
    # Get system resources for compatibility check
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        available_gb = psutil.virtual_memory().available / (1024**3)
    except:
        memory_gb = 8  # Default assumption
        available_gb = 4
    
    # Filter recommendations by system compatibility
    compatible_models = []
    for rec in recommendations:
        if rec["min_ram_gb"] <= available_gb:
            compatible_models.append(rec)
    
    result = {
        "success": True,
        "user_request": user_needs,
        "priority": priority,
        "system_info": {
            "total_memory_gb": round(memory_gb, 1),
            "available_memory_gb": round(available_gb, 1)
        },
        "recommendations": compatible_models[:5],  # Top 5 recommendations
        "next_steps": {
            "download": "Use 'download_model' tool with the model name",
            "example": "download_model with model_name='llama3.2'"
        }
    }
    
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _handle_download_model(arguments: Dict[str, Any], client: OllamaClient) -> List[TextContent]:
    """Start asynchronous model download"""
    model_name = arguments.get("model_name", "")
    
    if not model_name:
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": "Model name is required",
                "example": "download_model with model_name='llama3.2'"
            }, indent=2)
        )]
    
    # Use ModelManager for download
    model_manager = ModelManager(client)
    result = await model_manager.download_model_async(model_name, show_progress=True)
    
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _handle_check_progress(arguments: Dict[str, Any]) -> List[TextContent]:
    """Check download progress"""
    job_id = arguments.get("job_id", "")
    
    if not job_id:
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": "Job ID is required"
            }, indent=2)
        )]
    
    job_manager = get_job_manager()
    status = job_manager.get_job_status(job_id)
    
    if status:
        # Add progress visualization
        progress = status.get("progress_percent", 0)
        bar_length = 20
        filled = int(bar_length * progress / 100)
        progress_bar = "#" * filled + "-" * (bar_length - filled)
        
        status["progress_visualization"] = f"[{progress_bar}] {progress}%"
        
        return [TextContent(type="text", text=json.dumps(status, indent=2))]
    else:
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": f"Job {job_id} not found"
            }, indent=2)
        )]


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


async def _handle_search_models(arguments: Dict[str, Any]) -> List[TextContent]:
    """Search for available models in Ollama Hub"""
    search_term = arguments.get("search_term", "")
    category = arguments.get("category", "all")
    
    # Curated model database with categories
    available_models = [
        {
            "name": "llama3.2",
            "description": "Meta's Llama 3.2 - Excellent general purpose model",
            "size": "2GB",
            "category": ["chat", "reasoning"],
            "quality": "high",
            "speed": "medium",
            "min_ram_gb": 4
        },
        {
            "name": "llama3.2:1b",
            "description": "Compact Llama 3.2 - Fast and lightweight",
            "size": "1.3GB", 
            "category": ["chat", "small"],
            "quality": "medium",
            "speed": "fast",
            "min_ram_gb": 2
        },
        {
            "name": "qwen2.5:7b",
            "description": "Alibaba's Qwen 2.5 - Strong reasoning and coding",
            "size": "4.4GB",
            "category": ["code", "reasoning", "chat"],
            "quality": "high",
            "speed": "medium",
            "min_ram_gb": 6
        },
        {
            "name": "qwen2.5-coder:7b",
            "description": "Specialized coding model based on Qwen 2.5",
            "size": "4.4GB",
            "category": ["code"],
            "quality": "high",
            "speed": "medium", 
            "min_ram_gb": 6
        },
        {
            "name": "phi3.5",
            "description": "Microsoft's Phi 3.5 - Compact but capable",
            "size": "2.2GB",
            "category": ["chat", "reasoning", "small"],
            "quality": "high",
            "speed": "fast",
            "min_ram_gb": 3
        },
        {
            "name": "mistral:7b",
            "description": "Mistral 7B - Fast and efficient general model",
            "size": "4.1GB",
            "category": ["chat", "reasoning"],
            "quality": "high",
            "speed": "fast",
            "min_ram_gb": 5
        },
        {
            "name": "codellama:7b",
            "description": "Meta's Code Llama - Specialized for programming",
            "size": "3.8GB",
            "category": ["code"],
            "quality": "high",
            "speed": "medium",
            "min_ram_gb": 5
        }
    ]
    
    # Filter by search term and category
    filtered_models = []
    for model in available_models:
        # Category filter
        if category != "all" and category not in model["category"]:
            continue
        
        # Search term filter
        if search_term and search_term.lower() not in model["name"].lower() and search_term.lower() not in model["description"].lower():
            continue
        
        filtered_models.append(model)
    
    result = {
        "success": True,
        "search_term": search_term,
        "category": category,
        "models_found": len(filtered_models),
        "models": filtered_models,
        "download_instruction": "Use 'download_model' tool with the model name to download"
    }
    
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _handle_start_server(client: OllamaClient) -> List[TextContent]:
    """Attempt to start Ollama server using Desktop Commander pattern"""
    import platform
    import subprocess
    
    try:
        # First check if already running
        health = await client.health_check()
        if health["healthy"]:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "message": "Ollama server is already running",
                    "status": "already_active",
                    "server_url": health["host"]
                }, indent=2)
            )]
        
        # Use Desktop Commander approach: spawn with shell=True
        try:
            # Platform-specific spawn configuration
            spawn_options = {"shell": True}
            if platform.system() == "Windows":
                spawn_options["creationflags"] = subprocess.CREATE_NO_WINDOW
            
            # Start ollama serve process (Desktop Commander pattern)
            process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **spawn_options
            )
            
            if not process.pid:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "message": "Failed to get process ID for Ollama server",
                        "status": "spawn_failed"
                    }, indent=2)
                )]
            
            # Wait for server to become healthy (Desktop Commander timeout pattern)
            startup_timeout = 15  # seconds
            for attempt in range(startup_timeout):
                await asyncio.sleep(1)
                
                # Check if process is still running
                if process.poll() is not None:
                    # Process exited, read error output
                    stderr_output = process.stderr.read().decode() if process.stderr else "No error output"
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "success": False,
                            "message": f"Ollama process exited during startup (exit code: {process.returncode})",
                            "status": "process_exited",
                            "error_output": stderr_output,
                            "troubleshooting": {
                                "check_installation": "Verify Ollama is properly installed",
                                "check_permissions": "Ensure you have permission to start services",
                                "manual_start": "Try running 'ollama serve' manually in terminal"
                            }
                        }, indent=2)
                    )]
                
                # Check if server is healthy
                health_check = await client.health_check()
                if health_check["healthy"]:
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "success": True,
                            "message": f"Ollama server started successfully in {attempt + 1} seconds",
                            "status": "started",
                            "server_url": health_check["host"],
                            "process_id": process.pid,
                            "startup_time": f"{attempt + 1}s",
                            "next_steps": "You can now use other tools like 'list_local_models'"
                        }, indent=2)
                    )]
            
            # Timeout reached
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "message": f"Ollama server did not start within {startup_timeout} seconds",
                    "status": "startup_timeout",
                    "process_id": process.pid,
                    "troubleshooting": {
                        "check_logs": "Check Ollama logs for startup errors",
                        "port_check": "Verify port 11434 is not in use",
                        "manual_start": "Try running 'ollama serve' manually in terminal",
                        "system_resources": "Ensure sufficient system resources are available"
                    }
                }, indent=2)
            )]
                
        except FileNotFoundError:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "message": "Ollama not found on system",
                    "status": "not_installed",
                    "solution": {
                        "download": "Install Ollama from https://ollama.com",
                        "verify": "After installation, restart your terminal and ensure 'ollama' is in PATH"
                    }
                }, indent=2)
            )]
        
        except OSError as e:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "message": f"Operating system error starting Ollama: {str(e)}",
                    "status": "os_error",
                    "suggestion": "Check system permissions and PATH configuration"
                }, indent=2)
            )]
            
    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "message": f"Unexpected error starting Ollama: {str(e)}",
                "status": "error"
            }, indent=2)
        )]


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


def _analyze_user_needs(user_needs: str, priority: str) -> List[Dict[str, Any]]:
    """Analyze user requirements and return model recommendations"""
    needs_lower = user_needs.lower()
    
    # Base recommendations with scoring
    recommendations = []
    
    # Coding/Programming
    if any(word in needs_lower for word in ["code", "programming", "develop", "script", "python", "javascript"]):
        recommendations.extend([
            {
                "model_name": "qwen2.5-coder:7b",
                "score": 95,
                "reasons": ["Specialized for coding", "Excellent code generation", "Strong debugging"],
                "size": "4.4GB",
                "min_ram_gb": 6,
                "estimated_speed": "medium",
                "quality": "high"
            },
            {
                "model_name": "codellama:7b", 
                "score": 85,
                "reasons": ["Meta's code specialist", "Good for multiple languages", "Strong comments"],
                "size": "3.8GB",
                "min_ram_gb": 5,
                "estimated_speed": "medium",
                "quality": "high"
            }
        ])
    
    # Creative Writing
    if any(word in needs_lower for word in ["write", "creative", "story", "novel", "article", "content"]):
        recommendations.extend([
            {
                "model_name": "llama3.2",
                "score": 90,
                "reasons": ["Excellent creative writing", "Natural language flow", "Good storytelling"],
                "size": "2GB",
                "min_ram_gb": 4,
                "estimated_speed": "medium",
                "quality": "high"
            },
            {
                "model_name": "mistral:7b",
                "score": 80,
                "reasons": ["Creative and coherent", "Good narrative skills", "Fast generation"],
                "size": "4.1GB", 
                "min_ram_gb": 5,
                "estimated_speed": "fast",
                "quality": "high"
            }
        ])
    
    # General Chat/Conversation
    if any(word in needs_lower for word in ["chat", "talk", "conversation", "help", "assistant", "general"]):
        recommendations.extend([
            {
                "model_name": "llama3.2",
                "score": 85,
                "reasons": ["Excellent conversational AI", "Helpful and informative", "Good reasoning"],
                "size": "2GB",
                "min_ram_gb": 4,
                "estimated_speed": "medium", 
                "quality": "high"
            },
            {
                "model_name": "phi3.5",
                "score": 80,
                "reasons": ["Compact but capable", "Good general knowledge", "Fast responses"],
                "size": "2.2GB",
                "min_ram_gb": 3,
                "estimated_speed": "fast",
                "quality": "high"
            }
        ])
    
    # Analysis/Reasoning
    if any(word in needs_lower for word in ["analyze", "analysis", "reasoning", "logic", "research", "understand"]):
        recommendations.extend([
            {
                "model_name": "qwen2.5:7b",
                "score": 92,
                "reasons": ["Strong analytical reasoning", "Good at complex problems", "Detailed explanations"],
                "size": "4.4GB",
                "min_ram_gb": 6,
                "estimated_speed": "medium",
                "quality": "high"
            }
        ])
    
    # If no specific category detected, provide general recommendations
    if not recommendations:
        recommendations.extend([
            {
                "model_name": "llama3.2",
                "score": 85,
                "reasons": ["Versatile general-purpose model", "Good balance of capabilities", "Reliable performance"],
                "size": "2GB",
                "min_ram_gb": 4,
                "estimated_speed": "medium",
                "quality": "high"
            },
            {
                "model_name": "phi3.5",
                "score": 75,
                "reasons": ["Lightweight but capable", "Good for general tasks", "Lower resource requirements"],
                "size": "2.2GB",
                "min_ram_gb": 3,
                "estimated_speed": "fast",
                "quality": "medium"
            }
        ])
    
    # Adjust scores based on priority
    if priority == "speed":
        for rec in recommendations:
            if rec["estimated_speed"] == "fast":
                rec["score"] += 10
            elif rec["estimated_speed"] == "slow":
                rec["score"] -= 10
    elif priority == "quality":
        for rec in recommendations:
            if rec["quality"] == "high":
                rec["score"] += 15
            elif rec["quality"] == "medium":
                rec["score"] -= 5
    
    # Sort by score and remove duplicates
    seen_models = set()
    unique_recommendations = []
    for rec in sorted(recommendations, key=lambda x: x["score"], reverse=True):
        if rec["model_name"] not in seen_models:
            seen_models.add(rec["model_name"])
            unique_recommendations.append(rec)
    
    return unique_recommendations
