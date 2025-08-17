"""
Ollama Client - MCP Server v2.0 Refactored
Fully asynchronous, resilient client for Ollama communication.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import ollama

logger = logging.getLogger(__name__)


def _format_datetime_safe(dt_value):
    """Convert datetime objects to string format safely using ISO format."""
    if isinstance(dt_value, datetime):
        return dt_value.isoformat()
    elif isinstance(dt_value, dict):
        # Recursively clean datetime objects in dictionaries
        clean_dict = {}
        for k, v in dt_value.items():
            clean_dict[k] = _format_datetime_safe(v)
        return clean_dict
    elif isinstance(dt_value, list):
        # Recursively clean datetime objects in lists
        return [_format_datetime_safe(item) for item in dt_value]
    else:
        return dt_value


def _format_date_display(date_str: str) -> str:
    """Format date string to readable format, following claude-ollama-bridge approach."""
    try:
        if not date_str:
            return "Unknown"
        
        # Simple date formatting - keep first part of ISO date
        if 'T' in date_str:
            date_part = date_str.split('T')[0]
            time_part = date_str.split('T')[1][:5]
            return f"{date_part} {time_part}"
        
        return date_str[:18]
        
    except:
        return "Unknown"


@dataclass
class ModelInfo:
    """Basic model information, following claude-ollama-bridge approach"""
    name: str
    size: int
    digest: str
    modified_at: str  # ← KEY: stored as STRING, not datetime object
    details: Optional[Dict[str, Any]] = None
    
    @property
    def modified_display(self) -> str:
        """Format date for display"""
        return _format_date_display(self.modified_at)
    
    @property 
    def modified(self) -> str:
        """Backward compatibility property"""
        return self.modified_at
    
    @property
    def size_human(self) -> str:
        """Human readable size"""
        s = self.size
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if s < 1024.0:
                return f"{s:.1f} {unit}"
            s /= 1024.0
        return f"{s:.1f} PB"

class OllamaClient:
    """
    Fully asynchronous and resilient Ollama client.
    
    Key features:
    - Uses native ollama.AsyncClient for better performance.
    - Lazy initialization (starts even if Ollama is offline).
    - Graceful error handling.
    """
    
    def __init__(self, host: str = "http://localhost:11434", timeout: int = 30):
        """Initialize client without connecting"""
        self.host = host
        self.timeout = timeout
        self.client: Optional[ollama.AsyncClient] = None
        self._initialized = False
        self._init_error: Optional[str] = None
        logger.debug(f"OllamaClient created for {host}")
    
    def _ensure_client(self) -> bool:
        """Ensure the async client is initialized, return success status"""
        if self._initialized:
            return self.client is not None
        
        try:
            # The ollama package is a required dependency, but we check for robustness.
            self.client = ollama.AsyncClient(host=self.host, timeout=self.timeout)
            self._initialized = True
            logger.debug("Ollama async client initialized successfully")
            return True
        except Exception as e:
            self._init_error = f"Failed to initialize ollama.AsyncClient: {e}"
            self._initialized = True
            self.client = None
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Ollama server health asynchronously."""
        if not self._ensure_client() or not self.client:
            return {"healthy": False, "error": self._init_error, "host": self.host}
        
        try:
            # The 'ps' command is a lightweight way to check for a response.
            await asyncio.wait_for(self.client.ps(), timeout=5.0)
            # A full list is not needed for health check, but can provide model count
            models = await self.list_models()
            return {
                "healthy": True,
                "models_count": models.get("count", 0),
                "host": self.host,
                "message": "Ollama server responsive"
            }
        except asyncio.TimeoutError:
            return {"healthy": False, "error": "Ollama server timeout", "host": self.host}
        except Exception as e:
            return {"healthy": False, "error": str(e), "host": self.host}
    
    async def list_models(self) -> Dict[str, Any]:
        """List available models asynchronously with error handling."""
        if not self._ensure_client() or not self.client:
            return {"success": False, "error": self._init_error, "models": []}
        
        try:
            response = await self.client.list()
            
            models = []
            for model_obj in response.models:
                # Convert pydantic model to dict, following claude-ollama-bridge approach
                model_data = model_obj.model_dump()
                
                # Create ModelInfo for compatibility but convert to dict for JSON serialization
                model_info = ModelInfo(
                    name=model_data.get("model", "unknown"),  # ← KEY: field is "model" not "name"
                    size=model_data.get("size", 0),
                    digest=model_data.get("digest", ""),
                    modified_at=_format_datetime_safe(model_data.get("modified_at", "unknown")),  # ← Convert datetime BEFORE ModelInfo creation
                    details=model_data.get("details")
                )
                
                # Convert to plain dict for JSON serialization 
                model_dict = {
                    "name": model_info.name,
                    "size": model_info.size,
                    "digest": model_info.digest,
                    "modified_at": model_info.modified_at,
                    "modified_display": model_info.modified_display,
                    "size_human": model_info.size_human,
                    "details": model_info.details
                }
                models.append(model_dict)
            
            return {"success": True, "models": models, "count": len(models)}
            
        except asyncio.TimeoutError:
            return {"success": False, "error": "Timeout listing models", "models": []}
        except Exception as e:
            return {"success": False, "error": str(e), "models": []}
    
    async def show(self, model_name: str) -> Dict[str, Any]:
        """Get details about a model asynchronously."""
        if not self._ensure_client() or not self.client:
            return {"success": False, "error": self._init_error}

        try:
            response = await self.client.show(model_name)
            # Convert pydantic response to dict and sanitize datetime objects
            response_dict = response.model_dump()
            response_dict["success"] = True
            response_dict["name"] = model_name # The show command doesn't return the name, so we add it.
            
            # Convert any datetime objects to strings following claude-ollama-bridge approach
            for key, value in response_dict.items():
                if isinstance(value, datetime):
                    response_dict[key] = value.isoformat()
            
            return response_dict
        except ollama.ResponseError as e:
            if e.status_code == 404:
                return {"success": False, "error": f"Model '{model_name}' not found."}
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def chat(self, model: str, prompt: str, temperature: float = 0.7) -> Dict[str, Any]:
        """Generate response using Ollama model asynchronously."""
        if not self._ensure_client() or not self.client:
            return {"success": False, "error": self._init_error, "response": ""}
        
        try:
            messages = [{"role": "user", "content": prompt}]
            options = {"temperature": temperature}
            
            response = await self.client.chat(model=model, messages=messages, options=options)
            
            content = response.get('message', {}).get('content', '')
            
            return {
                "success": True,
                "response": content,
                "model": model,
                "metadata": {
                    "eval_count": response.get('eval_count', 0),
                    "total_duration": response.get('total_duration', 0)
                }
            }
            
        except asyncio.TimeoutError:
            return {"success": False, "error": "Chat timeout - model may be loading", "response": ""}
        except Exception as e:
            return {"success": False, "error": str(e), "response": ""}
    
    async def pull_model(self, model_name: str, stream: bool = False) -> Dict[str, Any]:
        """Download/pull a model from Ollama Hub asynchronously."""
        if not self._ensure_client() or not self.client:
            return {"success": False, "error": self._init_error}
        
        try:
            # The async pull method can stream statuses, but for a simple pull we set stream=False.
            result = await self.client.pull(model_name, stream=stream)
            
            return {
                "success": True,
                "message": f"Model {model_name} downloaded successfully",
                "model_name": model_name,
                "status": result.get("status")
            }
            
        except asyncio.TimeoutError:
            return {"success": False, "error": "Download timeout - model may be very large", "model_name": model_name}
        except Exception as e:
            return {"success": False, "error": str(e), "model_name": model_name}

    async def pull_model_stream(self, model_name: str):
        """Yields progress for a model download."""
        if not self._ensure_client() or not self.client:
            yield {"success": False, "error": self._init_error}
            return

        try:
            async for progress in self.client.pull(model_name, stream=True):
                yield {"success": True, "progress": progress}
        except Exception as e:
            yield {"success": False, "error": str(e)}

    async def remove_model(self, model_name: str) -> Dict[str, Any]:
        """Remove a model from local storage asynchronously."""
        if not self._ensure_client() or not self.client:
            return {"success": False, "error": self._init_error}
        
        try:
            await self.client.delete(model_name)
            
            return {"success": True, "message": f"Model {model_name} removed successfully", "model_name": model_name}
            
        except asyncio.TimeoutError:
            return {"success": False, "error": "Remove timeout", "model_name": model_name}
        except ollama.ResponseError as e:
             # Handle case where model does not exist
            if e.status_code == 404:
                return {"success": False, "error": f"Model '{model_name}' not found.", "model_name": model_name}
            return {"success": False, "error": str(e), "model_name": model_name}
        except Exception as e:
            return {"success": False, "error": str(e), "model_name": model_name}
