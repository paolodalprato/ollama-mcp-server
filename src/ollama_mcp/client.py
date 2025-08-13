"""
Ollama Client - MCP Server v2.0 Refactored
Fully asynchronous, resilient client for Ollama communication.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import ollama

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Basic model information"""
    name: str
    size: int
    modified: str
    
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
            raw_models = response.get('models', [])
            
            models = [
                ModelInfo(
                    name=model.get('name', 'unknown'),
                    size=model.get('size', 0),
                    modified=model.get('modified_at', 'unknown')
                )
                for model in raw_models
            ]
            
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
            response["success"] = True
            response["name"] = model_name # The show command doesn't return the name, so we add it.
            return response
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
