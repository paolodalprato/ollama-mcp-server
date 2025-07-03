"""
Ollama Client - MCP Server v1.1 Simplified
Resilient client for Ollama communication with lazy initialization

Based on v0.9 resilience patterns but dramatically simplified
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

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
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if self.size < 1024.0:
                return f"{self.size:.1f} {unit}"
            self.size /= 1024.0
        return f"{self.size:.1f} PB"


class OllamaClient:
    """
    Simplified resilient Ollama client
    
    Key features:
    - Lazy initialization (starts even if Ollama offline)
    - Non-blocking health checks
    - Graceful error handling
    - Minimal dependencies
    """
    
    def __init__(self, host: str = "http://localhost:11434", timeout: int = 30):
        """Initialize client without connecting"""
        self.host = host
        self.timeout = timeout
        self.client = None
        self._initialized = False
        self._init_error = None
        logger.debug(f"OllamaClient created for {host}")
    
    def _ensure_client(self) -> bool:
        """Ensure client is initialized, return success status"""
        if self._initialized:
            return self.client is not None
        
        try:
            import ollama
            self.client = ollama.Client(host=self.host)
            self._initialized = True
            logger.debug("Ollama client initialized successfully")
            return True
        except ImportError:
            self._init_error = "ollama package not installed. Run: pip install ollama"
            self._initialized = True
            return False
        except Exception as e:
            self._init_error = f"Failed to initialize: {e}"
            self._initialized = True
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Ollama server health without blocking"""
        if not self._ensure_client():
            return {
                "healthy": False,
                "error": self._init_error,
                "host": self.host
            }
        
        try:
            # Try to list models as health check
            loop = asyncio.get_event_loop()
            models = await asyncio.wait_for(
                loop.run_in_executor(None, self._sync_list),
                timeout=5.0
            )
            
            return {
                "healthy": True,
                "models_count": len(models),
                "host": self.host,
                "message": "Ollama server responsive"
            }
        except asyncio.TimeoutError:
            return {
                "healthy": False,
                "error": "Ollama server timeout",
                "host": self.host
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "host": self.host
            }
    
    async def list_models(self) -> Dict[str, Any]:
        """List available models with error handling"""
        if not self._ensure_client():
            return {
                "success": False,
                "error": self._init_error,
                "models": []
            }
        
        try:
            loop = asyncio.get_event_loop()
            raw_models = await asyncio.wait_for(
                loop.run_in_executor(None, self._sync_list),
                timeout=10.0
            )
            
            models = []
            for model_data in raw_models:
                try:
                    models.append(ModelInfo(
                        name=getattr(model_data, 'model', model_data.get('name', 'unknown')),
                        size=getattr(model_data, 'size', model_data.get('size', 0)),
                        modified=str(getattr(model_data, 'modified_at', model_data.get('modified', 'unknown')))
                    ))
                except Exception as e:
                    logger.warning(f"Model parsing error: {e}")
                    continue
            
            return {
                "success": True,
                "models": models,
                "count": len(models)
            }
            
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "Timeout listing models",
                "models": []
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "models": []
            }
    
    async def chat(self, model: str, prompt: str, temperature: float = 0.7) -> Dict[str, Any]:
        """Generate response using Ollama model"""
        if not self._ensure_client():
            return {
                "success": False,
                "error": self._init_error,
                "response": ""
            }
        
        try:
            messages = [{"role": "user", "content": prompt}]
            options = {"temperature": temperature}
            
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(None, self._sync_chat, model, messages, options),
                timeout=120.0
            )
            
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
            return {
                "success": False,
                "error": "Chat timeout - model may be loading",
                "response": ""
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": ""
            }
    
    async def pull_model(self, model_name: str) -> Dict[str, Any]:
        """Download/pull a model from Ollama Hub"""
        if not self._ensure_client():
            return {
                "success": False,
                "error": self._init_error
            }
        
        try:
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self._sync_pull, model_name),
                timeout=1800.0  # 30 minutes for download
            )
            
            return {
                "success": True,
                "message": f"Model {model_name} downloaded successfully",
                "model_name": model_name
            }
            
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "Download timeout - model may be very large",
                "model_name": model_name
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model_name": model_name
            }
    
    async def remove_model(self, model_name: str) -> Dict[str, Any]:
        """Remove a model from local storage"""
        if not self._ensure_client():
            return {
                "success": False,
                "error": self._init_error
            }
        
        try:
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, self._sync_remove, model_name),
                timeout=30.0
            )
            
            return {
                "success": True,
                "message": f"Model {model_name} removed successfully",
                "model_name": model_name
            }
            
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "Remove timeout",
                "model_name": model_name
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model_name": model_name
            }
    
    # Sync methods for executor
    def _sync_list(self) -> List[Any]:
        """Sync list models"""
        if not self.client:
            raise Exception("Client not initialized")
        response = self.client.list()
        return getattr(response, 'models', response.get('models', []))
    
    def _sync_chat(self, model: str, messages: List[Dict], options: Dict) -> Dict:
        """Sync chat"""
        if not self.client:
            raise Exception("Client not initialized")
        return self.client.chat(model=model, messages=messages, options=options)
    
    def _sync_pull(self, model_name: str) -> Dict:
        """Sync pull model"""
        if not self.client:
            raise Exception("Client not initialized")
        return self.client.pull(model_name)
    
    def _sync_remove(self, model_name: str) -> Dict:
        """Sync remove model"""
        if not self.client:
            raise Exception("Client not initialized")
        return self.client.delete(model_name)
