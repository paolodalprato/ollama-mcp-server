"""
Ollama Client - Ollama MCP Server v1.0.0
Core client for communication with Ollama server

Design Principles:
- Type safety with full annotations
- Immutable configuration  
- Comprehensive error handling
- Clean separation of concerns
"""

import asyncio
import json
import logging
import concurrent.futures
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Information about an Ollama model"""
    name: str
    size: int
    modified: str
    digest: str
    
    @property
    def size_human(self) -> str:
        """Human readable size"""
        return self._format_bytes(self.size)
    
    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes to human readable string"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"


class OllamaConnectionError(Exception):
    """Raised when cannot connect to Ollama server"""
    pass


class OllamaAPIError(Exception):
    """Raised when Ollama API returns an error"""
    pass


class OllamaClient:
    """
    Core client for Ollama server communication
    
    Provides async interface for all Ollama operations including
    model management, chat, and server health checks.
    """
    
    def __init__(self, host: str = "http://localhost:11434", timeout: int = 30):
        """
        Initialize Ollama client
        
        Args:
            host: Ollama server host URL
            timeout: Request timeout in seconds
        """
        self.host = host
        self.timeout = timeout
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.client = None
        self._client_initialized = False
        self._client_error = None
        logger.debug(f"Initialized OllamaClient for {host}")
    
    def _ensure_client(self):
        """Ensure ollama client is initialized, return success status"""
        if self._client_initialized:
            return self.client is not None
        
        try:
            import ollama
            self.client = ollama.Client(host=self.host)
            self._client_initialized = True
            logger.debug(f"Ollama client created for {self.host}")
            return True
        except ImportError as e:
            self._client_error = "ollama package required. Install with: pip install ollama"
            logger.error(self._client_error)
            self._client_initialized = True
            return False
        except Exception as e:
            self._client_error = f"Failed to create ollama client: {e}"
            logger.error(self._client_error)
            self._client_initialized = True
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if Ollama server is available and responsive
        
        Returns:
            Dict with health status and available models count
        """
        try:
            # First ensure client is initialized
            if not self._ensure_client():
                return {
                    "healthy": False,
                    "error": self._client_error or "Failed to initialize ollama client",
                    "host": self.host,
                    "message": "Ollama client initialization failed"
                }
            
            loop = asyncio.get_event_loop()
            models = await loop.run_in_executor(
                self.executor,
                self._sync_list_models
            )
            
            logger.info(f"Ollama health check: OK. {len(models)} models available")
            return {
                "healthy": True,
                "models_count": len(models),
                "host": self.host,
                "message": "Ollama server is responsive"
            }
            
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "host": self.host,
                "message": "Ollama server is not accessible"
            }
    
    async def list_models(self) -> List[ModelInfo]:
        """
        Get list of available models from Ollama
        
        Returns:
            List of ModelInfo objects
            
        Raises:
            OllamaConnectionError: If cannot connect to server
            OllamaAPIError: If API returns error
        """
        try:
            # First ensure client is initialized
            if not self._ensure_client():
                raise OllamaConnectionError(f"Cannot initialize ollama client: {self._client_error}")
            
            loop = asyncio.get_event_loop()
            raw_models = await loop.run_in_executor(
                self.executor,
                self._sync_list_models
            )
            
            models = []
            for model_data in raw_models:
                try:
                    models.append(ModelInfo(
                        name=getattr(model_data, 'model', model_data.get('name', 'unknown')),
                        size=getattr(model_data, 'size', model_data.get('size', 0)),
                        modified=str(getattr(model_data, 'modified_at', model_data.get('modified', 'unknown'))),
                        digest=getattr(model_data, 'digest', model_data.get('digest', 'unknown'))
                    ))
                except (AttributeError, KeyError) as e:
                    logger.warning(f"Model data format issue: {e}")
                    # Create minimal ModelInfo for problematic entries
                    models.append(ModelInfo(
                        name=str(model_data).split()[0] if str(model_data) else 'unknown',
                        size=0,
                        modified='unknown',
                        digest='unknown'
                    ))
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            if "connection" in str(e).lower():
                raise OllamaConnectionError(f"Cannot connect to Ollama at {self.host}: {e}")
            else:
                raise OllamaAPIError(f"Ollama API error: {e}")
    
    async def chat(self, model: str, prompt: str, system_prompt: Optional[str] = None, 
                  max_tokens: Optional[int] = None, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate response using specified Ollama model
        
        Args:
            model: Model name to use
            prompt: User prompt  
            system_prompt: Optional system prompt
            max_tokens: Optional max tokens to generate
            temperature: Generation temperature (0.0-1.0)
            
        Returns:
            Dict with response and metadata
            
        Raises:
            OllamaConnectionError: If cannot connect to server
            OllamaAPIError: If generation fails
        """
        try:
            # First ensure client is initialized
            if not self._ensure_client():
                raise OllamaConnectionError(f"Cannot initialize ollama client: {self._client_error}")
            
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            messages.append({
                "role": "user", 
                "content": prompt
            })
            
            # Generation options
            options = {
                "temperature": temperature
            }
            if max_tokens:
                options["num_predict"] = max_tokens
            
            # Execute generation in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                self._sync_generate,
                model,
                messages,
                options
            )
            
            # Extract and format response
            content = response.get('message', {}).get('content', '')
            
            result = {
                "response": content,
                "model_used": model,
                "metadata": {
                    "total_duration": response.get('total_duration', 0),
                    "load_duration": response.get('load_duration', 0),
                    "prompt_eval_count": response.get('prompt_eval_count', 0),
                    "eval_count": response.get('eval_count', 0),
                    "prompt_eval_duration": response.get('prompt_eval_duration', 0),
                    "eval_duration": response.get('eval_duration', 0)
                }
            }
            
            logger.info(f"Generated response using {model}: {len(content)} chars")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate response with {model}: {e}")
            if "connection" in str(e).lower():
                raise OllamaConnectionError(f"Cannot connect to Ollama: {e}")
            else:
                raise OllamaAPIError(f"Generation failed: {e}")
    
    async def pull_model(self, model_name: str) -> Dict[str, Any]:
        """
        Download a model from Ollama Hub
        
        Args:
            model_name: Name of model to download
            
        Returns:
            Dict with pull status
            
        Raises:
            OllamaConnectionError: If cannot connect to server
            OllamaAPIError: If pull fails
        """
        try:
            # First ensure client is initialized
            if not self._ensure_client():
                raise OllamaConnectionError(f"Cannot initialize ollama client: {self._client_error}")
            
            logger.info(f"Starting pull for model: {model_name}")
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._sync_pull_model,
                model_name
            )
            
            return {
                "success": result,
                "model_name": model_name,
                "message": f"Model {model_name} {'pulled successfully' if result else 'pull failed'}"
            }
            
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            if "connection" in str(e).lower():
                raise OllamaConnectionError(f"Cannot connect to Ollama: {e}")
            else:
                raise OllamaAPIError(f"Pull failed: {e}")
    
    async def remove_model(self, model_name: str) -> Dict[str, Any]:
        """
        Remove a model from local storage
        
        Args:
            model_name: Name of model to remove
            
        Returns:
            Dict with removal status
            
        Raises:
            OllamaConnectionError: If cannot connect to server
            OllamaAPIError: If removal fails
        """
        try:
            # First ensure client is initialized
            if not self._ensure_client():
                raise OllamaConnectionError(f"Cannot initialize ollama client: {self._client_error}")
            
            logger.info(f"Removing model: {model_name}")
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._sync_remove_model,
                model_name
            )
            
            return {
                "success": result,
                "model_name": model_name,
                "message": f"Model {model_name} {'removed successfully' if result else 'removal failed'}"
            }
            
        except Exception as e:
            logger.error(f"Failed to remove model {model_name}: {e}")
            if "connection" in str(e).lower():
                raise OllamaConnectionError(f"Cannot connect to Ollama: {e}")
            else:
                raise OllamaAPIError(f"Removal failed: {e}")
    
    # Private sync methods for executor
    
    def _sync_list_models(self) -> List[Any]:
        """Sync version of list_models for executor"""
        try:
            if self.client is None:
                raise Exception("Ollama client not initialized")
            response = self.client.list()
            return getattr(response, 'models', response.get('models', []))
        except Exception as e:
            logger.error(f"Sync list models failed: {e}")
            raise
    
    def _sync_generate(self, model: str, messages: List[Dict], options: Dict) -> Dict:
        """Sync version of generate for executor"""
        try:
            if self.client is None:
                raise Exception("Ollama client not initialized")
            response = self.client.chat(
                model=model,
                messages=messages,
                options=options
            )
            return response
        except Exception as e:
            raise Exception(f"Ollama generation failed: {e}")
    
    def _sync_pull_model(self, model_name: str) -> bool:
        """Sync version of pull_model for executor"""
        try:
            if self.client is None:
                raise Exception("Ollama client not initialized")
            self.client.pull(model_name)
            return True
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    def _sync_remove_model(self, model_name: str) -> bool:
        """Sync version of remove_model for executor"""
        try:
            if self.client is None:
                raise Exception("Ollama client not initialized")
            self.client.delete(model_name)
            return True
        except Exception as e:
            logger.error(f"Failed to remove model {model_name}: {e}")
            return False
    
    def __del__(self):
        """Cleanup executor on destruction"""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
        except Exception:
            pass


# Export main classes
__all__ = [
    "OllamaClient",
    "ModelInfo", 
    "OllamaConnectionError",
    "OllamaAPIError"
]
