"""
Model Manager - Ollama MCP Server v0.9.0
Comprehensive model management for Ollama

Design Principles:
- Type safety with full annotations
- Immutable configuration
- Comprehensive error handling
- Clean separation of concerns
"""

import asyncio
import logging
import subprocess
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

from .client import OllamaClient

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Comprehensive model management for Ollama
    
    Handles model download, removal, switching, and provides
    intelligent recommendations based on task requirements.
    """
    
    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        """
        Initialize model manager
        
        Args:
            ollama_client: Optional OllamaClient instance
        """
        self.client = ollama_client or OllamaClient()
        self.default_model: Optional[str] = None
        self._models_cache: Optional[Dict[str, Any]] = None
        logger.debug("Initialized ModelManager")
    
    async def list_models(self) -> Dict[str, Any]:
        """
        List all locally available models with detailed info
        
        Returns:
            Dict with models list and metadata
        """
        try:
            models_result = await self.client.list_models()
            
            if not models_result["success"]:
                return {
                    "success": False,
                    "error": models_result.get("error", "Failed to list models"),
                    "models": []
                }
            
            # Format models for output
            formatted_models = [
                {
                    "name": model["name"],
                    "size_bytes": model["size"],
                    "size_human": model["size_human"],
                    "modified": model.get("modified_display", model.get("modified_at", "Unknown")),
                    "is_default": model["name"] == self.default_model
                }
                for model in models_result["models"]
            ]
            
            return {
                "success": True,
                "models": formatted_models,
                "total_models": len(formatted_models),
                "default_model": self.default_model
            }
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return {
                "success": False,
                "error": str(e),
                "models": []
            }
    
    async def _refresh_model_cache(self) -> bool:
        """
        Force refresh of model cache after state-changing operations
        
        Returns:
            True if cache refresh succeeded
        """
        try:
            # Clear any existing cache
            self._models_cache = None
            # Force fresh fetch from Ollama
            fresh_models = await self.client.list_models()
            if fresh_models["success"]:
                self._models_cache = fresh_models
                logger.debug("Model cache refreshed successfully")
                return True
            else:
                logger.warning(f"Cache refresh failed: {fresh_models.get('error', 'Unknown error')}")
                return False
        except Exception as e:
            logger.error(f"Cache refresh failed: {e}")
            return False
    
    async def remove_model(self, model_name: str, force: bool = False) -> Dict[str, Any]:
        """
        Remove a model from local storage
        
        Args:
            model_name: Name of model to remove
            force: Force removal even if model is default
            
        Returns:
            Dict with removal results
        """
        try:
            # Check if model exists
            models_result = await self.client.list_models()
            if not models_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to check models: {models_result.get('error', 'Unknown error')}",
                    "model_name": model_name
                }
            
            model_exists = any(m["name"] == model_name for m in models_result["models"])
            
            if not model_exists:
                return {
                    "success": False,
                    "error": f"Model {model_name} not found locally",
                    "model_name": model_name
                }
            
            # Check if it's the default model
            if not force and model_name == self.default_model:
                return {
                    "success": False,
                    "error": f"Model {model_name} is the default model. Use force=True to remove",
                    "model_name": model_name,
                    "is_default": True
                }
            
            # Get model size before removal
            model_info = next((m for m in models_result["models"] if m["name"] == model_name), None)
            model_size_mb = (model_info["size"] / (1024 * 1024)) if model_info else 0
            
            # Remove the model
            result = await self.client.remove_model(model_name)
            
            if result["success"]:
                # Clear default if this was the default model
                if model_name == self.default_model:
                    self.default_model = None
                
                # AUTO-REFRESH cache after successful removal
                await self._refresh_model_cache()
                
                return {
                    "success": True,
                    "message": f"Model {model_name} removed successfully",
                    "model_name": model_name,
                    "space_freed_mb": model_size_mb
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Removal failed"),
                    "model_name": model_name
                }
                
        except Exception as e:
            logger.error(f"Error removing model {model_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_name": model_name
            }
    
    async def switch_default_model(self, model_name: str) -> Dict[str, Any]:
        """
        Switch the default model
        
        Args:
            model_name: Name of model to set as default
            
        Returns:
            Dict with switch results
        """
        try:
            # Check if model exists locally
            models_result = await self.client.list_models()
            if not models_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to check models: {models_result.get('error', 'Unknown error')}",
                    "model_name": model_name
                }
            
            model_exists = any(m["name"] == model_name for m in models_result["models"])
            
            if not model_exists:
                return {
                    "success": False,
                    "error": f"Model {model_name} not found locally",
                    "model_name": model_name
                }
            
            # Set as default
            old_default = self.default_model
            self.default_model = model_name
            
            return {
                "success": True,
                "message": f"Default model switched from {old_default} to {model_name}",
                "old_default": old_default,
                "new_default": model_name
            }
            
        except Exception as e:
            logger.error(f"Error switching default model to {model_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_name": model_name
            }
    
    async def _test_model_responsiveness(self, model_name: str) -> Dict[str, Any]:
        """
        Test if a model responds quickly to simple queries
        
        Args:
            model_name: Model to test
            
        Returns:
            Dict with responsiveness info
        """
        try:
            import time
            start_time = time.time()
            
            # Simple test prompt
            result = await self.client.chat(model_name, "Hi", temperature=0.1)
            
            response_time = (time.time() - start_time) * 1000  # ms
            
            return {
                "responsive": result["success"],
                "response_time_ms": round(response_time, 2),
                "test_successful": result["success"]
            }
            
        except Exception as e:
            logger.debug(f"Responsiveness test failed for {model_name}: {e}")
            return {
                "responsive": False,
                "response_time_ms": None,
                "test_successful": False,
                "error": str(e)
            }
    
    def _validate_model_name(self, model_name: str) -> bool:
        """
        Validate model name format
        
        Args:
            model_name: Model name to validate
            
        Returns:
            True if valid format
        """
        if not model_name or not isinstance(model_name, str):
            return False
        
        # Basic validation - model names should not be empty and not contain dangerous chars
        invalid_chars = ['<', '>', '|', '&', ';', '`', '$']
        if any(char in model_name for char in invalid_chars):
            return False
        
        return True


# Export main classes
__all__ = [
    "ModelManager",
]
