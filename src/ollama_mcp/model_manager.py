"""
Model Manager - Ollama MCP Server v1.0.0
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

from .client import OllamaClient, ModelInfo
from .job_manager import get_job_manager, JobManager

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
        self.job_manager = get_job_manager()
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
                    "name": model.name,
                    "size_bytes": model.size,
                    "size_human": model.size_human,
                    "modified": model.modified,
                    "is_default": model.name == self.default_model
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
            return {
                "success": False,
                "error": str(e),
                "models": []
            }
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
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
    
    async def download_model_async(self, model_name: str, 
                                  show_progress: bool = True) -> Dict[str, Any]:
        """
        Start asynchronous model download
        
        Args:
            model_name: Name of model to download
            show_progress: Whether to track progress
            
        Returns:
            Dict with job ID for tracking
        """
        try:
            # Validate model name
            if not self._validate_model_name(model_name):
                return {
                    "success": False,
                    "error": "Invalid model name format",
                    "model_name": model_name
                }
            
            # Check if model already exists
            models_result = await self.client.list_models()
            
            if models_result["success"]:
                for m in models_result["models"]:
                    if m.name == model_name:
                        return {
                            "success": True,
                            "message": f"Model {model_name} already exists locally",
                            "model_name": model_name,
                            "already_exists": True
                        }
            
            # Create download job
            job_id = self.job_manager.create_job(
                job_type="model_download",
                metadata={
                    "model_name": model_name,
                    "show_progress": show_progress
                }
            )
            
            # Start background download
            job_started = await self.job_manager.start_job(
                job_id,
                self._download_model_job,
                model_name,
                show_progress
            )
            
            if job_started:
                return {
                    "success": True,
                    "job_id": job_id,
                    "message": f"Download started for {model_name}",
                    "model_name": model_name,
                    "status": "started"
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to start download job",
                    "model_name": model_name
                }
                
        except Exception as e:
            logger.error(f"Error starting download for {model_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_name": model_name
            }
    
    async def _download_model_job(self, progress_callback: Callable, 
                                 model_name: str, show_progress: bool) -> Dict[str, Any]:
        """
        Background job for model download
        
        Args:
            progress_callback: Function to report progress
            model_name: Model to download
            show_progress: Whether to track progress
            
        Returns:
            Dict with download results
        """
        try:
            progress_callback(10, "Starting download...")
            
            start_time = time.time()
            
            # Use Ollama client for download
            result = await self.client.pull_model(model_name)
            
            download_time = time.time() - start_time
            
            if result["success"]:
                progress_callback(90, "Verifying download...")
                
                # Verify model is available
                models_result = await self.client.list_models()
                if models_result["success"]:
                    model_found = any(m.name == model_name for m in models_result["models"])
                else:
                    model_found = False
                
                if model_found:
                    progress_callback(100, "Download completed")
                    
                    # AUTO-REFRESH cache after successful download
                    await self._refresh_model_cache()
                    
                    return {
                        "success": True,
                        "message": f"Model {model_name} downloaded successfully",
                        "model_name": model_name,
                        "download_time_seconds": download_time
                    }
                else:
                    return {
                        "success": False,
                        "error": "Model not found after download",
                        "model_name": model_name
                    }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Download failed"),
                    "model_name": model_name
                }
                
        except Exception as e:
            logger.error(f"Download job failed for {model_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_name": model_name
            }
    
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
            
            model_exists = any(m.name == model_name for m in models_result["models"])
            
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
            model_info = next((m for m in models_result["models"] if m.name == model_name), None)
            model_size_mb = (model_info.size / (1024 * 1024)) if model_info else 0
            
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
            return {
                "success": False,
                "error": str(e),
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
            
            model_exists = any(m.name == model_name for m in models_result["models"])
            
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
    "ModelRecommendation"
]
