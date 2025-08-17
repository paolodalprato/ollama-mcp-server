"""
Ollama Client - Bridge Compatible Version
Exactly matching the working claude-ollama-bridge approach
"""

import asyncio
import json
import logging
import concurrent.futures
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

try:
    import ollama
except ImportError:
    raise ImportError("ollama package required. Install with: pip install ollama")


@dataclass
class ModelInfo:
    """Bridge-compatible model information"""
    name: str
    size: int
    modified: str
    digest: str
    
    @property
    def size_human(self) -> str:
        """Human readable size"""
        s = self.size
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if s < 1024.0:
                return f"{s:.1f} {unit}"
            s /= 1024.0
        return f"{s:.1f} PB"
    
    @property
    def modified_display(self) -> str:
        """Format date for display"""
        try:
            if not self.modified:
                return "Unknown"
            
            # Simple date formatting - keep first part of ISO date
            if 'T' in self.modified:
                date_part = self.modified.split('T')[0]
                time_part = self.modified.split('T')[1][:5]
                return f"{date_part} {time_part}"
            
            return self.modified[:18]
            
        except:
            return "Unknown"


class OllamaClient:
    """Bridge-compatible Ollama client - exact copy of working version"""
    
    def __init__(self, host: str = "http://localhost:11434", timeout: int = 30):
        self.host = host
        self.timeout = timeout
        self.client = ollama.Client(host=host)
        self.logger = logging.getLogger(__name__)
        
        # Thread pool for sync operations
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
    async def health_check(self) -> Dict[str, Any]:
        """Bridge-compatible health check"""
        try:
            # Execute test connection in separate thread
            loop = asyncio.get_event_loop()
            models = await loop.run_in_executor(
                self.executor,
                self._sync_list_models_raw
            )
            
            self.logger.info(f"Ollama health check: OK. {len(models)} models available")
            return {
                "healthy": True,
                "models_count": len(models),
                "host": self.host,
                "message": "Ollama server responsive"
            }
            
        except Exception as e:
            self.logger.error(f"Ollama health check failed: {e}")
            return {"healthy": False, "error": str(e), "host": self.host}
    
    def _sync_list_models_raw(self) -> List[Dict]:
        """Raw sync version for health check"""
        try:
            response = self.client.list()
            return response.models if hasattr(response, 'models') else []
        except Exception as e:
            self.logger.error(f"Sync list models failed: {e}")
            return []
    
    async def list_models(self) -> Dict[str, Any]:
        """Bridge-compatible list models - returns dict with plain dicts only"""
        try:
            # Execute sync operation in thread
            loop = asyncio.get_event_loop()
            raw_models = await loop.run_in_executor(
                self.executor,
                self._sync_list_models_raw
            )
            
            models = []
            for model_data in raw_models:
                try:
                    # Return plain dict instead of ModelInfo object to avoid serialization issues
                    model_dict = {
                        "name": str(model_data.model),
                        "size": int(model_data.size),
                        "modified": str(model_data.modified_at),  # Direct str() conversion - key fix
                        "digest": str(model_data.digest),
                        "size_human": self._format_size_human(int(model_data.size)),
                        "modified_display": self._format_modified_display(str(model_data.modified_at))
                    }
                    models.append(model_dict)
                except AttributeError as e:
                    # Handle API differences
                    self.logger.warning(f"Model data format issue: {e}")
                    model_dict = {
                        "name": str(getattr(model_data, 'model', 'unknown')),
                        "size": int(getattr(model_data, 'size', 0)),
                        "modified": str(getattr(model_data, 'modified_at', 'unknown')),
                        "digest": str(getattr(model_data, 'digest', 'unknown')),
                        "size_human": self._format_size_human(int(getattr(model_data, 'size', 0))),
                        "modified_display": "Unknown"
                    }
                    models.append(model_dict)
            
            return {"success": True, "models": models, "count": len(models)}
            
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            return {"success": False, "error": str(e), "models": []}
    
    def _format_size_human(self, size: int) -> str:
        """Human readable size"""
        s = size
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if s < 1024.0:
                return f"{s:.1f} {unit}"
            s /= 1024.0
        return f"{s:.1f} PB"
    
    def _format_modified_display(self, modified: str) -> str:
        """Format date for display"""
        try:
            if not modified or modified == "unknown":
                return "Unknown"
            
            # Simple date formatting - keep first part of ISO date
            if 'T' in modified:
                date_part = modified.split('T')[0]
                time_part = modified.split('T')[1][:5]
                return f"{date_part} {time_part}"
            
            return modified[:18]
            
        except:
            return "Unknown"
    
    async def chat(self, model: str, prompt: str, temperature: float = 0.7) -> Dict[str, Any]:
        """Bridge-compatible chat method"""
        try:
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            options = {"temperature": temperature}
            
            # Execute in thread
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                self._sync_generate,
                model,
                messages,
                options
            )
            
            # Extract response
            content = response['message']['content']
            
            return {
                "success": True,
                "response": content,
                "model": model,
                "metadata": {
                    "total_duration": response.get('total_duration', 0),
                    "eval_count": response.get('eval_count', 0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate response with {model}: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": ""
            }
    
    def _sync_generate(self, model: str, messages: List[Dict], options: Dict) -> Dict:
        """Sync version for thread executor"""
        try:
            response = self.client.chat(
                model=model,
                messages=messages,
                options=options
            )
            return response
        except Exception as e:
            raise Exception(f"Ollama generation failed: {e}")
    
    async def show(self, model_name: str) -> Dict[str, Any]:
        """Bridge-compatible show method"""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                self._sync_show,
                model_name
            )
            
            return {
                "success": True,
                "name": model_name,
                **response
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _sync_show(self, model_name: str) -> Dict:
        """Sync show for thread executor"""
        try:
            response = self.client.show(model_name)
            # Convert to dict and ensure no datetime objects
            result = {}
            if hasattr(response, 'model_dump'):
                result = response.model_dump()
            else:
                result = dict(response)
            
            # Clean any datetime objects
            for key, value in result.items():
                if hasattr(value, 'isoformat'):  # datetime object
                    result[key] = str(value)
                    
            return result
        except Exception as e:
            raise Exception(f"Show failed: {e}")
    
    async def remove_model(self, model_name: str) -> Dict[str, Any]:
        """Bridge-compatible remove model"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self._sync_remove,
                model_name
            )
            
            return {"success": True, "message": f"Model {model_name} removed successfully", "model_name": model_name}
            
        except Exception as e:
            return {"success": False, "error": str(e), "model_name": model_name}
    
    def _sync_remove(self, model_name: str):
        """Sync remove for thread executor"""
        try:
            self.client.delete(model_name)
        except Exception as e:
            raise Exception(f"Remove failed: {e}")
    
    def __del__(self):
        """Cleanup executor"""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
        except Exception:
            pass
