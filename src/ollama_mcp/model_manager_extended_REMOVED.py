"""
Model Manager Extended Methods - Ollama MCP Server v1.0.0
Additional methods for comprehensive model management
"""

import time
from typing import Dict, List, Optional, Any

# Extended methods for ModelManager class (to be merged)

async def switch_default_model(self, model_name: str) -> Dict[str, Any]:
    """
    Switch the default model
    
    Args:
        model_name: Name of model to set as default
        
    Returns:
        Dict with switch results
    """
    try:
        # Check if model exists
        models = await self.client.list_models()
        model_exists = any(m.name == model_name for m in models)
        
        if not model_exists:
            return {
                "success": False,
                "error": f"Model {model_name} not found locally",
                "model_name": model_name,
                "suggestion": "Download the model first"
            }
        
        # Test model responsiveness
        responsive_test = await self._test_model_responsiveness(model_name)
        if not responsive_test["responsive"]:
            return {
                "success": False,
                "error": f"Model {model_name} is not responsive",
                "model_name": model_name,
                "test_details": responsive_test
            }
        
        previous_default = self.default_model
        self.default_model = model_name
        
        return {
            "success": True,
            "message": f"Default model switched to {model_name}",
            "model_name": model_name,
            "previous_default": previous_default,
            "response_test": responsive_test
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "model_name": model_name
        }

async def get_model_info(self, model_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific model
    
    Args:
        model_name: Name of model to get info for
        
    Returns:
        Dict with detailed model information
    """
    try:
        models = await self.client.list_models()
        model = next((m for m in models if m.name == model_name), None)
        
        if not model:
            return {
                "success": False,
                "error": f"Model {model_name} not found",
                "model_name": model_name
            }
        
        # Test responsiveness
        responsive_test = await self._test_model_responsiveness(model_name)
        
        # Estimate capabilities
        capabilities = self._estimate_model_capabilities(model_name)
        
        return {
            "success": True,
            "model_name": model_name,
            "info": {
                "name": model.name,
                "size_bytes": model.size,
                "size_human": model.size_human,
                "modified": model.modified,
                "digest": model.digest,
                "is_default": model_name == self.default_model,
                "responsive": responsive_test["responsive"],
                "response_time_ms": responsive_test.get("response_time_ms", 0),
                "capabilities": capabilities
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "model_name": model_name
        }

async def recommend_models_for_task(self, task_description: str, 
                                   hardware_limit_gb: Optional[int] = None,
                                   quality_preference: str = "balanced") -> Dict[str, Any]:
    """
    Recommend models based on task requirements
    
    Args:
        task_description: Description of the task
        hardware_limit_gb: RAM limit in GB
        quality_preference: "speed", "balanced", or "quality"
        
    Returns:
        Dict with model recommendations
    """
    try:
        # Analyze task requirements
        task_analysis = self._analyze_task_requirements(task_description)
        
        # Get available models
        models_result = await self.list_models()
        if not models_result["success"]:
            return {
                "success": False,
                "error": "Could not retrieve available models"
            }
        
        local_models = models_result["models"]
        
        # Generate recommendations from local models
        local_recommendations = self._generate_model_recommendations(
            task_analysis, local_models, hardware_limit_gb, quality_preference
        )
        
        # Generate recommendations for downloadable models
        downloadable_recommendations = self._generate_downloadable_recommendations(
            task_analysis, hardware_limit_gb, quality_preference
        )
        
        return {
            "success": True,
            "task_analysis": task_analysis,
            "recommendations": {
                "local_models": [
                    {
                        "model_name": r.model_name,
                        "score": r.score,
                        "reasons": r.reasons,
                        "hardware_compatible": r.hardware_compatible,
                        "estimated_ram_gb": r.estimated_ram_gb,
                        "estimated_speed": r.estimated_speed,
                        "quality_rating": r.quality_rating
                    } for r in local_recommendations
                ],
                "downloadable_models": [
                    {
                        "model_name": r.model_name,
                        "score": r.score,
                        "reasons": r.reasons,
                        "hardware_compatible": r.hardware_compatible,
                        "estimated_ram_gb": r.estimated_ram_gb,
                        "estimated_speed": r.estimated_speed,
                        "quality_rating": r.quality_rating
                    } for r in downloadable_recommendations
                ]
            },
            "hardware_limit_gb": hardware_limit_gb,
            "quality_preference": quality_preference
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "task_description": task_description
        }

# Private helper methods

def _validate_model_name(self, model_name: str) -> bool:
    """Validate model name format"""
    if not model_name or not isinstance(model_name, str):
        return False
    
    # Basic validation for Ollama model names
    if ':' in model_name:
        name, tag = model_name.split(':', 1)
        return bool(name.strip() and tag.strip())
    else:
        return bool(model_name.strip())

async def _test_model_responsiveness(self, model_name: str) -> Dict[str, Any]:
    """Test if model responds to basic requests"""
    try:
        start_time = time.time()
        
        result = await self.client.chat(
            model=model_name,
            prompt="Hello",
            max_tokens=5,
            temperature=0.1
        )
        
        response_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            "responsive": bool(result.get("response")),
            "response_time_ms": int(response_time),
            "test_response": result.get("response", "")[:50] + "..." if len(result.get("response", "")) > 50 else result.get("response", "")
        }
        
    except Exception as e:
        return {
            "responsive": False,
            "error": str(e),
            "response_time_ms": 0
        }

def _estimate_model_capabilities(self, model_name: str) -> Dict[str, Any]:
    """Estimate model capabilities based on name and size"""
    capabilities = {
        "general_chat": True,
        "code_generation": False,
        "multilingual": False,
        "reasoning": False,
        "vision": False,
        "math": False,
        "estimated_quality": "medium"
    }
    
    name_lower = model_name.lower()
    
    # Code models
    if any(keyword in name_lower for keyword in ['code', 'coder', 'starcoder', 'deepseek']):
        capabilities["code_generation"] = True
        capabilities["reasoning"] = True
    
    # Multilingual models
    if any(keyword in name_lower for keyword in ['llama3', 'qwen', 'command', 'aya']):
        capabilities["multilingual"] = True
    
    # Reasoning models
    if any(keyword in name_lower for keyword in ['deepseek-r1', 'o1', 'reasoning']):
        capabilities["reasoning"] = True
        capabilities["math"] = True
    
    # Vision models
    if any(keyword in name_lower for keyword in ['vision', 'vl', 'llava']):
        capabilities["vision"] = True
    
    # Quality estimation based on size and model family
    if '70b' in name_lower or '72b' in name_lower:
        capabilities["estimated_quality"] = "high"
    elif any(size in name_lower for size in ['32b', '34b']):
        capabilities["estimated_quality"] = "high"
    elif any(size in name_lower for size in ['13b', '14b', '15b']):
        capabilities["estimated_quality"] = "medium"
    elif any(size in name_lower for size in ['7b', '8b']):
        capabilities["estimated_quality"] = "medium"
    elif any(size in name_lower for size in ['3b', '1b']):
        capabilities["estimated_quality"] = "basic"
    
    return capabilities

def _analyze_task_requirements(self, task_description: str) -> Dict[str, Any]:
    """Analyze task description to determine requirements"""
    desc_lower = task_description.lower()
    
    requirements = {
        "task_type": "general",
        "requires_code": False,
        "requires_multilingual": False,
        "requires_reasoning": False,
        "requires_vision": False,
        "requires_math": False,
        "quality_importance": "medium",
        "speed_importance": "medium"
    }
    
    # Detect task type
    if any(keyword in desc_lower for keyword in ['code', 'program', 'script', 'debug', 'function']):
        requirements["task_type"] = "coding"
        requirements["requires_code"] = True
    elif any(keyword in desc_lower for keyword in ['translate', 'translation', 'language']):
        requirements["task_type"] = "translation"
        requirements["requires_multilingual"] = True
    elif any(keyword in desc_lower for keyword in ['math', 'calculate', 'equation', 'solve']):
        requirements["task_type"] = "math"
        requirements["requires_math"] = True
        requirements["requires_reasoning"] = True
    elif any(keyword in desc_lower for keyword in ['image', 'picture', 'visual', 'photo']):
        requirements["task_type"] = "vision"
        requirements["requires_vision"] = True
    elif any(keyword in desc_lower for keyword in ['analyze', 'reasoning', 'logic', 'think']):
        requirements["task_type"] = "reasoning"
        requirements["requires_reasoning"] = True
    
    # Detect quality/speed preferences
    if any(keyword in desc_lower for keyword in ['fast', 'quick', 'speed']):
        requirements["speed_importance"] = "high"
        requirements["quality_importance"] = "medium"
    elif any(keyword in desc_lower for keyword in ['accurate', 'precise', 'best', 'quality']):
        requirements["quality_importance"] = "high"
        requirements["speed_importance"] = "medium"
    
    return requirements

def _generate_model_recommendations(self, task_analysis: Dict, local_models: List[Dict], 
                                   hardware_limit_gb: Optional[int], 
                                   quality_preference: str) -> List:
    """Generate recommendations from locally available models"""
    from .model_manager import ModelRecommendation
    
    recommendations = []
    
    for model in local_models:
        score = self._calculate_model_score(model, task_analysis, quality_preference)
        
        # Hardware compatibility check
        estimated_ram = self._estimate_ram_usage(model["name"])
        hardware_compatible = True
        if hardware_limit_gb and estimated_ram > hardware_limit_gb:
            hardware_compatible = False
            score *= 0.3  # Heavily penalize incompatible models
        
        # Generate recommendation reasons
        reasons = self._generate_recommendation_reasons(model, task_analysis)
        
        recommendation = ModelRecommendation(
            model_name=model["name"],
            score=score,
            reasons=reasons,
            hardware_compatible=hardware_compatible,
            estimated_ram_gb=estimated_ram,
            estimated_speed=self._estimate_speed(model["name"]),
            quality_rating=self._estimate_quality_rating(model["name"])
        )
        
        recommendations.append(recommendation)
    
    # Sort by score (highest first)
    recommendations.sort(key=lambda x: x.score, reverse=True)
    
    return recommendations[:5]  # Return top 5

def _generate_downloadable_recommendations(self, task_analysis: Dict, 
                                         hardware_limit_gb: Optional[int],
                                         quality_preference: str) -> List:
    """Generate recommendations for models that can be downloaded"""
    from .model_manager import ModelRecommendation
    
    # Popular models database (simplified)
    popular_models = [
        {"name": "llama3.1:8b", "ram_gb": 8, "quality": "high", "speed": "fast"},
        {"name": "llama3.1:70b", "ram_gb": 48, "quality": "very_high", "speed": "slow"},
        {"name": "deepseek-coder:6.7b", "ram_gb": 7, "quality": "high", "speed": "fast"},
        {"name": "deepseek-r1:14b", "ram_gb": 14, "quality": "very_high", "speed": "medium"},
        {"name": "qwen2.5:14b", "ram_gb": 14, "quality": "high", "speed": "medium"},
        {"name": "command-r:35b", "ram_gb": 24, "quality": "very_high", "speed": "medium"},
    ]
    
    recommendations = []
    
    for model_data in popular_models:
        # Hardware compatibility
        hardware_compatible = True
        if hardware_limit_gb and model_data["ram_gb"] > hardware_limit_gb:
            hardware_compatible = False
            continue  # Skip incompatible models
        
        # Calculate score based on task requirements
        score = self._calculate_downloadable_model_score(model_data, task_analysis, quality_preference)
        
        reasons = self._generate_downloadable_reasons(model_data, task_analysis)
        
        recommendation = ModelRecommendation(
            model_name=model_data["name"],
            score=score,
            reasons=reasons,
            hardware_compatible=hardware_compatible,
            estimated_ram_gb=model_data["ram_gb"],
            estimated_speed=model_data["speed"],
            quality_rating=model_data["quality"]
        )
        
        recommendations.append(recommendation)
    
    # Sort by score and return top 3
    recommendations.sort(key=lambda x: x.score, reverse=True)
    return recommendations[:3]

def _calculate_model_score(self, model: Dict, task_analysis: Dict, quality_preference: str) -> float:
    """Calculate score for a local model"""
    score = 0.5  # Base score
    
    capabilities = self._estimate_model_capabilities(model["name"])
    
    # Task-specific scoring
    if task_analysis["requires_code"] and capabilities["code_generation"]:
        score += 0.3
    if task_analysis["requires_multilingual"] and capabilities["multilingual"]:
        score += 0.25
    if task_analysis["requires_reasoning"] and capabilities["reasoning"]:
        score += 0.25
    if task_analysis["requires_vision"] and capabilities["vision"]:
        score += 0.4
    if task_analysis["requires_math"] and capabilities["math"]:
        score += 0.3
    
    # Quality preference adjustment
    quality_multiplier = {
        "speed": {"basic": 1.2, "medium": 1.0, "high": 0.8},
        "balanced": {"basic": 0.8, "medium": 1.0, "high": 1.1},
        "quality": {"basic": 0.6, "medium": 0.9, "high": 1.3}
    }
    
    quality_rating = capabilities["estimated_quality"]
    score *= quality_multiplier.get(quality_preference, {}).get(quality_rating, 1.0)
    
    # Responsiveness bonus
    if model.get("responsive", False):
        score += 0.1
    
    return min(score, 1.0)  # Cap at 1.0

def _estimate_ram_usage(self, model_name: str) -> float:
    """Estimate RAM usage in GB based on model name"""
    name_lower = model_name.lower()
    
    # Extract parameter count and estimate RAM
    if '70b' in name_lower or '72b' in name_lower:
        return 48.0
    elif '35b' in name_lower or '34b' in name_lower:
        return 24.0
    elif '14b' in name_lower or '15b' in name_lower:
        return 14.0
    elif '8b' in name_lower:
        return 8.0
    elif '7b' in name_lower:
        return 7.0
    elif '3b' in name_lower:
        return 3.0
    elif '1b' in name_lower:
        return 1.5
    else:
        return 8.0  # Default estimate

def _estimate_speed(self, model_name: str) -> str:
    """Estimate inference speed based on model size"""
    ram_usage = self._estimate_ram_usage(model_name)
    
    if ram_usage <= 8:
        return "fast"
    elif ram_usage <= 20:
        return "medium"
    else:
        return "slow"

def _estimate_quality_rating(self, model_name: str) -> str:
    """Estimate quality rating based on model characteristics"""
    capabilities = self._estimate_model_capabilities(model_name)
    return capabilities["estimated_quality"]
