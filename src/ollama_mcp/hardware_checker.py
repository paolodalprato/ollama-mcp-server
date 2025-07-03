"""
Hardware Checker - Claude Ollama Bridge v0.9
Cross-platform hardware compatibility and resource checking with multi-GPU support

Design Principles:
- Type safety with full annotations
- Cross-platform compatibility (Windows/Linux/macOS)
- Comprehensive error handling
- Clean separation of concerns
"""

import logging
import platform
import psutil
import subprocess
import shutil
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

from config import HardwareConfig, get_config

logger = logging.getLogger(__name__)

class GPUVendor(Enum):
    """Supported GPU vendors"""
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    APPLE = "apple"
    CPU = "cpu"
    UNKNOWN = "unknown"


@dataclass
class GPUInfo:
    """
    GPU information with vendor-specific details
    
    Contains comprehensive GPU details for different vendors.
    """
    name: str
    vendor: GPUVendor
    total_memory_mb: int
    free_memory_mb: int
    used_memory_mb: int
    driver_support: str
    compute_capability: Optional[str] = None
    temperature_c: Optional[float] = None
    power_usage_w: Optional[float] = None
    utilization_percent: Optional[float] = None
    
    def __post_init__(self):
        """Validate GPU info after creation"""
        self._validate_memory_values()
    
    def _validate_memory_values(self):
        """Validate memory values are consistent"""
        if self.total_memory_mb < 0:
            self.total_memory_mb = 0
        if self.free_memory_mb < 0:
            self.free_memory_mb = 0
        if self.used_memory_mb < 0:
            self.used_memory_mb = 0
        
        # Ensure consistency
        if self.free_memory_mb + self.used_memory_mb > self.total_memory_mb:
            self.used_memory_mb = max(0, self.total_memory_mb - self.free_memory_mb)
    
    @property
    def total_memory_gb(self) -> float:
        """Get total memory in GB"""
        return self.total_memory_mb / 1024
    
    @property
    def free_memory_gb(self) -> float:
        """Get free memory in GB"""
        return self.free_memory_mb / 1024
    
    @property
    def memory_usage_percent(self) -> float:
        """Calculate memory usage percentage"""
        if self.total_memory_mb == 0:
            return 0.0
        return (self.used_memory_mb / self.total_memory_mb) * 100


@dataclass
class SystemInfo:
    """
    Comprehensive system hardware information
    
    Contains detailed system specs with cross-platform compatibility.
    """
    os_name: str
    os_version: str
    cpu_count: int
    cpu_frequency_mhz: Optional[float]
    total_ram_gb: float
    available_ram_gb: float
    gpu_info: List[GPUInfo]
    architecture: str
    platform_details: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate system info after creation"""
        self._validate_values()
    
    def _validate_values(self):
        """Validate system values"""
        if self.total_ram_gb < 0:
            self.total_ram_gb = 0
        if self.available_ram_gb < 0:
            self.available_ram_gb = 0
        if self.cpu_count < 1:
            self.cpu_count = 1
    
    @property
    def ram_usage_percent(self) -> float:
        """Calculate RAM usage percentage"""
        if self.total_ram_gb == 0:
            return 0.0
        return ((self.total_ram_gb - self.available_ram_gb) / self.total_ram_gb) * 100
    
    @property
    def has_gpu(self) -> bool:
        """Check if system has GPU acceleration"""
        return any(gpu.vendor != GPUVendor.CPU for gpu in self.gpu_info)
    
    @property
    def best_gpu(self) -> Optional[GPUInfo]:
        """Get GPU with most memory"""
        gpus = [gpu for gpu in self.gpu_info if gpu.vendor != GPUVendor.CPU]
        return max(gpus, key=lambda g: g.total_memory_mb) if gpus else None


@dataclass
class ModelCompatibility:
    """
    Model compatibility assessment with detailed analysis
    
    Provides comprehensive compatibility analysis for models.
    """
    model_name: str
    compatible: bool
    estimated_ram_gb: float
    estimated_vram_gb: float
    performance_tier: str  # "excellent", "good", "moderate", "poor"
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate compatibility after creation"""
        self._validate_values()
    
    def _validate_values(self):
        """Validate compatibility values"""
        valid_tiers = {"excellent", "good", "moderate", "poor"}
        if self.performance_tier not in valid_tiers:
            self.performance_tier = "moderate"


class CrossPlatformGPUDetector:
    """
    Cross-platform GPU detection with multi-vendor support
    
    Detects GPUs from NVIDIA, AMD, Intel, and Apple across different platforms.
    """
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        """
        Initialize GPU detector
        
        Args:
            config: Hardware configuration, uses global config if None
        """
        self.config = config or get_config().hardware
        self._detection_cache: Optional[List[GPUInfo]] = None
        
        logger.debug("Initialized CrossPlatformGPUDetector")
    
    async def detect_all_gpus(self) -> List[GPUInfo]:
        """
        Detect all available GPUs across vendors
        
        Returns:
            List of detected GPUs
        """
        if not self.config.enable_gpu_detection:
            logger.info("GPU detection disabled in configuration")
            return [self._create_cpu_fallback()]
        
        if self._detection_cache is not None:
            return self._detection_cache
        
        gpus = []
        
        # Detect different GPU types
        nvidia_gpus = await self._detect_nvidia_gpus()
        amd_gpus = await self._detect_amd_gpus()
        intel_gpus = await self._detect_intel_gpus()
        apple_gpus = await self._detect_apple_gpus()
        
        gpus.extend(nvidia_gpus)
        gpus.extend(amd_gpus)
        gpus.extend(intel_gpus)
        gpus.extend(apple_gpus)
        
        # Fallback to CPU if no GPUs found
        if not gpus:
            gpus.append(self._create_cpu_fallback())
        
        self._detection_cache = gpus
        logger.info(f"Detected {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
        
        return gpus
    
    async def _detect_nvidia_gpus(self) -> List[GPUInfo]:
        """Detect NVIDIA GPUs using nvidia-smi"""
        gpus = []
        
        try:
            # Check if nvidia-smi is available
            if not shutil.which("nvidia-smi"):
                logger.debug("nvidia-smi not found, skipping NVIDIA detection")
                return gpus
            
            # Query GPU information
            cmd = [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,memory.used,temperature.gpu,power.draw,utilization.gpu",
                "--format=csv,noheader,nounits"
            ]
            
            result = await self._run_command(cmd, timeout=10)
            
            if result and result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 7:
                            try:
                                gpu = GPUInfo(
                                    name=parts[0],
                                    vendor=GPUVendor.NVIDIA,
                                    total_memory_mb=int(parts[1]),
                                    free_memory_mb=int(parts[2]),
                                    used_memory_mb=int(parts[3]),
                                    driver_support="cuda",
                                    temperature_c=float(parts[4]) if parts[4] != "[Not Supported]" else None,
                                    power_usage_w=float(parts[5]) if parts[5] != "[Not Supported]" else None,
                                    utilization_percent=float(parts[6]) if parts[6] != "[Not Supported]" else None
                                )
                                gpus.append(gpu)
                            except (ValueError, IndexError) as e:
                                logger.warning(f"Failed to parse NVIDIA GPU info: {e}")
        
        except Exception as e:
            logger.debug(f"NVIDIA GPU detection failed: {e}")
        
        return gpus
    
    async def _detect_amd_gpus(self) -> List[GPUInfo]:
        """Detect AMD GPUs using rocm-smi"""
        gpus = []
        
        try:
            # Check if rocm-smi is available
            if not shutil.which("rocm-smi"):
                logger.debug("rocm-smi not found, skipping AMD detection")
                return gpus
            
            # Query GPU information
            cmd = ["rocm-smi", "--showmeminfo", "vram", "--csv"]
            
            result = await self._run_command(cmd, timeout=10)
            
            if result and result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # Skip header
                    for line in lines[1:]:
                        if line.strip():
                            parts = [p.strip() for p in line.split(',')]
                            if len(parts) >= 3:
                                try:
                                    # AMD GPU naming - try to get more details
                                    gpu_name = await self._get_amd_gpu_name(parts[0])
                                    total_mb = int(parts[1]) if parts[1] != "N/A" else 0
                                    used_mb = int(parts[2]) if parts[2] != "N/A" else 0
                                    
                                    gpu = GPUInfo(
                                        name=gpu_name,
                                        vendor=GPUVendor.AMD,
                                        total_memory_mb=total_mb,
                                        free_memory_mb=max(0, total_mb - used_mb),
                                        used_memory_mb=used_mb,
                                        driver_support="rocm"
                                    )
                                    gpus.append(gpu)
                                except (ValueError, IndexError) as e:
                                    logger.warning(f"Failed to parse AMD GPU info: {e}")
        
        except Exception as e:
            logger.debug(f"AMD GPU detection failed: {e}")
        
        return gpus
    
    async def _detect_intel_gpus(self) -> List[GPUInfo]:
        """Detect Intel GPUs using intel-gpu-top or other methods"""
        gpus = []
        
        try:
            # Method 1: Try intel-gpu-top (Linux)
            if shutil.which("intel-gpu-top"):
                cmd = ["intel-gpu-top", "-l"]
                result = await self._run_command(cmd, timeout=5)
                
                if result and result.returncode == 0:
                    # Parse intel-gpu-top output
                    gpu = GPUInfo(
                        name="Intel Integrated Graphics",
                        vendor=GPUVendor.INTEL,
                        total_memory_mb=0,  # Intel GPUs share system RAM
                        free_memory_mb=0,
                        used_memory_mb=0,
                        driver_support="intel"
                    )
                    gpus.append(gpu)
            
            # Method 2: Try lspci on Linux
            elif platform.system() == "Linux" and shutil.which("lspci"):
                cmd = ["lspci", "-nn"]
                result = await self._run_command(cmd, timeout=5)
                
                if result and result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'intel' in line.lower() and any(term in line.lower() for term in ['vga', 'display', 'graphics']):
                            gpu = GPUInfo(
                                name="Intel Integrated Graphics",
                                vendor=GPUVendor.INTEL,
                                total_memory_mb=0,
                                free_memory_mb=0,
                                used_memory_mb=0,
                                driver_support="intel"
                            )
                            gpus.append(gpu)
                            break
        
        except Exception as e:
            logger.debug(f"Intel GPU detection failed: {e}")
        
        return gpus
    
    async def _detect_apple_gpus(self) -> List[GPUInfo]:
        """Detect Apple Silicon GPUs on macOS"""
        gpus = []
        
        try:
            if platform.system() != "Darwin":
                return gpus
            
            # Check for Apple Silicon
            cmd = ["sysctl", "-n", "machdep.cpu.brand_string"]
            result = await self._run_command(cmd, timeout=5)
            
            if result and result.returncode == 0:
                cpu_info = result.stdout.strip()
                if "Apple" in cpu_info:
                    # Determine chip type
                    if "M1" in cpu_info:
                        gpu_name = "Apple M1 GPU"
                    elif "M2" in cpu_info:
                        gpu_name = "Apple M2 GPU"
                    elif "M3" in cpu_info:
                        gpu_name = "Apple M3 GPU"
                    else:
                        gpu_name = "Apple Silicon GPU"
                    
                    # Get system memory (Apple Silicon uses unified memory)
                    memory_cmd = ["sysctl", "-n", "hw.memsize"]
                    memory_result = await self._run_command(memory_cmd, timeout=5)
                    
                    total_memory_mb = 0
                    if memory_result and memory_result.returncode == 0:
                        try:
                            total_memory_bytes = int(memory_result.stdout.strip())
                            total_memory_mb = int(total_memory_bytes / (1024 * 1024))
                        except ValueError:
                            pass
                    
                    gpu = GPUInfo(
                        name=gpu_name,
                        vendor=GPUVendor.APPLE,
                        total_memory_mb=total_memory_mb,
                        free_memory_mb=0,  # Can't easily determine free memory
                        used_memory_mb=0,
                        driver_support="metal"
                    )
                    gpus.append(gpu)
        
        except Exception as e:
            logger.debug(f"Apple GPU detection failed: {e}")
        
        return gpus
    
    async def _get_amd_gpu_name(self, gpu_id: str) -> str:
        """Get detailed AMD GPU name"""
        try:
            cmd = ["rocm-smi", "--showproductname", "--gpu", gpu_id]
            result = await self._run_command(cmd, timeout=5)
            
            if result and result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.strip() and not line.startswith('GPU'):
                        return line.strip()
        except Exception:
            pass
        
        return f"AMD GPU {gpu_id}"
    
    async def _run_command(self, cmd: List[str], timeout: int = 10) -> Optional[subprocess.CompletedProcess]:
        """
        Run command with timeout
        
        Args:
            cmd: Command to run
            timeout: Timeout in seconds
            
        Returns:
            CompletedProcess or None if failed
        """
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False
            )
            return result
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return None
    
    def _create_cpu_fallback(self) -> GPUInfo:
        """Create CPU fallback GPU info"""
        return GPUInfo(
            name="CPU Only",
            vendor=GPUVendor.CPU,
            total_memory_mb=0,
            free_memory_mb=0,
            used_memory_mb=0,
            driver_support="cpu"
        )


class HardwareChecker:
    """
    Cross-platform system hardware compatibility checker
    
    Analyzes system resources and provides compatibility assessments
    for different Ollama models across multiple platforms and GPU vendors.
    """
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        """
        Initialize hardware checker
        
        Args:
            config: Hardware configuration, uses global config if None
        """
        self.config = config or get_config().hardware
        self.gpu_detector = CrossPlatformGPUDetector(self.config)
        
        logger.debug("Initialized HardwareChecker")
    
    async def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information
        
        Returns:
            Dict with detailed system hardware info
        """
        try:
            # Basic system info
            memory = psutil.virtual_memory()
            cpu_freq = psutil.cpu_freq()
            
            # Platform details
            platform_details = {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version()
            }
            
            # Get GPU information
            gpu_info = await self.gpu_detector.detect_all_gpus()
            
            system_info = SystemInfo(
                os_name=platform.system(),
                os_version=platform.release(),
                cpu_count=psutil.cpu_count(),
                cpu_frequency_mhz=cpu_freq.current if cpu_freq else None,
                total_ram_gb=memory.total / (1024**3),
                available_ram_gb=memory.available / (1024**3),
                gpu_info=gpu_info,
                architecture=platform.machine(),
                platform_details=platform_details
            )
            
            return {
                "success": True,
                "system_info": {
                    "os_name": system_info.os_name,
                    "os_version": system_info.os_version,
                    "cpu_count": system_info.cpu_count,
                    "cpu_frequency_mhz": system_info.cpu_frequency_mhz,
                    "total_ram_gb": round(system_info.total_ram_gb, 2),
                    "available_ram_gb": round(system_info.available_ram_gb, 2),
                    "ram_usage_percent": round(system_info.ram_usage_percent, 1),
                    "architecture": system_info.architecture,
                    "has_gpu": system_info.has_gpu,
                    "gpu_info": [
                        {
                            "name": gpu.name,
                            "vendor": gpu.vendor.value,
                            "total_memory_gb": round(gpu.total_memory_gb, 2),
                            "free_memory_gb": round(gpu.free_memory_gb, 2),
                            "memory_usage_percent": round(gpu.memory_usage_percent, 1),
                            "driver_support": gpu.driver_support,
                            "temperature_c": gpu.temperature_c,
                            "power_usage_w": gpu.power_usage_w,
                            "utilization_percent": gpu.utilization_percent
                        }
                        for gpu in system_info.gpu_info
                    ],
                    "platform_details": system_info.platform_details
                }
            }
        
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {
                "success": False,
                "error": str(e),
                "troubleshooting": {
                    "check_permissions": "Ensure sufficient permissions for system monitoring",
                    "check_dependencies": "Verify required system tools are installed"
                }
            }
    
    async def check_model_compatibility(self, model_name: str) -> Dict[str, Any]:
        """
        Check if a specific model is compatible with current system
        
        Args:
            model_name: Name of model to check
            
        Returns:
            Dict with comprehensive compatibility assessment
        """
        try:
            # Get system info
            system_result = await self.get_system_info()
            if not system_result["success"]:
                return {
                    "success": False,
                    "error": "Could not retrieve system information"
                }
            
            system_info = system_result["system_info"]
            
            # Estimate model requirements
            estimated_ram = self._estimate_model_ram_requirements(model_name)
            estimated_vram = self._estimate_model_vram_requirements(model_name)
            
            # Check compatibility
            compatibility = self._assess_model_compatibility(
                model_name, system_info, estimated_ram, estimated_vram
            )
            
            return {
                "success": True,
                "model_name": model_name,
                "compatibility": {
                    "compatible": compatibility.compatible,
                    "performance_tier": compatibility.performance_tier,
                    "estimated_ram_gb": compatibility.estimated_ram_gb,
                    "estimated_vram_gb": compatibility.estimated_vram_gb,
                    "reasons": compatibility.reasons,
                    "warnings": compatibility.warnings,
                    "recommendations": compatibility.recommendations
                },
                "system_requirements": {
                    "estimated_ram_gb": estimated_ram,
                    "estimated_vram_gb": estimated_vram,
                    "minimum_ram_gb": estimated_ram * 1.2,
                    "recommended_ram_gb": estimated_ram * 1.5,
                    "supports_gpu_acceleration": system_info["has_gpu"]
                }
            }
        
        except Exception as e:
            logger.error(f"Error checking model compatibility for {model_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_name": model_name
            }
    
    def _estimate_model_ram_requirements(self, model_name: str) -> float:
        """
        Estimate RAM requirements for a model in GB
        
        Enhanced estimation with better model family recognition.
        """
        name_lower = model_name.lower()
        
        # Parameter-based estimation with improved patterns
        size_patterns = [
            ('405b', 250.0), ('70b', 48.0), ('72b', 48.0),
            ('35b', 24.0), ('34b', 24.0), ('33b', 22.0),
            ('20b', 16.0), ('15b', 12.0), ('14b', 12.0),
            ('13b', 10.0), ('11b', 9.0), ('8b', 6.0),
            ('7b', 5.0), ('6b', 4.5), ('3b', 2.5),
            ('1.5b', 1.5), ('1b', 1.0)
        ]
        
        for pattern, ram in size_patterns:
            if pattern in name_lower:
                return ram
        
        # Model family defaults
        family_defaults = {
            'llama3': 6.0, 'llama2': 5.0, 'mistral': 5.0,
            'mixtral': 24.0, 'command': 24.0, 'qwen': 6.0,
            'codellama': 8.0, 'deepseek': 8.0, 'phi': 2.5,
            'gemma': 4.0, 'neural-chat': 5.0, 'starcode': 6.0
        }
        
        for family, default_ram in family_defaults.items():
            if family in name_lower:
                return default_ram
        
        return 4.0  # Conservative default
    
    def _estimate_model_vram_requirements(self, model_name: str) -> float:
        """Estimate VRAM requirements for GPU acceleration"""
        base_ram = self._estimate_model_ram_requirements(model_name)
        # VRAM typically needs less due to optimizations
        return base_ram * 0.75
    
    def _assess_model_compatibility(self, model_name: str, system_info: Dict,
                                   estimated_ram: float, estimated_vram: float) -> ModelCompatibility:
        """
        Assess model compatibility with comprehensive analysis
        
        Enhanced compatibility assessment with performance tiers.
        """
        reasons = []
        warnings = []
        recommendations = []
        compatible = True
        performance_tier = "excellent"
        
        available_ram = system_info["available_ram_gb"]
        total_ram = system_info["total_ram_gb"]
        has_gpu = system_info["has_gpu"]
        
        # RAM compatibility check
        if estimated_ram > available_ram:
            compatible = False
            performance_tier = "poor"
            reasons.append(f"Insufficient RAM: {estimated_ram:.1f}GB needed, {available_ram:.1f}GB available")
            
            if estimated_ram > total_ram:
                recommendations.append(f"Upgrade RAM to at least {estimated_ram * 1.2:.0f}GB")
            else:
                recommendations.append("Close applications to free up RAM")
        
        elif estimated_ram > available_ram * 0.9:
            performance_tier = "moderate"
            warnings.append(f"High RAM usage: {estimated_ram:.1f}GB needed, {available_ram:.1f}GB available")
            recommendations.append("Consider closing other applications")
        
        elif estimated_ram > available_ram * 0.7:
            performance_tier = "good"
            warnings.append("Moderate RAM usage expected")
        
        # GPU acceleration assessment
        if has_gpu:
            best_gpu = max(system_info["gpu_info"], key=lambda g: g["total_memory_gb"])
            gpu_memory = best_gpu["total_memory_gb"]
            
            if estimated_vram <= gpu_memory * self.config.gpu_memory_fraction:
                reasons.append(f"GPU acceleration available ({best_gpu['vendor']} {best_gpu['name']})")
                if performance_tier == "excellent":
                    performance_tier = "excellent"
            else:
                warnings.append(f"Limited GPU memory: {estimated_vram:.1f}GB needed, {gpu_memory:.1f}GB available")
                recommendations.append("Model may fall back to CPU processing")
                if performance_tier in ["excellent", "good"]:
                    performance_tier = "good"
        else:
            if self.config.enable_cpu_fallback:
                warnings.append("No GPU detected - CPU-only processing")
                reasons.append("CPU processing supported")
                if performance_tier == "excellent":
                    performance_tier = "good"
            else:
                compatible = False
                reasons.append("GPU required but not available")
                recommendations.append("Install GPU drivers or enable CPU fallback")
        
        # Performance predictions based on model size
        if estimated_ram <= 4:
            reasons.append("Fast inference expected")
        elif estimated_ram <= 8:
            reasons.append("Good inference speed expected")
        elif estimated_ram <= 16:
            warnings.append("Moderate inference speed expected")
        else:
            warnings.append("Slow inference expected due to large model size")
            if performance_tier not in ["poor"]:
                performance_tier = "moderate"
        
        # Memory threshold check
        if available_ram < self.config.memory_threshold_gb:
            warnings.append(f"System RAM below recommended threshold ({self.config.memory_threshold_gb}GB)")
            recommendations.append("Close unnecessary applications or upgrade RAM")
        
        return ModelCompatibility(
            model_name=model_name,
            compatible=compatible,
            estimated_ram_gb=estimated_ram,
            estimated_vram_gb=estimated_vram,
            performance_tier=performance_tier,
            reasons=reasons,
            warnings=warnings,
            recommendations=recommendations
        )


# Export main classes
__all__ = [
    "HardwareChecker",
    "CrossPlatformGPUDetector",
    "SystemInfo",
    "GPUInfo",
    "ModelCompatibility",
    "GPUVendor"
]
