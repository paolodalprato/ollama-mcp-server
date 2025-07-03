"""
Configuration Manager - Claude Ollama Bridge v0.9
Cross-platform configuration management with YAML/JSON support

Design Principles:
- Type safety with full annotations
- Immutable configuration
- Comprehensive error handling
- Clean separation of concerns
"""

import logging
import os
import yaml
import json
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

class ConfigFormat(Enum):
    """Supported configuration file formats"""
    YAML = "yaml"
    JSON = "json"
    AUTO = "auto"


@dataclass
class ServerConfig:
    """
    Ollama server configuration settings
    
    Contains all server-related configuration options with validation.
    """
    host: str = "localhost"
    port: int = 11434
    startup_timeout: int = 10
    shutdown_timeout: int = 5
    custom_executable_path: Optional[str] = None
    environment_vars: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate server configuration after creation"""
        self._validate_configuration()
    
    def _validate_configuration(self):
        """Validate server configuration values"""
        if not 1024 <= self.port <= 65535:
            raise ValueError(f"Invalid port: {self.port}. Must be between 1024-65535")
        
        if self.startup_timeout < 1:
            logger.warning(f"Invalid startup_timeout: {self.startup_timeout}. Using default: 10")
            self.startup_timeout = 10
            
        if self.shutdown_timeout < 1:
            logger.warning(f"Invalid shutdown_timeout: {self.shutdown_timeout}. Using default: 5")
            self.shutdown_timeout = 5
        
        if self.custom_executable_path:
            path = Path(self.custom_executable_path)
            if not path.exists():
                logger.warning(f"Custom executable path does not exist: {self.custom_executable_path}")
    
    @property
    def full_url(self) -> str:
        """Get full server URL"""
        return f"http://{self.host}:{self.port}"


@dataclass
class ModelConfig:
    """
    Model management configuration settings
    
    Contains settings for model downloads, caching, and recommendations.
    """
    auto_pull_updates: bool = False
    cache_directory: Optional[str] = None
    max_concurrent_downloads: int = 2
    download_timeout: int = 300
    preferred_models: List[str] = field(default_factory=list)
    model_size_limit_gb: Optional[float] = None
    
    def __post_init__(self):
        """Validate model configuration after creation"""
        self._validate_configuration()
    
    def _validate_configuration(self):
        """Validate model configuration values"""
        if self.max_concurrent_downloads < 1:
            logger.warning(f"Invalid max_concurrent_downloads: {self.max_concurrent_downloads}. Using default: 2")
            self.max_concurrent_downloads = 2
        
        if self.download_timeout < 30:
            logger.warning(f"Invalid download_timeout: {self.download_timeout}. Using minimum: 30")
            self.download_timeout = 30
        
        if self.model_size_limit_gb is not None and self.model_size_limit_gb <= 0:
            logger.warning(f"Invalid model_size_limit_gb: {self.model_size_limit_gb}. Disabling limit")
            self.model_size_limit_gb = None


@dataclass
class HardwareConfig:
    """
    Hardware detection and optimization settings
    
    Contains configuration for GPU detection and resource management.
    """
    enable_gpu_detection: bool = True
    gpu_memory_fraction: float = 0.8
    enable_cpu_fallback: bool = True
    memory_threshold_gb: float = 4.0
    enable_performance_monitoring: bool = True
    custom_gpu_command: Optional[str] = None
    
    def __post_init__(self):
        """Validate hardware configuration after creation"""
        self._validate_configuration()
    
    def _validate_configuration(self):
        """Validate hardware configuration values"""
        if not 0.1 <= self.gpu_memory_fraction <= 1.0:
            logger.warning(f"Invalid gpu_memory_fraction: {self.gpu_memory_fraction}. Using default: 0.8")
            self.gpu_memory_fraction = 0.8
        
        if self.memory_threshold_gb < 1.0:
            logger.warning(f"Invalid memory_threshold_gb: {self.memory_threshold_gb}. Using minimum: 1.0")
            self.memory_threshold_gb = 1.0


@dataclass
class LoggingConfig:
    """
    Logging configuration settings
    
    Contains all logging-related configuration options.
    """
    level: str = "INFO"
    enable_file_logging: bool = True
    log_file_path: Optional[str] = None
    max_log_size_mb: int = 10
    backup_count: int = 5
    enable_console_logging: bool = True
    
    def __post_init__(self):
        """Validate logging configuration after creation"""
        self._validate_configuration()
    
    def _validate_configuration(self):
        """Validate logging configuration values"""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.level.upper() not in valid_levels:
            logger.warning(f"Invalid log level: {self.level}. Using default: INFO")
            self.level = "INFO"
        
        if self.max_log_size_mb < 1:
            logger.warning(f"Invalid max_log_size_mb: {self.max_log_size_mb}. Using default: 10")
            self.max_log_size_mb = 10
        
        if self.backup_count < 1:
            logger.warning(f"Invalid backup_count: {self.backup_count}. Using default: 5")
            self.backup_count = 5


@dataclass
class OllamaMCPConfig:
    """
    Complete Ollama MCP Server configuration
    
    Contains all configuration sections with validation and defaults.
    """
    server: ServerConfig = field(default_factory=ServerConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def __post_init__(self):
        """Validate complete configuration after creation"""
        self._validate_configuration()
    
    def _validate_configuration(self):
        """Validate complete configuration"""
        # All validation is handled by individual config sections
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OllamaMCPConfig':
        """Create configuration from dictionary"""
        server_data = data.get('server', {})
        models_data = data.get('models', {})
        hardware_data = data.get('hardware', {})
        logging_data = data.get('logging', {})
        
        return cls(
            server=ServerConfig(**server_data),
            models=ModelConfig(**models_data),
            hardware=HardwareConfig(**hardware_data),
            logging=LoggingConfig(**logging_data)
        )


class ConfigurationManager:
    """
    Cross-platform configuration manager with multiple sources
    
    Provides comprehensive configuration management with priority hierarchy:
    1. Environment variables (highest priority)
    2. User config file (~/.ollama-mcp/config.yaml)
    3. System config file (/etc/ollama-mcp/config.yaml or equivalent)
    4. Package defaults (lowest priority)
    """
    
    # Environment variable prefix
    ENV_PREFIX = "OLLAMA_MCP_"
    
    # Default config file names
    CONFIG_FILES = ["config.yaml", "config.yml", "config.json"]
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration manager
        
        Args:
            config_dir: Custom configuration directory, uses defaults if None
        """
        self.config_dir = config_dir or self._get_default_config_dir()
        self._config: Optional[OllamaMCPConfig] = None
        
        logger.debug(f"Initialized ConfigurationManager with config_dir: {self.config_dir}")
    
    def _get_default_config_dir(self) -> Path:
        """
        Get default configuration directory for current platform
        
        Returns:
            Path to default configuration directory
        """
        system = os.name
        
        if system == "nt":  # Windows
            base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        else:  # Unix-like (Linux, macOS)
            base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        
        return base / "ollama-mcp"
    
    def _get_system_config_paths(self) -> List[Path]:
        """
        Get system-wide configuration paths for current platform
        
        Returns:
            List of potential system config paths
        """
        system = os.name
        paths = []
        
        if system == "nt":  # Windows
            # Windows system-wide config locations
            program_data = Path(os.environ.get("PROGRAMDATA", "C:/ProgramData"))
            paths.append(program_data / "ollama-mcp")
        else:  # Unix-like
            # Unix system-wide config locations
            paths.extend([
                Path("/etc/ollama-mcp"),
                Path("/usr/local/etc/ollama-mcp")
            ])
        
        return paths
    
    def _find_config_file(self, search_dirs: List[Path]) -> Optional[Path]:
        """
        Find configuration file in search directories
        
        Args:
            search_dirs: Directories to search for config files
            
        Returns:
            Path to first found config file or None
        """
        for directory in search_dirs:
            if not directory.exists():
                continue
                
            for filename in self.CONFIG_FILES:
                config_path = directory / filename
                if config_path.exists() and config_path.is_file():
                    logger.debug(f"Found config file: {config_path}")
                    return config_path
        
        return None
    
    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """
        Load configuration from file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            ValueError: If file format is unsupported or invalid
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Determine format from extension
            suffix = config_path.suffix.lower()
            
            if suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(content)
            elif suffix == '.json':
                data = json.loads(content)
            else:
                # Try to auto-detect format
                try:
                    data = yaml.safe_load(content)
                except yaml.YAMLError:
                    data = json.loads(content)
            
            return data or {}
            
        except Exception as e:
            logger.error(f"Error loading config file {config_path}: {e}")
            raise ValueError(f"Invalid config file format: {e}")
    
    def _load_environment_overrides(self) -> Dict[str, Any]:
        """
        Load configuration overrides from environment variables
        
        Returns:
            Dictionary with environment variable overrides
        """
        overrides = {}
        
        # Environment variable mapping
        env_mappings = {
            f"{self.ENV_PREFIX}SERVER_HOST": ("server", "host"),
            f"{self.ENV_PREFIX}SERVER_PORT": ("server", "port"),
            f"{self.ENV_PREFIX}SERVER_STARTUP_TIMEOUT": ("server", "startup_timeout"),
            f"{self.ENV_PREFIX}SERVER_SHUTDOWN_TIMEOUT": ("server", "shutdown_timeout"),
            f"{self.ENV_PREFIX}SERVER_EXECUTABLE_PATH": ("server", "custom_executable_path"),
            f"{self.ENV_PREFIX}MODELS_AUTO_PULL": ("models", "auto_pull_updates"),
            f"{self.ENV_PREFIX}MODELS_CACHE_DIR": ("models", "cache_directory"),
            f"{self.ENV_PREFIX}MODELS_MAX_DOWNLOADS": ("models", "max_concurrent_downloads"),
            f"{self.ENV_PREFIX}HARDWARE_ENABLE_GPU": ("hardware", "enable_gpu_detection"),
            f"{self.ENV_PREFIX}HARDWARE_GPU_MEMORY": ("hardware", "gpu_memory_fraction"),
            f"{self.ENV_PREFIX}LOG_LEVEL": ("logging", "level"),
            f"{self.ENV_PREFIX}LOG_ENABLE_FILE": ("logging", "enable_file_logging"),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                # Type conversion
                if key in ["port", "startup_timeout", "shutdown_timeout", "max_concurrent_downloads"]:
                    try:
                        value = int(value)
                    except ValueError:
                        logger.warning(f"Invalid integer value for {env_var}: {value}")
                        continue
                elif key in ["gpu_memory_fraction"]:
                    try:
                        value = float(value)
                    except ValueError:
                        logger.warning(f"Invalid float value for {env_var}: {value}")
                        continue
                elif key in ["auto_pull_updates", "enable_gpu_detection", "enable_file_logging"]:
                    value = value.lower() in ("true", "1", "yes", "on")
                
                # Set nested value
                if section not in overrides:
                    overrides[section] = {}
                overrides[section][key] = value
        
        return overrides
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge configuration dictionaries
        
        Args:
            base: Base configuration
            override: Override configuration
            
        Returns:
            Merged configuration
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def load_configuration(self) -> OllamaMCPConfig:
        """
        Load complete configuration from all sources
        
        Returns:
            Complete validated configuration
        """
        # Start with defaults
        config_data = {}
        
        # Load system config
        system_paths = self._get_system_config_paths()
        system_config_file = self._find_config_file(system_paths)
        if system_config_file:
            try:
                system_config = self._load_config_file(system_config_file)
                config_data = self._merge_configs(config_data, system_config)
                logger.info(f"Loaded system config from: {system_config_file}")
            except Exception as e:
                logger.warning(f"Failed to load system config: {e}")
        
        # Load user config
        user_config_file = self._find_config_file([self.config_dir])
        if user_config_file:
            try:
                user_config = self._load_config_file(user_config_file)
                config_data = self._merge_configs(config_data, user_config)
                logger.info(f"Loaded user config from: {user_config_file}")
            except Exception as e:
                logger.warning(f"Failed to load user config: {e}")
        
        # Apply environment overrides
        env_overrides = self._load_environment_overrides()
        if env_overrides:
            config_data = self._merge_configs(config_data, env_overrides)
            logger.debug(f"Applied environment overrides: {list(env_overrides.keys())}")
        
        # Create configuration object
        self._config = OllamaMCPConfig.from_dict(config_data)
        
        logger.info("Configuration loaded successfully")
        return self._config
    
    def save_configuration(self, config: OllamaMCPConfig, format: ConfigFormat = ConfigFormat.YAML) -> Path:
        """
        Save configuration to user config file
        
        Args:
            config: Configuration to save
            format: File format to use
            
        Returns:
            Path where configuration was saved
        """
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine filename based on format
        if format == ConfigFormat.YAML:
            filename = "config.yaml"
        elif format == ConfigFormat.JSON:
            filename = "config.json"
        else:  # AUTO
            filename = "config.yaml"  # Default to YAML
        
        config_path = self.config_dir / filename
        config_data = config.to_dict()
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                if format == ConfigFormat.JSON:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                else:  # YAML
                    yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"Configuration saved to: {config_path}")
            return config_path
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def get_config(self) -> OllamaMCPConfig:
        """
        Get current configuration, loading if necessary
        
        Returns:
            Current configuration
        """
        if self._config is None:
            self._config = self.load_configuration()
        return self._config
    
    def create_default_config_file(self) -> Path:
        """
        Create default configuration file with documentation
        
        Returns:
            Path to created config file
        """
        default_config = OllamaMCPConfig()
        
        # Add documentation comments for YAML
        config_path = self.config_dir / "config.yaml"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        config_content = f"""# Ollama MCP Server Configuration
# Auto-generated default configuration file
# Edit this file to customize your Ollama MCP server settings

# Server configuration
server:
  host: "{default_config.server.host}"                    # Server host (localhost, 0.0.0.0, etc.)
  port: {default_config.server.port}                                      # Server port (1024-65535)
  startup_timeout: {default_config.server.startup_timeout}                                # Seconds to wait for server startup
  shutdown_timeout: {default_config.server.shutdown_timeout}                               # Seconds to wait for graceful shutdown
  custom_executable_path: null                           # Custom path to Ollama executable
  environment_vars: {{}}                                  # Custom environment variables

# Model management configuration
models:
  auto_pull_updates: {str(default_config.models.auto_pull_updates).lower()}                            # Automatically check for model updates
  cache_directory: null                                  # Custom model cache directory
  max_concurrent_downloads: {default_config.models.max_concurrent_downloads}                         # Maximum parallel downloads
  download_timeout: {default_config.models.download_timeout}                               # Download timeout in seconds
  preferred_models: []                                   # List of preferred models
  model_size_limit_gb: null                             # Maximum model size in GB

# Hardware detection and optimization
hardware:
  enable_gpu_detection: {str(default_config.hardware.enable_gpu_detection).lower()}                        # Enable GPU detection
  gpu_memory_fraction: {default_config.hardware.gpu_memory_fraction}                           # Fraction of GPU memory to use
  enable_cpu_fallback: {str(default_config.hardware.enable_cpu_fallback).lower()}                         # Enable CPU fallback
  memory_threshold_gb: {default_config.hardware.memory_threshold_gb}                           # Minimum memory threshold
  enable_performance_monitoring: {str(default_config.hardware.enable_performance_monitoring).lower()}              # Enable performance monitoring
  custom_gpu_command: null                              # Custom GPU detection command

# Logging configuration
logging:
  level: "{default_config.logging.level}"                               # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  enable_file_logging: {str(default_config.logging.enable_file_logging).lower()}                       # Enable file logging
  log_file_path: null                                   # Custom log file path
  max_log_size_mb: {default_config.logging.max_log_size_mb}                                  # Maximum log file size in MB
  backup_count: {default_config.logging.backup_count}                                     # Number of log backups to keep
  enable_console_logging: {str(default_config.logging.enable_console_logging).lower()}                   # Enable console logging
"""
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        logger.info(f"Created default configuration file: {config_path}")
        return config_path


# Global configuration instance
_config_manager: Optional[ConfigurationManager] = None

def get_config_manager(config_dir: Optional[Path] = None) -> ConfigurationManager:
    """
    Get global configuration manager instance
    
    Args:
        config_dir: Custom configuration directory
        
    Returns:
        Configuration manager instance
    """
    global _config_manager
    
    if _config_manager is None or config_dir is not None:
        _config_manager = ConfigurationManager(config_dir)
    
    return _config_manager

def get_config() -> OllamaMCPConfig:
    """
    Get current configuration
    
    Returns:
        Current configuration
    """
    return get_config_manager().get_config()


# Export main classes
__all__ = [
    "OllamaMCPConfig",
    "ServerConfig",
    "ModelConfig", 
    "HardwareConfig",
    "LoggingConfig",
    "ConfigurationManager",
    "ConfigFormat",
    "get_config_manager",
    "get_config"
]
