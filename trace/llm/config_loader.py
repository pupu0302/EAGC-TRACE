#!/usr/bin/env python3
"""
Configuration loading utility
Supports:
1. Multi-layer config merging (models.yaml + models.local.yaml)
2. Environment variable injection
3. Model validation
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Configuration loader"""

    def __init__(self, config_dir: str = None):
        """
        Initialize the configuration loader

        Args:
            config_dir: configuration directory path
                - If None, automatically locates configs/ under the project root.
                - If a relative path is given, it is resolved relative to the project root.
                - If an absolute path is given, it is used as-is.
        """
        # ============================================================
        # Auto-locate the project root directory
        # ============================================================
        if config_dir is None:
            # Get the directory containing this file (scripts/llm/)
            current_file = Path(__file__).resolve()
            # Go up two levels to reach the project root (E_DAG/)
            project_root = current_file.parent.parent.parent
            config_dir = project_root / "configs"
        else:
            config_dir = Path(config_dir)
            # If a relative path is given, resolve it against the project root
            if not config_dir.is_absolute():
                current_file = Path(__file__).resolve()
                project_root = current_file.parent.parent.parent
                config_dir = project_root / config_dir

        self.config_dir = config_dir
        self.config = {}

        # ============================================================
        # Debug info
        # ============================================================
        logger.info(f"ConfigLoader initialized with config_dir: {self.config_dir}")
        logger.info(f"Config directory exists: {self.config_dir.exists()}")
        # ============================================================

        self._load_configs()

    def _load_configs(self):
        """Load and merge configuration files"""
        # 1. Load the main configuration
        main_config_path = self.config_dir / "models.yaml"

        # ============================================================
        # Detailed error reporting
        # ============================================================
        if main_config_path.exists():
            try:
                with open(main_config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f) or {}
                logger.info(f"✓ Loaded main config from {main_config_path}")
                logger.info(f"  - Found {len(self.config.get('models', {}))} models")
                logger.info(f"  - Found {len(self.config.get('module_configs', {}))} module configs")
            except Exception as e:
                logger.error(f"✗ Failed to load {main_config_path}: {e}")
                self.config = {}
        else:
            logger.warning(f"✗ Main config not found: {main_config_path}")
            logger.warning(f"  - Searched in: {self.config_dir}")
            logger.warning(f"  - Working directory: {Path.cwd()}")
            self.config = {}
        # ============================================================

        # 2. Load local override configuration
        local_config_path = self.config_dir / "models.local.yaml"
        if local_config_path.exists():
            try:
                with open(local_config_path, 'r', encoding='utf-8') as f:
                    local_config = yaml.safe_load(f) or {}
                self._deep_merge(self.config, local_config)
                logger.info(f"✓ Merged local config from {local_config_path}")
            except Exception as e:
                logger.error(f"✗ Failed to load local config: {e}")

    def _deep_merge(self, base: Dict, override: Dict):
        """Deep-merge two dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def get_model_config(self, model_key: str) -> Dict[str, Any]:
        """
        Get the configuration for the given model

        Args:
            model_key: model key name (e.g. "gpt4o", "claude35_sonnet")

        Returns:
            Model configuration dict
        """
        models = self.config.get("models", {})

        # ============================================================
        # Detailed error reporting
        # ============================================================
        if not models:
            logger.error("No models found in config! Check if models.yaml was loaded correctly.")
            raise ValueError("No models configured. Please create configs/models.yaml")

        if model_key not in models:
            available = list(models.keys())
            logger.error(f"Model '{model_key}' not found. Available models: {available}")
            raise ValueError(f"Model '{model_key}' not found in config. Available: {available}")
        # ============================================================

        model_config = models[model_key].copy()

        # Inject the API key
        api_key_env = model_config.get("api_key_env")
        if api_key_env:
            api_key = os.getenv(api_key_env)
            if not api_key:
                logger.warning(f"API Key '{api_key_env}' not found in environment. "
                               f"Model '{model_key}' may not work.")
                # Do not raise; allow falling back to mock mode
            else:
                model_config["api_key"] = api_key
                logger.debug(f"✓ API Key loaded for {model_key}")

        # Check whether the model is enabled
        if not model_config.get("enabled", True):
            logger.warning(f"Model '{model_key}' is disabled in config")

        return model_config

    def get_module_config(self, module_name: str) -> Dict[str, Any]:
        """
        Get the module configuration (including which model it uses)

        Args:
            module_name: module name (e.g. "module_g", "module_sr_l")

        Returns:
            Module configuration dict (including the full model configuration)
        """
        module_configs = self.config.get("module_configs", {})

        # ============================================================
        # Detailed logging
        # ============================================================
        if module_name not in module_configs:
            default_model = self.config.get("default_model", "mock")
            logger.warning(f"Module '{module_name}' not configured in module_configs")
            logger.warning(f"Using default model: {default_model}")
            return self.get_model_config(default_model)
        # ============================================================

        module_config = module_configs[module_name].copy()
        model_key = module_config.pop("model", self.config.get("default_model", "mock"))

        logger.info(f"Loading config for {module_name}: model={model_key}")

        # Merge the model configuration
        model_config = self.get_model_config(model_key)
        model_config.update(module_config)  # module-level config overrides model-level config

        return model_config

    def list_available_models(self) -> list:
        """List all available models"""
        models = self.config.get("models", {})
        return [k for k, v in models.items() if v.get("enabled", True)]


# ==========================================
# Global singleton
# ==========================================
_config_loader = None


def get_config_loader(config_dir: str = None) -> ConfigLoader:
    """Get the global configuration loader (singleton pattern)"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_dir)
    return _config_loader
