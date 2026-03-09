"""
Configuration Manager for Gesture2Speech Project
Handles loading and accessing configuration from YAML file
"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict
from loguru import logger


class ConfigManager:
    """Singleton class to manage project configuration"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self.load_config()
    
    def load_config(self, config_path: str = None):
        """Load configuration from YAML file"""
        if config_path is None:
            # Get project root directory
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yaml"
        
        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        Example: config.get('model.lstm.hidden_size')
        """
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            logger.warning(f"Configuration key not found: {key_path}")
            return None
    
    def get_all(self) -> Dict:
        """Get entire configuration dictionary"""
        return self._config
    
    def get_paths(self) -> Dict[str, Path]:
        """Get all configured paths as Path objects"""
        project_root = Path(__file__).parent.parent.parent
        paths = self.get('paths', {})
        
        return {
            key: project_root / path 
            for key, path in paths.items()
        }
    
    def create_directories(self):
        """Create all configured directories if they don't exist"""
        paths = self.get_paths()
        
        for name, path in paths.items():
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {path}")
        
        logger.info("All configured directories created/verified")
    
    def get_label_mapping(self) -> Dict[str, int]:
        """Generate label to index mapping from categories"""
        categories = self.get('dataset.categories', {})
        label_mapping = {}
        idx = 0
        
        for category, labels in categories.items():
            for label in labels:
                label_mapping[label] = idx
                idx += 1
        
        return label_mapping
    
    def get_index_to_label(self) -> Dict[int, str]:
        """Generate index to label mapping"""
        label_mapping = self.get_label_mapping()
        return {v: k for k, v in label_mapping.items()}
    
    def get_num_classes(self) -> int:
        """Get total number of classes"""
        return len(self.get_label_mapping())


# Global configuration instance
config = ConfigManager()


if __name__ == "__main__":
    # Test configuration manager
    config = ConfigManager()
    
    print("Project Name:", config.get('project.name'))
    print("Model Architecture:", config.get('model.architecture'))
    print("Number of Classes:", config.get_num_classes())
    print("Label Mapping:", config.get_label_mapping())
    print("\nAll Paths:")
    for name, path in config.get_paths().items():
        print(f"  {name}: {path}")

# Made with Bob
