"""
Configuration Manager - Easy switching between modes
"""
from .base_config import BaseConfig


def get_config(use_forecast=True, mode='train'):
    """
    Get configuration based on mode
    
    Args:
        use_forecast: True for state_dim=15, False for state_dim=14
        mode: 'train' or 'eval'
        
    Returns:
        config: Configuration instance
    """
    config = BaseConfig()
    config.USE_FORECAST = use_forecast
    
    # Evaluation-specific settings
    if mode == 'eval':
        config.CURRICULUM_ENABLED = False  # Use final bands
    
    return config


def get_train_config(use_forecast=True):
    """Get training configuration"""
    return get_config(use_forecast=use_forecast, mode='train')


def get_eval_config(use_forecast=True):
    """Get evaluation configuration"""
    return get_config(use_forecast=use_forecast, mode='eval')
