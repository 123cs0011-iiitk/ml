"""
SVM Model Configuration

Configuration settings and hyperparameters for Support Vector Machine model.
"""

from typing import Dict, Any


class SVMConfig:
    """Configuration for SVM model training."""
    
    # Model Information
    MODEL_NAME = 'svm'
    MODEL_DISPLAY_NAME = 'Support Vector Machine (SVM)'
    MODEL_CLASS = 'SVMModel'
    MODEL_FILE = 'svm.py'
    
    # Training Parameters
    EXPECTED_TRAINING_MINUTES = 15
    STATUS_MESSAGE = 'Training SVM with kernel optimization'
    
    # Model supports verbose output
    VERBOSE = True
    
    # Hyperparameters
    KERNEL = 'rbf'
    C = 1.0
    EPSILON = 0.1
    GAMMA = 'scale'
    MAX_ITER = 1000
    CACHE_SIZE = 500  # MB
    
    @classmethod
    def get_model_params(cls) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return {
            'kernel': cls.KERNEL,
            'C': cls.C,
            'epsilon': cls.EPSILON,
            'gamma': cls.GAMMA,
            'max_iter': cls.MAX_ITER,
            'cache_size': cls.CACHE_SIZE,
            'verbose': cls.VERBOSE
        }
    
    @classmethod
    def get_training_info(cls) -> Dict[str, Any]:
        """Get training information for display and logging."""
        return {
            'model_name': cls.MODEL_NAME,
            'display_name': cls.MODEL_DISPLAY_NAME,
            'expected_minutes': cls.EXPECTED_TRAINING_MINUTES,
            'status_message': cls.STATUS_MESSAGE,
            'verbose': cls.VERBOSE
        }

