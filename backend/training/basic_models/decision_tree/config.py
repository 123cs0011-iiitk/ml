"""
Decision Tree Model Configuration

Configuration settings and hyperparameters for Decision Tree model.
"""

from typing import Dict, Any


class DecisionTreeConfig:
    """Configuration for Decision Tree model training."""
    
    # Model Information
    MODEL_NAME = 'decision_tree'
    MODEL_DISPLAY_NAME = 'Decision Tree'
    MODEL_CLASS = 'DecisionTreeModel'
    MODEL_FILE = 'decision_tree.py'
    
    # Training Parameters
    EXPECTED_TRAINING_MINUTES = 3
    STATUS_MESSAGE = 'Training decision tree with splits'
    
    # Model is not verbose
    VERBOSE = False
    
    # Hyperparameters
    MAX_DEPTH = 10
    MIN_SAMPLES_SPLIT = 10
    MIN_SAMPLES_LEAF = 5
    MAX_FEATURES = 'sqrt'
    RANDOM_STATE = 42
    
    @classmethod
    def get_model_params(cls) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return {
            'max_depth': cls.MAX_DEPTH,
            'min_samples_split': cls.MIN_SAMPLES_SPLIT,
            'min_samples_leaf': cls.MIN_SAMPLES_LEAF,
            'max_features': cls.MAX_FEATURES,
            'random_state': cls.RANDOM_STATE
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

