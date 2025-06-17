from typing import List, Dict, Any
import numpy as np
from .q_learning import QLearningModel

class ModelConfig:
    """Configuration for model initialization."""
    
    # Default parameter ranges for different models
    PARAM_RANGES = {
        'qlearning': {
            'learning_rate': np.linspace(0.1, 0.5, 3),  # 3 values
            'temperature': np.linspace(0.5, 2.0, 3),    # 3 values
            'initial_value': [0.5]                      # 1 value
        }
        # Add other models here
        # 'ddm': {
        #     'drift': np.linspace(0.1, 0.5, 3),
        #     'threshold': np.linspace(1.0, 2.0, 3),
        #     'noise': np.linspace(0.1, 0.3, 3)
        # }
    }
    
    # Model classes
    MODEL_CLASSES = {
        'qlearning': QLearningModel,
        # 'ddm': DriftDiffusionModel
    }
    
    @classmethod
    def get_model_class(cls, model_type: str):
        """Get the model class for a given type."""
        if model_type not in cls.MODEL_CLASSES:
            raise ValueError(f"Unknown model type: {model_type}")
        return cls.MODEL_CLASSES[model_type]
    
    @classmethod
    def get_param_ranges(cls, model_type: str) -> Dict[str, List[float]]:
        """Get parameter ranges for a given model type."""
        if model_type not in cls.PARAM_RANGES:
            raise ValueError(f"No parameter ranges defined for model type: {model_type}")
        return cls.PARAM_RANGES[model_type]
    
    @classmethod
    def generate_param_combinations(cls, model_type: str) -> List[Dict[str, float]]:
        """Generate all combinations of parameters for a model type."""
        param_ranges = cls.get_param_ranges(model_type)
        
        # Create a list of parameter values for each parameter
        param_values = [param_ranges[param] for param in param_ranges.keys()]
        
        # Generate all combinations
        from itertools import product
        param_names = list(param_ranges.keys())
        combinations = []
        
        for values in product(*param_values):
            param_dict = dict(zip(param_names, values))
            combinations.append(param_dict)
        
        return combinations

class ModelFactory:
    """Factory for creating model instances."""
    
    @staticmethod
    def create_models(model_type: str, 
                     specific_params: Dict[str, float] = None) -> List[Any]:
        """
        Create model instances with systematic parameter initialization.
        
        Args:
            model_type: Type of model to create (e.g., 'qlearning')
            specific_params: Optional specific parameter values to use
                           (if None, uses all combinations from config)
        
        Returns:
            List of model instances
        """
        model_class = ModelConfig.get_model_class(model_type)
        
        if specific_params:
            # Create single model with specific parameters
            return [model_class(**specific_params)]
        else:
            # Create models with all parameter combinations
            param_combinations = ModelConfig.generate_param_combinations(model_type)
            return [model_class(**params) for params in param_combinations]
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available model types."""
        return list(ModelConfig.MODEL_CLASSES.keys())
    
    @staticmethod
    def get_model_params(model_type: str) -> List[str]:
        """Get list of parameters for a model type."""
        return list(ModelConfig.get_param_ranges(model_type).keys()) 