import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from models.base_model import BaseModel

def compare_models(models: List[BaseModel], data: pd.DataFrame) -> pd.DataFrame:
    """
    Compare multiple models using BIC and other metrics.
    
    Args:
        models: List of fitted models to compare
        data: DataFrame containing the behavioral data
        
    Returns:
        DataFrame with comparison metrics
    """
    results = []
    
    for model in models:
        if not model.fitted:
            raise ValueError(f"Model {model.name} must be fitted before comparison")
            
        bic = model.calculate_bic(data)
        n_params = len(model.get_parameters())
        
        results.append({
            'Model': model.name,
            'BIC': bic,
            'Parameters': n_params,
            'Parameters': model.get_parameters()
        })
    
    return pd.DataFrame(results)

def plot_model_comparison(results: pd.DataFrame, save_path: str = None):
    """
    Create visualization of model comparison results.
    
    Args:
        results: DataFrame from compare_models
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    # Plot BIC values
    sns.barplot(data=results, x='Model', y='BIC')
    plt.xticks(rotation=45)
    plt.title('Model Comparison (BIC)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def calculate_bic_weights(results: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate BIC weights for model comparison.
    
    Args:
        results: DataFrame from compare_models
        
    Returns:
        DataFrame with BIC weights added
    """
    # Calculate delta BIC
    min_bic = results['BIC'].min()
    results['Delta_BIC'] = results['BIC'] - min_bic
    
    # Calculate BIC weights
    results['BIC_Weight'] = np.exp(-0.5 * results['Delta_BIC'])
    results['BIC_Weight'] = results['BIC_Weight'] / results['BIC_Weight'].sum()
    
    return results 