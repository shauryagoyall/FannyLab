import pandas as pd
import numpy as np
from utils.model_comparison import compare_models, plot_model_comparison, calculate_bic_weights
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import json
import os

def fit_models(models: List[Any], data: pd.DataFrame) -> Dict[str, Any]:
    """
    Fit all models to the data and collect results.
    """
    results = []
    
    for model in models:
        print(f"\nFitting {model.name}...")
        
        # Fit model with parameter optimization
        model.fit(data, optimize=True)
        
        # Calculate metrics
        bic = model.calculate_bic(data)
        n_params = len(model.get_parameters())
        
        # Store results
        model_results = {
            'model_name': model.name,
            'parameters': model.get_parameters(),
            'bic': bic,
            'n_parameters': n_params
        }
        
        # Add model-specific results
        if hasattr(model, 'q_values'):
            model_results['q_values'] = model.q_values.tolist()
        
        results.append(model_results)
        
        print(f"BIC: {bic:.2f}")
        print("Optimized Parameters:", model.get_parameters())
    
    return results

def save_results(results: Dict[str, Any], output_dir: str):
    """
    Save fitting results and create visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save numerical results
    with open(os.path.join(output_dir, 'fitting_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create comparison plot
    results_df = pd.DataFrame([{
        'Model': r['model_name'],
        'BIC': r['bic'],
        'Parameters': r['n_parameters']
    } for r in results])
    
    results_df = calculate_bic_weights(results_df)
    plot_model_comparison(results_df, os.path.join(output_dir, 'model_comparison.png'))
    
    # Plot learning curves for Q-learning models
    for result in results:
        if 'q_values' in result:
            model_name = result['model_name']
            q_values = np.array(result['q_values'])
            
            plt.figure(figsize=(10, 4))
            plt.plot(q_values[:, 0], label='Q-value Action 1')
            plt.plot(q_values[:, 1], label='Q-value Action 2')
            plt.title(f'Q-values over time - {model_name}')
            plt.xlabel('Trial')
            plt.ylabel('Q-value')
            plt.legend()
            plt.savefig(os.path.join(output_dir, f'q_values_{model_name}.png'))
            plt.close()

def main():
    # Configuration
    data_path = "path/to/your/data.csv"  # TODO: Set your data path
    output_dir = "results"
    
    # Load data
    print("Loading data...")
    data = load_data(data_path)
    
    # Create models
    print("Creating models...")
    models = create_models()
    
    # Fit models
    print("Fitting models...")
    results = fit_models(models, data)
    
    # Save results
    print("Saving results...")
    save_results(results, output_dir)
    
    print("\nDone! Results saved in:", output_dir)

if __name__ == "__main__":
    main() 