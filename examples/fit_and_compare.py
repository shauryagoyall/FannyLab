import pandas as pd
import numpy as np
from models.example_models import SimpleRLModel, DriftDiffusionModel
from utils.model_comparison import compare_models, plot_model_comparison, calculate_bic_weights

def main():
    # Load your data
    # data = pd.read_csv('path_to_your_data.csv')
    
    # For demonstration, create some synthetic data
    np.random.seed(42)
    n_trials = 1000
    data = pd.DataFrame({
        'choice': np.random.binomial(1, 0.7, n_trials),
        'reward': np.random.binomial(1, 0.8, n_trials),
        'rt': np.random.normal(0.5, 0.1, n_trials)
    })
    
    # Initialize models
    models = [
        SimpleRLModel(learning_rate=0.1),
        DriftDiffusionModel(drift=0.1, threshold=1.0, noise=0.1)
    ]
    
    # Fit models
    for model in models:
        model.fit(data)
    
    # Compare models
    results = compare_models(models, data)
    
    # Calculate BIC weights
    results = calculate_bic_weights(results)
    
    # Print results
    print("\nModel Comparison Results:")
    print(results[['Model', 'BIC', 'Delta_BIC', 'BIC_Weight']])
    
    # Plot results
    plot_model_comparison(results, 'results/model_comparison.png')

if __name__ == "__main__":
    main() 