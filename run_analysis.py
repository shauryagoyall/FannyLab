import os
import argparse
import json
from fit_models import fit_models, save_results
from models.model_factory import ModelFactory
from utils.data_loader import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='Run behavioral model fitting analysis')
    
    # Required arguments
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the data file')
    
    # Model selection
    parser.add_argument('--model_type', type=str, required=True,
                      choices=ModelFactory.get_available_models(),
                      help='Type of model to fit')
    
    # Optional arguments
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Directory to save results (default: results)')
    parser.add_argument('--mouse_id', type=str, default=None,
                      help='Mouse ID to analyze (default: None)')
    parser.add_argument('--date', type=str, default=None,
                      help='Date to analyze (default: None)')
    
    # Model parameters (optional)
    parser.add_argument('--params', type=str,
                      help='JSON string of specific parameter values to use')
    
    # Add help for available parameters
    parser.add_argument('--list_params', action='store_true',
                      help='List available parameters for the selected model')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # List parameters if requested
    if args.list_params:
        params = ModelFactory.get_model_params(args.model_type)
        print(f"\nAvailable parameters for {args.model_type}:")
        for param in params:
            print(f"- {param}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and preprocess data
    print(f"\nLoading data from {args.data_path}...")
    data_loader = DataLoader(args.data_path)
    data = data_loader.preprocess_data(
        mouse_id=args.mouse_id if args.mouse_id != 'all' else None,
        date=args.date
    )
    
    # Print data information
    data_info = data_loader.get_data_info()
    print("\nData information:")
    for key, value in data_info.items():
        print(f"{key}: {value}")
    
    # Parse specific parameters if provided
    specific_params = None
    if args.params:
        try:
            specific_params = json.loads(args.params)
            print("\nUsing specific parameters:", specific_params)
        except json.JSONDecodeError:
            print("Error: Invalid JSON in --params argument")
            return
    
    # Create models
    print(f"\nCreating {args.model_type} models...")
    models = ModelFactory.create_models(args.model_type, specific_params)
    print(f"Created {len(models)} model instances")
    
    # Fit models
    print("\nFitting models...")
    results = fit_models(models, data)
    
    # Save results
    print("\nSaving results...")
    save_results(results, args.output_dir)
    
    # Save processed data
    data_loader.save_processed_data(os.path.join(args.output_dir, 'processed_data.csv'))
    
    print(f"\nAnalysis complete! Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main() 