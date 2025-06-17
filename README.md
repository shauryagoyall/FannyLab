# Rat Decision Making Model Fitting

This repository contains tools for fitting and comparing behavioral models to rat decision-making data.

## Project Structure

```
├── data/                  # Data directory (place your data files here)
│   └── raw/              # Raw data files
│   └── processed/        # Processed data files
├── models/               # Model definitions
│   ├── base_model.py    # Base model class
│   ├── q_learning.py    # Q-learning implementation
│   └── model_factory.py # Model creation and configuration
├── utils/               # Utility functions
│   ├── data_loader.py   # Data loading and preprocessing
│   └── model_comparison.py # Model comparison utilities
├── results/             # Results and figures
└── run_analysis.py      # Main analysis script
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create required directories:
```bash
mkdir -p data/raw data/processed results
```

## Usage

1. Place your rat behavioral data in the `data/raw/` directory
2. Run the analysis script:
```bash
python run_analysis.py --data-file your_data.csv --subject-id SUBJECT_ID --session SESSION_ID
```

The script will:
- Load and preprocess your data
- Fit the Q-learning model
- Generate results and visualizations in the `results/` directory

## Data Format

Your data file should be a CSV file containing the following columns:
- `subject_id`: Identifier for each rat
- `session`: Session identifier
- `trial`: Trial number
- `choice`: Rat's choice (0 or 1)
- `reward`: Reward received (0 or 1)
- `correct_choice`: The correct choice for the trial (0 or 1)

## Model Comparison

The repository includes tools to:
- Fit behavioral models to your data
- Calculate BIC (Bayesian Information Criterion) for model comparison
- Visualize model fits and comparisons
- Perform statistical tests between models

## Contributing

Feel free to add new models or improve existing ones by following the established structure. 