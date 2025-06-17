import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

class DataLoader:
    """Class for loading and preprocessing behavioral data."""
    
    def __init__(self, data_path: str):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to the data file
        """
        self.data_path = Path(data_path)
        self.raw_data = None
        self.processed_data = None
    
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw data from file.
        Override this method with your specific data loading code.
        """
        # TODO: Replace with your actual data loading code
        # Example:
        # if self.data_path.suffix == '.csv':
        #     return pd.read_csv(self.data_path)
        # elif self.data_path.suffix == '.mat':
        #     return self._load_matlab_data()
        
        # Placeholder for testing
        return pd.DataFrame({
            'choice': np.random.binomial(1, 0.5, 1000),
            'reward': np.random.binomial(1, 0.7, 1000),
            'rt': np.random.normal(0.5, 0.1, 1000)
        })
    
    def preprocess_data(self, 
                       subject: Optional[str] = None,
                       session: Optional[str] = None,
                       **kwargs) -> pd.DataFrame:
        """
        Preprocess the raw data.
        
        Args:
            subject: Optional subject ID to filter
            session: Optional session ID to filter
            **kwargs: Additional preprocessing parameters
        
        Returns:
            Preprocessed DataFrame with required columns
        """
        if self.raw_data is None:
            self.raw_data = self.load_raw_data()
        
        data = self.raw_data.copy()
        
        # Filter by subject if specified
        if subject is not None:
            data = data[data['subject'] == subject]
        
        # Filter by session if specified
        if session is not None:
            data = data[data['session'] == session]
        
        # Add your preprocessing steps here
        # Example:
        # - Remove trials with missing data
        # - Calculate additional variables
        # - Normalize variables
        # - Add trial numbers
        # - Add block information
        # - etc.
        
        # Ensure required columns exist
        required_columns = ['choice', 'reward']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        self.processed_data = data
        return data
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded data.
        
        Returns:
            Dictionary containing data information
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Call preprocess_data first.")
        
        info = {
            'n_trials': len(self.processed_data),
            'columns': list(self.processed_data.columns),
            'subjects': list(self.processed_data['subject'].unique()) if 'subject' in self.processed_data.columns else None,
            'sessions': list(self.processed_data['session'].unique()) if 'session' in self.processed_data.columns else None
        }
        
        return info
    
    def save_processed_data(self, output_path: str):
        """
        Save processed data to file.
        
        Args:
            output_path: Path to save the processed data
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Call preprocess_data first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.processed_data.to_csv(output_path, index=False)
        print(f"Processed data saved to: {output_path}")

def load_data(data_path: str, 
             subject: Optional[str] = None,
             session: Optional[str] = None,
             **kwargs) -> pd.DataFrame:
    """
    Convenience function to load and preprocess data in one step.
    
    Args:
        data_path: Path to the data file
        subject: Optional subject ID to filter
        session: Optional session ID to filter
        **kwargs: Additional preprocessing parameters
    
    Returns:
        Preprocessed DataFrame
    """
    loader = DataLoader(data_path)
    return loader.preprocess_data(subject=subject, session=session, **kwargs) 