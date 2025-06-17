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
        """
        return pd.read_feather(str(self.data_path))
        
        # Placeholder for testing
        # return pd.DataFrame({
        #     'choice': np.random.binomial(1, 0.5, 1000),
        #     'reward': np.random.binomial(1, 0.7, 1000),
        #     'rt': np.random.normal(0.5, 0.1, 1000)
        # })
    
    def preprocess_data(self, 
                       mouse_id: Optional[str] = None,
                       date: Optional[str] = None,
                       **kwargs) -> pd.DataFrame:
        """
        Preprocess the raw data.
        
        Args:
            mouse_id: Optional mouse_id ID to filter
            date: Optional session ID to filter
            **kwargs: Additional preprocessing parameters
        
        Returns:
            Preprocessed DataFrame with required columns
        """
        if self.raw_data is None:
            self.raw_data = self.load_raw_data()
        
        data = self.raw_data.copy()
        
        # Filter by mouse_id if specified
        if mouse_id is not None:
            data = data[data['Mouse id'] == mouse_id]

        # Filter by session if specified
        if date is not None:
            data = data[data['Days'] == date]
        
        # Add preprocessing steps here
        # -- Create a new dataset only with the variables of interest
        #--have the choice variable and reward variable
        #-- use ludovica's parameters for reward and reward after switch and incorrect switch etc to get these columns so that it can be fit properly
        
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
            'mouse_ids': list(self.processed_data['mouse_id'].unique()) if 'mouse_id' in self.processed_data.columns else None,
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
             mouse_id: Optional[str] = None,
             session: Optional[str] = None,
             **kwargs) -> pd.DataFrame:
    """
    Convenience function to load and preprocess data in one step.
    
    Args:
        data_path: Path to the data file
        mouse_id: Optional mouse_id ID to filter
        session: Optional session ID to filter
        **kwargs: Additional preprocessing parameters
    
    Returns:
        Preprocessed DataFrame
    """
    loader = DataLoader(data_path)
    return loader.preprocess_data(mouse_id=mouse_id, session=session, **kwargs)

if __name__ == "__main__":
    # Test the DataLoader functionality
    import os
    
    # Test 1: Basic initialization and data loading
    print("\nTest 1: Basic initialization and data loading")
    try:
        # Replace this path with your actual data path
        test_data_path = "C:\\Users\\shaur\\Desktop\\FannyLab\\data\\raw\\AllData_fused_bySession_WorkingVer.feather"
        loader = DataLoader(test_data_path)
        raw_data = loader.load_raw_data()
        print(f"Successfully loaded raw data with shape: {raw_data.shape}")
    except Exception as e:
        print(f"Error in Test 1: {str(e)}")

    # Test 2: Data preprocessing
    print("\nTest 2: Data preprocessing")
    try:
        # Test preprocessing with optional filters
        processed_data = loader.preprocess_data(
            mouse_id="VF016",
            session="2024-07-16"   # Replace with actual session ID
        )
        print(f"Successfully preprocessed data with shape: {processed_data.shape}")
        print("Required columns present:", all(col in processed_data.columns for col in ['choice', 'reward']))
    except Exception as e:
        print(f"Error in Test 2: {str(e)}")

    # # Test 3: Get data information
    # print("\nTest 3: Get data information")
    # try:
    #     info = loader.get_data_info()
    #     print("Data information:")
    #     for key, value in info.items():
    #         print(f"{key}: {value}")
    # except Exception as e:
    #     print(f"Error in Test 3: {str(e)}")

    # # Test 4: Save processed data
    # print("\nTest 4: Save processed data")
    # try:
    #     # Replace with your desired output path
    #     output_path = "path/to/output/processed_data.csv"
    #     loader.save_processed_data(output_path)
    #     print(f"Data saved successfully to: {output_path}")
    # except Exception as e:
    #     print(f"Error in Test 4: {str(e)}")

    # # Test 5: Convenience function
    # print("\nTest 5: Convenience function")
    # try:
    #     data = load_data(
    #         test_data_path,
    #         mouse_id="test_mouse_id",
    #         session="test_session"
    #     )
    #     print(f"Successfully loaded and preprocessed data using convenience function. Shape: {data.shape}")
    # except Exception as e:
    #     print(f"Error in Test 5: {str(e)}") 