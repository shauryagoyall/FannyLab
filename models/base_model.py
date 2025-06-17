from abc import ABC, abstractmethod
import numpy as np
from scipy import stats

class BaseModel(ABC):
    """Base class for all behavioral models."""
    
    def __init__(self, name):
        self.name = name
        self.parameters = {}
        self.fitted = False
        
    @abstractmethod
    def fit(self, data):
        """Fit the model to the data."""
        pass
    
    @abstractmethod
    def predict(self, data):
        """Generate predictions from the model."""
        pass
    
    def calculate_bic(self, data):
        """Calculate Bayesian Information Criterion (BIC)."""
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating BIC")
            
        n = len(data)  # number of observations
        k = len(self.parameters)  # number of parameters
        
        # Calculate log-likelihood
        log_likelihood = self.calculate_log_likelihood(data)
        
        # Calculate BIC
        bic = k * np.log(n) - 2 * log_likelihood
        return bic
    
    @abstractmethod
    def calculate_log_likelihood(self, data):
        """Calculate the log-likelihood of the data under the model."""
        pass
    
    def get_parameters(self):
        """Return the model parameters."""
        return self.parameters.copy() 