import numpy as np
from scipy import stats
from scipy.optimize import minimize
from .base_model import BaseModel

class QLearningModel(BaseModel):
    """Q-learning model for 2-choice tasks."""
    
    def __init__(self, learning_rate=0.1, temperature=1.0, initial_value=0.5):
        """
        Initialize Q-learning model.
        
        Args:
            learning_rate (float): Learning rate (alpha) for updating Q-values
            temperature (float): Inverse temperature parameter for softmax choice rule
            initial_value (float): Initial Q-value for both actions
        """
        super().__init__("Q-Learning")
        self.parameters = {
            'learning_rate': learning_rate,
            'temperature': temperature,
            'initial_value': initial_value
        }
        self.q_values = None
        self.data = None
    
    def _softmax(self, q_values):
        """Apply softmax function to Q-values."""
        exp_values = np.exp(self.parameters['temperature'] * q_values)
        return exp_values / np.sum(exp_values)
    
    def _update_q_values(self, params):
        """Update Q-values using given parameters."""
        learning_rate = params[0]
        initial_value = params[2]
        
        n_trials = len(self.data)
        q_values = np.zeros((n_trials + 1, 2))
        q_values[0] = initial_value
        
        for t in range(n_trials):
            choice = self.data['choice'].iloc[t]
            reward = self.data['reward'].iloc[t]
            
            q_values[t + 1] = q_values[t].copy()
            q_values[t + 1, choice] = (q_values[t, choice] + 
                                     learning_rate * 
                                     (reward - q_values[t, choice]))
        return q_values
    
    def _negative_log_likelihood(self, params):
        """Calculate negative log-likelihood for optimization."""
        # Update parameters
        self.parameters['learning_rate'] = params[0]
        self.parameters['temperature'] = params[1]
        self.parameters['initial_value'] = params[2]
        
        # Update Q-values
        q_values = self._update_q_values(params)
        
        # Calculate choice probabilities
        choice_probs = np.zeros((len(self.data), 2))
        for t in range(len(self.data)):
            choice_probs[t] = self._softmax(q_values[t])
        
        # Calculate log-likelihood
        choices = self.data['choice'].values
        log_likelihood = np.sum(np.log(choice_probs[np.arange(len(choices)), choices]))
        
        return -log_likelihood  # Negative because we're minimizing
    
    def fit(self, data, optimize=True):
        """
        Fit the Q-learning model to the data.
        
        Args:
            data: DataFrame containing columns:
                - 'choice': 0 or 1 for each trial
                - 'reward': 0 or 1 for each trial
            optimize: Whether to optimize parameters using scipy.minimize
        """
        self.data = data
        
        if optimize:
            # Initial parameter values
            initial_params = [
                self.parameters['learning_rate'],
                self.parameters['temperature'],
                self.parameters['initial_value']
            ]
            
            # Parameter bounds
            bounds = [
                (0.01, 1.0),    # learning_rate bounds
                (0.01, 10.0),   # temperature bounds
                (0.0, 1.0)      # initial_value bounds
            ]
            
            # Optimize parameters
            result = minimize(
                self._negative_log_likelihood,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )
            
            # Update parameters with optimized values
            self.parameters['learning_rate'] = result.x[0]
            self.parameters['temperature'] = result.x[1]
            self.parameters['initial_value'] = result.x[2]
            
            print(f"Optimization successful: {result.success}")
            print(f"Final negative log-likelihood: {result.fun:.2f}")
        
        # Update Q-values with final parameters
        self.q_values = self._update_q_values([
            self.parameters['learning_rate'],
            self.parameters['temperature'],
            self.parameters['initial_value']
        ])
        
        self.fitted = True
        return self
    
    def predict(self, data):
        """
        Generate choice probabilities for each trial.
        
        Args:
            data: DataFrame containing the same format as fit()
            
        Returns:
            Array of choice probabilities for each trial
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        n_trials = len(data)
        choice_probs = np.zeros((n_trials, 2))
        
        # Calculate choice probabilities using current Q-values
        for t in range(n_trials):
            choice_probs[t] = self._softmax(self.q_values[t])
        
        return choice_probs
    
    def calculate_log_likelihood(self, data):
        """
        Calculate the log-likelihood of the choices under the model.
        
        Args:
            data: DataFrame containing the same format as fit()
            
        Returns:
            float: Log-likelihood of the data
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating likelihood")
        
        choice_probs = self.predict(data)
        choices = data['choice'].values
        
        # Calculate log-likelihood of each choice
        log_likelihood = np.sum(np.log(choice_probs[np.arange(len(choices)), choices]))
        
        return log_likelihood 