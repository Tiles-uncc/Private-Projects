import numpy as np

from itcs4156.models.LinearModel import LinearModel

class OrdinaryLeastSquares(LinearModel): 
    """ 
        Performs regression using ordinary least squares
        
        attributes:
            w (np.ndarray): weight matrix
            
    """
    def __init__(self):
        super().__init__()
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Used to train our model to learn optimal weights.
        
            TODO:
                Finish this method by adding code to perform OLS in order to learn the 
                weights `self.w`.
        """
        X = self.add_ones(X)
        self.w = np.linalg.pinv(X.T @ X) @ X.T @ y
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Used to make a prediction using the learned weights.
        
            TODO:
                Finish this method by adding code to make a prediction given the learned
                weights `self.w`.
        """
        # TODO (REQUIRED) Add code below

        # TODO (REQUIRED) Store predictions below by replacing np.ones()
        X = self.add_ones(X)
        y_hat = X @ self.w
        
        return y_hat
