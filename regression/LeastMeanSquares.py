import numpy as np

from itcs4156.models.LinearModel import LinearModel

class LeastMeanSquares(LinearModel):
    """
        Performs regression using least mean squares (gradient descent)
    
        attributes:
            w (np.ndarray): weight matrix
            
            alpha (float): learning rate or step size
            
            epochs (int): Number of epochs to run for mini-batch
                gradient descent
                
            seed (int): Seed to be used for NumPy's RandomState class
                or universal seed np.random.seed() function.
    """
    def __init__(self, alpha: float, epochs: int, seed: int = None):
        super().__init__()
        self.w = None
        self.alpha = alpha
        self.epochs = epochs
        self.seed = seed
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Used to train our model to learn optimal weights.
        
            TODO:
                Finish this method by adding code to perform LMS in order to learn the 
                weights `self.w`.
        """
        m_samples = len(X)
        X = self.add_ones(X)

        rng = np.random.RandomState(self.seed)
        self.w = rng.rand(X.shape[1])

        for e in range(self.epochs):
            for i in range(m_samples):
                y_hat = X[i, :] @ self.w
                self.w = self.w - (self.alpha * (y_hat - y[i]) * X[i])


                # We need to reshape our preds to be a 2D array
                # otherwise when we compute (preds - y)**2 we will
                # get an array of shape (100, 100) instead of a (100, 1)
                # this is due to the automatic broadcasting NumPy does!
                preds = X @ self.w
                preds = preds.reshape(-1, 1)



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
