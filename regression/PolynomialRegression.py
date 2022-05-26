import numpy as np

from itcs4156.assignments.regression.OrdinaryLeastSquares import OrdinaryLeastSquares
from sklearn.preprocessing import PolynomialFeatures
    
class PolynomialRegression(OrdinaryLeastSquares):
    """
        Performs polynomial regression using ordinary least squares algorithm
    
        attributes:
            w (np.ndarray): weight matrix that is inherited from OrdinaryLeastSquares
            
            degree (int): the number of polynomial degrees to include when adding
                polynomial features.
    """

    def __init__(self, degree: int):
        super().__init__()
        self.degree = degree

    def add_polynomial_features(self, X: np.ndarray) -> np.ndarray:
        """ Computes polynomial features given the pass data.
        
            TODO:
                Finish this method by adding code to compute the polynomial features
                for X. Be sure to return the new data with the polynomial features!
            
            Hint: 
                Feel free to use sklearn.preprocessing.PolynomialFeatures but remember
                it includes the bias so make sure to disable said feature!
        """
        poly_feat = PolynomialFeatures(degree=self.degree, include_bias=False)
        X_poly = poly_feat.fit_transform(X)

        return X_poly
        

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Used to train our model to learn optimal weights.
        
            TODO:
                Finish this method by adding code to perform polynomial regression using
                the closed form solution OLS to learn the weights `self.w`.
                
            Hint:
                Since we inherit from OrdinaryLeastSquares you can simply just call 
                super().train(X, y) instead of copying the code from OrdinaryLeastSquares 
                after you run self.add_polynomial_features(X).
        """
        X = self.add_polynomial_features(X)
        super().fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Used to make a prediction using the learned weights.
        
            TODO:
                Finish this method by adding code to make a prediction given the learned
                weights `self.w`.
                
            Hint:
                Since we inherit from OrdinaryLeastSquares you can simply just call 
                super().predict(X) instead of copying the code from OrdinaryLeastSquares 
                after you run self.add_polynomial_features(X).
        """
        # TODO (REQUIRED) Add code below

        # TODO (REQUIRED) Store predictions below by replacing np.ones()
        X = self.add_polynomial_features(X)
        y_hat = super().predict(X)
        return y_hat
        