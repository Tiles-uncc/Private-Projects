import numpy as np

from itcs4156.assignments.neural_networks.NeuralNetwork import NeuralNetwork

class NeuralNetworkClassifier(NeuralNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Code for extracting kwargs and storing them in _param_names
        # to be used later with get_params() and set_params() methods
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._param_names = list(kwargs.keys())
    
    def get_params(self, deep=True):
        """ Gets all class variables
        
            This is method is for compatibility with Sklearn's GridSearchCV 
        """
        return {param: getattr(self, param)
                for param in self._param_names}

    def set_params(self, **parameters):
        """ Sets all class variables
        
            This is method is for compatibility with Sklearn's GridSearchCV 
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """ Predict probabilities using parameters learned during training.
        
            This is method is for compatibility with Sklearn's GridSearchCV 
                
            Args:
                X: Features/data to make predictions with 

        """
        return self.forward(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Make predictions using parameters learned during training.
        
            Args:
                X: Features/data to make predictions with 

            TODO:
                Finish this method by adding code to make a prediction. 
                Store the predicted labels into `y_hat`.
        """
        # TODO (REQUIRED) Add code below
        y_hat_probs = self.forward(X)
        # TODO 13.6
        y_hat = np.argmax(y_hat_probs, axis=1)

        # TODO (REQUIRED) Store predictions below by replacing np.ones()

        # Makes sure predictions are given as a 2D array
        return y_hat.reshape(-1, 1)
