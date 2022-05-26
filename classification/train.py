from pdb import set_trace
import numpy as np
from sklearn.pipeline import Pipeline

from itcs4156.util.data import AddBias, Standardization, ImageNormalization, OneHotEncoding

class HyperParametersAndTransforms():
    
    @staticmethod
    def get_params(name):
        model = getattr(HyperParametersAndTransforms, name)
        params = {}
        for key, value in model.__dict__.items():
            if not key.startswith('__') and not callable(key):
                if not callable(value) and not isinstance(value, staticmethod):
                    params[key] = value
        return params
    
    class Perceptron():
        """Kwargs for classifier the Perceptron class and data prep"""
        model_kwargs = dict(
            alpha = .2,  # TODO (REQUIRED) Set learning rate
            epochs = 10,  # TODO (REQUIRED) Set epochs
            seed = None, # TODO (OPTIONAL) Set seed for reproducible results
        )

        data_prep_kwargs = dict(
            # TODO (OPTIONAL) Add Pipeline() definitions below
            target_pipe = None,
        # TODO (REQUIRED) Add Pipeline() definitions below
            feature_pipe = Pipeline([('standard', Standardization()), ('addbias', AddBias())]),
        )
        
    class NaiveBayes():
        """Kwargs for classifier the NaiveBayes class and data prep"""
        model_kwargs = dict(
            smoothing = .01, # (OPTIONAL) TODO Set smoothing parameter for STD
        )
        
        data_prep_kwargs = dict(
            # TODO (OPTIONAL) Add Pipeline() definitions below
            target_pipe = None,
            # TODO (REQUIRED) Add Pipeline() definitions below
            feature_pipe = Pipeline([('standard', ImageNormalization())]),
        )
        
    class LogisticRegression():
        model_kwargs = dict(
            alpha = .2, # TODO (REQUIRED) Set learning rate
            epochs = 150, # TODO (REQUIRED) Set epochs
            seed = 42, # TODO (OPTIONAL) Set seed for reproducible results
            batch_size = None, # TODO (OPTIONAL) Set mini-batch size if using mini-batch gradient descent
        )
      
        data_prep_kwargs = dict(
            # TODO (REQUIRED) Add Pipeline() definitions below
            target_pipe = Pipeline([('onehot', OneHotEncoding())]),
            # TODO (REQUIRED) Add Pipeline() definitions below
            feature_pipe = Pipeline([('standard', Standardization()), ('addbias', AddBias())]),
        )