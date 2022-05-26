class HyperParameters():
    
    @staticmethod
    def get_params(name):
        model = getattr(HyperParameters, name)
        return {key:value for key, value in model.__dict__.items() 
            if not key.startswith('__') and not callable(key)}
    
    class OrdinaryLeastSquares():
        pass # No hyperparamters to set
        
    class LeastMeanSquares():
        model_kwargs = dict(
            alpha = 0.12,# TODO (REQUIRED) Set your learning rate
            epochs = 2, # TODO (OPTIONAL) Set number of epochs
            seed = 42 # TODO (OPTIONAL) Set seed for randomly generated weights
        )

        data_prep_kwargs = dict(
            # TODO (OPTIONAL) Set the names of the features/columns to use for the Housing dataset
            #['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
            #DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'],
            use_features = ['RM', 'NOX'],
        )

    class PolynomialRegression():
        model_kwargs = dict(
            degree = 6, # TODO (REQUIRED) Set your polynomial degree
        )
        
    class PolynomialRegressionRegularized():
        model_kwargs = dict(
            degree = 6, # TODO (REQUIRED) Set your polynomial degree
            lamb = .65, # TODO (REQUIRED) Set your regularization value for lambda
        )

        data_prep_kwargs = dict(
            # TODO (OPTIONAL) Set the names of the features/columns to use for the Housing dataset
            #use_features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
            #                'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'],
            use_features = ['INDUS', 'DIS', 'NOX' ]
        )