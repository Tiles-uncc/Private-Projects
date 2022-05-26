import numpy as np
import pandas as pd

from scipy.stats import norm

from itcs4156.models.ClassificationModel import ClassificationModel

class NaiveBayes(ClassificationModel):
    """
        Performs Gaussian Naive Bayes
    
        attributes:
            smoothing: smoothing hyperparameter used to prevent numerical instability and 
                divide by zero errors
                
            class_labels (np.ndarray or list): Unique labels for the passed data. This 
                should be set in the fit() method.
            
            priors (np.ndarray): NumPy array which stores the priors.
            
            log_priors (np.ndarray): NumPy array which stores the log of the priors and
                used by the predict() method.
            
            means (np.ndarray): NumPy array of means used by the
                log_gaussian_distribution() method to compute the log likelihoods
            
            stds (np.ndarray): NumPy array of standard deviations used by the
                log_gaussian_distribution() method to compute the log likelihoods
            
    """
    def __init__(self, smoothing: float = 10e-3):
        ClassificationModel.__init__(self)
        self.smoothing = smoothing
        # All class variables that need to be set somewhere within the below methods.
        self.class_labels = None
        self.priors = None
        self.log_priors = None
        self.means = None
        self.stds = None

    def log_gaussian_distribution(self, X: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
        """ Computes the log of a value at a given point in a Gaussian distribution
        
            Args:
                X: Data for which an output value is computed for.
                
                mu: Feature means
                
                var: Feature variance
        """
        return norm.logpdf(X, mu, std**2)
        
    def compute_priors(self, y: np.ndarray) -> None:
        """ Computes the priors and log priors for each class.
    
            Args:
                y: Lables
                
            TODO: 
                Finish this method by computing the priors and log priors to be used when
                making predictions using MAP. Store the computed priors and log priors 
                into self.priors and self.log_priors.
                
        """
        self.class_labels, class_counts = np.unique(y, return_counts=True)
        # TODO 5.2
        total_data_samples = len(y)
        # TODO 5.3
        priors = class_counts / total_data_samples
        # TODO 5.4
        log_priors = np.log(priors)
        
        self.priors = priors  # TODO Store priors here by replacing None
        self.log_priors = log_priors # TODO Store log priors here by replacing None
    
    def compute_parameters(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Computes the means and standard deviations for classes and features
        
            Args:
                X: Data 
                
                y: Targets/labels

            TODO: 
                Finish this method by computing the means and stds for the Gaussian
                distribution which will then be used to compute the likelihoods. Store
                the computed means and stds into self.means and self.stds.
        """
        # TODO Add code below
        self.class_labels = np.unique(y)

        means = []
        stds = []
        # TODO 9.2
        for label in self.class_labels:
            # TODO 9.3
            class_locs = np.where(y == label)[0]
            class_X = X[class_locs]

            # TODO 9.4
            class_mean = np.mean(class_X, axis=0)
            means.append(class_mean)

            # TODO 9.5
            class_std = np.std(class_X, axis=0)
            stds.append(class_std)
        
        self.means = np.vstack(means) # TODO Store means here by replacing None
        self.stds = np.vstack(stds) # TODO Store stds here by replacing None
        
        # Add smoothing term to standard deviation to
        # help prevent numerical instability when STD equal
        # or near 0.
        self.stds += self.smoothing
        
    def compute_log_likelihoods(self, X: np.ndarray) -> np.ndarray:
        """ Computes and returns log likelihoods using the means and stds
                
            Args:
                X: Data 
        
            TODO:
                Finish this method by computing the log likelihoods of the passed data 
                `X`. Use the `self.means` and `self.stds` class variables you set in 
                the `compute_parameters()` method along with the `log_gaussian_distribution()` 
                method which is defined for you. The `log_gaussian_distribution()`  
                will apply the log to your feature likelihoods for you so you don't need to!
                This method should return the computed log likelihoods.
        """
        log_likelihoods = []

        for class_mean, class_std in zip(self.means, self.stds):
            feature_log_likelihoods = self.log_gaussian_distribution(X, class_mean, class_std)

            class_log_likelihoods = np.sum(feature_log_likelihoods, axis=1)

            log_likelihoods.append(class_log_likelihoods)

        log_likelihoods = np.vstack(log_likelihoods)
        log_likelihoods = log_likelihoods.T
        return log_likelihoods
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Computes the priors and Gaussian parameters used for predicting.
        
            Args:
                X: Data 
                
                y: Targets/labels
                
             TODO:
                Finish this method by computing the priors and Gaussian parameters. 
                To do so, first finish and then call the compute_parameters() and 
                compute_priors() methods.
        """
        self.class_labels = np.unique(y)
        self.compute_priors(y)
        self.compute_parameters(X, y)

    def predict(self, X) -> np.ndarray:
        """ Comptues a prediction using log likelihoods and log priors.
        
            Args:
                X: Data 
        
             TODO:
                Finish this method by computing the log likelihoods and log priors.
                To do so, first finishing and then call the compute_log_likelihoods() 
                method. You'll also need to access the class variables self.log_priors 
                and self.class_labels you set when running the fit(), compute_parameters() 
                and compute_priors() methods. Store the predicted labels into `y_hat`.
        """
        # TODO Add code below
        log_likelihoods = self.compute_log_likelihoods(X)

        joint_log_likelihoods = log_likelihoods  + self.log_priors
        # TODO 12.2
        most_likely_locs = np.argmax(joint_log_likelihoods, axis=1)
        # TODO 12.3
        y_hat = self.class_labels[most_likely_locs]
        return y_hat.reshape(-1, 1)
