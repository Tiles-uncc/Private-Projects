import numpy as np

from itcs4156.models.ClassificationModel import ClassificationModel

class Perceptron(ClassificationModel):
    """
        Performs Gaussian Naive Bayes
    
        attributes:
            alpha: learning rate or step size used by gradient descent.
                
            epochs (int): Number of times data is used to update the weights `self.w`.
                Each epoch means a data sample was used to update the weights at least
                once.
                
            seed (int): Seed to be used for NumPy's RandomState class
                or universal seed np.random.seed() function.
            
            batch_size (int): Mini-batch size used to determine the size of mini-batches
                if mini-batch gradient descent is used.
            
            w (np.ndarray): NumPy array which stores the learned weights.
    """
    def __init__(self, alpha: float, epochs: int = 1, seed: int = None):
        ClassificationModel.__init__(self)
        self.alpha = alpha
        self.epochs = epochs
        self.seed = seed
        self.w = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """ Train model to learn optimal weights when performing binary classification.
        
            Args:
                X: Data 
                
                y: Targets/labels
                
             TODO:
                Finish this method by using Rosenblatt's Perceptron algorithm to learn
                the best weights to classify the binary data. There is no need to
                implement th pocket algorithm unless you choose to do so. Also, update 
                and store the learned weights into `self.w`.
        """
        # Number of data samples
        m_samples = X.shape[0]

        rng = np.random.RandomState(42)
        # TODO 12.1
        self.w = rng.rand(X.shape[1])

        # TODO 12.2
        self.w_best = self.w.copy()


        # Loop over dataset multiple times
        e = 0
        for e in range(self.epochs):
            print(f"Epoch {e + 1}")
            misclassified = 0
            # Loop over all samples
            for i in range(m_samples):

                # TODO 12.3
                z = self.w @ X[i]

                # TODO 12.4
                y_hat = np.sign(z)

                # TODO 12.5
                if y_hat != y[i]:
                    # TODO 12.6
                    self.w = self.w + (self.alpha * X[i] * y[i])

                    # Update best weights
                    replace_wights = self.replace_best_weights(X, y)
                    if replace_wights is True:
                        print(f"\tUpdating best weights based on data sample {i + 1}...")
                        self.w_best[:] = self.w[:]

                    misclassified += 1

            print(f"\tData samples misclassified: {misclassified}")

            # Convergence check to see if we are no longer
            # misclassifing data samples
            if misclassified == 0:
                print(f"Converged at epoch: {e + 1}")
                break

        print(f"Epochs trained: {e + 1}")

    def replace_best_weights(self, X, y):
        # TODO 12.7
        preds_w = np.sign(X @ self.w).reshape(-1, 1)
        # TODO 12.8
        preds_w_best = np.sign(X @ self.w_best).reshape(-1, 1)

        # TODO 12.9
        total_corr_w = np.sum(preds_w == y)
        # TODO 12.10
        total_corr_w_best = np.sum(preds_w_best == y)

        # Check if total_corr_w correctly classified more
        # samples than total_corr_w_best. If so, return True.
        if total_corr_w > total_corr_w_best:
            return True
        else:
            return False
   
    def predict(self, X: np.ndarray):
        """ Make predictions using the learned weights.
        
            Args:
                X: Data 

            TODO:
                Finish this method by adding code to make a prediction given the learned
                weights `self.w`. Store the predicted labels into `y_hat`.
        """
        # TODO Add code below
        
        y_hat = np.sign(X @ self.w_best).reshape(-1,1)
        # Makes sure predictions are given as a 2D array
        return y_hat.reshape(-1, 1)

