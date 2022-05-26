from typing import List, Tuple, Union 

import numpy as np

from itcs4156.models.ClassificationModel import ClassificationModel

class LogisticRegression(ClassificationModel):
    """
        Performs Logistic Regression using the softmax function.
    
        attributes:
            alpha: learning rate or step size used by gradient descent.
                
            epochs: Number of times data is used to update the weights `self.w`.
                Each epoch means a data sample was used to update the weights at least
                once.
            
            seed (int): Seed to be used for NumPy's RandomState class
                or universal seed np.random.seed() function.
            
            batch_size: Mini-batch size used to determine the size of mini-batches
                if mini-batch gradient descent is used.
            
            w (np.ndarray): NumPy array which stores the learned weights.
    """
    def __init__(self, alpha: float, epochs: int = 1,  seed: int = None, batch_size: int = None):
        ClassificationModel.__init__(self)
        self.alpha = alpha
        self.epochs = epochs
        self.seed = seed
        self.batch_size = batch_size
        self.w = None

    def softmax(self, z: np.ndarray) -> np.ndarray:
        """ Computes probabilities for multi-class classification given continuous inputs z.
        
            Args:
                z: Continuous outputs after dotting the data with the current weights 

            TODO:
                Finish this method by adding code to return the softmax. Don't forget
                to subtract the max from `z` to maintain  numerical stability!
        """
        # TODO 12.1
        z = z - np.max(z, axis=-1, keepdims=True)
        # TODO 12.2
        e_z = np.exp(z)
        # TODO 12.3
        denominator = np.sum(e_z, axis=-1, keepdims=True)
        # TODO 12.4
        softmax = e_z / denominator
        return softmax

    def nll(self, y, probs, epsilon=1e-4):
        loss = y * np.log(probs)
        cost = -np.sum(loss) / len(y)
        return cost


    def init_weights(self, X, y, seed):
        rng = np.random.RandomState(seed)
        # TODO 14.3
        self.w = rng.rand(X.shape[1], y.shape[1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Train our model to learn optimal weights for classifying data.
        
            Args:
                X: Data 
                
                y: Targets/labels
                
             TODO:
                Finish this method by using either batch or mini-batch gradient descent
                to learn the best weights to classify the data. You'll need to finish and 
                also call the `softmax()` method to complete this method. Also, update 
                and store the learned weights into `self.w`. 
        """
        self.init_weights(X, y, self.seed)
        self.epoch_losses = []
        self.vld_epoch_losses = []
        g = self.softmax
        loss_func = self.nll
        for e in range(self.epochs):
            epoch_info = f"Epoch: {e + 1}/{self.epochs}"

            # TODO 9.4
            z = X @ self.w

            # TODO 9.5
            probs = g(z)
            # TODO 9.6
            avg_gradient = (X.T @ (probs - y)) / len(y)
            # TODO 9.7
            self.w = self.w - (self.alpha * avg_gradient)

            # TODO 9.8
            avg_loss = loss_func(y, probs)
            # TODO 9.9
            self.epoch_losses.append(avg_loss)
            epoch_info += f"\n\tAvg Training loss: {np.round(avg_loss, 4)}"

            '''if X_vld is not None and y_vld is not None:
                # TODO 9.10
                z_vld = X_vld @ self.w
                # TODO 9.11
                probs_vld = self.g(z_vld)
                # TODO 9.12
                avg_vld_loss = self.loss_func(y_vld, probs_vld)
                # TODO 9.13
                self.vld_epoch_losses.append(avg_vld_loss)
                epoch_info += f"\n\tAvg Validation loss: {np.round(avg_vld_loss, 4)}"
            if self.verbose: print(epoch_info)'''

       
    def predict(self, X: np.ndarray):
        """ Make predictions using the learned weights.
        
            Args:
                X: Data 

            TODO:
                Finish this method by adding code to make a prediction given the learned
                weights `self.w`. Store the predicted labels into `y_hat`.
        """
        # TODO Add code below
        g = self.softmax
        z = X @ self.w
        # TODO 14.5
        probs = g(z)
        # TODO 14.6
        y_hat = np.argmax(probs, axis=1)

        return y_hat.reshape(-1, 1)
