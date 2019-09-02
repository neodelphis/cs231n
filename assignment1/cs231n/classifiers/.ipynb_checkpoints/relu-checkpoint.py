#from builtins import range
import numpy as np
#from random import shuffle
#from past.builtins import xrange

class ReluLayer:
    """
    Propagation et rétropropagation dans une couche complètement connectée
    avec relu comme fonction d'activation
    """
    
    def forward(self, W, X):
        """
        relu forward pass function

        Inputs have dimension D, there are C classes, and we operate on minibatches
        of N examples.

        Inputs:
        - W: A numpy array of shape (D, C) containing weights.
        - X: A numpy array of shape (N, D) containing a minibatch of data.

        Returns:
        - Y: A numpy array of shape (N, C) containing a minibatch of data.
        """
        
        # Forward pass
        Y  = np.zeros_like(X)
        # logits
        self.logits = np.matmul(X,W)
        # relu
        Y = np.maximum(self.logits, 0)

        return Y


    def backprop(self, W, X, dL, reg):
        """
        relu back propagation
        relu backward pass function
        Inputs:
        Computed values

        returns:
        - dW = dL/dW
        - dX
        """

        # Backprop
        dlogits = dL*(self.logits>0)
        dW = X.T.dot(dlogits)
        dW /= X.shape[0]

        # Add regularization to the gradient.
        dW += 2* reg * W

        return dW


class ReluLayerWithDropout:
    """
    Propagation et rétropropagation dans une couche complètement connectée
    avec relu comme fonction d'activation
    """
    def __init__(self, dropout_rate = 0.1):
        self.dropout_rate = dropout_rate
    
    def forward(self, W, X):
        """
        relu forward pass function

        Inputs have dimension D, there are C classes, and we operate on minibatches
        of N examples.

        Inputs:
        - W: A numpy array of shape (D, C) containing weights.
        - X: A numpy array of shape (N, D) containing a minibatch of data.

        Returns:
        - Y: A numpy array of shape (N, C) containing a minibatch of data.
        """
        
        # Forward pass
        Y  = np.zeros_like(X)
        # logits
        self.logits = np.matmul(X,W)
        
        # dropout
        C = self.logits.shape[1]
        number_of_dropped_cells = int(self.dropout_rate * C)
        self.dropout_mask = np.random.choice(C, number_of_dropped_cells, replace=False)
        self.logits[:, self.dropout_mask] = 0
        
        # relu
        Y = np.maximum(self.logits, 0)

        return Y


    def backprop(self, W, X, dL, reg):
        """
        relu back propagation
        relu backward pass function
        Inputs:
        Computed values

        returns:
        - dW = dL/dW
        - dX
        """

        # Backprop
        #dropout 
        dL[:, self.dropout_mask] = 0
        dlogits = dL*(self.logits>0)
        dW = X.T.dot(dlogits)
        dW /= X.shape[0]

        # Add regularization to the gradient.
        dW += 2* reg * W

        return dW