from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_train = X.shape[0]
    for i in range(num_train):
        logits =  X[i].dot(W)
        logits -= np.max(logits) # Pour éviter les instabilités axis = 1
        correct_class_score = logits[y[i]]
        loss += -correct_class_score + np.log(np.sum(np.exp(logits)))
        
        # Calcul du gradient: dérivée partielle de L par rapport à W
        # Gradient de L par rapport aux logits
        dL = np.exp(logits) / np.sum(np.exp(logits))
        # Prise en compte du cas particulier dL[y[i]]
        dL[y[i]] = dL[y[i]] - 1
        # dW
        dW += np.outer(X[i], dL)
        

        
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2* reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    y = y.astype(int) # Par défault y est un tableau de float qui génère une erreur pour la transformation en 1-hot-E
    y_one_hot = np.zeros((y.size, num_classes))
    y_one_hot[np.arange(y.size),y] = 1 # Matrice N,C
    
    logits = np.matmul(X,W)
    #logits = np.max(logits, axis=1)
    logits = logits - np.array([np.max(logits, axis=1),] * num_classes).T

    
    # Utilisation du produit de Hadamard: multiplication terme à terme de matrices
    correct_class_terms = np.sum(np.multiply(y_one_hot, logits), axis=1)
    other_classes_terms = np.log(np.sum(np.exp(logits), axis=1))
    
    loss = np.sum( - correct_class_terms + other_classes_terms) 
    
    # Gradient de L par rapport aux logits
    # Softmax vectorisée
    numerateur   = np.exp(logits)
    denominateur = np.array([np.sum(np.exp(logits), axis=1),] * num_classes).T
    softmax      = numerateur / denominateur # element
    
    # Prise en compte du cas particulier dL[y[i]] qui donne S(y_i)-1
    dL = softmax - y_one_hot

    # Calcul du gradient: dérivée partielle de L par rapport à W
    # dW
    dW = np.matmul(X.T, dL) # Dans la multiplication matricielle on a la somme sur tous les X
    
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2* reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    Inputs and outputs are the same as softmax_loss_naive.
    Gradients of Loss with respect to W and X
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    dX = np.zeros_like(X)

    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    y = y.astype(int) # Par défault y est un tableau de float qui génère une erreur pour la transformation en 1-hot-E
    y_one_hot = np.zeros((y.size, num_classes))
    y_one_hot[np.arange(y.size),y] = 1 # Matrice N,C
    
    logits = np.matmul(X,W)
    logits = logits - np.array([np.max(logits, axis=1),] * num_classes).T
    
    # Utilisation du produit de Hadamard: multiplication terme à terme de matrices
    correct_class_terms = np.sum(np.multiply(y_one_hot, logits), axis=1)
    other_classes_terms = np.log(np.sum(np.exp(logits), axis=1))
    
    loss = np.sum( - correct_class_terms + other_classes_terms) 
    
    # Gradient de L par rapport aux logits
    # Softmax vectorisée
    numerateur   = np.exp(logits)
    denominateur = np.array([np.sum(np.exp(logits), axis=1),] * num_classes).T
    softmax      = numerateur / denominateur # element
    # Prise en compte du cas particulier dL[y[i]] qui donne S(y_i)-1
    dL = softmax - y_one_hot

    # Calcul des gradients
    # dW: dérivée partielle de L par rapport à W
    dW = np.matmul(X.T, dL) # Dans la multiplication matricielle on a la somme sur tous les X
    # dX: dérivée partielle de L par rapport à X
    dX = np.matmul(dL, W.T)
    
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2* reg * W

    return loss, dW, dX
