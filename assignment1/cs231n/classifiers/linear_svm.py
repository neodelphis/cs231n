from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    y = y.astype(int) # Par défault y est un tableau de float qui génère une erreur pour la transformation en 1-hot-E

    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        count = 0
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                count += 1
        # Calcul du gradient en un point
        # On peut l'intégrer dans la boucle précédente, mais les maths en deviennent moins lisibles
        for j in range(num_classes):
            if j == y[i]:
                dW[:,j] += - count * X[i]
            else:
                margin = scores[j] - correct_class_score + 1 # note delta = 1
                if margin > 0:
                    dW[:,j] += X[i]
                #    dW[:,j] += 1. * X[i]
                # else: # Ne sert à rien sauf à expliquer le calcul
                #    dW[:,j] += 0. * X[i]

            

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2* reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    delta = 1.0
    num_classes = W.shape[1]  # C
    num_train = X.shape[0]    # N
    y = y.astype(int) # Par défault y est un tableau de float qui génère une erreur pour la transformation en 1-hot-E


    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # compute the loss
    scores = np.matmul(X,W) # Matrice N,C
    y_one_hot = np.zeros((y.size, num_classes))
    y_one_hot[np.arange(y.size),y] = 1 # Matrice N,C
    
    # numpy.multiply : Multiply arguments element-wise.
    correct_class_score = np.sum(np.multiply(scores, y_one_hot), axis=1) # Vecteur de taille N
    # On le transforme en matrice pour enlever le score de la bonne classe à tous les éléments d'une ligne de scores
    margins = scores - np.array([correct_class_score,]*num_classes).T + delta
    margins = np.maximum(0, margins)
    
    # on y-th position scores[y] - scores[y] canceled and gave delta. We want
    # to ignore the y-th position and only consider margin on max wrong class
    # multiplication terme à terme par (1-y_one_hot) qui va renvoyer 0 juste pour l'élément s_yi
    margins = np.multiply(margins, (1-y_one_hot))
    
    loss = np.sum(margins)/num_train
    
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
                
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # cf notes

    s_grad = 1*(margins>0) + np.multiply(y_one_hot, - np.array([np.sum(margins>0, axis=1),]*num_classes).T )
    dW = np.matmul(X.T, s_grad)
    dW /= num_train
    
    # Add regularization to the gradient.
    dW += 2* reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
