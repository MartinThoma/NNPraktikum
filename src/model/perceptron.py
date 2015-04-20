from model.classifier import Classifier
import numpy as np
from util.activation_functions import Activation

from sklearn.metrics import accuracy_score

import logging
import sys

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class Perceptron(Classifier):
    """
    A digit-7 recognizer based on perceptron algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    learningRate : float
    epochs : int
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    """
    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small random values
        # around 0 and0.1
        self.weight = np.random.rand(self.trainingSet.input.shape[1], 1)/1000

    def train(self, verbose=True):
        """Train the perceptron with the perceptron learning algorithm.

        Parameters
        ----------
        verbose : bool
            Print logging messages with validation accuracy if verbose is True.
        """
        for i in range(1, self.epochs+1):
            pred = self.evaluate(self.validationSet.input)
            if verbose:
                val_acc = accuracy_score(self.validationSet.label, pred)*100
                logging.info("Epoch: %i (Validation acc: %0.4f%%)", i, val_acc)
            for X, y in zip(self.trainingSet, self.trainingSet.label):
                pred = self.classify(X)
                X = np.array([X]).reshape(784, 1)
                self.weight += self.learningRate * (y - pred) * X * (-1)

    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        return self.fire(testInstance)

    def evaluate(self, data=None):
        if data is None:
            data = self.testSet.input
        # One you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, data))

    def fire(self, input):
        return Activation.sign(np.dot(np.array(input), self.weight))
