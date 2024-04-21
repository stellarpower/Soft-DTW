#!/usr/bin/env python


import numpy as np
import tensorflow as tf
import pickle, os
from softdtwkeras.SDTWLoss import SDTWLoss

ScriptDirectory = os.path.dirname(os.path.realpath(__file__))

# PRevent Tensorflow from pre-emptively allocating all the GPU memory.
# Apart from inconvenience, this lets us see how much the implementation is using.
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class TestSDTWLoss:


    GammaValues = [1.0, 0.1, 0.01]

    #TestShape = (32, 512, 16)
    TestShape = (8, 32, 8)
    #TestShape = (2, 32, 4)

    Filename = f"{ ScriptDirectory }/TestingLosses.pkl"

    def __init__(self):
        # We will test multiple gamma values, and store results for each value of gamma we test against.
        self.referenceLosses = {}
        self.losses          = {}


    # Just computes the loss for random numbers
    # We can save this file to disc with the master branch
    # And then load it and compare for any modifications
    # Unsophisticated but easy test to check we haven't deviated in the implementation.
    def computeLoss(self):
        # Set seed for reproducibility
        np.random.seed(0)

        # Generate 3D tensors of random numbers
        # Fixme - the loss should be parameterised on the intermediate computation type.
        y_true = np.random.rand( *TestSDTWLoss.TestShape ).astype('float32')
        y_pred = np.random.rand( *TestSDTWLoss.TestShape ).astype('float32')

        # Convert numpy arrays to tensors
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)




        for gamma in TestSDTWLoss.GammaValues:

            print(f"Computing array of shape { TestSDTWLoss.TestShape } for gamma of { gamma }")

            # Test against an instance for particular gamma
            # This should engage the graphcompiler and call through the backend

            sdtwLoss = SDTWLoss(gamma = gamma)
            loss, backwardsFunction = sdtwLoss.call(y_true, y_pred)
            # turned off in this reverted branch
            #loss = sdtwLoss.call(y_true, y_pred)

            # NOw compute the backwards pass.
            # Dummy upstream gradient of 1
            gradients = backwardsFunction(1)

            # Store the loss for current gamma; we will compare all of them with known values later.
            self.losses[gamma] = loss.numpy()

        
    
    def saveReferenceValues(self):
        # Save the result to a pickled array
        with open(TestSDTWLoss.Filename, 'wb') as f:
            pickle.dump(self.losses, f)

    def loadReferenceValues(self):
        # Load the pickled array
        with open(TestSDTWLoss.Filename, 'rb') as f:
            self.referenceLosses = pickle.load(f)

    def assertReferenceValues(self):
        def toArray(dict):
            return np.array(list(dict.values()))
        # Assert that the loaded loss is equal to the computed loss
        computedLosses, referneceLosses = toArray(self.losses), toArray(self.referenceLosses)
        
        assert np.allclose(computedLosses, referneceLosses), "Tests failed"

        print("Tests passed")
    

if __name__ == '__main__':

    tf.config.run_functions_eagerly(True)

    testCase = TestSDTWLoss()

    testCase.computeLoss()

    # Run me to generate the reference values against the master branch.
    # This will be rather slow currently given the implementation seems to fall back to CPU.
    if False: #True:
        testCase.saveReferenceValues()
    
    # Or run me to verify our branch matches the master.
    else:
        testCase.loadReferenceValues()
        testCase.assertReferenceValues()


    