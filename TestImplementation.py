#!/usr/bin/env python


import numpy as np
import tensorflow as tf
import pickle, os

from   softdtwkeras.SDTWLoss         import SDTWLoss
#from  softdtwkeras.SDTWLossUnedited import SDTWLoss




ScriptDirectory = os.path.dirname(os.path.realpath(__file__))

# PRevent Tensorflow from pre-emptively allocating all the GPU memory.
# Apart from inconvenience, this lets us see how much the implementation is using.
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)



def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(        f)

def save(filename, object):
    with open(filename, 'wb') as f:
        return pickle.dump(object, f)

# EDIT:  Can just use np.load etc.


# For debugging the gradient tape logic.
class L1Loss:
    def __init__(self):
        pass

    def call(self, y_true, y_pred):
        return y_true - y_pred





class TestSDTWLoss:


    GammaValues = [1.0, 0.1, 0.01]

    #TestShape = ( 32, 512,  16)
    #TestShape = (  8,  32,   8)
    #TestShape = (  2,  32,   4)
    # This hsould offer fast computation.
    TestShape  = (  4,   8,   2)

    CacheDirectory = f"{ ScriptDirectory }/Test Arrays"
    
    # Stupid fucking language
    #InputsFilename = tuple( f"{ CacheDirectory }/{ type      }.pkl"   for type in ("y_true", "y_pred") )
    InputsFilename = (
        f"{ CacheDirectory }/y_true.pkl",
        f"{ CacheDirectory }/y_pred.pkl"
    )
    LossesFilename    = f"{ CacheDirectory }/TestingLosses.pkl"
    GradientsFilename = f"{ CacheDirectory }/TestingGradients.pkl"



    def __init__(self):
        # We will test multiple gamma values, and store results for each value of gamma we test against.
        self.referenceLosses    = {}
        self.losses             = {}
        self.referenceGradients = {}
        self.gradients          = {}





    # Just computes the loss for random numbers
    # We can save this file to disc with the master branch
    # And then load it and compare for any modifications
    # Unsophisticated but easy test to check we haven't deviated in the implementation.
    def computeLoss(self):

        # Convert numpy arrays to tensors
        y_true = tf.convert_to_tensor(self.y_true)
        y_pred = tf.convert_to_tensor(self.y_pred)



        for gamma in TestSDTWLoss.GammaValues:

            print(f"Computing array of shape { TestSDTWLoss.TestShape } for gamma of { gamma }")

            # Test against an instance for particular gamma
            # This should engage the graphcompiler and call through the backend

            sdtwLoss = SDTWLoss(gamma = gamma)
            

            with tf.GradientTape() as tape:

                # We only want/need to watch for changes in y_pred; y_true is a constant.
                tape.watch(y_pred)
                

                # Backwards function is not available - the tf.custom_gradient decorator
                # staches it away, but it's not returned, even though we explicitly return it.
                loss = sdtwLoss.call(y_true, y_pred)


                # Store the loss for current gamma; we will compare all of them with known values later.
                self.losses[gamma] = loss.numpy()


            # Now compute the backwards pass.
            # If we get hold of the backwards function, we could test with
            # a dummy upstream gradient of 1
            #gradients = backwardsFunction(1)

            
            # Tuple for gradient with respect to each variable
            gradients = tape.gradient(loss, [y_pred])
            gradients = gradients[0].numpy()

            # Store the loss for current gamma; we will compare all of them with known values later.
            # This should have the same shape as y_pred and y_true.
            self.gradients[gamma] = gradients


    def generateFreshNumbers(self):
        # Set seed for reproducibility
        np.random.seed(0)
        
        # Fixme - the loss should be parameterised on the intermediate computation type.
        self.y_true = np.random.rand( *TestSDTWLoss.TestShape ).astype('float32')
        self.y_pred = np.random.rand( *TestSDTWLoss.TestShape ).astype('float32')
    

    def saveReferenceValues(self):

        # Save the results and the input tensors to pickled arrays
        save(TestSDTWLoss.   InputsFilename[0],    self.y_true   )
        save(TestSDTWLoss.   InputsFilename[1],    self.y_pred   )
        # Careful - save the coputed, not reference here.
        save(TestSDTWLoss.   LossesFilename   ,    self.losses   )
        save(TestSDTWLoss.GradientsFilename   ,    self.gradients)


    def loadReferenceValues(self):
        # Load the pickled array
        self.referenceLosses    = load(TestSDTWLoss.   LossesFilename   )
        self.referenceGradients = load(TestSDTWLoss.GradientsFilename   )
        self.y_true             = load(TestSDTWLoss.   InputsFilename[0])
        self.y_pred             = load(TestSDTWLoss.   InputsFilename[1])

        assert self.y_true.shape == self.y_pred.shape == TestSDTWLoss.TestShape, \
            "Loaded cached arrays do not have the same shape as the hardcoded tuple in this file."
        
        #assert self.referenceLosses == TestSDTWLoss.GammaValues, \
        #    "Loaded cached arrays do not have the same gamma values as the hardcoded tuple in this file."


    def assertReferenceValues(self):
        def toArray(dict):
            return np.array(list(dict.values()))
        
        # Assert that the loaded loss is equal to the computed loss
        computedLosses,    referenceLosses    = toArray(self.losses   ), toArray(self.referenceLosses   )
        computedGradients, referenceGradients = toArray(self.gradients), toArray(self.referenceGradients)
        
        assert np.allclose(computedLosses,    referenceLosses                ), "Tests failed - losses    differ"

        # We need a slightly larger tolerance here
        assert np.allclose(computedGradients, referenceGradients, rtol = 1e-4), "Tests failed - gradients differ"

    
        print("Tests passed")
    

if __name__ == '__main__':

    tf.config.run_functions_eagerly(True ) #False)

    testCase = TestSDTWLoss()

    # Run me to generate the reference values against the master branch.
    # This will be rather slow currently given the implementation seems to fall back to CPU.
    if False: #True:
        testCase.generateFreshNumbers()
        testCase.computeLoss()
        testCase.saveReferenceValues()
    
    # Or run me to verify our branch matches the master.
    else:
        testCase.  loadReferenceValues()
        testCase.computeLoss()
        testCase.assertReferenceValues()


    