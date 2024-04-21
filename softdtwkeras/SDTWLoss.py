import tensorflow as tf
import numpy as np



# Can't seem to force tensorflow to run eagerly in general.
# Maps etc. seem always to be compiled ot the graph.
# This decorator effectively disables the tf.function decorator if eager execution is enabled.
RunEagerly = True 

def OptionalGraphFunction(func):
    return func if RunEagerly else tf.function(func)
    


class SDTWLoss(tf.keras.losses.Loss):
    def __init__(self, gamma: float = 1.0):
        super(SDTWLoss, self).__init__()
        self.gamma = tf.convert_to_tensor(gamma)


    # Native python implementation of the Cython version
    # This allows tensorflow to compile it in the graph, and so is actually optimal over using Cython
    @staticmethod
    @tf.function
    def softmin3(a, b, c, gamma):
        a /= -gamma
        b /= -gamma
        c /= -gamma

        max_val = tf.reduce_max([a, b, c])

        rTotal = \
              tf.exp(a - max_val) \
            + tf.exp(b - max_val) \
            + tf.exp(c - max_val)
        
        logarithm = tf.math.log(rTotal) + max_val
        result = -gamma * logarithm

        return result

    
    # Not sure if instance methods can be tf.functions
    # So forward to the static version
    def call      (self, y_true, y_pred):
        return SDTWLoss.callStatic(y_true, y_pred, self.gamma)


    @staticmethod
    #@tf.custom_gradient
    @OptionalGraphFunction
    def callStatic(y_true, y_pred, gamma):


        # Maps over axis 0 (sequences in batch) and compute the loss for each separate sequence independently.
        individualLossesForEachSequence, distanceMatricesForEachSequence = tf.map_fn(

            # map_fn does not expand the tuple; we need to explode it ourselves.
            lambda asTuple: SDTWLoss.computeSingleSequenceLoss(*asTuple, gamma),
            (y_true, y_pred),
            
            # We have to specify that the output is just a scalar value, and not a tensor, when te input and output shapes differ
            # FIXME: The second represents a tensor for the distance matrix, and it _not_ a scalar - but the dtype is sufficient to allow this to run.
            # FIXME This is hardcoded single-precision float for now.
            fn_output_signature = (tf.float32, tf.float32)
        )

        # Now we just sum over all sequences in the batch for a scalar return value.
        result = tf.reduce_sum(
            tf.convert_to_tensor(individualLossesForEachSequence)
        )

        forwardCalculations = (
            #batch_Distances_, result, "lengths", gamma, "bandwidth"
            batch_Distances_, result, gamma,
        )

        # Now we have to return both the forward pass and the gradient function
        return result, lambda upstream: SDTWLoss.backwardPass(y_true, y_pred, forwardCalculations, upstream)




    # This should be applied on each sequence in the batch.
    # These are separate, so we can do in parallel and make life easier and help the graph optimise.
    @staticmethod
    @OptionalGraphFunction
    def computeSingleSequenceLoss(y_true, y_pred, gamma):

        pairwiseDistanceMatrix = SDTWLoss.computePairwiseDistanceMatrix(y_true, y_pred)

        unitLoss = SDTWLoss.unit_loss_from_D(pairwiseDistanceMatrix, gamma)

        # The distance matrix is needed for the backward pass, so return it too.
        return (unitLoss, pairwiseDistanceMatrix)




    
    @staticmethod
    @OptionalGraphFunction
    def computePairwiseDistanceMatrix(a: tf.Tensor, b: tf.Tensor) -> None:
        """
        # return pairwise euclidean difference matrix
        Args:
          A,                    [m,d] matrix
          B,                    [n,d] matrix
        Returns:
          pairwiseDistances,    [m,n] matrix of pairwise distances
        """
        pairwiseDistances = tf.reduce_sum(
            (  tf.expand_dims(a, 1) - tf.expand_dims(b, 0)  ) ** 2,
            2
        )
        return pairwiseDistances
    

    
    

    

    @staticmethod
    @OptionalGraphFunction
    def unit_loss_from_D(D_,  gamma : tf.Tensor):
        m, n = tf.shape(D_)[0], tf.shape(D_)[1]

        # Allocate memory.
        loss = tf.fill(
            (m + 2, n + 2),
            tf.constant(np.inf) # , dtype = tf.float32) # Todo - parameterise me later.
        )

        loss = tf.tensor_scatter_nd_update(
            loss,
            [[0, 0]],
            [0.0]
        )
        
        # The graph compiler will automatically convert these loops to the appropriate backend-compatible loops
        # but only if the loop condition is a tensor itself.
        for i in tf.range(1, m + 1):
            for j in tf.range(1, n + 1):

                # D is indexed starting from 0.

                softMinimum = SDTWLoss.softmin3(
                    loss[i - 1, j    ],
                    loss[i - 1, j - 1],
                    loss[i    , j - 1],
                    
                    gamma
                )

                loss = tf.tensor_scatter_nd_update(
                    loss,
                    [ [i, j] ],
                    [ D_[i - 1, j - 1] + softMinimum ]
                )

        return loss[m, n]


    

    @staticmethod
    @tf.function
    def backwardPass(y_true, y_pred, forwardCalculations, upstream):
        # Think we should return a tensor, not a scalar, but not sure
        
        batch_Distances_, result, gamma, = forwardCalculations
        

        gradients = tf.map_fn(
            lambda resultsThisBatch: SDTWLoss.backwardsOneBatch(resultsThisBatch, y_true, y_pred, forwardCalculations, upstream),
            tf.zip(result, batch_Distances_)
        )
        
        # y_true is not a parameter, so, we return None.
        # FIXME - tensorflow thinks the gamma is aparameter, so, it expects a gradient for it.
        # Return None for now, to disable computation
        return None, upstream * gradients, None
        


    # One sequence in the batch.
    @staticmethod
    @tf.function
    def backwardsOneBatch(resultsThisBatch, y_true, y_pred, forwardCalculations, upstream):
        
        # Unzip to get the distance matrix too
        (resultThisBatch, batch_Distances_) = resultsThisBatch

        #m, n, _lengths, gamma, _bandwidth = forwardCalculations
        batch_Distances_, result, gamma, = forwardCalculations


        m, n = tf.shape(batch_Distances_)[0], tf.shape(batch_Distances_)[1]

        # The gradient array needs to be padded to be larger than the original distances matrix
        gradientsShape = (m + 2, n + 2)
        
        # Result for the gradients
        # Called E in the paper
        gradients = tf.zeros(gradientsShape)

        # The graph compiler will automatically convert these loops to the appropriate backend-compatible loops
        # but only if the loop condition is a tensor itself.
        for j in tf.range(m, 0, -1):
            for i in tf.range(n, 0, -1):
                a =  resultThisBatch[i + 1, j    ]   -   resultThisBatch[i, j]   -   batch_Distances_[i + 1, j    ]
                b =  resultThisBatch[i,     j + 1]   -   resultThisBatch[i, j]   -   batch_Distances_[i,     j + 1]
                c =  resultThisBatch[i + 1, j + 1]   -   resultThisBatch[i, j]   -   batch_Distances_[i + 1, j + 1]

                a = tf.exp(a / gamma)
                b = tf.exp(b / gamma)
                c = tf.exp(c / gamma)

                gradient = (

                        a * gradients[i + 1, j    ]
                     +  b * gradients[i,     j + 1]
                     +  c * gradients[i + 1, j + 1]
                )

                gradients[i, j] = gradient



        ## Andthen we need to remove the padding before returning
        unpadded = gradients[1:(n + 1), 1:(m + 1)]

        return unpadded
    

                
