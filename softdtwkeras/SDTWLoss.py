import tensorflow as tf
import numpy as np



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
    @tf.function
    def callStatic(y_true, y_pred, gamma):
        # tmp = [] # execution time : 14 seconds
        # for b_i in range(0, y_true.shape[0]):
        #     dis_ = self.unit_loss(y_true[b_i], y_pred[b_i])
        #     tmp.append(dis_)
        # return tf.reduce_sum(tf.convert_to_tensor(tmp))
    
        # batch execution loop -> execution time : 13
        batch_Distances_ =  SDTWLoss.batch_squared_euclidean_compute_tf(y_true, y_pred)


        # Maps over axis 0 (sequences in batch) and compute the loss for each separate sequence independently.
        individualLossesForEachSequence = tf.map_fn(
            lambda sequence: SDTWLoss.unit_loss_from_D(sequence, gamma),
            batch_Distances_
        )

        # Now we just sum over all sequences in the batch for a scalar return value.
        return tf.reduce_sum(
            tf.convert_to_tensor(individualLossesForEachSequence)
        )


    
    @staticmethod
    @tf.function
    def squared_euclidean_compute_tf(a: tf.Tensor, b: tf.Tensor) -> None:
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
    @tf.function
    def batch_squared_euclidean_compute_tf(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """
        Computes pairwise distances between each elements of A and each elements of B.
        Args:
          A,                    [m,d] matrix
          B,                    [n,d] matrix
        Returns:
          pairwiseDistances,    [m,n] matrix of pairwise distances
        """

        # Expand dimensions to enable broadcasting
        a_expanded = tf.expand_dims(a, axis = 2)  # Shape: [batch, m, 1, d]
        b_expanded = tf.expand_dims(b, axis = 1)  # Shape: [batch, 1, n, d]

        # Compute pairwise squared Euclidean distances
        squared_diff = tf.reduce_sum(
            tf.square(a_expanded - b_expanded),
            axis = -1
        )  # Shape: [batch, m, n]

        return squared_diff
    

    @staticmethod
    @tf.function
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
    def unit_loss(y_true, y_pred, gamma):

        D_ = SDTWLoss.squared_euclidean_compute_tf(y_true, y_pred)

        return SDTWLoss.unit_loss_from_D(D_, gamma)
    

