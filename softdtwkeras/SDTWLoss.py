import tensorflow as tf
import numpy as np

from softdtwkeras.soft_dtw_fast import py_softmin3


class SDTWLoss(tf.keras.losses.Loss):
    def __init__(self, gamma: float = 1.0):
        super(SDTWLoss, self).__init__()
        self.gamma = tf.convert_to_tensor(gamma)


    # Native python implementation of the Cython version
    # This allows tensorflow to compile it in the graph, and so is actually optimal over using Cython
    @staticmethod
    def softmin3(a, b, c, gamma):
        a /= -gamma
        b /= -gamma
        c /= -gamma

        max_val = tf.reduce_max([a, b, c])

        tmp = \
              tf.exp(a - max_val) \
            + tf.exp(b - max_val) \
            + tf.exp(c - max_val)
        
        logarithm = tf.math.log(tmp) + max_val
        result = -gamma * logarithm

        return result


    def call(self, y_true, y_pred):
        # tmp = [] # execution time : 14 seconds
        # for b_i in range(0, y_true.shape[0]):
        #     dis_ = self.unit_loss(y_true[b_i], y_pred[b_i])
        #     tmp.append(dis_)
        # return tf.reduce_sum(tf.convert_to_tensor(tmp))

        # batch execution loop -> execution time : 13
        batch_Distances_ = self.batch_squared_euclidean_compute_tf(y_true, y_pred)
        tmp = []
        for b_i in range(0, batch_Distances_.shape[0]):
            dis_ = self.unit_loss_from_D( batch_Distances_[b_i])
            tmp.append(dis_)
        return tf.reduce_sum(tf.convert_to_tensor(tmp))



    def squared_euclidean_compute_tf(self, a: tf.Tensor, b: tf.Tensor) -> None:
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
    

    def batch_squared_euclidean_compute_tf(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
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
    


    def unit_loss_from_D(self, D_):
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

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # D is indexed starting from 0.

                softMinimum = SDTWLoss.softmin3(
                    loss[i - 1, j    ],
                    loss[i - 1, j - 1],
                    loss[i    , j - 1],
                    
                    self.gamma
                )

                loss = tf.tensor_scatter_nd_update(
                    loss,
                    [ [i, j] ],
                    [ D_[i - 1, j - 1] + softMinimum ]
                )

        return loss[m, n]


    def unit_loss(self, y_true, y_pred):

        D_ = self.squared_euclidean_compute_tf(y_true, y_pred)

        return self.unit_loss_from_D(D_)
