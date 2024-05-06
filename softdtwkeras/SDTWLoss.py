import tensorflow as tf
import numpy as np, jax, functools, time
import jax.numpy as jnp
from jax.experimental import jax2tf
from jax import lax
from termcolor import colored



# Can't seem to force tensorflow to run eagerly in general.
# Maps etc. seem always to be compiled ot the graph.
# This decorator effectively disables the tf.function decorator if eager execution is enabled.
RunEagerly = False #True 




class SDTWLoss(tf.keras.losses.Loss):


    def __init__(self, outputShape, gamma: float = 1.0, preCompile : bool = True):
        super(SDTWLoss, self).__init__()

        # We will see if we can use these like constants, as when providing them as arguments
        # - as is advised for tf.function - the JIT-compiler is trying to trace them and getting itsely
        # completely tangled in knots.
        self.outputShape = outputShape
        self.gamma       = gamma

        self.m = self.n = self.outputShape[1]

        
        # Looks like if this appears in call, it is recompiled each time.
        # So for now ,cache it for the lifetime of the loss function.
        self.jitFunction = jax2tf.convert(
            self.callStatic,

            # The batch size will often be None at the point TensorFlow calls in; speify that this needs to be worked out later on.
            # We need to specify twice (as we have two parameters - GT and predicted.)
            polymorphic_shapes = 2 * ["(b, _, _)"],
        )

        if preCompile:
            self.preCompile()



    # Attempt to force the JIT compiler to instantiate everything with the correct shape 
    def preCompile(self):
        # Invokle the JIT compiler early to pre-cache the compilation before we want to do something real with it.
        print(colored("Precompiling XLA, for whatever reason this appears very slow.", "cyan"))

        # Create random numbers of the same shape to invoke.
        y_true = tf.random.uniform(self.outputShape)
        y_pred = tf.random.uniform(self.outputShape)

        # Invoke the JIT compiler early by sending through some random numbers.
        # Compute the gradients too, to make sure that the custom gradient gets invoked too.

        startTime = time.time()
        with tf.GradientTape() as tape:

            # We only want/need to watch for changes in y_pred; y_true is a constant.
            tape.watch(y_pred)
            

            # Backwards function is not available - the tf.custom_gradient decorator
            # staches it away, but it's not returned, even though we explicitly return it.
            loss = self.call(y_true, y_pred)


        # Now compute the backwards pass.   
        # Tuple for gradient with respect to each variable
        gradients = tape.gradient(loss, [y_pred])[0]
        
        functionTime = time.time() - startTime

        print(colored(f"Precompilation complete in {functionTime:.2g}s", "cyan"))




    # Native python implementation of the Cython version
    # This allows tensorflow to compile it in the graph, and so is actually optimal over using Cython
    def softmin3(self, a, b, c):
        a /= -self.gamma
        b /= -self.gamma
        c /= -self.gamma

        max_val = jnp.max(jnp.array([a, b, c]))

        rTotal = \
              jnp.exp(a - max_val) \
            + jnp.exp(b - max_val) \
            + jnp.exp(c - max_val)
        
        logarithm = jnp.log(rTotal) + max_val
        result = -self.gamma * logarithm

        return result

    
    # Not sure if instance methods can be tf.functions
    # So forward to the static version
    #@tf.function(jit_compile = True, autograph = False)
    @tf.function(jit_compile = False, autograph = False)
    def call      (self, y_true, y_pred):

        result = self.jitFunction(y_true, y_pred)#, self.gamma, m, n)
        return result




    # This function does not have a custom gradient - we map over the sequences in the batch. 
    # So only computeSingleSequenceLoss() has a custom gradient
    
    # TODO - should specify gamma in the same way
    #####@functools.partial(jax.jit, static_argnames=['m', 'n', 'gamma'])# NEed to specify that m and n are fixed and not inferred at runtime.
    def callStatic(self, y_true, y_pred): #, gamma, m, n):

        #@functools.partial(jax.jit, static_argnames=['m', 'n', 'gamma'])
        def t(yT, yP): 
            thing = self.computeSingleSequenceLoss(yT, yP) #, gamma, m, n)
            return thing


        # Maps over axis 0 (sequences in batch) and compute the loss for each separate sequence independently.
        unitLossesForEachSequence = jax.vmap(

            # JAX does expand the tuple.
            #lambda yT, yP: SDTWLoss.computeSingleSequenceLoss(yT, yP, gamma, m, n),
            t,
            in_axes=(0, 0) # Take the first axis (sequences in batch)

        )(y_true, y_pred)

        # Now we just sum over all sequences in the batch for a scalar return value.
        summedLossForAllSequences = jnp.sum(unitLossesForEachSequence)

        return summedLossForAllSequences




    # This should be applied on each sequence in the batch.
    # These are separate, so we can do in parallel and make life easier and help the graph optimise.
    #@functools.partial(jax.jit, static_argnames=['m', 'n', 'gamma'])# NEed to specify that m and n are fixed and not inferred at runtime.
    def computeSingleSequenceLoss(self, y_true, y_pred): #, gamma, m, n):
        
        pairwiseDistanceMatrix = SDTWLoss.computePairwiseDistanceMatrix(y_true, y_pred)

        #m, n = jnp.shape(pairwiseDistanceMatrix)[0], jnp.shape(pairwiseDistanceMatrix)[1]

        # Scalar loss
        # We no loner need to cache the full versions for the backward pass, seeing as we handle this
        # on a per-sequence basis.
        lossMatrix = self.computeLossMatrixFromDistanceMatrix_wrapped(pairwiseDistanceMatrix)#, m, n, gamma)

        unitLoss = lossMatrix #[m, n]

        return unitLoss




    # We can now slot in a different distance function if we want - the graph compiler will automatically
    # handle he differentiation for us here - we only need to return the alignment matrix, and the rest will
    # be taken care of.
    @staticmethod
    # This may need to be handled in TF for differentiation, but we'll try it in JAX for now
    def computePairwiseDistanceMatrix(a, b) -> None:
        """
        # return pairwise euclidean difference matrix
        Args:
          A,                    [m,d] matrix
          B,                    [n,d] matrix
        Returns:
          pairwiseDistances,    [m,n] matrix of pairwise distances
        """
        pairwiseDistances = jnp.sum(
            (  jnp.expand_dims(a, 1) - jnp.expand_dims(b, 0)  ) ** 2,
            axis = 2
        )
        
        return pairwiseDistances
    

    def computeFullLossMAtrix(self, distanceMatrix):

        # Fill the matrix with infinities - presum to represent infinite distance
        # An prevent an overflow at the edges.
        # https://github.com/Sleepwalking/pytorch-softdtw/blob/ddff7e3237a3520711f5b48b9e1ffc4647e9ef4a/soft_dtw.py#L11
        lossMatrix = jnp.full((self.m + 2, self.n + 2), jnp.inf)

        # Set the top-left item to be zero - it has zero distance form itself(?)
        # https://github.com/Sleepwalking/pytorch-softdtw/blob/ddff7e3237a3520711f5b48b9e1ffc4647e9ef4a/soft_dtw.py#L12
        lossMatrix = lossMatrix.at[0, 0].set(0.0)

        

             
            
        
        def loopBody(i, j, lossMatrix):  
            # https://github.com/toinsson/pysdtw/blob/c902025cf8d8926fd4a85ea3620002be9b4715d7/pysdtw/sdtw_cpu.py#L98C1-L100C1
            # The if statement is not symbolic - it is evaluated within python, so it can't contain deferred evaluation - 
            # what in effect is happening is the literal value of the predicate is deciding what AST gets passed to the JIT compiler.
            # Treat it like if constexpr rather than if.
            lossMatrix = lax.cond(
                # This is called a predicate, but is not actually a function but the evaluated condition.
                jnp.isinf(lossMatrix[i, j]),
                
                # Set it in this case.
                lambda array: array.at[i, j].set(-jnp.inf),
                # Do nothing
                lambda array: array,

                # NEed to pass in functionally; assume this helps it optimise.
                lossMatrix
            )

            # D is indexed starting from 0.

            softMinimum = self.softmin3(
                lossMatrix[i - 1, j    ],
                lossMatrix[i - 1, j - 1],
                lossMatrix[i    , j - 1],
                
                #gamma
            )
            lossMatrix = lossMatrix.at[i, j].set(
                distanceMatrix[i - 1, j - 1]
                + softMinimum
            )
            return lossMatrix

        # for loops are unrolled inline in Jax - this means we have enormous compile times, as the entire body is flattened out to source code
        # and then the optimiser works over it.
        # By using the backend loop, this should pass down to HLO in a more idiomatic form.
        
        # for i in range(1, self.m + 1):
        lossMatrix = jax.lax.fori_loop(
            1, self.m + 1,

            # for j in range(1, self.n + 1):
            lambda i, outerAccumulator: jax.lax.fori_loop(
                1, self.n + 1,

                # innerAccumulator is the updated loss matrix from one iteration of the inner loop.
                lambda j, innerAccumulator: loopBody(i, j, innerAccumulator),

                # outerAccumulator is the updated loss matrix from one full pass of the inner loop (i.e. one outer loop iteration)
                outerAccumulator
            ),
            lossMatrix
        )
        

        #stash = (distanceMatrix, lossMatrix, m, n, gamma)
        
        return lossMatrix
    
    ## wrapped()


    #@functools.partial(jax.jit, static_argnames=['m', 'n', 'gamma'])
    def computeLossMatrixFromDistanceMatrix_wrapped(self, distanceMatrix): #, m, n, gamma):
        @jax.custom_vjp
        def wrapSelf(distanceMatrix): #, m, n, gamma):
            fullMatrix = self.computeFullLossMAtrix(distanceMatrix)

            return fullMatrix[self.m, self.n]
        

        wrapSelf.defvjp(
            lambda distanceMatrix : self.computeLossMatrixFromDistanceMatrix_forwards     (distanceMatrix ),
            lambda stash, upstream: self.computeLossMatrixFromDistanceMatrix_backwardsPass(stash, upstream),
        )

        return wrapSelf(distanceMatrix)



    # _Think_ this is called "R" in the sleepwalking version, for Result.
    # This is the function that returns a cutom gradient. It has direct access to the intermediate calculations
    # through capture and as it is just handling tihs sequence, this allows better parallelisation and is also conceptually
    # easier to step through and debug.
    #@tf.custom_gradient
    #@jax.custom_gradient

    # NOT HERE
    # @jax.custom_vjp
    def computeLossMatrixFromDistanceMatrix_forwards(self, distanceMatrix): #, m, n, gamma):
        
        ##~~
        ##~~# Because of the semantics of custom_gradient, unfortunately the range loop below won;t work in the main function
        ##~~# We simply need to wrap the whole body in a tf.function - it is working eagerly or someething like that with the forward pass,
        ##~~# not wrapping it properly
        ##~~####@functools.partial(jax.jit, static_argnames=['m', 'n'])# NEed to specify that m and n are fixed and not inferred at runtime.
        ##~~
        ##~~#@functools.partial(jax.jit, static_argnames=['m', 'n', 'gamma'])
        ##~~def wrapped2():
        ##~~
        ##~~    lossMatrix = SDTWLoss.wrapped(distanceMatrix, m, n, gamma)
        ##~~
        ##~~    stash = (distanceMatrix, lossMatrix, m, n, gamma)
        ##~~
        ##~~    return stash, lossMatrix
        ##~~
        ##~~stash, lossMatrix = wrapped2()

        #return lossMatrix[m, n], backwardsPass

        lossMatrix = self.computeFullLossMAtrix(distanceMatrix)  #, m, n, gamma)
        stash = (distanceMatrix, lossMatrix) #, m, n, gamma)

        return lossMatrix[self.m, self.n], stash
    
    ## computeLossMatrixFromDistanceMatrix()

    def computeLossMatrixFromDistanceMatrix_backwardsPass(self, stash, upstream):
        distanceMatrix, lossMatrix = stash # , m, n, gamma = stash

        alignmentGradients = self.backwardsOneSequence(distanceMatrix, lossMatrix) #, m, n, gamma)
        gradients = jnp.multiply(upstream, alignmentGradients)

        # These are with reference ot the original arguments for the function above.
        return (gradients,)
    
    


    # One sequence in the batch.
    def backwardsOneSequence(self, distanceMatrix, lossMatrix): #, m, n, gamma):

        ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # The alignments array needs to be padded to be larger than the original distances matrix
        alignmentsShape = (self.m + 2, self.n + 2)
        

        # Called E in the paper
        # Set to zero in general
        alignmentsMatrix = jnp.zeros(alignmentsShape)

        # Set the bottom-right value as 1.
        alignmentsMatrix = alignmentsMatrix.at[self.m + 1, self.n + 1].set(1)


        ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Set the bottom and right edges of the **loss** matrix to be negative infinity
        # https://github.com/toinsson/pysdtw/blob/c902025cf8d8926fd4a85ea3620002be9b4715d7/pysdtw/sdtw_cpu.py#L91
        paddings = [[0, 1], [0, 1]]
        lossMatrix = jnp.pad(
            lossMatrix[0 : -1,   0 : -1],    # Take off the bottom and right edges
            paddings,
            constant_values = -jnp.inf
        )
        
        
        # Copy the bottom-right value into the -infinity padding in the _loss_ matrix.
        lossMatrix = lossMatrix.at[self.m + 1, self.n + 1].set(lossMatrix[self.m, self.n])
        
        
        ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


        
        # Pad with one row/column of zeros at the beginning and at the end
        # In order to match the loss matrix.
        # TODO: We could decrement the indices below, but, thisi s easier to see for debugging.
        paddings = [[1, 1], [1, 1]]
        paddedDistanceMatrix = jnp.pad(distanceMatrix, paddings, constant_values = 0)

        ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #for j in tf.range(m, 0, -1):
        #    for i in tf.range(n, 0, -1):
        # n then m? The torch versions assign the dimensions n then m, which is just confusing.
        # We use m then n.
        for j in range(self.n, 0, -1):
            for i in range(self.m, 0, -1):

                a =  lossMatrix[i + 1, j    ]   -   lossMatrix[i, j]   -   paddedDistanceMatrix[i + 1, j    ]
                b =  lossMatrix[i,     j + 1]   -   lossMatrix[i, j]   -   paddedDistanceMatrix[i,     j + 1]
                c =  lossMatrix[i + 1, j + 1]   -   lossMatrix[i, j]   -   paddedDistanceMatrix[i + 1, j + 1]

                a = jnp.exp(a / self.gamma)
                b = jnp.exp(b / self.gamma)
                c = jnp.exp(c / self.gamma)


                alignment = (

                        a * alignmentsMatrix[i + 1, j    ]
                     +  b * alignmentsMatrix[i,     j + 1]
                     +  c * alignmentsMatrix[i + 1, j + 1]
                )
                
                alignmentsMatrix = alignmentsMatrix.at[i, j].set(alignment)


        ## Andthen we need to remove the padding before returning
        unpadded = alignmentsMatrix[1:(self.n + 1), 1:(self.m + 1)]

        return unpadded
    
    ## backwardsOneSequence()
                




##~~
##~~SDTWLoss.computeLossMatrixFromDistanceMatrix_wrapped.defvjp(
##~~    SDTWLoss.computeLossMatrixFromDistanceMatrix_forwards,
##~~    SDTWLoss.computeLossMatrixFromDistanceMatrix_backwardsPass
##~~)




