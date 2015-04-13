--{-# LANGUAGE BangPatterns #-}
module NeuralNetworks where
import Util
import Data.List
import System.Random
import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Util
import Numeric.GSL.Minimization
import Control.Parallel (par,pseq)
import Debug.Trace
import System.IO
import System.Directory

readThetaList :: [Int] -> IO [Matrix Double]
readThetaList layers = do
                          let readThetaListInternal fname = do
                                                              contents <- readFile fname
                                                              let thetas = contents `seq` read contents :: [Matrix Double]
                                                              return thetas
                          let generateThetaList = do
                                                    rgen <- getStdGen
                                                    return $ randomInitializeAllWeights rgen layers
                          let fileName = show layers
                          fexist <- doesFileExist fileName
                          if fexist
                          then
                            readThetaListInternal fileName
                          else
                            generateThetaList

writeThetaList :: [Int] -> [Matrix Double] -> IO ()
writeThetaList layers thetas = do
                                let filename = show layers
                                let contents = show thetas
                                writeFile filename contents

{- RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
   incoming connections and L_out outgoing connections
   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights
   of a layer with L_in incoming connections and L_out outgoing
   connections.
   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
   the first row of W handles the "bias" terms -}
randomInitializeWeights :: RandomGen g => g -> Int -> Int -> Matrix Double
randomInitializeWeights gen l_in l_out = randommatrix * 2 * epsilon_init - epsilon_init
     where
      randomlist = randoms gen :: [Double]
      epsilon_init = 0.12
      randommatrix = (l_out><(1+l_in)) randomlist

randomInitializeAllWeights :: RandomGen g => g -> [Int] -> [Matrix Double]
randomInitializeAllWeights gen (outputs:[]) = []
randomInitializeAllWeights gen (inputs:outputs:rest) = (randomInitializeWeights gen inputs outputs):(randomInitializeAllWeights gen (outputs:rest))

{- Implements the neural network cost function for a neural network which performs
   classification.
   [J, grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
   X, y, lambda) computes the cost and gradient of the neural network. The
   parameters for the neural network are "unrolled" into the vector
   nn_params and need to be converted back into the weight matrices.
   The returned parameter grad should be a "unrolled" vector of the
   partial derivatives of the neural network.

   NOTE: cost does not work on NNs with no hidden layers.  TODO fix this if
   I ever need it.

   -}
costFunction :: Vector Double -> [Int] -> Matrix Double -> Matrix Double -> Double -> (Double, Vector Double)
costFunction thetaVector layers x targetYsMatrix lambda = (j_reg, gradientVector)
    where
      m = fromIntegral $ rows targetYsMatrix
      thetaList = reshapeVector layers thetaVector
      (inputMatrices, activationMatrices, output_layer) = forwardPropagate x thetaList
      --j = trace (show $ -((1/m) * sumElements (targetYsMatrix * (log output_layer) + (1 - targetYsMatrix)  * (log (1 - output_layer))))) (-((1/m) * ( sumElements (targetYsMatrix * (log output_layer) + (1 - targetYsMatrix) * (log (1 - output_layer))))))
      --j = trace (show $ ((targetYsMatrix * (log output_layer) + (1 - targetYsMatrix) * (log (1 - output_layer))))) (-((1/m) * ( sumElements (targetYsMatrix * (log output_layer) + (1 - targetYsMatrix) * (log (1 - output_layer))))))
      j = -((1/m) * ( sumElements (targetYsMatrix * (log output_layer) + (1 - targetYsMatrix) * (log (1 - output_layer)))))
      j_reg = j + (lambda/(2*m)) * ( sum ( map (\ theta -> sumElements ((dropColumns 1 theta) ^ 2)) thetaList ))
      deltaList = backPropagate thetaList inputMatrices activationMatrices output_layer targetYsMatrix
      gradientVector = createGradientVector thetaList deltaList
        where
          -- create list of gradient matrices and then flatten them into a single vector
          createGradientVector :: [Matrix Double] -> [Matrix Double] -> Vector Double
          createGradientVector tlist dlist = flattenConcat $ iterate tlist dlist
            where
              iterate [] []         = []
              iterate (t:ts) (d:ds) = grad:(iterate ts ds)
                where
                  -- liminate j = 0 column from theta for purposes of regularization
                  tmpTheta = padLeft (dropColumns 1 t) 0
                  -- regularized gradients
                  grad =  (1/(scalar m)) * d + (scalar (lambda/m)) * tmpTheta

-- |Performs forward propagation on a neural network, returning the resulting output layer
forwardPropagate :: Matrix Double -> [Matrix Double] -> ([Matrix Double],[Matrix Double],Matrix Double)
--forwardPropagate inputMatrix _ | trace ("Forward propagate input matrix dimensions " ++ (show (rows inputMatrix)) ++ "x" ++ (show (cols inputMatrix)) ) False = undefined
forwardPropagate inputMatrix thetaList = (inputMatrices, activationMatrices, output_layer)
  where
    resultList' = doForwardProp inputMatrix thetaList
    --add x to the start of the input list, for both input and activation... testing
    resultList = (inputMatrix,inputMatrix):resultList'
    (inputMatrices, activationMatrices) = unzip resultList
    output_layer = last activationMatrices
    doForwardProp input []         = []
    doForwardProp input (t:thetas) = (initial_value,activation):(doForwardProp activation thetas)
      where
        input_with_bias = addBias input
        initial_value = input_with_bias <> (ctrans t)
        activation = sigmoid initial_value

-- |Performs backpropagation on a neural network, returning the resulting Delta matrices
backPropagate :: [Matrix Double] -> [Matrix Double] -> [Matrix Double] -> Matrix Double -> Matrix Double -> [Matrix Double]
backPropagate thetaList inputMatrices activationMatrices output_layer targetYs = output_deltas `par` doBackProp 0 deltaList
  where
    deltaList = makeZeroDeltaList thetaList
    m = rows targetYs
    -- reverse the lists we'll be iterating through, since we're going backwards from the output layer
    revInputs = reverse inputMatrices
    revActivations = reverse activationMatrices
    revThetaList = reverse thetaList
    output_deltas = (head revActivations) - targetYs
    --for each example in training set:
    doBackProp index ds | index >= m = ds
                        | otherwise  = doBackProp (index +1) updatedDs
      where
        -- reverse the lists we'll be iterating through, since we're going backwards from the output layer
        revDs = reverse ds
        --do output layer separately:
        output_delta = ctrans $ extractRows [index] output_deltas
        --do the rest of the layers:
        updatedDs = reverse $ backPropagateOneExample revThetaList output_delta (tail revInputs) (tail revActivations) revDs
        backPropagateOneExample _ _ _ _ [] = []
        backPropagateOneExample (thetaI:thetas) previous_delta (inputI:inputs) (activationI:activations) (deltaI:deltas) = current_delta `pseq` updatedDelta:(backPropagateOneExample thetas current_delta inputs activations deltas)
          where
            current_delta_with_bias = (ctrans thetaI) <> previous_delta * (sigmoidGradient (ctrans (addBias (extractRows [index] inputI))))
            current_delta = updatedDelta `par` dropRows 1 current_delta_with_bias
            -- capital Delta accumulation
            updatedDelta = deltaI + previous_delta <> addBias (extractRows [index] activationI)

-- |Makes a list of 0-filled Delta matrices from a list of theta matrices
makeZeroDeltaList :: [Matrix Double] -> [Matrix Double]
makeZeroDeltaList [] = []
makeZeroDeltaList (m:ms) = (zeros (rows m) (cols m)):(makeZeroDeltaList ms)

-- |Reshapes a vector into a list of weight matrices for a network of the given number and size of layers
reshapeVector :: [Int] -> Vector Double -> [Matrix Double]
reshapeVector layers vector = iterate layers (toList vector)
  where
    iterate _ [] = []
    iterate [] _ = []
    iterate (x:[]) _ = []
    iterate (l_in:l_out:xs) vlist = m:(iterate (l_out:xs) (drop msize vlist))
      where
        msize = l_out * (l_in + 1)
        m = reshape (l_in + 1) (fromList (take msize vlist))

-- |The prediction function
predict :: [Matrix Double] -> Matrix Double -> [Double] -> Matrix Double
predict [] input targetMapping = selectTarget
  where
    maxIndex xs = snd $ maximum $ zip xs targetMapping
    selectTarget = asColumn $ fromList $ map maxIndex $ toLists input
predict (theta:thetas) input targetMapping = predict thetas layer targetMapping
  where
    layer = sigmoid $ (addBias input) <> (ctrans theta)

{-
Minimization methods:
ConjugateFR
ConjugatePR
VectorBFGS
VectorBFGS2
SteepestDescent
-}

minimizeVDGeneric method x targetYsMatrix layers num_iters lambda precision initial_thetas = minimizeVD method precision num_iters first_step tol traceCostFunc gradientFunc (flattenConcat initial_thetas)
    where
      first_step = precision * 100
      tol = 0.1 -- see http://www.gnu.org/software/gsl/manual/html_node/Multimin-Algorithms-with-Derivatives.html
      costFunc thetaVect = fst $ costGradFunction thetaVect
      gradientFunc thetaVect = snd $ costGradFunction thetaVect
      costGradFunction t = (costFunction t layers x targetYsMatrix lambda)
      traceGradientFunc thetaVect = traceShow ((sumElements grad)) grad
        where grad = snd $ costGradFunction thetaVect
      traceCostFunc thetaVect = traceShow cost cost
        where cost = fst $ costGradFunction thetaVect
      --memoizedCostGradFunction = unsafeMemoize costGradFunction

-- Generate learning curves
learningCurvesNeuralNetwork::
  RandomGen g =>   MinimizeMethodD ->
                   Matrix Double ->
                   Matrix Double ->
                   Matrix Double ->
                   Matrix Double ->
                   [Int] ->
                   Double ->
                   Int ->
                   Double ->
                   g ->
                   Int ->
                   [(Double,Double,Int)]
learningCurvesNeuralNetwork method x y xval yval layers lambda iter precision gen step_size = learningCurves trainFunc costFunc x y xval yval step_size
  where
    initial_thetas = randomInitializeAllWeights gen layers
    trainFunc tx ty = asColumn $ fst $ minimizeVDGeneric method tx ty layers iter lambda precision initial_thetas
    costFunc tx ty tt = fst $ costFunction (flatten tt) layers tx ty  0

