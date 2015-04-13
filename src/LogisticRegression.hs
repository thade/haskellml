module LogisticRegression where
import Util
import GradientDescent
import Data.List
import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Util
import Numeric.GSL.Minimization

{- Trains multiple logistic regression classifiers and returns all
  the classifiers in a list of matrices [theta], where the i-th element
  corresponds to the classifier for label i -}
classifyOneVsAll :: Matrix Double -> Matrix Double -> Double -> Double -> Int -> ([Matrix Double],[Double])
classifyOneVsAll x y lambda precision max_iters =  (map (\ target -> trainClassifier x target lambda precision max_iters) targetMatrixList,targets)
  where
    (targetMatrix,targets) = mapOutputsVectorToTargetMatrix y
    targetMatrixList = map asColumn $ toColumns targetMatrix

{- Trains a logistic regression classifier, returning the learned theta -}
trainClassifier :: Matrix Double -> Matrix Double -> Double -> Double -> Int -> Matrix Double
trainClassifier x y lambda precision max_iters = asColumn $ fst $ minimizeVDGeneric VectorBFGS2 x y (zeros (cols x) 1) max_iters lambda precision

{- The cost function J
   Compute cost and gradient for logistic regression with regularization.
   Computes the cost of using theta as the parameter for regularized logistic regression and the
   gradient of the cost w.r.t. to the parameters.
   Returns double of cost and gradient -}
computeCost :: Matrix Double -> Matrix Double -> Matrix Double -> Double -> (Double, Matrix Double)
computeCost x y theta lambda = (cost, gradient)
    where
      m = fromIntegral $ rows y
      prediction = sigmoid $ x <> theta
      tempTheta = accum theta (*) [((0,0),0)]
      cost = - (1/m) * sumElements ( y * (log prediction) + (1 - y) * (log (1 - prediction)) ) + (lambda/(2*m)) * sumElements (tempTheta ^ 2)
      gradient = ctrans ((scalar (1/m)) * ( sumColumns  ((prediction - y) * x) ) ) + (scalar (lambda/m)) * tempTheta

{- Variants that only compute cost OR gradient (for min functions) -}
costOnly :: Matrix Double -> Matrix Double -> Matrix Double -> Double -> Double
costOnly x y theta lambda = - (1/m) * sumElements ( y * (log prediction) + (1 - y) * (log (1 - prediction)) ) + (lambda/(2*m)) * sumElements (tempTheta ^ 2)
  where
      m = fromIntegral $ rows y
      prediction = sigmoid $ x <> theta
      tempTheta = accum theta (*) [((0,0),0)]

gradientOnly :: Matrix Double -> Matrix Double -> Matrix Double -> Double -> Matrix Double
gradientOnly x y theta lambda = ctrans ((scalar (1/m)) * ( sumColumns  ((prediction - y) * x) ) ) + (scalar (lambda/m)) * tempTheta
  where
      m = fromIntegral $ rows y
      prediction = sigmoid $ x <> theta
      tempTheta = accum theta (*) [((0,0),0)]

-- Minimize with gradient descent
minimizeGD x y initial_theta alpha num_iters lambda = gradientDescent gradFunc initial_theta alpha num_iters
    where
      gradFunc t = snd (computeCost x y t lambda)

{-
Minimization methods:
ConjugateFR
ConjugatePR
VectorBFGS
VectorBFGS2
SteepestDescent
-}

minimizeVDGeneric method x y initial_theta num_iters lambda precision = minimizeVD method precision num_iters first_step tol costFunction gradientFunction (flatten initial_theta)
    where
      first_step = precision * 100
      tol = 0.1 -- see http://www.gnu.org/software/gsl/manual/html_node/Multimin-Algorithms-with-Derivatives.html
      costFunction thetaVect = costOnly x y (asColumn thetaVect) lambda
      gradientFunction thetaVect = flatten $ gradientOnly x y (asColumn thetaVect) lambda

-- Predict probability that the label is 0 or 1 using learned logistic
predict xi theta = sigmoid (xi <> theta)

-- Predict whether the label is 0 or 1 using learned logistic
predictThreshold :: Matrix Double -> Matrix Double -> Double -> Matrix Double
predictThreshold xi theta threshold = cond (predict xi theta) (scalar threshold) 0 1 1

-- Gauge the accuracy of a prediction
accuracy p y = mean $ toList $ flatten (cond p y 0 1 0) * 100

{- Predict the label for a trained one-vs-all classifier. The labels
   are in the range 1..K, where K = size(all_theta, 1).
   p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
   for each example in the matrix X. Note that X contains the examples in
   rows. all_theta is a matrix where the i-th row is a trained logistic
   regression theta vector for the i-th class. You should set p to a vector
   of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
   for 4 examples) -}
predictOneVsAll :: [Matrix Double] -> Matrix Double -> [Double] -> Matrix Double
predictOneVsAll all_theta x targetMapping = selectTarget
  where
    at_matrix = fromBlocks [all_theta]
    maxIndex xs = snd $ maximum $ zip xs targetMapping
    selectTarget = asColumn $ fromList $ map maxIndex $ toLists $ predict x at_matrix
