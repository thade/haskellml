module LinearRegression where
import Util
import GradientDescent
import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Util
import Debug.Trace

-- The cost function J
-- Compute regularized cost for linear regression with multiple variables
-- computes the cost of using theta as the parameter for linear 
-- regression to fit the data points in X and y, returns double of cost and gradient
computeCost :: Matrix Double -> Matrix Double -> Matrix Double -> Double -> (Double, Matrix Double)
computeCost x y theta lambda = (cost, gradient)
    where
      m = fromIntegral $ rows y
      tempTheta = accum theta (*) [((0,0),0)]
      cost = ( 1/(2*m) ) * sumElements (((x <> theta) - y) ^2 ) + (lambda/(2*m)) * sumElements (tempTheta ^ 2)
      gradient = ctrans $ (scalar (1/m)) * ( sumColumns  ((x <> theta - y) * x) ) + (scalar (lambda/m)) * (ctrans tempTheta)


linearRegressionGD x y initial_theta alpha num_iters lambda = gradientDescent gradFunc initial_theta alpha num_iters
    where
      gradFunc t = snd (computeCost x y t lambda)

-- The prediction function
predict xi theta = xi <> theta

-- Generate learning curves
learningCurvesLinearRegression :: 
                   Matrix Double ->
                   Matrix Double ->
                   Matrix Double ->
                   Matrix Double ->
                   Double ->
                   Double ->
                   Int ->
                   [(Double,Double,Int)]
learningCurvesLinearRegression x y xval yval lambda alpha iter = learningCurves trainFunc costFunc x y xval yval 1
  where
    initial_theta = ((cols x)><1) (repeat 1)
    trainFunc tx ty = linearRegressionGD tx ty initial_theta alpha iter lambda
    costFunc tx ty tt = fst $ computeCost tx ty tt 0
