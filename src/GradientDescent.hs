module GradientDescent where
import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Util

--  Performs gradient descent to learn theta
--  theta = GRADIENTDESCENT(x, y, theta, alpha, num_iters) updates theta by
--  taking num_iters gradient steps with learning rate alpha
gradientDescent :: (Matrix Double -> Matrix Double) -> Matrix Double -> Double -> Int -> Matrix Double
gradientDescent gradFunc initialTheta alpha num_iters = iterate initialTheta num_iters
            where
                iterate theta 0 = theta
                iterate theta iters = iterate (updateTheta theta) (iters-1)
                updateTheta t = t - (scalar alpha) * (gradFunc t)
