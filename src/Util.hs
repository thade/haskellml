{-# LANGUAGE BangPatterns #-}
module Util where
import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Util
import Data.List
import Data.Ord (comparing)
import Control.Applicative
import System.Random
import System.IO.Unsafe
import Data.IORef
import qualified Data.Map as Map

partitionList :: [Double] -> [a] -> [[a]]
partitionList ps xs = iterate (proportional ps) xs
  where
    sumP = sum ps
    lenX = length xs
    proportional = map (/sumP)
    iterate [] leftover = [leftover]
    iterate (p:rest) li = (take proportion li):iterate rest (drop proportion li)
      where 
        proportion = floor $ p * (fromIntegral lenX)

{- shuffle the input and partition into training, test and validation sets in the
   given proportions 
   -}
shuffleAndPartitionInput :: RandomGen g => g -> (Matrix Double,Matrix Double) -> 
                                           (Double,Double,Double) -> 
                                           ((Matrix Double,Matrix Double), 
                                            (Matrix Double,Matrix Double), 
                                            (Matrix Double,Matrix Double))
shuffleAndPartitionInput gen (inputX,inputY) (trainProp,valProp,testProp) = (train,val,test)
  where
   m = rows inputY 
   (trainIndices:valIndices:testIndices:rest) = partitionList [trainProp,valProp,testProp] (shuffle gen m [0..])
   train   = get trainIndices
   val     = get valIndices
   test    = get testIndices
   get ind = (extractRows ind inputX,extractRows ind inputY)

-- |shuffle a list
shuffle :: RandomGen g => g -> Int -> [a] -> [a]
shuffle gen 0   _  = []
shuffle gen len xs = 
        let
                n = fst $ randomR (0, len - 1) gen
                (y, ys) =  choose n xs
                ys' = shuffle gen (len - 1) ys
        in y:ys'

choose _ [] = error "choose: index out of range"
choose 0 (x:xs) = (x, xs)
choose i (x:xs) = let (y, ys) = choose (i - 1) xs in (y, x:ys)  

-- | choose a random sample from x and y 
randomSample :: RandomGen g => g -> (Matrix Double,Matrix Double) -> Int -> (Matrix Double, Matrix Double)
randomSample gen (x,y) size = (newX,newY)
  where
    m = rows y
    indices = sort $ uniqueRandoms size m gen
    newX = extractRows indices x
    newY = extractRows indices y

uniqueRandoms :: RandomGen g => Int -> Int -> g -> [Int] 
uniqueRandoms size max gen | max < size = error "size greater than max"
                           | otherwise = take size $ shuffle gen max [0..(max-1)]



-- fatureNormalize(X)
-- Normalizes the features in X
-- FEATURENORMALIZE(X) returns a normalized version of X where
-- the mean value of each feature is 0 and the standard deviation
-- is 1. This is often a good preprocessing step to do when
-- working with learning algorithms.
featureNormalize :: Matrix Double -> (Matrix Double, Matrix Double, Matrix Double)
featureNormalize x = ((doNormalize x mu sigma), mu, sigma)
                      where
                        mu = fromRows $ [fromList $ map mean $ map toList $ toColumns x]
                        sigma = fromRows $ [fromList $ map stddev $ map toList $ toColumns x]

-- normalize an input matrix given sigma and mu
doNormalize :: Matrix Double -> Matrix Double -> Matrix Double -> Matrix Double
doNormalize x mu sigma = (x - mu) / sigma


-- split a list into words by a function
wordsBy :: (a -> Bool) -> [a] -> [[a]]
wordsBy p s = case dropWhile p s of
    []      -> []
    s':rest -> (s':w) : wordsBy p (drop 1 s'')
        where (w, s'') = break p rest

-- read a csv file of numbers into a matrix
readMatrixCSV :: FilePath -> IO (Matrix Double)
readMatrixCSV path = do
    s <- map (wordsBy (\ x -> x == ',' )) <$> lines <$> readFile path
    let d = map (map read) $ s :: [[Double]]
    let cols = length $ head d
    return $ reshape cols $ fromList (concat d)

-- sum up the rows in a matrix, returning a column vector of sums
sumRows m = fromColumns $ [ sum $ toColumns m ]

-- sum up the columns in a matrix, returning a row vector of sums
sumColumns m = fromRows $ [ sum $ toRows m ]

-- pad a matrix on the left with a value
padLeft :: Matrix Double -> Double -> Matrix Double
padLeft !m num = fromBlocks [[scalar num, m]]

-- convenience function similar to above
addBias !m = fromBlocks [[1,m]]

-- |Numerically stable mean
mean :: Floating a => [a] -> a
mean x = fst $ foldl' (\(!m, !n) x -> (m+(x-m)/(n+1),n+1)) (0,0) x

-- |Same as 'mean' 
average :: Floating a => [a] -> a
average = mean

-- |Median
median :: (Floating a, Ord a) => [a] -> a
median x | odd n  = head  $ drop (n `div` 2) x'
         | even n = mean $ take 2 $ drop i x'
                  where i = (length x' `div` 2) - 1
                        x' = sort x
                        n  = length x
-- |Range
range :: (Num a, Ord a) => [a] -> a
range xs = maximum xs - minimum xs

-- |Standard deviation of sample
stddev :: (Floating a) => [a] -> a
stddev xs = sqrt $ var xs

-- |Sample variance
var xs = (var' 0 0 0 xs) / (fromIntegral $ length xs - 1)
    where
      var' _ _ s [] = s
      var' m n s (x:xs) = var' nm (n + 1) (s + delta * (x - nm)) xs
         where
           delta = x - m
           nm = m + delta/(fromIntegral $ n + 1)

-- |Replace items in a list
replace :: Eq a => a -> a -> [a] -> [a]
replace _ _ [] = []
replace target replacement (x:xs) | target == x = replacement:(replace target replacement xs)
                                  | otherwise   = x:(replace target replacement xs) 

-- The Sigmoid function
sigmoid x = 1.0 / (1 + exp (-x))

sigmoidGradient z = (sigmoid z) * (1 - (sigmoid z))

-- |Find the maximum value of each row and return a column vector of their indices
rowMaxIndex :: Matrix Double -> Matrix Double
rowMaxIndex m = asColumn $ fromList $ map maxIndex $ toLists m
  where
    maxIndex xs = snd $ maximum $ zip xs [0,1..]

-- |Turn a list of matrices into a vector
flattenConcat ms = fromList $ concat $ map (toList . flatten) ms

-- |Memoizes a function with unsafePerformIO
{-# NOINLINE unsafeMemoize #-}
unsafeMemoize :: Ord a => (a -> b) -> (a -> b)
unsafeMemoize f = unsafePerformIO $ do 
    r <- newIORef Map.empty
    return $ \ x -> unsafePerformIO $ do 
        m <- readIORef r
        case Map.lookup x m of
            Just y  -> return y
            Nothing -> do 
                    let y = f x
                    writeIORef r (Map.insert x y m)
                    return y

{- Generates the train and cross validation set errors needed
  to plot a learning curve
   [error_train, error_val] = ...
       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
       cross validation set errors for a learning curve. In particular,
       it returns two vectors of the same length - error_train and
       error_val. Then, error_train(i) contains the training error for
       i examples (and similarly for error_val(i)).

   In this function, you will compute the train and test errors for
   dataset sizes from 1 up to m. In practice, when working with larger
   datasets, you might want to do this in larger intervals.
-}

learningCurves :: (Matrix Double -> Matrix Double -> Matrix Double) -> 
                  (Matrix Double -> Matrix Double -> Matrix Double -> Double) -> 
                   Matrix Double -> 
                   Matrix Double -> 
                   Matrix Double -> 
                   Matrix Double -> 
                   Int ->
                   [(Double,Double,Int)]
learningCurves train cost x y xval yval step_size = iterate 1
  where
    m = rows y
    iterate i | i >= m    = []
              | otherwise = (train_cost,cval_cost,i):iterate (i+step_size)
      where
        x' = takeRows i x
        y' = takeRows i y
        theta = train x' y'
        train_cost = cost x' y' theta
        cval_cost = cost xval yval theta

{- POLYFEATURES Maps X (1D vector) into the p-th power
   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
   maps each example into its polynomial features where
   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
-}
oneDPolyFeatures :: Matrix Double -> Int -> Matrix Double
oneDPolyFeatures x p = fromColumns $ iterate 1
  where
    xVect = (toColumns x) !! 0
    iterate i | i > p     = []
              | otherwise = (xVect ^ i):iterate (i+1)

{- VALIDATIONCURVE Generate the train and validation errors needed to
   plot a validation curve that we can use to select lambda
   [lambda_vec, error_train, error_val] = ...
   VALIDATIONCURVE(X, y, Xval, yval) returns the train
   and validation errors (in error_train, error_val)
   for different values of lambda. You are given the training set (X,
   y) and validation set (Xval, yval).
-}
validationCurve :: (Double -> Matrix Double) -> (Matrix Double -> Double) -> (Matrix Double -> Double) -> [Double] -> [(Double,Double)]
validationCurve trainF trainErrorF valErrorF candidates = iterate candidates
  where
    iterate [] = []
    iterate (c:cs) = (trainEr,valEr):(iterate cs)
      where 
        theta = trainF c
        trainEr = trainErrorF theta
        valEr = valErrorF theta

{- mapOutputsVectorToTargetMatrix
   Takes an m-dimension column vector of outputs in space {0:(k-1)} and maps it to an m><k target 
   matrix in space {0,1} where each row represents an output and a 1 is used to indicate which
   class (column) an output is in.  Eg if there are 3 output classes and the input vector looks
   like this:

   [ 0,
     0,
     1,
     2 ]
  
  The output matrix will look like this:

  [ 1, 0, 0;
    1, 0, 0;
    0, 1, 0;
    0, 0, 1 ]
-}
mapOutputsVectorToTargetMatrix y = (targetYsMatrix,targets)
  where
    targets = nub $ toList $ flatten y
    targetYsList = map (\t -> cond y t 0 1 0) (map scalar targets)
    targetYsMatrix = fromBlocks [targetYsList]
