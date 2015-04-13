module Paths_haskellml (
    version,
    getBinDir, getLibDir, getDataDir, getLibexecDir,
    getDataFileName, getSysconfDir
  ) where

import qualified Control.Exception as Exception
import Data.Version (Version(..))
import System.Environment (getEnv)
import Prelude

catchIO :: IO a -> (Exception.IOException -> IO a) -> IO a
catchIO = Exception.catch

version :: Version
version = Version [0,1,0,0] []
bindir, libdir, datadir, libexecdir, sysconfdir :: FilePath

bindir     = "/home/thade/code/haskellml/.cabal-sandbox/bin"
libdir     = "/home/thade/code/haskellml/.cabal-sandbox/lib/i386-linux-ghc-7.8.3/haskellml-0.1.0.0"
datadir    = "/home/thade/code/haskellml/.cabal-sandbox/share/i386-linux-ghc-7.8.3/haskellml-0.1.0.0"
libexecdir = "/home/thade/code/haskellml/.cabal-sandbox/libexec"
sysconfdir = "/home/thade/code/haskellml/.cabal-sandbox/etc"

getBinDir, getLibDir, getDataDir, getLibexecDir, getSysconfDir :: IO FilePath
getBinDir = catchIO (getEnv "haskellml_bindir") (\_ -> return bindir)
getLibDir = catchIO (getEnv "haskellml_libdir") (\_ -> return libdir)
getDataDir = catchIO (getEnv "haskellml_datadir") (\_ -> return datadir)
getLibexecDir = catchIO (getEnv "haskellml_libexecdir") (\_ -> return libexecdir)
getSysconfDir = catchIO (getEnv "haskellml_sysconfdir") (\_ -> return sysconfdir)

getDataFileName :: FilePath -> IO FilePath
getDataFileName name = do
  dir <- getDataDir
  return (dir ++ "/" ++ name)
