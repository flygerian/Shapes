# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/olabode/repos/Shapes/deps/OpenBLAS")
  file(MAKE_DIRECTORY "/home/olabode/repos/Shapes/deps/OpenBLAS")
endif()
file(MAKE_DIRECTORY
  "/home/olabode/repos/Shapes/build/openblas-build"
  "/home/olabode/repos/Shapes/build/openblas-install"
  "/home/olabode/repos/Shapes/build/openblas_ext-prefix/tmp"
  "/home/olabode/repos/Shapes/build/openblas_ext-prefix/src/openblas_ext-stamp"
  "/home/olabode/repos/Shapes/build/openblas_ext-prefix/src"
  "/home/olabode/repos/Shapes/build/openblas_ext-prefix/src/openblas_ext-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/olabode/repos/Shapes/build/openblas_ext-prefix/src/openblas_ext-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/olabode/repos/Shapes/build/openblas_ext-prefix/src/openblas_ext-stamp${cfgdir}") # cfgdir has leading slash
endif()
