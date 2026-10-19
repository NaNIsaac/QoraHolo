#include "backends.hpp"
#include "cuda/util.cuh"

namespace qoraholo::cuda {

bool device_present() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

}  // namespace qoraholo::cuda
