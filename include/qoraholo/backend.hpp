#pragma once

namespace qoraholo {

enum class Backend { cpu, cuda };

// True when the library was built with CUDA and a device is visible.
[[nodiscard]] bool cuda_available();

}  // namespace qoraholo
