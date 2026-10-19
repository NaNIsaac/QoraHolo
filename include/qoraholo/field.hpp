#pragma once

#include <array>
#include <optional>

#include "qoraholo/array.hpp"
#include "qoraholo/backend.hpp"

namespace qoraholo {

// Peak of the +1 sideband, as an offset from the DC pixel of the centred spectrum, in pixels.
// "+1" is the peak in the half plane x > 0 (or x == 0, y > 0).
struct Sideband {
  int iy = 0;
  int ix = 0;
  float y = 0;  // sub-pixel refined
  float x = 0;
  float magnitude = 0;
};

struct FieldOptions {
  // Sideband centre {y, x} in pixels from DC. Empty: detected per hologram. For multi-angle
  // data pass the reference carrier so that the illumination tilt stays in the field.
  std::optional<std::array<float, 2>> center;
  float radius = 0;        // crop radius in spectrum pixels; <= 0: |center| / 3
  float dc_exclusion = 0;  // radius around DC skipped by the peak search; <= 0: N / 8
  int output_size = 0;     // side of the returned field; 0: same as the hologram
};

// Holograms are square with an even side. Searches the first hologram of the stack.
[[nodiscard]] Sideband find_sideband(const Array3<float>& holograms, float dc_exclusion = 0);

// Off-axis hologram -> complex field: FFT, circular crop around the sideband, shift of the
// sideband centre to DC, inverse FFT at output_size. Field pixel = hologram pixel * N /
// output_size.
[[nodiscard]] Array3<cfloat> extract_fields(const Array3<float>& holograms,
                                            const FieldOptions& options = {},
                                            Backend backend = Backend::cpu);

// sample / background, pixelwise. background holds one field or one per sample field.
[[nodiscard]] Array3<cfloat> field_ratio(const Array3<cfloat>& sample,
                                         const Array3<cfloat>& background);

}  // namespace qoraholo
