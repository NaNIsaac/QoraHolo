#pragma once

#include "qoraholo/array.hpp"
#include "qoraholo/backend.hpp"

namespace qoraholo {

// Unweighted least-squares phase unwrapping (Ghiglia & Romero 1994): the Poisson equation for
// the wrapped phase gradients with Neumann boundaries, solved with DCT-II computed via FFT.
// Each image of the stack is unwrapped separately; image sides must be even. The free constant
// is chosen so that the result is congruent to the input modulo 2 pi.
[[nodiscard]] Array3<float> unwrap(const Array3<float>& wrapped, Backend backend = Backend::cpu);

// Complex Rytov phase of a field ratio u / u_in: log|r| + i * unwrap(arg r). The remaining
// 2 pi multiple is fixed by putting the mean phase on the image border in (-pi, pi], so the
// border must be background.
[[nodiscard]] Array3<cfloat> rytov(const Array3<cfloat>& ratio, Backend backend = Backend::cpu);

}  // namespace qoraholo
