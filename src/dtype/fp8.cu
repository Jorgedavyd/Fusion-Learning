/* Features of Fusion floats:
Always parse by copy on instance for broadcasting?.
Callable just on device.
Move not allowed.
Copy allowed.
Ensure coalescing patterns
*/

#include <cassert>
#include <cstdio>
#include "Fusion/errorMacros.h"
#include "floatBase.h"
#include <cuda_fp8.h>

template <typename T, unsigned int align_ = 1>
class SPBase : private FusionFloatBase<T, align_> {
public:
    __device__ __forceinline__ SPBase<T, align_> operator*(SPBase<T, align_>&& a) const {
        return SPBase<T, align_>(_data * a.get_value());
    }

    __device__ __forceinline__ SPBase<T, align_> operator+(SPBase<T, align_>&& a) const {
        return SPBase<T, align_>(_data + a.get_value());
    }

    __device__ __forceinline__ SPBase<T, align_> operator-(SPBase<T, align_>&& a) const {
        return SPBase<T, align_>(_data - a.get_value());
    }

    __device__ __forceinline__ SPBase<T, align_> operator/(SPBase<T, align_>&& a) const {
        return SPBase<T, align_>(_data / a.get_value());
    }
};

using fp8_e4m3 = SPBase<__nv_fp8_e4m3, 1>;
using fp8_e5m2 = SPBase<__nv_fp8_e5m2, 1>;
using fp8_x2_e4m3 = SPBase<__nv_fp8_e4m3, 2>;
using fp8_x2_e5m2 = SPBase<__nv_fp8_e5m2, 2>;
using fp8_x4_e4m3 = SPBase<__nv_fp8_e4m3, 4>;
using fp8_x4_e5m2 = SPBase<__nv_fp8_e5m2, 4>;
