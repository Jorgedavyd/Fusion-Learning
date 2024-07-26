/* Features of Fusion floats:
Always parse by copy on instance for broadcasting?.
Callable just on device.
Move not allowed.
Copy allowed.
Ensure coalescing patterns
*/

#include <cassert>
#include <cstdio>
#include "errMacros.h"
#include "floatBase.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
class HalfPrecisionFusionFloat : private FusionFloatBase<T, 2> {
public:
    __device__ __forceinline__ HalfPrecisionFusionFloat operator*(HalfPrecisionFusionFloat&& a) const {
        return HalfPrecisionFusionFloat<T>(__hmul(_data, a.get_value()));
    }

    __device__ __forceinline__ HalfPrecisionFusionFloat operator+(HalfPrecisionFusionFloat&& a) const {
        return HalfPrecisionFusionFloat<T>(__hadd(_data, a.get_value()));
    }

    __device__ __forceinline__ HalfPrecisionFusionFloat operator-(HalfPrecisionFusionFloat&& a) const {
        return HalfPrecisionFusionFloat<T>(__hsub(_data, a.get_value()));
    }

    __device__ __forceinline__ HalfPrecisionFusionFloat operator/(HalfPrecisionFusionFloat&& a) const {
        return HalfPrecisionFusionFloat<T>(__hdiv(_data, a.get_value()));
    }
};

template <typename T>
class Half2PrecisionFusionFloat : private FusionFloatBase<T, 4> {
public:
    __device__ __forceinline__ Half2PrecisionFusionFloat operator*(Half2PrecisionFusionFloat&& a) const {
        return Half2PrecisionFusionFloat<T>(__hmul2(_data, a.get_value()));
    }

    __device__ __forceinline__ Half2PrecisionFusionFloat operator+(Half2PrecisionFusionFloat&& a) const {
        return Half2PrecisionFusionFloat<T>(__hadd2(_data, a.get_value()));
    }

    __device__ __forceinline__ Half2PrecisionFusionFloat operator-(Half2PrecisionFusionFloat&& a) const {
        return Half2PrecisionFusionFloat<T>(__hsub2(_data, a.get_value()));
    }

    __device__ __forceinline__ Half2PrecisionFusionFloat operator/(Half2PrecisionFusionFloat&& a) const {
        return Half2PrecisionFusionFloat<T>(__h2div(_data, a.get_value()));
    }
};

using fp16 = HalfPrecisionFusionFloat<__half> ;
using bf16 = HalfPrecisionFusionFloat<__nv_bfloat16>;
using fp16_2 = HalfPrecisionFusionFloat<__half2> ;
using bf16_2 = HalfPrecisionFusionFloat<__nv_bfloat162>;

