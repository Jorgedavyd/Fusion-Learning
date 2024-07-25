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
#include "float.h"
#include <cuda_fp8.h>
/*
Implementation of float8 single, dual_vector, quad_vector
*/
class SinglePrecisionFusionFloat : private FusionFloatBase<__nv_fp8_storage_t, 1> {
public:
    __device__ __forceinline__ SinglePrecisionFusionFloat operator*(const SinglePrecisionFusionFloat& a) const {
        return SinglePrecisionFusionFloat<T>(__hmul(_data, a.get_value()));
    }

    __device__ __forceinline__ SinglePrecisionFusionFloat operator+(const SinglePrecisionFusionFloat& a) const {
        return SinglePrecisionFusionFloat<T>(__hadd(_data, a.get_value()));
    }

    __device__ __forceinline__ SinglePrecisionFusionFloat operator-(const SinglePrecisionFusionFloat& a) const {
        return SinglePrecisionFusionFloat<T>(__hsub(_data, a.get_value()));
    }

    __device__ __forceinline__ SinglePrecisionFusionFloat operator/(const SinglePrecisionFusionFloat& a) const {
        return SinglePrecisionFusionFloat<T>(__hdiv(_data, a.get_value()));
    }
};
