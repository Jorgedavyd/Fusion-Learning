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

template <typename T, unsigned int align_, typename Args>
class alignas(align_) FusionFloatBase {
    static_assert(align_ > 0 && (align_ & (align_ -1)) == 0, "Alignement must be a positive power of 2");
public:
    __device__ FusionFloatBase (T&& value) : _data(value) {};

    __device__ FusionFloatBase (T& value) : _data(value) {};

    __device__ FusionFloatBase (float&& value, ...) : _data(T(value, args)) {};

    __device__ FusionFloatBase (FusionFloatBase<T, align_>&& value) noexcept = delete;

    __device__ FusionFloatBase (FusionFloatBase<T, align_>& value) noexcept = delete;

    __device__ __forceinline__ FusionFloatBase<T, align_>& operator=(FusionFloatBase<T, align_>&& a) const noexcept = delete;

    __device__ __forceinline__ FusionFloatBase<T, align_>& operator=(FusionFloatBase<T, align_>& a) const noexcept = delete;

    virtual __device__ __forceinline__ FusionFloatBase<T, align_> operator*(const FusionFloatBase<T, align_>& a) const {
        printf("CUDA error with code:", NOT_IMPLEMENTED_ERROR);
    }

    virtual __device__ __forceinline__ FusionFloatBase<T, align_> operator+(const FusionFloatBase<T, align_>& a) const {
        printf("CUDA error with code:", NOT_IMPLEMENTED_ERROR);
    }

    virtual __device__ __forceinline__ FusionFloatBase<T, align_> operator-(const FusionFloatBase<T, align_>& a) const {
        printf("CUDA error with code:", NOT_IMPLEMENTED_ERROR);
    }

    virtual __device__ __forceinline__ FusionFloatBase<T, align_> operator/(const FusionFloatBase<T, align_>& a) const {
        printf("CUDA error with code:", NOT_IMPLEMENTED_ERROR);
    }

    __device__ __forceinline__ void operator++() {
        ++_data;
    }

    __device__ __forceinline__ void operator--() {
        --_data;
    }

    __device__ __forceinline__ void operator+=(FusionFloatBase<T, align_>&& a) {
        _data += a;
    }

    __device__ __forceinline__ void operator-=(FusionFloatBase<T, align_>&& a) {
        _data -= a;
    }

    __device__ __forceinline__ void operator*=(FusionFloatBase<T, align_>&& a) {
        _data *= a;
    }

    __device__ __forceinline__ void operator/=(FusionFloatBase<T, align_>&& a) {
        _data /= a;
    }

    __device__ __forceinline__ void atomicAdd(const FusionFloatBase<T, align_>* address, const FusionFloatBase<T, align_>& val) {
        atomicAdd(address, val);
    }

    __device__ T* get_value (void) const {
        return &_data;
    };

    virtual ~FusionFloatBase (void) = default;
private:
    T _data;
};
