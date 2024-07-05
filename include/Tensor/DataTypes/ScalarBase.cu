#include <cuda_runtime.h>

template <typename T, const bool _16 = false>
class FusionFloatBase {
public:
    T value_;

    // Constructor for scalar types
    template <typename scalar_t>
    __device__ FloatBase(scalar_t value) : value_(static_cast<T>(value)) {}

    // Constructor for float
    template <>
    __device__ FloatBase(float value) : value_(static_cast<T>(value)) {}

    // Constructor for the same type
    __device__ FloatBase(const T& value) : value_(value) {}

    // Arithmetic operators
    __device__ __forceinline__ FloatBase operator*(const FloatBase& a) const {
        return FloatBase(value_ * a.value_);
    }

    __device__ __forceinline__ FloatBase operator+(const FloatBase& a) const {
        return FloatBase(value_ + a.value_);
    }

    __device__ __forceinline__ FloatBase operator-(const FloatBase& a) const {
        return FloatBase(value_ - a.value_);
    }

    __device__ __forceinline__ FloatBase operator/(const FloatBase& a) const {
        return FloatBase(value_ / a.value_);
    }

    // Increment and decrement operators
    __device__ __forceinline__ void operator++() {
        value_ += static_cast<T>(1.0f);
    }

    __device__ __forceinline__ void operator--() {
        value_ -= static_cast<T>(1.0f);
    }

    // Compound assignment operators
    __device__ __forceinline__ void operator+=(const FloatBase& a) {
        value_ += a.value_;
    }

    __device__ __forceinline__ void operator-=(const FloatBase& a) {
        value_ -= a.value_;
    }

    __device__ __forceinline__ void operator*=(const FloatBase& a) {
        value_ *= a.value_;
    }

    __device__ __forceinline__ void operator/=(const FloatBase& a) {
        value_ /= a.value_;
    }

    // Atomic add operation
    __device__ __forceinline__ void atomicAdd(const FloatBase* address, const FloatBase& val) {
        ::atomicAdd(&address->value_, val.value_);
    }

    // Getter for value
    __device__ __forceinline__ T* value() {
        return &value_;
    }
};
