#include <cuda.h>

template <typename T, const bool _16 = false>
class FusionFloatBase {
public:
    T value_;

    // Constructor for scalar types
    template <typename scalar_t>
    __device__ FusionFloatBase(scalar_t value) : value_(static_cast<T>(value)) {}

    // Constructor for float
    template <>
    __device__ FusionFloatBase(float value) : value_(static_cast<T>(value)) {}

    // Constructor for the same type
    __device__ FusionFloatBase(const T& value) : value_(value) {}

    // Arithmetic operators
    __device__ __forceinline__ FusionFloatBase operator*(const FusionFloatBase& a) const {
        return FusionFloatBase(value_ * a.value_);
    }

    __device__ __forceinline__ FusionFloatBase operator+(const FusionFloatBase& a) const {
        return FusionFloatBase(value_ + a.value_);
    }

    __device__ __forceinline__ FusionFloatBase operator-(const FusionFloatBase& a) const {
        return FusionFloatBase(value_ - a.value_);
    }

    __device__ __forceinline__ FusionFloatBase operator/(const FusionFloatBase& a) const {
        return FusionFloatBase(value_ / a.value_);
    }

    // Increment and decrement operators
    __device__ __forceinline__ void operator++() {
        value_ += static_cast<T>(1.0f);
    }

    __device__ __forceinline__ void operator--() {
        value_ -= static_cast<T>(1.0f);
    }

    // Compound assignment operators
    __device__ __forceinline__ void operator+=(const FusionFloatBase& a) {
        value_ += a.value_;
    }

    __device__ __forceinline__ void operator-=(const FusionFloatBase& a) {
        value_ -= a.value_;
    }

    __device__ __forceinline__ void operator*=(const FusionFloatBase& a) {
        value_ *= a.value_;
    }

    __device__ __forceinline__ void operator/=(const FusionFloatBase& a) {
        value_ /= a.value_;
    }

    // Atomic add operation
    __device__ __forceinline__ void atomicAdd(const FusionFloatBase* address, const FusionFloatBase& val) {
        ::atomicAdd(&address->value_, val.value_);
    }

    // Getter for value
    __device__ __forceinline__ T* value() {
        return &value_;
    }
};
