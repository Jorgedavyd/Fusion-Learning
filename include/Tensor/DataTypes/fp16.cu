#include <cuda_fp16.h>

class fp16 {
public:
    __half value_;

    __device__ fp16() : value_(__float2half(0.0f)) {}

    __device__ fp16(float value) : value_(__float2half(value)) {}

    __device__ fp16(const __half& value) : value_(value) {}

    // Arithmetic operators
    __device__ __forceinline__ fp16 operator*(const fp16& a) const {
        return fp16(__hmul(value_, a.value_));
    }

    __device__ __forceinline__ fp16 operator+(const fp16& a) const {
        return fp16(__hadd(value_, a.value_));
    }

    __device__ __forceinline__ fp16 operator-(const fp16& a) const {
        return fp16(__hsub(value_, a.value_));
    }

    __device__ __forceinline__ fp16 operator/(const fp16& a) const {
        return fp16(__hdiv(value_, a.value_));
    }

    // Increment and decrement operators
    __device__ __forceinline__ void operator++() {
        value_ = __hadd(value_, __float2half(1.0f));
    }

    __device__ __forceinline__ void operator--() {
        value_ = __hsub(value_, __float2half(1.0f));
    }

    // Compound assignment operators
    __device__ __forceinline__ void operator+=(const fp16& a) {
        value_ = __hadd(value_, a.value_);
    }

    __device__ __forceinline__ void operator-=(const fp16& a) {
        value_ = __hsub(value_, a.value_);
    }

    __device__ __forceinline__ void operator*=(const fp16& a) {
        value_ = __hmul(value_, a.value_);
    }

    __device__ __forceinline__ void operator/=(const fp16& a) {
        value_ = __hdiv(value_, a.value_);
    }

    // Atomic add operation
    __device__ __forceinline__ void atomicAdd(const fp16* address, const fp16& val) {
        atomicAdd(&address->value_, val.value_);
    }

    // Getter for value
    __device__ __forceinline__ __half* value() {
        return &value_;
    }
};

