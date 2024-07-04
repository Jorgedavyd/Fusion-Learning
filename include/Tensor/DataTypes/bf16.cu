#include <cuda_bf16.h>

class bf16{
	public:
		__nv_bfloat16 value_;
		__device__ bf16() : value_(__float2half(0.0f)) {}

		__device__ bf16(float value) : value_(__float2half(value)) {}

		__device__ bf16(const __half& value) : value_(value) {}

		// Arithmetic operators
		__device__ __forceinline__ bf16 operator*(const bf16& a) const {
			return bf16(__hmul(value_, a.value_));
		}

		__device__ __forceinline__ bf16 operator+(const bf16& a) const {
			return bf16(__hadd(value_, a.value_));
		}

		__device__ __forceinline__ bf16 operator-(const bf16& a) const {
			return bf16(__hsub(value_, a.value_));
		}

		__device__ __forceinline__ bf16 operator/(const bf16& a) const {
			return bf16(__hdiv(value_, a.value_));
		}

		// Increment and decrement operators
		__device__ __forceinline__ void operator++() {
			value_ = __hadd(value_, __float2half(1.0f));
		}

		__device__ __forceinline__ void operator--() {
			value_ = __hsub(value_, __float2half(1.0f));
		}

		// Compound assignment operators
		__device__ __forceinline__ void operator+=(const bf16& a) {
			value_ = __hadd(value_, a.value_);
		}

		__device__ __forceinline__ void operator-=(const bf16& a) {
			value_ = __hsub(value_, a.value_);
		}

		__device__ __forceinline__ void operator*=(const bf16& a) {
			value_ = __hmul(value_, a.value_);
		}

		__device__ __forceinline__ void operator/=(const bf16& a) {
			value_ = __hdiv(value_, a.value_);
		}

		// Atomic add operation
		__device__ __forceinline__ void atomicAdd(const bf16* address, const bf16& val) {
			atomicAdd(&address->value_, val.value_);
		}

		// Getter for value
		__device__ __forceinline__ __nv_bfloat16* value() {
			return &value_;
		}
};

