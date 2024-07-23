#include "ScalarBase.cu"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

template <typename T, typename to_type>
class FusionFloatBase<T, true> {
	public:
		T value_;
		__device__ FusionFloatBase () : value_(to_type(0.0f)) {}
		__device__ FusionFloatBase (float value) : value_(static_cast<float>(value)) {}
		__device__ FusionFloatBase (const T& value) : value_(to_type(value)) {}
		__device__ __forceinline__ FusionFloatBase operator*(const FusionFloatBase& a) const {
			return FusionFloatBase(__hmul(value_, a.value_));
		}

		__device__ __forceinline__ FusionFloatBase operator+(const FusionFloatBase& a) const {
			return FusionFloatBase(__hadd(value_, a.value_));
		}

		__device__ __forceinline__ FusionFloatBase operator-(const FusionFloatBase& a) const {
			return FusionFloatBase(__hsub(value_, a.value_));
		}

		__device__ __forceinline__ FusionFloatBase operator/(const FusionFloatBase& a) const {
			return FusionFloatBase(__hdiv(value_, a.value_));
		}

		__device__ __forceinline__ void operator++() {
			value_ = __hadd(value_, to_type(1.0f));
		}

		__device__ __forceinline__ void operator--() {
			value_ = __hsub(value_, to_type(1.0f));
		}

		__device__ __forceinline__ void operator+=(const FusionFloatBase& a) {
			value_ = __hadd(value_, a.value_);
		}

		__device__ __forceinline__ void operator-=(const FusionFloatBase& a) {
			value_ = __hsub(value_, a.value_);
		}

		__device__ __forceinline__ void operator*=(const FusionFloatBase& a) {
			value_ = __hmul(value_, a.value_);
		}

		__device__ __forceinline__ void operator/=(const FusionFloatBase& a) {
			value_ = __hdiv(value_, a.value_);
		}

		__device__ __forceinline__ void atomicAdd(const FusionFloatBase* address, const FusionFloatBase& val) {
			atomicAdd(&address->value_, val.value_);
		}

		__device__ __forceinline__ T* value() {
			return &value_;
		}
};
template <typename T>
class FusionFloatBase<T, true> {
	public:

};

template <typename T>
using Float16Base = FusionFloatBase<T, true>;
using bfloat16 = Float16Base<__nv_bfloat16>;
using float16 = Float16Base<__half>;

