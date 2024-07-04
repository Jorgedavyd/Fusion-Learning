#include <cstdint>
#include <cuda.h>
#include <torch/extension.h>
#include "fusion/tensor/DeviceDescriptor"
#include <vector>

// Asynchronous methods
// Being an interface between the optimal tensor and the kernels
template <typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int32_t>
class FusionTensorAccessor {
	protected:
		PtrTraits data_;
		size_t strides_;
		size_t sizes_;
		__device__ __forceinline__ index_t mapping (index_t index) const {
			// This defines the mapping that takes b, c, h, w -> best data manipulation routines
		};
	public:
		__device__ T& operator[](unsigned index_t idx) {
			// Define how to access data with the mapping
			return mapping(idx);
		};
			
		__device__ T& operator[](unsigned index_t idx) const {
			// Define how to access data with the mapping
		};
};

template <typename index_t = uint32_t>
std::vector<index_t> get_dims (torch::Tensor input) {
	std::vector<index_t> out;
	for (auto dim: input.size()) {
		out.push_back(dim);
	};
	return out;
};

// create coalesced data (one timer), just at the beginning
template <typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = uint32_t>
class FusionTensorDescriptor {
	protected:
		std::vector<index_t> dims;
		size_t size_ = sizeof(T);
		static const torch::GenericPackedTensor<T, N, PtrTraits, index_t> accessor;
		static T* cuda_t;
		// Given the input index_t and the new structure of the data it creates a mapping for the optimized data
	public:
		FusionTensorDescriptor (torch::Tensor* tensor) : accessor((index_t==32) ? tensor.packed_accessor32<T, N, PtrTraits> : tensor.packed_accessor64<T, N, PtrTraits>) {
			// Define the dimensions;
			dims = get_dims<index_t>(*tensor);
			// Define the space in memory
			for (index_t dim: this->dims) {
				this->size_*=dim;
			}
			// Optimize tensor allocation
			this->optimizeTensor();
		};
	private:	
		void optimizeTensor (void) {
			static cudaMalloc((void**)&this->cuda_t, this->size_);
			// Save tensor data in the coalesced way;
		}; // See if that would make optimizeTensor be called just once

		~FusionTensorDescriptor (void) {
			// Deallocate or free CUDA or just define it as a one timer
		};

};	

template <typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int32_t>
class FusionTensor: private FusionTensorBase<T, N, PtrTraits, index_t>, private FusionTensorAccessor<T, N, PtrTraits, index_t> {
	public:
		FusionTensor (const torch::Tensor* tensor)  : FusionTensorDescriptor(tensor), FusionTensorAccessor(tensor) {};
};

