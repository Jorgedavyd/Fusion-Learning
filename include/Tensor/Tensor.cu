#include <cstdint>
#include <cuda.h>
#include <torch/extension.h>
#include "fusion/tensor/DeviceDescriptor"
#include <vector>
// Asynchronous methods
// Being an interface between the optimal tensor and the kernels
template <typename FusionScalar, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int32_t>
class FusionTensorAccessor {
	protected:
		PtrTraits data_;
		size_t size_;
		size_t stride;
		__device__ __forceinline__ FusionScalar& mapping (index_t index) {
			// This defines the mapping that takes b, c, h, w -> best data manipulation routines
			
		};
	public:
		__device__ FusionScalar& operator[](unsigned index_t idx) {
			// Define how to access data with the mapping
			return mapping(idx);
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
template <typename FusionScalar, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = uint32_t>
class FusionTensorDescriptor {
	protected:
		std::vector<index_t> dims;
		size_t size_ = sizeof(FusionScalar);
		static const torch::GenericPackedTensor<FusionScalar, N, PtrTraits, index_t> accessor;
		static FusionScalar* cuda_t;
		// Given the input index_t and the new structure of the data it creates a mapping for the optimized data
	public:
		FusionTensorDescriptor (torch::Tensor* tensor) : accessor((index_t==32) ? tensor.packed_accessor32<FusionScalar, N, PtrTraits> : tensor.packed_accessor64<FusionScalar, N, PtrTraits>) {
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

template <typename FusionScalar, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int32_t>
class FusionTensor: private FusionTensorBase<FusionScalar, N, PtrTraits, index_t>, private FusionTensorAccessor<FusionScalar, N, PtrTraits, index_t> {
	public:
		FusionTensor (const torch::Tensor* tensor)  : FusionTensorDescriptor(tensor), FusionTensorAccessor(tensor) {};
};

