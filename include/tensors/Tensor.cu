#include <cuda.h>
#include <torch/extension.h>
#include "fusion/tensor/DeviceDescriptor"
class FusionTensorDescriptor; // Creates optimal tensors from torch::Tensor
class FusionTensorAccessor; // Creates an accessor to the data through mappings from (n, c, w, h) to its optimal structure
// Asynchronous methods
template <typename scalar_t, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int32_t>
// Being an interface between the optimal tensor and the kernels
class FusionTensorAccessor {
	protected:
		PtrTraits data_;
		size_t size_;
		size_t stride;

	public:
		const scalar_t& operator[](index_t ); // Define how to access data
};

// create coalesced data (one timer), just at the beginning
template <typename scalar_t, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int32_t>
class FusionTensorDescriptor {
	protected:
		const torch::GenericPackedTensor<scalar_t, N, PtrTraits, index_t> accessor;
		// Given the input index_t and the new structure of the data it creates a mapping for the optimized data
		index_t idx_mapping (index_t idx) {
			return ...;
		} 
	public:
		FusionTensorBase(const torch::Tensor* tensor) {
			TORCH_CHECK(index_t == 32 || index_t == 64, "Not valid index type");
			CHECK(*tensor);
			// Create torch accessor to access the data to re allocate
			accessor = (index_t==32) ? tensor.packed_accessor32<scalar_t, N, PtrTraits> : tensor.packed_accessor64<scalar_t, N, PtrTraits>;
			// Optimize tensor allocation
			this->optimizeTensor();
			// Delete default tensor allocation (torch) (before sending from shared memory to global)	
		};
	private:
		//Purpose: Creating asynchronous data optimization step	
		__global__ void optimizeTensorKernelParent (void) {
		}
		
		__global__ void optimizeTensorKernelChild (void) {
		}
		
		static void optimizeTensor (void) {

		} // See if that would make optimizeTensor be called just once
		void to_host (void) {};
		void to_device (void) {};

}	
template <typename scalar_t, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int32_t>
class FusionTensor: protected FusionTensorBase, protected FusionTensorAccessor {
	private:
	public:
		FusionTensorDescriptor (const torch::Tensor* tensor)  : FusionTensorBase(tensor), FusionTensorAccessor(tensor) {
		__device__ __forceinline__ T& operator[](index_t index) {
			return this->data_[this->strides_[0]*i];
		}

		}

};

template <>
using FusionTensorDescriptor32 = FusionTensorDescriptor<int32_t>;

template <>
using FusionTensorDescriptor64 = FusionTensorDescriptor<int64_t>;
