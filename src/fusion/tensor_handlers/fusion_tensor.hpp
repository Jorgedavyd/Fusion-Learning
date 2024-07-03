#include <iostream>
#include <torch/extension.h>
#include "fusion/check.h"
#include "fusion/typing"
	
template <typename scalar_t>
class FusedTensorDescriptor: Tensor {
	private:
		const Tensor* ptr; //data allocation descriptor
		const int stride;
		
	public:
		__device__ FusedTensorDescriptor(Tensor* tensor) {
			CHECK(tensor);
			static const Tensor* = tensor.data_ptr<scalar_t>();
		}

		__device__ __forceinline__ float getItem(const unsigned int n, const unsigned int c, const unsigned int h, const unsigned int w) {
			

		
