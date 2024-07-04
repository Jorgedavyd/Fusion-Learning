#include <cstdint>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

template <typename index_t = uint32_t>
std::vector<index_t> get_dims (torch::Tensor input) {
	std::vector<index_t> out;
	for (auto dim: input.size()) {
		out.push_back(dim);
	};
	return out;
};

template <typename index_t = uint32_t>
std::vector<index_t> get_dims (std::vector<torch::Tensor> input) {
	return get_dims(input[0]);
};

template <typename scalar_t, typename index_t = uint32_t>
class FusionDataLoader {
	public: 
		FusionDataLoader (torch::utils::data::DataLoader* dataloader) {
			dims = get_dims(*dataloader[0]);
			for (index_t dim: dims) {
				size_ *= dim;
			};
		};

		__device__ scalar_t& operator[](index_t idx) {
			return *accessor_ptr[idx];
		}
};
