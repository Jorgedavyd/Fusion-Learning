#include <cstring>
#include <cuda.h>
#include "Data/TorchDataHandler.h"
#include "Tensor.h"

template<typename index_t>
__host__ FusionTensorDescriptor* toFusion (torch::Tensor* tensor_ptr) {
    // Create Tensor Descriptor
};

template<typename index_t>
__host__ DataHandler<index_t>::DataHandler (data::DataLoader& dataloader) {
    // Allocate Tensor descriptor based on the max size
};

template<typename index_t>
__host__ FusionTensorDescriptor* DataHandler<index_t>::operator[]() const {
    // Create Tensor Descriptor
};


