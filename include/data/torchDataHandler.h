#include <cstdint>
#include "Tensor.h"
#include <torch/extension.h>

#ifndef DATALOADER_H
#define DATALOADER_H

namespace data = torch::utils::data;

template<typename index_t = uint32_t>
class DataHandler {
public:
    explicit __host__ DataHandler(data::DataLoader* dataloader);

    FusionTensorDescriptor<index_t>* __host__ operator[](index_t& index);

private:
    FusionTensorDescriptor</*Add all the template things*/index_t> tensor;
    data::DataLoader* dataloader_ptr;
    FusionTensorDescriptor* __host__ toFusion (torch::Tensor* tensor_ptr);
};
#endif //DATALOADER_H

