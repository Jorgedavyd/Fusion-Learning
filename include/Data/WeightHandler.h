#ifndef WEIGHT_HANDLER_H
#define WEIGHT_HANDLER_H
#include <cstdint>

// Handles all things related to the weights and contains them
template<typename index_t = uint32_t>
class WeightHandler{
public:
    WeightHandler ();
    //Allocate weights in constant memory as a function of the architecture
    //Make the inference process a for loop per step
    FusionTensorDescriptor* __host__ __device__ operator[] (index_t& index);

private:
    FusionTensorDescriptor* weights;
    FusionTensorDescriptor* __host__ toConstant (FusionTensorDescriptor& tensor) const;

};

#endif//WEIGHT_HANDLER_H
