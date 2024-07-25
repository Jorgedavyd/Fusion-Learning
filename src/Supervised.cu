#pragma once
#include <cuda.h>
#include <torch/extension.h>
#include "fusion/Tensor/DeviceDescriptor"
#include "fusion/optimizer"
#include "fusion/data"

namespace DataLoader = torch::utils::data::DataLoader;

template <typename scalar_t, typename grad_step_t, typename index_t = uint32_t>
__host__ struct HyperParameters {
    const unsigned index_t epochs;
    const scalar_t learning_rate;
    const scalar_t weight_decay;
    const scalar_t gradient_clipping;
    const FusionOptimizer<grad_step_t> optimizer;

    HyperParameters<scalar_t, index_t, grad_step_t>(const unsigned index_t epochs, const scalar_t learning_rate, const scalar_t weight_decay, const FusionOptimizer<grad_step_t> optimizer, const scalar_t gradient_clipping) : learning_rate(learning_rate), epochs(epochs), weight_decay(weight_decay), optimizer(optimizer), gradient_clipping(gradient_clipping) {};

};

template <typename training_type, typename model_type>
class SupervisedTrainer : protected DeviceDescriptor {
private:
    //Defining cuda base
    void* kernels[];
    static cudaGraphEdgeData edgeData;
    FusionCost* J;
    FusionOptimizer* opt;
public:
    SupervisedTrainer (void* kernels[]) : kernels(kernels), DeviceDescriptor() {

    };
    __global__ void fused_propagation() {
        // Define the runtime based on the available kernels and runtimes
        // Contains all the kernels on a graph
        // Contains programmaticed kernels (model) -> loss (kernel) -> compute the backward pass with the input.
    }; // Fused propagation

    __host__ __forceinline__ void epochLaunch (unsigned int idx, FusionDataLoader* train_loader) {
        // pass the batch pointer to the fused propagation.
        const FusionAccessor data_ptr = train_loader[idx];
        this->fused_propagation(data_ptr);
        // The secondary kernel will be launch to preemptively bring the next batch of data over the already deleted tensors
        const FusionAccessor next_data_ptr = train_loader[idx];
        this->allocate_next_batch(next_data_ptr);
        // By allowing programmaticed behaviour, we will be able to load the next batch of data by hiding latency intervals
    }
    torch::Tensor* operator()(DataLoader* train_dataloader, DataLoader* val_dataloader, HyperParameters* hyperparameters, FusionOptimizer* optimizer) {// Fused propagation
        // Torch dataloder -> Fusion DataLoader
        // Define the hyperparameters into the optimizer
        // create a stream that defines the graph
        #pragma unroll
        for (epoch = 0; epoch < hyperparameters.epochs ; epoch++) {
            #pragma unroll
            for (i=0; i<length(train_loader); i++) {
                this->epochLaunch(i, &train_loader)
            }
        };

    };
};
/*
   torch::Tensor -> Fusion Tensor -> fused propagation -> next iteration with programmaticed workflow
 */
