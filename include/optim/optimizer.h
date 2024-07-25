#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <cstdint>
template<typename index_t = uint32_t>
class OptmHandler{};

template<typename index_t = uint32_t>
class OptimizerBase {
public:
    virtual explicit OptimizerBase (WeightHandler<index_t>* weight_handler);
    virtual void step (void);
    virtual void zero (void);

private:
    FusionTensorDescriptor* cum_grad;
};


#endif //OPTIMIZER_H
