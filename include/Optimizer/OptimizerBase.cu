#ifndef OPTIMIZERBASE_H
#define OPTIMIZERBASE_H

#include <cstdint>
template<typename index_t = uint32_t>
class OptimizerHandler {};

template<typename index_t = uint32_t>
class OptimizerBase {
public:
    virtual explicit OptimizerBase (WeightHandler<index_t>* weight_handler);
    virtual void step (void);
    virtual void zero (void);

private:
    FusionTensorDescriptor* cum_grad;
};


#endif //OPTIMIZERBASE_H
