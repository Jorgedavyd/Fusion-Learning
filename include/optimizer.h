#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <cstdint>

template<typename index_t = uint32_t>
class OptimHandler{};

template<typename index_t = uint32_t>
class OptimizerBase {
public:
    virtual explicit OptimizerBase (WeightHandler<index_t>* weight_handler);
    virtual void step (void);
    virtual void zero (void);

private:
    FusionTensorDescriptor* cum_grad;
};


template<typename T>
class GradientDescent : OptimizerBase<> {};

template<typename T>
class Adam : GradientDescent {};

template<typename T>
class SGD : GradientDescent {};

template<typename T>
class RSM : GradientDescent {};

#endif //OPTIMIZER_H
