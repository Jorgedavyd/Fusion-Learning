#ifndef TRANSFORMERS_H
#define TRANSFORMERS_H
#include "Module.h"

class TransformerBase : Module {
public:
    TransformerBase();
private:
    virtual void __global__ default_fwd_kernel ();
    virtual void __global__ enhanced_fusion_kernel ();
};

#endif //TRANSFORMERS_H
