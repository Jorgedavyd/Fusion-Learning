#ifndef ATTENTION_H
#define ATTENTION_H
#include "Module.h"

class GeneralAttention: Module {
public:
      GeneralAttention();
private:
    virtual void __global__ default_fwd_kernel (void);
    virtual void __global__ enhanced_fusion_kernel (void);
};

using GeneralAttention<> = MultiHeadAttention;
using GeneralAttention<> = MultiQueryAttention;
using GeneralAttention<> = GroupedQueryAttention;

#endif //ATTENTION_H
