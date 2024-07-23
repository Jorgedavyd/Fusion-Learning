#ifndef ATTENTION_H
#define ATTENTION_H
#include "Module.h"

class GeneralAttention: Module {
public:
      GeneralAttention();
private:
    virtual void __global__ default_fwd_kernel ();
    virtual void __global__ enhanced_fusion_kernel ();
};

using GeneralAttention<> = MultiHeadAttention;
using GeneralAttention<> = MultiQueryAttention;
using GeneralAttention<> = GroupedQueryAttention;

#endif //ATTENTION_H
