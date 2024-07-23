#ifndef LINEAR_TRANSFORMATION_H
#define LINEAR_TRANSFORMATION_H

#include <Module.h>
template <bool train = false>
class linear_transformation : Module<train> {
public:
    linear_transformation (FusionTensorDescriptor& input, FusionTensorDescriptor& weight, FusionTensorDescriptor& bias);
};

#endif // LINEAR_TRANSFORMATION
