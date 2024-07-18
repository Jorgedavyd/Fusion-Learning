#ifndef LINEAR_TRANSFORMATION_H
#define LINEAR_TRANSFORMATION_H

template <bool train = false>
class linear_transformation {
public:
    linear_transformation (FusionTensorDescriptor& input, FusionTensorDescriptor& weight, FusionTensorDescriptor& bias);
};

#endif // LINEAR_TRANSFORMATION
