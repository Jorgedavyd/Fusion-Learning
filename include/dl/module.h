#ifndef MODULE_H
#define MODULE_H

template <bool train = false>
class Module {
public:
    Module ();
    void operator() ();
private:
    virtual void __global__ default_fwd_kernel (void);
    virtual void __global__ enhanced_fusion_kernel (void);
};

#endif //MODULE_H


