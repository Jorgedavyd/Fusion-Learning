#include "ScalarBase"
#include <cuda_fp8.h>

template <>
class FusionFloatBase<__nv_fp8_storage_t, false> {
	public:

};
