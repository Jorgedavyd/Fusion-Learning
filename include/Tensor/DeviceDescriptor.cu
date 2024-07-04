//Describes the device with its properties
#include <cuda.h>
#include <torch/extension.h>
#include <string>

struct DeviceConfig{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp,0);
		TORCH_CHECK((float)deviceProp.major >= 9., "Not valid compute version");
		const float constant_memory = deviceProp.totalConstMem;
		const float global_memory = deviceProp.totalGlobalMem;
		const float shared_memory = deviceProp.sharedMemPerMultiprocessor;
		const int number_single_multiprocessors = deviceProp.multiProcessorCount;
		const int max_thread_per_sm = deviceProp.maxThreadsPerMultiProcessor;
};
