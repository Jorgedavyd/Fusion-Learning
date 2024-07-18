#ifndef CNN_H
#define CNN_H
#include <Module.h>

template <bool train = false>
class Convolution1d : Module<train>{};
template <bool train = false>
class Convolution2d : Module<train>{};
template <bool train = false>
class Convolution3d : Module<train>{};

#endif //CNN_H
