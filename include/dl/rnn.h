#ifndef VANILLA_H
#define VANILLA_H
#include <Module.h>

template <bool train = false>
class GRU : Module<train> {
public:
    GRU();
};

template <bool train = false>
class LSTM: Module<train> {
public:
    LSTM ();
};

template <bool train = false>
class RNN : Module<train> {
public:
    RNN();
};

#endif //VANILLA_H

