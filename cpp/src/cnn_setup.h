#ifndef LOADER_H
#define LOADER_H
#include <torch/torch.h>


namespace CNNSetup {

    torch::jit::script::Module load_model(std::string model_path);

    torch::Device get_device_available();

};






#endif //LOADER_H
