#ifndef LOADER_H
#define LOADER_H
#include <torch/torch.h>


namespace CNNSetup {

    torch::jit::script::Module load_model(const std::string& model_path);

    torch::Device get_device_available();

    void print_num_parameters(const torch::nn::Module& model);

    void print_trainable_parameters(const torch::nn::Module& model);

};



#endif //LOADER_H
