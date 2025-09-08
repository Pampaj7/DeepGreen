#include "cnn_setup.h"

#include <torch/script.h>

torch::jit::script::Module CNNSetup::load_model(const std::string& model_path)
{
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        throw std::runtime_error(e.msg());
    }
    std::cout << model_path << " loaded\n";
    return module;
}

torch::Device CNNSetup::get_device_available()
{
    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }
    return {device_type};
}

void CNNSetup::print_num_parameters(const torch::nn::Module& model)
{
    std::size_t total_params = 0;
    for (const auto& param : model.parameters(/*recurse=*/true)) {
        total_params += param.numel();
    }
    std::cout << "Total parameters number: " << total_params << std::endl;
}

void CNNSetup::print_trainable_parameters(const torch::nn::Module& model)
{
    std::size_t trainable_params = 0;
    for (const auto& param : model.parameters(/*recurse=*/true)) {
        if (param.requires_grad())
            trainable_params += param.numel();
    }
    std::cout << "Total trainable parameters: " << trainable_params << std::endl;
}
