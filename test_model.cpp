#include <iostream>
#include <memory>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <tuple>
#include <vector>

int main(int argc, const char *argv[]) {

  // Get the path of model
  if (argc != 2) {
    std::cerr << "usage: test_model <path-to-exported-script-module>\n";
    return -1;
  }

  torch::jit::script::Module model;
  torch::Device device(torch::kCUDA);
  try {
    // model = torch::jit::load(argv[1]);
    model = torch::jit::load(argv[1], device);
    std::cout << "successfully loaded the model\n";
  }

  catch (const c10::Error &e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  // model.to(device);
  // std::cout << model.device().type() << std::endl;
  torch::Tensor coords = torch::tensor(
      {{{0.03192167, 0.00638559, 0.01301679},
        {-0.83140486, 0.39370209, -0.26395324},
        {-0.66518241, -0.84461308, 0.20759389},
        {0.45554739, 0.54289633, 0.81170881},
        {0.66091919, -0.16799635, -0.91037834}},
       {{-4.1862600, 0.0575700, -0.0381200},
        {-3.1689400, 0.0523700, 0.0200000},
        {-4.4978600, 0.8211300, 0.5604100},
        {-4.4978700, -0.8000100, 0.4155600},
        {0.00000000, -0.00000000, -0.00000000}}},
      torch::TensorOptions().requires_grad(true).dtype(torch::kDouble));

  torch::Tensor species = torch::tensor(
      {{1, 0, 0, 0, 0}, {2, 0, 0, 0, -1}},
      torch::TensorOptions().requires_grad(false).dtype(torch::kLong));

  // Define the input variables
  coords = coords.to(device);
  species = species.to(device);
  std::vector<torch::jit::IValue> inputs;
  std::vector<torch::jit::IValue> tuple;
  tuple.push_back(species);
  tuple.push_back(coords);
  inputs.push_back(torch::ivalue::Tuple::create(tuple));

  // Run the model
  auto aev = model.forward(inputs).toTuple()->elements()[1].toTensor();
  std::cout << aev.sizes() << std::endl;
}
