#include <torch/torch.h>
#include <iostream>
#include <torch/script.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <vector>

int main(int argc, const char* argv[]) {

  // Get the path of model
  if (argc != 2) {
    std::cerr << "usage: test_model <path-to-exported-script-module>\n";
    return -1;
  }

  torch::jit::script::Module model;
  torch::Device device(torch::kCPU);
  try {
	
    model = torch::jit::load(argv[1]);
    std::cout << "successfully loaded the model\n";
  }

  catch (const c10::Error& e) {
	std::cerr << "error loading the model\n";
	return -1;
  }
}
