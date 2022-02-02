# cuaev_cpp

Make sure using cuda-11.3, and CUDA_HOME is set properly.

## Build Instruction

Requirement
```bash
# environment
conda create -n torch3
conda activate torch3
# torch
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
# torchani
git clone git@github.com:aiqm/torchani.git
cd torchani
python setup.py install --cuaev
```

Build
```bash
cd ..
cp torchani/torchani/cuaev/* ./cuaev
python save_ani.py
# should print torch.Size([2, 5, 384])
rm -rf build; mkdir build; cd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
make
./test_model ../model.pt
# should print [2, 5, 384]
```
