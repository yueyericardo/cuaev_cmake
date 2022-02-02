# cuaev_cpp

Instruction
```
git clone git@github.com:aiqm/torchani.git
mv torchani/torchani/cuaev/* ./cuaev
rm -rf torchani
python save_ani.py
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
make
./test_model ../model.pt
```
