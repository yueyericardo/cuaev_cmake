import sys
import os
import os.path as osp
import torch
import torchani

#torch.classes.load_library('/work/dev/torchani/torchani/libcuaev.so')

def save_cuda_aev():
    device = torch.device('cuda')
    tolerance = 5e-5
    Rcr = 5.2000e+00
    Rca = 3.5000e+00
    EtaR = torch.tensor([1.6000000e+01], device=device)
    ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00,
                         3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00], device=device)
    Zeta = torch.tensor([3.2000000e+01], device=device)
    ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00,
                         1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=device)
    EtaA = torch.tensor([8.0000000e+00], device=device)
    ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00,
                         2.2000000e+00, 2.8500000e+00], device=device)
    num_species = 5

    device = torch.device('cuda')
    cuaev_computer = torchani.AEVComputer(
        Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species, use_cuda_extension=True)

    # torch.jit.save()
    # _, aev = aev_computer((species, coordinates))
    script_module = torch.jit.script(cuaev_computer.eval())
    # script_module = torch.jit.freeze(script_module.eval())
    script_module.save('model.pt')
    cu_aev = torch.jit.load('model.pt').eval()


if __name__ == '__main__':
    save_cuda_aev()
