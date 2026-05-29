import os
import math
import glog
import torch
from torch import nn

from ..utils.matmul_had import matmul_hadU_cuda


_LI_CODESZ = 4
_LI_CODEBIT = 2

class Sphere_codebook(nn.Module):
    
    def __init__(self, inference=False, **kwargs):
        super().__init__()
        self.id = "linear_index"
        self.codesz = _LI_CODESZ
        self.opt_scale = 1.0
        self.idx_dtype = torch.uint8
        self.packsz = 1
        self.pack_out = False
        self.version = 0.1

        self.min_var = 0
        self.max_var = 2**_LI_CODEBIT - 1

        input_index_sample = torch.cartesian_prod(*([torch.arange(2**_LI_CODEBIT)]*_LI_CODESZ)).to(dtype=torch.float)
        input_index_sample = input_index_sample-((2**_LI_CODEBIT-1)/2)
        sphere_norm = (1/torch.norm(input_index_sample, p=2, dim=1, keepdim=True)).nan_to_num()
        r_norm = torch.norm(input_index_sample, p=1, dim=1, keepdim=True)
        fake_codebook = r_norm*input_index_sample*sphere_norm
        self.register_buffer('grid', fake_codebook)
        if not inference:
            self.register_buffer('grid_norm', fake_codebook.norm(dim=-1)**2)

    def quantize(self, w, return_idx=True, **kwargs):
        assert w.shape[-1] == self.codesz
        wqidx = (2 * w @ self.grid.T - self.grid_norm).argmax(1)
        wq = self.grid[wqidx, :]
        if return_idx:
            return wq, wqidx.to(self.idx_dtype)
        return wq
    
    def maybe_pack_idxs(self, idxs):
        return idxs

    def by_idxs(self, Qidxs, **kwargs):
        m = Qidxs.shape[0]
        Qidxs = Qidxs.to(torch.long).flatten()
        w = self.grid[Qidxs, :].reshape(m, -1)
        return w


class SphereLinear(nn.Module):
    
    def __init__(self, device):
        super().__init__()
        self.codebook = Sphere_codebook().to(torch.float16).to(device)
        self.scale = 32

    def maybe_unpack_idxs(self, idxs):
        return (idxs, )

    def cache_WH(self, **kwargs):
        return 
    
    def forward(self, input, Qidxs_list, SU, SV, had_left_T, had_right, K_left, K_right, rescale_WH=False, scaleWH=None, train_mode=False, **kwargs):
        n, m = len(SU), len(SV)
        x = input.view(-1, n).to(torch.float32)
        if rescale_WH:
            x /= scaleWH
        x = x * SU

        x = matmul_hadU_cuda(x, had_left_T, K_left) / self.scale
    
        W_decompressed = self.codebook.by_idxs(Qidxs_list[0])
        x = (x.to(torch.float16) @ W_decompressed.T).to(torch.float32)

        x = matmul_hadU_cuda(x, had_right, K_right)
        x = x * SV * self.scale

        output = x.view(*input.shape[:-1], m)
        
        return output
    
