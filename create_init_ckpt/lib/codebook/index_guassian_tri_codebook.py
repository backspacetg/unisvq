import os
import math

import torch
from torch import nn

from ..utils.matmul_had import matmul_hadU_cuda

import glog

_LI_CODESZ = 4
_LI_CODEBIT = 2
CODE_MAX_VAR = 3

ortho_mat = torch.load(os.path.join(os.path.dirname(__file__), "ortho_dim4.pt"), weights_only=True)

class LinearGuassian_codebook(nn.Module):
    
    def __init__(self, inference=False, **kwargs):
        super().__init__()
        self.id = "linear_index"
        self.codesz = _LI_CODESZ
        self.opt_scale = 1.0
        self.idx_dtype = torch.uint8
        self.packsz = 1
        self.pack_out = False
        self.version = 0

        self.min_var = 0
        self.max_var = CODE_MAX_VAR - 1
        linear_proj = nn.Linear(_LI_CODESZ, _LI_CODESZ, bias=False)
        linear_proj.weight.requires_grad = False
        input_index_sample = torch.cartesian_prod(*([torch.arange(CODE_MAX_VAR)]*_LI_CODESZ)).to(device=linear_proj.weight.device, dtype=linear_proj.weight.dtype)
        linear_proj.weight.data = ortho_mat.to(device=linear_proj.weight.device, dtype=linear_proj.weight.dtype)
        mean = (CODE_MAX_VAR-1)/2
        scale = (torch.pow(torch.arange(CODE_MAX_VAR, dtype=torch.float), 2).mean().item() - mean**2)
        input_index_sample = 1/math.sqrt(scale)*(input_index_sample-mean)
        fake_codebook = linear_proj(input_index_sample) 
        self.register_buffer('linear_proj', linear_proj.weight.data.clone())
        self.register_buffer('grid', fake_codebook)
        if not inference:
            self.register_buffer('grid_norm', fake_codebook.norm(dim=-1)**2)


    def quantize(self, w, return_idx=True, **kwargs):
        assert w.shape[-1] == self.codesz
        # glog.debug(f'before quantize: {w.shape}, {w.mean().item()}, {w.std().item()}')
        wqidx = (2 * w @ self.grid.T - self.grid_norm).argmax(1)
        wq = self.grid[wqidx, :]
        # glog.debug(f'after quantize: {wq.shape}, {wq.mean().item()}, {wq.std().item()}')
        if return_idx:
            return wq, wqidx.to(self.idx_dtype)
        return wq
    
    def maybe_pack_idxs(self, idxs):
        return idxs

    def by_idxs(self, Qidxs, **kwargs):
        m = Qidxs.shape[0]
        Qidxs = Qidxs.to(torch.long).flatten()
        # print(Qidxs.shape, Qidxs.max(), Qidxs.min())
        # print('========================')
        w = self.grid[Qidxs, :].reshape(m, -1)
        return w

class LinearGuassianLinear(nn.Module):
    
    def __init__(self, device):
        super().__init__()
        self.codebook = LinearGuassian_codebook().to(torch.float16).to(device)
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
        try:
            W_decompressed = self.codebook.by_idxs(Qidxs_list[0])
        except:
            print(Qidxs_list[0].max())
            exit(0)
        try:
            x = (x.to(torch.float16) @ W_decompressed.T).to(torch.float32)
        except:
            print(W_decompressed.dtype, W_decompressed.shape, Qidxs_list[0].max())
            exit(0)

        x = matmul_hadU_cuda(x, had_right, K_right)
        x = x * SV * self.scale

        output = x.view(*input.shape[:-1], m)
        
        return output
    
