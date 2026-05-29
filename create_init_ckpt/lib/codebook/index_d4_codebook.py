import os
import math

import torch
from torch import nn

from ..utils.matmul_had import matmul_hadU_cuda


_LI_CODESZ = 4
_LI_CODEBIT = 2

ortho_mat = torch.load(os.path.join(os.path.dirname(__file__), "d4.pt"), weights_only=True)

class LinearD4_codebook(nn.Module):
    
    def __init__(self, inference=False, **kwargs):
        super().__init__()
        self.id = "linear_d4_tuneable"
        self.codesz = _LI_CODESZ
        self.opt_scale = 1.21
        self.idx_dtype = torch.uint8
        self.packsz = 1
        self.pack_out = False
        self.version = 0

        self.min_var = 0
        self.max_var = 2**_LI_CODEBIT - 1
        linear_proj = nn.Linear(_LI_CODESZ, _LI_CODESZ, bias=False, dtype=torch.bfloat16)
        linear_proj.weight.requires_grad = False
        input_index_sample = torch.cartesian_prod(*([torch.arange(2**_LI_CODEBIT)]*_LI_CODESZ)).to(device=linear_proj.weight.device, dtype=linear_proj.weight.dtype)
        linear_proj.weight.data = ortho_mat.to(device=linear_proj.weight.device, dtype=linear_proj.weight.dtype)
        mean = (2**_LI_CODEBIT-1)/2
        scale = 1 # (torch.pow(torch.arange(2**_LI_CODEBIT, dtype=torch.float), 2).mean().item() - mean**2)
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
        w = self.grid[Qidxs, :].reshape(m, -1)
        return w

class LinearD4Linear(nn.Module):
    
    def __init__(self, device, in_features, out_features, group_size):
        super().__init__()
        self.codebook = LinearD4_codebook().to(torch.bfloat16).to(device)
        self.group_size = group_size
        self.group_scale = nn.Parameter(torch.zeros(out_features//group_size, in_features, device=device))
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

        W_decompressed = self.codebook.by_idxs(Qidxs_list[0]) * (1.0+torch.repeat_interleave(self.group_scale, repeats=self.group_size, dim=0).to(torch.float16))

        x = (x.to(torch.bfloat16) @ W_decompressed.T).to(torch.float32)

        x = matmul_hadU_cuda(x, had_right, K_right)
        x = x * SV * self.scale

        output = x.view(*input.shape[:-1], m)
        
        return output
    
