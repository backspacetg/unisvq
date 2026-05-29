from os.path import join, dirname

import torch
from safetensors import safe_open
import fast_hadamard_transform

from .. import utils

hadamard_mats = {}
with safe_open(join(dirname(__file__), "hadamard.safetensors"), framework="pt", device="cpu") as f:
    k_sort = []
    for k in f.keys():
        k_sort.append((int(k), f.get_tensor(k)))
k_sort.sort(key=lambda x:x[0], reverse=True)
hadamard_mats = {x[0]: x[1] for x in k_sort}
del k_sort 


def get_hadK(n, transpose=False):
    hadK, K = None, None
    for prefetch_k in hadamard_mats.keys():
        if n % prefetch_k == 0:
            assert (is_pow2(n//prefetch_k))
            K = prefetch_k
            hadK = hadamard_mats[prefetch_k].T if transpose else hadamard_mats[prefetch_k]
            break
    return hadK, K


def matmul_hadU(X, transpose=False, skip_mix=False):
    n = X.shape[-1]
    hadK, K = get_hadK(n, transpose)
    input = X.clone().view(-1, n, 1)
    output = input.clone()
    while input.shape[1] > K:
        input = input.view(input.shape[0], input.shape[1] // 2, 2, input.shape[2])
        output = output.view(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.view(input.shape[0], input.shape[1], -1)
        (input, output) = (output, input)
    del output
    utils.clean()
    if K > 1 and not skip_mix:
        input = torch.bmm(hadK.repeat(len(input), 1, 1).to(input.device).to(input.dtype), input)
    return input.view(X.shape) / torch.tensor(n).sqrt()


def matmul_hadUt(X, skip_mix=False):
    return matmul_hadU(X, transpose=True, skip_mix=skip_mix)


torch.library.define("quip_lib::hadamard", "(Tensor x, float scale) -> Tensor")

@torch.library.register_fake("quip_lib::hadamard")
def hadamard_abstract(x: torch.Tensor, scale: float) -> torch.Tensor:
    return x

@torch.library.impl("quip_lib::hadamard", "default")
def hadamard(x: torch.Tensor, scale: float) -> torch.Tensor:
    return fast_hadamard_transform.hadamard_transform(x, scale)


def matmul_hadU_cuda(X, hadK, K, transpose=False, skip_mix=False):
    n = X.shape[-1]
    if K == 1:
        return torch.ops.quip_lib.hadamard(X.contiguous(), 1/(n**0.5))

    if transpose:
        hadK = hadK.T.contiguous()
    input = X.float().view(-1, K, n // K)
    input = torch.ops.quip_lib.hadamard(input.contiguous(), 1/(n**0.5))
    if not skip_mix:
        input = hadK.to(input.device).to(input.dtype) @ input
    return input.to(X.device).to(X.dtype).reshape(X.shape)


def matmul_hadUt_cuda(X, hadK, K):
    return matmul_hadU_cuda(X, hadK, K, transpose=True)

def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)