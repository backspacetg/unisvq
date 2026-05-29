import torch
import fast_hadamard_transform


if not hasattr(torch.ops.hadamard, "hadamard"):
    print("register hadamard::hadamard")
    torch.library.define("hadamard::hadamard", "(Tensor x, float scale) -> Tensor")

    @torch.library.register_fake("hadamard::hadamard")
    def hadamard_abstract(x: torch.Tensor, scale: float) -> torch.Tensor:
        return x

    @torch.library.impl("hadamard::hadamard", "default")
    def hadamard(x: torch.Tensor, scale: float) -> torch.Tensor:
        return fast_hadamard_transform.hadamard_transform(x, scale)
else:
    print("skip register hadamard::hadamard")


class HadamardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        output = torch.ops.hadamard.hadamard(x, scale)
        ctx.scale = scale
        ctx.save_for_backward()
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        scale = ctx.scale
        return torch.ops.hadamard.hadamard(grad_output, scale), None
    
    
def matmul_hadU_cuda_blockwise(X):
    BLOCK_SIZE = 128
    input_x = X.view(-1, 1, BLOCK_SIZE)
    return HadamardFunction.apply(input_x, BLOCK_SIZE**(-0.5)).view(X.shape)
