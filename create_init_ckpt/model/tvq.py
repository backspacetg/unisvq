import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from lib.utils.matmul_had import matmul_hadU_cuda, get_hadK

def power_derivative(w, delta, k, power_clamp_max):
    abs_term = torch.abs(2*w/delta-1)
    val = abs_term.pow(1.0/k-1)/k
    return torch.clamp(val, max=power_clamp_max)

class DGEFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, w, qmax:int=3, k:float=5.0, scale_weight: bool=False, power_clamp_max: float=3.0):
        rmin, rmax = w.min(), w.max()
        scale = (rmax-rmin).clamp(min=1e-8)/float(qmax)
        zero_point = -rmin/scale
        ctx.save_for_backward(w, scale, zero_point)
        ctx.k = k
        ctx.power_clamp_max = power_clamp_max
        ctx.qmax = qmax
        # ctx.mask_rate = mask_rate
        q = torch.round(w/scale+zero_point).clamp(0, qmax)
        if scale_weight:
            q = (q - zero_point) * scale
        return q

    @staticmethod
    def backward(ctx, grad_output):
        w, scale, zero_point = ctx.saved_tensors
        k = ctx.k
        power_clamp_max = ctx.power_clamp_max
        qmax = ctx.qmax
        # mask_rate = ctx.mask_rate
        delta = scale
        q_idx = torch.floor(w/scale+zero_point).clamp(0, qmax)
        q_center = (q_idx-zero_point)*delta
        rel = w-q_center
        dy = power_derivative(rel, delta, k, power_clamp_max=power_clamp_max)
        # grad_mask = (torch.rand(*dy.shape, device=dy.device, dtype=dy.dtype) > mask_rate).to(dtype=bool)
        grad_input = grad_output * dy # * grad_mask

        return grad_input, None, None, None, None

def weight_quant(w: torch.Tensor, qmin=-1, qmax=2, eps=1e-6, scale_weight=False):
    rmin, rmax = w.min(), w.max()
    scale = (rmax - rmin).clamp(min=eps)/float(qmax - qmin)
    zero_point = qmin - torch.round(rmin / scale)
    q = torch.round(w / scale + zero_point).clamp(qmin, qmax)
    if scale_weight:
        q = (q - zero_point) * scale
    return (q - w).detach() + w


def is_meta_module(module: nn.Module):
    for p in module.parameters():
        if p.is_meta:
            return True
    return False


class CodeBookGenerator(nn.Module):
    
    def __init__(self, vec_size, vec_bit, in_features, out_features, device=None, dtype=None, proj_type='bypass'):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.proj_type = proj_type
        self.min_var = 0
        self.max_var = 2**vec_bit-1
        assert proj_type in ['bi_linear', 'linear', 'bitnet', 'bypass', 'hadamard_linear']
        if self.proj_type == "bi_linear":
            self.g1 = nn.Linear(vec_size, vec_size, bias=False, **factory_kwargs)
            self.fn1 = nn.SiLU()
            self.g2 = nn.Linear(vec_size, vec_size, bias=False, **factory_kwargs)
        elif self.proj_type == "linear":
            self.g1 = nn.Linear(vec_size, vec_size, bias=False, **factory_kwargs)
            if not is_meta_module(self.g1):
                self.g1.weight.data = torch.eye(vec_size, vec_size, **factory_kwargs)
        elif self.proj_type == "hadamard_linear":
            self.g1 = nn.Linear(vec_size, vec_size, bias=True, **factory_kwargs)
            if not is_meta_module(self.g1):
                self.g1.weight.data = torch.eye(vec_size, vec_size, **factory_kwargs)
        elif self.proj_type == "bypass" or self.proj_type == "bitnet":
            pass
        else:
            raise NotImplementedError(self.proj_type)
        self.weight_zero_point = (2**vec_bit-1)/2
        self.weight_scale = math.sqrt(out_features+in_features)/2
        self.mask_rate = 0

    def quant_weight(self, ori_weight) -> torch.Tensor:
        if "linear" in self.proj_type:
            ori_weight = self.weight_scale*ori_weight + self.weight_zero_point
            # w = weight_quant(ori_weight, qmin=self.min_var, qmax=self.max_var)
            w = DGEFunction.apply(ori_weight, self.max_var, 5.0)
        elif self.proj_type == "bitnet":
            w = weight_quant(ori_weight, qmin=self.min_var, qmax=self.max_var, scale_weight=True)
        else:
            w = ori_weight
        return w
    
    def forward(self, ori_weight) -> torch.Tensor:
        w_q = self.quant_weight(ori_weight)
        if "linear" in self.proj_type:
            if self.proj_type == "bi_linear":
                g1y = self.g1(w_q)
                w = self.g2(self.fn1(g1y)) + g1y
            elif self.proj_type == "linear" or self.proj_type == "hadamard_linear":
                w = self.g1(w_q)
            else:
                raise NotImplementedError(self.proj_type)
        else:
            w = w_q
        return w
    
    def set_rate(self, mask_rate):
        self.mask_rate = mask_rate


class TVQLinear(nn.Module):

    def __init__(self, in_features, out_features, group_size=128, vec_size=4, vec_bit=2, bias=False, device=None, dtype=None, proj_type="bypass"):
        assert out_features%group_size == 0, (out_features, group_size)
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.vec_size = vec_size

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.scales = nn.Parameter(torch.zeros(out_features, int(in_features//group_size), **factory_kwargs))    
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.code_generator = CodeBookGenerator(
            vec_size=vec_size, 
            vec_bit=vec_bit, 
            in_features=in_features,
            out_features=out_features,
            proj_type=proj_type
        )

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
            
    def forward(self, input):
        # if self.weight.grad is not None:
        #     print("linear", self.weight.grad.isnan().sum(), self.weight.grad.isnan().numel())
        weight = self.code_generator(self.weight.reshape(-1, self.vec_size)).reshape(self.out_features, -1) * (1.0+torch.repeat_interleave(self.scales, self.group_size, dim=1))
        # print("linear when forward", weight.mean().item(), weight.std().item())
        output = F.linear(input, weight)

        if self.bias is not None:
            output += self.bias
        
        return output


class TVQHadLinear(nn.Module):

    def __init__(
            self, 
            in_features, 
            out_features, 
            group_size=128, 
            vec_size=4, 
            vec_bit=2, 
            bias=False, 
            device=None, 
            dtype=None,
            proj_type="bypass",
            codebook_trainable=True
        ):
        assert out_features%group_size == 0, (out_features, group_size)
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.vec_size = vec_size
        
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.SU = nn.Parameter(torch.empty(in_features, **factory_kwargs))
        self.SV = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        self.code_generator = CodeBookGenerator(
            vec_size=vec_size, 
            vec_bit=vec_bit, 
            in_features=in_features,
            out_features=out_features,
            proj_type=proj_type
        )
        if not codebook_trainable:
            for p in self.code_generator.parameters():
                p.requires_grad = False
        hadamard_left, K_left = get_hadK(in_features)
        self.hadamard_left_T = nn.Parameter(hadamard_left.to(**factory_kwargs).T.contiguous().clone(), requires_grad=False)
        self.K_left = K_left
        hadamard_right, K_right = get_hadK(out_features)
        self.hadamard_right = nn.Parameter(hadamard_right.to(**factory_kwargs).clone(), requires_grad=False)
        self.K_right = K_right    

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
            
    def forward(self, inputs):
        inputs = inputs * self.SU
        inputs = matmul_hadU_cuda(inputs, self.hadamard_left_T, self.K_left)
        weight = self.code_generator.quant_weight(self.weight)
        B, T, D = inputs.shape
        inputs = inputs.view(B, T, D//self.vec_size, self.vec_size)
        outputs = F.linear(inputs, self.code_generator.g1.weight.T)
        outputs = outputs.reshape(B, T, D)
        outputs = F.linear(outputs, weight) + (inputs * self.code_generator.g1.bias).sum(dim=(2, 3)).unsqueeze(2)
        outputs = matmul_hadU_cuda(outputs, self.hadamard_right, self.K_right)
        outputs = outputs * self.SV

        if self.bias is not None:
            outputs += self.bias
        
        return outputs
