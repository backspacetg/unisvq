import torch

def pack_2bit(tensor):
    tensor = tensor.to(torch.uint8)
    original_shape = tensor.shape
    flat_tensor = tensor.view(-1, 4)
    
    packed = (flat_tensor[:, 0] << 6) | \
             (flat_tensor[:, 1] << 4) | \
             (flat_tensor[:, 2] << 2) | \
             (flat_tensor[:, 3])
    new_shape = list(original_shape[:-1]) + [original_shape[-1] // 4]
    return packed.view(new_shape)

def unpack_2bit(packed_tensor, original_shape, dtype=torch.float16):

    flat_packed = packed_tensor.view(-1, 1)
    
    v0 = (flat_packed >> 6) & 0b11
    v1 = (flat_packed >> 4) & 0b11
    v2 = (flat_packed >> 2) & 0b11
    v3 = (flat_packed)      & 0b11

    unpacked = torch.cat([v0, v1, v2, v3], dim=-1)
    return unpacked.view(original_shape).to(dtype)

def pack_tri(tensor):

    original_shape = tensor.shape
    if original_shape[-1] % 5 != 0:
        pad_size = 5 - (original_shape[-1] % 5)
        tensor = torch.nn.functional.pad(tensor, (0, pad_size), value=0)
    
    tensor = tensor.to(torch.uint8)
    flat_tensor = tensor.view(-1, 5)
    
    weights = torch.tensor([81, 27, 9, 3, 1], device=tensor.device, dtype=torch.uint8)
    packed = (flat_tensor * weights).sum(dim=-1, dtype=torch.uint8)

    new_shape = list(original_shape[:-1]) + [ (original_shape[-1] + 4) // 5 ]
    return packed.view(new_shape)

def unpack_tri(packed_tensor, original_shape, dtype=torch.float16):
    n_row = packed_tensor.shape[0]
    flat_packed = packed_tensor.view(-1, 1).to(torch.int32)
    v0 = (flat_packed // 81) % 3
    v1 = (flat_packed // 27) % 3
    v2 = (flat_packed // 9) % 3
    v3 = (flat_packed // 3) % 3
    v4 = flat_packed % 3
    unpacked = torch.cat([v0, v1, v2, v3, v4], dim=-1)
    res = unpacked.reshape(n_row, -1)[:, :original_shape[1]]
    return res.to(dtype)             


def simple_rtn(weight: torch.Tensor, nbits=8, group_size=128):
    ori_shape = weight.shape
    weight = weight.reshape([-1, group_size])
    _min = weight.min(axis=1, keepdim=True)[0]
    _max = weight.max(axis=1, keepdim=True)[0]
    max_v = round(2**nbits - 1)
    min_v = 0
    min_max = [min_v, max_v]
    scale = (max_v / (_max - _min)).clamp(max=2e4)  # clamp to avoid half-precision problems
    zero = -_min * scale

    weight = torch.round(weight * scale + zero).clamp(min_max[0], min_max[1]).to(torch.uint8)
    scale_r = 1.0 / scale

    return weight, scale_r, zero, ori_shape


def pack_lm_head(lm_head_weight: torch.Tensor):
    qweight, scale_r, zero, ori_shape = simple_rtn(
        weight=lm_head_weight, 
        nbits=4, 
        group_size=128)
    qweight = (qweight[:, 0::2] << 4) | qweight[:, 1::2]

    return qweight, scale_r, zero, ori_shape


def unpack_lm_head(q_weight_packed: torch.Tensor, scale_r, zero, ori_shape):
    q_weight = q_weight_packed.new_zeros(size=[q_weight_packed.shape[0], q_weight_packed.shape[1]*2])
    q_weight[:, 0::2] = q_weight_packed >> 4
    q_weight[:, 1::2] = q_weight_packed & 0x0F
    q_weight_unpacked = ((q_weight - zero) * scale_r).reshape(ori_shape)

    return q_weight_unpacked
